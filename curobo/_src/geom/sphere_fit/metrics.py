# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Quality metrics for sphere fitting results."""

from __future__ import annotations

import numpy as np
import torch
import trimesh

from curobo._src.geom.sphere_fit.types import SphereFitMetrics, SphereFitResult
from curobo._src.geom.sphere_fit.wp_mesh_query import WarpMeshQuery


def _solid_components(mesh):
    """Interpret closed collision components as a union, without repairing them."""
    parts = list(mesh.split(only_watertight=False, repair=False))
    if len(parts) > 1 and all(
        part.is_watertight and part.is_winding_consistent and part.volume > 0 for part in parts
    ):
        return parts
    return [mesh]


def _union_sdf(queries, points):
    """Outside distance is distance to the nearest solid, not an internal face."""
    distance = None
    for query in queries:
        current, _ = query.query_sdf(points.contiguous())
        distance = current.clone() if distance is None else torch.minimum(distance, current)
    return distance


def compute_sphere_fit_metrics(
    mesh: trimesh.Trimesh,
    centers: np.ndarray,
    radii: np.ndarray,
    n_interior: int = 10000,
    n_surface: int = 5000,
    n_sphere_surface: int = 200,
    device: torch.device = torch.device("cuda", 0),
) -> SphereFitMetrics:
    """Compute quality metrics for a sphere fit.

    Args:
        mesh: Original triangle mesh.
        centers: ``(N, 3)`` sphere centres (torch.Tensor or numpy array).
        radii: ``(N,)`` sphere radii (torch.Tensor or numpy array).
        n_interior: Number of interior points for coverage measurement.
        n_surface: Number of mesh surface points for gap measurement.
        n_sphere_surface: Points per sphere for protrusion measurement.
        device: Torch device for computation.

    Returns:
        :class:`SphereFitMetrics` with quality measurements.
    """
    dev = device
    num_spheres = len(centers)
    if num_spheres == 0:
        return SphereFitMetrics(
            num_spheres=0,
            protrusion=1.0,
            surface_gap_mean=float("inf"),
            surface_gap_p95=float("inf"),
            max_uncovered_gap=float("inf"),
        )

    centers_t = torch.as_tensor(centers, dtype=torch.float32, device=dev)
    radii_t = torch.as_tensor(radii, dtype=torch.float32, device=dev).ravel()

    components = _solid_components(mesh)
    mesh_queries = [WarpMeshQuery(part, dev) for part in components]

    # --- Coverage: fraction of interior points inside >= 1 sphere -----------
    candidates = np.random.random((n_interior * 2, 3)) * mesh.extents + mesh.bounds[0]
    contained = np.logical_or.reduce([part.contains(candidates) for part in components])
    interior_pts = candidates[contained][:n_interior]
    if not len(interior_pts):
        raise ValueError("sphere-fit coverage sampling found no interior points")
    interior_t = torch.as_tensor(interior_pts, dtype=torch.float32, device=dev)

    # Matrix-product distances can lose millimeter-scale separation when TF32
    # is enabled elsewhere. Direct differences preserve the coverage witness.
    dists = torch.cdist(
        interior_t.unsqueeze(0), centers_t.unsqueeze(0),
        compute_mode="donot_use_mm_for_euclid_dist",
    ).squeeze(0)
    covered = (dists < radii_t.unsqueeze(0)).any(dim=1)
    coverage = covered.float().mean().item()

    # --- Protrusion: fraction of sphere surface points outside mesh ---------
    sphere_surface_pts = []
    for i in range(num_spheres):
        phi = torch.rand(n_sphere_surface, device=dev) * 2 * np.pi
        cos_theta = torch.rand(n_sphere_surface, device=dev) * 2 - 1
        sin_theta = torch.sqrt(1 - cos_theta**2)
        pts = centers_t[i] + radii_t[i] * torch.stack(
            [
                sin_theta * torch.cos(phi),
                sin_theta * torch.sin(phi),
                cos_theta,
            ],
            dim=1,
        )
        sphere_surface_pts.append(pts)
    sphere_surf_t = torch.cat(sphere_surface_pts, dim=0)
    signed_distance = _union_sdf(mesh_queries, sphere_surf_t)
    outside_mask = signed_distance > 0
    protrusion = outside_mask.float().mean().item()

    if outside_mask.any():
        protrusion_dists = signed_distance[outside_mask]
        protrusion_dist_mean = protrusion_dists.mean().item()
        protrusion_dist_p95 = torch.quantile(protrusion_dists, 0.95).item()
    else:
        protrusion_dist_mean = 0.0
        protrusion_dist_p95 = 0.0

    # --- Surface gap: distance from mesh surface to nearest sphere surface --
    surface_pts, _ = trimesh.sample.sample_surface(mesh, n_surface)
    surface_t = torch.as_tensor(surface_pts, dtype=torch.float32, device=dev)
    surf_dists = torch.cdist(
        surface_t.unsqueeze(0), centers_t.unsqueeze(0),
        compute_mode="donot_use_mm_for_euclid_dist",
    ).squeeze(0)
    sphere_surf_dist = surf_dists - radii_t.unsqueeze(0)
    min_gap, _ = sphere_surf_dist.min(dim=1)
    gap_values = torch.relu(min_gap)
    surface_gap_mean = gap_values.mean().item()
    surface_gap_p95 = torch.quantile(gap_values, 0.95).item()

    # --- Volume ratio -------------------------------------------------------
    sphere_vol = float((4.0 / 3.0 * np.pi * (radii_t**3).sum()).item())
    mesh_vol = (
        float(mesh.volume)
        if mesh.is_watertight and mesh.volume > 0
        else float(np.prod(mesh.bounds[1] - mesh.bounds[0]))
    )
    volume_ratio = sphere_vol / max(mesh_vol, 1e-12)

    max_uncovered_gap = gap_values.max().item()

    return SphereFitMetrics(
        num_spheres=num_spheres,
        coverage=coverage,
        protrusion=protrusion,
        protrusion_dist_mean=protrusion_dist_mean,
        protrusion_dist_p95=protrusion_dist_p95,
        surface_gap_mean=surface_gap_mean,
        surface_gap_p95=surface_gap_p95,
        max_uncovered_gap=max_uncovered_gap,
        volume_ratio=volume_ratio,
    )


def populate_metrics(
    result: SphereFitResult,
    mesh: trimesh.Trimesh,
    n_interior: int = 10000,
    n_surface: int = 5000,
    n_sphere_surface: int = 200,
    device: torch.device = torch.device("cuda", 0),
) -> SphereFitMetrics:
    """Compute metrics, store them on ``result.metrics``, and return them."""
    metrics = compute_sphere_fit_metrics(
        mesh,
        result.centers,
        result.radii,
        n_interior=n_interior,
        n_surface=n_surface,
        n_sphere_surface=n_sphere_surface,
        device=device,
    )
    result.metrics = metrics
    return metrics
