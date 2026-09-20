# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Top-level sphere fitting dispatcher.

This module provides the main entry point :func:`fit_spheres_to_mesh` which delegates
to the appropriate fitting backend based on the requested :class:`SphereFitType`.
"""

# Standard Library
from __future__ import annotations

import time
from typing import Optional

# Third Party
import numpy
import torch
import trimesh

from curobo._src.geom.sphere_fit.fit_morphit import MorphItLossWeights, morphit_sphere_fit
from curobo._src.geom.sphere_fit.fit_voxel import sample_even_fit_mesh, voxel_fit_mesh
from curobo._src.geom.sphere_fit.metrics import populate_metrics

# CuRobo
from curobo._src.geom.sphere_fit.sphere_count import estimate_sphere_count
from curobo._src.geom.sphere_fit.types import SphereFitResult, SphereFitType
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.logging import log_info, log_warn


def _is_hollow_mesh(mesh: trimesh.Trimesh, thickness_ratio: float = 0.1) -> bool:
    """Detect if a mesh is a degenerate thin shell that needs convex-hull replacement.

    Only triggers for watertight meshes whose volume is an extremely small
    fraction of their bounding box (e.g. a closed thin panel).  Non-watertight
    meshes are left as-is; Warp SDF queries handle triangle soups correctly.
    """
    if not mesh.is_watertight:
        return False
    bbox_dims = mesh.bounds[1] - mesh.bounds[0]
    bbox_vol = float(numpy.prod(bbox_dims))
    if bbox_vol < 1e-12:
        return True
    fill_ratio = abs(float(mesh.volume)) / bbox_vol
    return fill_ratio < thickness_ratio


def _apply_clip_plane(
    result: SphereFitResult,
    clip_plane: tuple,
    buffer: float = 0.02,
) -> None:
    """Discard or shrink spheres that cross a half-plane boundary.

    Centers on or behind the buffered plane (offset plus *buffer*) are removed.
    Remaining radii are clamped to that boundary. Modifies *result* in place.
    """
    normal = numpy.array(clip_plane[0], dtype=numpy.float64)
    normal = normal / numpy.linalg.norm(normal)
    offset = float(clip_plane[1])

    clearance = result.centers @ normal - (offset + buffer)
    keep = clearance > 0.0
    if not numpy.all(keep):
        result.centers = result.centers[keep]
        result.radii = result.radii[keep]
        result.num_spheres = len(result.centers)
        clearance = clearance[keep]

    if result.num_spheres > 0:
        max_radii = numpy.maximum(clearance, 0.0)
        result.radii = numpy.minimum(result.radii, max_radii)


def _preserve_clip_plane_precision(result: SphereFitResult, clip_plane: tuple) -> None:
    """Keep the half-plane bound after centers and radii change precision."""
    normal = torch.as_tensor(clip_plane[0], dtype=torch.float64, device=result.centers.device)
    normal = normal / torch.linalg.vector_norm(normal)
    distance = result.centers.to(torch.float64) @ normal - (float(clip_plane[1]) + 0.02)
    radii = torch.minimum(result.radii.to(torch.float64), distance).to(result.radii.dtype)
    # Nearest rounding can cross the bound. Choose the adjacent interior value
    # only in that case, rather than adding a geometric clearance or tolerance.
    radii = torch.where(
        radii.to(torch.float64) > distance,
        torch.nextafter(radii, torch.zeros_like(radii)),
        radii,
    )
    keep = (distance > 0) & (radii > 0)
    result.centers = result.centers[keep].contiguous()
    result.radii = radii[keep].contiguous()
    result.num_spheres = len(result.radii)


def fit_spheres_to_mesh(
    mesh: trimesh.Trimesh,
    num_spheres: Optional[int] = None,
    sphere_density: float = 1.0,
    surface_radius: float = 0.005,
    fit_type: SphereFitType = SphereFitType.FAST,
    iterations: int = 200,
    compute_metrics: bool = False,
    coverage_weight: Optional[float] = None,
    protrusion_weight: Optional[float] = None,
    clip_plane: Optional[tuple] = None,
    device_cfg: DeviceCfg = DeviceCfg(),
) -> SphereFitResult:
    """Approximate a mesh with spheres.

    Args:
        mesh: Input mesh.
        num_spheres: Sphere budget. FAST accepts integers in [1, 256] and may
            return fewer spheres. When None, FAST uses ceil(32*sphere_density),
            clamped to [1, 256]; legacy methods use their volume-based estimate.
        sphere_density: Dimensionless density multiplier used when *num_spheres*
            is ``None``.  Scales both the sphere count estimate and the
            per-link cap.  ``1.0`` (default) gives a balanced count; ``2.0``
            doubles it; ``0.5`` halves it.  Practical range: ``0.1`` -- ``10.0``.
        surface_radius: Radius added to surface-sampled spheres.  Only affects
            the ``SURFACE`` fit type and the surface-sampling fallback.
        fit_type: Fitting algorithm, FAST by default; see :class:`SphereFitType`.
        iterations: Optimization iterations (only used by ``MORPHIT``).
        compute_metrics: When True, compute quality metrics (coverage,
            protrusion, surface gap, volume ratio) on the result.
        coverage_weight: MorphIt coverage loss weight.  Higher values force
            spheres to fill the mesh volume more completely.  Only used by
            ``MORPHIT``.  When ``None``, uses the default (1000.0).
        protrusion_weight: MorphIt protrusion loss weight.  Higher values
            penalise sphere surface area outside the mesh more aggressively.
            Only used by ``MORPHIT``.  When ``None``, uses the default (10.0).
        clip_plane: Half-plane constraint ``((nx, ny, nz), offset)`` in
            mesh-local coordinates.  Spheres that cross the plane are penalised
            during MorphIt optimization and hard-clamped afterwards.  For
            non-MorphIt fit types, only the hard clamp is applied.  ``None``
            (default) disables clipping.
        device_cfg: Device and floating-point dtype for the returned spheres.

    Returns:
        A :class:`SphereFitResult` with sphere positions, radii, and
        optionally quality metrics.
    """
    requested_n_spheres = num_spheres
    used_convex_hull = False

    if fit_type in (SphereFitType.VOXEL, SphereFitType.MORPHIT) and _is_hollow_mesh(mesh):
        log_info("sphere_fit: hollow/thin mesh detected, using convex hull")
        mesh = mesh.convex_hull
        used_convex_hull = True

    auto_mode = num_spheres is None
    if auto_mode:
        if fit_type == SphereFitType.FAST:
            if not numpy.isfinite(sphere_density) or sphere_density <= 0:
                raise ValueError("FAST sphere_density must be positive and finite")
            num_spheres = max(1, int(numpy.ceil(32 * min(float(sphere_density), 8.0))))
        else:
            num_spheres = estimate_sphere_count(mesh, sphere_density=sphere_density)
        log_info(f"sphere_fit: auto num_spheres={num_spheres}")

    n_pts = n_radius = None
    history = []
    fallback_used = False
    fast_diagnostics = None

    t0 = time.time()

    device = device_cfg.device

    if fit_type == SphereFitType.FAST:
        # Lazy import keeps explicitly selected legacy backends independent.
        from curobo._src.geom.sphere_fit.fit_fast import fast_fit_mesh

        if isinstance(num_spheres, bool) or not isinstance(num_spheres, (int, numpy.integer)):
            raise ValueError("FAST num_spheres must be an integer budget in [1, 256]")
        n_pts, n_radius, fast_diagnostics = fast_fit_mesh(mesh, num_spheres)

    elif fit_type == SphereFitType.SURFACE:
        n_pts, n_radius = sample_even_fit_mesh(mesh, num_spheres, surface_radius)

    elif fit_type == SphereFitType.VOXEL:
        n_pts, n_radius = voxel_fit_mesh(mesh, num_spheres, device=device)

    elif fit_type == SphereFitType.MORPHIT:
        init_pts, init_rad = voxel_fit_mesh(mesh, num_spheres, device=device)
        if init_pts is not None and len(init_pts) > 0:
            loss_weights = None
            if coverage_weight is not None or protrusion_weight is not None:
                loss_weights = MorphItLossWeights(
                    coverage=coverage_weight if coverage_weight is not None else 1000.0,
                    protrusion=protrusion_weight if protrusion_weight is not None else 10.0,
                )
            n_pts, n_radius, history = morphit_sphere_fit(
                mesh, num_spheres, iterations=iterations,
                init_centers=init_pts, init_radii=init_rad,
                loss_weights=loss_weights,
                clip_plane=clip_plane,
                max_spheres=num_spheres if requested_n_spheres is not None else 0,
                device=device,
            )

    if (n_pts is None or len(n_pts) < 1) and num_spheres > 0:
        log_warn("sphere_fit: primary method failed, falling back to voxel volume")
        n_pts, n_radius = voxel_fit_mesh(mesh, num_spheres, device=device)
        fallback_used = True

    if (n_pts is None or len(n_pts) < 1) and num_spheres > 0:
        log_warn("sphere_fit: voxel fallback empty (thin shell?), using surface sampling")
        n_pts, n_radius = sample_even_fit_mesh(mesh, num_spheres, surface_radius)
        fallback_used = True

    fit_time = time.time() - t0

    if n_pts is None:
        n_pts = numpy.zeros((0, 3))
    if n_radius is None:
        n_radius = numpy.zeros((0,))
    n_radius = numpy.ravel(n_radius)

    if requested_n_spheres is not None and len(n_pts) > requested_n_spheres:
        order = numpy.argsort(-n_radius)[:requested_n_spheres]
        n_pts = n_pts[order]
        n_radius = n_radius[order]

    result = SphereFitResult(
        centers=n_pts,
        radii=n_radius,
        num_spheres=len(n_pts),
        fit_time_s=fit_time,
        used_mesh=mesh,
        history=history,
        debug_info={
            "fit_type": fit_type.value,
            "used_convex_hull": used_convex_hull,
            "auto_n_spheres": auto_mode,
            "requested_n_spheres": requested_n_spheres,
            "resolved_n_spheres": num_spheres,
            "fallback_used": fallback_used,
        },
    )

    if fast_diagnostics is not None:
        result.debug_info["fast"] = fast_diagnostics

    if clip_plane is not None and result.num_spheres > 0:
        _apply_clip_plane(result, clip_plane)

    result.centers = torch.as_tensor(
        result.centers, dtype=device_cfg.dtype, device=device
    ).contiguous()
    result.radii = torch.as_tensor(
        result.radii, dtype=device_cfg.dtype, device=device
    ).contiguous()

    if clip_plane is not None and result.num_spheres > 0:
        _preserve_clip_plane_precision(result, clip_plane)

    if compute_metrics:
        populate_metrics(result, mesh, device=device)

    return result
