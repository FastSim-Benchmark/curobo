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


def _preserve_clip_plane_precision(
    result: SphereFitResult, clip_plane: tuple, buffer: float = 0.02,
) -> None:
    """Keep the half-plane bound after centers and radii change precision."""
    normal = torch.as_tensor(clip_plane[0], dtype=torch.float64, device=result.centers.device)
    normal = normal / torch.linalg.vector_norm(normal)
    distance = result.centers.to(torch.float64) @ normal - (float(clip_plane[1]) + buffer)
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


def _select_auto_fit(
    mesh: trimesh.Trimesh, budget: Optional[int], outward: Optional[float],
) -> tuple[SphereFitType, str]:
    """Select a backend without weakening an explicit outward constraint."""
    if outward is not None:
        return SphereFitType.BOUNDED, "explicit_outward_bound"
    if (isinstance(budget, (bool, numpy.bool_))
            or not isinstance(budget, (int, numpy.integer)) or not 4 <= budget <= 50):
        return SphereFitType.FAST, "budget_outside_explicit_rim_range"
    vertices = numpy.unique(numpy.asarray(mesh.vertices), axis=0)
    if (len(vertices) != 8 or not mesh.is_volume or mesh.body_count != 1):
        return SphereFitType.FAST, "not_single_rectangular_plate"
    transform, extents = trimesh.bounds.oriented_bounds(mesh)
    dimensions = numpy.sort(extents)
    if dimensions[0] > 0.2 * dimensions[1]:
        return SphereFitType.FAST, "not_thin_plate"
    local = trimesh.transform_points(vertices, transform)
    tolerance = float(extents.max()) * 1e-8
    if not numpy.all(numpy.abs(numpy.abs(local) - extents / 2) <= tolerance):
        return SphereFitType.FAST, "not_rectangular_plate"
    return SphereFitType.RIM, "single_rectangular_plate_with_small_budget"


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
    max_bottom_protrusion_m: Optional[float] = None,
    max_protrusion_m: Optional[float] = None,
) -> SphereFitResult:
    """Approximate a mesh with spheres.

    Args:
        mesh: Input mesh.
        num_spheres: Sphere budget. FAST accepts integers in [1, 256] and may
            return fewer spheres. When None, FAST uses ceil(32*sphere_density),
            clamped to [1, 256]; RIM requires an explicit budget in [4, 50].
            Legacy methods use their volume-based estimate.
        sphere_density: Dimensionless density multiplier used when *num_spheres*
            is ``None``.  Scales both the sphere count estimate and the
            per-link cap.  ``1.0`` (default) gives a balanced count; ``2.0``
            doubles it; ``0.5`` halves it.  Practical range: ``0.1`` -- ``10.0``.
        surface_radius: Radius added to surface-sampled spheres.  Only affects
            the ``SURFACE`` fit type and the surface-sampling fallback.
        fit_type: Fitting algorithm, FAST by default. AUTO routes an explicit
            outward bound to BOUNDED; a single rectangular plate with thickness
            at most 20% of its intermediate extent and an explicit 4--50 budget
            to RIM; other requests to FAST. RIM certifies only the contour band.
            Explicit backend selections are never rerouted.
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
        max_bottom_protrusion_m: Optional nonnegative bound below the mesh's
            minimum local Z. FAST and RIM consider this plane during fitting.
            RIM retains its source-bound contour witnesses within the source's
            local-Z extent. Final clipping clamps radii, removes spheres
            entirely below the bound, and preserves it after dtype conversion.
            Metrics describe the final clipped spheres.
            This is an object-local plane, not a world-gravity constraint.
        max_protrusion_m: Required positive source-relative outward bound for
            BOUNDED. AUTO selects BOUNDED when this is supplied; explicitly
            selected other methods reject this parameter. BOUNDED requires an
            explicit sphere capacity and closed components with a verified
            interior kernel; unsupported geometry or insufficient capacity
            raises instead of returning an approximate cover.

    Returns:
        A :class:`SphereFitResult` with sphere positions, radii, and
        optionally quality metrics.
    """
    routing = None
    if fit_type == SphereFitType.AUTO:
        fit_type, reason = _select_auto_fit(mesh, num_spheres, max_protrusion_m)
        routing = {"requested_fit_type": "auto", "selected_fit_type": fit_type.value,
                   "reason": reason}
    if fit_type == SphereFitType.RIM:
        if num_spheres is None:
            raise ValueError("RIM requires an explicit sphere capacity in [4, 50]")
        if device_cfg.dtype not in (torch.float32, torch.float64):
            raise ValueError("RIM supports float32 and float64 output")
    if fit_type == SphereFitType.BOUNDED:
        if num_spheres is None:
            raise ValueError("BOUNDED requires an explicit sphere capacity")
        if device_cfg.dtype not in (torch.float32, torch.float64):
            raise ValueError("BOUNDED supports float32 and float64 output")
        if (isinstance(max_protrusion_m, bool) or max_protrusion_m is None
                or not numpy.isfinite(max_protrusion_m) or max_protrusion_m <= 1e-5):
            raise ValueError("BOUNDED max_protrusion_m must be finite and greater than 1e-5")
    elif max_protrusion_m is not None:
        raise ValueError("max_protrusion_m requires BOUNDED fitting")
    bottom_plane = None
    if max_bottom_protrusion_m is not None:
        if (isinstance(max_bottom_protrusion_m, bool)
                or not numpy.isfinite(max_bottom_protrusion_m)
                or max_bottom_protrusion_m < 0):
            raise ValueError("max_bottom_protrusion_m must be finite and nonnegative")
        vertices = numpy.asarray(mesh.vertices)
        if vertices.size == 0 or not numpy.isfinite(vertices).all():
            raise ValueError("bottom-plane constraint requires a finite nonempty mesh")
        bottom_plane = ((0.0, 0.0, 1.0), float(vertices[:, 2].min()) - max_bottom_protrusion_m)
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
    bounded_diagnostics = None
    rim_diagnostics = None

    t0 = time.time()

    device = device_cfg.device

    if fit_type == SphereFitType.FAST:
        # Lazy import keeps explicitly selected legacy backends independent.
        from curobo._src.geom.sphere_fit.fit_fast import fast_fit_mesh

        if isinstance(num_spheres, bool) or not isinstance(num_spheres, (int, numpy.integer)):
            raise ValueError("FAST num_spheres must be an integer budget in [1, 256]")
        if bottom_plane is None:
            n_pts, n_radius, fast_diagnostics = fast_fit_mesh(mesh, num_spheres)
        else:
            n_pts, n_radius, fast_diagnostics = fast_fit_mesh(
                mesh, num_spheres, minimum_z=bottom_plane[1]
            )

    elif fit_type == SphereFitType.RIM:
        from curobo._src.geom.sphere_fit.fit_rim import rim_fit_mesh

        output_dtype = numpy.float32 if device_cfg.dtype == torch.float32 else numpy.float64
        n_pts, n_radius, rim_diagnostics = rim_fit_mesh(
            mesh, num_spheres, output_dtype,
            minimum_z=None if bottom_plane is None else bottom_plane[1],
        )

    elif fit_type == SphereFitType.BOUNDED:
        from curobo._src.geom.sphere_fit.fit_bounded import bounded_fit_mesh

        effective_bound = max_protrusion_m
        if max_bottom_protrusion_m is not None and effective_bound is not None:
            effective_bound = min(effective_bound, max_bottom_protrusion_m)
        n_pts, n_radius, bounded_diagnostics = bounded_fit_mesh(
            mesh, num_spheres, effective_bound,
        )

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

    if fit_type in (SphereFitType.FAST, SphereFitType.BOUNDED, SphereFitType.RIM) and (
        n_pts is None or len(n_pts) < 1
    ):
        raise ValueError(
            f"{fit_type.name} produced no spheres; legacy fitting fallback is disabled"
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
        if fit_type == SphereFitType.BOUNDED:
            raise ValueError("BOUNDED cannot truncate a volume cover to fit capacity")
        if fit_type == SphereFitType.RIM:
            raise ValueError("RIM cannot truncate a contour cover to fit capacity")
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

    if routing is not None:
        result.debug_info["routing"] = routing
    if fast_diagnostics is not None:
        result.debug_info["fast"] = fast_diagnostics

    if clip_plane is not None and result.num_spheres > 0:
        _apply_clip_plane(result, clip_plane)

    if bottom_plane is not None and result.num_spheres > 0:
        _apply_clip_plane(result, bottom_plane, buffer=0.0)

    result.centers = torch.as_tensor(
        result.centers, dtype=device_cfg.dtype, device=device
    ).contiguous()
    result.radii = torch.as_tensor(
        result.radii, dtype=device_cfg.dtype, device=device
    ).contiguous()

    if clip_plane is not None and result.num_spheres > 0:
        _preserve_clip_plane_precision(result, clip_plane)

    if bottom_plane is not None:
        if result.num_spheres > 0:
            _preserve_clip_plane_precision(result, bottom_plane, buffer=0.0)
        result.debug_info["bottom_plane"] = {
            "minimum_z_m": bottom_plane[1],
            "max_bottom_protrusion_m": max_bottom_protrusion_m,
            "frame": "mesh_local",
        }
    if bounded_diagnostics is not None:
        from curobo._src.geom.sphere_fit.fit_bounded import verify_bounded_result

        verify_bounded_result(
            mesh, result.centers.detach().cpu().numpy(), result.radii.detach().cpu().numpy(),
            bounded_diagnostics,
        )
        result.debug_info["bounded"] = bounded_diagnostics
        result.fit_time_s = time.time() - t0
    if rim_diagnostics is not None:
        from curobo._src.geom.sphere_fit.fit_rim import verify_rim_result

        verify_rim_result(
            mesh, result.centers.detach().cpu().numpy(), result.radii.detach().cpu().numpy(),
            rim_diagnostics,
        )
        result.debug_info["rim"] = rim_diagnostics
        result.fit_time_s = time.time() - t0

    if compute_metrics:
        populate_metrics(result, mesh, device=device)

    return result
