# SPDX-License-Identifier: Apache-2.0

"""Explicit outer-contour proxies for plates, without a volume-cover claim."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import ConvexHull, QhullError

if TYPE_CHECKING:
    import trimesh

_BOUNDARY_BAND_M = 0.0015
_CONTAINMENT_RESERVE_M = 1e-6


def _clip_section_to_source_bottom(corners: np.ndarray, bottom: float) -> np.ndarray:
    """Intersect a convex contour prism with the source's true local-Z extent."""
    inside = corners[corners[:, 2] >= bottom]
    below = corners[corners[:, 2] < bottom]
    if not len(below):
        return corners
    if not len(inside):
        raise ValueError("RIM source-bottom intersection leaves an uncovered contour section")
    # Every hull edge is among these pairs. Extra pair intersections lie inside
    # the same convex prism, so their convex hull is the exact clipped section.
    delta = inside[None] - below[:, None]
    fraction = (bottom - below[:, 2])[:, None] / delta[:, :, 2]
    crossings = below[:, None] + fraction[:, :, None] * delta
    crossings[:, :, 2] = bottom
    return np.unique(np.concatenate((inside, crossings.reshape(-1, 3))), axis=0)


def _geometry(
    mesh: trimesh.Trimesh, num_spheres: int, *, source_bottom_bound: bool = False,
) -> dict:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces)
    if (
        vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4
        or not np.isfinite(vertices).all() or faces.ndim != 2 or faces.shape[1] != 3
        or not len(faces) or not np.issubdtype(faces.dtype, np.integer)
        or faces.min() < 0 or faces.max() >= len(vertices)
    ):
        raise ValueError("RIM requires finite nondegenerate triangle-mesh geometry")
    # A complete compound is projected once. Internal convex-part seams do not
    # become separate contours, and duplicated component vertices cannot bias PCA.
    triangles = vertices[faces]
    if not np.any(np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
    ) > 0):
        raise ValueError("RIM requires nondegenerate source triangles")
    unique = np.unique(triangles.reshape(-1, 3), axis=0)
    if len(unique) < 4:
        raise ValueError("RIM requires at least four distinct plate vertices")
    shift = (unique.min(axis=0) + unique.max(axis=0)) * 0.5
    local = unique - shift
    centered = local - local.mean(axis=0)
    values, basis = np.linalg.eigh(centered.T @ centered / len(centered))
    if values[1] <= 0 or not np.isfinite(values).all():
        raise ValueError("RIM plate has no two-dimensional extent")
    basis = basis[:, ::-1]
    for axis in range(3):
        if basis[np.argmax(np.abs(basis[:, axis])), axis] < 0:
            basis[:, axis] *= -1
    points = local @ basis
    extent = np.ptp(points, axis=0)
    if min(extent[:2]) <= 1e-12 or extent[2] <= 1e-12:
        raise ValueError("RIM requires nondegenerate plate geometry")
    try:
        hull = ConvexHull(points[:, :2])
    except QhullError as error:
        raise ValueError("RIM projected outer contour is degenerate") from error
    contour = points[hull.vertices, :2]
    maximum_side = np.full(len(points), -np.inf)
    for equations in np.array_split(hull.equations, max(1, (len(contour) + 127) // 128)):
        maximum_side = np.maximum(
            maximum_side,
            (points[:, :2] @ equations[:, :2].T + equations[:, 2]).max(axis=1),
        )
    boundary = points[-maximum_side <= _BOUNDARY_BAND_M]
    low, high = float(boundary[:, 2].min()), float(boundary[:, 2].max())
    if high - low <= 1e-12:
        raise ValueError("RIM outer boundary has no nondegenerate height band")
    lengths = np.linalg.norm(np.roll(contour, -1, axis=0) - contour, axis=1)
    cumulative = np.r_[0.0, lengths.cumsum()]
    perimeter = float(cumulative[-1])
    if not np.isfinite(perimeter) or perimeter <= 1e-12 or np.any(lengths <= 0):
        raise ValueError("RIM projected outer contour is degenerate")

    def point_at(distance):
        index = min(
            int(np.searchsorted(cumulative, distance, side="right") - 1), len(contour) - 1
        )
        fraction = (distance - cumulative[index]) / lengths[index]
        return contour[index] + fraction * (contour[(index + 1) % len(contour)] - contour[index])

    sections, centers = [], []
    for index in range(num_spheres):
        start, end = perimeter * index / num_spheres, perimeter * (index + 1) / num_spheres
        distances = np.r_[
            start, cumulative[(cumulative > start) & (cumulative < end)], end
        ]
        xy = np.asarray([point_at(distance) for distance in distances])
        corners = np.concatenate(
            (np.c_[xy, np.full(len(xy), low)], np.c_[xy, np.full(len(xy), high)])
        )
        center = np.r_[(xy.min(axis=0) + xy.max(axis=0)) * 0.5, (low + high) * 0.5]
        source_corners = corners @ basis.T + shift
        if source_bottom_bound:
            source_corners = _clip_section_to_source_bottom(
                source_corners, float(vertices[:, 2].min())
            )
        sections.append(source_corners)
        centers.append(center @ basis.T + shift)
    return {
        "vertices": vertices,
        "faces": faces,
        "projected_vertices": (vertices - shift) @ basis[:, :2],
        "contour": contour,
        "centers": np.asarray(centers),
        "sections": sections,
        "boundary": boundary @ basis.T + shift,
        "pca_extents_m": extent.tolist(),
        "rim_height_range_m": [low, high],
        "perimeter_m": perimeter,
    }


def _source_bridges(geometry):
    """Measure hull-edge intervals unsupported by the source projected triangles."""
    xy, faces, contour = (
        geometry["projected_vertices"], geometry["faces"], geometry["contour"]
    )
    tolerance = max(1e-12, max(geometry["pca_extents_m"]) * 1e-9)
    gaps = []
    for start, end in zip(contour, np.roll(contour, -1, axis=0)):
        edge = end - start
        length = float(np.linalg.norm(edge))
        tangent = edge / length
        normal = np.array([tangent[1], -tangent[0]])
        on_side = np.abs((xy - start) @ normal) <= tolerance
        selected = on_side[faces]
        indices = faces[np.count_nonzero(selected, axis=1) >= 2]
        intervals = []
        for face in indices:
            distance = (xy[face[on_side[face]]] - start) @ tangent
            intervals.append((max(0.0, float(distance.min())), min(length, float(distance.max()))))
        cursor = 0.0
        for lower, upper in sorted(intervals):
            if lower > cursor + tolerance:
                gaps.append(lower - cursor)
            cursor = max(cursor, upper)
        if length > cursor + tolerance:
            gaps.append(length - cursor)
    return float(sum(gaps)), float(max(gaps, default=0.0))


def rim_fit_mesh(
    mesh: trimesh.Trimesh, num_spheres: int, dtype: type = np.float32,
    *, minimum_z: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit a complete convex projected contour band within an explicit 4--50 budget.

    The entire compound defines one PCA plane and one outer convex contour.
    Each sphere contains its assigned contour-section corners. Spheres are
    convex, so these corners cover every intervening contour segment at all
    heights in the selected rim band. A bottom-constrained fit intersects that
    band with the source's true local-Z extent, then raises sphere centers to
    obey the plane without shrinking away coverage. Filled plate faces and
    volume are not certified. The source mesh remains unchanged.
    """
    if (
        isinstance(num_spheres, (bool, np.bool_))
        or not isinstance(num_spheres, (int, np.integer)) or not 4 <= num_spheres <= 50
    ):
        raise ValueError("RIM num_spheres must be an explicit integer budget in [4, 50]")
    if np.dtype(dtype) not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("RIM supports float32 and float64 output")
    geometry = _geometry(mesh, int(num_spheres), source_bottom_bound=minimum_z is not None)
    centers = geometry["centers"].astype(dtype).astype(np.float64)
    boundary = geometry["boundary"]
    # Retain actual vertices near the external contour as additional witnesses,
    # rather than assuming a corner-only contour proof includes the inward band.
    assigned = np.linalg.norm(boundary[:, None] - centers[None], axis=-1).argmin(axis=1)
    radii = []
    for index, (center, corners) in enumerate(zip(centers, geometry["sections"])):
        witnesses = np.concatenate((corners, boundary[assigned == index]))
        if minimum_z is not None:
            reserve = 2 * _CONTAINMENT_RESERVE_M
            height = witnesses[:, 2] - minimum_z
            if np.any(height <= reserve):
                raise ValueError(
                    "RIM bottom bound leaves uncovered contour witnesses on its plane"
                )
            horizontal_squared = np.square(witnesses[:, :2] - center[:2]).sum(axis=1)
            # For center height H above the plane and witness height h,
            # sqrt(xy_distance**2 + (H-h)**2) + reserve <= H.
            # Solve this inequality for H for every assigned witness at once.
            required_height = float(np.max(
                (horizontal_squared + height * height - reserve * reserve)
                / (2 * (height - reserve))
            ))
            required_z = minimum_z + required_height
            if center[2] < required_z:
                rounded_z = np.asarray(required_z, dtype=dtype)
                if float(rounded_z) < required_z:
                    rounded_z = np.nextafter(rounded_z, np.asarray(np.inf, dtype=dtype))
                center[2] = float(rounded_z)
        radius = np.linalg.norm(witnesses - center, axis=1).max() + _CONTAINMENT_RESERVE_M
        radii.append(float(np.nextafter(
            np.asarray(radius, dtype=dtype), np.asarray(np.inf, dtype=dtype)
        )))
    total_bridge, maximum_bridge = _source_bridges(geometry)
    diagnostics = {
        "schema": "curobo-rim-fit/1",
        "frame": "mesh_local",
        "contour_kind": "outer_convex_projected_contour",
        "concavity_handling": "outer_convex_projection_bridges_concavities",
        "convex_contour_has_source_gaps": maximum_bridge > 0,
        "convex_contour_source_gap_length_m": total_bridge,
        "maximum_convex_contour_bridge_m": maximum_bridge,
        "full_volume_covered": False,
        "whole_plate_volume_covered": False,
        "outer_contour_height_band_covered": False,
        "boundary_vertices_covered": False,
        "sphere_count": int(num_spheres),
        "outer_contour_vertex_count": len(geometry["contour"]),
        "boundary_vertex_count": len(boundary),
        "boundary_band_m": _BOUNDARY_BAND_M,
        "metric_containment_reserve_m": _CONTAINMENT_RESERVE_M,
        "contour_band_source_bottom_clipped": minimum_z is not None,
        "bottom_constrained_fit": minimum_z is not None,
        "pca_extents_m": geometry["pca_extents_m"],
        "rim_height_range_m": geometry["rim_height_range_m"],
        "perimeter_m": geometry["perimeter_m"],
        "section_corners_m": [corners.tolist() for corners in geometry["sections"]],
        "boundary_vertices_m": boundary.tolist(),
    }
    return centers, np.asarray(radii), diagnostics


def verify_rim_result(mesh, centers, radii, diagnostics):
    """Recheck continuous contour and vertex coverage after conversion and clipping."""
    c, r = np.asarray(centers, dtype=np.float64), np.asarray(radii, dtype=np.float64)
    count = diagnostics["sphere_count"]
    if c.shape != (count, 3) or r.shape != (count,):
        raise ValueError("RIM clipping removed spheres required by the contour certificate")
    if not np.isfinite(c).all() or not np.isfinite(r).all() or np.any(r <= 0):
        raise ValueError("RIM result requires finite positive spheres")
    geometry = _geometry(
        mesh, count,
        source_bottom_bound=diagnostics.get("contour_band_source_bottom_clipped", False),
    )
    source_sections = geometry["sections"]
    declared_sections = diagnostics["section_corners_m"]
    if len(declared_sections) != count or any(
        not np.array_equal(declared, original)
        for declared, original in zip(declared_sections, source_sections)
    ) or not np.array_equal(diagnostics["boundary_vertices_m"], geometry["boundary"]):
        raise ValueError("RIM contour witnesses do not match the original mesh")
    section_margin = min(
        float(radius - np.linalg.norm(corners - center, axis=1).max())
        for center, radius, corners in zip(c, r, source_sections)
    )
    boundary_gaps = (
        np.linalg.norm(geometry["boundary"][:, None] - c[None], axis=-1) - r[None]
    ).min(axis=1)
    vertex_margin = float(-boundary_gaps.max())
    if section_margin <= 0 or vertex_margin <= 0:
        raise ValueError(
            "RIM clipping or conversion leaves an uncovered outer contour or rim vertex"
        )
    diagnostics.update(
        outer_contour_height_band_covered=True,
        boundary_vertices_covered=True,
        minimum_assigned_outer_contour_margin_m=section_margin,
        minimum_boundary_vertex_margin_m=vertex_margin,
        containment_margin_m=min(section_margin, vertex_margin),
        maximum_assigned_outer_contour_gap_m=0.0,
        maximum_boundary_vertex_gap_m=0.0,
        uncovered_boundary_vertices=0,
        object_local_bottom_protrusion_m=max(
            0.0, float(geometry["vertices"][:, 2].min() - (c[:, 2] - r).min())
        ),
    )
