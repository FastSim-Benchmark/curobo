# SPDX-License-Identifier: Apache-2.0

"""Explicit plate outer-contour fitting and its converted-geometry evidence."""

import copy

import numpy as np
import pytest
import torch
import trimesh
from scipy.spatial import Delaunay

from curobo._src.geom.sphere_fit.fit_rim import verify_rim_result
from curobo.sphere_fit import SphereFitType, fit_spheres_to_mesh
from curobo.types import DeviceCfg


def _plate():
    return trimesh.creation.cylinder(radius=0.1, height=0.006, sections=32)


def _fit(mesh=None, **kwargs):
    return fit_spheres_to_mesh(
        _plate() if mesh is None else mesh,
        fit_type=SphereFitType.RIM,
        **{"num_spheres": 50, "device_cfg": DeviceCfg(device="cpu"), **kwargs},
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("budget", [4, 50, np.int32(12)])
def test_rim_explicit_budget_covers_returned_contour_and_vertices(dtype, budget):
    """Every final sphere contains its continuous section's upper/lower corners."""
    mesh = _plate()
    result = _fit(mesh, num_spheres=budget, device_cfg=DeviceCfg(device="cpu", dtype=dtype))
    proof = result.debug_info["rim"]
    assert result.num_spheres == budget
    assert result.centers.dtype == result.radii.dtype == dtype
    assert result.centers.device.type == "cpu"
    assert result.used_mesh is mesh
    assert not result.debug_info["fallback_used"]
    assert not result.debug_info["used_convex_hull"]
    assert result.debug_info["fit_type"] == "rim"
    assert proof["schema"] == "curobo-rim-fit/1" and proof["frame"] == "mesh_local"
    assert proof["outer_contour_height_band_covered"]
    assert proof["boundary_vertices_covered"]
    assert proof["containment_margin_m"] > 0
    assert proof["uncovered_boundary_vertices"] == 0
    assert not proof["full_volume_covered"] and not proof["whole_plate_volume_covered"]
    for center, radius, corners in zip(
        result.centers.double().numpy(), result.radii.double().numpy(), proof["section_corners_m"]
    ):
        assert len(corners) >= 4
        assert np.linalg.norm(np.asarray(corners) - center, axis=1).max() < radius


@pytest.mark.parametrize("budget", [None, True, np.bool_(True), 3, 51, 4.0, "50"])
def test_rim_rejects_implicit_or_invalid_budgets(budget):
    """Rim fitting never estimates or silently expands its capacity."""
    with pytest.raises(ValueError, match="RIM"):
        _fit(num_spheres=budget)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("budget", [4, 50])
@pytest.mark.parametrize("height_to_width", [0.350001, 0.424, 0.7])
def test_rim_deep_plate_has_no_height_to_width_cutoff(dtype, budget, height_to_width):
    """Deep plates retain the returned contour proof beyond the former 35% cutoff."""
    mesh = trimesh.creation.cylinder(radius=0.1, height=0.2 * height_to_width, sections=32)
    result = _fit(mesh, num_spheres=budget, device_cfg=DeviceCfg(device="cpu", dtype=dtype))
    proof = result.debug_info["rim"]
    extents = proof["pca_extents_m"]
    assert mesh.extents[2] / min(mesh.extents[:2]) == pytest.approx(height_to_width)
    assert extents[2] / min(extents[:2]) > 0.35
    assert result.num_spheres == budget
    assert result.centers.dtype == result.radii.dtype == dtype
    assert result.used_mesh is mesh
    assert not result.debug_info["fallback_used"]
    assert proof["outer_contour_height_band_covered"]
    assert proof["boundary_vertices_covered"]
    assert proof["containment_margin_m"] > 0
    centers, radii = result.centers.double().numpy(), result.radii.double().numpy()
    for center, radius, corners in zip(centers, radii, proof["section_corners_m"]):
        assert np.linalg.norm(np.asarray(corners) - center, axis=1).max() < radius
    gaps = (
        np.linalg.norm(np.asarray(proof["boundary_vertices_m"])[:, None] - centers, axis=-1)
        - radii
    ).min(axis=1)
    assert np.all(gaps < 0)
    assert not proof["full_volume_covered"] and not proof["whole_plate_volume_covered"]


def test_rim_equal_extent_geometry_is_not_rejected_as_too_thick():
    """Height alone does not reject a nondegenerate projected outer contour."""
    result = _fit(trimesh.creation.box([0.1, 0.1, 0.1]))
    proof = result.debug_info["rim"]
    assert result.used_mesh.extents == pytest.approx([0.1, 0.1, 0.1])
    extents = proof["pca_extents_m"]
    assert extents[2] / min(extents[:2]) > 0.35
    assert proof["outer_contour_height_band_covered"]
    assert proof["boundary_vertices_covered"]
    assert proof["containment_margin_m"] > 0
    assert result.num_spheres == 50


@pytest.mark.parametrize("kind", ["line", "flat", "nonfinite", "empty"])
def test_rim_rejects_unsupported_geometry(kind):
    """Invalid or degenerate input fails instead of changing fitting methods."""
    mesh = _plate()
    if kind == "line":
        mesh = trimesh.Trimesh(
            vertices=[[0, 0, 0], [0.1, 0, 0], [0.2, 0, 0], [0.3, 0, 0]],
            faces=[[0, 1, 2], [1, 2, 3]], process=False,
        )
    elif kind == "flat":
        mesh.vertices[:, 2] = 0
    elif kind == "nonfinite":
        mesh.vertices[0, 0] = np.nan
    else:
        mesh = trimesh.Trimesh(vertices=[], faces=[], process=False)
    with pytest.raises(ValueError, match="RIM"):
        _fit(mesh)


def test_rim_compound_uses_one_external_contour_and_excludes_internal_seams():
    """Closed convex-part edges inside the full plate are not external rims."""
    parts = []
    for x in [-0.05, 0.05]:
        for y in [-0.05, 0.05]:
            part = trimesh.creation.box([0.1, 0.1, 0.006])
            part.apply_translation([x, y, 0])
            parts.append(part)
    mesh = trimesh.util.concatenate(parts)
    proof = _fit(mesh).debug_info["rim"]
    assert proof["outer_contour_vertex_count"] == 4
    assert not proof["convex_contour_has_source_gaps"]
    boundary = np.asarray(proof["boundary_vertices_m"])
    assert not np.any(np.all(np.isclose(boundary[:, :2], [0, 0]), axis=1))


def test_rim_concave_source_reports_its_convex_contour_approximation():
    """A convex outer contour may bridge a concavity; the evidence says so."""
    parts = []
    for x, y in [(-0.05, -0.05), (-0.05, 0.05), (0.05, -0.05)]:
        part = trimesh.creation.box([0.1, 0.1, 0.006])
        part.apply_translation([x, y, 0])
        parts.append(part)
    proof = _fit(trimesh.util.concatenate(parts)).debug_info["rim"]
    assert proof["contour_kind"] == "outer_convex_projected_contour"
    assert proof["concavity_handling"] == "outer_convex_projection_bridges_concavities"
    assert proof["convex_contour_has_source_gaps"]
    assert proof["maximum_convex_contour_bridge_m"] > 0.02
    assert not proof["full_volume_covered"]


def test_rim_rotated_translated_plate_keeps_original_frame_evidence():
    """PCA handles the plane orientation while witnesses stay mesh-local."""
    mesh = _plate()
    mesh.apply_transform(trimesh.transformations.rotation_matrix(0.6, [0.3, 0.5, 0.2]))
    mesh.apply_translation([0.123456789, -0.234567891, 0.345678912])
    result = _fit(mesh)
    verify_rim_result(mesh, result.centers.numpy(), result.radii.numpy(), result.debug_info["rim"])
    assert result.debug_info["rim"]["containment_margin_m"] > 0


def test_rim_pca_centers_asymmetrically_tessellated_rotated_geometry():
    """A bbox offset must not turn a rigidly rotated six-mm plate into a thick solid."""
    xy = np.r_[
        [[0, 0], [0.2, 0], [0, 0.2]],
        np.random.default_rng(5).uniform(size=(200, 2)) * 0.004 + [0.003, 0.003],
    ]
    count = len(xy)
    vertices = np.r_[np.c_[xy, np.full(count, -0.003)], np.c_[xy, np.full(count, 0.003)]]
    triangles = Delaunay(xy).simplices
    faces = np.r_[triangles[:, ::-1], triangles + count,
                  [[i, j, j + count] for i, j in [(0, 1), (1, 2), (2, 0)]],
                  [[i, j + count, i + count] for i, j in [(0, 1), (1, 2), (2, 0)]]]
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    rotation = np.array([
        [0.58212667, 0.58020611, -0.56963971],
        [-0.04389914, 0.72198443, 0.69051527],
        [0.81191219, -0.37696067, 0.44575695],
    ])
    mesh.vertices = np.asarray(mesh.vertices) @ rotation.T
    proof = _fit(mesh).debug_info["rim"]
    assert proof["pca_extents_m"][2] == pytest.approx(0.006, abs=1e-9)
    assert proof["containment_margin_m"] > 0


@pytest.mark.parametrize("constraint", ["clip", "bottom"])
def test_rim_clipping_cannot_silently_remove_contour_coverage(constraint):
    """Shared hard clipping must retain the final rim proof or fail explicitly."""
    kwargs = (
        {"clip_plane": ([1, 0, 0], 0)}
        if constraint == "clip" else {"max_bottom_protrusion_m": 0.0}
    )
    with pytest.raises(ValueError, match="RIM.*(removed|uncovered)"):
        _fit(**kwargs)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_rim_native_bottom_bound_can_preserve_a_raised_outer_lip(dtype):
    """A raised lip can obey the object's bottom plane and retain every section."""
    lip = trimesh.creation.annulus(r_min=0.09, r_max=0.1, height=0.006, sections=32)
    lip.apply_translation([0, 0, 0.015])
    base = trimesh.creation.cylinder(radius=0.085, height=0.002, sections=32)
    base.apply_translation([0, 0, 0.001])
    mesh = trimesh.util.concatenate((lip, base))
    result = _fit(
        mesh, max_bottom_protrusion_m=0.0015, device_cfg=DeviceCfg(device="cpu", dtype=dtype)
    )
    assert bool((result.centers[:, 2].double() - result.radii.double() >= -0.0015).all())
    assert result.debug_info["rim"]["containment_margin_m"] > 0
    assert result.debug_info["rim"]["object_local_bottom_protrusion_m"] <= 0.0015


@pytest.mark.parametrize("budget", [4, 50])
def test_rim_bottom_bound_fits_flat_plate_without_shrinking_away_coverage(budget):
    """Constrained centers retain the rim with the same explicit sphere capacity."""
    mesh = _plate()
    result = _fit(mesh, num_spheres=budget, max_bottom_protrusion_m=0.0015)
    assert result.num_spheres == budget
    assert result.debug_info["rim"]["containment_margin_m"] > 0
    assert result.debug_info["rim"]["object_local_bottom_protrusion_m"] <= 0.0015
    assert not result.debug_info["fallback_used"]
    assert np.all(result.centers.numpy()[:, 2] - result.radii.numpy()
                  >= float(mesh.vertices[:, 2].min()) - 0.0015 - 1e-9)


def test_rim_bottom_bound_uses_source_bottom_for_asymmetric_curved_plate():
    """PCA extrusion below the real source bottom must not make a fit impossible."""
    mesh = trimesh.creation.annulus(r_min=0.04, r_max=0.15, height=0.006, sections=32)
    vertices = np.asarray(mesh.vertices)
    vertices[:, 2] += 0.04 * (vertices[:, 0] / 0.15) ** 2 + 0.015 * vertices[:, 0] / 0.15
    mesh.vertices = vertices
    result = _fit(mesh, max_bottom_protrusion_m=0.0015)
    proof = result.debug_info["rim"]
    bottom = float(vertices[:, 2].min())
    assert proof["outer_contour_height_band_covered"]
    assert proof["boundary_vertices_covered"]
    assert proof["containment_margin_m"] > 0
    assert proof["object_local_bottom_protrusion_m"] <= 0.0015
    assert all(np.asarray(corners)[:, 2].min() >= bottom
               for corners in proof["section_corners_m"])
    assert result.num_spheres == 50
    assert result.used_mesh is mesh


def test_rim_verifier_rejects_missing_shrunken_or_forged_geometry():
    """The final verifier checks source-bound witnesses and returned spheres."""
    mesh = _plate()
    result = _fit(mesh)
    c, r, proof = result.centers.numpy(), result.radii.numpy(), result.debug_info["rim"]
    with pytest.raises(ValueError, match="removed"):
        verify_rim_result(mesh, c[:-1], r[:-1], proof)
    with pytest.raises(ValueError, match="uncovered"):
        verify_rim_result(mesh, c, r * 0.01, proof)
    forged = copy.deepcopy(proof)
    forged["section_corners_m"][0][0][0] += 0.01
    with pytest.raises(ValueError, match="original mesh"):
        verify_rim_result(mesh, c, r, forged)


def test_rim_preserves_original_mesh_metrics_without_claiming_face_coverage():
    """Metrics measure the whole source mesh, including its uncovered face."""
    result = _fit(compute_metrics=True)
    assert result.metrics.coverage < 1.0
    assert result.metrics.max_uncovered_gap > 0.02
    assert result.metrics.volume_ratio > 0
    assert not result.debug_info["rim"]["full_volume_covered"]


def test_rim_rejects_global_outward_bound_and_unsupported_dtype():
    """A rim proof cannot promise a full-source outward-distance bound."""
    with pytest.raises(ValueError, match="requires BOUNDED"):
        _fit(max_protrusion_m=0.0015)
    with pytest.raises(ValueError, match="RIM supports"):
        _fit(device_cfg=DeviceCfg(device="cpu", dtype=torch.float16))
