# SPDX-License-Identifier: Apache-2.0

"""Full-volume and local-clearance contracts for attached-object proxies."""

import numpy as np
import pytest
import torch
import trimesh

from curobo._src.geom.sphere_fit.fit_bounded import _hull_excess, verify_bounded_result
from curobo.sphere_fit import SphereFitType, fit_spheres_to_mesh
from curobo.types import DeviceCfg


def fit(mesh, **kwargs):
    return fit_spheres_to_mesh(
        mesh, fit_type=SphereFitType.BOUNDED, device_cfg=DeviceCfg(device="cpu"),
        **{"num_spheres": 1024, "max_protrusion_m": 0.002, **kwargs},
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_two_solids_keep_empty_gap_and_cover_complete_returned_volume(dtype):
    left, right = trimesh.creation.box([0.01, 0.014, 0.012]), trimesh.creation.box([0.01]*3)
    left.apply_translation([-0.016, 0, 0])
    right.apply_translation([0.016, 0, 0])
    mesh = trimesh.util.concatenate([left, right])
    result = fit_spheres_to_mesh(
        mesh, num_spheres=1024, fit_type=SphereFitType.BOUNDED,
        max_protrusion_m=0.002, max_bottom_protrusion_m=0.002,
        device_cfg=DeviceCfg(device="cpu", dtype=dtype),
    )
    proof = result.debug_info["bounded"]
    assert proof["full_volume_covered"]
    assert proof["verified_max_protrusion_m"] <= 0.002
    assert result.centers.dtype == result.radii.dtype == dtype
    assert not result.debug_info["fallback_used"]
    assert result.num_spheres <= 1024
    # A probe in the original inter-component gap must not collide with a ball.
    assert bool((torch.linalg.vector_norm(result.centers, dim=1) > result.radii+0.005).all())
    verify_bounded_result(mesh, result.centers.numpy(), result.radii.numpy(), proof)


@pytest.mark.parametrize("kwargs", [
    {"num_spheres": None}, {"num_spheres": True}, {"num_spheres": 0},
    {"num_spheres": 16385}, {"max_protrusion_m": None},
    {"max_protrusion_m": True}, {"max_protrusion_m": 0},
    {"max_protrusion_m": float("nan")},
])
def test_invalid_capacity_or_distance_rejected(kwargs):
    with pytest.raises(ValueError, match="BOUNDED"):
        fit(trimesh.creation.box([0.01]*3), **kwargs)


def test_insufficient_capacity_is_not_silent_truncation():
    with pytest.raises(ValueError, match="budget|capacity"):
        fit(trimesh.creation.box([0.02]*3), num_spheres=1)


def test_open_mesh_cannot_claim_volume_coverage():
    mesh = trimesh.creation.box([0.01]*3)
    mesh.update_faces(np.arange(len(mesh.faces)-1))
    with pytest.raises(ValueError, match="closed"):
        fit(mesh)


def test_clipping_cannot_invalidate_returned_volume_certificate():
    with pytest.raises(ValueError, match="BOUNDED"):
        fit(trimesh.creation.box([0.01]*3), clip_plane=([0, 0, 1], 0))


def test_post_conversion_verifier_rejects_inflated_or_missing_geometry():
    mesh = trimesh.creation.box([0.01]*3)
    result = fit(mesh)
    c, r, proof = result.centers.numpy(), result.radii.numpy(), result.debug_info["bounded"]
    with pytest.raises(ValueError, match="outward"):
        verify_bounded_result(mesh, c, r+0.01, proof)
    with pytest.raises(ValueError, match="uncovered"):
        verify_bounded_result(mesh, c, r*0.01, proof)


def test_hull_excess_is_accounted_for_before_sphere_fitting():
    mesh = trimesh.creation.box([0.02]*3)
    top = np.all(mesh.triangles[:, :, 2] > 0, axis=1)
    faces = mesh.faces[~top].tolist()
    for face in mesh.faces[top]:
        for i, j in zip(face, np.roll(face, -1), strict=True):
            # Omit the interior diagonal shared by the original top triangles.
            if np.count_nonzero(mesh.vertices[i, :2] != mesh.vertices[j, :2]) == 1:
                faces.append([int(i), int(j), len(mesh.vertices)])
    mesh = trimesh.Trimesh(vertices=np.vstack([mesh.vertices, [0, 0, 0.0098]]), faces=faces)
    assert mesh.is_watertight and mesh.is_winding_consistent
    _, excess = _hull_excess(mesh)
    assert excess > 0.0001
    result = fit(mesh, num_spheres=2048)
    assert result.debug_info["bounded"]["full_volume_covered"]
    assert result.debug_info["bounded"]["verified_max_protrusion_m"] <= 0.002


def test_fast_does_not_silently_ignore_requested_bound():
    with pytest.raises(ValueError, match="requires BOUNDED"):
        fit_spheres_to_mesh(trimesh.creation.box(), max_protrusion_m=0.002)


@pytest.mark.parametrize("transformed", [False, True])
def test_thin_phone_sized_box_has_bounded_refinement(monkeypatch, transformed):
    """Cover a thin solid without exhausting refinement on sphere seams."""
    import curobo._src.geom.sphere_fit.fit_bounded as module

    monkeypatch.setattr(module, "_MAX_CELLS", 20000)
    mesh = trimesh.creation.box([0.0705, 0.130, 0.0085])
    if transformed:
        mesh.apply_transform(trimesh.transformations.rotation_matrix(0.7, [1, 2, 3]))
        mesh.apply_translation([0.1, -0.2, 0.3])
    result = fit(mesh, num_spheres=8192, max_protrusion_m=0.0005, compute_metrics=False)
    assert result.debug_info["bounded"]["full_volume_covered"]
    assert result.debug_info["bounded"]["verified_max_protrusion_m"] <= 0.0005
    assert result.num_spheres <= 8192
    assert not result.debug_info["fallback_used"]


def test_skewed_hexahedron_is_not_treated_as_rectangular_volume():
    """A near-box cannot replace the actual source with an approximate OBB."""
    import curobo._src.geom.sphere_fit.fit_bounded as module

    mesh = trimesh.creation.box([0.01, 0.02, 0.003])
    mesh.vertices[:, 0] += mesh.vertices[:, 2] * 0.1
    assert module._volume_cells(mesh).shape == (12, 4, 3)
    result = fit(mesh)
    assert result.debug_info["bounded"]["full_volume_covered"]


@pytest.mark.parametrize("reverse", [False, True])
def test_many_thin_components_preserve_ownership_and_gap(reverse):
    """Keep each of 44 generated solids independently certified."""
    parts = []
    for index in range(44):
        part = trimesh.creation.box([0.003, 0.008, 0.0006])
        part.apply_translation([index * 0.006, 0, 0])
        parts.append(part)
    mesh = trimesh.util.concatenate(parts[::-1] if reverse else parts)
    result = fit(mesh, num_spheres=8192, max_protrusion_m=0.0005, compute_metrics=False)
    proof = result.debug_info["bounded"]
    assert proof["full_volume_covered"]
    assert len(proof["components"]) == 44
    assert set(proof["sphere_components"]) == set(range(44))
    gap = torch.tensor([0.003, 0, 0])
    assert (torch.linalg.vector_norm(result.centers - gap, dim=1) > result.radii).all()


@pytest.mark.parametrize("masks", [
    [0b0011, 0b1100, 0b0101, 0b1000],
    [0b0111, 0b1011, 0b1100, 0b1000, 0],
    [0b0001, 0b0010, 0b0100, 0b1000],
])
def test_lazy_compression_preserves_greedy_selection(monkeypatch, masks):
    """Stale heap bounds preserve all cells and the original tie order."""
    import curobo._src.geom.sphere_fit.fit_bounded as module

    monkeypatch.setattr(module, "_cover_masks", lambda *_: (masks, 4))
    uncovered, expected = 0b1111, []
    while uncovered:
        index = max(range(len(masks)), key=lambda i: (masks[i] & uncovered).bit_count())
        expected.append(index)
        uncovered &= ~masks[index]
    centers = np.arange(len(masks) * 3).reshape(-1, 3)
    selected, _ = module._compress(None, centers, np.ones(len(masks)))
    np.testing.assert_array_equal(selected, centers[expected])


def test_compression_cannot_accept_incomplete_cell_masks(monkeypatch):
    """An exhausted heap cannot silently discard an uncovered volume cell."""
    import curobo._src.geom.sphere_fit.fit_bounded as module

    monkeypatch.setattr(module, "_cover_masks", lambda *_: ([0b0011], 4))
    with pytest.raises(ValueError, match="lost a certified cell"):
        module._compress(None, np.zeros((1, 3)), np.ones(1))


@pytest.mark.parametrize("outward,accepted", [(0.002, True), (0.00001, False)])
def test_depth_terminal_ball_still_obeys_outward_constraint(monkeypatch, outward, accepted):
    import curobo._src.geom.sphere_fit.fit_bounded as module

    monkeypatch.setattr(module, "_MAX_DEPTH", 0)
    monkeypatch.setattr(module, "_witness_ball", lambda p, *_: (p, 1e-6))
    mesh = trimesh.creation.box([.001]*3)
    if not accepted:
        with pytest.raises(ValueError, match="outward bound"):
            module._cover(mesh, outward, 256)
    else:
        c, r = module._cover(mesh, outward, 256)
        _, count = module._cover_masks(mesh, c, r)
        assert count == len(module._volume_cells(mesh))
