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
        assert count == len(mesh.faces)
