# SPDX-License-Identifier: Apache-2.0
"""Wide thin collision solids need economical, complete volume covers."""

import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from curobo._src.geom.sphere_fit.fit_bounded import bounded_fit_mesh, verify_bounded_result


@pytest.mark.parametrize("transformed", [False, True])
def test_wide_slab_fits_capacity_after_float32_conversion(transformed):
    mesh = trimesh.creation.box([.20, .20, .003])
    if transformed:
        mesh.apply_transform(trimesh.transformations.translation_matrix([.1, -.2, .3]))
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_euler("xyz", [.3, -.2, .7]).as_matrix()
        mesh.apply_transform(transform)
    centers, radii, proof = bounded_fit_mesh(mesh, 100, .0149)
    assert 0 < len(radii) <= 100
    verify_bounded_result(mesh, centers.astype(np.float32), radii.astype(np.float32), proof)
    assert proof["full_volume_covered"] is True
    # A certificate must reject a hole in the volume, even with unchanged owners.
    with pytest.raises(ValueError):
        verify_bounded_result(mesh, centers, radii * .1, proof)


def test_box_tiles_never_truncate_to_an_impossible_capacity():
    mesh = trimesh.creation.box([.20, .20, .003])
    with pytest.raises(ValueError, match="capacity|budget"):
        bounded_fit_mesh(mesh, 1, .0149)


@pytest.mark.parametrize("transformed", [False, True])
def test_checked_box_tiles_verify_a_tapered_wall_after_public_conversion(transformed):
    """A different tetrahedral partition must not reject a valid box cover."""
    from curobo.sphere_fit import SphereFitType, fit_spheres_to_mesh
    from curobo.types import DeviceCfg

    profile = np.asarray([
        [.0864974226, 0.], [.094, .028], [.090, .028], [.0833012702, .003],
    ], dtype=np.float32)
    points = np.asarray([[x, y, z] for x, z in profile for y in (-.08, .08)], dtype=np.float32)
    wall = trimesh.convex.convex_hull(points)
    floor = trimesh.creation.box([.16, .16, .003])
    floor.apply_translation([0., 0., .0015])
    mesh = trimesh.util.concatenate([floor, wall])
    if transformed:
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_euler("xyz", [.3, -.2, .7]).as_matrix()
        pose[:3, 3] = [.1, -.2, .3]
        mesh.apply_transform(pose)
    result = fit_spheres_to_mesh(
        mesh, num_spheres=200, fit_type=SphereFitType.BOUNDED,
        max_protrusion_m=.0149, max_bottom_protrusion_m=.0149,
        compute_metrics=False, device_cfg=DeviceCfg(device="cpu"),
    )
    assert result.debug_info["bounded"]["full_volume_covered"] is True
    assert result.num_spheres <= 200
    with pytest.raises(ValueError):
        verify_bounded_result(mesh, result.centers.numpy(), result.radii.numpy() * .1,
                              result.debug_info["bounded"])
