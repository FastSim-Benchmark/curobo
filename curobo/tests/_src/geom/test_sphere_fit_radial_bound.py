# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import trimesh

from curobo._src.geom.sphere_fit.fit_bounded import _radial_hull_excess as radial_bound


def dented_box(depth):
    mesh = trimesh.creation.box([.02] * 3)
    top = np.all(mesh.triangles[:, :, 2] > 0, axis=1)
    faces = mesh.faces[~top].tolist()
    for face in mesh.faces[top]:
        for i, j in zip(face, np.roll(face, -1), strict=True):
            if np.count_nonzero(mesh.vertices[i, :2] != mesh.vertices[j, :2]) == 1:
                faces.append([int(i), int(j), len(mesh.vertices)])
    return trimesh.Trimesh(np.vstack([mesh.vertices, [0, 0, .01 - depth]]), faces)


def test_convex_box_requires_only_arithmetic_reserve():
    assert radial_bound(trimesh.creation.box([.02] * 3), np.zeros(3)) <= 1.00001e-7


@pytest.mark.parametrize('depth', [.0002, .002, .005])
def test_known_dent_distance_is_bounded(depth):
    mesh = dented_box(depth)
    value = radial_bound(mesh, np.zeros(3))
    # Hull top-center has exact distance depth/sqrt(1+(depth/.01)**2)
    # from any of the four symmetric sloped source faces.
    exact_distance = depth / np.sqrt(1 + (depth / .01) ** 2)
    assert exact_distance <= value <= depth + 1.01e-7


@pytest.mark.parametrize('origin', [[1, 0, 0], [0, 0, .01], [np.nan, 0, 0]])
def test_invalid_kernel_origin_rejected(origin):
    with pytest.raises(ValueError):
        radial_bound(trimesh.creation.box([.02] * 3), origin)


def test_open_source_rejected():
    mesh = trimesh.creation.box([.02] * 3)
    mesh.update_faces(np.arange(len(mesh.faces) - 1))
    with pytest.raises(ValueError, match='closed outward'):
        radial_bound(mesh, np.zeros(3))


def test_shallow_asymmetric_dent_fits_without_relaxing_outward_bound():
    from curobo.sphere_fit import SphereFitType, fit_spheres_to_mesh
    from curobo.types import DeviceCfg

    mesh = trimesh.creation.box([.08, .035, .01])
    top = np.all(mesh.triangles[:, :, 2] > 0, axis=1)
    faces = mesh.faces[~top].tolist()
    for face in mesh.faces[top]:
        for i, j in zip(face, np.roll(face, -1), strict=True):
            if np.count_nonzero(mesh.vertices[i, :2] != mesh.vertices[j, :2]) == 1:
                faces.append([int(i), int(j), len(mesh.vertices)])
    source = trimesh.Trimesh(np.vstack([mesh.vertices, [.038, .015, .0048]]), faces)
    result = fit_spheres_to_mesh(
        source, num_spheres=2048, fit_type=SphereFitType.BOUNDED,
        max_protrusion_m=.002, device_cfg=DeviceCfg(device="cpu"),
    )
    proof = result.debug_info["bounded"]
    assert proof["full_volume_covered"]
    assert proof["verified_max_protrusion_m"] <= .002
    assert result.num_spheres <= 2048
