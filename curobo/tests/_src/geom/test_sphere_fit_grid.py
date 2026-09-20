# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounding-box seeds remain internal even for thin collision geometry."""

import numpy as np
import pytest
import trimesh

from curobo._src.geom.sphere_fit.fit_voxel import _build_bbox_grid


@pytest.mark.parametrize("thin_axis", [0, 1, 2])
@pytest.mark.parametrize("count", [1, 32, 256])
def test_thin_box_seeds_stay_inside_and_centered(thin_axis, count):
    extents = np.full(3, 0.2)
    extents[thin_axis] = 0.001
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation([0.3, -0.7, 1.2])
    points = _build_bbox_grid(mesh, count)
    assert len(points) > 0
    assert np.all(points > mesh.bounds[0])
    assert np.all(points < mesh.bounds[1])
    np.testing.assert_allclose(points.mean(axis=0), mesh.bounds.mean(axis=0), atol=1e-12)
    np.testing.assert_allclose(points[:, thin_axis], mesh.bounds.mean(axis=0)[thin_axis])


def test_single_cube_cell_is_at_center():
    mesh = trimesh.creation.box(extents=[0.2, 0.2, 0.2])
    np.testing.assert_allclose(_build_bbox_grid(mesh, 1), [[0.0, 0.0, 0.0]], atol=1e-15)


def test_zero_volume_mesh_has_no_seeds():
    mesh = trimesh.creation.box(extents=[0.2, 0.2, 0.0])
    assert _build_bbox_grid(mesh, 32).shape == (0, 3)
