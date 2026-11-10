# SPDX-License-Identifier: Apache-2.0

"""Thin tapered solids must retain an admissible interior witness center."""

import numpy as np
import pytest
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from curobo._src.geom.sphere_fit.fit_bounded import _MARGIN, _witness_ball


def _floor():
    return np.array([
        [x * half, y * half, z]
        for half, z in ((.08, 0.), (.075, .003))
        for x in (-1., 1.) for y in (-1., 1.)
    ])


@pytest.mark.parametrize("transformed", [False, True])
def test_interior_witness_keeps_valid_center_when_shifted_proposals_leave_solid(transformed):
    vertices = _floor()
    point = np.array([0., 0., .001])
    if transformed:
        rotation = Rotation.from_euler("xyz", [.3, -.2, .7]).as_matrix()
        translation = np.array([.1, -.2, .3])
        vertices = vertices @ rotation.T + translation
        point = rotation @ point + translation
    equations = ConvexHull(vertices).equations
    center, radius = _witness_ball(point, equations, point.copy(), .0149)
    inside = float(-(equations[:, :3] @ center + equations[:, 3]).max())
    assert inside >= 0.
    assert radius - inside <= .0149
    assert radius - np.linalg.norm(center - point) > _MARGIN
    np.testing.assert_array_equal(center, center.astype(np.float32).astype(float))


def test_interior_candidate_cannot_admit_a_witness_outside_the_source_hull():
    equations = ConvexHull(_floor()).equations
    with pytest.raises(ValueError, match="witness lies outside"):
        _witness_ball(np.array([0., 0., -.001]), equations, np.array([0., 0., .001]), .0149)
