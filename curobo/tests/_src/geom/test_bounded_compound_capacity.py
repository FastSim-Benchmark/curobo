# SPDX-License-Identifier: Apache-2.0
"""A larger capacity must not discard sharing across overlapping components."""

import pytest
import trimesh

from curobo._src.geom.sphere_fit.fit_bounded import bounded_fit_mesh, verify_bounded_result


@pytest.mark.parametrize("capacity", [50, 100, 200])
def test_overlapping_components_share_coverage_above_fifty(capacity):
    # Multiple collision parts can describe the same source volume. Reserving
    # one independent budget per part wastes capacity without improving coverage.
    mesh = trimesh.util.concatenate([trimesh.creation.box([0.01] * 3) for _ in range(101)])
    centers, radii, proof = bounded_fit_mesh(mesh, capacity, 0.0149)
    assert len(radii) == 1
    verify_bounded_result(mesh, centers, radii, proof)


def test_large_capacity_preserves_empty_space_between_components():
    left = trimesh.creation.box([0.01] * 3)
    right = left.copy()
    left.apply_translation([-0.04, 0, 0])
    right.apply_translation([0.04, 0, 0])
    mesh = trimesh.util.concatenate([left, right])
    centers, radii, proof = bounded_fit_mesh(mesh, 100, 0.002)
    verify_bounded_result(mesh, centers, radii, proof)
    assert all(abs(center[0]) > radius + 0.02 for center, radius in zip(centers, radii))
