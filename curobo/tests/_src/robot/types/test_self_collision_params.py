# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Self-collision pair enumeration preserves the exact upper-triangle policy."""

from __future__ import annotations

import pytest
import torch

from curobo._src.robot.types.self_collision_params import SelfCollisionKinematicsCfg


@pytest.mark.parametrize("transpose", [False, True])
def test_pair_order_and_exclusions(transpose: bool) -> None:
    """Asymmetric and noncontiguous inputs retain lexicographic pair ordering."""
    generator = torch.Generator().manual_seed(29)
    distances = torch.rand((19, 19), generator=generator, dtype=torch.float32)
    distances[distances < 0.6] = -torch.inf
    distances[4] = -torch.inf
    distances[1, 7] = torch.inf
    distances[2, 8] = torch.nan
    if transpose:
        distances = distances.T
    padding = torch.arange(19, dtype=torch.float32) / 1000
    expected = [(i, j) for i in range(19) for j in range(i + 1, 19)
                if distances[i, j] != -torch.inf]

    result = SelfCollisionKinematicsCfg.create_from_sphere_pair_distances(distances, padding)

    assert result.collision_pairs.tolist() == [list(pair) for pair in expected]
    assert result.collision_pairs.dtype == torch.int16
    assert result.collision_pairs.device == distances.device
    assert result.collision_pairs.is_contiguous()
    assert result.sphere_padding is padding
    assert result.num_spheres == 19


@pytest.mark.parametrize("count", [0, 1, 8])
def test_no_collision_pairs(count: int) -> None:
    """Models with no eligible sphere pair return an empty two-column tensor."""
    distances = torch.full((count, count), -torch.inf, dtype=torch.float32)
    result = SelfCollisionKinematicsCfg.create_from_sphere_pair_distances(
        distances, torch.zeros(count, dtype=torch.float32)
    )
    assert result.collision_pairs.shape == (0, 2)
    assert result.collision_pairs.dtype == torch.int16


@pytest.mark.parametrize("shape,padding_shape", [((2, 3), (2,)), ((2, 2), (2, 1))])
def test_incompatible_shapes(shape: tuple[int, ...], padding_shape: tuple[int, ...]) -> None:
    """Reject malformed distance and padding dimensions before enumeration."""
    with pytest.raises(ValueError):
        SelfCollisionKinematicsCfg.create_from_sphere_pair_distances(
            torch.zeros(shape, dtype=torch.float32),
            torch.zeros(padding_shape, dtype=torch.float32),
        )


def test_rejects_unrepresentable_sphere_indices() -> None:
    """An int16 pair representation cannot silently wrap oversized indices."""
    count = torch.iinfo(torch.int16).max + 2
    with pytest.raises(ValueError, match="int16"):
        SelfCollisionKinematicsCfg.create_from_sphere_pair_distances(
            torch.empty((count, count), device="meta", dtype=torch.float32),
            torch.empty(count, device="meta", dtype=torch.float32),
        )
