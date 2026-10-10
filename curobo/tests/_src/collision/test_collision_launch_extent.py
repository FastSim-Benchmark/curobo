# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Collision queries retain their last sphere beyond a signed 32-bit pair count."""

import pytest
import torch

from curobo._src.geom.collision.buffer_collision import CollisionBuffer
from curobo._src.geom.collision.collision_scene import SceneCollision, SceneCollisionCfg
from curobo._src.geom.types import Cuboid, SceneCfg
from curobo._src.types.device_cfg import DeviceCfg


@pytest.mark.parametrize("swept", [False, True])
def test_last_sphere_collision_above_int32_pair_count(
    cuda_device_cfg: DeviceCfg, swept: bool
) -> None:
    """Only the final sphere is active; compare its cost and gradient to a small query."""
    capacity = 4096
    horizon = 2 if swept else 1
    count = (2**31 // capacity // horizon) + 1
    assert horizon * count * capacity > 2**31
    checker = SceneCollision.from_config(SceneCollisionCfg(
        device_cfg=cuda_device_cfg,
        scene_model=SceneCfg(cuboid=[Cuboid(
            name="box", dims=[0.1, 0.1, 0.1], pose=[0, 0, 0, 1, 0, 0, 0]
        )]),
        cache={"cuboid": capacity},
    ))
    spheres = torch.zeros((1, horizon, count, 4), device=cuda_device_cfg.device)
    spheres[..., 3] = -1.0
    spheres[:, :, -1, 0] = 0.06
    spheres[:, :, -1, 3] = 0.05
    weight = torch.ones(1, device=cuda_device_cfg.device)
    eta = torch.full((1,), 0.01, device=cuda_device_cfg.device)

    def query(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        buffer = CollisionBuffer.from_shape(values.shape, cuda_device_cfg)
        if swept:
            costs = checker.checker.get_swept_sphere_distance(
                scene=checker.data, query_sphere=values, collision_buffer=buffer,
                weight=weight, activation_distance=eta,
                trajectory_dt=torch.full((1,), 0.02, device=cuda_device_cfg.device),
            )
        else:
            costs = checker.get_sphere_distance_raw(
                query_spheres=values, collision_buffer=buffer,
                weight=weight, activation_distance=eta,
            )
        return costs.clone(), buffer.gradient.clone()

    expected_cost, expected_grad = query(spheres[:, :, -1:].contiguous())
    actual_cost, actual_grad = query(spheres)
    torch.cuda.synchronize()
    assert bool((expected_cost > 0).all())
    torch.testing.assert_close(actual_cost[:, :, -1:], expected_cost)
    torch.testing.assert_close(actual_grad[:, :, -1:], expected_grad)
    assert int(torch.count_nonzero(actual_cost[:, :, :-1])) == 0
    assert int(torch.count_nonzero(actual_grad[:, :, :-1])) == 0
