# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare enabled-slot dispatch to full-capacity queries, including graph replay."""

import pytest
import torch
import warp as wp

from curobo._src.geom.collision.buffer_collision import CollisionBuffer
from curobo._src.geom.collision.checker_collision import CollisionChecker
from curobo._src.geom.collision.wp_collision_kernel import sphere_obstacle_collision_kernel
from curobo._src.geom.collision.wp_speed_metric import apply_speed_metric
from curobo._src.geom.collision.wp_sweep_collision_kernel import (
    swept_sphere_obstacle_collision_kernel,
)
from curobo._src.geom.data.data_scene import SceneData
from curobo._src.geom.types import Cuboid
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.warp import get_warp_device_stream


def _scene(kind, capacity=4096):
    cfg = DeviceCfg(device=torch.device("cuda:0"))
    kwargs = {f"{kind}_cache": capacity}
    if kind == "voxel":
        kwargs = {"voxel_cache": {"layers": capacity, "dims": [0.2] * 3, "voxel_size": 0.1}}
    scene = SceneData.create_cache(2, cfg, **kwargs)
    data = scene.get_valid_data()[0]
    for env in range(2):
        box = Cuboid(name=f"box{env}", pose=[0, 0, 0, 1, 0, 0, 0], dims=[0.2] * 3)
        if kind == "voxel":
            data.params[..., :3] = 2
            data.params[..., 3] = 0.1
            data.dims[..., :3] = 0.2
            data.features[:] = -0.01
        else:
            data.load_batch([box.get_mesh() if kind == "mesh" else box], env)
        # Keep valid geometry even at the last slot. No CPU name/count assumption
        # is allowed to hide valid GPU enable changes in the query path.
        for key, value in vars(data).items():
            if (
                isinstance(value, torch.Tensor)
                and value.ndim >= 2
                and value.shape[:2] == (2, capacity)
            ):
                value[env] = value[env, :1].expand_as(value[env]).clone()
    data.enable.zero_()
    data.count[:] = capacity
    return scene, data, cfg


def _inputs(cfg):
    generator = torch.Generator(device="cuda:0").manual_seed(42)
    q = torch.rand((2, 5, 17, 4), generator=generator, device="cuda:0") * 0.04
    q[..., 0] += 0.09
    q[..., 3] = 0.05
    q[:, :, -1, 3] = -1  # Disabled query spheres retain zero cost/gradient.
    q.requires_grad_(True)
    env = torch.tensor([1, 0], device="cuda:0", dtype=torch.int32)
    weight, eta, dt = [cfg.to_device([v]) for v in (2.0, 0.02, 0.05)]
    pairs = torch.full((2, 17, 2), -1, device="cuda:0", dtype=torch.int32)
    return q, env, weight, eta, dt, pairs


def _dense(scene, q, env, weight, eta, dt, pairs, swept, speed=False):
    """Launch the unchanged full-capacity pair dispatch as the numeric reference."""
    buffer = CollisionBuffer.from_shape(q.shape, scene.device_cfg)
    device, stream = get_warp_device_stream(q)
    b, h, n, _ = q.shape
    spheres = wp.from_torch(q.detach().view(-1, 4), dtype=wp.vec4)
    cost = wp.from_torch(buffer.distance.view(-1))
    gradient = wp.from_torch(buffer.gradient.view(-1))
    for data in scene.get_valid_data():
        kernel = (
            swept_sphere_obstacle_collision_kernel if swept else sphere_obstacle_collision_kernel
        )
        wp.launch(
            kernel,
            dim=b * h * n * data.max_n,
            inputs=[
                data.to_warp(),
                spheres,
                wp.from_torch(weight),
                wp.from_torch(eta),
                wp.from_torch(env),
                cost,
                gradient,
                b,
                h,
                n,
                data.max_n,
                wp.uint8(1),
                wp.from_torch(pairs.view(-1)),
                0 if data is scene.voxels else 2,
            ],
            device=device,
            stream=stream,
        )
    if swept and speed:
        wp.launch(
            apply_speed_metric,
            dim=b * h * n,
            inputs=[spheres, cost, gradient, wp.from_torch(dt), b, h, n],
            device=device,
            stream=stream,
        )
    return buffer.distance, buffer.gradient


def _query(scene, q, env, weight, eta, dt, pairs, swept, speed=False, buffer=None, checker=None):
    checker = checker or CollisionChecker(scene.device_cfg)
    buffer = buffer or CollisionBuffer.from_shape(q.shape, scene.device_cfg)
    kwargs = dict(
        env_query_idx=env,
        return_loss=True,
        replacement_cuboid_ids=pairs,
        replacement_mesh_ids=pairs,
    )
    if swept:
        return checker.get_swept_sphere_distance(
            scene, q, buffer, weight, eta, dt, enable_speed_metric=speed, **kwargs
        )
    return checker.get_sphere_distance(scene, q, buffer, weight, eta, **kwargs)


@pytest.mark.parametrize("kind", ["cuboid", "mesh", "voxel"])
@pytest.mark.parametrize("swept,speed", [(False, False), (True, False), (True, True)])
def test_enabled_dispatch_matches_dense_after_dynamic_flag_changes(kind, swept, speed):
    """Empty, sparse, strided, full and re-enabled scenes retain every valid pair."""
    scene, data, cfg = _scene(kind)
    args = _inputs(cfg)
    q, env, weight, eta, dt, pairs = args
    pairs[:, :3, 0] = data.max_n - 1
    pairs[:, :3, 1] = 0
    for slots in (
        [],
        [data.max_n - 1],
        list(range(0, data.max_n, 31)),
        list(range(data.max_n)),
        [],
        [0, 127, 128, 511, 512, 513, data.max_n - 1],
    ):
        data.enable.zero_()
        data.enable[0, slots] = 1
        data.enable[1, list(reversed(slots))[::2]] = 1
        expected, grad = _dense(scene, *args, swept, speed)
        result = _query(scene, *args, swept, speed)
        scale = torch.linspace(0.2, 1.5, result.numel(), device=q.device).reshape_as(result)
        actual_grad = torch.autograd.grad(result, q, grad_outputs=scale)[0]
        torch.testing.assert_close(result, expected, rtol=3e-5, atol=2e-6)
        torch.testing.assert_close(actual_grad, grad * scale.unsqueeze(-1), rtol=3e-5, atol=2e-6)
        assert not result[..., -1].any()
        assert not actual_grad[..., -1, :].any()
        if not slots:
            assert not result.any()
        else:
            assert result.max() > 0


@pytest.mark.parametrize("kind", ["cuboid", "mesh", "voxel"])
@pytest.mark.parametrize("swept", [False, True])
def test_graph_replay_reads_current_flags_and_original_slot_replacement_ids(kind, swept):
    """A graph captured while empty must see later high-slot updates and removals."""
    scene, data, cfg = _scene(kind)
    args = _inputs(cfg)
    q, env, weight, eta, dt, pairs = args
    buffer = CollisionBuffer.from_shape(q.shape, cfg)
    checker = CollisionChecker(cfg)
    for _ in range(3):
        _query(scene, *args, swept, buffer=buffer)
        _dense(scene, *args, swept)
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        result = _query(scene, *args, swept, buffer=buffer, checker=checker)
    for count, slots in (
        (data.max_n, [data.max_n - 1]),
        (129, list(range(129))),
        (0, [data.max_n - 1]),
        (data.max_n, [1, 128, 511, 512, 513, data.max_n - 1]),
    ):
        data.count[:] = count
        data.enable.zero_()
        data.enable[:, slots] = 1
        pairs[:, :5, 0] = data.max_n - 1
        expected, expected_grad = _dense(scene, *args, swept)
        graph.replay()
        torch.testing.assert_close(result, expected, rtol=3e-5, atol=2e-6)
        torch.testing.assert_close(buffer.gradient, expected_grad, rtol=3e-5, atol=2e-6)


@pytest.mark.parametrize("capacity", [1, 7, 127, 128, 129, 511, 512, 513])
def test_small_and_non_multiple_capacities(capacity):
    """Direct and tail dispatch must not read outside allocated obstacle storage."""
    scene, data, cfg = _scene("cuboid", capacity)
    data.enable[:, -1] = 1
    args = _inputs(cfg)
    expected, _ = _dense(scene, *args, False)
    torch.testing.assert_close(_query(scene, *args, False), expected, rtol=0, atol=0)
