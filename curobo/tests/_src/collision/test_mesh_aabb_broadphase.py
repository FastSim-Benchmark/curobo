"""Mesh broad phase must preserve the unpruned collision cost and gradient."""

import pytest
import torch
import warp as wp

from curobo._src.geom.collision.buffer_collision import CollisionBuffer
from curobo._src.geom.collision.checker_collision import CollisionChecker
from curobo._src.geom.collision.wp_collision_kernel import query_aabb_overlaps
from curobo._src.geom.data.data_mesh import MeshDataWarp
from curobo._src.geom.data.data_scene import SceneData
from curobo._src.geom.types import Cuboid, Mesh


def _box(name, offset=0.0, pose=None):
    mesh = Cuboid(name=name, dims=[0.2, 0.3, 0.4], pose=[0, 0, 0, 1, 0, 0, 0]).get_trimesh_mesh()
    mesh.vertices[:, 0] += offset
    return Mesh(name=name, vertices=mesh.vertices.tolist(), faces=mesh.faces.tolist(),
                pose=pose or [0, 0, 0, 1, 0, 0, 0])


def _query(scene, q, swept, env=None, speed=False, buffer=None):
    cfg = scene.device_cfg
    buffer = buffer or CollisionBuffer.from_shape(q.shape, cfg)
    checker = CollisionChecker(cfg)
    args = (scene, q, buffer, cfg.to_device([1.0]), cfg.to_device([0.01]))
    if swept:
        result = checker.get_swept_sphere_distance(
            *args, cfg.to_device([0.05]), env_query_idx=env, enable_speed_metric=speed)
    else:
        result = checker.get_sphere_distance(*args, env_query_idx=env)
    return result.clone(), buffer.gradient.clone()


def _compare(scene, q, swept, env=None, speed=False):
    actual = _query(scene, q, swept, env, speed)
    bounds = scene.meshes.aabb.clone()
    try:
        # Infinite bounds disable only the broad phase; SDF and sweep are identical.
        scene.meshes.aabb[..., :3] = -torch.inf
        scene.meshes.aabb[..., 3:] = torch.inf
        reference = _query(scene, q, swept, env, speed)
    finally:
        scene.meshes.aabb.copy_(bounds)
    for observed, expected in zip(actual, reference):
        torch.testing.assert_close(observed, expected, atol=1e-6, rtol=1e-5)
    return actual


@pytest.mark.parametrize("swept,speed", [(False, False), (True, False), (True, True)])
def test_mesh_bounds_preserve_random_cost_and_gradient(cuda_device_cfg, swept, speed):
    scene = SceneData.create_cache(2, cuda_device_cfg, mesh_cache=8)
    scene.meshes.load_batch(
        [_box("offset", 1.0), _box("rotated", pose=[0.4, 0, 0, .8, 0, 0, .6])], 0)
    scene.meshes.load_batch([_box("other", -0.4)], 1)
    generator = torch.Generator(device="cuda:0").manual_seed(76)
    q = torch.rand((2, 7, 300, 4), device="cuda:0", generator=generator) * 2 - 1
    q[..., 3] = 0.06
    q[:, :, -1, 3] = -1
    env = torch.tensor([1, 0], device="cuda:0", dtype=torch.int32)
    result, _ = _compare(scene, q, swept, env, speed)
    assert torch.count_nonzero(result) > 0


@wp.kernel
def _crossing_bounds(data: MeshDataWarp, output: wp.array(dtype=wp.int32)):
    lower = wp.vec3(-0.2, 0.0, 0.0)
    upper = wp.vec3(0.2, 0.0, 0.0)
    output[0] = wp.int32(query_aabb_overlaps(data, 0, 0, lower, upper, 0.03))
    output[1] = wp.int32(query_aabb_overlaps(data, 0, 0, lower, lower, 0.03))
    output[2] = wp.int32(query_aabb_overlaps(data, 0, 0, upper, upper, 0.03))


def test_swept_bounds_retain_thin_wall_crossing(cuda_device_cfg):
    wall = Mesh(name="wall", vertices=[[0, -1, -1], [0, 1, -1], [0, 1, 1], [0, -1, 1]],
                faces=[[0, 1, 2], [0, 2, 3]], pose=[0, 0, 0, 1, 0, 0, 0])
    scene = SceneData.create_cache(1, cuda_device_cfg, mesh_cache=1)
    scene.meshes.load_batch([wall], 0)
    q = cuda_device_cfg.to_device([[[[-.2, 0, 0, .02]], [[.2, 0, 0, .02]]]])
    endpoint, _ = _compare(scene, q, False)
    _compare(scene, q, True)
    assert torch.count_nonzero(endpoint) == 0
    output = torch.empty(3, device="cuda:0", dtype=torch.int32)
    wp.launch(_crossing_bounds, dim=1, inputs=[scene.meshes.to_warp(), wp.from_torch(output)],
              device="cuda:0")
    assert output.tolist() == [1, 0, 0]


def test_mesh_bounds_follow_replace_remove_and_reload(cuda_device_cfg):
    scene = SceneData.create_cache(1, cuda_device_cfg, mesh_cache=3)
    data = scene.meshes
    address = data.aabb.data_ptr()
    data.load_batch([_box("first", 1), _box("second", -1)], 0)
    q = cuda_device_cfg.to_device([[[[1, 0, 0, .02], [-1, 0, 0, .02], [2, 0, 0, .02]]]])
    assert (_compare(scene, q, False)[0][0, 0, :2] > 0).all()
    data.remove("first")
    assert _compare(scene, q, False)[0][0, 0, 1] > 0
    data.replace(_box("second", 2))
    assert _compare(scene, q, False)[0][0, 0, 2] > 0
    data.clear()
    data.add(_box("added", 1))
    assert _compare(scene, q, False)[0][0, 0, 0] > 0
    data.load_batch([_box("reloaded", -1)], 0)
    assert _compare(scene, q, False)[0][0, 0, 1] > 0
    assert data.aabb.data_ptr() == address


@pytest.mark.parametrize("swept", [False, True])
def test_captured_graph_reads_replaced_mesh_bounds(cuda_device_cfg, swept):
    scene = SceneData.create_cache(1, cuda_device_cfg, mesh_cache=2)
    scene.meshes.load_batch([_box("changing", 1)], 0)
    q = cuda_device_cfg.to_device([[[[1, 0, 0, .02], [-1, 0, 0, .02]]]])
    buffer = CollisionBuffer.from_shape(q.shape, cuda_device_cfg)
    checker = CollisionChecker(cuda_device_cfg)
    weight, eta, dt = [cuda_device_cfg.to_device([v]) for v in (1.0, .01, .05)]

    def query():
        args = (scene, q, buffer, weight, eta)
        if swept:
            return checker.get_swept_sphere_distance(*args, dt)
        return checker.get_sphere_distance(*args)

    for _ in range(3):
        query()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = query()
    for offset in (-1, 1):
        scene.meshes.replace(_box("changing", offset))
        expected, gradient = _compare(scene, q, swept)
        graph.replay()
        torch.testing.assert_close(result, expected)
        torch.testing.assert_close(buffer.gradient, gradient)
        assert result[0, 0, 0 if offset == 1 else 1] > 0
