"""Cuboid bounds and grouping preserve analytic collision and sweep behavior."""
import pytest
import torch
import warp as wp

from curobo._src.geom.collision import wp_autograd
from curobo._src.geom.collision.buffer_collision import CollisionBuffer
from curobo._src.geom.collision.checker_collision import CollisionChecker
from curobo._src.geom.collision.wp_collision_kernel import query_aabb_overlaps
from curobo._src.geom.data.data_cuboid import CuboidDataWarp
from curobo._src.geom.data.data_scene import SceneData
from curobo._src.geom.types import Cuboid
from curobo.types import Pose


def query(scene, spheres, speed=False, env=None, swept=True):
    cfg = scene.device_cfg
    buffer = CollisionBuffer.from_shape(spheres.shape, cfg)
    checker = CollisionChecker(cfg)
    args = (scene, spheres, buffer, cfg.to_device([1.0]), cfg.to_device([.01]))
    if swept:
        value = checker.get_swept_sphere_distance(
            *args, cfg.to_device([.05]), env_query_idx=env, enable_speed_metric=speed)
    else:
        value = checker.get_sphere_distance(*args, env_query_idx=env)
    return value.clone(), buffer.gradient.clone()


def compare(monkeypatch, scene, spheres, speed=False, env=None):
    actual = query(scene, spheres, speed, env)
    with monkeypatch.context() as context:
        context.setattr(wp_autograd, 'SWEEP_GROUP_SIZE', spheres.shape[2] + 1)
        reference = query(scene, spheres, speed, env)
    for a, b in zip(actual, reference, strict=True):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
    return actual


@wp.kernel
def edge_queries(data: CuboidDataWarp, output: wp.array(dtype=wp.int32)):
    outside = wp.vec3(.3, 0.0, 0.0)
    origin = wp.vec3(0.0)
    tangent = wp.vec3(.11, 0.0, 0.0)
    output[0] = wp.int32(query_aabb_overlaps(data, 0, 0, outside, outside, .01))
    output[1] = wp.int32(query_aabb_overlaps(data, 0, 0, -outside, outside, .01))
    output[2] = wp.int32(query_aabb_overlaps(data, 0, 0, origin, origin, .01))
    output[3] = wp.int32(query_aabb_overlaps(data, 0, 0, tangent, tangent, .01))
    nonfinite = wp.vec3(wp.inf, 0.0, 0.0)
    output[4] = wp.int32(query_aabb_overlaps(data, 0, 0, nonfinite, nonfinite, .01))
    output[5] = wp.int32(query_aabb_overlaps(data, 0, 0, outside, -outside, .01))


def test_rejection_keeps_crossings_interiors_tangencies_and_nonfinite(cuda_device_cfg):
    scene = SceneData.create_cache(1, cuda_device_cfg, cuboid_cache=1)
    scene.cuboids.load_batch([Cuboid(name='box', dims=[.2, .2, .2],
                                   pose=[0, 0, 0, 1, 0, 0, 0])], 0)
    output = torch.empty(6, device='cuda:0', dtype=torch.int32)
    wp.launch(edge_queries, dim=1, inputs=[scene.cuboids.to_warp(), wp.from_torch(output)],
              device='cuda:0')
    assert output.tolist() == [0, 1, 1, 1, 1, 1]


@pytest.mark.parametrize('speed', [False, True])
def test_grouped_cost_gradient_rotated_multi_environment(cuda_device_cfg, monkeypatch, speed):
    scene = SceneData.create_cache(2, cuda_device_cfg, cuboid_cache=4)
    scene.cuboids.load_batch([
        Cuboid(name='first', dims=[.2, .3, .4], pose=[0, 0, 0, 1, 0, 0, 0]),
        Cuboid(name='rotated', dims=[.5, .1, .3], pose=[.4, 0, 0, .8, 0, 0, .6])], 0)
    scene.cuboids.load_batch([Cuboid(name='other', dims=[.4, .3, .2],
                                   pose=[-.4, 0, 0, 1, 0, 0, 0])], 1)
    generator = torch.Generator(device='cuda:0').manual_seed(82)
    spheres = torch.rand((2, 5, 67, 4), device='cuda:0', generator=generator) * 2 - 1
    spheres[..., 3] = .06
    spheres[:, :, -1, 3] = -1
    env = torch.tensor([1, 0], device='cuda:0', dtype=torch.int32)
    value, _ = compare(monkeypatch, scene, spheres, speed, env)
    assert torch.count_nonzero(value) > 0


def test_thin_wall_crossing_and_empty_group(cuda_device_cfg, monkeypatch):
    scene = SceneData.create_cache(1, cuda_device_cfg, cuboid_cache=1)
    scene.cuboids.load_batch([Cuboid(name='wall', dims=[.0002, 1, 1],
                                   pose=[0, 0, 0, 1, 0, 0, 0])], 0)
    spheres = torch.zeros((1, 2, 65, 4), device='cuda:0')
    spheres[:, 0, :, 0] = -.15
    spheres[:, 1, :, 0] = .15
    spheres[..., 3] = .02
    spheres[..., 32:64, 3] = -1
    endpoints, _ = query(scene, spheres, swept=False)
    assert torch.count_nonzero(endpoints) == 0
    value, _ = compare(monkeypatch, scene, spheres)
    assert (value[..., :32] > 0).all() and (value[..., 64:] > 0).all()
    assert torch.count_nonzero(value[..., 32:64]) == 0


def test_graph_replay_reads_changed_dimensions_and_pose(cuda_device_cfg):
    cfg = cuda_device_cfg
    scene = SceneData.create_cache(1, cfg, cuboid_cache=1)
    scene.cuboids.load_batch([Cuboid(name='box', dims=[.2, .2, .2],
                                   pose=[0, 0, 0, 1, 0, 0, 0])], 0)
    spheres = torch.zeros((1, 2, 64, 4), device='cuda:0')
    spheres[..., 0] = .5
    spheres[..., 3] = .02
    checker = CollisionChecker(cfg)
    buffer = CollisionBuffer.from_shape(spheres.shape, cfg)
    weight, eta, dt = (cfg.to_device([value]) for value in (1.0, .01, .05))

    def launch():
        value = checker.get_swept_sphere_distance(
            scene, spheres, buffer, weight, eta, dt)
        return value.clone(), buffer.gradient.clone()

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            launch()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        value, gradient = launch()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.count_nonzero(value) == 0
    scene.cuboids.update_dims('box', cfg.to_device([2, 2, 2]))
    graph.replay()
    torch.cuda.synchronize()
    assert (value > 0).all()
    scene.cuboids.update_pose('box', w_obj_pose=Pose(
        position=cfg.to_device([[10, 0, 0]]), quaternion=cfg.to_device([[1, 0, 0, 0]])))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.count_nonzero(value) == torch.count_nonzero(gradient) == 0
