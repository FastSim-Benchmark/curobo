"""Per-query world bounds remain conservative and retain uncertain geometry."""
import math

import torch
import warp as wp

from curobo._src.geom.collision.wp_sweep_collision_kernel import obstacle_world_bounds
from curobo._src.geom.data.data_scene import SceneData
from curobo._src.geom.types import Cuboid


def evaluate(data, local):
    bounds = torch.empty((2, data.num_envs * data.max_n, 4), device='cuda:0')
    wp.launch(obstacle_world_bounds, dim=data.num_envs * data.max_n,
              inputs=[data.to_warp(), wp.from_torch(local.reshape(-1, 6)),
                      wp.from_torch(bounds[0], dtype=wp.vec4),
                      wp.from_torch(bounds[1], dtype=wp.vec4)], device='cuda:0')
    return bounds.cpu()


def test_rotated_world_bounds_and_disabled_slots(cuda_device_cfg):
    scene = SceneData.create_cache(2, cuda_device_cfg, cuboid_cache=3)
    q = math.sqrt(.5)
    scene.cuboids.load_batch([Cuboid(name='rotated', dims=[2, 4, 6],
                                    pose=[1, 2, 3, q, 0, 0, q])], 0)
    scene.cuboids.load_batch([Cuboid(name='other', dims=[2, 2, 2],
                                    pose=[10, 0, 0, 1, 0, 0, 0])], 1)
    data = scene.cuboids
    half = data.dims[..., :3] * .5
    local = torch.cat((-half, half), -1)
    bounds = evaluate(data, local)
    assert bounds[0, :, 3].tolist() == [1, 0, 0, 1, 0, 0]
    expected_lower = torch.tensor([-1., 1., 0.])
    expected_upper = torch.tensor([3., 3., 6.])
    assert (bounds[0, 0, :3] <= expected_lower).all()
    assert (bounds[1, 0, :3] >= expected_upper).all()
    assert (expected_lower - bounds[0, 0, :3]).max() < .002
    assert (bounds[1, 0, :3] - expected_upper).max() < .002
    assert bounds[0, 3, 0] < 9 and bounds[1, 3, 0] > 11


def test_nonfinite_inverted_and_nonunit_bounds_cannot_exclude(cuda_device_cfg):
    scene = SceneData.create_cache(1, cuda_device_cfg, cuboid_cache=3)
    scene.cuboids.load_batch([Cuboid(name=str(i), dims=[1, 1, 1],
                                    pose=[0, 0, 0, 1, 0, 0, 0]) for i in range(3)], 0)
    data = scene.cuboids
    half = data.dims[..., :3] * .5
    local = torch.cat((-half, half), -1)
    local[0, 0, 0] = float('nan')
    local[0, 1, 0] = 2.  # Inverted interval.
    data.inv_pose[0, 2, 3] = 2.  # An inverse with a non-unit quaternion.
    assert evaluate(data, local)[0, :, 3].tolist() == [2, 2, 2]
