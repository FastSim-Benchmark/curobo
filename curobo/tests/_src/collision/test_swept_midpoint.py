"""Adjacent half sweeps must not leave their shared midpoint unchecked."""

import pytest
import torch

from curobo._src.geom.collision.buffer_collision import CollisionBuffer
from curobo._src.geom.collision.checker_collision import CollisionChecker
from curobo._src.geom.data.data_scene import SceneData
from curobo._src.geom.types import Mesh


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("scale", [0.5, 1.0, 10.0])
@pytest.mark.parametrize("reverse", [False, True])
def test_crossing_midpoint_of_two_sided_wall(cuda_device_cfg, axis, scale, reverse):
    vertices = torch.tensor([[0, -1, -1], [0, 1, -1], [0, 1, 1], [0, -1, 1]])
    vertices = vertices.roll(axis, dims=1).tolist()
    scene = SceneData.create_cache(1, cuda_device_cfg, mesh_cache=1)
    scene.meshes.load_batch([Mesh(
        name="wall", vertices=vertices, faces=[[0, 1, 2], [0, 2, 3]],
        pose=[0, 0, 0, 1, 0, 0, 0])], 0)
    spheres = torch.zeros((1, 2, 1, 4), device="cuda:0")
    spheres[0, :, 0, axis] = cuda_device_cfg.to_device([-.2, .2]) * scale
    spheres[..., 3] = .02 * scale
    if reverse:
        spheres = spheres.flip(1)
    checker = CollisionChecker(cuda_device_cfg)
    buffer = CollisionBuffer.from_shape(spheres.shape, cuda_device_cfg)
    weight = cuda_device_cfg.to_device([1.0])
    eta = cuda_device_cfg.to_device([.01])
    args = (scene, spheres, buffer, weight, eta)
    endpoints = checker.get_sphere_distance(*args).clone()
    swept = checker.get_swept_sphere_distance(*args, cuda_device_cfg.to_device([.05]))
    assert torch.count_nonzero(endpoints) == 0
    assert (swept > 0).all()
    assert torch.isfinite(buffer.gradient).all()

    # The same segment translated beyond the finite wall must remain clear.
    spheres[..., (axis + 1) % 3] = 3
    clear = checker.get_swept_sphere_distance(*args, cuda_device_cfg.to_device([.05]))
    assert torch.count_nonzero(clear) == 0
