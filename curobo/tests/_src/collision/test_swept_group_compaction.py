"""Compacted sweep queries preserve ungrouped checks, including overflow."""

import pytest
import torch

from curobo._src.geom.collision import wp_autograd
from curobo._src.geom.collision.buffer_collision import CollisionBuffer
from curobo._src.geom.collision.wp_autograd import SweptSphereObstacleCollision as Original
from curobo._src.geom.data.data_scene import SceneData
from curobo._src.geom.types import Cuboid, Mesh
from curobo._src.types.device_cfg import DeviceCfg

Candidate = Original


@pytest.fixture
def cfg():
    from curobo._src.util.warp import init_warp

    init_warp()
    return DeviceCfg(device=torch.device("cuda:0"))


def query(cls, scene, spheres, speed=False, replacements=None, buffer=None, constants=None):
    cfg = scene.device_cfg
    buffer = buffer or CollisionBuffer.from_shape(spheres.shape, cfg)
    if constants is None:
        constants = [cfg.to_device([v]) for v in (1.0, 0.01, 0.1, 0.05)] + [
            torch.arange(spheres.shape[0] - 1, -1, -1, device=cfg.device, dtype=torch.int32)
        ]
    weight, eta, maximum, dt, env = constants
    cost = cls.apply(
        spheres,
        buffer,
        scene,
        weight,
        eta,
        maximum,
        dt,
        speed,
        env,
        scene.num_envs > 1,
        False,
        replacements,
        replacements,
    )
    return cost.clone(), buffer.gradient.clone()


def reference(scene, spheres, speed=False, replacements=None):
    with pytest.MonkeyPatch.context() as context:
        context.setattr(wp_autograd, "SWEEP_GROUP_SIZE", spheres.shape[2] + 1)
        return query(Original, scene, spheres, speed, replacements)


def compare(scene, spheres, speed=False, replacements=None):
    actual = query(Candidate, scene, spheres, speed, replacements)
    expected = reference(scene, spheres, speed, replacements)
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
    return actual


@pytest.mark.parametrize("kind", ["cuboids", "meshes"])
@pytest.mark.parametrize("speed", [False, True])
@pytest.mark.parametrize("count", [1, 20])
def test_multi_environment_and_overflow(cfg, kind, speed, count):
    scene = SceneData.create_cache(
        2, cfg, **{("cuboid_cache" if kind == "cuboids" else "mesh_cache"): 24}
    )
    data = getattr(scene, kind)
    for env in range(2):
        shapes = []
        for i in range(count):
            box = Cuboid(
                name=f"box-{env}-{i}",
                dims=[0.4, 0.4, 0.4],
                pose=[i * 0.001, env * 0.02, 0, 1, 0, 0, 0],
            )
            if kind == "meshes":
                mesh = box.get_trimesh_mesh()
                shapes.append(
                    Mesh(
                        name=box.name,
                        vertices=mesh.vertices.tolist(),
                        faces=mesh.faces.tolist(),
                        pose=[0, 0, 0, 1, 0, 0, 0],
                    )
                )
            else:
                shapes.append(box)
        data.load_batch(shapes, env)
    gen = torch.Generator(device=cfg.device).manual_seed(873)
    spheres = torch.rand((2, 4, 65, 4), generator=gen, device=cfg.device) * 0.3 - 0.15
    spheres[..., 3] = 0.04
    spheres[:, :, -1, 3] = -1
    cost, _ = compare(scene, spheres, speed)
    assert torch.count_nonzero(cost) > 0
    # Contact replacement must still apply to both stored and overflow paths.
    replacements = torch.full((2, 65, 2), -1, device=cfg.device, dtype=torch.int32)
    replacements[:, :, 0] = 0
    replaced, _ = compare(scene, spheres, speed, replacements)
    if count == 1:
        assert torch.count_nonzero(replaced) == 0


@pytest.mark.parametrize("n", [1, 31, 32, 33, 67])
def test_partial_disabled_groups_and_surface_crossing(cfg, n):
    scene = SceneData.create_cache(1, cfg, mesh_cache=1)
    scene.meshes.load_batch(
        [
            Mesh(
                name="wall",
                vertices=[[0, -1, -1], [0, 1, -1], [0, 1, 1], [0, -1, 1]],
                faces=[[0, 1, 2], [0, 2, 3]],
                pose=[0, 0, 0, 1, 0, 0, 0],
            )
        ],
        0,
    )
    spheres = torch.zeros((1, 2, n, 4), device=cfg.device)
    spheres[:, 0, :, 0] = -0.15
    spheres[:, 1, :, 0] = 0.15
    spheres[..., 3] = 0.02
    if n > 32:
        spheres[:, :, 32:, 3] = -1
    cost, _ = compare(scene, spheres)
    assert torch.count_nonzero(cost) > 0
    scene.meshes.enable.zero_()
    cost, _ = compare(scene, spheres)
    assert torch.count_nonzero(cost) == 0


def test_graph_replay_reads_pose_enable_and_count(cfg):
    scene = SceneData.create_cache(1, cfg, cuboid_cache=24)
    scene.cuboids.load_batch(
        [
            Cuboid(name=f"box-{i}", dims=[0.2, 0.2, 0.2], pose=[0, 0, 0, 1, 0, 0, 0])
            for i in range(20)
        ],
        0,
    )
    spheres = torch.zeros((1, 3, 65, 4), device=cfg.device)
    spheres[..., 0] = 0.08
    spheres[..., 3] = 0.03
    buffer = CollisionBuffer.from_shape(spheres.shape, cfg)
    constants = [cfg.to_device([v]) for v in (1.0, 0.01, 0.1, 0.05)] + [
        torch.zeros(1, device=cfg.device, dtype=torch.int32)
    ]
    for _ in range(3):
        query(Candidate, scene, spheres, buffer=buffer, constants=constants)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = query(Candidate, scene, spheres, buffer=buffer, constants=constants)
    for count, displacement, enabled in [(20, 0.0, 1), (1, 2.0, 1), (1, 0.0, 1), (20, 0.0, 0)]:
        scene.cuboids.count.fill_(count)
        scene.cuboids.inv_pose[..., 0] = displacement
        scene.cuboids.enable.fill_(enabled)
        expected = reference(scene, spheres)
        graph.replay()
        for a, b in zip(actual, expected, strict=True):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
