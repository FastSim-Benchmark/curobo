# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Regression tests for mesh obstacle SDF collision queries."""

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.geom.collision.buffer_collision import CollisionBuffer
from curobo._src.geom.collision.collision_scene import SceneCollision, SceneCollisionCfg
from curobo._src.geom.types import Cuboid, Mesh, MeshDistanceMode, SceneCfg


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_small_mesh_collision_cost_matches_cuboid(cuda_device_cfg):
    """Small mesh collision costs must match the same geometry as an analytic cuboid."""
    cube_edge = 0.05
    pose = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    sphere_radius = 0.05
    query_distances_m = [0.08, 0.10, 0.50, 1.00]

    def make_checker(scene_cfg: SceneCfg) -> SceneCollision:
        cfg = SceneCollisionCfg(
            device_cfg=cuda_device_cfg,
            scene_model=scene_cfg,
            cache={"cuboid": 4, "mesh": 4},
        )
        return SceneCollision.from_config(cfg)

    def collision_cost(checker: SceneCollision) -> torch.Tensor:
        spheres = torch.tensor(
            [[[[d, 0.0, 0.0, sphere_radius] for d in query_distances_m]]],
            device=cuda_device_cfg.device,
            dtype=torch.float32,
        )
        buf = CollisionBuffer.from_shape(spheres.shape, cuda_device_cfg)
        return checker.get_sphere_distance_raw(
            query_spheres=spheres,
            collision_buffer=buf,
            weight=torch.tensor([1.0], device=cuda_device_cfg.device),
            activation_distance=torch.tensor([0.01], device=cuda_device_cfg.device),
        )

    cuboid = Cuboid(name="box", dims=[cube_edge, cube_edge, cube_edge], pose=pose)
    trimesh_box = cuboid.get_trimesh_mesh()
    mesh = Mesh(
        name="box",
        vertices=trimesh_box.vertices.tolist(),
        faces=trimesh_box.faces.reshape(-1).tolist(),
        pose=pose,
    )

    cuboid_cost = collision_cost(make_checker(SceneCfg(cuboid=[cuboid])))
    mesh_checker = make_checker(SceneCfg(mesh=[mesh]))
    assert bool(mesh_checker.data.meshes.use_signed_distance[0, 0].item())
    mesh_cost = collision_cost(mesh_checker)

    assert torch.allclose(mesh_cost, cuboid_cost)
    assert mesh_cost.flatten()[0] > 0.0
    assert torch.all(mesh_cost.flatten()[1:] == 0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_open_mesh_uses_two_sided_unsigned_distance(cuda_device_cfg):
    """Open surfaces must collide identically from either winding side."""
    vertices = [
        [-0.5, -0.5, 0.0],
        [0.5, -0.5, 0.0],
        [0.5, 0.5, 0.0],
        [-0.5, 0.5, 0.0],
    ]
    faces = [[0, 1, 2], [0, 2, 3]]
    spheres = torch.tensor(
        [
            [
                [
                    [0.0, 0.0, 0.20, 0.05],
                    [0.0, 0.0, -0.20, 0.05],
                    [0.0, 0.0, 0.04, 0.05],
                    [0.0, 0.0, -0.04, 0.05],
                ]
            ]
        ],
        device=cuda_device_cfg.device,
        dtype=torch.float32,
    )

    def collision_cost(face_data):
        mesh = Mesh(
            name="open_wall",
            vertices=vertices,
            faces=face_data,
            pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        )
        assert mesh.distance_mode == MeshDistanceMode.AUTO
        checker = SceneCollision.from_config(
            SceneCollisionCfg(
                device_cfg=cuda_device_cfg,
                scene_model=SceneCfg(mesh=[mesh]),
                cache={"mesh": 1},
            )
        )
        assert not bool(checker.data.meshes.use_signed_distance[0, 0].item())
        buf = CollisionBuffer.from_shape(spheres.shape, cuda_device_cfg)
        return checker.get_sphere_distance_raw(
            query_spheres=spheres,
            collision_buffer=buf,
            weight=torch.tensor([1.0], device=cuda_device_cfg.device),
            activation_distance=torch.tensor([0.01], device=cuda_device_cfg.device),
        )

    forward_cost = collision_cost(faces)
    reverse_cost = collision_cost([face[::-1] for face in faces])

    assert torch.all(forward_cost.flatten()[:2] == 0.0)
    assert torch.all(forward_cost.flatten()[2:] > 0.0)
    assert torch.allclose(forward_cost, reverse_cost)


def test_mesh_distance_mode_accepts_config_strings():
    """Scene dictionaries expose stable string values for mesh semantics."""
    scene = SceneCfg.create(
        {
            "mesh": {
                "wall": {
                    "vertices": [[0.0, 0.0, 0.0]],
                    "faces": [0, 0, 0],
                    "distance_mode": "surface",
                }
            }
        }
    )

    assert scene.mesh[0].distance_mode == MeshDistanceMode.SURFACE


def test_mesh_distance_mode_rejects_unknown_values():
    """Invalid distance semantics must fail at configuration admission."""
    with pytest.raises(ValueError, match="distance_mode must be one of"):
        Mesh(
            name="bad",
            vertices=[[0.0, 0.0, 0.0]],
            faces=[0, 0, 0],
            distance_mode="inside-ish",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_auto_mode_does_not_trust_inward_closed_mesh(cuda_device_cfg):
    """Watertight topology alone is insufficient for signed-distance queries."""
    cuboid = Cuboid(
        name="inside_out_box",
        dims=[0.1, 0.1, 0.1],
        pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    )
    trimesh_box = cuboid.get_trimesh_mesh()
    mesh = Mesh(
        name="inside_out_box",
        vertices=trimesh_box.vertices.tolist(),
        faces=trimesh_box.faces[:, ::-1].tolist(),
        pose=cuboid.pose,
    )

    checker = SceneCollision.from_config(
        SceneCollisionCfg(
            device_cfg=cuda_device_cfg,
            scene_model=SceneCfg(mesh=[mesh]),
            cache={"mesh": 1},
        )
    )

    assert not bool(checker.data.meshes.use_signed_distance[0, 0].item())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_open_mesh_declared_solid_fails_scene_admission(cuda_device_cfg):
    """Explicit solid semantics invoke strict topology validation before upload."""
    mesh = Mesh(
        name="invalid_wall_solid",
        vertices=[
            [-0.5, -0.5, 0.0],
            [0.5, -0.5, 0.0],
            [0.5, 0.5, 0.0],
            [-0.5, 0.5, 0.0],
        ],
        faces=[[0, 1, 2], [0, 2, 3]],
        pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        distance_mode="solid",
    )

    with pytest.raises(ValueError, match="invalid_wall_solid.*watertight"):
        SceneCollision.from_config(
            SceneCollisionCfg(
                device_cfg=cuda_device_cfg,
                scene_model=SceneCfg(mesh=[mesh]),
                cache={"mesh": 1},
            )
        )
