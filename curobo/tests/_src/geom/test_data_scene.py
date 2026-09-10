# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for aggregate scene obstacle data."""

from unittest.mock import Mock

from curobo._src.geom.data.data_scene import SceneData


def test_enable_obstacle_routes_to_storage_containing_name() -> None:
    """Test that obstacle enablement uses the storage containing its name."""
    cuboids = Mock()
    cuboids.has_name.return_value = False
    meshes = Mock()
    meshes.has_name.return_value = True
    scene_data = SceneData(cuboids=cuboids, meshes=meshes)

    scene_data.enable_obstacle("mesh_obstacle", enabled=False, env_idx=0)

    cuboids.has_name.assert_called_once_with("mesh_obstacle", 0)
    meshes.has_name.assert_called_once_with("mesh_obstacle", 0)
    cuboids.get_names.assert_not_called()
    meshes.get_names.assert_not_called()
    cuboids.set_enabled.assert_not_called()
    meshes.set_enabled.assert_called_once_with("mesh_obstacle", False, 0)


def test_update_obstacle_pose_routes_to_storage_containing_name() -> None:
    """Test that pose updates use the storage containing the obstacle name."""
    cuboids = Mock()
    cuboids.has_name.return_value = False
    meshes = Mock()
    meshes.has_name.return_value = True
    pose = Mock()
    scene_data = SceneData(cuboids=cuboids, meshes=meshes)

    scene_data.update_obstacle_pose("mesh_obstacle", pose=pose, env_idx=1)

    cuboids.has_name.assert_called_once_with("mesh_obstacle", 1)
    meshes.has_name.assert_called_once_with("mesh_obstacle", 1)
    cuboids.get_names.assert_not_called()
    meshes.get_names.assert_not_called()
    cuboids.update_pose.assert_not_called()
    meshes.update_pose.assert_called_once_with("mesh_obstacle", w_obj_pose=pose, env_idx=1)


def test_scene_delta_preserves_buffers_and_unmentioned_meshes() -> None:
    import pytest
    import torch

    from curobo.scene import Cuboid, Mesh
    from curobo.scene import Scene as SceneCfg
    from curobo.types import DeviceCfg

    device = DeviceCfg(device=torch.device("cuda:0"))
    a = Cuboid(name="a", pose=[0, 0, 0, 1, 0, 0, 0], dims=[1, 1, 1])
    b = Cuboid(name="b", pose=[2, 0, 0, 1, 0, 0, 0], dims=[1, 1, 1])
    mesh = Mesh(name="mesh", pose=[4, 0, 0, 1, 0, 0, 0],
                vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces=[[0, 1, 2]])
    data = SceneData.from_scene_cfg(SceneCfg(cuboid=[a, b], mesh=[mesh]), device,
                                    cuboid_cache=2, mesh_cache=2)
    pointer = data.cuboids.inv_pose.data_ptr()
    mesh_pointer = data.meshes.mesh_ids.data_ptr()
    cached_mesh = data.meshes.wp_cache["mesh"]
    c = Cuboid(name="c", pose=[3, 0, 0, 1, 0, 0, 0], dims=[2, 1, 1])
    data.apply_obstacle_updates([c], ["a"])
    assert set(data.get_obstacle_names()) == {"b", "c", "mesh"}
    assert data.cuboids.inv_pose.data_ptr() == pointer
    assert data.meshes.mesh_ids.data_ptr() == mesh_pointer
    assert data.meshes.wp_cache["mesh"] is cached_mesh
    with pytest.raises(ValueError, match="capacity"):
        data.apply_obstacle_updates([a], [])
    assert set(data.get_obstacle_names()) == {"b", "c", "mesh"}
    with pytest.raises(ValueError, match="existing"):
        data.apply_obstacle_updates([], ["missing"])
    changed = Mesh(
        name="mesh", pose=mesh.pose,
        vertices=[[0, 0, 0], [2, 0, 0], [0, 1, 0]], faces=mesh.faces,
    )
    data.apply_obstacle_updates([changed], [])
    assert data.meshes.mesh_ids.data_ptr() == mesh_pointer
    assert data.meshes.wp_cache["mesh"] is not cached_mesh
    data.apply_obstacle_updates([], ["mesh"])
    assert not data.meshes.get_names()
    data.apply_obstacle_updates([mesh], [])
    assert data.meshes.get_names() == ["mesh"]


def test_scene_delta_rejects_voxel_replacement_before_any_write() -> None:
    import pytest

    from curobo.scene import Cuboid

    cuboids, voxels = Mock(), Mock()
    cuboids.has_name.return_value = False
    voxels.has_name.side_effect = lambda name, env_idx: name == "voxel"
    scene = SceneData(cuboids=cuboids, voxels=voxels)
    replacement = Cuboid(name="voxel", pose=[0, 0, 0, 1, 0, 0, 0], dims=[1, 1, 1])
    with pytest.raises(ValueError, match="voxel"):
        scene.apply_obstacle_updates([replacement], [])
    cuboids.add.assert_not_called()
    voxels.remove.assert_not_called()
    for invalid in (-1, 1, True, 0.5):
        with pytest.raises(ValueError, match="environment"):
            scene.apply_obstacle_updates([], [], env_idx=invalid)


def test_scene_delta_cpu_reference_keeps_other_environments() -> None:
    from curobo._src.geom.collision.collision_scene import SceneCollision
    from curobo.scene import Cuboid

    storage = Mock(num_envs=2)
    checker = SceneCollision(data=storage, checker=Mock(), device_cfg=Mock())
    box = Cuboid(name="box", pose=[0, 0, 0, 1, 0, 0, 0], dims=[1, 1, 1])
    checker.apply_obstacle_updates([box], [], env_idx=1)
    assert not checker.scene_model[0].objects
    other = checker.scene_model[1]
    assert other.get_obstacle("box") is box
    checker.apply_obstacle_updates([box], [], env_idx=0)
    assert checker.scene_model[1] is other
    checker.apply_obstacle_updates([], ["box"], env_idx=0)
    assert checker.scene_model[0].get_obstacle("box") is None
    assert checker.scene_model[1].get_obstacle("box") is box
