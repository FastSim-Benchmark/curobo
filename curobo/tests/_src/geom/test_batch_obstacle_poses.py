# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contract and device equivalence tests for public batched scene pose updates."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from curobo._src.geom.collision.buffer_collision import CollisionBuffer
from curobo._src.geom.collision.collision_scene import SceneCollision, SceneCollisionCfg
from curobo._src.geom.data.data_scene import SceneData
from curobo._src.geom.types import Cuboid, SceneCfg
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import Pose


def _cpu_data() -> SceneData:
    """Provide CPU storage for validation without constructing a CUDA context."""
    def storage(names: list[str]) -> SimpleNamespace:
        inv_pose = torch.zeros((2, 4, 8), dtype=torch.float32)
        inv_pose[..., 3] = 1
        return SimpleNamespace(names=[names + [None] * (4 - len(names))] * 2,
                               inv_pose=inv_pose)
    return SceneData(cuboids=storage(["box", "untouched"]), meshes=storage(["mesh"]),
                     voxels=storage(["voxel"]), num_envs=2,
                     device_cfg=DeviceCfg(device=torch.device("cpu")))


def _poses(n: int, device: str = "cpu") -> Pose:
    position = torch.arange(n * 3, dtype=torch.float32, device=device).reshape(n, 3) / 10
    quaternion = torch.zeros((n, 4), dtype=torch.float32, device=device)
    quaternion[:, 0] = 1
    return Pose(position, quaternion)


@pytest.mark.parametrize("invalid", [-1, 2, True, False, 0.5, "0", None])
def test_batch_rejects_invalid_environment_before_writes(invalid: object) -> None:
    data = _cpu_data()
    with patch.object(Pose, "inverse") as inverse, pytest.raises(ValueError, match="environment"):
        data.update_obstacle_poses(["box"], _poses(1), env_idx=invalid)
    inverse.assert_not_called()


@pytest.mark.parametrize("names", ["box", b"box", iter(["box"]), [""], [None], [4],
                                   ["box", "box"], ["box", "missing"]])
def test_batch_rejects_invalid_names_without_partial_write(names: object) -> None:
    data = _cpu_data()
    before = data.cuboids.inv_pose.clone()
    n = len(names) if isinstance(names, list) else 1
    with patch.object(Pose, "inverse") as inverse, pytest.raises(ValueError):
        data.update_obstacle_poses(names, _poses(n))
    inverse.assert_not_called()
    assert torch.equal(before, data.cuboids.inv_pose)


@pytest.mark.parametrize("failure", ["shape_position", "shape_quaternion", "rank", "dtype",
                                     "device", "nan_position", "inf_quaternion", "zero_quaternion",
                                     "nonunit", "storage_dtype", "ambiguous", "not_pose"])
def test_batch_preflights_every_pose_before_inverse(failure: str) -> None:
    data, poses = _cpu_data(), _poses(2)
    names = ["box", "mesh"]
    if failure == "shape_position":
        poses.position = poses.position[:1]
    elif failure == "shape_quaternion":
        poses.quaternion = poses.quaternion[:, :3]
    elif failure == "rank":
        poses.position = poses.position.unsqueeze(0)
    elif failure == "dtype":
        poses.quaternion = poses.quaternion.to(torch.float16)
    elif failure == "device":
        poses.quaternion = poses.quaternion.to("meta")
    elif failure == "nan_position":
        poses.position[-1, 0] = float("nan")
    elif failure == "inf_quaternion":
        poses.quaternion[-1, 2] = float("inf")
    elif failure == "zero_quaternion":
        poses.quaternion[-1] = 0
    elif failure == "nonunit":
        poses.quaternion[-1, 0] = 1.01
    elif failure == "storage_dtype":
        data.meshes.inv_pose = data.meshes.inv_pose.to(torch.float16)
    elif failure == "ambiguous":
        data.voxels.names[0][1] = "mesh"
    elif failure == "not_pose":
        poses = None
    before = [storage.inv_pose.clone() for storage in data.get_valid_data()]
    with patch.object(Pose, "inverse") as inverse, pytest.raises(ValueError):
        data.update_obstacle_poses(names, poses)
    inverse.assert_not_called()
    assert all(torch.equal(old, storage.inv_pose)
               for old, storage in zip(before, data.get_valid_data()))


def test_batch_empty_validates_but_does_not_invert_or_write() -> None:
    data = _cpu_data()
    with patch.object(Pose, "inverse") as inverse:
        data.update_obstacle_poses([], _poses(0), env_idx=1)
    inverse.assert_not_called()
    with pytest.raises(ValueError, match="shapes"):
        data.update_obstacle_poses([], _poses(1))


def test_public_collision_routes_batch_without_changing_cpu_model() -> None:
    data, model, poses = Mock(), SceneCfg(), _poses(1)
    checker = SceneCollision(data=data, checker=Mock(), device_cfg=Mock(), scene_model=model)
    checker.update_obstacle_poses(["box"], poses, env_idx=1)
    data.update_obstacle_poses.assert_called_once_with(["box"], poses, 1)
    assert checker.scene_model is model


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_batch_matches_scalar_all_storage_types_and_preserves_buffers(device: str) -> None:
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    cfg = DeviceCfg(device=torch.device(device))
    data = SceneData.create_cache(2, cfg, cuboid_cache=4, mesh_cache=4,
                                 voxel_cache={"layers": 2, "dims": [0.1] * 3, "voxel_size": 0.05})
    for storage, name in zip(data.get_valid_data(), ("box", "mesh", "voxel")):
        storage.names[0][0] = storage.names[1][0] = name
        storage.names[0][1] = storage.names[1][1] = name + "_untouched"
        storage.count[:] = 2
        storage.enable[:, 1] = 1  # The named slot stays disabled after moving.
        storage.inv_pose[..., 7] = 42
    initial = {id(storage): {key: (value.data_ptr(), value.clone())
                            for key, value in vars(storage).items()
                            if isinstance(value, torch.Tensor)}
               for storage in data.get_valid_data()}
    original_names = [deepcopy(storage.names) for storage in data.get_valid_data()]
    names = ["voxel", "box", "mesh"]
    poses = _poses(3, device)
    poses.quaternion[:] = torch.tensor([0.5, 0.5, 0.5, 0.5], device=device)
    for index, name in enumerate(names):
        data.update_obstacle_pose(name, Pose(poses.position[index], poses.quaternion[index]), 0)
    with patch.object(Pose, "inverse", autospec=True, side_effect=Pose.inverse) as inverse:
        data.update_obstacle_poses(names, poses, env_idx=1)
    assert inverse.call_count == 1
    for storage, old_names in zip(data.get_valid_data(), original_names):
        torch.testing.assert_close(storage.inv_pose[0], storage.inv_pose[1], rtol=0, atol=0)
        assert storage.names == old_names
        for key, value in vars(storage).items():
            if not isinstance(value, torch.Tensor):
                continue
            pointer, before = initial[id(storage)][key]
            assert value.data_ptr() == pointer
            if key == "inv_pose":
                assert torch.equal(value[..., 7], before[..., 7])
                assert torch.equal(value[:, 1:], before[:, 1:])
            else:
                assert torch.equal(value, before)


def test_batch_cpu_reindexes_compacted_slots_and_accepts_strided_inputs() -> None:
    cfg = DeviceCfg(device=torch.device("cpu"))
    data = SceneData.create_cache(1, cfg, cuboid_cache=3)
    boxes = [Cuboid(name=name, pose=[0, 0, 0, 1, 0, 0, 0], dims=[1, 1, 1])
             for name in ("a", "b", "c")]
    data.cuboids.load_batch(boxes, 0)
    data.cuboids.remove("a", 0)  # c moves from slot 2 to slot 0.
    poses = _poses(4)
    strided = Pose(poses.position[::2], poses.quaternion[::2])
    assert not strided.position.is_contiguous()
    data.update_obstacle_poses(["b", "c"], strided)
    expected = Pose(strided.position.contiguous(), strided.quaternion.contiguous()).inverse()
    torch.testing.assert_close(data.cuboids.inv_pose[0, 1, :7], expected.get_pose_vector()[0])
    torch.testing.assert_close(data.cuboids.inv_pose[0, 0, :7], expected.get_pose_vector()[1])


def test_batch_gpu_preserves_mesh_bvh_and_existing_query_graph_on_custom_stream() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    cfg = DeviceCfg(device=torch.device("cuda:0"))
    box = Cuboid(name="box", pose=[0, 0, 0, 1, 0, 0, 0], dims=[0.4] * 3)
    mesh = Cuboid(name="mesh", pose=[3, 0, 0, 1, 0, 0, 0], dims=[0.4] * 3).get_mesh()
    scene = SceneCollision.from_config(SceneCollisionCfg(
        device_cfg=cfg, scene_model=SceneCfg(cuboid=[box], mesh=[mesh]),
        cache={"cuboid": 4, "mesh": 4},
    ))
    cached = scene.data.meshes.wp_cache["mesh"]
    ids = scene.data.meshes.mesh_ids.clone()
    address = scene.data.meshes.inv_pose.data_ptr()
    query = torch.tensor([[[[0.15, 0, 0, 0.05], [3.15, 0, 0, 0.05]]]], device=cfg.device)
    buffer = CollisionBuffer.from_shape(query.shape, cfg)
    weight, eta = torch.tensor([1.0], device=cfg.device), torch.tensor([0.02], device=cfg.device)
    def distance() -> torch.Tensor:
        return scene.get_sphere_distance_raw(query, buffer, weight, eta)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            before = distance().clone()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        graph_result = distance().clone()
    poses = _poses(2, "cuda:0")
    poses.position[:] = torch.tensor([[10, 0, 0], [13, 0, 0]], device=cfg.device)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        scene.update_obstacle_poses(["box", "mesh"], poses)
        graph.replay()
        eager_result = distance().clone()
    stream.synchronize()
    assert torch.all(before > 0)
    assert torch.equal(graph_result, eager_result)
    assert torch.count_nonzero(graph_result) == 0
    assert scene.data.meshes.wp_cache["mesh"] is cached
    assert scene.data.meshes.inv_pose.data_ptr() == address
    assert torch.equal(ids, scene.data.meshes.mesh_ids)
