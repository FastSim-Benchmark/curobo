# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic coverage and index fidelity for grouped endpoint retries."""

from types import SimpleNamespace

import pytest
import torch

from curobo._src.motion.motion_goalset import plan_goalset_fallback
from curobo._src.motion.motion_planner import MotionPlanner
from curobo._src.types.tool_pose import GoalToolPose
from curobo.types import JointState


def fixture(*, goal_count=5, ik_success=None, trajectory_success=None, finetune=None):
    """Encode different tool targets so accidental cross-group pairing is visible."""
    position = torch.zeros((1, 1, 2, goal_count, 3))
    position[0, 0, 0, :, 0] = torch.arange(goal_count)
    position[0, 0, 1, :, 0] = 100 + torch.arange(goal_count)
    quaternion = torch.zeros((1, 1, 2, goal_count, 4))
    quaternion[..., 0] = 1
    goals = GoalToolPose(["left", "right"], position, quaternion)
    current = JointState.from_position(torch.zeros((1, 2)))
    calls = SimpleNamespace(ik=[], trajectory=[], results=[])

    def solve_ik(target, *, current_state, return_seeds):
        assert current_state is current
        assert return_seeds == 2
        assert target.tool_frames == goals.tool_frames
        assert target.position.shape == (1, 1, 2, 1, 3)
        index = int(target.position[0, 0, 0, 0, 0])
        torch.testing.assert_close(target.position, goals.position[..., index : index + 1, :])
        torch.testing.assert_close(target.quaternion, goals.quaternion[..., index : index + 1, :])
        calls.ik.append(index)
        success = ik_success[index] if ik_success is not None else True
        return SimpleNamespace(
            success=torch.tensor([[success, False]]),
            solution=torch.tensor([[[index, 1.0], [-99.0, -99.0]]]),
            total_time=0.1,
            solve_time=0.05,
        )

    def solve_trajectory(target, state, *, seed_config, use_implicit_goal, **kwargs):
        assert state is current
        assert use_implicit_goal is False
        assert kwargs == ({} if finetune is None else {"finetune_attempts": finetune})
        index = int(target.position[0, 0, 0, 0, 0])
        assert float(target.position[0, 0, 1, 0, 0]) == 100 + index
        torch.testing.assert_close(seed_config, torch.tensor([[[index, 1.0], [index, 1.0]]]))
        calls.trajectory.append(index)
        success = trajectory_success[index] if trajectory_success is not None else True
        result = SimpleNamespace(
            success=torch.tensor([success]),
            goalset_index=torch.tensor([[[0, 0, -1]]]),
            debug_info={"existing": "preserved"},
            total_time=0.2,
            solve_time=0.1,
        )
        calls.results.append(result)
        return result

    planner = SimpleNamespace(
        config=SimpleNamespace(trajopt_finetune_attempts=finetune),
        ik_solver=SimpleNamespace(config=SimpleNamespace(num_seeds=2), solve_pose=solve_ik),
        trajopt_solver=SimpleNamespace(
            config=SimpleNamespace(num_seeds=2), solve_pose=solve_trajectory
        ),
    )
    return planner, goals, current, calls


@pytest.mark.parametrize("finetune", [None, 0, 3])
def test_original_index_and_multi_tool_group_survive_fallback(finetune):
    planner, goals, current, calls = fixture(
        ik_success=[False, True, True, True, True],
        trajectory_success=[False, False, True, True, True],
        finetune=finetune,
    )
    before = goals.clone()
    result = plan_goalset_fallback(planner, goals, current, False, max_attempts=2)
    assert calls.ik == [0, 1, 2]
    assert calls.trajectory == [1, 2]
    assert result.success.all()
    assert result.goalset_index.tolist() == [[[2, 2, -1]]]
    assert result.debug_info["existing"] == "preserved"
    assert result.debug_info["goalset_fallback"] == {
        "goal_count": 5,
        "goal_budget": 4,
        "attempted_goal_indices": [0, 1, 2],
        "ik_feasible_goal_count": 2,
        "trajectory_attempt_count": 2,
        "selected_original_goal_index": 2,
    }
    assert result.total_time == pytest.approx(0.7)
    assert result.solve_time == pytest.approx(0.35)
    torch.testing.assert_close(goals.position, before.position)
    torch.testing.assert_close(goals.quaternion, before.quaternion)


@pytest.mark.parametrize("attempts,expected", [(0, []), (1, [0, 1]), (10, [0, 1, 2, 3, 4])])
def test_all_ik_rejected_and_budget_exhaustion(attempts, expected, caplog):
    planner, goals, current, calls = fixture(ik_success=[False] * 5)
    assert plan_goalset_fallback(planner, goals, current, False, attempts) is None
    assert calls.ik == expected
    assert calls.trajectory == []
    assert "exhausted its endpoint-search budget" in caplog.text
    assert "'trajectory_attempt_count': 0" in caplog.text


def test_failed_trajectories_remain_failed(caplog):
    planner, goals, current, calls = fixture(trajectory_success=[False] * 5)
    result = plan_goalset_fallback(planner, goals, current, False, 1)
    assert calls.trajectory == [0, 1]
    assert not result.success.any()
    assert result.goalset_index.tolist() == [[[1, 1, -1]]]
    assert "selected_original_goal_index" not in result.debug_info["goalset_fallback"]
    assert "exhausted its endpoint-search budget" in caplog.text


def test_missing_redundant_single_goal_index_is_restored():
    planner, goals, current, _ = fixture(ik_success=[False, True, True, True, True])
    solve = planner.trajopt_solver.solve_pose

    def omit_local_index(*args, **kwargs):
        result = solve(*args, **kwargs)
        result.goalset_index = None
        return result

    planner.trajopt_solver.solve_pose = omit_local_index
    result = plan_goalset_fallback(planner, goals, current, False, 1)
    assert result.goalset_index.tolist() == [[1, 1]]


@pytest.mark.parametrize("solver", ["ik_solver", "trajopt_solver"])
def test_solver_errors_are_not_hidden(solver):
    planner, goals, current, _ = fixture()

    def broken(*args, **kwargs):
        raise ValueError("unexpected native solver failure")

    getattr(planner, solver).solve_pose = broken
    with pytest.raises(ValueError, match="unexpected native solver failure"):
        plan_goalset_fallback(planner, goals, current, False, 1)


def test_normal_goalset_failure_invokes_grouped_fallback():
    planner, goals, current, calls = fixture(ik_success=[False, True, True, True, True])
    single_solve = planner.ik_solver.solve_pose
    broad_calls = []

    def solve(target, **kwargs):
        if target.num_goalset > 1:
            broad_calls.append(target.num_goalset)
            return SimpleNamespace(success=torch.tensor([False]), debug_info={"failed": True})
        return single_solve(target, **kwargs)

    planner.ik_solver.solve_pose = solve
    result = MotionPlanner._plan_pose_goalset(planner, goals, current, False, max_attempts=1)
    assert broad_calls == [5]
    assert calls.ik == [0, 1]
    assert result.goalset_index.tolist() == [[[1, 1, -1]]]
    assert result.success.all()


def test_normal_goalset_success_does_not_repeat_candidates():
    planner, goals, current, calls = fixture()
    result = SimpleNamespace(success=torch.tensor([True]))
    planner.ik_solver.solve_pose = lambda *args, **kwargs: SimpleNamespace(
        success=torch.tensor([True]), solution=torch.zeros((1, 2))
    )
    planner.trajopt_solver.solve_pose = lambda *args, **kwargs: result
    assert MotionPlanner._plan_pose_goalset(planner, goals, current, False, 1) is result
    assert calls.ik == []


@pytest.mark.parametrize("obstacle_kind", ["mesh", "cuboid"])
@pytest.mark.parametrize("held", [False, True])
def test_native_fallback_rejects_blocked_goal_and_preserves_later_index(
    tmp_path, obstacle_kind, held
):
    """Use real IK, collision queries and TrajOpt for the fallback's selected group."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from curobo.examples.reference.contact_separation import box_clearance, robot_config
    from curobo.motion_planner import MotionPlannerCfg
    from curobo.scene import Cuboid, Scene
    from curobo.types import AxisHold, DeviceCfg, Pose

    obstacle = Cuboid(name="blocked_target", dims=[0.1, 0.3, 0.3], pose=[0, 0, 0.2, 1, 0, 0, 0])
    scene = Scene(
        cuboid=[obstacle] if obstacle_kind == "cuboid" else [],
        mesh=[obstacle.get_mesh()] if obstacle_kind == "mesh" else [],
    )
    device = DeviceCfg()
    planner = MotionPlanner(
        MotionPlannerCfg.create(
            robot_config(tmp_path),
            scene_model=scene,
            device_cfg=device,
            num_ik_seeds=8,
            num_trajopt_seeds=4,
            use_cuda_graph=True,
            random_seed=123,
            interpolation_dt=0.01,
            interpolation_buffer_size=1200,
        )
    )
    current = JointState.from_position(device.to_device([[-0.35, 0, 0.2]]), planner.joint_names)
    positions = [[0, 0, 0.2], [-0.2, 0, 0.2]]
    if held:
        limits = planner.kinematics.get_joint_limits().position.clone()
        limits[:, planner.joint_names.index("y")] = 0.0
        planner.update_joint_limits(position=limits)
        positions.insert(1, [-0.2, 0.1, 0.2])
    selected = len(positions) - 1
    goals = GoalToolPose.from_poses(
        {
            "gripper": Pose(
                position=device.to_device(positions),
                quaternion=device.to_device([[1, 0, 0, 0]] * len(positions)),
            )
        },
        ordered_tool_frames=["gripper"],
        num_goalset=len(positions),
    )
    try:
        with planner._hold_axis_scope({"gripper": AxisHold()}, current):
            result = plan_goalset_fallback(planner, goals, current, True, max_attempts=1)
        assert result is not None and result.success.all()
        assert result.goalset_index.reshape(-1).tolist() == [selected]
        summary = result.debug_info["goalset_fallback"]
        assert summary["attempted_goal_indices"] == list(range(len(positions)))
        assert summary["ik_feasible_goal_count"] == 1
        assert summary["trajectory_attempt_count"] == 1
        trajectory = result.get_interpolated_plan()
        if held:
            assert torch.count_nonzero(trajectory.position[..., 1]) == 0
        states = JointState.from_position(trajectory.position.reshape(-1, 3), planner.joint_names)
        spheres = planner.compute_kinematics(states).robot_spheres
        spheres = spheres.reshape(-1, spheres.shape[-2], 4)
        enabled = spheres[0, :, 3] > 0
        assert box_clearance(spheres.cpu(), obstacle)[:, enabled.cpu()].min() >= 0
        for data in (
            planner.scene_collision_checker.data.cuboids,
            planner.scene_collision_checker.data.meshes,
        ):
            if data is not None:
                assert all(data.enable[0, i] == 1 for i, name in enumerate(data.names[0]) if name)
    finally:
        planner.destroy()
