# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Bind target, captured endpoint, validation and trajectory seed through retries."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from curobo._src.motion import motion_contact_goalset as module
from curobo.tests._src.motion.test_goalset_fallback import fixture
from curobo.types import JointState


def terminal_fixture(
    monkeypatch,
    *,
    endpoint_valid=lambda i, c: True,
    trajectory_valid=lambda i, c: i == 1 and c == 1,
):
    planner, goals, current, _ = fixture(goal_count=3)
    calls = SimpleNamespace(active=None, captured=[], metrics=[], trajectory=[], restored=[])

    def candidates(owner, target, start):
        assert owner is planner and start is current and calls.active is None
        index = int(target.position[0, 0, 0, 0, 0])
        for candidate in range(3):
            assert calls.active is None
            calls.captured.append((index, candidate))
            yield JointState.from_position(torch.tensor([[float(index), float(candidate)]]))

    @contextmanager
    def scope(owner, start, endpoint, mode):
        assert calls.active is None
        assert owner is planner and start is current and mode == "end"
        calls.active = tuple(int(q) for q in endpoint.position.reshape(-1))
        try:
            yield
        finally:
            calls.restored.append(calls.active)
            calls.active = None

    def metrics(target, *, current_state, seed_config, return_seeds, run_optimizer):
        assert current_state is current and run_optimizer is False and return_seeds == 2
        assert calls.active is not None
        index, candidate = calls.active
        assert float(target.position[0, 0, 1, 0, 0]) == 100 + index
        torch.testing.assert_close(seed_config, torch.tensor([[list(calls.active)] * 2]).float())
        calls.metrics.append(calls.active)
        return SimpleNamespace(success=torch.tensor([endpoint_valid(index, candidate)]))

    def trajectory(
        target, start, *, seed_config, use_implicit_goal, finetune_attempts, finetune_dt_scale
    ):
        assert start is current and use_implicit_goal is False
        assert finetune_attempts == 1 and finetune_dt_scale == 0.55
        assert calls.active is not None
        index, candidate = calls.active
        assert float(target.position[0, 0, 1, 0, 0]) == 100 + index
        torch.testing.assert_close(seed_config, torch.tensor([[list(calls.active)] * 2]).float())
        calls.trajectory.append(calls.active)
        return SimpleNamespace(
            success=torch.tensor([trajectory_valid(index, candidate)]),
            goalset_index=torch.tensor([[0, 0]]),
            debug_info={},
        )

    monkeypatch.setattr(module, "contact_goal_candidates", candidates)
    monkeypatch.setattr(module, "boundary_contact_scope", scope)
    planner.ik_solver.solve_pose = metrics
    planner.trajopt_solver.solve_pose = trajectory
    return planner, goals, current, calls


def test_failed_endpoint_and_trajectory_continue_with_exact_target_seed_binding(monkeypatch):
    planner, goals, current, calls = terminal_fixture(
        monkeypatch, endpoint_valid=lambda i, c: (i, c) != (1, 0)
    )
    result = module.plan_terminal_pose(planner, goals, current, False, 2, "end")
    assert result.success.all() and result.goalset_index.tolist() == [[1, 1]]
    assert calls.captured == calls.metrics == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert calls.trajectory == [(0, 0), (0, 1), (1, 1)]
    assert calls.restored == calls.captured and calls.active is None
    summary = result.debug_info["terminal_contact_search"]
    assert summary["endpoint_metric_rejection_count"] == 1
    assert summary["trajectory_attempt_count"] == 3
    assert summary["selected_original_goal_index"] == 1


def test_all_candidate_trajectories_rejected_obeys_both_budgets(monkeypatch, caplog):
    planner, goals, current, calls = terminal_fixture(
        monkeypatch, trajectory_valid=lambda i, c: False
    )
    result = module.plan_terminal_pose(planner, goals, current, False, 1, "end")
    assert not result.success.any()
    assert calls.captured == [(0, 0), (1, 0)]
    assert calls.restored == calls.captured and calls.active is None
    assert "exhausted its bounded candidate search" in caplog.text
    assert "selected_original_goal_index" not in result.debug_info["terminal_contact_search"]


@pytest.mark.parametrize("solver", ["ik_solver", "trajopt_solver"])
def test_terminal_scope_restored_but_unknown_solver_error_propagates(monkeypatch, solver):
    planner, goals, current, calls = terminal_fixture(monkeypatch)

    def broken(*args, **kwargs):
        raise ValueError("native solver defect")

    getattr(planner, solver).solve_pose = broken
    with pytest.raises(ValueError, match="native solver defect"):
        module.plan_terminal_pose(planner, goals, current, False, 1, "end")
    assert calls.active is None and calls.restored == [(0, 0)]


@pytest.mark.parametrize("kind", ["mesh", "cuboid"])
def test_native_contact_goalset_rejects_blocked_path_then_uses_other_endpoint(tmp_path, kind):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from curobo.examples.reference.contact_separation import box_clearance, robot_config
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    from curobo.scene import Cuboid, Scene
    from curobo.types import AxisHold, DeviceCfg, GoalToolPose, Pose

    support = Cuboid(name="support", dims=[1.0, 1.0, 0.1], pose=[0, 0, -0.05, 1, 0, 0, 0])
    wall = Cuboid(name="barrier", dims=[0.05, 2.0, 2.0], pose=[-0.15, 0, 0.5, 1, 0, 0, 0])
    device = DeviceCfg()
    planner = MotionPlanner(
        MotionPlannerCfg.create(
            robot_config(tmp_path),
            scene_model=Scene(
                mesh=[support.get_mesh()] if kind == "mesh" else [],
                cuboid=[wall] + ([support] if kind == "cuboid" else []),
            ),
            device_cfg=device,
            num_ik_seeds=8,
            num_trajopt_seeds=4,
            use_cuda_graph=True,
            position_tolerance=1e-5,
            interpolation_dt=0.01,
            interpolation_buffer_size=1200,
            random_seed=123,
        )
    )
    current = JointState.from_position(device.to_device([[-0.35, 0, 0.2]]), planner.joint_names)
    planner.attachment_manager.update(device.to_device([[0, 0, -0.08, 0.041]]), current)
    goals = GoalToolPose.from_poses(
        {
            "gripper": Pose(
                position=device.to_device([[0.2, 0, 0.1205], [-0.4, 0, 0.1205]]),
                quaternion=device.to_device([[1, 0, 0, 0], [1, 0, 0, 0]]),
            ),
        },
        ordered_tool_frames=["gripper"],
        num_goalset=2,
    )
    try:
        result = planner.plan_pose(
            goals,
            current,
            max_attempts=1,
            hold_axis={"gripper": AxisHold()},
            allow_boundary_collision="end",
        )
        assert result is not None and result.success.all()
        assert result.goalset_index.reshape(-1).tolist() == [1]
        summary = result.debug_info["terminal_contact_search"]
        assert summary["trajectory_attempt_count"] == 2
        assert summary["attempts"][0]["endpoint_metrics_success"]
        assert not summary["attempts"][0]["trajectory_success"]
        positions = result.get_interpolated_plan().position.reshape(-1, 3)
        torch.testing.assert_close(
            positions[-1], device.to_device([-0.4, 0, 0.1205]), atol=1e-5, rtol=0
        )
        spheres = planner.compute_kinematics(
            JointState.from_position(positions, planner.joint_names)
        ).robot_spheres
        spheres = spheres.reshape(-1, spheres.shape[-2], 4).cpu()
        enabled = spheres[0, :, 3] > 0
        assert box_clearance(spheres, wall)[:, enabled].min() >= 0
        assert box_clearance(spheres, support)[:, enabled].min() >= -0.002
    finally:
        planner.destroy()
