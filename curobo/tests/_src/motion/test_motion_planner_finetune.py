# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Optional refinement preserves initial trajectory optimization and validation."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from curobo._src.motion.motion_planner import MotionPlanner
from curobo._src.motion.motion_planner_cfg import MotionPlannerCfg
from curobo._src.state.state_joint import JointState
from curobo._src.types.tool_pose import GoalToolPose


@pytest.mark.parametrize("value", [True, False, -1, 0.0, 1.5, "0"])
def test_invalid_refinement_count_rejected_before_loading_robot(value: object) -> None:
    with pytest.raises(ValueError, match="trajopt_finetune_attempts"):
        MotionPlannerCfg(None, None, trajopt_finetune_attempts=value)
    with pytest.raises(ValueError, match="trajopt_finetune_attempts"):
        MotionPlannerCfg.create("must-not-load.yml", trajopt_finetune_attempts=value)


@pytest.mark.parametrize("value", [None, 0, 1, 3])
def test_refinement_count_factory_preserves_configuration(monkeypatch, value) -> None:
    import curobo._src.motion.motion_planner_cfg as module

    for cls in (module.RobotCfg, module.IKSolverCfg, module.TrajOptSolverCfg,
                module.PRMGraphPlannerCfg):
        monkeypatch.setattr(cls, "create", lambda *args, **kwargs: SimpleNamespace())
    config = MotionPlannerCfg.create({"robot_cfg": {}}, trajopt_finetune_attempts=value)
    assert config.trajopt_finetune_attempts == value


@pytest.mark.parametrize("value", [None, 0, 2])
@pytest.mark.parametrize("branch", ["single", "graph_pose", "goalset", "cspace", "graph_cspace"])
@pytest.mark.parametrize("success", [True, False])
def test_public_planning_preserves_branch_defaults_and_forwards_override(
    value, branch: str, success: bool,
) -> None:
    planner = MotionPlanner.__new__(MotionPlanner)
    planner._destroyed = True
    planner.config = MotionPlannerCfg(None, None, trajopt_finetune_attempts=value)
    planner._hold_axis_scope = lambda *args: nullcontext()
    planner.graph_planner = object()
    planner._get_graph_seed_trajectories = lambda *args: torch.zeros((1, 2, 2, 2))
    planner.ik_solver = SimpleNamespace(solve_pose=lambda *args, **kwargs: SimpleNamespace(
        success=torch.ones((1, 2), dtype=torch.bool), solution=torch.zeros((1, 2, 2)),
        total_time=0.1, solve_time=0.1,
    ))
    calls = []

    def solve(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(success=torch.tensor([success]), total_time=0.1, solve_time=0.1)

    planner.trajopt_solver = SimpleNamespace(
        config=SimpleNamespace(num_seeds=2), solve_pose=solve, solve_cspace=solve,
    )
    start = JointState.from_position(torch.zeros((1, 2)))
    graph_attempt = 0 if branch.startswith("graph_") else 1
    if "cspace" in branch:
        result = planner.plan_cspace(
            start, start, max_attempts=1, enable_graph_attempt=graph_attempt,
        )
    else:
        goals = SimpleNamespace(num_goalset=2 if branch == "goalset" else 1)
        result = planner.plan_pose(
            goals, start, max_attempts=1, enable_graph_attempt=graph_attempt,
        )
    assert result.success.item() is success
    assert len(calls) == 1
    if value is None and branch == "goalset":
        assert "finetune_attempts" not in calls[0]
    else:
        default = 3 if "cspace" in branch or branch == "graph_pose" else 1
        assert calls[0]["finetune_attempts"] == (default if value is None else value)
    if branch != "goalset":
        assert calls[0]["finetune_dt_scale"] == (0.75 if branch != "single" else 0.55)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("use_cuda_graph", [False, True])
@pytest.mark.parametrize("branch", ["single", "goalset", "cspace", "posture"])
def test_zero_refinement_keeps_native_initial_solve_and_full_metrics(
    monkeypatch, use_cuda_graph: bool, branch: str,
) -> None:
    config = MotionPlannerCfg.create(
        "franka.yml", num_ik_seeds=16, num_trajopt_seeds=4, max_goalset=2,
        use_cuda_graph=use_cuda_graph, trajopt_finetune_attempts=0,
    )
    assert config.trajopt_finetune_attempts == 0
    with MotionPlanner(config) as planner:
        names = planner.joint_names
        start = JointState.from_position(
            planner.default_joint_state.position.unsqueeze(0), joint_names=names,
        )
        goal_positions = start.position.clone()
        goal_positions[:, 0] += 0.1
        goal = JointState.from_position(goal_positions, joint_names=names)
        validations = []
        original_validate = planner.trajopt_solver._interpolate_and_compute_metrics

        def validate(*args, **kwargs):
            output = original_validate(*args, **kwargs)
            validations.append(bool(output[0].costs_and_constraints.get_feasible(
                include_all_hybrid=False, sum_horizon=True,
            ).any()))
            return output

        monkeypatch.setattr(planner.trajopt_solver, "_interpolate_and_compute_metrics", validate)
        if branch == "cspace":
            result = planner.plan_cspace(goal, start, max_attempts=1)
        elif branch == "posture":
            result = planner.plan_posture(
                goal, start, free_joints=tuple(names[-3:]),
                held_joints=tuple(names[1:4]), max_attempts=1,
            )
        else:
            tool = planner.compute_kinematics(goal).tool_poses
            position = tool.position.unsqueeze(3)
            quaternion = tool.quaternion.unsqueeze(3)
            if branch == "goalset":
                position = position.repeat(1, 1, 1, 2, 1)
                quaternion = quaternion.repeat(1, 1, 1, 2, 1)
                position[..., 1, 0] += 0.01
            goals = GoalToolPose(tool.tool_frames, position, quaternion)
            result = planner.plan_pose(goals, start, max_attempts=1)
        assert result is not None and result.success.all()
        assert validations == [True]
        trajectory = result.get_interpolated_plan().reorder(names)
        assert torch.isfinite(trajectory.position).all()
        assert result.motion_time().gt(0).all()
        if branch == "posture":
            q = trajectory.position.reshape(-1, len(names))
            assert (q[:, 1:4] - start.position[:, 1:4]).abs().max() <= 1.0e-5
            assert (q[-1, 0] - goal.position[0, 0]).abs() <= 0.01
