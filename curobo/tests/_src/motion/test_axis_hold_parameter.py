# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Per-query axis constraints, endpoint semantics, and CUDA graph isolation."""

import math

import pytest
import torch

from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import AxisHold, GoalToolPose, JointState, Pose, ToolPoseCriteria


@pytest.fixture(scope="module")
def planner():
    with MotionPlanner(
        MotionPlannerCfg.create(
            "franka.yml",
            num_ik_seeds=16,
            num_trajopt_seeds=4,
            interpolation_dt=0.01,
        )
    ) as value:
        yield value


def start_state(planner):
    return JointState.from_position(
        planner.default_joint_state.position.unsqueeze(0),
        joint_names=planner.joint_names,
    )


def maximum_tilt(planner, result, start, axis):
    trajectory = result.get_interpolated_plan().reorder(planner.joint_names)
    positions = trajectory.position.reshape(-1, planner.action_dim)
    fractions = torch.arange(8, device=positions.device) / 8
    dense = (
        positions[:-1, None] + fractions[None, :, None] * positions.diff(dim=0)[:, None]
    ).reshape(-1, planner.action_dim)
    dense = torch.cat([dense, positions[-1:]])
    q = planner.compute_kinematics(
        JointState.from_position(dense, joint_names=planner.joint_names)
    ).tool_poses.quaternion
    q0 = planner.compute_kinematics(start).tool_poses.quaternion
    direction = Pose(
        quaternion=q.reshape(-1, 4)
    ).get_rotation_matrix() @ planner.device_cfg.to_device(list(axis))
    reference = Pose(
        quaternion=q0.reshape(-1, 4)
    ).get_rotation_matrix() @ planner.device_cfg.to_device(list(axis))
    cross = torch.linalg.cross(direction, reference.expand_as(direction), dim=-1)
    return torch.atan2(cross.norm(dim=-1), (direction * reference).sum(-1)).max().item()


def test_joint_goal_twists_freely_then_incompatible_goal_fails_and_next_call_is_clear(planner):
    start = start_state(planner)
    yaw_goal = start.clone()
    yaw_goal.position[..., -1] += 0.3
    hold = {planner.tool_frames[0]: AxisHold()}
    # First capture unconfigured graphs, then replay with changed parameter values.
    assert planner.plan_cspace(yaw_goal, start, max_attempts=1).success.all()
    result = planner.plan_cspace(yaw_goal, start, max_attempts=2, hold_axis=hold)
    assert result is not None and result.success.all()
    assert maximum_tilt(planner, result, start, (0, 0, 1)) <= 0.01
    endpoint = (
        result.get_interpolated_plan()
        .reorder(planner.joint_names)
        .position.reshape(-1, planner.action_dim)[-1]
    )
    torch.testing.assert_close(endpoint, yaw_goal.position[0], atol=1e-5, rtol=0)

    tilted_goal = start.clone()
    tilted_goal.position[..., 3] += 0.2
    result = planner.plan_cspace(tilted_goal, start, max_attempts=1, hold_axis=hold)
    assert result is None or not result.success.any()
    assert planner.plan_cspace(tilted_goal, start, max_attempts=1).success.all()
    # Each invocation captures its own reference, including a changed start pose.
    assert planner.plan_cspace(
        tilted_goal, tilted_goal, max_attempts=1, hold_axis=hold
    ).success.all()


def test_pose_motion_holds_arbitrary_tool_local_axis_and_clears_on_exception(planner, monkeypatch):
    start = start_state(planner)
    tool = planner.compute_kinematics(start).tool_poses
    axis = (
        (
            Pose(quaternion=tool.quaternion.reshape(-1, 4)).get_rotation_matrix().transpose(-1, -2)
            @ planner.device_cfg.to_device([0.0, 0.0, 1.0])
        )
        .reshape(3)
        .tolist()
    )
    goal = GoalToolPose(
        tool.tool_frames,
        tool.position.unsqueeze(3) + planner.device_cfg.to_device([0, 0.12, 0.06]),
        tool.quaternion.unsqueeze(3).clone(),
    )
    hold = {planner.tool_frames[0]: AxisHold(tuple(axis))}
    result = planner.plan_pose(goal, start, max_attempts=2, hold_axis=hold)
    assert result is not None and result.success.all()
    assert maximum_tilt(planner, result, start, axis) <= 0.01
    with monkeypatch.context() as patch:

        def fail(*args, **kwargs):
            raise RuntimeError("injected planning failure")

        patch.setattr(planner, "_plan_pose", fail)
        with pytest.raises(RuntimeError, match="injected"):
            planner.plan_pose(goal, start, hold_axis=hold)
    for solver in (planner.ik_solver, planner.trajopt_solver):
        for rollout in [
            *solver.core.get_all_rollout_instances(),
            *solver.core.additional_metrics_rollouts.values(),
        ]:
            for cost in rollout.get_cost_component_by_name("axis_hold"):
                assert not cost._stacked_tool_pose_criteria.terminal_pose_axes_weight_factor.any()


@pytest.mark.parametrize("axis", [(0, 0, 0), (1, 2), (math.nan, 0, 1), (True, 0, 1)])
def test_invalid_axis_is_rejected(axis):
    with pytest.raises(ValueError):
        AxisHold(axis)


@pytest.mark.parametrize("tolerance", [0, -1, math.pi, math.inf, math.nan, True])
def test_invalid_tolerance_is_rejected(tolerance):
    with pytest.raises(ValueError):
        AxisHold(tolerance_rad=tolerance)


def test_initial_contact_combines_with_per_motion_parameter():
    from curobo.examples.reference.cup_upright import make_cup_planner, validate_cup

    planner, start, goal, axis, support, contact = make_cup_planner()
    try:
        # Exercise the per-query cost without the earlier global axis-hold recipe.
        planner.update_tool_pose_criteria(
            {
                frame: ToolPoseCriteria(terminal_pose_convergence_tolerance=[0.0, 0.01])
                for frame in planner.tool_frames
            }
        )
        goal.quaternion.copy_(planner.compute_kinematics(start).tool_poses.quaternion.unsqueeze(3))
        result = planner.plan_pose(
            goal,
            start,
            max_attempts=2,
            enable_graph_attempt=2,
            hold_axis={planner.tool_frames[0]: AxisHold(tuple(axis.tolist()))},
        )
        assert result is not None and result.success.all()
        assert maximum_tilt(planner, result, start, axis.tolist()) <= 0.01
        assert contact is not None
        assert validate_cup(planner, result.get_interpolated_plan(), axis, support, contact)[
            "contact_valid"
        ]
    finally:
        planner.destroy()


def test_goal_contact_and_normal_landing_combine_with_axis_hold(tmp_path):
    from curobo.examples.reference.contact_placement import validate_landing
    from curobo.examples.reference.contact_separation import make_planner, validate_trajectory

    planner, start, goal, scene = make_planner(
        tmp_path,
        "cabinet",
        True,
        placement=True,
        normal_landing=True,
        placement_goal=(0.0, 0.0, 0.12),
    )
    try:
        result = planner.plan_cspace(
            goal,
            start,
            max_attempts=2,
            enable_graph_attempt=2,
            hold_axis={"gripper": AxisHold()},
        )
        assert result is not None and result.success.all()
        positions = result.get_interpolated_plan().position.reshape(-1, 3)
        assert validate_trajectory(planner, positions, scene, True, placement=True)["valid"]
        metrics = planner.config.trajopt_solver_config.core_cfg.metrics_rollout_config
        contact = metrics.constraint_cfg.scene_collision_cfg.goal_contact
        landing = validate_landing(planner, positions, scene, contact)
        assert landing["landing_valid"]
        assert landing["contact_before_alignment_count"] == 0
    finally:
        planner.destroy()
