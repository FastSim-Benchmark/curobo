# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""Partial joint goal sets preserve held coordinates and isolate requests."""

import pytest
import torch

from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import AxisHold, JointState


@pytest.mark.parametrize("axis_hold", [False, True])
@pytest.mark.parametrize("use_cuda_graph", [False, True])
def test_posture_goalset_and_next_joint_query(axis_hold, use_cuda_graph):
    with MotionPlanner(
        MotionPlannerCfg.create(
            "franka.yml",
            num_ik_seeds=16,
            num_trajopt_seeds=4,
            use_cuda_graph=use_cuda_graph,
        )
    ) as planner:
        names = planner.joint_names
        start = JointState.from_position(
            planner.default_joint_state.position.unsqueeze(0), joint_names=names
        )
        assert planner.plan_cspace(start, start, max_attempts=1).success.all()
        positions = start.position.repeat(2, 1)
        positions[:, 0] += positions.new_tensor([0.1, 0.2])
        goals = JointState.from_position(positions, joint_names=names)
        original_limits = planner.kinematics.get_joint_limits().position.clone()
        result = planner.plan_posture(
            goals,
            start,
            free_joints=tuple(names[-3:]),
            held_joints=tuple(names[1:4]),
            max_attempts=2,
            hold_axis={planner.tool_frames[0]: AxisHold()} if axis_hold else None,
        )
        assert result is not None and result.success.all()
        trajectory = result.get_interpolated_plan().reorder(names).position.reshape(-1, len(names))
        assert (trajectory[-1, 0] - positions[:, 0]).abs().min() <= 0.01
        assert int(result.goalset_index.reshape(-1)[0]) == int(
            (trajectory[-1, 0] - positions[:, 0]).abs().argmin()
        )
        assert (trajectory[:, 1:4] - start.position[:, 1:4]).abs().max() <= 1.0e-5
        assert torch.equal(planner.kinematics.get_joint_limits().position, original_limits)
        if axis_hold:
            from curobo.tests._src.motion.test_axis_hold_parameter import maximum_tilt

            assert maximum_tilt(planner, result, start, (0, 0, 1)) <= 0.01
        assert planner.plan_cspace(start, start, max_attempts=1).success.all()
        with pytest.raises(ValueError, match="overlap or are unknown"):
            planner.plan_posture(goals, start, free_joints=("missing",))
        conflicting = positions.clone()
        conflicting[:, 1] += 0.001
        with pytest.raises(ValueError, match="different endpoint targets"):
            planner.plan_posture(
                JointState.from_position(conflicting, joint_names=names), start,
                held_joints=(names[1],), tolerance=0.01,
            )
        from curobo._src.motion.motion_posture import posture_scope

        with pytest.raises(RuntimeError, match="abort posture"):
            with posture_scope(planner, goals, start, (), (names[1],), 0.01):
                assert torch.equal(
                    planner.kinematics.get_joint_limits().position[:, 1],
                    start.position[0, 1].expand(2),
                )
                raise RuntimeError("abort posture")
        assert torch.equal(planner.kinematics.get_joint_limits().position, original_limits)


def test_goalset_cost_cannot_mix_candidates_and_rejects_held_motion():
    from curobo._src.cost.cost_base_cfg import BaseCostCfg
    from curobo._src.cost.cost_posture import PostureCost

    cost = PostureCost(BaseCostCfg(weight=1.0), 3)
    device = cost.goals.device
    goals = torch.tensor([[0.0, 1.0, 0.3], [1.0, 0.0, 0.3]], device=device)
    cost.configure(
        goals,
        torch.ones(3, device=device),
        torch.tensor([0.0, 0.0, 1.0], device=device),
        goals[0],
        torch.full((3,), 0.005, device=device),
    )
    assert cost.forward(goals[0].reshape(1, 1, 3)).sum() == 0
    mixed = torch.tensor([[[0.0, 0.0, 0.3]]], device=device)
    assert cost.forward(mixed).sum() > 0
    path = goals[0].repeat(1, 3, 1)
    path[0, 1, 2] += 0.1
    assert cost.forward(path).sum() > 0
    cost.active.zero_()
    assert cost.forward(path).sum() == 0


def test_posture_departs_mesh_support_and_restores_ordinary_checks(tmp_path):
    from curobo.examples.reference.contact_separation import robot_config, scene_config
    from curobo.scene import Scene

    table = scene_config(False).cuboid[0]
    panel = table.get_mesh()
    panel.name = "table_panel"
    config = MotionPlannerCfg.create(
        robot_config(tmp_path),
        scene_model=Scene(mesh=[table.get_mesh(), panel]),
        num_ik_seeds=8,
        num_trajopt_seeds=4,
        use_cuda_graph=True,
    )
    with MotionPlanner(config) as planner:
        start = JointState.from_position(
            planner.device_cfg.to_device([[0, 0, 0.12]]),
            joint_names=planner.joint_names,
        )
        planner.attachment_manager.update(
            planner.device_cfg.to_device([[0, 0, -0.08, 0.041]]),
            start,
        )
        goals = JointState.from_position(
            planner.device_cfg.to_device([[-0.6, 0, 0.17], [-0.65, 0, 0.17]]),
            joint_names=planner.joint_names,
        )
        result = planner.plan_posture(
            goals,
            start,
            held_joints=("y",),
            allow_boundary_collision="start",
            contact_links=("attached_object",),
            max_attempts=3,
        )
        assert result is not None and result.success.all()
        ordinary = planner.plan_posture(goals, start, held_joints=("y",), max_attempts=1)
        assert ordinary is not None and not ordinary.success.any()
        assert "selected_constraint_maxima" in ordinary.debug_info


def test_posture_retry_projects_waypoints_and_keeps_native_acceptance(monkeypatch):
    """Exercise the retry branch even when a simple direct path already works."""
    with MotionPlanner(MotionPlannerCfg.create(
        "franka.yml", num_ik_seeds=16, num_trajopt_seeds=4, use_cuda_graph=False,
    )) as planner:
        names = planner.joint_names
        start = JointState.from_position(
            planner.default_joint_state.position.unsqueeze(0), joint_names=names,
        )
        goal = start.clone()
        goal.position[:, 0] += .1
        solve = planner.trajopt_solver.solve_pose
        seeded_calls = []

        def reject_first(*args, **kwargs):
            result = solve(*args, **kwargs)
            seeded_calls.append(kwargs.get("seed_traj") is not None)
            if len(seeded_calls) == 1:
                result.success.fill_(False)
            return result

        monkeypatch.setattr(planner.trajopt_solver, "solve_pose", reject_first)
        result = planner.plan_posture(
            goal, start, free_joints=(names[-1],), held_joints=tuple(names[1:-1]),
            max_attempts=3,
        )
        assert result is not None and result.success.all()
        assert seeded_calls[0] is False and any(seeded_calls[1:])
        assert any(row["accepted"] > 0 for row in result.debug_info["posture_seed_attempts"])
        q = result.get_interpolated_plan().reorder(names).position.reshape(-1, len(names))
        assert (q[-1, 0] - goal.position[0, 0]).abs() <= .01
        assert (q[:, 1:-1] - start.position[:, 1:-1]).abs().max() <= 1e-5
