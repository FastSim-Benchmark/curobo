# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime speed and held-coordinate updates preserve native allocations."""

import pytest
import torch

from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg


@pytest.fixture(params=[False, True])
def planner(request):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    config = MotionPlannerCfg.create(
        robot="franka.yml",
        device_cfg=DeviceCfg(device="cuda:0"),
        num_ik_seeds=8,
        num_trajopt_seeds=2,
        use_cuda_graph=request.param,
    )
    instance = MotionPlanner(config)
    yield instance
    instance.destroy()


def test_runtime_limits_hold_joints_restore_speed_and_keep_instances(planner):
    components = (planner.ik_solver, planner.trajopt_solver, planner.graph_planner)
    limits = planner.kinematics.get_joint_limits()
    original = {
        name: getattr(limits, name).clone()
        for name in ("position", "velocity", "acceleration", "jerk")
    }
    pointers = {name: getattr(limits, name).data_ptr() for name in original}
    start = planner.default_joint_state.position.reshape(1, -1).clone()
    start[0, 0] = 0.2
    goal = start.clone()
    goal[0, 1] += 0.12
    native_start = JointState.from_position(start, joint_names=planner.joint_names)
    native_goal = JointState.from_position(goal, joint_names=planner.joint_names)
    # Capture optimizer graphs before changing limits.
    first = planner.plan_cspace(native_goal, native_start, max_attempts=2)
    assert bool(first.success.all())
    for scale in (0.3, 1.0):
        position = original["position"].clone()
        position[:, 0] = start[0, 0]
        planner.update_joint_limits(
            position=position,
            velocity=original["velocity"] * scale,
            acceleration=original["acceleration"] * scale,
            jerk=original["jerk"] * scale,
        )
        result = planner.plan_cspace(native_goal, native_start, max_attempts=2)
        assert bool(result.success.all())
        trajectory = result.interpolated_trajectory.reorder(planner.joint_names)
        assert torch.max(torch.abs(trajectory.position[..., 0] - start[0, 0])) < 1e-5
        assert torch.max(torch.abs(trajectory.velocity[..., 0])) < 1e-5
        assert torch.all(trajectory.velocity.abs() <= original["velocity"][1] * scale * 1.01)
        assert torch.all(
            trajectory.acceleration.abs() <= original["acceleration"][1] * scale * 1.01
        )
        assert torch.all(trajectory.jerk.abs() <= original["jerk"][1] * scale * 1.01)
        assert components == (planner.ik_solver, planner.trajopt_solver, planner.graph_planner)
        assert pointers == {name: getattr(limits, name).data_ptr() for name in original}
    planner.update_joint_limits(**original)
    assert torch.equal(limits.position, original["position"])


def test_invalid_limits_are_atomic(planner):
    limits = planner.kinematics.get_joint_limits()
    before = limits.position.clone()
    with pytest.raises(ValueError):
        planner.update_joint_limits(
            position=before * 0.9, velocity=torch.zeros_like(limits.velocity)
        )
    assert torch.equal(limits.position, before)


def test_held_projection_preserves_real_violations_and_active_gradients(planner):
    """Only constant numerical roundoff is projected, without masking real motion."""
    transition = planner.trajopt_solver.core.auxiliary_rollout.transition_model
    limits = planner.kinematics.get_joint_limits().position.clone()
    limits[:, 0] = 0.2
    planner.update_joint_limits(position=limits)
    position = planner.default_joint_state.position.reshape(1, 1, -1).repeat(1, 8, 1)
    position[..., 0] = 0.2 + 1e-7
    state = JointState.from_position(position.requires_grad_(), joint_names=planner.joint_names)
    state.velocity = torch.ones_like(position) * 1e-5
    projected = transition.project_held_coordinates(state)
    assert torch.all(projected.position[..., 0] == limits[0, 0])
    assert torch.count_nonzero(projected.velocity[..., 0]) == 0
    projected.position.sum().backward()
    assert torch.count_nonzero(position.grad[..., 0]) == 0
    assert torch.all(position.grad[..., 1:] == 1.0)
    moved = state.clone()
    moved.position = moved.position.detach()
    moved.position[:, 3, 0] += 1e-4
    rejected = transition.project_held_coordinates(moved)
    assert torch.equal(rejected.position, moved.position)
    assert torch.equal(rejected.velocity, moved.velocity)
