# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Exact complete posture endpoints retain the native GPU trajectory checks."""

import pytest
import torch

from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import JointState


@pytest.mark.parametrize("use_cuda_graph", [False, True])
def test_complete_posture_preserves_a_wrist_target_near_the_original_joint_limit(use_cuda_graph):
    """A fully specified target is not moved by an unnecessary IK projection."""
    with MotionPlanner(
        MotionPlannerCfg.create(
            "franka.yml",
            num_ik_seeds=16,
            num_trajopt_seeds=4,
            use_cuda_graph=use_cuda_graph,
        )
    ) as planner:
        start = JointState.from_position(
            planner.default_joint_state.position.unsqueeze(0), joint_names=planner.joint_names
        )
        limits = planner.kinematics.get_joint_limits().position.clone()
        position = start.position.clone()
        position[0, -1] = limits[0, -1] + 0.04
        goal = JointState.from_position(position, joint_names=planner.joint_names)
        result = planner.plan_posture(goal, start, tolerance=1.0e-5, max_attempts=3)
        assert result is not None and result.success.all()
        path = result.get_interpolated_plan().reorder(planner.joint_names).position
        assert torch.isfinite(path).all()
        assert (path.reshape(-1, len(planner.joint_names))[-1] - position[0]).abs().max() <= 1.e-5
        assert result.debug_info["posture"]["fully_specified_cspace"]
        assert torch.equal(planner.kinematics.get_joint_limits().position, limits)
