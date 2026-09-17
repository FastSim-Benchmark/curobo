# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Per-call joint goal sets sharing the native IK + TrajOpt pipeline."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch

from curobo._src.cost.cost_posture import PATH_CONSTRAINT_PRIORITY
from curobo._src.motion.motion_contact import boundary_contact_scope
from curobo._src.solver.solver_trajopt_result import TrajOptSolverResult
from curobo._src.state.state_joint import JointState
from curobo._src.types.axis_hold import AxisHold
from curobo._src.types.tool_pose import GoalToolPose

if TYPE_CHECKING:
    from curobo._src.motion.motion_planner import MotionPlanner


@contextmanager
def posture_scope(
    planner: MotionPlanner,
    goals: JointState,
    start: JointState,
    free_joints: tuple[str, ...],
    held_joints: tuple[str, ...],
    tolerance: float,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Install posture constraints and restore Cartesian criteria on every exit."""
    names = tuple(planner.joint_names)
    if len(set(free_joints)) != len(free_joints) or len(set(held_joints)) != len(held_joints):
        raise ValueError("posture joint declarations must be unique")
    if set(free_joints) & set(held_joints) or (set(free_joints) | set(held_joints)) - set(names):
        raise ValueError("posture free/held joints overlap or are unknown")
    if isinstance(tolerance, bool) or not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("posture tolerance must be positive and finite")
    positions = goals.reorder(names).position
    initial = start.reorder(names).position
    if positions.ndim != 2 or not 1 <= len(positions) <= 256 or initial.shape != (1, len(names)):
        raise ValueError("posture requires 1..256 goals and one start state")
    if not torch.isfinite(positions).all() or not torch.isfinite(initial).all():
        raise ValueError("posture states must be finite")
    mask = positions.new_tensor([name not in free_joints for name in names])
    held = positions.new_tensor([name in held_joints for name in names])
    if not mask.any():
        raise ValueError("posture must constrain at least one joint")
    if ((positions - initial).abs() * held > 1.0e-5).any():
        raise ValueError("held joints cannot have different endpoint targets")
    original_limits = planner.kinematics.get_joint_limits().position.clone()
    held_mask = held.bool()
    if held_mask.any() and (
        (initial[0, held_mask] < original_limits[0, held_mask]).any()
        or (initial[0, held_mask] > original_limits[1, held_mask]).any()
    ):
        raise ValueError("held joint start exceeds position limits")
    costs, saved, axis_weights = [], [], []
    solvers = (planner.ik_solver, planner.trajopt_solver)
    try:
        # Enabling posture changes which residuals execute in the rollout.
        # Resident CUDA graphs must include the current request's constraint.
        for solver in solvers:
            solver.core.invalidate_parameter_graphs()
        if held_mask.any():
            constrained_limits = original_limits.clone()
            constrained_limits[:, held_mask] = initial[:, held_mask]
            planner.update_joint_limits(position=constrained_limits)
        for solver in solvers:
            rollouts = [
                *solver.core.get_all_rollout_instances(),
                *solver.core.additional_metrics_rollouts.values(),
            ]
            for rollout in rollouts:
                for cost in rollout.get_cost_component_by_name("axis_hold"):
                    if (
                        cost is not None
                        and float(cost.config.weight.max()) > 1
                        and all(cost is not item[0] for item in axis_weights)
                    ):
                        axis_weights.append((cost, cost._weight.clone()))
                        # Joint goals leave tool orientation unconstrained. Give
                        # the small angular residual priority over smoothness;
                        # acceptance retains the caller's original tolerance.
                        cost._weight.mul_(PATH_CONSTRAINT_PRIORITY)
                for cost in rollout.get_cost_component_by_name("posture"):
                    if cost is not None and cost not in costs:
                        costs.append(cost)
                        # Optimize to the center; retain numerical slack only for acceptance.
                        slack = tolerance if float(cost.config.weight.max()) <= 1 else 0.0
                        cost.configure(
                            positions,
                            mask,
                            held,
                            initial[0],
                            positions.new_full((len(names),), slack),
                        )
                for cost in rollout.get_cost_component_by_name("tool_pose"):
                    if cost is not None and all(cost is not item[0] for item in saved):
                        criteria = cost._stacked_tool_pose_criteria
                        a = criteria.terminal_pose_axes_weight_factor
                        b = criteria.non_terminal_pose_axes_weight_factor
                        saved.append((cost, a.clone(), b.clone()))
                        a.zero_()
                        b.zero_()
        yield positions, mask, held, initial
    finally:
        if held_mask.any():
            planner.update_joint_limits(position=original_limits)
        for cost, weight in axis_weights:
            cost._weight.copy_(weight)
        for cost in costs:
            cost.deactivate()
        for cost, a, b in saved:
            cost._stacked_tool_pose_criteria.terminal_pose_axes_weight_factor.copy_(a)
            cost._stacked_tool_pose_criteria.non_terminal_pose_axes_weight_factor.copy_(b)
        for solver in solvers:
            solver.core.invalidate_parameter_graphs()


def plan_posture(
    planner: MotionPlanner,
    goal_states: JointState,
    current_state: JointState,
    *,
    free_joints: tuple[str, ...] = (),
    held_joints: tuple[str, ...] = (),
    tolerance: float = 0.01,
    max_attempts: int = 5,
    hold_axis: Mapping[str, AxisHold] | None = None,
    allow_boundary_collision: str = "none",
    max_initial_penetration: float = 0.002,
    contact_links: tuple[str, ...] | None = None,
) -> TrajOptSolverResult | None:
    """Solve one disjunctive terminal joint goal set, with optional free joints.

    This uses native goal-set IK/TrajOpt and axis/contact scopes. Undeclared
    coordinates are provided by the caller as held_joints, not silently freed.
    """
    if allow_boundary_collision not in {"none", "start"}:
        raise ValueError("posture currently supports none/start boundary contact only")
    if (
        isinstance(max_initial_penetration, bool)
        or not isinstance(max_initial_penetration, (int, float))
        or not math.isfinite(max_initial_penetration)
        or max_initial_penetration <= 0
    ):
        raise ValueError("max_initial_penetration must be positive and finite")
    with posture_scope(
        planner, goal_states, current_state, free_joints, held_joints, tolerance
    ) as data:
        positions, mask, held, initial = data
        tool = planner.compute_kinematics(current_state).tool_poses
        # Cartesian tracking is disabled in this scope. The joint cost owns
        # the entire candidate set; do not duplicate dummy Cartesian goals.
        goals = GoalToolPose(
            tool.tool_frames,
            tool.position.unsqueeze(3),
            tool.quaternion.unsqueeze(3),
        )
        with boundary_contact_scope(
            planner,
            current_state,
            None,
            allow_boundary_collision,
            max_initial_penetration,
            contact_links,
        ):
            result = planner.plan_pose(
                goals,
                current_state,
                max_attempts=max_attempts,
                enable_graph_attempt=max_attempts,
                hold_axis=hold_axis,
                allow_boundary_collision="none",
            )
        if result is None:
            return result
        trajectory = result.get_interpolated_plan().reorder(planner.joint_names)
        q = trajectory.position.reshape(-1, planner.action_dim)
        error = ((q[-1] - positions).abs() * mask).amax(-1)
        held_error = ((q - initial).abs() * held).max()
        if result.goalset_index is not None:
            result.goalset_index.fill_(int(error.argmin()))
        debug = {} if result.debug_info is None else dict(result.debug_info)
        debug["posture"] = {
            "candidate_count": len(positions),
            "selected_index": int(error.argmin()),
            "terminal_error": float(error.min()),
            "held_error": float(held_error),
            "terminal_positions": q[-1].tolist(),
            "tolerance": tolerance,
            "native_success": result.success.tolist(),
        }
        result.debug_info = debug
        if not torch.isfinite(q).all() or error.min() > tolerance or held_error > 1.0e-5:
            result.success.fill_(False)
        return result
