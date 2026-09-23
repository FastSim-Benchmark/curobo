# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Bounded grouped-goal fallback after ordinary goalset search is exhausted."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from curobo._src.types.tool_pose import GoalToolPose
from curobo._src.util.logging import log_warn

if TYPE_CHECKING:
    from curobo._src.motion.motion_planner import MotionPlanner
    from curobo._src.state.state_joint import JointState


def goal_group(goals: GoalToolPose, index: int) -> GoalToolPose:
    """Preserve every tool's matching target from one original group."""
    return GoalToolPose(
        tool_frames=goals.tool_frames.copy(),
        position=goals.position[..., index : index + 1, :].contiguous(),
        quaternion=goals.quaternion[..., index : index + 1, :].contiguous(),
    )


def remap_goalset_index(result, index: int, tool_count: int) -> None:
    """A one-group solver result still refers to its original caller goalset."""
    if result.goalset_index is None:
        result.goalset_index = torch.full(
            (*result.success.shape, tool_count),
            index,
            dtype=torch.long,
            device=result.success.device,
        )
    else:
        result.goalset_index = torch.where(
            result.goalset_index >= 0,
            torch.full_like(result.goalset_index, index),
            result.goalset_index,
        )


def plan_goalset_fallback(
    planner: MotionPlanner,
    goals: GoalToolPose,
    current: JointState,
    use_implicit_goal: bool,
    max_attempts: int,
):
    """Try original target groups individually with all ordinary solver checks.

    A group retains every tool's target at the same original goalset index.
    The extra endpoint-search budget is bounded by the existing attempt and
    IK-seed budgets, and never revisits a group. Returned indices remain in the
    original goalset even though each internal query contains one group.
    """
    budget = min(goals.num_goalset, max_attempts * planner.ik_solver.config.num_seeds)
    summary = {
        "goal_count": goals.num_goalset,
        "goal_budget": budget,
        "attempted_goal_indices": [],
        "ik_feasible_goal_count": 0,
        "trajectory_attempt_count": 0,
    }
    last_result = None
    elapsed = 0.0
    solve_time = 0.0
    for index in range(budget):
        target = goal_group(goals, index)
        summary["attempted_goal_indices"].append(index)
        ik = planner.ik_solver.solve_pose(
            target, current_state=current, return_seeds=planner.trajopt_solver.config.num_seeds
        )
        elapsed += ik.total_time
        solve_time += ik.solve_time
        if not bool(ik.success.any()):
            continue
        summary["ik_feasible_goal_count"] += 1
        seeds = ik.solution.clone()
        seeds[~ik.success] = seeds[ik.success][0].clone()
        kwargs = (
            {}
            if planner.config.trajopt_finetune_attempts is None
            else {"finetune_attempts": planner.config.trajopt_finetune_attempts}
        )
        result = planner.trajopt_solver.solve_pose(
            target, current, seed_config=seeds, use_implicit_goal=use_implicit_goal, **kwargs
        )
        elapsed += result.total_time
        solve_time += result.solve_time
        summary["trajectory_attempt_count"] += 1
        remap_goalset_index(result, index, len(target.tool_frames))
        last_result = result
        if bool(result.success.any()):
            summary["selected_original_goal_index"] = index
            break
    if last_result is not None:
        debug = dict(last_result.debug_info or {})
        debug["goalset_fallback"] = summary
        last_result.debug_info = debug
        last_result.total_time = elapsed
        last_result.solve_time = solve_time
    if last_result is None or not bool(last_result.success.any()):
        log_warn(f"Goalset fallback exhausted its endpoint-search budget: {summary}")
    return last_result
