# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Keep each terminal contact bound to the exact endpoint that justified it."""

from __future__ import annotations

from itertools import islice
from typing import TYPE_CHECKING

from curobo._src.motion.motion_contact import boundary_contact_scope, contact_goal_candidates
from curobo._src.motion.motion_goalset import goal_group, remap_goalset_index
from curobo._src.util.logging import log_warn

if TYPE_CHECKING:
    from curobo._src.motion.motion_planner import MotionPlanner
    from curobo._src.state.state_joint import JointState
    from curobo._src.types.tool_pose import GoalToolPose


def plan_terminal_pose(
    planner: MotionPlanner,
    goals: GoalToolPose,
    current: JointState,
    use_implicit_goal: bool,
    max_attempts: int,
    mode: str,
):
    """Validate and seed each endpoint inside its own immutable contact scope.

    Existing budgets bound groups by attempts times IK seeds, and each group's
    captured candidates by attempts. No optimizer reruns the endpoint IK after
    capture: complete IK metrics validate that same state before native TrajOpt.
    """
    budget = min(goals.num_goalset, max_attempts * planner.ik_solver.config.num_seeds)
    summary = {
        "goal_count": goals.num_goalset,
        "goal_budget": budget,
        "candidate_budget_per_goal": max_attempts,
        "attempted_goal_indices": [],
        "captured_candidate_count": 0,
        "endpoint_metric_rejection_count": 0,
        "trajectory_attempt_count": 0,
        "attempts": [],
    }
    last_result = None
    for index in range(budget):
        target = goal_group(goals, index)
        summary["attempted_goal_indices"].append(index)
        candidates = contact_goal_candidates(planner, target, current)
        for candidate_index, endpoint in enumerate(islice(candidates, max_attempts)):
            summary["captured_candidate_count"] += 1
            record = {"goal_index": index, "candidate_index": candidate_index}
            summary["attempts"].append(record)
            with boundary_contact_scope(planner, current, endpoint, mode):
                seeds = endpoint.position.reshape(1, 1, -1).repeat(
                    1, planner.ik_solver.config.num_seeds, 1
                )
                checked = planner.ik_solver.solve_pose(
                    target,
                    current_state=current,
                    seed_config=seeds,
                    return_seeds=planner.trajopt_solver.config.num_seeds,
                    run_optimizer=False,
                )
                record["endpoint_metrics_success"] = bool(checked.success.any())
                if not record["endpoint_metrics_success"]:
                    summary["endpoint_metric_rejection_count"] += 1
                    continue
                seeds = endpoint.position.reshape(1, 1, -1).repeat(
                    1, planner.trajopt_solver.config.num_seeds, 1
                )
                finetune = planner.config.trajopt_finetune_attempts
                result = planner.trajopt_solver.solve_pose(
                    target,
                    current,
                    seed_config=seeds,
                    use_implicit_goal=use_implicit_goal,
                    finetune_attempts=1 if finetune is None else finetune,
                    finetune_dt_scale=0.55,
                )
                summary["trajectory_attempt_count"] += 1
                record["trajectory_success"] = bool(result.success.any())
                remap_goalset_index(result, index, len(target.tool_frames))
                last_result = result
                if not record["trajectory_success"] and "selected_constraint_maxima" in (
                    result.debug_info or {}
                ):
                    from curobo._src.motion.motion_failure_diagnostics import (
                        terminal_failure_summary,
                    )

                    evidence = terminal_failure_summary(planner, result)
                    log_warn(f"Terminal trajectory failure evidence: {evidence}")
            if record["trajectory_success"]:
                summary["selected_original_goal_index"] = index
                break
        if last_result is not None and bool(last_result.success.any()):
            break
    if last_result is not None:
        debug = dict(last_result.debug_info or {})
        debug["terminal_contact_search"] = summary
        last_result.debug_info = debug
    if last_result is None or not bool(last_result.success.any()):
        log_warn(f"Terminal contact planning exhausted its bounded candidate search: {summary}")
    return last_result
