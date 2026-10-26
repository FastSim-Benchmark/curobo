"""Try one paired continuous terminal-contact seed without weakening acceptance."""

import math
import time

import numpy as np
from scipy.spatial.transform import Rotation

from curobo._src.motion.motion_goalset import goal_group, remap_goalset_index

from .contact_continuation import plan_contact_continuation


def _contact_seed_summary(record, outcome):
    """Retain fixed-size scalar evidence, without copying per-joint arrays."""
    summary = {"outcome": outcome if type(outcome) is str and outcome in (
        "interrupted", "no_seed", "success", "rejected",
    ) else "unrecognized"}
    index = record.get("original_goal_index")
    if type(index) is int and 0 <= index < 1_000_000:
        summary["original_goal_index"] = index
    for name in ("endpoint_proposed", "continuous_seed_present", "endpoint_metrics_success"):
        value = record.get(name)
        if type(value) is bool:
            summary[name] = value
    continuation = record.get("continuation")
    if type(continuation) is dict:
        step = continuation.get("last_step")
        if type(step) is int and 0 <= step <= 16:
            summary["continuation_last_step"] = step
        delta = continuation.get("remaining_endpoint_delta")
        if type(delta) in (list, tuple) and len(delta) <= 4096:
            if all(
                (type(value) is float and math.isfinite(value))
                or (type(value) is int and value.bit_length() <= 1023)
                for value in delta
            ):
                summary["remaining_endpoint_delta_absmax"] = max(
                    (abs(float(value)) for value in delta), default=0.0
                )
    seed = record.get("seed_record")
    tail = seed.get("terminal_tail_seed_proposal") if type(seed) is dict else None
    summary["tail_seed_proposal"] = type(tail) is dict
    if type(tail) is dict:
        step = tail.get("from_ik_step")
        if type(step) is int and 1 <= step <= 16:
            summary["tail_seed_proposal_from_ik_step"] = step
        reason = tail.get("reason")
        summary["tail_seed_proposal_reason"] = (
            "native_ik_no_valid_seed"
            if type(reason) is str and reason == "native_ik_no_valid_seed" else "unrecognized"
        )
        qualification = tail.get("qualification")
        summary["tail_seed_proposal_qualification"] = (
            "seed_proposal_only_not_native_trajectory_acceptance"
            if (type(qualification) is str
                and qualification == "seed_proposal_only_not_native_trajectory_acceptance")
            else "unrecognized"
        )
    return summary


def plan_contact_axis_seed(
    planner, goals, current, holds, use_implicit_goal=True, *, diagnostics=None
):
    """Return a native result or no seed; the caller retains ordinary fallback.

    The caller owns the original start-referenced axis scope and shared deadline.
    No task identity, robot link spelling, or scene-specific target is selected.
    """
    record = {}
    outcome = "interrupted"
    try:
        return _plan_contact_axis_seed(
            planner, goals, current, holds, use_implicit_goal, record,
        )
    finally:
        outcome = record.pop("_outcome", outcome)
        summary = _contact_seed_summary(record, outcome)
        if diagnostics is not None:
            diagnostics.clear()
            diagnostics.update(summary)


def _plan_contact_axis_seed(planner, goals, current, holds, use_implicit_goal, record):
    """Keep native planning and timing independent from diagnostic retention."""
    started = time.perf_counter()
    poses = planner.compute_kinematics(current).tool_poses
    frames = list(poses.tool_frames)
    goals = goals.reorder_links(frames)
    p0 = poses.position.reshape(-1, len(frames), 3)[0].detach().cpu().numpy()
    q0 = poses.quaternion.reshape(-1, len(frames), 4)[0].detach().cpu().numpy()
    positions = goals.position[0, 0].detach().cpu().numpy()
    quaternions = goals.quaternion[0, 0].detach().cpu().numpy()
    held = [frames.index(frame) for frame in holds]
    if not held:
        raise ValueError("A continuous axis seed requires an axis-hold frame")

    def rank(index):
        angles = [
            (
                Rotation.from_quat(q0[j, [1, 2, 3, 0]]).inv()
                * Rotation.from_quat(quaternions[j, index, [1, 2, 3, 0]])
            ).magnitude()
            for j in held
        ]
        return sum(angles), float(np.linalg.norm(positions[held, index] - p0[held]))

    index = min(range(goals.num_goalset), key=rank)
    record["original_goal_index"] = index
    result = plan_contact_continuation(
        planner,
        goal_group(goals, index),
        current,
        holds,
        use_implicit_goal,
        diagnostics=record,
    )
    if result is None:
        record["_outcome"] = "no_seed"
        return None
    remap_goalset_index(result, index, len(frames))
    result.total_time = max(result.total_time, time.perf_counter() - started)
    prefix = record.get("seed_record", {})
    result.solve_time += prefix.get("ik_solve_time", 0.0)
    record["_outcome"] = "success" if bool(result.success.any()) else "rejected"
    debug = dict(result.debug_info or {})
    debug.pop("continuous_contact_seed", None)
    result.debug_info = {
        "continuous_contact_seed": _contact_seed_summary(record, record["_outcome"]), **debug
    }
    return result
