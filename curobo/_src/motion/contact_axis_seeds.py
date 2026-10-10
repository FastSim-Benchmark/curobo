"""Try one paired continuous terminal-contact seed without weakening acceptance."""

import time

import numpy as np
from scipy.spatial.transform import Rotation

from curobo._src.motion.motion_goalset import goal_group, remap_goalset_index

from .contact_continuation import plan_contact_continuation


def plan_contact_axis_seed(planner, goals, current, holds, use_implicit_goal=True):
    """Return a native result or no seed; the caller retains ordinary fallback.

    The caller owns the original start-referenced axis scope and shared deadline.
    No task identity, robot link spelling, or scene-specific target is selected.
    """
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
    record = {"original_goal_index": index}
    result = plan_contact_continuation(
        planner,
        goal_group(goals, index),
        current,
        holds,
        use_implicit_goal,
        diagnostics=record,
    )
    if result is None:
        return None
    remap_goalset_index(result, index, len(frames))
    result.total_time = max(result.total_time, time.perf_counter() - started)
    prefix = record.get("seed_record", {})
    result.solve_time += prefix.get("ik_solve_time", 0.0)
    result.debug_info = {**(result.debug_info or {}), "continuous_contact_seed": record}
    return result
