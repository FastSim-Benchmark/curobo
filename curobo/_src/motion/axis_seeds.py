# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Request-local continuous IK seeds for Cartesian axis-constrained paths."""

import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp

from curobo._src.motion.motion_goalset import goal_group, remap_goalset_index
from curobo._src.state.state_joint import JointState
from curobo._src.types.tool_pose import GoalToolPose
from curobo._src.util.time_budget import check_time_budget


def local_search_radius(kinematics, position):
    """Use joint types, including negative axes, rather than robot joint names."""
    radius = torch.full_like(position, 0.15)
    config = kinematics.config.kinematics_config
    for joint, kind in zip(config.joint_map.tolist(), config.joint_map_type.tolist()):
        if 0 <= joint < len(position) and kind in (0, 1, 2, 6, 7, 8):
            radius[joint] = 0.02
    return radius


def resample_seed(path, horizon, count):
    """Resample within each segment's convex hull, preserving both endpoints."""
    if path.ndim != 2 or len(path) < 2 or horizon < 2 or count < 1:
        raise ValueError("continuous seed requires a path, >=2 knots and positive count")
    if not bool(torch.isfinite(path).all()):
        raise ValueError("continuous seed positions must be finite")
    locations = torch.linspace(0, len(path) - 1, horizon, device=path.device, dtype=path.dtype)
    lower = locations.long().clamp(max=len(path) - 2)
    seed = path[lower] + (locations - lower)[:, None] * (path[lower + 1] - path[lower])
    seed[0], seed[-1] = path[0], path[-1]
    return seed[None, None].repeat(1, count, 1, 1)


def continuous_axis_seed(planner, goals, current, holds, *, terminal_proposal=None,
                         terminal_hint=None, diagnostics=None):
    """Try one paired target with 16 bounded IK steps; always restore limits.

    The caller owns the original start-referenced axis scope and time budget.
    This produces a seed only. Native trajectory checks still decide acceptance.
    """
    diagnostics = {} if diagnostics is None else diagnostics
    names = planner.joint_names
    state = current.reorder(names).clone()
    starting = planner.compute_kinematics(state).tool_poses
    frames = list(starting.tool_frames)
    goals = goals.reorder_links(frames)
    p0 = starting.position.reshape(-1, len(frames), 3)[0].detach().cpu().numpy().copy()
    q0 = starting.quaternion.reshape(-1, len(frames), 4)[0].detach().cpu().numpy().copy()
    positions = goals.position[0, 0].detach().cpu().numpy()
    quaternions = goals.quaternion[0, 0].detach().cpu().numpy()
    held = [frames.index(frame) for frame in holds]

    def rank(index):
        angles = [(Rotation.from_quat(q0[j, [1, 2, 3, 0]]).inv()
                   * Rotation.from_quat(quaternions[j, index, [1, 2, 3, 0]])).magnitude()
                  for j in held]
        return sum(angles), float(np.linalg.norm(positions[held, index] - p0[held]))

    index = min(range(goals.num_goalset), key=rank)
    rotations = [Slerp([0, 1], Rotation.from_quat(
        np.array([q0[j], quaternions[j, index]])[:, [1, 2, 3, 0]],
    )) for j in range(len(frames))]
    bounds = planner.kinematics.get_joint_limits().position.clone()
    radius = local_search_radius(planner.kinematics, state.position[0])
    hint = None if terminal_hint is None else terminal_hint.reorder(names).position.reshape(-1)
    if hint is not None and (hint.shape != state.position[0].shape
            or not bool(torch.isfinite(hint).all())
            or bool(((hint < bounds[0]) | (hint > bounds[1])).any())):
        raise ValueError("Invalid terminal hint")
    initial = state.position[0].clone()
    points = [initial]
    record = {"original_goal_index": index, "ik_steps": 0, "ik_time": 0.0,
              "ik_solve_time": 0.0}
    count = planner.ik_solver.config.num_seeds
    try:
        for step in range(1, 17):
            check_time_budget()
            alpha = step / 16
            p = p0 + alpha * (positions[:, index] - p0)
            q = np.stack([rotation(alpha).as_quat()[[3, 0, 1, 2]]
                          for rotation in rotations])
            target = GoalToolPose(
                frames, torch.as_tensor(p, device=state.position.device,
                                        dtype=state.position.dtype)[None, None, :, None],
                torch.as_tensor(q, device=state.position.device,
                                dtype=state.position.dtype)[None, None, :, None],
            )
            local = torch.stack((torch.maximum(bounds[0], state.position[0] - radius),
                                 torch.minimum(bounds[1], state.position[0] + radius)))
            if hint is not None:
                remaining = 16 - step
                local[0] = torch.maximum(local[0], hint - remaining * radius)
                local[1] = torch.minimum(local[1], hint + remaining * radius)
            diagnostics["last_step"] = step
            diagnostics["remaining_endpoint_delta"] = (
                None if hint is None else (hint-state.position[0]).detach().cpu().tolist())
            if bool((local[0] > local[1]).any()):
                return None
            planner.update_joint_limits(position=local)
            if step == 16 and terminal_proposal is not None:
                proposed = terminal_proposal(planner, target, state)
                record["ik_steps"] = step
                if proposed is None:
                    return None
                chosen = proposed.reorder(names).position.reshape(-1)
                if (chosen.shape != state.position[0].shape
                        or not bool(torch.isfinite(chosen).all())
                        or bool(((chosen < local[0]) | (chosen > local[1])).any())):
                    raise ValueError("Terminal proposal violated local physical bounds")
                points.append(chosen.clone())
                break
            preferred = state.position[0] if hint is None else initial + alpha * (hint-initial)
            warm = torch.maximum(local[0], torch.minimum(local[1], preferred))
            result = planner.ik_solver.solve_pose(
                target, current_state=state,
                seed_config=warm[None, None].repeat(1, count, 1),
                return_seeds=count,
            )
            record["ik_steps"] = step
            record["ik_time"] += result.total_time
            record["ik_solve_time"] += result.solve_time
            valid = result.success.reshape(-1)
            candidates = result.solution.reshape(-1, len(names))
            valid = (valid & torch.isfinite(candidates).all(dim=-1)
                     & ((candidates >= local[0]) & (candidates <= local[1])).all(dim=-1))
            if not bool(valid.any()):
                return None
            candidates = candidates[valid]
            distances = torch.linalg.vector_norm(candidates - preferred, dim=-1)
            chosen = candidates[distances.argmin()].clone()
            points.append(chosen)
            state = JointState.from_position(chosen[None], names)
    finally:
        planner.update_joint_limits(position=bounds)
    path = torch.stack(points)
    seeds = resample_seed(path, planner.trajopt_solver.action_horizon,
                          planner.trajopt_solver.config.num_seeds)
    return goal_group(goals, index), path[-1], seeds, record


def plan_continuous_axis_seed(planner, goals, current, holds, use_implicit_goal):
    """Keep native endpoint, dynamics, contact and dense collision acceptance."""
    prepared = continuous_axis_seed(planner, goals, current, holds)
    if prepared is None:
        return None
    target, endpoint, seeds, record = prepared
    check_time_budget()
    finetune = planner.config.trajopt_finetune_attempts
    result = planner.trajopt_solver.solve_pose(
        target, current, seed_config=endpoint[None, None].repeat(1, seeds.shape[1], 1),
        seed_traj=seeds, use_implicit_goal=use_implicit_goal,
        finetune_attempts=1 if finetune is None else finetune,
    )
    remap_goalset_index(result, record["original_goal_index"], len(target.tool_frames))
    result.total_time += record["ik_time"]
    result.solve_time += record["ik_solve_time"]
    result.debug_info = {**(result.debug_info or {}), "continuous_axis_seed": record}
    return result
