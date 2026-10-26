# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Soft motion preferences; feasibility always comes from native solver metrics."""
from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from curobo._src.robot.kinematics.kinematics import Kinematics
    from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg


def create_motion_quality_kinematics(config: KinematicsCfg) -> Kinematics:
    """Append middle-joint landmarks to independent seven-joint tool chains.

    This private scoring model preserves the native robot and collision models.
    Unsupported or shared chains retain the original tool-only preferences.
    For G2's seven-joint arms the fourth joint origin is the elbow link4 origin.
    """
    from curobo._src.robot.kinematics.kinematics import Kinematics

    params = config.kinematics_config
    parents = params.link_map.cpu().tolist()
    joints = params.joint_map.cpu().tolist()
    types = params.joint_map_type.cpu().tolist()
    tools = params.tool_frame_map.cpu().tolist()
    midpoint, masks, chains = [], [], []
    for tool in tools:
        chain = []
        while tool:
            if joints[tool] >= 0 and types[tool] >= 0:
                chain.append(tool)
            tool = parents[tool]
        chain.reverse()
        active = {joints[index] for index in chain}
        if len(chain) != 7 or len(active) != 7 or any(active & previous for previous in chains):
            return Kinematics(config, compute_spheres=False)
        chains.append(active)
        midpoint.append(chain[3])
        masks.append({joints[index] for index in chain[:4]})
    if not midpoint:
        return Kinematics(config, compute_spheres=False)
    names = {index: name for name, index in params.link_name_to_idx_map.items()}
    extra = [names[index] for index in midpoint]
    extended = params.clone()
    extended.tool_frames = list(params.tool_frames) + extra
    extended.tool_frame_map = torch.cat([
        params.tool_frame_map, params.tool_frame_map.new_tensor(midpoint),
    ])
    affects = torch.zeros((params.num_dof, len(extra)), dtype=torch.bool,
                          device=params.joint_affects_endeffector.device)
    for index, mask in enumerate(masks):
        affects[list(mask), index] = True
    extended.joint_affects_endeffector = torch.cat([
        params.joint_affects_endeffector.reshape(params.num_dof, len(tools)),
        affects.to(params.joint_affects_endeffector.dtype),
    ], dim=1).flatten()
    return Kinematics(replace(config, tool_frames=extended.tool_frames,
                              kinematics_config=extended), compute_spheres=False)


def height_overshoot(
    height: torch.Tensor, lengths: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return rise beyond the higher actual endpoint for heights [B,S,H,L]."""
    horizon = height.shape[2]
    if lengths is None:
        return (height.amax(dim=2) - torch.maximum(height[:, :, 0],
                                                 height[:, :, -1])).clamp_min(0.)
    lengths = lengths.reshape(*height.shape[:2]).long().clamp(1, horizon)
    active = torch.arange(horizon, device=height.device)[None, None, :] < lengths[..., None]
    peak = torch.where(active[..., None], height, -torch.inf).amax(dim=2)
    last = torch.gather(height, 2, (lengths - 1)[..., None, None].expand(
        -1, -1, 1, height.shape[-1])).squeeze(2)
    return (peak - torch.maximum(height[:, :, 0], last)).clamp_min(0.)


def posture_motion_penalty(
    height: torch.Tensor, lengths: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Penalize excess landmark rise only above a five-centimetre deadband."""
    extra = height_overshoot(height, lengths)
    return (4. * (extra - .05).clamp_min(0.) / .05).amax(dim=-1)


def rank_balanced_motion_candidates(
    position: torch.Tensor, lengths: torch.Tensor,
    success: torch.Tensor, quality: torch.Tensor,
    posture_height: Optional[torch.Tensor] = None,
    tcp_baseline_quality: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Rank complete native-validated paths [B,S,H,L,3] without large arm exchanges.

    Each arm's reference is its shortest path among feasible pool members. These
    per-arm minima are comparison bounds, not a combined feasible trajectory.
    Screen TCP increases exceeding both 0.10m and 50% of that reference and,
    when supplied, elbow overshoot exceeding the feasible minimum by 0.05m.
    These are large-motion preferences, not the separate five-percent contract.
    If a pre-posture TCP score is supplied, preserve its complete feasible
    incumbent: elbow preferences may add at most max(0.02m, 5%) per-arm travel.
    This pool-local protection does not compare against an unavailable baseline.
    If no candidate meets every bound, minimize worst normalized overrun while
    preserving native success and the old quality order for ties.
    """
    horizon = position.shape[2]
    steps = torch.arange(horizon - 1, device=position.device)[None, None, :]
    active = steps < lengths.reshape(*success.shape, 1).clamp(1, horizon) - 1
    distances = torch.linalg.vector_norm(position.diff(dim=2), dim=-1)
    path = torch.where(active[..., None], distances, 0.).sum(dim=2)
    finite = torch.isfinite(path).all(dim=-1)
    usable = success & finite
    reference = torch.where(usable[..., None], path, torch.inf).amin(dim=1)
    has_reference = usable.any(dim=1)
    reference = torch.where(has_reference[:, None], reference, 0.)
    allowance = torch.maximum(.5 * reference, torch.full_like(reference, .10))
    overrun = ((path - reference[:, None] - allowance[:, None])
               / allowance[:, None]).clamp_min(0.).amax(dim=-1)
    eligible = usable
    protection = {}
    if tcp_baseline_quality is not None:
        # Preserve a complete feasible TCP-ranked incumbent before considering
        # elbow improvements. Never trade a large hand detour for lower elbows.
        baseline_order = torch.argsort(tcp_baseline_quality, dim=1, stable=True)
        risk = torch.argsort(torch.gather(overrun, 1, baseline_order), dim=1, stable=True)
        baseline_order = torch.gather(baseline_order, 1, risk)
        feasible = torch.argsort(torch.gather(success, 1, baseline_order).to(torch.int32),
                                dim=1, descending=True, stable=True)
        baseline = torch.gather(baseline_order, 1, feasible)[:, :1]
        protected = torch.gather(path, 1, baseline[..., None].expand(-1, -1, path.shape[-1]))
        tolerance = torch.maximum(.05 * protected, torch.full_like(protected, .02))
        eligible = usable & (path <= protected + tolerance + 1e-6).all(dim=-1)
        protection = dict(tcp_protected_reference_m=protected.squeeze(1),
                          tcp_protection_eligible=eligible)
    posture = {}
    if posture_height is not None:
        extra = height_overshoot(posture_height, lengths)
        elbow_reference = torch.where(eligible[..., None], extra, torch.inf).amin(dim=1)
        elbow_reference = torch.where(eligible.any(dim=1)[:, None], elbow_reference, 0.)
        elbow_overrun = ((extra - elbow_reference[:, None] - .05) / .05).clamp_min(0.)
        overrun = torch.maximum(overrun, elbow_overrun.amax(dim=-1))
        posture = dict(elbow_overshoot_m=extra, elbow_reference_m=elbow_reference)
    overrun = torch.where(torch.isfinite(overrun), overrun, torch.inf)
    overrun = torch.where(has_reference[:, None], overrun, 0.)
    overrun = torch.where(overrun <= 1e-6, 0., overrun)
    order = torch.argsort(quality, dim=1, stable=True)
    risk_order = torch.argsort(torch.gather(overrun, 1, order), dim=1, stable=True)
    order = torch.gather(order, 1, risk_order)
    if tcp_baseline_quality is not None:
        preferred = torch.argsort(torch.gather(eligible, 1, order).to(torch.int32),
                                  dim=1, descending=True, stable=True)
        order = torch.gather(order, 1, preferred)
    feasible_order = torch.argsort(torch.gather(success, 1, order).to(torch.int32),
                                  dim=1, descending=True, stable=True)
    return torch.gather(order, 1, feasible_order), dict(
        tcp_path_m=path, reference_m=reference, allowance_m=allowance,
        normalized_overrun=overrun, within_limits=eligible & (overrun <= 1e-6),
        **posture, **protection,
    )


def combine_independent_tool_candidates(
    solutions: torch.Tensor,
    success: torch.Tensor,
    tool_cost: torch.Tensor,
    joint_masks: torch.Tensor,
    count: int,
) -> Optional[torch.Tensor]:
    """Propose endpoints for disjoint chains; callers must revalidate collisions.

    Shapes: solutions [batch, pool, dof], tool_cost [batch, pool, tools],
    joint_masks [tools, dof]. Shared actuated joints disable recombination.
    Success may be per-tool [batch, pool, tools]: one unconverged chain need
    not discard another chain's converged portion of the same proposal.
    These are proposals, never an assertion of whole-robot feasibility.
    """
    if bool((joint_masks.sum(dim=0) > 1).any()):
        return None
    combined = solutions[:, :1].expand(-1, count, -1).clone()
    for tool in range(joint_masks.shape[0]):
        valid = success[:, :, tool] if success.ndim == 3 else success
        order = torch.argsort(torch.where(valid, tool_cost[:, :, tool], torch.inf), dim=1, stable=True)
        index = order[:, :count]
        part = torch.gather(solutions, 1, index[..., None].expand(-1, -1, solutions.shape[-1]))
        combined = torch.where(joint_masks[tool][None, None], part, combined)
    return combined


def motion_quality_terms(
    position: torch.Tensor, quaternion: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Return per-tool excess translation and rotation for [B,S,H,L,3/4] paths."""
    lengths = torch.linalg.vector_norm(position.diff(dim=2), dim=-1).sum(dim=2)
    chords = torch.linalg.vector_norm(position[:, :, -1] - position[:, :, 0], dim=-1)
    excess = torch.clamp(lengths - chords, min=0.0) / (0.05 + chords)
    if quaternion is not None:
        def angle(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
            vector = (first[..., :1] * second[..., 1:] - second[..., :1] * first[..., 1:]
                      - torch.linalg.cross(first[..., 1:], second[..., 1:], dim=-1))
            scalar = (first * second).sum(dim=-1).abs()
            return 2 * torch.atan2(torch.linalg.vector_norm(vector, dim=-1), scalar)
        travel = angle(quaternion[:, :, :-1], quaternion[:, :, 1:]).sum(dim=2)
        chord = angle(quaternion[:, :, 0], quaternion[:, :, -1])
        excess = excess + 4.0 * (travel - chord).clamp_min(0.0) / (0.25 + chord)
    return excess


def motion_quality_cost(
    position: torch.Tensor, joints: torch.Tensor, quaternion: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Score [batch, seeds, steps, tools, 3] paths and matching joint paths.

    Penalize each tool's excess travel relative to its endpoint chord. The
    worst tool dominates, so one arm cannot hide the other arm's detour.
    A nonzero length scale also handles rotation-only and stationary goals.
    """
    excess = motion_quality_terms(position, quaternion)
    joint_travel = torch.linalg.vector_norm(joints.diff(dim=2), dim=-1).sum(dim=2)
    variation = joints.diff(dim=2).abs().sum(dim=2)
    backtrack = (variation - (joints[:, :, -1] - joints[:, :, 0]).abs()).clamp_min(0.0)
    return excess.amax(dim=-1) + 0.25 * excess.sum(dim=-1) + 0.02 * joint_travel + 0.1 * backtrack.amax(dim=-1)


def select_motion_candidates(
    solutions: torch.Tensor,
    success: torch.Tensor,
    cost: torch.Tensor,
    count: int,
    old_count: int = 0,
    reserve_old: int = 0,
) -> torch.Tensor:
    """Choose diverse feasible endpoints [batch, count], reserving old options.

    Inputs have shapes [batch, pool, dof] and [batch, pool]. Failures never
    displace feasible solutions. If unique successes run out, repeats are
    allowed before admitting failures. No modulo wrapping of bounded joints.
    """
    batch, pool, _ = solutions.shape
    available = torch.ones_like(success)
    selected = []
    for slot in range(count):
        eligible = available & success
        if slot < reserve_old and old_count:
            old = torch.arange(pool, device=solutions.device)[None] < old_count
            preferred = eligible & old
            eligible = torch.where(preferred.any(dim=1, keepdim=True), preferred, eligible)
        eligible = torch.where(eligible.any(dim=1, keepdim=True), eligible, success)
        eligible = torch.where(eligible.any(dim=1, keepdim=True), eligible, available)
        eligible = torch.where(eligible.any(dim=1, keepdim=True), eligible, torch.ones_like(eligible))
        index = torch.where(eligible, cost, torch.inf).argmin(dim=1)
        selected.append(index)
        picked = solutions[torch.arange(batch, device=solutions.device), index]
        duplicate = (solutions - picked[:, None]).abs().amax(dim=-1) < 0.01
        available &= ~duplicate
    return torch.stack(selected, dim=1)
