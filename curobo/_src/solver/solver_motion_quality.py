# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Soft motion preferences; feasibility always comes from native solver metrics."""
from __future__ import annotations

from typing import Optional

import torch


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
