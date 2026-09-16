# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""In-place propagation of motion limits to resident optimizer and validation buffers."""

from __future__ import annotations

import torch


def update_joint_limits(planner, *, position=None, velocity=None, acceleration=None, jerk=None):
    """Apply a complete validated update without reallocating planner components."""
    limits = planner.kinematics.get_joint_limits()
    changes = {}
    for name, value in (
        ("position", position),
        ("velocity", velocity),
        ("acceleration", acceleration),
        ("jerk", jerk),
    ):
        if value is None:
            continue
        current = getattr(limits, name)
        candidate = torch.as_tensor(value, device=current.device, dtype=current.dtype)
        if candidate.shape != current.shape or not bool(torch.isfinite(candidate).all()):
            raise ValueError(f"{name} limits must be finite with shape {tuple(current.shape)}")
        if bool((candidate[0] > candidate[1]).any()):
            raise ValueError(f"{name} lower limits exceed upper limits")
        if name != "position" and bool(((candidate[0] >= 0) | (candidate[1] <= 0)).any()):
            raise ValueError(f"{name} limits must bracket zero strictly")
        changes[name] = candidate.clone()
    if not changes:
        return
    for name, value in changes.items():
        getattr(limits, name).copy_(value)
    # C-space costs own copies; kinematics and transitions share the robot limits.
    rollouts = []
    for solver in (planner.ik_solver, planner.trajopt_solver):
        core = solver.core
        rollouts.extend(core.get_all_rollout_instances())
        rollouts.extend(core.additional_metrics_rollouts.values())
        sampler = core.seed_manager.action_sample_generator
        sampler.update_bounds(
            core.auxiliary_rollout.action_bound_lows, core.auxiliary_rollout.action_bound_highs
        )
        for optimizer in core.optimizers:
            optimizer_core = optimizer._core
            optimizer_core._bounds.refresh(
                optimizer_core.rollout_fn.action_bound_lows,
                optimizer_core.rollout_fn.action_bound_highs,
                optimizer_core.action_horizon,
            )
    seed = planner.ik_solver.seed_ik_solver
    if seed is not None:
        margin = (limits.position[1] - limits.position[0]) * seed.config.joint_limit_margin
        seed.action_min.copy_(limits.position[0] + margin)
        seed.action_max.copy_(limits.position[1] - margin)
        seed.action_step_max.copy_(
            seed.config.max_step_size * (seed.action_max - seed.action_min).abs()
        )
        seed.act_sample_gen.update_bounds(seed.action_min, seed.action_max)
    if planner.graph_planner is not None:
        graph = planner.graph_planner
        rollouts.append(graph.auxiliary_rollout)
        graph.sampling_strategy.action_sample_generator.update_bounds(
            graph.action_bound_lows, graph.action_bound_highs
        )
    for rollout in rollouts:
        for cost in rollout.get_cost_component_by_name("cspace"):
            for name, value in changes.items():
                getattr(cost.config.joint_limits, name).copy_(value)
    planner.reset_seed()
