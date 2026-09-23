# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Bounded failed-trajectory diagnostics; never used for planning acceptance."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from curobo._src.cost.cost_cspace_type import CSpaceCostType

if TYPE_CHECKING:
    from curobo._src.cost.cost_cspace_cfg import CSpaceCostCfg
    from curobo._src.state.state_joint import JointState


def joint_bound_diagnostics(state: JointState, config: CSpaceCostCfg) -> dict:
    """Report actual selected states against the metric component's own limits.

    Costs combine several physical quantities, so a cspace maximum alone does
    not identify a joint-position violation. These raw-SI diagnostics preserve
    that distinction and cap individual violation records at eight per term.
    """
    names = config.joint_limits.joint_names
    position = state.position.detach().cpu().reshape(-1, state.position.shape[-2], len(names))
    limits = config.joint_limits.position.detach().cpu()
    result = {
        "source": "optimized_joint_state",
        "joint_names": list(names),
        "position_start": position[0, 0].tolist(),
        "position_end": position[0, -1].tolist(),
        "position_limits": limits.tolist(),
        "reported_endpoint_trajectory_index": 0,
        "trajectory_count": len(position),
        "terms": {},
        "effort": "Not present in JointState; consult torque metrics separately",
    }
    weights = config.weight.detach().cpu().reshape(-1)
    activation = config.activation_distance.detach().cpu().reshape(-1)
    for index, name in enumerate(("position", "velocity", "acceleration", "jerk")):
        value = getattr(state, name)
        if index and config.cost_type == CSpaceCostType.POSITION:
            result["terms"][name] = {
                "available": value is not None,
                "evaluated_by_cost": False,
            }
            continue
        if value is None:
            result["terms"][name] = {"available": False}
            continue
        values = value.detach().cpu().reshape_as(position)
        raw_bounds = getattr(config.joint_limits, name).detach().cpu()
        margin = activation[index] * (raw_bounds[1] - raw_bounds[0])
        lower, upper = raw_bounds[0] + margin, raw_bounds[1] - margin
        excess = torch.maximum(lower - values, values - upper).clamp_min(0)
        flat = excess.reshape(-1)
        positive = torch.nonzero(flat > 0).reshape(-1)
        count = min(8, len(positive))
        indices = positive[torch.topk(flat[positive], count).indices] if count else positive
        rows = []
        horizon, dof = position.shape[-2:]
        for offset in indices.tolist():
            trajectory, remainder = divmod(offset, horizon * dof)
            step, joint = divmod(remainder, dof)
            rows.append(
                {
                    "trajectory_index": trajectory,
                    "step": step,
                    "joint": names[joint],
                    "value": float(values[trajectory, step, joint]),
                    "lower": float(lower[joint]),
                    "upper": float(upper[joint]),
                    "excess": float(excess[trajectory, step, joint]),
                }
            )
        result["terms"][name] = {
            "available": True,
            "weight": float(weights[index]),
            "activation_distance": float(activation[index]),
            "maximum_excess": float(excess.max()),
            "start_maximum_excess": float(excess[:, 0].max()),
            "end_maximum_excess": float(excess[:, -1].max()),
            "violation_count": len(positive),
            "largest_violations": rows,
        }
    return result
