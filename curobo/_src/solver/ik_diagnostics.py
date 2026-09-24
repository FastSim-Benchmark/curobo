# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Bounded diagnostics for failed IK batches; no effect on acceptance."""

from __future__ import annotations

import torch

from curobo._src.solver.trajopt_diagnostics import joint_bound_diagnostics
from curobo._src.state.state_joint import JointState


def self_collision_pair_diagnostics(state, selected, seed_count, collision, kinematics):
    """Report bounded enabled pairs from the exact FK state used for IK metrics.

    No FK is repeated and no attachment is reconstructed. Distances are measured
    from the captured floating-point sphere values in the kinematic base frame;
    this is diagnostic arithmetic, not a replacement for native acceptance.
    """
    spheres = state.robot_spheres
    if spheres is None:
        return {"available": False, "reason": "metrics_state_has_no_robot_spheres"}
    count = collision.num_spheres
    if spheres.shape[-2:] != (count, 4):
        raise ValueError("IK diagnostic sphere shape does not match the active collision model")
    states = spheres.detach().reshape(seed_count, -1, count, 4)
    if states.shape[1] != 1:
        raise ValueError("IK endpoint diagnostics require a single FK horizon sample")
    indices = selected.detach().reshape(-1)[:4]
    states = states[indices].to(device="cpu", dtype=torch.float64)
    pairs = collision.collision_pairs.detach().to(device="cpu", dtype=torch.long)
    padding = collision.sphere_padding.detach().to(device="cpu", dtype=torch.float64)
    mapping = kinematics.link_sphere_idx_map.detach().cpu().tolist()
    names = {index: name for name, index in kinematics.link_name_to_idx_map.items()}
    positions = state.joint_state.position.detach()
    positions = positions.reshape(seed_count, -1, positions.shape[-1])
    positions = positions[indices].cpu()
    if len(kinematics.joint_names) != positions.shape[-1]:
        raise ValueError("IK diagnostic joint names do not match the metrics coordinate layout")
    result = {
        "available": True,
        "source": "same_ik_metrics_robot_spheres_and_enabled_collision_pairs",
        "frame": "kinematic_base",
        "sphere_count": count,
        "enabled_pair_count": len(pairs),
        "selected_seed_count": selected.numel(),
        "reported_seed_count": len(indices),
        "seed_sample_truncated": selected.numel() > len(indices),
        "maximum_pairs_per_seed": 8,
        "joint_names": list(kinematics.joint_names),
        "seeds": [],
    }
    first, second = pairs.T
    for row, seed_index in enumerate(indices.cpu().tolist()):
        sample = states[row, 0]
        effective_radii = sample[:, 3] + padding
        # The native kernel checks validity after applying per-sphere padding.
        valid = (effective_radii[first] >= 0) & (effective_radii[second] >= 0)
        distance_squared = ((sample[first, :3]-sample[second, :3])**2).sum(dim=-1)
        radius_sum = effective_radii[first]+effective_radii[second]
        squared_overlap = radius_sum**2-distance_squared
        positive = torch.nonzero(valid & (squared_overlap > 0), as_tuple=False).flatten()
        order = torch.argsort(squared_overlap[positive], descending=True, stable=True)
        top = positive[order[:8]].tolist()
        details = []
        for pair_index in top:
            sphere_indices = pairs[pair_index].tolist()
            details.append({
                "enabled_pair_index": pair_index,
                "spheres": [{
                    "native_index": i,
                    "link": names[mapping[i]],
                    "center_m": sample[i, :3].tolist(),
                    "radius_m": float(sample[i, 3]),
                    "padding_m": float(padding[i]),
                    "effective_radius_m": float(effective_radii[i]),
                } for i in sphere_indices],
                "center_distance_m": float(distance_squared[pair_index].sqrt()),
                "overlap_m": float(radius_sum[pair_index]-distance_squared[pair_index].sqrt()),
                "squared_overlap_m2": float(squared_overlap[pair_index]),
            })
        result["seeds"].append({
            "seed_index": seed_index,
            "joint_positions": positions[row, 0].tolist(),
            "positive_pair_count": len(positive),
            "pair_sample_truncated": len(positive) > len(details),
            "maximum_squared_overlap_m2": float(squared_overlap[top[0]]) if top else 0.0,
            "pairs": details,
        })
    return result


def ik_failure_diagnostics(
    metrics, feasible, converged, success, selected, config,
    self_collision_config=None, kinematics_config=None,
):
    """Separate convergence, collision and bounds for all and selected IK seeds."""
    indices = selected.reshape(-1)
    seed_count = success.numel()
    group = metrics.costs_and_constraints.constraints
    constraints = {}
    for name, value in zip(group.names, group.values):
        costs = value.detach().reshape(seed_count, -1)
        constraints[name] = {
            "positive_cost_seed_count": int((costs > 0).any(dim=1).sum()),
            "maximum": float(costs.max()),
            "selected_maximum": float(costs[indices].max()),
        }
    result = {
        "seed_count": seed_count,
        "feasible_seed_count": int(feasible.sum()),
        "converged_seed_count": int(converged.sum()),
        "successful_seed_count": int(success.sum()),
        "selected_seed_indices": indices[:8].tolist(),
        "selected_seed_count": indices.numel(),
        "constraints": constraints,
    }
    rejected = torch.nonzero(
        converged.reshape(-1) & ~success.reshape(-1), as_tuple=False
    ).flatten()
    sampled = rejected[:8]
    result["converged_rejected_seeds"] = {
        "count": rejected.numel(),
        "sample_truncated": rejected.numel() > sampled.numel(),
        "seeds": [
            {
                "seed_index": int(index),
                "feasible": bool(feasible.reshape(-1)[index]),
                "constraint_maxima": {
                    name: float(value.detach().reshape(seed_count, -1)[index].max())
                    for name, value in zip(group.names, group.values)
                },
            }
            for index in sampled.tolist()
        ],
    }
    if config is not None:
        all_positions = metrics.state.joint_state.position
        dof = all_positions.shape[-1]
        positions = all_positions.reshape(seed_count, -1, dof)[indices]
        state = JointState.from_position(positions, joint_names=config.joint_limits.joint_names)
        bounds = joint_bound_diagnostics(state, config)
        bounds["source"] = "ik_endpoint_joint_state"
        result["selected_joint_bounds"] = bounds
    if self_collision_config is not None and kinematics_config is not None:
        result["selected_self_collision_pairs"] = self_collision_pair_diagnostics(
            metrics.state, selected, seed_count, self_collision_config, kinematics_config
        )
    return result
