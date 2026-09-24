# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Bounded raw geometry evidence for a rejected terminal trajectory."""

import math

import torch

from curobo._src.collision.contact_mesh import MeshClearance
from curobo._src.state.state_joint import JointState


def _self_collision_sample(planner, state, peak, metrics):
    """Report loaded, enabled sphere pairs at the selected start and peak."""
    if metrics is None or not metrics.has_cost("self_collision"):
        return None
    config = metrics.get_cost("self_collision").config.self_collision_kin_config
    pairs = config.collision_pairs.long()
    params = planner.attachment_manager.kinematics_params
    names = {i: name for name, i in params.link_name_to_idx_map.items()}
    links = [names[i] for i in params.link_sphere_idx_map.tolist()]
    positions = state.position.reshape(-1, state.position.shape[-2], state.position.shape[-1])
    trajectory = peak["maximum_trajectory"]
    samples = []
    for step in dict.fromkeys((0, peak["maximum_step"])):
        selected = JointState.from_position(
            positions[trajectory, step : step + 1], joint_names=state.joint_names
        )
        with torch.no_grad():
            spheres = planner.compute_kinematics(selected).robot_spheres.reshape(-1, 4)
            a, b = pairs.unbind(-1)
            raw = spheres[a, 3] + spheres[b, 3] - torch.linalg.vector_norm(
                spheres[a, :3] - spheres[b, :3], dim=-1
            )
            padded = raw + config.sphere_padding[a] + config.sphere_padding[b]
            valid = (spheres[a, 3] > 0) & (spheres[b, 3] > 0) & (padded > 0)
            indices = torch.nonzero(valid).reshape(-1)
            order = torch.argsort(padded[indices], descending=True)[:8]
            rows = []
            for index in indices[order].tolist():
                first, second = pairs[index].tolist()
                rows.append({
                    "links": [links[first], links[second]],
                    "spheres": [first, second],
                    "raw_overlap_m": float(raw[index]),
                    "padded_overlap_m": float(padded[index]),
                })
        samples.append({"step": step, "collision_pair_count": int(valid.sum()), "pairs": rows})
    return {"trajectory": trajectory, "samples": samples, "acceptance_modified": False}


def terminal_failure_summary(planner, result):
    """Explain the first selected trajectory without changing acceptance or inputs.

    Raw mesh gaps include permitted endpoint contact and exclude cost weights.
    They are diagnostic geometry, not a replacement for contact-aware metrics.
    """
    debug = result.debug_info or {}
    summary = {
        "constraints": debug.get("selected_constraint_maxima", {}),
        "joint_bounds": (debug.get("selected_joint_bounds") or {}).get("terms", {}),
    }
    optimized = summary["constraints"].get("optimized", {})
    self_peak = optimized.get("self_collision", {})
    state = getattr(result, "js_solution", None)
    if state is not None and "maximum_step" in self_peak:
        metrics = planner.trajopt_solver.metrics_rollout.metrics_constraint_manager
        sample = _self_collision_sample(planner, state, self_peak, metrics)
        if sample is not None:
            summary["raw_self_collision_sample"] = sample
    peak = optimized.get("scene_collision", {})
    if "maximum_step" not in peak:
        return summary
    scene = planner.scene_collision_checker
    if scene is None or scene.data.meshes is None or state is None:
        return summary
    metrics = planner.trajopt_solver.metrics_rollout.metrics_constraint_manager
    if metrics is not None and metrics.has_cost("scene_collision"):
        cost = metrics.get_cost("scene_collision")
        if cost._contact is not None:
            full = state.position.reshape(-1, state.position.shape[-2], state.position.shape[-1])
            fk = planner.compute_kinematics(
                JointState.from_position(full, joint_names=state.joint_names)
            )
            spheres_full = fk.robot_spheres.reshape(full.shape[0], full.shape[1], -1, 4)
            with torch.no_grad():
                contact = cost._contact.cost(spheres_full)
            summary["terminal_contact_component"] = {
                "unweighted_maximum_m": float(contact.max()),
                "weighted_maximum": float((contact * cost._weight).max()),
                "activation_distance_m": cost.config.activation_distance.tolist(),
                "use_sweep": cost.config.use_sweep,
                "type": type(cost._contact).__name__,
            }
    meshes = scene.data.meshes
    indices = torch.nonzero(meshes.enable[0]).reshape(-1).to(torch.int32)
    if not indices.numel():
        return summary
    positions = state.position.reshape(-1, state.position.shape[-2], state.position.shape[-1])
    step = peak["maximum_step"]
    trajectory = peak["maximum_trajectory"]
    selected = JointState.from_position(
        positions[trajectory, step : step + 1], joint_names=state.joint_names
    )
    spheres = planner.compute_kinematics(selected).robot_spheres.reshape(-1, 4)
    with torch.no_grad():
        gaps = MeshClearance.apply(spheres, meshes, indices)
        gaps = gaps.masked_fill(spheres[:, 3:4] <= 0, float("inf"))
        values, offsets = torch.topk(gaps.reshape(-1), min(8, gaps.numel()), largest=False)
    params = planner.attachment_manager.kinematics_params
    link_names = {i: name for name, i in params.link_name_to_idx_map.items()}
    mapping = params.link_sphere_idx_map.tolist()
    rows = []
    for gap, offset in zip(values.tolist(), offsets.tolist()):
        if math.isinf(gap):
            continue
        sphere, mesh = divmod(offset, len(indices))
        rows.append(
            {
                "link": link_names[mapping[sphere]],
                "sphere": sphere,
                "obstacle": meshes.names[0][int(indices[mesh])],
                "raw_gap_m": gap,
            }
        )
    summary["raw_mesh_gap_sample"] = {
        "trajectory": trajectory,
        "step": step,
        "pairs": rows,
        "contact_allowance_applied": False,
    }
    return summary
