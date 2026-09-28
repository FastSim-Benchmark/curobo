# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Bounded raw geometry evidence for rejected trajectories."""

import math

import torch

from curobo._src.collision.contact_approach import ContactApproach
from curobo._src.collision.contact_departure_set import ContactDepartureSet
from curobo._src.collision.contact_mesh import MeshClearance
from curobo._src.collision.contact_separation import ContactSeparation
from curobo._src.state.state_joint import JointState


def _approach_gap_summary(
    contact: ContactApproach, spheres: torch.Tensor, trajectory: int
) -> dict:
    """Report bounded raw gap evidence without changing contact acceptance.

    Rebound measures motion away from the goal after entering the approach band.
    It can be positive even when every measured sphere gap is nonnegative.
    Values are geometric distances before the declaration's numerical tolerance.
    """
    separation = contact.separation
    selected = spheres[trajectory : trajectory + 1].index_select(-2, separation.indices)
    dense = separation.interpolate_samples(selected)
    gap = separation.clearance(dense)[0]
    reverse_gap = gap.flip(0)
    best = torch.cummax(
        reverse_gap.clamp_max(separation.declaration.release_clearance), dim=0
    ).values
    rebound = (best - reverse_gap).clamp_min(0).flip(0)
    below_goal = (separation.initial_clearance - gap).clamp_min(0)
    ranking = (rebound + below_goal).amax(0)
    rows = []
    for index in torch.argsort(ranking, descending=True)[:8].tolist():
        peak = int((rebound[:, index] + below_goal[:, index]).argmax())
        sample_indices = sorted(
            {0, len(gap) - 1, *range(max(0, peak - 1), min(len(gap), peak + 2))}
        )
        rows.append({
            "sphere": int(separation.indices[index]),
            "goal_gap_m": float(separation.initial_clearance[index]),
            "minimum_gap_m": float(gap[:, index].min()),
            "maximum_approach_rebound_m": float(rebound[:, index].max()),
            "maximum_below_goal_gap_m": float(below_goal[:, index].max()),
            "samples": [{"dense_step": step, "gap_m": float(gap[step, index])}
                        for step in sample_indices],
        })
    return {
        "obstacle": contact.declaration.obstacle_name,
        "trajectory": trajectory,
        "dense_sample_count": len(gap),
        "sphere_count": len(separation.indices),
        "approach_clearance_m": separation.declaration.release_clearance,
        "numerical_tolerance_m": separation.declaration.numerical_tolerance,
        "goal_sphere_mismatch_m": float(
            (selected[:, -1] - separation.initial_spheres).abs().max()
        ),
        "initial_clearance_shortfall_m": float(
            (separation.declaration.release_clearance - gap[0]).clamp_min(0).max()
        ),
        "spheres": rows,
        "acceptance_modified": False,
    }


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


def _departure_gap_summary(contact, spheres, trajectory):
    """Separate support penetration from nonmonotonic departure, without exemptions."""
    selected = spheres[trajectory : trajectory + 1].index_select(-2, contact.indices)
    gap = contact.clearance(contact.interpolate_samples(selected))[0]
    best = torch.cummax(gap.clamp_max(contact.declaration.release_clearance), dim=0).values
    rebound = (best - gap).clamp_min(0)
    below_start = (contact.initial_clearance - gap).clamp_min(0)
    ranking = (rebound + below_start).amax(0)
    rows = []
    for index in torch.argsort(ranking, descending=True)[:8].tolist():
        peak = int((rebound[:, index] + below_start[:, index]).argmax())
        steps = sorted({0, len(gap) - 1, *range(max(0, peak - 1), min(len(gap), peak + 2))})
        rows.append({
            "sphere": int(contact.indices[index]),
            "initial_gap_m": float(contact.initial_clearance[index]),
            "minimum_gap_m": float(gap[:, index].min()),
            "maximum_departure_rebound_m": float(rebound[:, index].max()),
            "maximum_below_initial_gap_m": float(below_start[:, index].max()),
            "samples": [{"dense_step": step, "gap_m": float(gap[step, index])} for step in steps],
        })
    return {
        "obstacle": contact.declaration.obstacle_name,
        "trajectory": trajectory,
        "dense_sample_count": len(gap),
        "sphere_count": len(contact.indices),
        "release_clearance_m": contact.declaration.release_clearance,
        "numerical_tolerance_m": contact.declaration.numerical_tolerance,
        "start_sphere_mismatch_m": float((selected[:, 0] - contact.initial_spheres).abs().max()),
        "spheres": rows,
        "acceptance_modified": False,
    }


def terminal_failure_summary(planner, result, *, trajectory_kind="optimized"):
    """Explain the selected optimized or dense grid without changing acceptance.

    Raw mesh gaps include permitted endpoint contact and exclude cost weights.
    They are diagnostic geometry, not a replacement for contact-aware metrics.
    Dense-grid peaks must index the interpolated state and its metrics rollout.
    """
    if trajectory_kind not in {"optimized", "interpolated"}:
        raise ValueError("trajectory_kind must be optimized or interpolated")
    debug = result.debug_info or {}
    summary = {
        "constraints": debug.get("selected_constraint_maxima", {}),
        "joint_bounds": (debug.get("selected_joint_bounds") or {}).get("terms", {}),
    }
    selected_metrics = summary["constraints"].get(trajectory_kind, {})
    self_peak = selected_metrics.get("self_collision", {})
    state = getattr(result, "js_solution" if trajectory_kind == "optimized"
                    else "interpolated_trajectory", None)
    if selected_metrics and state is None:
        summary.update(trajectory_kind=trajectory_kind, trajectory_state_available=False)
        return summary
    metrics = None
    if state is not None and selected_metrics:
        summary["trajectory_kind"] = trajectory_kind
        rollout = (planner.trajopt_solver.metrics_rollout if trajectory_kind == "optimized"
                   else planner.trajopt_solver.additional_metrics_rollouts["interpolated_rollout"])
        metrics = rollout.metrics_constraint_manager
    if state is not None and "maximum_step" in self_peak:
        sample = _self_collision_sample(planner, state, self_peak, metrics)
        if sample is not None:
            summary["raw_self_collision_sample"] = sample
    peak = selected_metrics.get("scene_collision", {})
    if "maximum_step" not in peak:
        return summary
    scene = planner.scene_collision_checker
    if scene is None or scene.data.meshes is None or state is None:
        return summary
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
            if isinstance(cost._contact, ContactApproach):
                with torch.no_grad():
                    summary["terminal_approach_gaps"] = _approach_gap_summary(
                        cost._contact, spheres_full, peak["maximum_trajectory"]
                    )
            if isinstance(cost._contact, ContactDepartureSet):
                departures = cost._contact.contacts
            elif isinstance(cost._contact, ContactSeparation):
                departures = (cost._contact,)
            else:
                departures = ()
            if departures:
                with torch.no_grad():
                    summary["departure_gaps"] = [
                        _departure_gap_summary(item, spheres_full, peak["maximum_trajectory"])
                        for item in departures[:8]
                    ]
                summary["departure_contact_count"] = len(departures)
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
