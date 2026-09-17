# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Request-scoped endpoint contact capture and collision rollout installation."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from typing import TYPE_CHECKING, Iterator

import torch

from curobo._src.collision.contact_approach import ContactApproach, GoalContact
from curobo._src.collision.contact_departure_set import ContactDepartureSet
from curobo._src.collision.contact_mesh import MeshClearance
from curobo._src.collision.contact_separation import ContactSeparation, StartContact
from curobo._src.collision.contact_transfer import ContactTransfer
from curobo._src.state.state_joint import JointState
from curobo._src.types.pose import Pose
from curobo._src.util.logging import log_and_raise

if TYPE_CHECKING:
    from curobo._src.cost.cost_scene_collision import SceneCollisionCost
    from curobo._src.motion.motion_planner import MotionPlanner
    from curobo._src.solver.solver_ik import IKSolver
    from curobo._src.solver.solver_trajopt import TrajOptSolver
    from curobo._src.types.tool_pose import GoalToolPose


def validate_boundary_mode(value: str) -> None:
    """Keep the public policy finite; an unknown value never weakens checks."""
    if type(value) is not str or value not in {"none", "start", "end", "both"}:
        log_and_raise("allow_boundary_collision must be none, start, end, or both")


def scene_costs(solver: IKSolver | TrajOptSolver) -> list[SceneCollisionCost]:
    """Find cost, constraint, and interpolated-metric components once by identity."""
    costs = []
    for rollout in [
        *solver.core.get_all_rollout_instances(),
        *solver.core.additional_metrics_rollouts.values(),
    ]:
        for cost in rollout.get_cost_component_by_name("scene_collision"):
            if cost is not None and cost not in costs:
                costs.append(cost)
    return costs


def contact_goal_seed(
    planner: MotionPlanner, goals: GoalToolPose, current: JointState
) -> JointState | None:
    """Obtain kinematic endpoint evidence, then restore full scene checks.

    This candidate is never executed or accepted as a trajectory. The subsequent
    contact-aware goal IK, trajectory optimization, and metrics must all succeed.
    Self collision and joint limits remain active while constructing this seed.
    """
    costs = scene_costs(planner.ik_solver)
    enabled = [cost for cost in costs if cost.enabled]
    weights = [cost._weight.clone() for cost in enabled]
    planner.ik_solver.core.invalidate_parameter_graphs()
    try:
        for cost in enabled:
            cost.disable_cost()
        result = planner.ik_solver.solve_pose(goals, return_seeds=1, current_state=current)
    finally:
        for cost, weight in zip(enabled, weights):
            cost.enable_cost()
            cost._weight.copy_(weight)
        planner.ik_solver.core.invalidate_parameter_graphs()
    if not bool(result.success.any()):
        return None
    position = result.solution[result.success][0].reshape(1, -1)
    return JointState.from_position(position, joint_names=planner.joint_names)


def capture_contact(
    planner: MotionPlanner,
    state: JointState | None,
    *,
    terminal: bool,
    max_initial_penetration: float = 0.002,
    contact_links: tuple[str, ...] | None = None,
) -> StartContact | GoalContact | tuple[StartContact, ...] | None:
    """Capture bounded static support contacts in the current planning frame."""
    scene = planner.scene_collision_checker
    if state is None or scene is None:
        return None
    if state.shape[0] != 1 or scene.data.num_envs != 1:
        log_and_raise("Boundary contact currently requires one planning problem and environment")
    spheres = planner.compute_kinematics(state).robot_spheres.reshape(-1, 4)
    eligible = torch.ones(len(spheres), device=spheres.device, dtype=torch.bool)
    if contact_links is not None:
        if not contact_links or len(set(contact_links)) != len(contact_links):
            log_and_raise("contact_links must name distinct robot links")
        eligible.zero_()
        for link in contact_links:
            ids = planner.attachment_manager.kinematics_params.get_sphere_index_from_link_name(
                link
            )
            eligible[ids] = True
    cuboids = scene.data.cuboids
    contacts = []
    # Freeze the small enable mask once. Reading one CUDA scalar per reserved
    # slot synchronizes thousands of times even when almost all slots are empty.
    enabled_cuboids = [] if cuboids is None else cuboids.enable[0].tolist()
    for index, name in enumerate(cuboids.names[0] if cuboids is not None else ()):
        if not bool(enabled_cuboids[index]):
            continue
        inverse = cuboids.inv_pose[0, index, :7]
        rotation = Pose(position=inverse[:3], quaternion=inverse[3:]).get_rotation_matrix()[0]
        local = (
            spheres[:, 0:1] * rotation[:, 0]
            + spheres[:, 1:2] * rotation[:, 1]
            + spheres[:, 2:3] * rotation[:, 2]
            + inverse[:3]
        )
        outside = local.abs() - cuboids.dims[0, index, :3] * 0.5
        gap = outside.clamp_min(0).norm(dim=-1) + outside.amax(-1).clamp_max(0) - spheres[:, 3]
        indices = torch.nonzero(eligible & (spheres[:, 3] > 0) & (gap <= 1.0e-5)).reshape(-1)
        if indices.numel() == 0:
            continue
        selected = tuple(indices.tolist())
        geometry = tuple(tuple(sphere) for sphere in spheres[indices].tolist())
        declaration = (
            GoalContact(name, selected, geometry)
            if terminal
            else StartContact(
                name, selected, geometry, max_initial_penetration=max_initial_penetration
            )
        )
        # Validate depth, identity, and enabled geometry before installing any exemption.
        captured = (
            declaration.as_departure() if isinstance(declaration, GoalContact) else declaration
        )
        ContactSeparation(captured, scene, len(spheres))
        contacts.append(declaration)
    meshes = scene.data.meshes
    if not terminal and meshes is not None:
        mesh_indices = torch.nonzero(meshes.enable[0]).reshape(-1).to(torch.int32)
        if mesh_indices.numel():
            gaps = MeshClearance.apply(spheres, meshes, mesh_indices)
            touching = eligible[:, None] & (spheres[:, 3:4] > 0) & (gaps <= 1.0e-5)
            for column in torch.nonzero(touching.any(0)).reshape(-1).tolist():
                index = int(mesh_indices[column])
                indices = torch.nonzero(touching[:, column]).reshape(-1)
                declaration = StartContact(
                    meshes.names[0][index],
                    tuple(indices.tolist()),
                    tuple(tuple(row) for row in spheres[indices].tolist()),
                    max_initial_penetration=max_initial_penetration,
                )
                ContactSeparation(declaration, scene, len(spheres))
                contacts.append(declaration)
    if len(contacts) > 1 and not terminal:
        if len(contacts) > 8:
            log_and_raise("Start contact supports at most 8 initial supports")
        return tuple(contacts)
    if len(contacts) > 1:
        log_and_raise(
            "Boundary contact supports one static support per endpoint; "
            "multiple supports are ambiguous"
        )
    return contacts[0] if contacts else None


@contextmanager
def boundary_contact_scope(
    planner: MotionPlanner,
    start: JointState,
    goal: JointState | None,
    mode: str,
    max_initial_penetration: float = 0.002,
    contact_links: tuple[str, ...] | None = None,
) -> Iterator[None]:
    """Install contact constraints before graph capture and restore them on every exit."""
    departure = (
        capture_contact(
            planner,
            start,
            terminal=False,
            max_initial_penetration=max_initial_penetration,
            contact_links=contact_links,
        )
        if mode in {"start", "both"}
        else None
    )
    arrival = capture_contact(planner, goal, terminal=True) if mode in {"end", "both"} else None
    if isinstance(departure, tuple) and mode == "both":
        log_and_raise("Multiple supports currently require start-only contact")
    if (
        departure is not None
        and mode == "both"
        and planner.scene_collision_checker.data.meshes is not None
    ):
        if departure.obstacle_name in planner.scene_collision_checker.data.meshes.names[0]:
            log_and_raise("Mesh support contact currently supports start-only queries")
    if departure is None and arrival is None:
        yield
        return
    saved = []
    solvers = (planner.ik_solver, planner.trajopt_solver)
    try:
        for solver in solvers:
            solver.core.invalidate_parameter_graphs()
            for cost in scene_costs(solver):
                if cost._contact is not None:
                    log_and_raise(
                        "Per-call boundary contact cannot override configured contact declarations"
                    )
                saved.append((cost, cost._contact))
                scene = cost.config.scene_collision_checker
                count = cost.config.num_spheres
                if solver is planner.ik_solver:
                    if arrival is not None:
                        cost._contact = ContactApproach(
                            replace(arrival, terminal_only=True), scene, count
                        )
                elif departure is not None and arrival is not None:
                    cost._contact = ContactTransfer(departure, arrival, scene, count)
                elif isinstance(departure, tuple):
                    cost._contact = ContactDepartureSet(departure, scene, count)
                elif departure is not None:
                    cost._contact = ContactSeparation(departure, scene, count)
                else:
                    cost._contact = ContactApproach(arrival, scene, count)
        yield
    finally:
        for cost, previous in saved:
            cost._contact = previous
        for solver in solvers:
            solver.core.invalidate_parameter_graphs()
