# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Request-scoped endpoint contact capture and collision rollout installation."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from typing import TYPE_CHECKING, Iterator

import torch

from curobo._src.collision.contact_approach import ContactApproach, GoalContact
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
    planner: MotionPlanner, state: JointState | None, *, terminal: bool
) -> StartContact | GoalContact | None:
    """Capture shallow sphere/box endpoint contacts in the planner's current frame."""
    scene = planner.scene_collision_checker
    if state is None or scene is None or scene.data.cuboids is None:
        return None
    if state.shape[0] != 1 or scene.data.num_envs != 1:
        log_and_raise("Boundary contact currently requires one planning problem and environment")
    spheres = planner.compute_kinematics(state).robot_spheres.reshape(-1, 4)
    cuboids = scene.data.cuboids
    contacts = []
    for index, name in enumerate(cuboids.names[0]):
        if not bool(cuboids.enable[0, index].item()):
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
        indices = torch.nonzero((spheres[:, 3] > 0) & (gap <= 1.0e-5)).reshape(-1)
        if indices.numel() == 0:
            continue
        selected = tuple(indices.tolist())
        geometry = tuple(tuple(sphere) for sphere in spheres[indices].tolist())
        declaration = (
            GoalContact(name, selected, geometry)
            if terminal
            else StartContact(name, selected, geometry)
        )
        # Validate depth, identity, and enabled geometry before installing any exemption.
        captured = (
            declaration.as_departure() if isinstance(declaration, GoalContact) else declaration
        )
        ContactSeparation(captured, scene, len(spheres))
        contacts.append(declaration)
    if len(contacts) > 1:
        log_and_raise(
            "Boundary contact supports one static cuboid per endpoint; "
            "multiple supports are ambiguous"
        )
    return contacts[0] if contacts else None


@contextmanager
def boundary_contact_scope(
    planner: MotionPlanner, start: JointState, goal: JointState | None, mode: str
) -> Iterator[None]:
    """Install contact constraints before graph capture and restore them on every exit."""
    departure = (
        capture_contact(planner, start, terminal=False) if mode in {"start", "both"} else None
    )
    arrival = capture_contact(planner, goal, terminal=True) if mode in {"end", "both"} else None
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
