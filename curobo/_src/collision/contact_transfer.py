# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Compose bounded departure and landing within one optimized trajectory."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import torch

from curobo._src.collision.contact_approach import ContactApproach, GoalContact
from curobo._src.collision.contact_separation import ContactSeparation, StartContact
from curobo._src.util.logging import log_and_raise

if TYPE_CHECKING:
    from curobo._src.geom.collision.collision_scene import SceneCollision
    from curobo._src.types.tool_pose import ToolPose


class ContactTransfer:
    """Preserve every unselected pair and require clearance between endpoint contacts."""

    def __init__(
        self, start: StartContact, goal: GoalContact, scene: SceneCollision, num_spheres: int
    ) -> None:
        """Bind both endpoint declarations to the same captured sphere layout and scene."""
        if goal.terminal_only:
            log_and_raise("Combined contact requires a trajectory, not terminal_only IK")
        departure = ContactSeparation(start, scene, num_spheres)
        arrival = ContactApproach(goal, scene, num_spheres)
        shared = (
            tuple(index for index in start.sphere_indices if index in goal.sphere_indices)
            if start.obstacle_name == goal.obstacle_name
            else ()
        )
        self.replacement_ids = torch.stack(
            (departure.replacement_ids, arrival.replacement_ids), dim=-1
        ).contiguous()
        self.departure = None
        self.arrival = None
        self.shared_start = None
        self.shared_goal = None
        start_only = tuple(index for index in start.sphere_indices if index not in shared)
        goal_only = tuple(index for index in goal.sphere_indices if index not in shared)

        def select_start(indices: tuple[int, ...]) -> StartContact:
            return replace(
                start,
                sphere_indices=indices,
                initial_spheres=tuple(
                    start.initial_spheres[start.sphere_indices.index(index)] for index in indices
                ),
            )

        def select_goal(indices: tuple[int, ...]) -> GoalContact:
            return replace(
                goal,
                sphere_indices=indices,
                goal_spheres=tuple(
                    goal.goal_spheres[goal.sphere_indices.index(index)] for index in indices
                ),
            )

        if start_only:
            self.departure = ContactSeparation(select_start(start_only), scene, num_spheres)
        if goal_only:
            self.arrival = ContactApproach(select_goal(goal_only), scene, num_spheres)
        if shared:
            self.shared_start = ContactSeparation(select_start(shared), scene, num_spheres)
            self.shared_goal = ContactApproach(select_goal(shared), scene, num_spheres)
            if start.subdivisions != goal.subdivisions:
                log_and_raise("Combined contact declarations require matching subdivisions")

    def cost(
        self,
        robot_spheres: torch.Tensor,
        tool_poses: ToolPose | None = None,
        activation_distance: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return (batch, horizon, spheres) violations without prescribing a phase time."""
        cost = torch.zeros_like(robot_spheres[..., 0])
        if self.departure is not None:
            cost = cost + self.departure.cost(robot_spheres)
        if self.arrival is not None:
            cost = cost + self.arrival.cost(robot_spheres, tool_poses, activation_distance)
        if self.shared_start is None:
            return cost
        start = self.shared_start
        goal = self.shared_goal
        end = goal.separation
        spheres = robot_spheres.index_select(-2, start.indices)
        batch, horizon = spheres.shape[:2]
        if horizon < 2:
            log_and_raise("Combined contact requires a trajectory with at least two states")
        gap = start.clearance(start.interpolate_samples(spheres))
        start_tolerance = start.declaration.numerical_tolerance
        goal_tolerance = end.declaration.numerical_tolerance
        prefix = torch.cummax(gap.clamp_max(start.declaration.release_clearance), dim=1).values
        suffix = torch.cummax(
            gap.flip(1).clamp_max(end.declaration.release_clearance), dim=1
        ).values.flip(1)
        departure = (start.initial_clearance - gap - start_tolerance).relu()
        departure = departure + (prefix - gap - start_tolerance).relu()
        approach = (end.initial_clearance - gap - goal_tolerance).relu()
        approach = approach + (suffix - gap - goal_tolerance).relu()
        violation = torch.minimum(departure, approach).sum(-1)
        interval = violation[:, :-1].reshape(batch, horizon - 1, -1).amax(-1)
        shared = torch.cat((interval, violation[:, -1:]), dim=1)
        clearance = max(start.declaration.release_clearance, end.declaration.release_clearance)
        # A path that stays on the support is never a valid pick/place transfer.
        release_error = (
            (clearance - gap.amax(1) - min(start_tolerance, goal_tolerance)).relu().sum(-1)
        )
        first = torch.arange(horizon, device=spheres.device) == 0
        last = torch.arange(horizon, device=spheres.device) == horizon - 1
        start_error = (
            (spheres[:, 0] - start.initial_spheres).abs().amax((-1, -2)) - start_tolerance
        ).relu()
        end_error = (
            (spheres[:, -1] - end.initial_spheres).abs().amax((-1, -2)) - goal_tolerance
        ).relu()
        shared = shared + (release_error + start_error)[:, None] * first
        shared = shared + end_error[:, None] * last
        shared = shared + start.geometry_cost(spheres) + end.geometry_cost(spheres)
        cost = cost + shared[..., None] * start.output_mask
        if goal.landing is not None:
            released = torch.cummax(start.clearance(spheres), dim=1).values
            released = (released >= start.declaration.release_clearance - start_tolerance).all(-1)
            cost = (
                cost
                + goal.landing.cost(robot_spheres, tool_poses, False, activation_distance)
                * released[..., None]
            )
        return cost
