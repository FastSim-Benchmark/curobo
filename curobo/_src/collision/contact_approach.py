# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Bounded terminal contact using the time reversal of contact separation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from curobo._src.collision.contact_landing import LandingAlignment, NormalLanding
from curobo._src.collision.contact_separation import ContactSeparation, StartContact
from curobo._src.util.logging import log_and_raise

if TYPE_CHECKING:
    from curobo._src.geom.collision.collision_scene import SceneCollision
    from curobo._src.types.tool_pose import ToolPose


@dataclass(frozen=True)
class GoalContact:
    """One immutable terminal-contact declaration for a single-environment query."""

    #: Name of the static cuboid support.
    obstacle_name: str
    #: Robot sphere indices whose final support contact is being replaced.
    sphere_indices: tuple[int, ...]
    #: Captured world-frame goal spheres, one (x, y, z, radius) per index.
    goal_spheres: tuple[tuple[float, float, float, float], ...]
    #: Maximum accepted terminal sphere-model penetration, in meters.
    max_goal_penetration: float = 0.002
    #: Clearance before entering the final monotonic approach, in meters.
    approach_clearance: float = 0.005
    #: Absolute numerical tolerance for gap and captured-geometry comparisons.
    numerical_tolerance: float = 1.0e-5
    #: Linear sphere-center samples per trajectory interval.
    subdivisions: int = 4
    #: Explicit single-state goal validation for IK; never use on trajectory rollouts.
    terminal_only: bool = False
    #: Optional final-pose alignment; requires normal-only motion inside the approach band.
    landing: NormalLanding | None = None

    def __post_init__(self) -> None:
        """Validate the same bounded geometry contract as a reversed departure."""
        if type(self.terminal_only) is not bool:
            log_and_raise("GoalContact terminal_only must be a bool")
        self.as_departure()

    def as_departure(self) -> StartContact:
        """Describe this contact as the starting state of a reversed trajectory."""
        return StartContact(
            obstacle_name=self.obstacle_name,
            sphere_indices=self.sphere_indices,
            initial_spheres=self.goal_spheres,
            max_initial_penetration=self.max_goal_penetration,
            release_clearance=self.approach_clearance,
            numerical_tolerance=self.numerical_tolerance,
            subdivisions=self.subdivisions,
        )


class ContactApproach:
    """Constrain a final approach while preserving the captured support geometry."""

    def __init__(self, declaration: GoalContact, scene: SceneCollision, num_spheres: int) -> None:
        """Bind the placement declaration before CUDA graph capture."""
        self.declaration = declaration
        self.separation = ContactSeparation(declaration.as_departure(), scene, num_spheres)
        self.replacement_ids = self.separation.replacement_ids
        self.landing = (
            LandingAlignment(declaration.landing, self.separation)
            if declaration.landing is not None
            else None
        )

    def cost(
        self,
        robot_spheres: torch.Tensor,
        tool_poses: ToolPose | None = None,
        activation_distance: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return per-step violations for (batch, horizon, num_spheres, 4) spheres.

        Outside the approach clearance, normal motion is allowed. Once inside,
        the gap can only decrease toward the captured terminal contact, without
        overshoot or rebound. Other sphere/obstacle pairs keep their own checks.
        """
        if self.declaration.terminal_only:
            cost = self.separation.captured_state_cost(robot_spheres)
        else:
            cost = self.separation.cost(robot_spheres.flip(1)).flip(1)
        if self.landing is not None:
            cost = cost + self.landing.cost(
                robot_spheres, tool_poses, self.declaration.terminal_only, activation_distance
            )
        return cost
