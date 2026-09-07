# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Align a held object's tool frame before permitting final support contact."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from curobo._src.util.logging import log_and_raise

if TYPE_CHECKING:
    from curobo._src.collision.contact_separation import ContactSeparation
    from curobo._src.types.tool_pose import ToolPose


@dataclass(frozen=True)
class NormalLanding:
    """A fixed-orientation normal approach to a captured world-frame tool pose."""

    #: Tool frame rigidly connected to the held payload.
    tool_frame: str
    #: User-provided final world position, in meters.
    goal_position: tuple[float, float, float]
    #: User-provided final world quaternion, in wxyz order.
    goal_quaternion: tuple[float, float, float, float]
    #: Unit normal pointing out of the support toward free space.
    outward_normal: tuple[float, float, float]
    #: Maximum lateral offset from the final normal line, in meters.
    position_tolerance: float = 1e-4
    #: Maximum orientation error during final approach, in radians.
    orientation_tolerance: float = 1e-3

    def __post_init__(self) -> None:
        """Reject malformed target poses, normals, and alignment tolerances."""
        if not self.tool_frame:
            log_and_raise("NormalLanding requires a tool frame")
        for values, length in (
            (self.goal_position, 3),
            (self.goal_quaternion, 4),
            (self.outward_normal, 3),
        ):
            if len(values) != length or any(not math.isfinite(value) for value in values):
                log_and_raise("NormalLanding requires finite world-frame pose and normal values")
        if (
            abs(sum(value * value for value in self.goal_quaternion) - 1.0) > 1e-5
            or abs(sum(value * value for value in self.outward_normal) - 1.0) > 1e-5
        ):
            log_and_raise("NormalLanding quaternion and outward normal must have unit length")
        if (
            not math.isfinite(self.position_tolerance)
            or self.position_tolerance <= 0
            or not math.isfinite(self.orientation_tolerance)
            or not 0 < self.orientation_tolerance < math.pi
        ):
            log_and_raise("NormalLanding requires positive finite alignment tolerances")


class LandingAlignment:
    """Require lateral and orientation alignment inside the contact approach band."""

    def __init__(self, declaration: NormalLanding, separation: ContactSeparation) -> None:
        """Capture the landing target and validate its direction against the support."""
        self.declaration = declaration
        self.separation = separation
        device = separation.initial_spheres.device
        self.goal_position = torch.tensor(
            declaration.goal_position, device=device, dtype=torch.float32
        )
        self.goal_quaternion = torch.tensor(
            declaration.goal_quaternion, device=device, dtype=torch.float32
        )
        self.normal = torch.tensor(declaration.outward_normal, device=device, dtype=torch.float32)
        self.clearance = separation.declaration.release_clearance
        if declaration.position_tolerance >= self.clearance:
            log_and_raise("NormalLanding lateral tolerance must be below approach clearance")
        self.quaternion_tolerance = 2.0 * math.sin(declaration.orientation_tolerance / 4.0)
        raised = separation.initial_spheres.clone()
        raised[:, :3] += self.normal * self.clearance
        clearance_gain = separation.clearance(raised) - separation.initial_clearance
        if bool(
            (clearance_gain < self.clearance - separation.declaration.numerical_tolerance).any()
        ):
            log_and_raise("NormalLanding outward direction must agree with the support normal")

    def cost(
        self,
        robot_spheres: torch.Tensor,
        tool_poses: ToolPose | None,
        terminal_only: bool,
        activation_distance: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return (batch, horizon, num_spheres) alignment violations.

        Above the approach band, transport and rotation remain free. Inside it,
        the tool must stay on the final normal line with its final orientation.
        The terminal-only branch applies the same complete pose target to goal IK.
        Positive collision activation distance extends the optimizer's guidance
        band and shrinks its alignment dead zone, without changing the declared
        tolerances used by zero-activation validity checks.
        """
        if tool_poses is None or self.declaration.tool_frame not in tool_poses.tool_frames:
            log_and_raise("NormalLanding requires the declared tool pose in the kinematics state")
        if tool_poses.position.shape[:2] != robot_spheres.shape[:2]:
            log_and_raise("NormalLanding tool pose and sphere batch/horizon shapes must match")
        index = tool_poses.tool_frames.index(self.declaration.tool_frame)
        position = tool_poses.position[:, :, index]
        quaternion = tool_poses.quaternion[:, :, index]
        # Pick the target hemisphere before interpolation, so q and -q remain equivalent.
        quaternion = torch.where(
            ((quaternion * self.goal_quaternion).sum(-1) < 0)[..., None],
            -quaternion,
            quaternion,
        )
        horizon = robot_spheres.shape[1]
        position_tolerance = self.declaration.position_tolerance
        quaternion_tolerance = self.quaternion_tolerance
        if not terminal_only:
            spheres = robot_spheres.index_select(-2, self.separation.indices)
            gap = self.separation.clearance(self.separation.interpolate_samples(spheres)).amin(-1)
            clearance = self.clearance
            if activation_distance is not None:
                activation_distance = activation_distance.clamp_min(0)
                clearance = clearance + activation_distance
                position_tolerance = (position_tolerance - activation_distance).clamp_min(0)
                quaternion_tolerance = (
                    quaternion_tolerance - activation_distance / self.clearance
                ).clamp_min(0)
            clearance_error = (clearance - gap).relu()
            position = self.separation.interpolate_samples(position)
            quaternion = self.separation.interpolate_samples(quaternion)
        quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        displacement = position - self.goal_position
        if not terminal_only:
            displacement = (
                displacement - (displacement * self.normal).sum(-1, keepdim=True) * self.normal
            )
        lateral_error = (displacement.norm(dim=-1) - position_tolerance).relu()
        orientation_error = (
            torch.minimum(
                (quaternion - self.goal_quaternion).norm(dim=-1),
                (quaternion + self.goal_quaternion).norm(dim=-1),
            )
            - quaternion_tolerance
        ).relu()
        alignment_error = lateral_error + self.clearance * orientation_error
        # The feasible set is the union of clear transport and aligned landing.
        # Taking the minimum avoids a vanishing product gradient near
        # the clearance boundary while preserving the same zero-cost states.
        violation = (
            alignment_error if terminal_only else torch.minimum(clearance_error, alignment_error)
        )
        if not terminal_only:
            interval = violation[:, :-1].reshape(violation.shape[0], horizon - 1, -1).amax(-1)
            violation = torch.cat((interval, violation[:, -1:]), dim=1)
        return violation[..., None] * self.separation.output_mask
