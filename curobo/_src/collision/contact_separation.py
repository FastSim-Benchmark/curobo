# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Bounded initial contact constraints for a static cuboid support.

This experimental constraint replaces only explicitly named sphere/cuboid pairs.
It preserves the captured initial gap, prevents loss of separation until release,
and requires the terminal configuration to clear the support. It does not model
contact forces or make guarantees about a physical mesh behind a sphere proxy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from curobo._src.types.pose import Pose
from curobo._src.util.logging import log_and_raise

if TYPE_CHECKING:
    from curobo._src.geom.collision.collision_scene import SceneCollision


@dataclass(frozen=True)
class StartContact:
    """One immutable initial-contact declaration for a single-environment query."""

    #: Name of the static cuboid support in the collision scene.
    obstacle_name: str
    #: Robot sphere indices whose support contact is being replaced.
    sphere_indices: tuple[int, ...]
    #: Captured world-frame spheres, one (x, y, z, radius) per index.
    initial_spheres: tuple[tuple[float, float, float, float], ...]
    #: Maximum accepted initial sphere-model penetration, in meters.
    max_initial_penetration: float = 0.002
    #: Required support clearance at release and at the terminal state, in meters.
    release_clearance: float = 0.005
    #: Absolute tolerance for numerical gap comparisons, in meters.
    numerical_tolerance: float = 1.0e-5
    #: Linear sphere-center samples per trajectory interval.
    subdivisions: int = 4

    def __post_init__(self) -> None:
        """Reject malformed or unbounded contact declarations."""
        if not self.obstacle_name or not self.sphere_indices:
            log_and_raise("StartContact requires a support name and sphere indices")
        if len(self.sphere_indices) != len(set(self.sphere_indices)) or any(
            type(index) is not int or index < 0 for index in self.sphere_indices
        ):
            log_and_raise("StartContact sphere indices must be distinct nonnegative integers")
        if len(self.initial_spheres) != len(self.sphere_indices) or any(
            len(sphere) != 4
            or any(not math.isfinite(value) for value in sphere)
            or sphere[3] <= 0.0
            for sphere in self.initial_spheres
        ):
            log_and_raise("StartContact requires finite positive-radius captured spheres")
        if (
            any(
                not math.isfinite(value) or value <= 0.0
                for value in (
                    self.max_initial_penetration,
                    self.release_clearance,
                    self.numerical_tolerance,
                )
            )
            or self.numerical_tolerance >= self.release_clearance
        ):
            log_and_raise("StartContact distances must be positive, with tolerance below release")
        if type(self.subdivisions) is not int or not 1 <= self.subdivisions <= 32:
            log_and_raise("StartContact subdivisions must be an integer from 1 to 32")


class ContactSeparation:
    """Evaluate one bound declaration using GPU tensors and differentiable box distances."""

    def __init__(self, declaration: StartContact, scene: SceneCollision, num_spheres: int) -> None:
        """Bind a declaration to the current immutable scene and robot sphere layout."""
        self.declaration = declaration
        self.scene = scene
        cuboids = scene.data.cuboids
        if scene.data.num_envs != 1 or cuboids is None:
            log_and_raise("StartContact currently requires one environment with a cuboid support")
        if declaration.obstacle_name not in cuboids.names[0]:
            log_and_raise(f"StartContact cuboid {declaration.obstacle_name!r} does not exist")
        obstacle_index = cuboids.names[0].index(declaration.obstacle_name)
        self.support_index = obstacle_index
        if not bool(cuboids.enable[0, obstacle_index].item()):
            log_and_raise("StartContact support must be enabled")
        if max(declaration.sphere_indices) >= num_spheres:
            log_and_raise("StartContact sphere index is outside the robot collision model")
        device = scene.device_cfg.device
        self.indices = torch.tensor(declaration.sphere_indices, device=device, dtype=torch.long)
        self.initial_spheres = torch.tensor(
            declaration.initial_spheres, device=device, dtype=torch.float32
        )
        self.half_extents = cuboids.dims[0, obstacle_index, :3].clone() * 0.5
        inverse = cuboids.inv_pose[0, obstacle_index, :7].clone()
        self.captured_inverse_pose = inverse.clone()
        self.inverse_position = inverse[:3]
        self.inverse_rotation = Pose(
            position=inverse[:3], quaternion=inverse[3:7]
        ).get_rotation_matrix()[0]
        self.initial_clearance = self.clearance(self.initial_spheres)
        if bool((self.initial_clearance < -declaration.max_initial_penetration).any()):
            log_and_raise(
                "StartContact initial penetration exceeds its declared geometry tolerance"
            )
        if bool((self.initial_clearance > declaration.numerical_tolerance).any()):
            log_and_raise("StartContact may only replace spheres actually touching the support")
        self.replacement_ids = torch.full((1, num_spheres), -1, device=device, dtype=torch.int32)
        self.replacement_ids[0, self.indices] = obstacle_index
        self.output_mask = torch.zeros(num_spheres, device=device, dtype=torch.float32)
        self.output_mask[declaration.sphere_indices[0]] = 1.0
        self.fractions = (
            torch.arange(declaration.subdivisions, device=device, dtype=torch.float32)
            / declaration.subdivisions
        )

    def clearance(self, spheres: torch.Tensor) -> torch.Tensor:
        """Return signed sphere/OBB clearance for (..., 4) world-frame spheres."""
        local = spheres[..., :3] @ self.inverse_rotation.T + self.inverse_position
        outside = local.abs() - self.half_extents
        return (
            torch.linalg.vector_norm(outside.clamp_min(0.0), dim=-1)
            + outside.amax(dim=-1).clamp_max(0.0)
            - spheres[..., 3]
        )

    def cost(self, robot_spheres: torch.Tensor) -> torch.Tensor:
        """Return per-step violations with shape (batch, horizon, num_spheres).

        Gaps are capped at release clearance before taking a running maximum,
        so retreat after release is allowed as long as normal clearance remains.
        Prefix maxima prevent small successive inward moves from accumulating.
        """
        spheres = robot_spheres.index_select(-2, self.indices)
        batch, horizon, count, _ = spheres.shape
        if horizon < 2:
            log_and_raise("StartContact is a trajectory constraint, not a single-state exemption")
        dense = spheres[:, :-1, None] + self.fractions[None, None, :, None, None] * (
            spheres[:, 1:, None] - spheres[:, :-1, None]
        )
        dense = torch.cat((dense.reshape(batch, -1, count, 4), spheres[:, -1:]), dim=1)
        gap = self.clearance(dense)
        tolerance = self.declaration.numerical_tolerance
        capped = gap.clamp_max(self.declaration.release_clearance)
        best_gap = torch.cummax(capped, dim=1).values
        violations = (
            (self.initial_clearance - gap - tolerance).relu() + (best_gap - gap - tolerance).relu()
        ).sum(dim=-1)
        interval = violations[:, :-1].reshape(batch, horizon - 1, -1).amax(dim=-1)
        terminal = violations[:, -1:] + (
            self.declaration.release_clearance - gap[:, -1] - tolerance
        ).relu().sum(dim=-1, keepdim=True)
        cost = torch.cat((interval, terminal), dim=1)
        # The declaration must match the actual initial geometry, not merely its gap.
        start_error = (
            (spheres[:, 0] - self.initial_spheres).abs().amax(dim=(-1, -2)) - tolerance
        ).relu()
        first_mask = torch.arange(horizon, device=spheres.device) == 0
        cost = cost + start_error[:, None] * first_mask[None]
        cuboids = self.scene.data.cuboids
        support_changed = (
            (cuboids.inv_pose[0, self.support_index, :7] - self.captured_inverse_pose).abs().amax()
            + (cuboids.dims[0, self.support_index, :3] - 2.0 * self.half_extents).abs().amax()
            + (1.0 - cuboids.enable[0, self.support_index].to(torch.float32))
        )
        radius_error = (
            (spheres[..., 3] - self.initial_spheres[:, 3]).abs().amax(dim=-1) - tolerance
        ).relu()
        cost = cost + support_changed + radius_error
        return cost[..., None] * self.output_mask
