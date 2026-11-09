# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Conjunction of bounded departures from a support assembled from several meshes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from curobo._src.collision.contact_separation import ContactSeparation, StartContact
from curobo._src.geom.collision.collision_scene import SceneCollision
from curobo._src.util.logging import log_and_raise

if TYPE_CHECKING:
    from curobo._src.collision.contact_stationary import StationaryContactSet


class ContactDepartureSet:
    """Every captured sphere/support pair must independently separate."""

    def __init__(
        self,
        declarations: tuple[StartContact, ...],
        scene: SceneCollision,
        num_spheres: int,
        *,
        stationary: StationaryContactSet | None = None,
    ) -> None:
        """Bind a finite set without disabling any obstacle."""
        if not 1 <= len(declarations) + (len(stationary.contacts) if stationary else 0) <= 8:
            log_and_raise("Start contact supports 1..8 initial supports")
        if stationary is not None:
            held = {i for c in stationary.contacts for i in c.declaration.sphere_indices}
            if any(held.intersection(d.sphere_indices) for d in declarations):
                log_and_raise("Stationary and departure spheres must be disjoint")
        self.stationary = stationary
        self.contacts = tuple(ContactSeparation(d, scene, num_spheres) for d in declarations)
        self.replacement_ids = torch.stack([c.replacement_ids for c in self.contacts], dim=-1)
        self.replacement_mesh_ids = torch.stack(
            [
                c.replacement_mesh_ids if c.is_mesh else torch.full_like(c.replacement_ids, -1)
                for c in self.contacts
            ],
            dim=-1,
        )
        if stationary is not None:
            self.replacement_ids = torch.cat(
                (self.replacement_ids, stationary.replacement_ids), -1
            )
            self.replacement_mesh_ids = torch.cat(
                (self.replacement_mesh_ids, stationary.replacement_mesh_ids), -1)

    def cost(self, robot_spheres: torch.Tensor) -> torch.Tensor:
        """Sum violations; satisfying one support cannot compensate for another."""
        result = torch.stack([c.cost(robot_spheres) for c in self.contacts]).sum(0)
        return result if self.stationary is None else result + self.stationary.cost(robot_spheres)
