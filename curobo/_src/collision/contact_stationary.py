# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Bounded existing contacts whose sphere geometry stays fixed throughout a query."""

from __future__ import annotations

import torch

from curobo._src.collision.contact_separation import ContactSeparation, StartContact
from curobo._src.geom.collision.collision_scene import SceneCollision
from curobo._src.util.logging import log_and_raise


class StationaryContactSet:
    """Replace only captured pairs; reject any center, radius or support change."""

    def __init__(
        self, declarations: tuple[StartContact, ...], scene: SceneCollision, num_spheres: int,
    ) -> None:
        """Apply the existing depth admission to one through eight static supports."""
        if not 1 <= len(declarations) <= 8:
            log_and_raise("Stationary contact supports 1..8 captured supports")
        self.contacts = tuple(ContactSeparation(item, scene, num_spheres) for item in declarations)
        self.replacement_ids = torch.stack([item.replacement_ids for item in self.contacts], dim=-1)
        self.replacement_mesh_ids = torch.stack([
            item.replacement_mesh_ids if item.is_mesh else torch.full_like(item.replacement_ids, -1)
            for item in self.contacts
        ], dim=-1)

    def cost(self, robot_spheres: torch.Tensor) -> torch.Tensor:
        """Return (batch, horizon, sphere) violations, including single-state IK."""
        costs = []
        for contact in self.contacts:
            spheres = robot_spheres.index_select(-2, contact.indices)
            change = (spheres - contact.initial_spheres).abs().amax(dim=(-1, -2))
            violation = (change - contact.declaration.numerical_tolerance).relu()
            violation = violation + contact.geometry_cost(spheres)
            costs.append(violation[..., None] * contact.output_mask)
        return torch.stack(costs).sum(0)
