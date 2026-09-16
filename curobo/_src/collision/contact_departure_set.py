# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Conjunction of bounded departures from a support assembled from several meshes."""

import torch

from curobo._src.collision.contact_separation import ContactSeparation, StartContact
from curobo._src.geom.collision.collision_scene import SceneCollision
from curobo._src.util.logging import log_and_raise


class ContactDepartureSet:
    """Every captured sphere/support pair must independently separate."""

    def __init__(
        self,
        declarations: tuple[StartContact, ...],
        scene: SceneCollision,
        num_spheres: int,
    ) -> None:
        """Bind a finite set without disabling any obstacle."""
        if not 1 <= len(declarations) <= 8:
            log_and_raise("Start contact supports 1..8 initial supports")
        self.contacts = tuple(ContactSeparation(d, scene, num_spheres) for d in declarations)
        self.replacement_ids = torch.stack([c.replacement_ids for c in self.contacts], dim=-1)
        self.replacement_mesh_ids = torch.stack(
            [
                c.replacement_mesh_ids if c.is_mesh else torch.full_like(c.replacement_ids, -1)
                for c in self.contacts
            ],
            dim=-1,
        )

    def cost(self, robot_spheres: torch.Tensor) -> torch.Tensor:
        """Sum violations; satisfying one support cannot compensate for another."""
        return torch.stack([c.cost(robot_spheres) for c in self.contacts]).sum(0)
