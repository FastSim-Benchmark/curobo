# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Endpoint admission uses the same declared numerical gap tolerance as the path."""

import pytest
import torch

from curobo._src.collision.contact_approach import ContactApproach, GoalContact
from curobo._src.collision.contact_separation import ContactSeparation, StartContact
from curobo._src.geom.collision.collision_scene import SceneCollision, SceneCollisionCfg
from curobo._src.geom.types import Cuboid, SceneCfg


@pytest.mark.parametrize("mesh", [False, True])
@pytest.mark.parametrize("arrival", [False, True])
@pytest.mark.parametrize(
    "excess,tolerance,accepted",
    [(0.0, 1e-5, True), (7e-6, 1e-5, True), (20e-6, 1e-5, False), (7e-6, 1e-6, False)],
)
def test_endpoint_gap_tolerance(mesh, arrival, excess, tolerance, accepted):
    """Keep bounded contact at large coordinates without admitting deeper penetration."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    table = Cuboid(name="table", dims=[1, 1, 0.1], pose=[2.3, 6.5, 0.95, 1, 0, 0, 0])
    model = SceneCfg(mesh=[table.get_mesh()]) if mesh else SceneCfg(cuboid=[table])
    scene = SceneCollision.from_config(SceneCollisionCfg(scene_model=model))
    initial = (2.3, 6.5, 1.05 - 0.002 - excess, 0.05)
    kind = GoalContact if arrival else StartContact
    declaration = kind("table", (0,), (initial,), numerical_tolerance=tolerance)
    bind = ContactApproach if arrival else ContactSeparation
    if not accepted:
        with pytest.raises(ValueError, match="initial penetration"):
            bind(declaration, scene, 1)
        return
    contact = bind(declaration, scene, 1)
    values = [initial, (2.3, 6.5, 1.06, 0.05)]
    if arrival:
        values.reverse()
    spheres = torch.tensor([[[v] for v in values]], device="cuda")
    assert bool((contact.cost(spheres) == 0).all())
    inward = spheres.clone()
    # A second inward waypoint cannot consume another tolerance or bypass monotonic contact.
    inward[0, 1 if arrival else 0, 0, 2] -= 0.001
    assert bool((contact.cost(inward) > 0).any())
    assert scene.data.meshes.enable[0, 0] == 1 if mesh else scene.data.cuboids.enable[0, 0] == 1
