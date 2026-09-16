# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Mesh support replacement preserves all other sphere/obstacle pairs."""

from types import SimpleNamespace

import pytest
import torch

from curobo._src.collision.contact_separation import ContactSeparation, StartContact
from curobo._src.cost.cost_scene_collision import SceneCollisionCost
from curobo._src.cost.cost_scene_collision_cfg import SceneCollisionCostCfg
from curobo._src.geom.collision.collision_scene import SceneCollision, SceneCollisionCfg
from curobo._src.geom.types import Cuboid, SceneCfg


@pytest.mark.parametrize("sweep", [False, True])
def test_mesh_departure_other_pairs_and_gradient(sweep):
    table = Cuboid(name="table", dims=[1, 1, 0.1], pose=[0, 0, -0.05, 1, 0, 0, 0])
    ceiling = Cuboid(name="ceiling", dims=[1, 1, 0.1], pose=[0, 0, 0.2, 1, 0, 0, 0])
    scene = SceneCollision.from_config(
        SceneCollisionCfg(
            scene_model=SceneCfg(mesh=[table.get_mesh()], cuboid=[ceiling]),
        )
    )
    declaration = StartContact("table", (0,), ((0, 0, 0.049, 0.05),))
    contact = ContactSeparation(declaration, scene, 2)
    config = SceneCollisionCostCfg(
        weight=1.0, num_spheres=2, use_sweep=sweep, start_contact=declaration
    )
    config.scene_collision_checker = scene
    cost = SceneCollisionCost(config)
    cost.setup_batch_tensors(1, 3)
    spheres = torch.tensor(
        [
            [
                [[0, 0, 0.049, 0.05], [0.2, 0, 0.04, 0.05]],
                [[0, 0, 0.06, 0.05], [0.2, 0, 0.06, 0.05]],
                [[0, 0, 0.16, 0.05], [0.2, 0, 0.07, 0.05]],
            ]
        ],
        device="cuda",
    )
    result = cost.forward(
        SimpleNamespace(robot_spheres=spheres), trajectory_dt=torch.tensor([0.01], device="cuda")
    )
    assert result[0, 0, 0] == 0
    assert result[0, 0, 1] > 0
    assert result[0, 2, 0] > 0
    assert scene.data.meshes.enable[0, 0] == 1
    inward = spheres.clone()
    inward[0, 1, 0, 2] = 0.047
    inward.requires_grad_(True)
    contact.cost(inward).sum().backward()
    assert inward.grad[0, 1, 0, 2] < 0
    with pytest.raises(ValueError, match="initial penetration"):
        ContactSeparation(StartContact("table", (0,), ((0, 0, 0.01, 0.05),)), scene, 2)
    moved_table = table.get_mesh()
    moved_table.pose[2] += 0.02
    replacement = SceneCollision.from_config(
        SceneCollisionCfg(scene_model=SceneCfg(mesh=[moved_table]))
    )
    clear = spheres.clone()
    clear[0, 2, 0, 2] = 0.07
    assert contact.cost(clear).sum() == 0
    scene.data.meshes = replacement.data.meshes
    assert contact.cost(clear).sum() > 0
