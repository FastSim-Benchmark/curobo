# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Authored exclusions affect only the declared sphere/obstacle pairs."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from curobo._src.cost.cost_scene_collision import SceneCollisionCost
from curobo._src.cost.cost_scene_collision_cfg import SceneCollisionCostCfg
from curobo._src.geom.collision.collision_scene import SceneCollision, SceneCollisionCfg
from curobo._src.geom.types import Cuboid, SceneCfg
from curobo._src.motion.motion_contact import capture_contact


@pytest.fixture(params=["cuboid", "mesh"])
def scene(request: pytest.FixtureRequest) -> SceneCollision:
    """Provide a cable and a distinct table, with their collisions enabled."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    shapes = [
        Cuboid(name="cable", dims=[0.1, 0.1, 0.1], pose=[0, 0, 0, 1, 0, 0, 0]),
        Cuboid(name="table", dims=[0.1, 0.1, 0.1], pose=[0.3, 0, 0, 1, 0, 0, 0]),
    ]
    model = (SceneCfg(cuboid=shapes) if request.param == "cuboid"
             else SceneCfg(mesh=[shape.get_mesh() for shape in shapes]))
    return SceneCollision.from_config(SceneCollisionCfg(scene_model=model))


def evaluate(scene: SceneCollision, sweep: bool = False) -> torch.Tensor:
    """Two overlapping spheres: payload index 0 and unrelated robot index 1."""
    cfg = SceneCollisionCostCfg(weight=1.0, num_spheres=2, use_sweep=sweep)
    cfg.scene_collision_checker = scene
    cost = SceneCollisionCost(cfg)
    cost.setup_batch_tensors(1, 3)
    spheres = torch.tensor(
        [
            [
                [[0, 0, 0, 0.02], [0, 0, 0, 0.02]],
                [[0.15, 0, 0, 0.02], [0.15, 0, 0, 0.02]],
                [[0.3, 0, 0, 0.02], [0.3, 0, 0, 0.02]],
            ]
        ],
        device="cuda",
    )
    return cost.forward(
        SimpleNamespace(robot_spheres=spheres), trajectory_dt=torch.tensor([0.1], device="cuda")
    )


@pytest.mark.parametrize("sweep", [False, True])
def test_filtered_payload_keeps_robot_and_table_collision(
    scene: SceneCollision, sweep: bool
) -> None:
    """Do not remove the cable globally or exempt the payload from the table."""
    scene.set_collision_pair_filters({"cable": (0,)}, num_spheres=2)
    result = evaluate(scene, sweep)
    assert result[0, 0, 0] == 0
    assert result[0, 0, 1] > 0
    assert result[0, 2, 0] > 0
    for data in (scene.data.cuboids, scene.data.meshes):
        if data is not None:
            assert data.enable[0, :2].tolist() == [1, 1]


def test_clear_restores_pair_and_invalid_update_is_atomic(scene: SceneCollision) -> None:
    """Release must restore collisions; invalid names/indices cannot partially apply."""
    scene.set_collision_pair_filters({"cable": (0,)}, num_spheres=2)
    for invalid in ({"missing": (0,)}, {"cable": (-1,)}, {"cable": (2,)}):
        with pytest.raises(ValueError):
            scene.set_collision_pair_filters(invalid, num_spheres=2)
        assert evaluate(scene)[0, 0, 0] == 0
    scene.set_collision_pair_filters({}, num_spheres=2)
    assert evaluate(scene)[0, 0, 0] > 0


def test_boundary_capture_skips_only_filtered_pair(scene: SceneCollision) -> None:
    """An authored overlap is not a support contact; real deep contact still fails."""
    spheres = torch.tensor([[[[0, 0, 0, 0.02], [0.15, 0, 0, 0.02]]]], device="cuda")
    planner = SimpleNamespace(
        scene_collision_checker=scene,
        compute_kinematics=lambda state: SimpleNamespace(robot_spheres=spheres),
    )
    state = SimpleNamespace(shape=(1,))
    scene.set_collision_pair_filters({"cable": (0,)}, num_spheres=2)
    assert capture_contact(planner, state, terminal=False) is None
    spheres[0, 0, 0, 0] = 0.3
    with pytest.raises(ValueError, match="initial penetration"):
        capture_contact(planner, state, terminal=False)


def test_world_replacement_clears_stale_slot_exclusion(scene: SceneCollision) -> None:
    """Reusing a cable's slot for a table must not inherit the cable exclusion."""
    scene.set_collision_pair_filters({"cable": (0,)}, num_spheres=2)
    obstacle = Cuboid(name="new_table", dims=[0.1, 0.1, 0.1], pose=[0, 0, 0, 1, 0, 0, 0])
    replacement = (SceneCfg(cuboid=[obstacle]) if scene.data.cuboids is not None
                   else SceneCfg(mesh=[obstacle.get_mesh()]))
    scene.load_collision_model(replacement)
    assert evaluate(scene)[0, 0, 0] > 0
