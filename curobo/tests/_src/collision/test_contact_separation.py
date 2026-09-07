# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for selective, bounded contact departure."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from curobo._src.collision.contact_separation import ContactSeparation, StartContact
from curobo._src.cost.cost_scene_collision import SceneCollisionCost
from curobo._src.cost.cost_scene_collision_cfg import SceneCollisionCostCfg
from curobo._src.geom.collision.buffer_collision import CollisionBuffer
from curobo._src.geom.collision.collision_scene import SceneCollision, SceneCollisionCfg
from curobo._src.geom.types import Cuboid, SceneCfg


@pytest.fixture
def scene() -> SceneCollision:
    """Keep a table and a separate ceiling active for every query."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    return SceneCollision.from_config(
        SceneCollisionCfg(
            scene_model=SceneCfg(
                cuboid=[
                    Cuboid(name="table", dims=[1.0, 1.0, 0.1], pose=[0, 0, -0.05, 1, 0, 0, 0]),
                    Cuboid(name="ceiling", dims=[1.0, 1.0, 0.1], pose=[0, 0, 0.2, 1, 0, 0, 0]),
                ]
            )
        )
    )


def declaration() -> StartContact:
    """Describe a sphere with one millimeter of known model overlap."""
    return StartContact("table", (0,), ((0.0, 0.0, 0.049, 0.05),))


def trajectory(heights: list[float]) -> torch.Tensor:
    """Construct a sphere trajectory without changing its actual radius."""
    return torch.tensor([[[[0.0, 0.0, z, 0.05]] for z in heights]], device="cuda")


@pytest.mark.parametrize(
    "heights,accepted",
    [
        ([0.049, 0.051, 0.06], True),
        ([0.049, 0.049, 0.06], True),
        ([0.049, 0.047, 0.06], False),
        ([0.049, 0.049, 0.049], False),
        ([0.049, 0.06, 0.049, 0.07], False),
        ([0.049, 0.07, 0.056, 0.06], True),
        ([0.05, 0.055, 0.06], False),
    ],
)
def test_contact_departure_rules(
    scene: SceneCollision, heights: list[float], accepted: bool
) -> None:
    """Accept departure, but reject deeper contact, recontact, or a mismatched start."""
    contact = ContactSeparation(declaration(), scene, 1)
    cost = contact.cost(trajectory(heights))
    assert bool((cost == 0.0).all()) is accepted


def test_deep_initial_penetration_is_rejected(scene: SceneCollision) -> None:
    """Do not automatically convert an arbitrary colliding state into allowed contact."""
    contact = StartContact("table", (0,), ((0.0, 0.0, 0.04, 0.05),))
    with pytest.raises(ValueError, match="initial penetration"):
        ContactSeparation(contact, scene, 1)


@pytest.mark.parametrize("sweep", [False, True])
def test_other_pairs_remain_checked(scene: SceneCollision, sweep: bool) -> None:
    """Replacing payload/table contact must preserve ceiling and gripper/table collision."""
    config = SceneCollisionCostCfg(
        weight=1.0,
        num_spheres=2,
        use_sweep=sweep,
        start_contact=declaration(),
    )
    config.scene_collision_checker = scene
    cost = SceneCollisionCost(config)
    cost.setup_batch_tensors(1, 3)
    payload = trajectory([0.049, 0.06, 0.16])
    gripper = trajectory([0.04, 0.06, 0.07])
    state = SimpleNamespace(robot_spheres=torch.cat((payload, gripper), dim=2))
    result = cost.forward(state, trajectory_dt=torch.tensor([0.01], device="cuda"))
    assert result[0, 0, 0] == 0.0
    assert result[0, 0, 1] > 0.0
    assert result[0, 2, 0] > 0.0
    assert scene.data.cuboids.enable[0, :2].tolist() == [1, 1]


def test_contact_gradient_points_out_of_support(scene: SceneCollision) -> None:
    """The separation constraint must push an inward waypoint away from the support."""
    contact = ContactSeparation(declaration(), scene, 1)
    spheres = trajectory([0.049, 0.047, 0.06]).requires_grad_(True)
    contact.cost(spheres).sum().backward()
    assert torch.isfinite(spheres.grad).all()
    assert spheres.grad[0, 1, 0, 2] < 0.0


@pytest.mark.parametrize("sweep", [False, True])
@pytest.mark.parametrize("invalid", ["shape", "dtype", "device", "stride"])
def test_invalid_replacement_map_is_rejected(
    scene: SceneCollision, sweep: bool, invalid: str
) -> None:
    """Reject unsafe map storage before either native collision kernel is launched."""
    spheres = trajectory([0.049, 0.055, 0.06]).repeat(1, 1, 2, 1)
    maps = {
        "shape": torch.zeros((1, 1), dtype=torch.int32, device="cuda"),
        "dtype": torch.zeros((1, 2), dtype=torch.int64, device="cuda"),
        "device": torch.zeros((1, 2), dtype=torch.int32, device="cpu"),
        "stride": torch.zeros((1, 4), dtype=torch.int32, device="cuda")[:, ::2],
    }
    arguments = dict(
        scene=scene.data,
        query_sphere=spheres,
        collision_buffer=CollisionBuffer.from_shape(spheres.shape, scene.device_cfg),
        weight=torch.ones(1, device="cuda"),
        activation_distance=torch.zeros(1, device="cuda"),
        replacement_cuboid_ids=maps[invalid],
    )
    with pytest.raises(ValueError, match="replacement_cuboid_ids must"):
        if sweep:
            scene.checker.get_swept_sphere_distance(
                **arguments, trajectory_dt=torch.tensor([0.01], device="cuda")
            )
        else:
            scene.checker.get_sphere_distance(**arguments)


def test_changed_support_is_rejected(scene: SceneCollision) -> None:
    """A cached declaration must not exempt a moved or resized support."""
    contact = ContactSeparation(declaration(), scene, 1)
    spheres = trajectory([0.049, 0.055, 0.06])
    assert contact.cost(spheres).sum() == 0.0
    scene.data.cuboids.dims[0, 0, 2] += 0.01
    assert contact.cost(spheres).sum() > 0.0


def test_contact_checks_between_waypoints(scene: SceneCollision) -> None:
    """Do not tunnel through the support between two clear endpoint spheres."""
    contact = ContactSeparation(declaration(), scene, 1)
    spheres = torch.tensor(
        [[[[0, 0, 0.049, 0.05]], [[-0.7, 0, -0.05, 0.05]], [[0.7, 0, -0.05, 0.05]]]], device="cuda"
    )
    assert (contact.clearance(spheres[:, 1:]) > 0).all()
    assert contact.cost(spheres).sum() > 0.0


def test_rotated_support_does_not_assume_world_z() -> None:
    """A support normal along world X must allow separation in world X."""
    half_sqrt = 0.5**0.5
    scene = SceneCollision.from_config(
        SceneCollisionCfg(
            scene_model=SceneCfg(
                cuboid=[
                    Cuboid(
                        name="rotated",
                        dims=[1, 1, 0.1],
                        pose=[-0.05, 0, 0, half_sqrt, 0, half_sqrt, 0],
                    ),
                ]
            )
        )
    )
    initial = ((0.049, 0.0, 0.0, 0.05),)
    contact = ContactSeparation(StartContact("rotated", (0,), initial), scene, 1)
    spheres = torch.tensor([[[[x, 0, 0, 0.05]] for x in [0.049, 0.055, 0.06]]], device="cuda")
    assert contact.cost(spheres).sum() == 0.0
