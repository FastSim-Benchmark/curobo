# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for bounded terminal contact and explicit goal-state checks."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from curobo._src.collision.contact_approach import ContactApproach, GoalContact
from curobo._src.collision.contact_separation import ContactSeparation, StartContact
from curobo._src.cost.cost_scene_collision import SceneCollisionCost
from curobo._src.cost.cost_scene_collision_cfg import SceneCollisionCostCfg
from curobo._src.geom.collision.collision_scene import SceneCollision, SceneCollisionCfg
from curobo._src.geom.types import Cuboid, SceneCfg


@pytest.fixture
def scene() -> SceneCollision:
    """Keep the support and a ceiling active throughout placement."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    return SceneCollision.from_config(
        SceneCollisionCfg(
            scene_model=SceneCfg(
                cuboid=[
                    Cuboid(name="table", dims=[1, 1, 0.1], pose=[0, 0, -0.05, 1, 0, 0, 0]),
                    Cuboid(name="ceiling", dims=[1, 1, 0.1], pose=[0, 0, 0.2, 1, 0, 0, 0]),
                ]
            )
        )
    )


def declaration(terminal_only: bool = False) -> GoalContact:
    """Capture a final sphere with one millimeter of model overlap."""
    return GoalContact("table", (0,), ((0, 0, 0.049, 0.05),), terminal_only=terminal_only)


def trajectory(heights: list[float]) -> torch.Tensor:
    """Build fixed-radius sphere poses with shape (1, horizon, 1, 4)."""
    return torch.tensor([[[[0, 0, z, 0.05]] for z in heights]], device="cuda")


@pytest.mark.parametrize(
    "heights,accepted",
    [
        ([0.07, 0.055, 0.049], True),
        ([0.07, 0.049, 0.049], True),
        ([0.07, 0.056, 0.08, 0.049], True),
        ([0.07, 0.047, 0.049], False),
        ([0.07, 0.05, 0.06, 0.049], False),
        ([0.049, 0.055, 0.049], False),
        ([0.07, 0.06, 0.051], False),
    ],
)
def test_placement_rules(scene: SceneCollision, heights: list[float], accepted: bool) -> None:
    """Accept final approach; reject overshoot, rebound, or a mismatched endpoint."""
    cost = ContactApproach(declaration(), scene, 1).cost(trajectory(heights))
    assert bool((cost == 0).all()) is accepted


@pytest.mark.parametrize("terminal_only", [False, True])
def test_terminal_validation_requires_explicit_mode(
    scene: SceneCollision, terminal_only: bool
) -> None:
    """A terminal-only IK exemption must never silently validate a trajectory."""
    contact = ContactApproach(declaration(terminal_only), scene, 1)
    wrong_input = trajectory([0.07, 0.049] if terminal_only else [0.049])
    with pytest.raises(ValueError, match="trajectory|single-state"):
        contact.cost(wrong_input)


@pytest.mark.parametrize("height,accepted", [(0.049, True), (0.04, False), (0.07, False)])
def test_goal_state_collision_rules(scene: SceneCollision, height: float, accepted: bool) -> None:
    """IK must reach the declared contact without treating deeper overlap as valid."""
    cost = ContactApproach(declaration(True), scene, 1).cost(trajectory([height]))
    assert bool((cost == 0).all()) is accepted


@pytest.mark.parametrize("sweep", [False, True])
def test_placement_preserves_other_collision_pairs(scene: SceneCollision, sweep: bool) -> None:
    """The payload still hits the ceiling and the gripper still hits the table."""
    cfg = SceneCollisionCostCfg(
        weight=1.0, num_spheres=2, use_sweep=sweep, goal_contact=declaration()
    )
    cfg.scene_collision_checker = scene
    cost = SceneCollisionCost(cfg)
    cost.setup_batch_tensors(1, 3)
    payload = trajectory([0.16, 0.06, 0.049])
    gripper = trajectory([0.07, 0.06, 0.04])
    state = SimpleNamespace(robot_spheres=torch.cat((payload, gripper), dim=2))
    result = cost.forward(state, trajectory_dt=torch.tensor([0.01], device="cuda"))
    assert result[0, 0, 0] > 0
    assert result[0, -1, 0] == 0
    assert result[0, -1, 1] > 0
    assert scene.data.cuboids.enable[0, :2].tolist() == [1, 1]


def test_placement_gradient_rejects_overshoot(scene: SceneCollision) -> None:
    """An inward waypoint must receive an outward corrective gradient."""
    spheres = trajectory([0.07, 0.047, 0.049]).requires_grad_(True)
    ContactApproach(declaration(), scene, 1).cost(spheres).sum().backward()
    assert torch.isfinite(spheres.grad).all()
    assert spheres.grad[0, 1, 0, 2] < 0


@pytest.mark.parametrize("terminal_only", [False, True])
def test_changed_support_invalidates_placement(scene: SceneCollision, terminal_only: bool) -> None:
    """Both trajectory and IK validation detect a changed support."""
    contact = ContactApproach(declaration(terminal_only), scene, 1)
    spheres = trajectory([0.049] if terminal_only else [0.07, 0.049])
    assert contact.cost(spheres).sum() == 0
    scene.data.cuboids.dims[0, 0, 2] += 0.01
    assert contact.cost(spheres).sum() > 0


def test_deep_goal_contact_is_rejected(scene: SceneCollision) -> None:
    """A place declaration cannot allow an arbitrarily colliding goal."""
    deep = GoalContact("table", (0,), ((0, 0, 0.04, 0.05),))
    with pytest.raises(ValueError, match="penetration"):
        ContactApproach(deep, scene, 1)


@pytest.mark.parametrize("heights,accepted", [
    ([0.049, 0.06, 0.049], True),
    ([0.049, 0.049, 0.049], False),
    ([0.049, 0.047, 0.06, 0.049], False),
    ([0.049, 0.06, 0.049, 0.06, 0.049], False),
    ([0.049, 0.06, 0.047, 0.049], False),
    ([0.049, 0.16, 0.049], False),
])
@pytest.mark.parametrize("sweep", [False, True])
def test_combined_endpoint_contacts(
    scene: SceneCollision, heights: list[float], accepted: bool, sweep: bool
) -> None:
    """A transfer must clear the support and may recontact it only at the end."""
    cfg = SceneCollisionCostCfg(
        weight=1.0,
        num_spheres=1,
        use_sweep=sweep,
        start_contact=StartContact("table", (0,), ((0, 0, 0.049, 0.05),)),
        goal_contact=declaration(),
    )
    cfg.scene_collision_checker = scene
    cost = SceneCollisionCost(cfg)
    cost.setup_batch_tensors(1, len(heights))
    result = cost.forward(
        SimpleNamespace(robot_spheres=trajectory(heights)),
        trajectory_dt=torch.tensor([0.01], device="cuda"),
    )
    assert bool((result == 0).all()) is accepted


def test_contact_clearance_preserves_float32_precision(
    scene: SceneCollision, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TF32 matmul settings must not erase micrometer-scale contact changes."""
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    contact = ContactSeparation(declaration().as_departure(), scene, 1)
    spheres = trajectory([0.048994 + i * 5e-8 for i in range(301)])
    gap = contact.clearance(spheres)
    expected = spheres[..., 2] - spheres[..., 3]
    assert torch.allclose(gap, expected, atol=2e-8, rtol=0)


@pytest.mark.parametrize("sweep", [False, True])
def test_transfer_between_distinct_supports_keeps_other_pairs_checked(sweep: bool) -> None:
    """One sphere may depart one cuboid and arrive at another in the same query."""
    world = SceneCollision.from_config(SceneCollisionCfg(scene_model=SceneCfg(cuboid=[
        Cuboid(name="left", dims=[0.4, 0.4, 0.1], pose=[-0.4, 0, -0.05, 1, 0, 0, 0]),
        Cuboid(name="right", dims=[0.4, 0.4, 0.1], pose=[0.4, 0, -0.05, 1, 0, 0, 0]),
    ])))
    cfg = SceneCollisionCostCfg(
        weight=1.0, num_spheres=2, use_sweep=sweep,
        start_contact=StartContact("left", (0,), ((-0.4, 0, 0.049, 0.05),)),
        goal_contact=GoalContact("right", (0,), ((0.4, 0, 0.049, 0.05),)),
    )
    cfg.scene_collision_checker = world
    cost = SceneCollisionCost(cfg)
    points = [(-0.4, 0.049), (-0.4, 0.07), (0.0, 0.09), (0.4, 0.07), (0.4, 0.049)]
    payload = torch.tensor([[[x, 0, z, 0.05]] for x, z in points], device="cuda")[None]
    gripper = payload.clone()
    gripper[..., 2] += 0.1
    spheres = torch.cat((payload, gripper), dim=2)
    cost.setup_batch_tensors(1, len(points))
    dt = torch.tensor([0.01], device="cuda")
    assert cost.forward(SimpleNamespace(robot_spheres=spheres), trajectory_dt=dt).sum() == 0
    spheres[0, -1, 1, 2] = 0.04
    assert cost.forward(SimpleNamespace(robot_spheres=spheres), trajectory_dt=dt)[0, -1, 1] > 0


def test_transfer_gradient_and_changed_support(scene: SceneCollision) -> None:
    """An inward move receives a corrective gradient; changed geometry invalidates capture."""
    from curobo._src.collision.contact_transfer import ContactTransfer

    contact = ContactTransfer(
        StartContact("table", (0,), ((0, 0, 0.049, 0.05),)), declaration(), scene, 1
    )
    spheres = trajectory([0.049, 0.047, 0.06, 0.049]).requires_grad_(True)
    contact.cost(spheres).sum().backward()
    assert torch.isfinite(spheres.grad).all()
    assert spheres.grad[0, 1, 0, 2] < 0
    assert contact.cost(trajectory([0.049, 0.06, 0.049])).sum() == 0
    scene.data.cuboids.dims[0, 0, 2] += 0.01
    assert contact.cost(trajectory([0.049, 0.06, 0.049])).sum() > 0


@pytest.mark.parametrize("placement", [False, True])
def test_micrometer_rebound_is_rejected_with_tf32_enabled(
    scene: SceneCollision, placement: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject a 15 micrometer rebound under the declared 10 micrometer tolerance."""
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    heights = torch.cat(
        (
            torch.linspace(0.07, 0.048994, 200),
            torch.linspace(0.048994, 0.049009, 50),
            torch.linspace(0.049009, 0.049, 51),
        )
    ).tolist()
    spheres = trajectory(heights)
    if placement:
        cost = ContactApproach(declaration(), scene, 1).cost(spheres)
    else:
        cost = ContactSeparation(declaration().as_departure(), scene, 1).cost(spheres.flip(1))
    assert cost.max() > 0
