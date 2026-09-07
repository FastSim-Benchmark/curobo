# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Reject support contact before final lateral and orientation alignment."""

from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from curobo._src.collision.contact_approach import ContactApproach, GoalContact
from curobo._src.collision.contact_landing import NormalLanding
from curobo._src.cost.cost_scene_collision import SceneCollisionCost
from curobo._src.cost.cost_scene_collision_cfg import SceneCollisionCostCfg
from curobo._src.geom.collision.collision_scene import SceneCollision, SceneCollisionCfg
from curobo._src.geom.types import Cuboid, SceneCfg
from curobo.types import ToolPose


@pytest.fixture
def scene() -> SceneCollision:
    """Create a planar table under one attached sphere."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    return SceneCollision.from_config(
        SceneCollisionCfg(
            scene_model=SceneCfg(
                cuboid=[Cuboid(name="table", dims=[1, 1, 0.1], pose=[0, 0, -0.05, 1, 0, 0, 0])]
            )
        )
    )


def declaration() -> GoalContact:
    """Keep the user's final height, but require a normal final approach."""
    return GoalContact(
        "table",
        (0,),
        ((0, 0, 0.049, 0.05),),
        landing=NormalLanding(
            tool_frame="gripper",
            goal_position=(0, 0, 0.049),
            goal_quaternion=(1, 0, 0, 0),
            outward_normal=(0, 0, 1),
        ),
    )


def state(points: list[list[float]], yaw: list[float] | None = None) -> SimpleNamespace:
    """Build sphere and tool poses, including rotation invisible to a single sphere."""
    xyz = torch.tensor(points, device="cuda", dtype=torch.float32).reshape(1, -1, 1, 3)
    radius = torch.full_like(xyz[..., :1], 0.05)
    angles = torch.tensor(yaw or [0.0] * len(points), device="cuda") * 0.5
    quat = torch.stack((angles.cos(), angles * 0, angles * 0, angles.sin()), -1)
    return SimpleNamespace(
        robot_spheres=torch.cat((xyz, radius), -1),
        tool_poses=ToolPose(["gripper"], xyz, quat.reshape(1, -1, 1, 4)),
    )


def evaluate(scene: SceneCollision, contact: GoalContact, query: SimpleNamespace) -> torch.Tensor:
    """Evaluate the production collision cost with its contact replacement enabled."""
    cfg = SceneCollisionCostCfg(weight=1.0, num_spheres=1, goal_contact=contact)
    cfg.scene_collision_checker = scene
    cost = SceneCollisionCost(cfg)
    cost.setup_batch_tensors(1, query.robot_spheres.shape[1])
    return cost.forward(query)


@pytest.mark.parametrize("yaw", [None, [0.4, 0, 0]])
def test_aligned_normal_landing_is_accepted(
    scene: SceneCollision, yaw: list[float] | None
) -> None:
    """Allow transport and rotation above the band, then normal-only placement."""
    query = state([[-0.1, 0, 0.07], [0, 0, 0.057], [0, 0, 0.049]], yaw)
    assert evaluate(scene, declaration(), query).sum() == 0


@pytest.mark.parametrize(
    "points",
    [
        [[0, 0, 0.07], [-0.02, 0, 0.049], [0, 0, 0.049]],
        [[-0.1, 0, 0.07], [-0.01, 0, 0.054], [0, 0, 0.049]],
    ],
)
def test_contact_before_alignment_is_rejected(
    scene: SceneCollision, points: list[list[float]]
) -> None:
    """Reject both surface sliding and diagonal entry before the tool is aligned."""
    assert evaluate(scene, declaration(), state(points)).sum() > 0


def test_rotation_during_contact_is_rejected(scene: SceneCollision) -> None:
    """Sphere-center alignment alone must not allow the held object to rotate on the table."""
    query = state([[0, 0, 0.07], [0, 0, 0.049], [0, 0, 0.049]], [0, 0.2, 0])
    assert evaluate(scene, declaration(), query).sum() > 0


def test_equivalent_quaternion_signs_are_accepted(scene: SceneCollision) -> None:
    """Quaternion sign changes do not represent physical rotation."""
    query = state([[0, 0, 0.07], [0, 0, 0.057], [0, 0, 0.049]])
    query.tool_poses.quaternion[:, 1] *= -1
    assert evaluate(scene, declaration(), query).sum() == 0


def test_landing_without_tool_pose_is_rejected(scene: SceneCollision) -> None:
    """Do not silently downgrade a pose-constrained landing to sphere-only checking."""
    query = state([[0, 0, 0.07], [0, 0, 0.049]])
    del query.tool_poses
    with pytest.raises(ValueError, match="tool pose"):
        evaluate(scene, declaration(), query)


def test_wrong_support_normal_is_rejected(scene: SceneCollision) -> None:
    """The final motion must approach along the actual support normal."""
    contact = declaration()
    contact = replace(contact, landing=replace(contact.landing, outward_normal=(1, 0, 0)))
    with pytest.raises(ValueError, match="support normal"):
        evaluate(scene, contact, state([[0, 0, 0.07], [0, 0, 0.049]]))


def test_goal_ik_checks_landing_orientation(scene: SceneCollision) -> None:
    """The same landing target must constrain goal IK orientation."""
    contact = replace(declaration(), terminal_only=True)
    assert evaluate(scene, contact, state([[0, 0, 0.049]])).sum() == 0
    assert evaluate(scene, contact, state([[0, 0, 0.049]], [0.2])).sum() > 0


@pytest.mark.parametrize("scale, accepted", [(0.9, True), (1.1, False)])
@pytest.mark.parametrize("rotation", [False, True])
def test_alignment_tolerance_boundary(
    scene: SceneCollision, scale: float, accepted: bool, rotation: bool
) -> None:
    """Enforce the declared tolerances on contact motion, including orientation."""
    offset = 0 if rotation else scale * 1e-4
    yaw = [0, scale * 1e-3, 0] if rotation else None
    query = state([[0, 0, 0.07], [offset, 0, 0.049], [0, 0, 0.049]], yaw)
    assert bool(evaluate(scene, declaration(), query).sum() == 0) is accepted


def test_rotated_support_normal_is_supported(scene: SceneCollision) -> None:
    """Use the declared world normal instead of assuming every support is horizontal."""
    half_sqrt = math.sqrt(0.5)
    rotated_scene = SceneCollision.from_config(
        SceneCollisionCfg(
            scene_model=SceneCfg(
                cuboid=[
                    Cuboid(
                        name="table",
                        dims=[1, 1, 0.1],
                        pose=[0, 0.05, 0, half_sqrt, half_sqrt, 0, 0],
                    )
                ]
            )
        )
    )
    contact = declaration()
    contact = replace(
        contact,
        goal_spheres=((0, -0.049, 0, 0.05),),
        landing=replace(
            contact.landing,
            goal_position=(0, -0.049, 0),
            goal_quaternion=(half_sqrt, half_sqrt, 0, 0),
            outward_normal=(0, -1, 0),
        ),
    )
    query = state([[0, -0.07, 0], [0, -0.057, 0], [0, -0.049, 0]])
    query.tool_poses.quaternion[:] = torch.tensor(contact.landing.goal_quaternion, device="cuda")
    assert evaluate(rotated_scene, contact, query).sum() == 0


def test_alignment_gradients_are_finite(scene: SceneCollision) -> None:
    """Provide a useful correction gradient for a misaligned contacting pose."""
    query = state([[0, 0, 0.07], [0.001, 0, 0.049], [0, 0, 0.049]], [0, 0.2, 0])
    query.tool_poses.position.requires_grad_()
    query.tool_poses.quaternion.requires_grad_()
    evaluate(scene, declaration(), query).sum().backward()
    for values in (query.tool_poses.position, query.tool_poses.quaternion):
        assert torch.isfinite(values.grad).all()
        assert values.grad.abs().sum() > 0


def test_alignment_boundary_has_nonvanishing_clearance_gradient(scene: SceneCollision) -> None:
    """A misaligned near-boundary pose must be pushed fully outside the landing band."""
    query = state([[0.001, 0, 0.054999], [0.001, 0, 0.054999]])
    query.robot_spheres.requires_grad_()
    alignment = ContactApproach(declaration(), scene, 1).landing
    alignment.cost(query.robot_spheres, query.tool_poses, False).sum().backward()
    assert query.robot_spheres.grad[..., 2].sum() <= -1.9


@pytest.mark.parametrize("activation", [-0.005, 0.0, 0.005])
def test_optimizer_activation_guides_alignment_before_required_band(
    scene: SceneCollision, activation: float
) -> None:
    """Reuse collision activation distance for early guidance without changing validity."""
    query = state([[-0.1, 0, 0.07], [-0.002, 0, 0.056], [0, 0, 0.055], [0, 0, 0.049]])
    assert evaluate(scene, declaration(), query).sum() == 0
    cfg = SceneCollisionCostCfg(
        weight=1.0, num_spheres=1, goal_contact=declaration(), activation_distance=activation
    )
    cfg.scene_collision_checker = scene
    cost = SceneCollisionCost(cfg)
    cost.setup_batch_tensors(1, 4)
    assert bool(cost.forward(query).sum() > 0) is (activation > 0)
