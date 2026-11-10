# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Stationary contact must not authorize motion through a held object."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from curobo._src.collision.contact_separation import StartContact
from curobo._src.collision.contact_stationary import StationaryContactSet
from curobo._src.collision.contact_departure_set import ContactDepartureSet
from curobo._src.cost.cost_scene_collision import SceneCollisionCost
from curobo._src.cost.cost_scene_collision_cfg import SceneCollisionCostCfg
from curobo._src.geom.collision.collision_scene import SceneCollision, SceneCollisionCfg
from curobo._src.geom.types import Cuboid, SceneCfg


@pytest.fixture
def scene() -> SceneCollision:
    """Supply one contacted surface and an unrelated active obstacle."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    return SceneCollision.from_config(SceneCollisionCfg(scene_model=SceneCfg(cuboid=[
        Cuboid(name="support", dims=[1., 1., .1], pose=[0, 0, -.05, 1, 0, 0, 0]),
        Cuboid(name="other", dims=[1., 1., .1], pose=[0, 0, .2, 1, 0, 0, 0]),
    ])))


def contact(scene: SceneCollision, count: int = 1) -> StationaryContactSet:
    """Declare only sphere zero's initial one-millimeter proxy overlap."""
    return StationaryContactSet((StartContact("support", (0,), ((0., 0., .049, .05),)),), scene, count)


@pytest.mark.parametrize("horizon", [1, 3])
def test_unchanged_contact_is_allowed(scene: SceneCollision, horizon: int) -> None:
    """IK and trajectories share the same frozen-contact constraint."""
    spheres = torch.tensor([[[[0., 0., .049, .05]]] * horizon], device="cuda")
    assert not contact(scene).cost(spheres).any()


@pytest.mark.parametrize("axis,change", [(0, .001), (2, -.001), (2, .001), (3, .001)])
def test_any_contact_motion_or_radius_change_is_rejected(
    scene: SceneCollision, axis: int, change: float,
) -> None:
    """Neither tangential motion, deeper contact, departure nor resize is exempt."""
    spheres = torch.tensor([[[[0., 0., .049, .05]]] * 3], device="cuda")
    spheres[0, 1, 0, axis] += change
    assert contact(scene).cost(spheres).max() > 0


def test_deep_contact_is_rejected(scene: SceneCollision) -> None:
    """Keep the existing bounded initial-penetration admission rule."""
    declaration = StartContact("support", (0,), ((0., 0., .04, .05),))
    with pytest.raises(ValueError, match="initial penetration"):
        StationaryContactSet((declaration,), scene, 1)


@pytest.mark.parametrize("sweep", [False, True])
def test_unselected_pairs_remain_checked(scene: SceneCollision, sweep: bool) -> None:
    """A frozen holding contact must not hide another sphere hitting the support."""
    cfg = SceneCollisionCostCfg(weight=1., num_spheres=2, use_sweep=sweep)
    cfg.scene_collision_checker = scene
    cost = SceneCollisionCost(cfg)
    cost._contact = contact(scene, 2)
    cost.setup_batch_tensors(1, 3)
    spheres = torch.tensor([[[0., 0., .049, .05], [.1, 0., .04, .05]]] * 3, device="cuda").unsqueeze(0)
    result = cost.forward(SimpleNamespace(robot_spheres=spheres), trajectory_dt=torch.tensor([.01], device="cuda"))
    assert result.max() > 0


def test_cuda_replay_rechecks_stationarity(scene: SceneCollision) -> None:
    """Captured execution must detect a changed input, not cache acceptance."""
    constraint = contact(scene)
    spheres = torch.tensor([[[[0., 0., .049, .05]]] * 3], device="cuda")
    for _ in range(3):
        constraint.cost(spheres)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = constraint.cost(spheres)
    graph.replay()
    assert not result.any()
    spheres[0, 1, 0, 0] += .001
    graph.replay()
    assert result.max() > 0


@pytest.mark.parametrize('violation', ['none', 'held_moves', 'moves_inward', 'no_release'])
def test_combined_stationary_and_departure_keep_independent_guards(
    scene: SceneCollision, violation: str,
) -> None:
    """Only the departing sphere may move, without deepening or remaining in contact."""
    held = contact(scene, 2)
    departure = StartContact('support', (1,), ((.1, 0., .049, .05),))
    constraint = ContactDepartureSet((departure,), scene, 2, stationary=held)
    spheres = torch.tensor([[[[0., 0., .049, .05], [.1, 0., z, .05]]
                              for z in [.049, .055, .06]]], device='cuda')
    if violation == 'held_moves':
        spheres[0, 1, 0, 0] += .001
    elif violation == 'moves_inward':
        spheres[0, 1, 1, 2] = .048
    elif violation == 'no_release':
        spheres[0, :, 1, 2] = .049
    assert bool(constraint.cost(spheres).any()) == (violation != 'none')


@pytest.mark.parametrize('sweep', [False, True])
def test_combined_contact_does_not_hide_other_obstacles(scene: SceneCollision, sweep: bool) -> None:
    """The full native cost still sees uncaptured sphere/obstacle pairs."""
    cfg = SceneCollisionCostCfg(weight=1., num_spheres=2, use_sweep=sweep)
    cfg.scene_collision_checker = scene
    cost = SceneCollisionCost(cfg)
    cost._contact = ContactDepartureSet(
        (StartContact('support', (1,), ((.1, 0., .049, .05),)),),
        scene, 2, stationary=contact(scene, 2))
    cost.setup_batch_tensors(1, 3)
    spheres = torch.tensor([[[[0., 0., .049, .05], [.1, 0., z, .05]]
                              for z in [.049, .1, .17]]], device='cuda')
    result = cost.forward(SimpleNamespace(robot_spheres=spheres),
                          trajectory_dt=torch.tensor([.01], device='cuda'))
    assert result.max() > 0


def test_combined_deep_or_overlapping_declarations_rejected(scene: SceneCollision) -> None:
    """Composition does not broaden depth admission or replace a held sphere twice."""
    with pytest.raises(ValueError, match='initial penetration'):
        ContactDepartureSet((StartContact('support', (1,), ((.1, 0., .04, .05),)),),
                            scene, 2, stationary=contact(scene, 2))
    with pytest.raises(ValueError, match='disjoint'):
        ContactDepartureSet((StartContact('support', (0,), ((0., 0., .049, .05),)),),
                            scene, 2, stationary=contact(scene, 2))


def test_combined_cuda_replay_rechecks_both_guards(scene: SceneCollision) -> None:
    """Neither holding nor departure may use stale captured acceptance."""
    constraint = ContactDepartureSet(
        (StartContact('support', (1,), ((.1, 0., .049, .05),)),),
        scene, 2, stationary=contact(scene, 2))
    spheres = torch.tensor([[[[0., 0., .049, .05], [.1, 0., z, .05]]
                              for z in [.049, .055, .06]]], device='cuda')
    for _ in range(3):
        constraint.cost(spheres)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = constraint.cost(spheres)
    graph.replay()
    assert not result.any()
    spheres[0, 1, 0, 0] += .001
    graph.replay()
    assert result.max() > 0
    spheres[0, 1, 0, 0] -= .001
    spheres[0, 1, 1, 2] = .048
    graph.replay()
    assert result.max() > 0
