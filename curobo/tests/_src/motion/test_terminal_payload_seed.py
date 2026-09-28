# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Keep body/world collision guidance while proposing payload contact endpoints."""

from types import SimpleNamespace

import pytest
import torch

from curobo._src.motion import motion_contact as module
from curobo.types import JointState


def fixture(monkeypatch, *, invalid_first=False, broken=False):
    spheres = torch.tensor([[[0., 0., 0., .1], [1., 2., 3., .05], [0., 0., 0., -100.]]])
    original = spheres.clone()
    calls = []
    cost = SimpleNamespace(enabled=True, _weight=torch.ones(1))
    cost.disable_cost = lambda: setattr(cost, "enabled", False)
    cost.enable_cost = lambda: setattr(cost, "enabled", True)
    core = SimpleNamespace(invalidate_parameter_graphs=lambda: calls.append("invalidate"))
    manager = SimpleNamespace(
        _attached_link_name="payload",
        kinematics_params=SimpleNamespace(
            link_spheres=spheres,
            get_sphere_index_from_link_name=lambda name: torch.tensor([1, 2]),
        ),
    )

    def solve(*args, **kwargs):
        assert cost.enabled, "body/environment constraints must remain enabled"
        assert spheres[0, 0, 3] == original[0, 0, 3]
        assert bool((spheres[0, 1:, 3] < 0).all())
        calls.append("solve")
        if broken:
            raise RuntimeError("native failure")
        return SimpleNamespace(solution=torch.tensor([[[0.], [1.]]]),
                               success=torch.tensor([[True, True]]))

    def capture(planner, state, *, terminal):
        torch.testing.assert_close(spheres, original, rtol=0, atol=0)
        calls.append("capture")
        if invalid_first and float(state.position[0, 0]) == 0:
            raise module._ContactPenetrationError("payload exceeds 2mm")

    planner = SimpleNamespace(attachment_manager=manager, joint_names=["q"],
                              ik_solver=SimpleNamespace(core=core,
                                  config=SimpleNamespace(num_seeds=2), solve_pose=solve))
    monkeypatch.setattr(module, "capture_contact", capture)
    monkeypatch.setattr(module, "scene_costs", lambda solver: [cost])
    return planner, spheres, original, calls


@pytest.mark.parametrize("invalid_first", [False, True])
def test_body_guided_seed_restores_payload_before_capture_and_yield(monkeypatch, invalid_first):
    planner, spheres, original, calls = fixture(monkeypatch, invalid_first=invalid_first)
    current = JointState.from_position(torch.zeros(1, 1))
    iterator = module.contact_goal_candidates(planner, object(), current)
    result = next(iterator)
    assert float(result.position[0, 0]) == float(invalid_first)
    torch.testing.assert_close(spheres, original, rtol=0, atol=0)
    iterator.close()
    assert calls.count("solve") == 1
    assert calls.count("invalidate") == 2


def test_native_error_restores_geometry_and_propagates(monkeypatch):
    planner, spheres, original, calls = fixture(monkeypatch, broken=True)
    current = JointState.from_position(torch.zeros(1, 1))
    with pytest.raises(RuntimeError, match="native failure"):
        next(module.contact_goal_candidates(planner, object(), current))
    torch.testing.assert_close(spheres, original, rtol=0, atol=0)
    assert calls.count("invalidate") == 2


def test_all_restored_payload_penetrations_rejected(monkeypatch):
    planner, spheres, original, _ = fixture(monkeypatch)

    def deep(*args, **kwargs):
        torch.testing.assert_close(spheres, original, rtol=0, atol=0)
        raise module._ContactPenetrationError("deep contact")

    monkeypatch.setattr(module, "capture_contact", deep)
    current = JointState.from_position(torch.zeros(1, 1))
    assert list(module._payload_goal_candidates(planner, object(), current)) == []
    torch.testing.assert_close(spheres, original, rtol=0, atol=0)


def test_scene_proposal_failure_retains_bounded_kinematic_pass(monkeypatch):
    planner, spheres, original, calls = fixture(monkeypatch)
    cost = module.scene_costs(planner.ik_solver)[0]

    def solve(*args, **kwargs):
        calls.append("solve")
        if cost.enabled:
            assert bool((spheres[0, 1:, 3] < 0).all())
        else:
            torch.testing.assert_close(spheres, original, rtol=0, atol=0)
        return SimpleNamespace(solution=torch.tensor([[[1.]]]),
                               success=torch.tensor([[not cost.enabled]]))

    planner.ik_solver.solve_pose = solve
    current = JointState.from_position(torch.zeros(1, 1))
    assert float(module.contact_goal_seed(planner, object(), current).position[0, 0]) == 1.
    assert calls.count("solve") == 2 and cost.enabled
    torch.testing.assert_close(spheres, original, rtol=0, atol=0)
