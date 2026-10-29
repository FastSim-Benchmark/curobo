# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Complete joint targets retain exact c-space endpoints and native rejection."""

from types import SimpleNamespace

import pytest
import torch

from curobo.motion_planner import MotionPlanner
from curobo.types import JointState


def case(delta=0.0, native_success=True):
    names = ["a", "b"]
    start = JointState.from_position(torch.tensor([[0.0, 0.0]]), joint_names=names)
    goal = JointState.from_position(torch.tensor([[0.2, -0.5]]), joint_names=names)
    path = JointState.from_position(
        torch.tensor([[[0.0, 0.0], [0.2 + delta, -0.5]]]), joint_names=names
    )
    result = SimpleNamespace(
        success=torch.tensor([native_success]),
        debug_info={"native": "preserved"},
        goalset_index=torch.tensor([5]),
        get_interpolated_plan=lambda: path,
    )
    calls = []
    planner = SimpleNamespace(
        joint_names=names,
        plan_cspace=lambda *args, **kwargs: calls.append((args, kwargs)) or result,
    )
    return planner, start, goal, result, calls


def test_complete_target_skips_ik_and_preserves_axis_budget_and_endpoint_contract():
    planner, start, goal, result, calls = case()
    axis = {"tcp": object()}
    got = MotionPlanner.plan_posture(
        planner, goal, start, tolerance=1.0e-5, max_attempts=3, hold_axis=axis, timeout_s=7.0
    )
    assert got is result and got.success.all()
    assert len(calls) == 1 and calls[0][0][1] is start
    assert torch.equal(calls[0][0][0].position, goal.position)
    assert calls[0][1] == {
        "max_attempts": 3,
        "hold_axis": axis,
        "allow_boundary_collision": "none",
        "timeout_s": 7.0,
    }
    assert got.goalset_index.tolist() == [0]
    assert got.debug_info["native"] == "preserved"
    assert got.debug_info["posture"]["fully_specified_cspace"]


@pytest.mark.parametrize("delta", [0.01018, float("nan")])
def test_native_success_cannot_relax_posture_endpoint_tolerance(delta):
    planner, start, goal, result, _ = case(delta)
    assert not MotionPlanner.plan_posture(planner, goal, start, tolerance=1.0e-5).success.any()
    assert result.debug_info["posture"]["tolerance"] == 1.0e-5


def test_native_failed_path_is_never_promoted_or_read_as_valid_geometry():
    planner, start, goal, result, _ = case(native_success=False)

    def no_path():
        raise AssertionError("failed native path is not a trajectory")

    result.get_interpolated_plan = no_path
    assert MotionPlanner.plan_posture(planner, goal, start) is result
    assert not result.success.any()


@pytest.mark.parametrize(
    "options",
    [
        {"free_joints": ("a",)},
        {"held_joints": ("b",)},
        {"allow_boundary_collision": "start"},
        {"multiple": True},
    ],
)
def test_partial_held_contact_and_multiple_goals_keep_existing_posture_owner(monkeypatch, options):
    import curobo._src.motion.motion_posture as module

    planner, start, goal, _, calls = case()
    options = dict(options)
    if options.pop("multiple", False):
        goal = JointState.from_position(goal.position.repeat(2, 1), joint_names=goal.joint_names)
    seen = []
    marker = object()
    monkeypatch.setattr(
        module, "plan_posture", lambda *args, **kwargs: seen.append((args, kwargs)) or marker
    )
    assert MotionPlanner.plan_posture(planner, goal, start, **options) is marker
    assert not calls and len(seen) == 1


@pytest.mark.parametrize(
    "options",
    [
        {"tolerance": 0.0},
        {"tolerance": float("nan")},
        {"tolerance": True},
        {"max_initial_penetration": 0.0},
    ],
)
def test_complete_target_does_not_bypass_existing_argument_validation(options):
    planner, start, goal, _, calls = case()
    with pytest.raises(ValueError):
        MotionPlanner.plan_posture(planner, goal, start, **options)
    assert not calls
