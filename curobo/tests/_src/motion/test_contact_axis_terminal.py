"""Candidate local-bound, equality, and restoration regression tests."""

import importlib.util
from pathlib import Path

import pytest
import torch

from curobo._src.motion import axis_seeds as module
from curobo._src.state.state_joint import JointState

_spec = importlib.util.spec_from_file_location(
    "contact_axis_test_base", Path(__file__).with_name("test_axis_seeds.py")
)
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)


@pytest.mark.parametrize("outcome", ["success", "reject", "raise", "outside", "nonfinite"])
def test_terminal_proposal_is_local_and_restores_bounds(outcome):
    planner, goals, start, active, original, calls = base.fixture()
    observed = []

    def terminal(p, target, previous):
        assert p is planner and len(calls) == 15
        assert torch.equal(active[:, 1], original[:, 1])
        assert (active[1, 0] - active[0, 0]).item() == pytest.approx(0.04)
        assert target.position.flatten()[0].item() == pytest.approx(0.16)
        observed.append(True)
        if outcome == "reject":
            return None
        if outcome == "raise":
            raise RuntimeError("proposal rejected")
        value = 0.16 if outcome == "success" else 0.3 if outcome == "outside" else float("nan")
        return JointState.from_position(torch.tensor([[value, 0.0]]), planner.joint_names)

    if outcome in ("raise", "outside", "nonfinite"):
        with pytest.raises(RuntimeError if outcome == "raise" else ValueError):
            module.continuous_axis_seed(
                planner, goals, start, {"tool": object()}, terminal_proposal=terminal
            )
    else:
        result = module.continuous_axis_seed(
            planner, goals, start, {"tool": object()}, terminal_proposal=terminal
        )
        if outcome == "reject":
            assert result is None
        else:
            _, endpoint, seeds, record = result
            assert record["original_goal_index"] == 1
            assert torch.equal(seeds[0, 0, -1], endpoint)
            assert not seeds[..., 1].any()
    assert observed == [True]
    assert torch.equal(active, original)


def test_no_terminal_override_retains_original_sixteen_steps():
    planner, goals, start, active, original, calls = base.fixture()
    result = module.continuous_axis_seed(planner, goals, start, {"tool": object()})
    assert result is not None and len(calls) == 16
    assert torch.equal(active, original)


def test_endpoint_corridor_rejects_native_candidate_outside_remaining_reach():
    from types import SimpleNamespace as NS

    planner, goals, start, active, original, calls = base.fixture()
    original[:, 1] = torch.tensor([-3.0, 3.0])
    active.copy_(original)
    endpoint = JointState.from_position(torch.tensor([[0.16, 0.0]]), planner.joint_names)
    seen = []

    def solve(target, **kwargs):
        step = len(seen) + 1
        seen.append(active.clone())
        assert active[0, 1] >= -(16 - step) * 0.15
        assert active[1, 1] <= (16 - step) * 0.15
        # A successful native flag cannot override the explicit corridor.
        solution = torch.tensor(
            [[[target.position.flatten()[0], 2.9], [target.position.flatten()[0], 0.0]]]
        )
        return NS(
            solution=solution,
            success=torch.ones(1, 2, dtype=torch.bool),
            total_time=0.0,
            solve_time=0.0,
        )

    planner.ik_solver.solve_pose = solve
    result = module.continuous_axis_seed(
        planner,
        goals,
        start,
        {"tool": object()},
        terminal_hint=endpoint,
        terminal_proposal=lambda *args: endpoint,
    )
    assert result is not None and len(seen) == 15
    assert not result[2][..., 1].any()
    assert torch.equal(result[1], endpoint.position[0])
    assert torch.equal(active, original)


@pytest.mark.parametrize("value", [float("nan"), 0.4])
def test_invalid_endpoint_hint_is_rejected_without_mutating_bounds(value):
    planner, goals, start, active, original, calls = base.fixture()
    hint = JointState.from_position(torch.tensor([[value, 0.0]]), planner.joint_names)
    with pytest.raises(ValueError, match="Invalid terminal hint"):
        module.continuous_axis_seed(planner, goals, start, {"tool": object()}, terminal_hint=hint)
    assert not calls and torch.equal(active, original)
