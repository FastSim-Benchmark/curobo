"""Contact seed goal pairing and rejected-result contracts."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from curobo._src.motion import contact_axis_seeds as module

p = Path(__file__).with_name("test_axis_seeds.py")
spec = importlib.util.spec_from_file_location("contact_candidate_base", p)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


@pytest.mark.parametrize("outcome", ["success", "rejected", "none", "exception"])
def test_pair_identity_remapping_and_no_false_acceptance(monkeypatch, outcome):
    planner, goals, current, *_ = base.fixture()
    observed = []

    def plan(p, target, start, holds, implicit, *, diagnostics):
        assert p is planner and start is current and implicit is False
        assert target.num_goalset == 1
        assert torch.equal(target.position, goals.position[..., 1:2, :])
        observed.append(target)
        if outcome == "exception":
            raise TimeoutError("deadline")
        if outcome == "none":
            return None
        diagnostics["seed_record"] = {"ik_solve_time": 0.03}
        return NS(
            success=torch.tensor([outcome == "success"]),
            goalset_index=torch.tensor([[[0]]]),
            total_time=0.01,
            solve_time=0.02,
            debug_info={"native": True},
        )

    monkeypatch.setattr(module, "plan_contact_continuation", plan)
    if outcome == "exception":
        with pytest.raises(TimeoutError):
            module.plan_contact_axis_seed(planner, goals, current, {"tool": object()}, False)
    else:
        result = module.plan_contact_axis_seed(planner, goals, current, {"tool": object()}, False)
        if outcome == "none":
            assert result is None
        else:
            assert result.success.item() == (outcome == "success")
            assert result.goalset_index.item() == 1
            assert result.solve_time == pytest.approx(0.05)
            assert result.debug_info["native"]
    assert len(observed) == 1


def test_axis_hold_required():
    planner, goals, current, *_ = base.fixture()
    with pytest.raises(ValueError, match="requires an axis-hold"):
        module.plan_contact_axis_seed(planner, goals, current, {})
