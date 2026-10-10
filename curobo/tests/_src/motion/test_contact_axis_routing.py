"""Exercise actual public planner routing with CPU solver doubles."""

from contextlib import nullcontext
from types import SimpleNamespace as NS

import pytest
import torch

from curobo._src.motion import contact_axis_seeds


@pytest.mark.parametrize("mode", ["none", "start", "end", "both"])
@pytest.mark.parametrize("held", [False, True])
@pytest.mark.parametrize("attempts", [0, 1])
@pytest.mark.parametrize("outcome", ["success", "rejected", "none", "timeout"])
def test_contact_routing_preserves_existing_fallback(monkeypatch, mode, held, attempts, outcome):
    import curobo._src.motion.axis_seeds as ordinary
    import curobo._src.motion.motion_contact as contact
    import curobo._src.motion.motion_contact_goalset as terminal
    import curobo._src.motion.motion_planner as production

    calls = []
    fallback = object()
    result = NS(success=torch.tensor([outcome == "success"]))

    def candidate(*args):
        calls.append("contact")
        if outcome == "timeout":
            raise TimeoutError("deadline")
        return None if outcome == "none" else result

    monkeypatch.setattr(contact_axis_seeds, "plan_contact_axis_seed", candidate)
    monkeypatch.setattr(ordinary, "plan_continuous_axis_seed", lambda *a: None)
    monkeypatch.setattr(contact, "boundary_contact_scope", lambda *a: nullcontext())

    def legacy(*a):
        calls.append("terminal")
        return fallback

    monkeypatch.setattr(terminal, "plan_terminal_pose", legacy)
    planner = NS(_hold_axis_scope=lambda *a: nullcontext(), _plan_pose=lambda *a: fallback)
    eligible = mode == "end" and held and attempts > 0
    kwargs = dict(
        hold_axis={"tool": object()} if held else None,
        allow_boundary_collision=mode,
        max_attempts=attempts,
    )
    if eligible and outcome == "timeout":
        with pytest.raises(TimeoutError):
            production.MotionPlanner.plan_pose(planner, NS(), NS(), **kwargs)
        assert calls == ["contact"]
    else:
        actual = production.MotionPlanner.plan_pose(planner, NS(), NS(), **kwargs)
        assert actual is (result if eligible and outcome == "success" else fallback)
        assert ("contact" in calls) == eligible
        assert ("terminal" in calls) == (
            mode in ("end", "both") and not (eligible and outcome == "success")
        )
