"""CPU identity/lifecycle tests, not physical contact validation."""

import importlib.util
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from curobo._src.motion import contact_continuation as module
from curobo._src.motion.motion_goalset import goal_group
from curobo._src.state.state_joint import JointState
from curobo._src.types.tool_pose import ToolPose

_spec = importlib.util.spec_from_file_location(
    "contact_axis_test_base", Path(__file__).with_name("test_axis_seeds.py")
)
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)


@pytest.mark.parametrize(
    "failure", [None, "proposal", "prefix", "endpoint", "trajectory", "exception"]
)
def test_exact_endpoint_contact_and_restoration(monkeypatch, failure):
    planner, goals, current, active, bounds, calls = base.fixture(
        fail_at=8 if failure == "prefix" else None
    )
    target = goal_group(goals, 1)
    target.position.mul_(0.1)
    flags = {"scope": False, "closed": False, "trajectory": False}
    endpoint = JointState.from_position(torch.tensor([[0.016, 0.0]]), planner.joint_names)

    def fk(state):
        positions = torch.zeros(1, 1, 1, 3)
        positions[..., 0] = state.position[0, 0]
        return NS(
            tool_poses=ToolPose(["tool"], positions, torch.tensor([[[[1.0, 0.0, 0.0, 0.0]]]]))
        )

    planner.compute_kinematics = fk

    def candidates(*args):
        try:
            assert active[1, 0].item() == pytest.approx(0.02)
            if failure == "exception":
                raise RuntimeError("proposal failed")
            if failure != "proposal":
                yield endpoint
        finally:
            flags["closed"] = True

    @contextmanager
    def scope(p, start, captured, mode):
        assert flags["closed"] and torch.equal(active, bounds)
        assert mode == "end" and torch.equal(captured.position, endpoint.position)
        flags["scope"] = True
        try:
            yield
        finally:
            flags["scope"] = False

    original_solve = planner.ik_solver.solve_pose

    def ik(target, **options):
        if options.get("run_optimizer") is False:
            assert flags["scope"]
            assert torch.equal(active, bounds)
            assert torch.equal(options["seed_config"][0, 0], endpoint.position[0])
            return NS(success=torch.tensor([failure != "endpoint"]))
        assert not flags["scope"]
        return original_solve(target, **options)

    def trajectory(*args, **options):
        flags["trajectory"] = True
        assert args[0] is target
        assert flags["scope"] and torch.equal(active, bounds)
        assert torch.equal(options["seed_traj"][0, 0, -1], endpoint.position[0])
        assert torch.equal(options["seed_config"][0, 0], endpoint.position[0])
        return NS(success=torch.tensor([failure != "trajectory"]))

    monkeypatch.setattr(module, "contact_goal_candidates", candidates)
    monkeypatch.setattr(module, "boundary_contact_scope", scope)
    planner.ik_solver.solve_pose = ik
    planner.trajopt_solver.solve_pose = trajectory
    if failure == "exception":
        with pytest.raises(RuntimeError, match="proposal failed"):
            module.plan_contact_continuation(planner, target, current, {"tool": object()})
    else:
        diagnostics = {}
        result = module.plan_contact_continuation(
            planner, target, current, {"tool": object()}, diagnostics=diagnostics
        )
        if failure in ("proposal", "endpoint"):
            assert result is None and not flags["trajectory"]
        else:
            assert result.success.item() == (failure in (None, "prefix"))
        if failure == "prefix":
            proposal = diagnostics["seed_record"]["terminal_tail_seed_proposal"]
            assert proposal["from_ik_step"] == 8
            assert proposal["reason"] == "native_ik_no_valid_seed"
    assert flags["closed"] and not flags["scope"]
    assert torch.equal(active, bounds)


def test_endpoint_iterator_close_failure_still_restores_limits(monkeypatch):
    planner, goals, current, active, bounds, _ = base.fixture()
    endpoint = JointState.from_position(torch.tensor([[0.016, 0.0]]), planner.joint_names)

    def candidates(*args):
        try:
            yield endpoint
        finally:
            raise RuntimeError("contact cleanup failed")

    monkeypatch.setattr(module, "contact_goal_candidates", candidates)
    with pytest.raises(RuntimeError, match="contact cleanup failed"):
        module.plan_contact_continuation(
            planner, goal_group(goals, 1), current, {"tool": object()}
        )
    assert torch.equal(active, bounds)
