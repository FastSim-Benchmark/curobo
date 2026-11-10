# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Failure evidence must retain actual raw terms and the request's active scope."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from curobo._src.cost.cost_cspace_type import CSpaceCostType
from curobo._src.motion.motion_planner import MotionPlanner
from curobo._src.solver.solver_trajopt import TrajOptSolver
from curobo._src.state.state_joint import JointState


@pytest.mark.parametrize("boundary", ["none", "start", "end", "both"])
@pytest.mark.parametrize("outcome", ["success", "failure", "none"])
def test_cspace_evidence_preserves_result_budgets_and_active_scope(
    monkeypatch, caplog, boundary, outcome
):
    scope = {"axis": False, "contact": False}
    calls = []
    result = (
        None
        if outcome == "none"
        else SimpleNamespace(success=torch.tensor([[outcome == "success"]]))
    )

    @contextmanager
    def hold(*args):
        scope["axis"] = True
        try:
            yield
        finally:
            scope["axis"] = False

    @contextmanager
    def contact(*args):
        scope["contact"] = True
        try:
            yield
        finally:
            scope["contact"] = False

    def plan(goal, start, attempts, graph):
        calls.append((goal, start, attempts, graph, dict(scope)))
        return result

    solver = SimpleNamespace(_hold_axis_scope=hold, _plan_cspace=plan)
    solver._plan_cspace_with_failure_evidence = lambda *args: (
        MotionPlanner._plan_cspace_with_failure_evidence(solver, *args)
    )
    evidence_calls = []

    def evidence(planner, actual, *, trajectory_kind):
        assert planner is solver and actual is result
        assert scope == {"axis": True, "contact": boundary != "none"}
        evidence_calls.append(trajectory_kind)
        return {
            "raw_mesh_gap_sample": {"step": 124, "pairs": [{"sphere": 7, "raw_gap_m": -0.0001}]}
        }

    monkeypatch.setattr("curobo._src.motion.motion_contact.boundary_contact_scope", contact)
    monkeypatch.setattr(
        "curobo._src.motion.motion_failure_diagnostics.terminal_failure_summary", evidence
    )
    goal, start = object(), object()
    returned = MotionPlanner.plan_cspace(
        solver,
        goal,
        start,
        max_attempts=7,
        enable_graph_attempt=3,
        allow_boundary_collision=boundary,
    )
    assert returned is result
    assert calls == [
        (
            goal,
            start,
            7,
            3 if boundary == "none" else 7,
            {"axis": True, "contact": boundary != "none"},
        )
    ]
    assert scope == {"axis": False, "contact": False}
    assert evidence_calls == (["optimized", "interpolated"] if outcome == "failure" else [])
    assert ("C-space interpolated trajectory failure evidence" in caplog.text) is (
        outcome == "failure"
    )


def test_raw_seed_one_ulp_position_violation_is_identified_without_projection(caplog):
    lower = torch.tensor(-1.3599998950958252, dtype=torch.float32)
    outside = torch.nextafter(lower, torch.tensor(-float("inf")))
    state = JointState.from_position(
        torch.tensor([[[[0.0, -1.3], [0.0, float(outside)], [0.0, -1.2]]]]),
        joint_names=["base", "arm"],
    )
    state.velocity = torch.zeros_like(state.position)
    state.acceleration = torch.zeros_like(state.position)
    state.jerk = torch.zeros_like(state.position)
    saved = state.position.clone()
    limits = SimpleNamespace(
        joint_names=["base", "arm"],
        position=torch.tensor([[-1.0, float(lower)], [1.0, -float(lower)]]),
        velocity=torch.tensor([[-2.0, -2.0], [2.0, 2.0]]),
        acceleration=torch.tensor([[-3.0, -3.0], [3.0, 3.0]]),
        jerk=torch.tensor([[-60.0, -60.0], [60.0, 60.0]]),
    )
    config = SimpleNamespace(
        joint_limits=limits,
        cost_type=CSpaceCostType.STATE,
        weight=torch.full((5,), 5000.0),
        activation_distance=torch.zeros(5),
    )
    manager = SimpleNamespace(
        has_cost=lambda name: name == "cspace",
        get_cost=lambda name: SimpleNamespace(config=config),
    )
    solver = SimpleNamespace(metrics_rollout=SimpleNamespace(metrics_constraint_manager=manager))
    maxima = {"optimized": {"cspace": {"maximum": 3.552713678800501e-11}}}
    selected = SimpleNamespace(
        js_solution=state, debug_info={"selected_constraint_maxima": maxima}
    )
    recovered = SimpleNamespace(get_topk_seeds=lambda count: selected if count == 1 else None)
    evidence = TrajOptSolver._initial_seed_failure_evidence(solver, recovered)
    torch.testing.assert_close(state.position, saved, rtol=0, atol=0)
    term = evidence["seed_joint_bounds"]["position"]
    assert term["maximum_excess"] == float(lower - outside)
    assert 0.5 * 5000.0 * term["maximum_excess"] ** 2 == pytest.approx(3.552713678800501e-11)
    assert term["largest_violations"][0]["joint"] == "arm"
    assert term["largest_violations"][0]["step"] == 1
    assert evidence["seed_constraint_maxima"] is maxima
    assert all(
        evidence["seed_joint_bounds"][name]["maximum_excess"] == 0
        for name in ("velocity", "acceleration", "jerk")
    )
    assert "Rejected initial seed raw joint-bound terms" in caplog.text


@pytest.mark.parametrize("manager", [None, SimpleNamespace(has_cost=lambda name: False)])
def test_seed_evidence_without_cspace_manager_remains_explicit(manager):
    selected = SimpleNamespace(
        debug_info={
            "selected_constraint_maxima": {"optimized": {"scene_collision": {"maximum": 1.0}}}
        }
    )
    solver = SimpleNamespace(metrics_rollout=SimpleNamespace(metrics_constraint_manager=manager))
    recovered = SimpleNamespace(get_topk_seeds=lambda count: selected)
    evidence = TrajOptSolver._initial_seed_failure_evidence(solver, recovered)
    assert "seed_joint_bounds" not in evidence
    assert evidence["seed_constraint_maxima"]["optimized"]["scene_collision"]["maximum"] == 1.0
