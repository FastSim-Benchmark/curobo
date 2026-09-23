# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Failed-state diagnostics distinguish limits without changing tensor inputs."""

from types import SimpleNamespace

import pytest
import torch

from curobo._src.cost.cost_cspace_type import CSpaceCostType
from curobo._src.solver.solver_trajopt import TrajOptSolver
from curobo._src.solver.solver_trajopt_result import TrajOptSolverResult
from curobo._src.solver.trajopt_diagnostics import joint_bound_diagnostics
from curobo._src.state.state_joint import JointState


def fixture():
    """Use two joints with distinct bounds and a three-step CPU trajectory."""
    limits = SimpleNamespace(
        joint_names=["base", "arm"],
        **{
            name: torch.tensor([[-1.0, -2.0], [1.0, 2.0]])
            for name in ("position", "velocity", "acceleration", "jerk")
        },
    )
    config = SimpleNamespace(
        cost_type=CSpaceCostType.STATE,
        joint_limits=limits,
        weight=torch.full((5,), 5000.0),
        activation_distance=torch.zeros(5),
    )
    state = JointState.from_position(torch.zeros((1, 1, 3, 2)), joint_names=limits.joint_names)
    state.velocity = torch.zeros_like(state.position)
    state.acceleration = torch.zeros_like(state.position)
    state.jerk = torch.zeros_like(state.position)
    return state, config


@pytest.mark.parametrize("term", ["position", "velocity", "acceleration", "jerk"])
def test_only_actual_violating_term_is_reported(term):
    """An endpoint derivative violation must not be mislabeled as a position limit."""
    state, config = fixture()
    getattr(state, term)[0, 0, -1, 1] = 3.25
    before = {
        name: getattr(state, name).clone()
        for name in ("position", "velocity", "acceleration", "jerk")
    }
    result = joint_bound_diagnostics(state, config)
    for name, value in before.items():
        torch.testing.assert_close(getattr(state, name), value)
        assert result["terms"][name]["maximum_excess"] == (1.25 if name == term else 0)
    row = result["terms"][term]["largest_violations"][0]
    assert row == {
        "trajectory_index": 0,
        "step": 2,
        "joint": "arm",
        "value": 3.25,
        "lower": -2.0,
        "upper": 2.0,
        "excess": 1.25,
    }
    assert result["position_limits"] == [[-1.0, -2.0], [1.0, 2.0]]


def test_exact_bounds_and_activation_distance():
    """Exact raw bounds pass; configured soft margins are reported explicitly."""
    state, config = fixture()
    state.position[0, 0, -1] = torch.tensor([1.0, -2.0])
    assert joint_bound_diagnostics(state, config)["terms"]["position"]["maximum_excess"] == 0
    config.activation_distance[0] = 0.1
    assert joint_bound_diagnostics(state, config)["terms"]["position"][
        "maximum_excess"
    ] == pytest.approx(0.4)


def test_missing_derivative_is_explicit_and_output_is_bounded():
    """Absent state components remain unknown, and large trajectories stay bounded."""
    state, config = fixture()
    state.position = torch.full((2, 3, 40, 2), 4.0)
    state.velocity = state.acceleration = state.jerk = None
    result = joint_bound_diagnostics(state, config)
    assert result["trajectory_count"] == 6
    assert result["terms"]["velocity"] == {"available": False}
    assert result["terms"]["position"]["violation_count"] == 480
    assert len(result["terms"]["position"]["largest_violations"]) == 8


def test_position_cost_does_not_mislabel_effort_weight_as_velocity():
    """Teleport position costs have two weights: position and effort only."""
    state, config = fixture()
    config.cost_type = CSpaceCostType.POSITION
    config.weight = torch.tensor([5000.0, 0.0])
    config.activation_distance = torch.zeros(2)
    state.velocity.fill_(4.0)
    result = joint_bound_diagnostics(state, config)
    assert result["terms"]["velocity"] == {"available": True, "evaluated_by_cost": False}
    assert result["terms"]["position"]["maximum_excess"] == 0


@pytest.mark.parametrize("success", [False, True])
def test_solver_result_adds_diagnostics_only_for_failure(success):
    """Exercise the real result-selection path without initializing a GPU solver."""
    state, config = fixture()
    state.position[0, 0, -1, 1] = 3.25
    state.dt = torch.tensor([[0.02]])
    manager = SimpleNamespace(
        has_cost=lambda name: name == "cspace",
        get_cost=lambda name: SimpleNamespace(config=config),
    )
    solver = SimpleNamespace(
        joint_names=config.joint_limits.joint_names,
        metrics_rollout=SimpleNamespace(metrics_constraint_manager=manager),
        auxiliary_rollout=SimpleNamespace(
            transition_model=SimpleNamespace(get_full_dof_from_solution=lambda state: state)
        ),
        interpolation_steps=4,
    )
    original_debug = {"selected_constraint_maxima": {"cspace": 3698.0}}
    result = TrajOptSolverResult(
        success=torch.tensor([[success]]),
        js_solution=state,
        solution=state.position.clone(),
        debug_info=original_debug,
        total_cost_reshaped=torch.zeros((1, 1)),
        seed_rank=torch.zeros((1, 1), dtype=torch.long),
        batch_size=1,
        num_seeds=1,
    )
    returned = TrajOptSolver._get_best_result(solver, result, 1)
    assert bool(returned.success.item()) is success
    assert returned.debug_info["selected_constraint_maxima"] == {"cspace": 3698.0}
    assert "selected_joint_bounds" not in original_debug
    if success:
        assert "selected_joint_bounds" not in returned.debug_info
    else:
        detail = returned.debug_info["selected_joint_bounds"]
        assert detail["source"] == "optimized_joint_state"
        assert detail["position_end"] == [0.0, 3.25]
        assert detail["terms"]["position"]["end_maximum_excess"] == 1.25
