# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""CUDA scalar arguments retain their declared float32 values."""

from types import SimpleNamespace

import pytest
import torch

from curobo._src.curobolib.backends.cuda_core_backend.optimization import launch_line_search
from curobo._src.optim.gradient.line_search_strategy import StrongWolfeLineSearchStrategy
from curobo._src.optim.gradient.update_best_solution import update_best_solution


@pytest.mark.parametrize("count", [4, 5])
@pytest.mark.parametrize(
    "armijo,curvature,delta,relative,expected_index,update_best",
    [
        (1e-5, 0.9, 0.0, 0.0, 2, True),
        (0.1, 0.3, 0.0, 0.0, 1, True),
        (0.9, 0.9, 0.0, 0.0, 1, True),
        (0.1, 0.9, 2.0, 0.0, 2, False),
        (0.1, 0.9, 0.5, 0.2, 2, False),
        (0.1, 0.9, 0.5, 0.1, 2, True),
    ],
)
def test_line_search_float_parameters_match_torch(
    count, armijo, curvature, delta, relative, expected_index, update_best
):
    """Both specialized and runtime kernels use configured Wolfe and best thresholds."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    dimension = 32
    costs = torch.tensor([10.0, 9.0, 8.5, 11.0, 12.0][:count], device="cuda").view(1, count, 1)
    actions = torch.arange(count * dimension, device="cuda", dtype=torch.float32)
    actions = actions.view(1, count, dimension)
    gradients = torch.zeros_like(actions)
    gradients[0, :, 0] = torch.tensor([-2.0, -0.4, -1.5, 1.0, 2.0][:count], device="cuda")
    direction = torch.zeros((1, 1, dimension), device="cuda")
    direction[..., 0] = 1
    scales = torch.tensor([0.0, 0.5, 1.0, 2.0, 3.0][:count], device="cuda")
    context = SimpleNamespace(
        line_search_c_1=armijo, line_search_c_2=curvature,
        line_search_scale=scales.view(1, count, 1, 1),
        opt_dim=dimension, action_dim=dimension, action_horizon=1,
        c_idx=torch.zeros(1, device="cuda", dtype=torch.int64),
    )
    strategy = StrongWolfeLineSearchStrategy()
    expected = strategy._torch_wolfe_search(
        actions, direction, costs, gradients, context,
        strategy._compute_curvature_condition, strategy._handle_no_valid_step,
    )
    assert expected.selected_state.idxs[0, 0] == expected_index

    def state():
        return SimpleNamespace(
            best_cost=torch.tensor([10.0], device="cuda"),
            best_action=torch.full((1, 1, dimension), -1.0, device="cuda"),
            best_iteration=torch.zeros(1, device="cuda", dtype=torch.int32),
            current_iteration=torch.zeros(1, device="cuda", dtype=torch.int32),
            converged=torch.zeros(1, device="cuda", dtype=torch.uint8),
            cost=torch.zeros(1, device="cuda"),
            action=torch.zeros((1, 1, dimension), device="cuda"),
            gradient=torch.zeros((1, 1, dimension), device="cuda"),
        )

    native = state()
    reference = state()
    reference.cost = expected.selected_state.cost
    reference.action = expected.selected_state.action
    update_best_solution(reference, 1, dimension, delta, relative, 10)
    exploration_cost = torch.empty_like(native.cost)
    exploration_action = torch.empty_like(native.action)
    exploration_gradient = torch.empty_like(native.gradient)
    exploration_idx = torch.empty((1, count), device="cuda", dtype=torch.int32)
    selected_idx = torch.empty_like(exploration_idx)
    launch_line_search(
        native.best_cost, native.best_action, native.best_iteration, native.current_iteration,
        native.converged, 10, delta, relative,
        exploration_cost, exploration_action, exploration_gradient, exploration_idx,
        native.cost, native.action, native.gradient, selected_idx,
        costs, actions, gradients, direction, scales, armijo, curvature,
        True, False, count, dimension, 1,
    )
    assert selected_idx[0, 0] == expected_index
    assert bool(native.best_iteration[0]) is update_best
    for field in ("cost", "action", "best_cost", "best_action", "best_iteration"):
        torch.testing.assert_close(getattr(native, field), getattr(reference, field))
    torch.testing.assert_close(native.gradient, expected.selected_state.gradient)
