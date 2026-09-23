# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Failed seeds retain cost order independently of the legacy failure penalty."""

import pytest
import torch

from curobo._src.solver.solver_trajopt_result import TrajOptSolverResult


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "success,expected",
    [
        ([[False, False, False, False]], [[3, 1, 2, 0]]),
        ([[True, False, True, False]], [[2, 0, 3, 1]]),
        ([[True, True, True, True]], [[3, 1, 2, 0]]),
    ],
)
def test_rank_preserves_cost_within_success_groups(device, success, expected):
    """Float32 costs must not collapse into ties after adding the failed penalty."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    costs = torch.tensor([[90.0, 10.0, 50.0, 2.0]], device=device)
    successful = torch.tensor(success, device=device)
    before = successful.clone()
    derivatives = torch.zeros((4, 24, 2), device=device)
    returned_costs, rank = TrajOptSolverResult._jit_compute_rank(
        torch.full((4,), 0.02, device=device),
        derivatives,
        derivatives,
        costs,
        successful,
        1,
        4,
    )
    assert rank.cpu().tolist() == expected
    torch.testing.assert_close(successful, before)
    assert returned_costs.dtype == torch.float32
    expected_costs = costs + 20.0
    expected_costs[~successful] += 1e16
    torch.testing.assert_close(returned_costs, expected_costs)


def test_rank_batches_independently_and_preserves_equal_cost_order():
    """Ties are deterministic and one batch's feasibility cannot affect another."""
    derivatives = torch.zeros((6, 24, 2))
    _, rank = TrajOptSolverResult._jit_compute_rank(
        torch.full((6,), 0.02),
        derivatives,
        derivatives,
        torch.tensor([[3.0, 3.0, 1.0], [1.0, 2.0, 3.0]]),
        torch.tensor([[False, False, False], [False, True, True]]),
        2,
        3,
    )
    assert rank.tolist() == [[2, 0, 1], [1, 2, 0]]
