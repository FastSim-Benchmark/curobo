# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for bounded interpolation rounding at joint limits."""

import pytest
import torch

from curobo._src.state.state_joint import JointState
from curobo._src.util.interpolation_roundoff import project_interpolation_position_roundoff


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_observed_wrist_excursion_and_larger_violations(device):
    """The recorded one-ULP wrist excursion is fixed; two ULPs still fail."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    limits = torch.tensor([[-1.0123], [1.0123]], device=device)
    lower, upper = limits.unbind(0)
    below = torch.nextafter(lower, torch.full_like(lower, -float("inf")))
    above = torch.nextafter(upper, torch.full_like(upper, float("inf")))
    twice_below = torch.nextafter(below, torch.full_like(lower, -float("inf")))
    twice_above = torch.nextafter(above, torch.full_like(upper, float("inf")))
    values = torch.stack([lower, below, twice_below, upper, above, twice_above]).unsqueeze(0)
    derivative = torch.ones_like(values)
    state = JointState(position=values, velocity=derivative)
    reference = JointState(position=torch.stack([lower, upper]).unsqueeze(0))
    result = project_interpolation_position_roundoff(state, reference, limits)
    expected = torch.stack([lower, lower, twice_below, upper, upper, twice_above]).unsqueeze(0)
    torch.testing.assert_close(result.position, expected, rtol=0, atol=0)
    assert result.velocity is derivative
    assert state.position is values
    assert state.position[0, 1, 0] < lower[0]
    # This was the false-positive weighted cspace cost in the native failure.
    assert (2500 * (below - lower).square()).item() == pytest.approx(3.552713678800501e-11)


def test_invalid_reference_is_not_repaired_across_batches_or_joints():
    """Eligibility is independent for each seed and coordinate."""
    limits = torch.tensor([[-1.0, -1.0], [1.0, 1.0]])
    below = torch.nextafter(limits[0], torch.full((2,), -float("inf")))
    reference = JointState(position=torch.tensor([[[-1.0, -1.1]], [[-1.1, -1.0]]]))
    state = JointState(position=below.expand(2, 3, 2).clone())
    result = project_interpolation_position_roundoff(state, reference, limits)
    assert torch.all(result.position[0, :, 0] == -1.0)
    assert torch.all(result.position[1, :, 1] == -1.0)
    assert torch.all(result.position[0, :, 1] == below[1])
    assert torch.all(result.position[1, :, 0] == below[0])


def test_nonfinite_and_interior_values_are_preserved():
    """No projection conceals nonfinite state or changes interior positions."""
    limits = torch.tensor([[-1.0], [1.0]])
    values = torch.tensor([[float("nan")], [float("inf")], [-float("inf")], [0.25]])
    state = JointState(position=values)
    reference = JointState(position=torch.zeros((2, 1)))
    result = project_interpolation_position_roundoff(state, reference, limits)
    torch.testing.assert_close(result.position, values, rtol=0, atol=0, equal_nan=True)
