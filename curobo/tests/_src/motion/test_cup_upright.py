# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Combine a real rotational arm, attached cup, free yaw, and initial support contact."""

import pytest
import torch

from curobo.examples.reference.cup_upright import make_cup_planner, validate_cup


@pytest.mark.parametrize("contact_enabled", [False, True])
def test_upright_contact_departure(contact_enabled: bool) -> None:
    """Only the declared contact exception permits this upright departure."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    planner, start, goal, axis, support, contact = make_cup_planner(contact_enabled)
    with planner:
        for _ in range(2):
            result = planner.plan_pose(goal, start, max_attempts=2, enable_graph_attempt=2)
            success = result is not None and bool(result.success.all())
            assert success == contact_enabled
            if success:
                report = validate_cup(
                    planner, result.get_interpolated_plan(), axis, support, contact
                )
                assert report["max_tilt_rad"] < 0.01
                assert report["contact_valid"]
                assert report["initial_support_gap_m"] == pytest.approx(-0.001, abs=1e-6)
                assert report["minimum_other_pair_gap_m"] > 0
