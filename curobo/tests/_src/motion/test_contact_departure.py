# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Exercise complete native trajectories from table and low-cabinet contact."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from curobo.examples.reference.contact_separation import run_case


@pytest.mark.parametrize("scene", ["table", "cabinet"])
def test_native_contact_departure(tmp_path: Path, scene: str) -> None:
    """Compare the unchanged collision contract and the selective contact contract."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    baseline, _ = run_case(tmp_path, scene, False, repeats=1)
    contact, positions = run_case(tmp_path, scene, True, repeats=1)
    assert baseline["warm_successes"] == 0
    assert contact["warm_successes"] == 1
    validation = contact["validation"]
    assert validation["valid"]
    assert validation["forbidden_collision_count"] == 0
    assert validation["native_max_forbidden_penetration_mm"] == 0
    assert validation["initial_contact_mm"] == pytest.approx(-1.0, abs=1e-4)
    assert positions[0].tolist() == pytest.approx([0, 0, 0.12], abs=1e-6)
    if scene == "cabinet":
        assert validation["initial_headroom_mm"] == pytest.approx(30.0, abs=1e-3)
        assert validation["ten_cm_lift_min_clearance_mm"] < -20.0
        assert validation["maximum_lift_inside_cabinet_mm"] < 30.0
        assert validation["minimum_forbidden_clearance_mm"] > 0
