# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Exercise placement with terminal contact through native IK and TrajOpt."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from curobo._src.solver.solver_trajopt_result import TrajOptSolverResult
from curobo.examples.reference.contact_placement import run_case
from curobo.motion_planner import MotionPlanner
from curobo.types import GoalToolPose, JointState


@pytest.mark.parametrize("scene", ["table", "tray", "cabinet"])
def test_native_pose_placement(tmp_path: Path, scene: str) -> None:
    """A colliding place target must pass only with the explicit goal declaration."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    baseline, _ = run_case(tmp_path, scene, False, repeats=1, goal_position=(0, 0, 0.12))
    contact, positions = run_case(tmp_path, scene, True, repeats=1, goal_position=(0, 0, 0.12))
    assert baseline["warm_successes"] == 0
    assert baseline["target_normal_collision"]["native_max_forbidden_penetration_mm"] > 0.9
    assert contact["warm_successes"] == 1
    assert contact["validation"]["valid"]
    assert contact["validation"]["forbidden_collision_count"] == 0
    assert contact["validation"]["terminal_contact_mm"] == pytest.approx(-1, abs=0.011)
    assert contact["normal_landing"]
    assert contact["validation"]["landing_valid"]
    assert contact["validation"]["contact_before_landing_phase_count"] == 0
    assert contact["validation"]["contact_before_alignment_count"] == 0
    assert contact["validation"]["transport_min_support_clearance_mm"] >= 4.99
    assert contact["validation"]["landing_max_lateral_error_mm"] <= 0.1
    assert contact["validation"]["landing_max_orientation_error_rad"] <= 0.001
    assert contact["joint_trajectory_optimization"]
    assert 0 < contact["validation"]["landing_start_fraction"] < 1
    assert positions[-1].tolist() == pytest.approx([0, 0, 0.12], abs=1e-5)
    if scene == "cabinet":
        assert contact["validation"]["minimum_forbidden_clearance_mm"] > 0
        assert contact["validation"]["goal_headroom_mm"] == pytest.approx(30, abs=0.011)


def test_native_cspace_placement(tmp_path: Path) -> None:
    """Goal-contact trajectory validation also works for a supplied joint target."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    contact, _ = run_case(
        tmp_path, "cabinet", True, repeats=1, goal_position=(0, 0, 0.12), pose_goal=False
    )
    assert contact["warm_successes"] == 1
    assert contact["validation"]["valid"]
    assert contact["validation"]["contact_before_landing_phase_count"] == 0


def test_sealed_cabinet_placement_is_rejected(tmp_path: Path) -> None:
    """An allowed final support contact cannot excuse a blocked approach."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    contact, _ = run_case(tmp_path, "sealed_cabinet", True, repeats=1, goal_position=(0, 0, 0.12))
    assert contact["warm_successes"] == 0


@pytest.mark.parametrize(
    "goal,support_height,approach_clearance",
    [((0.05, 0.02, 0.1195), 0, 0.003), ((-0.05, 0, 0.181), 0.06, 0.008)],
)
def test_place_uses_only_supplied_goal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    goal: tuple[float, float, float],
    support_height: float,
    approach_clearance: float,
) -> None:
    """Plan each complete trajectory to the caller's target without intermediate pose solves."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    goals = []
    original = MotionPlanner.plan_pose

    def record_goal(
        self: MotionPlanner, target: GoalToolPose, current: JointState, **kwargs: Any
    ) -> TrajOptSolverResult | None:
        goals.append(target.position.detach().cpu().reshape(-1, 3)[0].tolist())
        return original(self, target, current, **kwargs)

    monkeypatch.setattr(MotionPlanner, "plan_pose", record_goal)
    record, positions = run_case(
        tmp_path,
        "cabinet",
        True,
        repeats=1,
        goal_position=goal,
        support_height=support_height,
        approach_clearance=approach_clearance,
    )
    assert record["warm_successes"] == 1
    assert record["validation"]["landing_valid"]
    assert record["validation"]["transport_min_support_clearance_mm"] >= (
        approach_clearance * 1000 - 0.01
    )
    assert positions[-1].tolist() == pytest.approx(goal, abs=1e-5)
    assert len(goals) == 2  # One full-trajectory solve for cold and warm repetitions.
    for target in goals:
        assert target == pytest.approx(goal, abs=1e-6)
