# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""No-endpoint diagnostics preserve the real retry and acceptance behavior."""

from types import SimpleNamespace

import pytest
import torch

from curobo._src.motion.motion_planner import MotionPlanner
from curobo.types import JointState


def fixture(monkeypatch, outcomes, *, posture=False):
    warnings, calls = [], []
    diagnostics = {
        "seed_count": 128,
        "feasible_seed_count": 0,
        "converged_seed_count": 19,
        "successful_seed_count": 0,
        "constraints": {"posture": {"positive_cost_seed_count": 109},
                        "self_collision": {"positive_cost_seed_count": 17},
                        "scene_collision": {"positive_cost_seed_count": 128},
                        "cspace": {"positive_cost_seed_count": 2}},
    }
    monkeypatch.setattr("curobo._src.motion.motion_planner.log_warn", warnings.append)
    current = JointState.from_position(torch.zeros((1, 2)))
    goals = SimpleNamespace(num_goalset=1)
    def ik(target, **kwargs):
        torch.testing.assert_close(kwargs["current_state"].position, current.position)
        assert target is goals and kwargs["return_seeds"] == 2
        index = len(calls)
        calls.append(kwargs)
        outcome = outcomes[index]
        if isinstance(outcome, BaseException):
            raise outcome
        return SimpleNamespace(success=torch.tensor([[outcome, outcome]]),
                               solution=torch.zeros((1, 2, 2)), total_time=.1, solve_time=.05,
                               debug_info={"failed_ik": diagnostics} if not outcome else {})
    def trajectory(*args, **kwargs):
        return SimpleNamespace(success=torch.tensor([True]), debug_info={},
                               total_time=.2, solve_time=.1)
    planner = SimpleNamespace(ik_solver=SimpleNamespace(solve_pose=ik),
        trajopt_solver=SimpleNamespace(config=SimpleNamespace(num_seeds=2), solve_pose=trajectory),
        graph_planner=None, config=SimpleNamespace(trajopt_finetune_attempts=None))
    seed_calls = []
    seeds = None
    if posture:
        def endpoint(attempt):
            seed_calls.append(attempt)
            return torch.zeros((1, 128, 2))
        seeds = SimpleNamespace(goals=torch.zeros((16, 2)), endpoint_seeds=endpoint,
                                trajectory_seeds=lambda *args: None, records=[],
                                total_time=0., solve_time=0.)
    return planner, goals, current, seeds, warnings, calls, diagnostics, seed_calls


@pytest.mark.parametrize("posture", [False, True])
def test_all_ik_rejections_log_each_bounded_batch_without_changing_none(monkeypatch, posture):
    planner, goals, current, seeds, warnings, calls, diagnostics, seed_calls = fixture(
        monkeypatch, [False]*3, posture=posture)
    result = MotionPlanner._plan_pose_single(planner, goals, current, 3, 3, posture_seeds=seeds)
    assert result is None and len(calls) == len(warnings) == 3
    for index, warning in enumerate(warnings):
        assert f"attempt {index+1}/3" in warning
        assert f"posture={posture}" in warning
        assert f"candidate_count={16 if posture else 1}" in warning
        assert str(diagnostics) in warning
    assert seed_calls == ([0, 1, 2] if posture else [])


def test_rejected_batch_does_not_prevent_a_later_success(monkeypatch):
    planner, goals, current, seeds, warnings, calls, _, _ = fixture(monkeypatch, [False, True])
    result = MotionPlanner._plan_pose_single(planner, goals, current, 3, 3)
    assert result.success.all() and len(calls) == 2 and len(warnings) == 1
    assert result.total_time == pytest.approx(.4)


def test_solver_error_still_propagates_without_becoming_ik_rejection(monkeypatch):
    planner, goals, current, seeds, warnings, calls, _, _ = fixture(
        monkeypatch, [ValueError("solver bug")])
    with pytest.raises(ValueError, match="solver bug"):
        MotionPlanner._plan_pose_single(planner, goals, current, 3, 3)
    assert len(calls) == 1 and not warnings
