# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Fresh global retries retain graph opportunities and the caller's budgets."""
from types import SimpleNamespace

import pytest
import torch

from curobo._src.motion.motion_planner import MotionPlanner
from curobo._src.state.state_joint import JointState
from curobo._src.util.time_budget import time_budget


def make_planner(
    ik_outcomes: list[bool], trajectory_outcomes: list[bool],
    graph_outcomes: list[bool], graph_enabled: bool = True,
) -> SimpleNamespace:
    """Record stage order while feeding bounded native-stage outcomes."""
    events, ik_calls, trajectories, graph_calls = [], [], [], []

    def ik(goal: object, **kwargs: object) -> SimpleNamespace:
        index = len(ik_calls)
        ik_calls.append(kwargs)
        events.append("ik")
        return SimpleNamespace(
            success=torch.full((1, 2), ik_outcomes[index], dtype=torch.bool),
            solution=torch.zeros(1, 2, 2), total_time=0.1, solve_time=0.05,
            debug_info={"global_seed_config": torch.full((1, 4, 2), 0.25)},
        )

    def trajectory(*args: object, **kwargs: object) -> SimpleNamespace:
        index = len(trajectories)
        trajectories.append(kwargs)
        events.append("trajopt")
        return SimpleNamespace(
            success=torch.full((1, 2), trajectory_outcomes[index], dtype=torch.bool),
            total_time=0.2, solve_time=0.1, debug_info={},
        )

    def graph(*args: object) -> torch.Tensor | None:
        index = len(graph_calls)
        graph_calls.append(args)
        events.append("graph")
        return torch.ones(1, 2, 3, 2) if graph_outcomes[index] else None

    return SimpleNamespace(
        ik_solver=SimpleNamespace(solve_pose=ik),
        trajopt_solver=SimpleNamespace(
            config=SimpleNamespace(num_seeds=2), solve_pose=trajectory),
        graph_planner=object() if graph_enabled else None,
        _get_graph_seed_trajectories=graph,
        config=SimpleNamespace(trajopt_finetune_attempts=None),
        events=events, ik_calls=ik_calls, trajectories=trajectories,
        graph_calls=graph_calls,
    )


def solve(planner: SimpleNamespace, attempts: int = 3, graph_attempt: int = 1,
          posture: object = None) -> object:
    """Invoke the real retry loop on deterministic stage doubles."""
    return MotionPlanner._plan_pose_single(
        planner, SimpleNamespace(num_goalset=1),
        JointState.from_position(torch.zeros(1, 2)), attempts, graph_attempt,
        posture_seeds=posture,
    )


def test_failed_trajectory_uses_fresh_retry_and_second_round_graph() -> None:
    """Do not feed saved LM seeds back or postpone the requested graph attempt."""
    planner = make_planner([True, True], [False, True], [True])
    result = solve(planner)
    assert result.success.any()
    assert planner.events == ["ik", "trajopt", "ik", "graph", "trajopt"]
    assert "prefer_smooth_motion" not in planner.ik_calls[0]
    assert planner.ik_calls[1]["prefer_smooth_motion"] is False
    assert "seed_config" not in planner.ik_calls[1]
    assert planner.trajectories[1]["seed_traj"] is not None
    assert planner.trajectories[1]["finetune_attempts"] == 3


@pytest.mark.parametrize("attempts", [1, 2, 3])
def test_no_endpoint_never_exceeds_total_attempt_budget(attempts: int) -> None:
    """Every retry asks for new global seeds, even after an IK-only failure."""
    planner = make_planner([False] * attempts, [], [])
    assert solve(planner, attempts) is None
    assert planner.events == ["ik"] * attempts
    for kwargs in planner.ik_calls[1:]:
        assert "seed_config" not in kwargs
        assert kwargs["prefer_smooth_motion"] is False


def test_graph_failure_gets_another_fresh_global_attempt() -> None:
    """The third round does not return to local-only quality sampling."""
    planner = make_planner([True, True, True], [False, True], [False, True])
    assert solve(planner).success.any()
    assert planner.events == ["ik", "trajopt", "ik", "graph", "ik", "graph", "trajopt"]
    for kwargs in planner.ik_calls[1:]:
        assert "seed_config" not in kwargs
        assert kwargs["prefer_smooth_motion"] is False


@pytest.mark.parametrize("graph_enabled,graph_attempt", [(False, 1), (True, 3)])
def test_caller_can_disable_or_delay_graph(graph_enabled: bool, graph_attempt: int) -> None:
    """Fresh samples do not override the caller's graph schedule."""
    planner = make_planner([True, True], [False, True], [], graph_enabled)
    assert solve(planner, graph_attempt=graph_attempt).success.any()
    assert planner.events == ["ik", "trajopt", "ik", "trajopt"]


def test_first_round_success_does_not_trigger_a_retry() -> None:
    """Keep the successful quality fast path and its existing seed policy."""
    planner = make_planner([True], [True], [])
    assert solve(planner).success.any()
    assert planner.events == ["ik", "trajopt"]
    assert "prefer_smooth_motion" not in planner.ik_calls[0]


def test_explicit_posture_seeds_are_not_replaced() -> None:
    """Preserve caller-provided posture endpoint and trajectory proposals."""
    seeds = [torch.full((1, 4, 2), 0.1), torch.full((1, 4, 2), 0.2)]
    posture = SimpleNamespace(
        endpoint_seeds=lambda attempt: seeds[attempt],
        trajectory_seeds=lambda goal, endpoint, attempt: None,
        goals=[object()], total_time=0., solve_time=0., records=[],
    )
    planner = make_planner([True, True], [False, True], [True])
    assert solve(planner, posture=posture).success.any()
    for i, kwargs in enumerate(planner.ik_calls):
        assert kwargs["seed_config"] is seeds[i]
        assert "prefer_smooth_motion" not in kwargs


def test_retry_cannot_extend_the_shared_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """An exhausted call budget stops before another IK or graph invocation."""
    now = [0.]
    monkeypatch.setattr("curobo._src.util.time_budget.monotonic", lambda: now[0])
    planner = make_planner([False], [], [])
    original = planner.ik_solver.solve_pose

    def expire(*args: object, **kwargs: object) -> object:
        result = original(*args, **kwargs)
        now[0] = 2.
        return result

    planner.ik_solver.solve_pose = expire
    with pytest.raises(TimeoutError), time_budget(1.):
        solve(planner)
    assert planner.events == ["ik"]
