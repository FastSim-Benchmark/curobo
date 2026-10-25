# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Original seed restoration precedes PRM and consumes the existing attempt budget."""
from types import SimpleNamespace

import pytest
import torch

from curobo._src.motion.motion_planner import MotionPlanner
from curobo._src.state.state_joint import JointState


@pytest.mark.parametrize("attempts,success,call_count", [(1, False, 1), (2, True, 2)])
def test_restore_original_seeds_before_graph_within_budget(
    attempts: int, success: bool, call_count: int
) -> None:
    """A failed new path may retry its original seed pool without a fresh sample."""
    frozen = torch.full((1, 4, 2), 0.25)
    calls = []
    trajectories = []

    def ik(goal: object, **kwargs: object) -> SimpleNamespace:
        calls.append(kwargs)
        if len(calls) == 2:
            torch.testing.assert_close(kwargs["seed_config"], frozen)
            assert kwargs["prefer_smooth_motion"] is False
        return SimpleNamespace(success=torch.ones(1, 2, dtype=torch.bool),
            solution=torch.zeros(1, 2, 2), total_time=0.1, solve_time=0.05,
            debug_info={"global_seed_config": frozen} if len(calls) == 1 else {})

    def trajectory(*args: object, **kwargs: object) -> SimpleNamespace:
        trajectories.append(kwargs)
        assert kwargs["seed_traj"] is None
        return SimpleNamespace(success=torch.full((1, 2), len(trajectories) > 1),
                               total_time=0.2, solve_time=0.1, debug_info={})

    def graph(*args: object) -> None:
        pytest.fail("PRM must not precede original seed restoration")

    planner = SimpleNamespace(ik_solver=SimpleNamespace(solve_pose=ik),
        trajopt_solver=SimpleNamespace(config=SimpleNamespace(num_seeds=2), solve_pose=trajectory),
        graph_planner=object(), _get_graph_seed_trajectories=graph,
        config=SimpleNamespace(trajopt_finetune_attempts=None))
    result = MotionPlanner._plan_pose_single(planner, SimpleNamespace(num_goalset=1),
        JointState.from_position(torch.zeros(1, 2)), attempts, 1)
    assert bool(result.success.any()) is success
    assert len(calls) == len(trajectories) == call_count
