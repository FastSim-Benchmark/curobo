# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Alternative posture seeds preserve limits, held joints, and query state."""
from types import SimpleNamespace

import pytest
import torch

from curobo._src.motion.posture_seeds import (
    intermediate_goals,
    sample_posture_states,
    waypoint_trajectories,
)


@pytest.mark.parametrize("intermediate", [False, True])
def test_seed_diversity_limits_and_held_coordinates(intermediate):
    start = torch.tensor([0.1, -0.99, 0.3])
    goals = torch.tensor([[0.7, 0.5, 0.3], [-0.7, 0.8, 0.3]])
    limits = torch.tensor([[-1., -1., 0.3], [1., 1., 0.3]])
    free = torch.tensor([False, True, False])
    held = torch.tensor([False, False, True])
    args = (start, goals, limits, free, held, 16)
    rng = torch.random.get_rng_state().clone()
    first = sample_posture_states(*args, 1, intermediate=intermediate)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert torch.equal(first, sample_posture_states(*args, 1, intermediate=intermediate))
    assert not torch.equal(first, sample_posture_states(*args, 2, intermediate=intermediate))
    assert (first >= limits[0]).all() and (first <= limits[1]).all()
    assert torch.equal(first[:, 2], start[2].expand(16))
    if not intermediate:
        assert torch.equal(first[:, 0], goals[torch.arange(16) % 2, 0])
    else:
        assert not torch.equal(first[:, 0], goals[torch.arange(16) % 2, 0])
    with pytest.raises(ValueError):
        sample_posture_states(*args, 0, intermediate=intermediate)


def test_waypoint_paths_avoid_linear_seed_barrier_without_changing_endpoints():
    start = torch.tensor([-1., 0., .3])
    middle = torch.tensor([[0., .8, .3], [0., -.8, .3]])
    ends = torch.tensor([[1., 0., .3], [1., 0., .3]])
    held = torch.tensor([False, False, True])
    paths = waypoint_trajectories(start, middle, ends, 31, held)[0]
    assert torch.equal(paths[:, 0], start.expand(2, 3))
    assert torch.equal(paths[:, -1], ends)
    assert torch.equal(paths[:, :, 2], start[2].expand(2, 31))
    # A radius .4 obstacle at the origin blocks the direct path; both seeds avoid it.
    assert torch.linalg.vector_norm(paths[:, :, :2], dim=-1).min() > .4
    with pytest.raises(ValueError):
        waypoint_trajectories(start, middle, ends, 2, held)


@pytest.mark.parametrize("abort", [False, True])
def test_intermediate_ik_goals_restore_even_on_rejection(abort):
    original = torch.arange(24.).reshape(8, 3)
    valid = torch.tensor([True, True, False, False, False, False, False, False])
    cost = SimpleNamespace(goals=original.clone(), valid=valid.clone())
    rollout = SimpleNamespace(get_cost_component_by_name=lambda name: [cost])
    solver = SimpleNamespace(core=SimpleNamespace(
        get_all_rollout_instances=lambda: [rollout],
        additional_metrics_rollouts={"same": rollout},
    ))
    try:
        with intermediate_goals(solver, torch.ones(3, 3)):
            assert cost.valid.sum() == 3
            assert torch.equal(cost.goals[:3], torch.ones(3, 3))
            if abort:
                raise RuntimeError("rejected")
    except RuntimeError:
        assert abort
    assert torch.equal(cost.goals, original) and torch.equal(cost.valid, valid)


@pytest.mark.parametrize("accepted", [False, True])
def test_rejected_midpoints_never_become_paths(accepted):
    from curobo._src.motion.posture_seeds import PostureSeeds
    from curobo.types import JointState

    start = torch.tensor([[0., 0., .3]])
    goal = torch.tensor([[.7, .4, .3]])
    cost = SimpleNamespace(goals=goal.repeat(256, 1), valid=torch.zeros(256, dtype=torch.bool))
    cost.valid[0] = True
    original = cost.goals.clone()
    rollout = SimpleNamespace(get_cost_component_by_name=lambda name: [cost])
    calls = []

    def solve_pose(goals, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            success=torch.full((1, 4), accepted),
            solution=torch.tensor([[[.2, .1, .3]]]).repeat(1, 4, 1),
            total_time=.2, solve_time=.1,
        )

    planner = SimpleNamespace(
        joint_names=["a", "wrist", "held"],
        kinematics=SimpleNamespace(get_joint_limits=lambda: SimpleNamespace(
            position=torch.tensor([[-1., -1., .3], [1., 1., .3]])
        )),
        ik_solver=SimpleNamespace(
            config=SimpleNamespace(num_seeds=4), solve_pose=solve_pose,
            core=SimpleNamespace(get_all_rollout_instances=lambda: [rollout],
                                 additional_metrics_rollouts={}),
        ),
        trajopt_solver=SimpleNamespace(action_horizon=15),
    )
    context = PostureSeeds(
        planner, JointState.from_position(start, joint_names=planner.joint_names),
        goal, torch.tensor([1, 0, 1]), torch.tensor([0, 0, 1]),
    )
    assert context.endpoint_seeds(0) is None
    assert context.trajectory_seeds(None, goal.unsqueeze(0), 0) is None
    paths = context.trajectory_seeds(None, goal.unsqueeze(0), 1)
    assert (paths is not None) == accepted
    assert len(calls) == 1 and context.records == [
        {"attempt": 2, "sampled": 4, "accepted": 4 if accepted else 0}
    ]
    assert torch.equal(cost.goals, original)
    assert cost.valid.sum() == 1
    assert context.total_time == .2 and context.solve_time == .1
