# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Geometric LM residual has correct derivatives and preserves global seeds."""
import pytest
import torch

from curobo._src.solver.seed_ik.seed_ik_solver import SeedIKSolver
from curobo._src.solver.seed_ik.seed_ik_solver_cfg import SeedIKSolverCfg
from curobo._src.solver.seed_ik.seed_path_regularizer import SeedPathRegularizer


@pytest.fixture
def regularizer() -> SeedPathRegularizer:
    cfg = SeedIKSolverCfg.create(robot="franka.yml", num_seeds=8, use_cuda_graph=False)
    solver = SeedIKSolver(cfg)
    prior = SeedPathRegularizer(cfg.robot_config.kinematics, .05, 4, cfg.device_cfg)
    prior.setup_batch_tensors(2, 8)
    prior.active.fill_(1)
    return prior


def test_path_gradient_and_global_reserve(regularizer: SeedPathRegularizer) -> None:
    """Check every joint derivative by finite differences in two distinct batches."""
    prior = regularizer
    start = prior.kinematics.default_joint_position.view(1, -1).repeat(16, 1)
    q = start + torch.tensor([.4, -.3, .2, .5, -.2, .3, -.4], device=start.device)
    q[8:] += .15
    gradient, jacobian, cost = prior.compute(q, start)
    gradient, jacobian, cost = gradient.clone(), jacobian.clone(), cost.clone()
    for joint in range(7):
        delta = torch.zeros_like(q)
        delta[:, joint] = 5e-4
        plus = prior.compute(q + delta, start)[2].clone()
        minus = prior.compute(q - delta, start)[2].clone()
        torch.testing.assert_close(gradient[:, joint], (plus-minus)/1e-3, atol=2e-6, rtol=.02)
    assert cost[[3, 11]].min() > 1e-6
    reserved = torch.tensor([0, 4, 5, 6, 7, 8, 12, 13, 14, 15], device=q.device)
    assert torch.count_nonzero(gradient[reserved]) == 0
    assert torch.count_nonzero(jacobian[reserved]) == 0
    assert torch.count_nonzero(cost[reserved]) == 0


def test_stationary_and_missing_reference(regularizer: SeedPathRegularizer) -> None:
    """No motion and no supplied start must add no soft path objective."""
    prior = regularizer
    start = prior.kinematics.default_joint_position.view(1, -1).repeat(16, 1)
    gradient, jacobian, cost = prior.compute(start, start)
    assert torch.count_nonzero(gradient) == 0
    assert torch.count_nonzero(jacobian) == 0
    assert torch.count_nonzero(cost) == 0
    prior.active.zero_()
    gradient, jacobian, cost = prior.compute(start + .3, start)
    assert torch.count_nonzero(gradient) == 0
    assert torch.count_nonzero(jacobian) == 0
    assert torch.count_nonzero(cost) == 0


def test_graph_replay_updates_reference(regularizer: SeedPathRegularizer) -> None:
    """Graph replay uses the new request's start, not the capture-time value."""
    prior = regularizer
    start = prior.kinematics.default_joint_position.view(1, -1).repeat(16, 1)
    q = start + .4
    for _ in range(3):
        prior.compute(q, start)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = prior.compute(q, start)
    graph.replay()
    first = output[2].clone()
    start.add_(.15)
    graph.replay()
    actual = [x.clone() for x in output]
    expected = prior.compute(q, start)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b)
    assert not torch.allclose(first, actual[2])


def test_scene_switch_disables_and_restores_path_prior() -> None:
    """Shared scene updates change the graph input, without weakening collision checks."""
    from curobo._src.geom.types import SceneCfg
    from curobo._src.solver.solver_ik import IKSolver
    from curobo._src.solver.solver_ik_cfg import IKSolverCfg
    from curobo._src.state.state_joint import JointState
    from curobo._src.types.tool_pose import GoalToolPose

    obstacle = SceneCfg.create({"cuboid": {"far": {
        "pose": [5., 5., 5., 1., 0., 0., 0.], "dims": [.1, .1, .1]}}})
    cfg = IKSolverCfg.create(robot="franka.yml", num_seeds=8,
                             use_cuda_graph=True, scene_model=obstacle)
    cfg.prefer_smooth_motion = True
    solver = IKSolver(cfg)
    current = JointState.from_position(solver.kinematics.default_joint_position.view(1, -1),
                                       joint_names=solver.joint_names)
    tool = solver.kinematics.compute_kinematics(current).tool_poses
    goal = GoalToolPose.from_poses(tool.to_dict(), ordered_tool_frames=solver.tool_frames)
    try:
        for scene, expected in [(obstacle, 0.), (SceneCfg.create({}), 1.), (obstacle, 0.)]:
            solver.update_world(scene)
            result = solver.solve_pose(goal, current_state=current)
            assert result.success.any()
            prior = solver.seed_ik_solver.error_calculator.path_regularizer
            assert prior.active.item() == expected
    finally:
        solver.destroy()
