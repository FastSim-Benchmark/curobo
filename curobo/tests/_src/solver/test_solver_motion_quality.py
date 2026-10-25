# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Motion preferences must preserve feasibility, diversity, and both tools."""
import torch
import pytest

from curobo._src.solver.solver_motion_quality import (
    combine_independent_tool_candidates, motion_quality_cost, select_motion_candidates,
)


def test_worst_tool_prevents_transferring_detour() -> None:
    paths = torch.zeros(1, 2, 3, 2, 3)
    paths[0, 0, 1, :, 0] = torch.tensor([0.1, 0.1])
    paths[0, 1, 1, :, 0] = torch.tensor([0.0, 0.19])
    cost = motion_quality_cost(paths, torch.zeros(1, 2, 3, 2))
    assert cost[0, 0] < cost[0, 1]
    assert torch.isfinite(cost).all()


def test_feasible_old_candidate_survives_better_scored_new_candidates() -> None:
    q = torch.arange(6, dtype=torch.float32).reshape(1, 6, 1)
    success = torch.tensor([[True, True, True, True, True, False]])
    score = torch.tensor([[10.0, 11.0, 1.0, 2.0, 3.0, -100.0]])
    chosen = select_motion_candidates(q, success, score, 4, old_count=2, reserve_old=2)
    assert chosen.tolist() == [[0, 1, 2, 3]]
    assert torch.gather(success, 1, chosen).all()


def test_duplicate_successes_never_admit_a_failure() -> None:
    q = torch.tensor([[[0.0], [0.001], [2.0]], [[0.0], [1.0], [2.0]]])
    success = torch.tensor([[True, True, False], [False, True, False]])
    score = torch.tensor([[1.0, 2.0, 0.0], [0.0, 1.0, 0.0]])
    chosen = select_motion_candidates(q, success, score, 3)
    assert torch.gather(success, 1, chosen).all()


def test_bounded_joint_angles_are_not_wrapped() -> None:
    position = torch.zeros(1, 2, 2, 1, 3)
    joints = torch.tensor([[[[3.0], [-3.0]], [[3.0], [2.5]]]])
    cost = motion_quality_cost(position, joints)
    assert cost[0, 0] > 10 * cost[0, 1]


def test_disjoint_chains_can_share_good_parts_but_shared_joints_cannot() -> None:
    solutions = torch.tensor([[[1.0, 9.0], [8.0, 2.0], [0.0, 0.0]]])
    success = torch.tensor([[True, True, False]])
    costs = torch.tensor([[[0.0, 9.0], [8.0, 0.0], [-1.0, -1.0]]])
    masks = torch.eye(2, dtype=torch.bool)
    result = combine_independent_tool_candidates(solutions, success, costs, masks, 2)
    torch.testing.assert_close(result[:, 0], torch.tensor([[1.0, 2.0]]))
    masks[:, 0] = True
    assert combine_independent_tool_candidates(solutions, success, costs, masks, 2) is None


def test_partial_tool_convergence_can_propose_a_complete_endpoint() -> None:
    """Whole-batch rejection must not hide usable disjoint chain proposals."""
    solutions = torch.tensor([[[1.0, 9.0], [8.0, 2.0]]])
    per_tool = torch.tensor([[[True, False], [False, True]]])
    costs = torch.tensor([[[0.0, 9.0], [8.0, 0.0]]])
    result = combine_independent_tool_candidates(solutions, per_tool, costs,
                                                torch.eye(2, dtype=torch.bool), 2)
    torch.testing.assert_close(result[:, 0], torch.tensor([[1.0, 2.0]]))


def test_rotation_unwinding_is_penalized_independently_of_quaternion_sign() -> None:
    """A 270 degree route to a 90 degree goal must incur an orientation detour."""
    angles = torch.tensor([[[0.0, 0.785398, 1.570796], [0.0, -2.356194, -4.712389]]])
    quaternions = torch.zeros(1, 2, 3, 1, 4)
    quaternions[..., 0] = (angles / 2).cos().unsqueeze(-1)
    quaternions[..., 1] = (angles / 2).sin().unsqueeze(-1)
    positions, joints = torch.zeros(1, 2, 3, 1, 3), torch.zeros(1, 2, 3, 2)
    cost = motion_quality_cost(positions, joints, quaternions)
    assert cost[0, 1] > cost[0, 0] + 1
    quaternions[:, :, 1] *= -1
    torch.testing.assert_close(cost, motion_quality_cost(positions, joints, quaternions))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_local_tool_models_keep_original_robot_topology_unchanged() -> None:
    """Each auxiliary LM sees one tool while the native robot retains both."""
    from curobo._src.motion.motion_planner_cfg import MotionPlannerCfg
    from curobo._src.solver.solver_ik import IKSolver

    cfg = MotionPlannerCfg.create("dual_ur10e.yml", prefer_smooth_motion=True, num_ik_seeds=4)
    params = cfg.ik_solver_config.robot_config.kinematics.kinematics_config
    before = params.joint_affects_endeffector.clone()
    solver = IKSolver(cfg.ik_solver_config)
    try:
        assert len(solver._local_tool_solvers) == len(params.tool_frames) == 2
        for index, local in enumerate(solver._local_tool_solvers):
            assert local.tool_frames == [params.tool_frames[index]]
            masks = local.config.robot_config.kinematics.kinematics_config.joint_affects_endeffector
            torch.testing.assert_close(masks, before.reshape(-1, 2)[:, index].contiguous())
        torch.testing.assert_close(params.joint_affects_endeffector, before)
    finally:
        solver.destroy()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_local_guesses_preserve_sampler_advance_and_explicit_seeds() -> None:
    from curobo._src.solver.seed_ik.seed_ik_solver import SeedIKSolver
    from curobo._src.solver.seed_ik.seed_ik_solver_cfg import SeedIKSolverCfg

    plain = SeedIKSolver(SeedIKSolverCfg.create("franka.yml", num_seeds=32))
    local = SeedIKSolver(SeedIKSolverCfg.create("franka.yml", num_seeds=32, local_seed_count=8))
    seed = plain.default_joint_position.reshape(1, 1, -1)
    a = plain._generate_seed_configs(1, seed)
    b = local._generate_seed_configs(1, seed)
    torch.testing.assert_close(a[:, 8:], b[:, 8:])
    torch.testing.assert_close(plain.act_sample_gen._int_gen.get_state(), local.act_sample_gen._int_gen.get_state())
    assert (b[:, :8] - seed).abs().amax() <= 0.201
    torch.testing.assert_close(b[:, :1], seed)
    explicit = seed.expand(1, 32, -1).clone()
    torch.testing.assert_close(local._generate_seed_configs(1, explicit), explicit)
    assert not local._local_seed_active


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("scene", [None, "collision_test.yml"])
def test_quality_planner_retains_native_checks_with_obstacles(scene: str | None) -> None:
    from curobo._src.motion.motion_planner import MotionPlanner
    from curobo._src.motion.motion_planner_cfg import MotionPlannerCfg
    from curobo._src.state.state_joint import JointState

    cfg = MotionPlannerCfg.create("franka.yml", scene_model=scene,
        prefer_smooth_motion=True, quality_refinement_iters=20,
        num_ik_seeds=16, num_trajopt_seeds=2, use_cuda_graph=True)
    planner = MotionPlanner(cfg)
    try:
        start = JointState.from_position(planner.default_joint_state.position.unsqueeze(0),
                                        joint_names=planner.joint_names)
        endpoint = start.clone()
        endpoint.position[..., 0] += 0.1
        goal = planner.compute_kinematics(endpoint).tool_poses.as_goal()
        result = planner.plan_pose(goal, start, max_attempts=3, timeout_s=10)
        assert result is not None and bool(result.success.any())
        assert result.position_error.max() < cfg.trajopt_solver_config.position_tolerance
        assert result.rotation_error.max() < cfg.trajopt_solver_config.orientation_tolerance
        trajectory = result.get_interpolated_plan()
        active = planner.ik_solver.get_active_js(trajectory).position
        limits = cfg.ik_solver_config.robot_config.kinematics.kinematics_config.joint_limits.position
        assert (active >= limits[0] - 1e-5).all() and (active <= limits[1] + 1e-5).all()
        assert cfg.ik_solver_config.optimizer_configs[-1].num_iters == 100
    finally:
        planner.destroy()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_no_endpoint_keeps_full_optimizer_budget_and_rejects_unreachable_goal() -> None:
    from curobo._src.motion.motion_planner_cfg import MotionPlannerCfg
    from curobo._src.solver.solver_ik import IKSolver
    from curobo._src.state.state_joint import JointState
    from curobo._src.types.pose import Pose
    from curobo._src.types.tool_pose import GoalToolPose

    cfg = MotionPlannerCfg.create("franka.yml", prefer_smooth_motion=True,
        quality_refinement_iters=20, num_ik_seeds=16, use_cuda_graph=True)
    solver = IKSolver(cfg.ik_solver_config)
    try:
        start = JointState.from_position(solver.default_joint_state.position.unsqueeze(0),
                                        joint_names=solver.joint_names)
        pose = Pose(position=torch.tensor([[20.0, 0.0, 0.0]], device="cuda"),
                    quaternion=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda"))
        goal = GoalToolPose.from_poses({solver.kinematics.tool_frames[0]: pose},
                                     ordered_tool_frames=solver.kinematics.tool_frames)
        iterations = []
        optimizer = solver.optimizer.optimizers[-1]
        original = optimizer.optimize
        def observed(actions: torch.Tensor) -> torch.Tensor:
            iterations.append(optimizer.config.num_iters)
            return original(actions)
        optimizer.optimize = observed
        result = solver.solve_pose(goal, current_state=start)
        assert not bool(result.success.any())
        assert iterations == [100]
        assert optimizer.config.num_iters == 100
        assert "failed_ik" in result.debug_info
    finally:
        solver.destroy()
