# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""A failed optimizer must not discard a fully validated, retimed seed."""

from pathlib import Path

import pytest
import torch

from curobo.examples.reference.contact_separation import robot_config
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.scene import Cuboid, Scene
from curobo.types import AxisHold, JointState


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("use_cuda_graph", [False, True])
@pytest.mark.parametrize("blocked", [False, True])
def test_retime_clear_seed_but_reject_colliding_seed(
    tmp_path, monkeypatch, use_cuda_graph, blocked,
):
    scene = Scene(cuboid=[Cuboid(
        name="wall", dims=[0.04, 0.5, 0.5],
        pose=[0.1 if blocked else 4.0, 0, 0.2, 1, 0, 0, 0],
    )])
    config = MotionPlannerCfg.create(
        robot_config(tmp_path), scene_model=scene, num_ik_seeds=4,
        num_trajopt_seeds=2, use_cuda_graph=use_cuda_graph,
        trajopt_finetune_attempts=0,
    )
    with MotionPlanner(config) as planner:
        start = JointState.from_position(
            planner.default_joint_state.position.unsqueeze(0), planner.joint_names,
        )
        goal = start.clone()
        goal.position[:, 0] += 0.2
        calls = []

        def invalid_optimization(seed):
            calls.append(True)
            result = seed  # Recovery must preserve a copy even if an optimizer mutates input.
            result[:, 3:-3, 2] = 1.5  # Above the authored 0.6 m joint limit.
            return result

        monkeypatch.setattr(planner.trajopt_solver.optimizer, "optimize", invalid_optimization)
        result = planner.plan_cspace(
            goal, start, max_attempts=1, hold_axis={"gripper": AxisHold()},
        )
        assert calls == [True]
        assert result is not None
        assert bool(result.success.all()) is not blocked
        if blocked:
            recovery = result.debug_info["initial_seed_recovery"]
            assert not recovery["retimed_seed_success"]
            assert recovery["seed_constraint_maxima"]["interpolated"]["scene_collision"][
                "maximum"
            ] > 0
        else:
            assert result.debug_info["initial_seed_recovery"]["retimed_seed_success"]
            trajectory = result.get_interpolated_plan().reorder(planner.joint_names)
            positions = trajectory.position.reshape(-1, 3)
            torch.testing.assert_close(positions[0], start.position[0], atol=2e-6, rtol=0)
            torch.testing.assert_close(positions[-1], goal.position[0], atol=2e-6, rtol=0)
            assert (positions[:, 1:] - start.position[:, 1:]).abs().max() < 2e-6
            assert trajectory.velocity.abs().max() <= 0.3
            assert trajectory.acceleration.abs().max() <= 2.0
            assert trajectory.jerk.abs().max() <= 20.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("use_cuda_graph", [False, True])
@pytest.mark.parametrize("blocked", [False, True])
def test_partial_recovery_preserves_valid_path_and_native_collision_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, use_cuda_graph: bool, blocked: bool,
) -> None:
    """One surviving path must not hide a better seed or permit a colliding seed."""
    import curobo._src.solver.solver_trajopt as native_trajopt

    scene = Scene(cuboid=[Cuboid(name="wall", dims=[.04, .05, .5],
        pose=[.1 if blocked else 4., 0, .2, 1, 0, 0, 0])])
    config = MotionPlannerCfg.create(robot_config(tmp_path), scene_model=scene,
        num_ik_seeds=4, num_trajopt_seeds=2, use_cuda_graph=use_cuda_graph,
        trajopt_finetune_attempts=0, prefer_smooth_motion=True)
    with MotionPlanner(config) as planner:
        solver = planner.trajopt_solver
        start = JointState.from_position(planner.default_joint_state.position.unsqueeze(0),
                                         planner.joint_names)
        goal = start.clone()
        goal.position[:, 0] += .2
        evaluated, final = [], []
        native_get = solver._get_result
        native_rank = solver._get_best_result

        def mixed_optimization(seed: torch.Tensor) -> torch.Tensor:
            seed[0, 3:-3, 2] = 1.5
            phase = torch.linspace(0, torch.pi, seed.shape[1], device=seed.device)
            seed[1, :, 1] = .2 * phase.sin().square()
            return seed

        def capture_evaluated(*args: object, **kwargs: object) -> object:
            result = native_get(*args, **kwargs)
            evaluated.append((result.success.clone(), result.interpolated_trajectory.position.clone()))
            return result

        def capture_final(pool: object, count: int) -> object:
            final.append((pool.success.clone(), pool.interpolated_trajectory.position.clone()))
            return native_rank(pool, count)

        monkeypatch.setattr(solver.optimizer, "optimize", mixed_optimization)
        monkeypatch.setattr(solver, "_get_result", capture_evaluated)
        monkeypatch.setattr(solver, "_get_best_result", capture_final)
        # Force the quality trigger, while retaining every native geometric,
        # derivative and endpoint check and the real recovered seed trajectories.
        monkeypatch.setattr(native_trajopt, "motion_quality_cost",
            lambda position, joints, rotations: torch.full(joints.shape[:2], 10.,
                device=joints.device, dtype=joints.dtype))
        result = planner.plan_cspace(goal, start, max_attempts=1)
        assert result is not None and bool(result.success.any())
        assert evaluated[0][0].tolist() == [[False, True]]
        assert result.debug_info["initial_seed_recovery"]["partial_quality_recovery"]
        assert result.debug_info["initial_seed_recovery"]["retimed_seed_success"] is not blocked
        torch.testing.assert_close(final[0][1][:, 1], evaluated[0][1][:, 1], atol=0, rtol=0)
        assert final[0][0].tolist() == [[not blocked, True]]
