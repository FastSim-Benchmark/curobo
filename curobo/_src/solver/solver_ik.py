# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inverse Kinematics solver for reaching target tool-frame poses.

Wraps SolverCore with IK-specific logic: seed generation via a Levenberg-Marquardt
seed solver, multi-link goal handling, batch padding, and solution ranking by pose
error. Supports goalset IK, velocity-aware IK via optimization_dt, and optional
collision checking.
"""
from __future__ import annotations

# Standard Library
from dataclasses import replace
from typing import Dict, Optional

# Third Party
import torch
import torch.autograd.profiler as profiler

from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria

# CuRobo
from curobo._src.geom.collision.collision_scene import SceneCollision
from curobo._src.geom.types import SceneCfg
from curobo._src.robot.kinematics.kinematics import Kinematics, KinematicsState
from curobo._src.rollout.goal_registry import GoalRegistry
from curobo._src.rollout.metrics import RolloutMetrics
from curobo._src.solver.ik_diagnostics import ik_failure_diagnostics
from curobo._src.solver.seed_ik.seed_ik_solver import SeedIKSolver
from curobo._src.solver.seed_ik.seed_ik_solver_cfg import SeedIKSolverCfg
from curobo._src.solver.solve_mode import SolveMode
from curobo._src.solver.solve_state import SolveState
from curobo._src.solver.solver_core import SolverCore
from curobo._src.solver.solver_ik_cfg import IKSolverCfg
from curobo._src.solver.solver_ik_result import IKSolverResult
from curobo._src.solver.solver_motion_quality import (
    combine_independent_tool_candidates,
    create_motion_quality_kinematics,
    height_overshoot,
    motion_quality_cost,
    motion_quality_terms,
    posture_motion_penalty,
    select_motion_candidates,
)
from curobo._src.state.state_joint import JointState
from curobo._src.state.state_robot import RobotState
from curobo._src.types.tool_pose import GoalToolPose, ToolPose
from curobo._src.util.cuda_event_timer import CudaEventTimer
from curobo._src.util.logging import log_and_raise, log_warn
from curobo._src.util.tensor_util import stable_topk
from curobo._src.util.torch_util import get_torch_jit_decorator


def _pad_batch_inputs(
    goal_tool_poses: GoalToolPose,
    current_state: Optional[JointState],
    seed_config: Optional[torch.Tensor],
    batch_size: int,
    max_batch: int,
):
    """Pad inputs from batch_size to max_batch by repeating the first element."""
    pad = max_batch - batch_size
    goal_tool_poses = goal_tool_poses.clone()
    pos = goal_tool_poses.position
    quat = goal_tool_poses.quaternion
    expand_dims = [-1] * (pos.ndim - 1)
    goal_tool_poses.position = torch.cat(
        [pos, pos[:1].expand(pad, *expand_dims)], dim=0,
    )
    goal_tool_poses.quaternion = torch.cat(
        [quat, quat[:1].expand(pad, *[-1] * (quat.ndim - 1))], dim=0,
    )
    if current_state is not None:
        current_state = current_state.clone()
        current_state.position = torch.cat(
            [current_state.position, current_state.position[:1].expand(pad, -1)], dim=0,
        )
        if current_state.velocity is not None:
            current_state.velocity = torch.cat(
                [current_state.velocity, current_state.velocity[:1].expand(pad, -1)], dim=0,
            )
        if current_state.acceleration is not None:
            current_state.acceleration = torch.cat(
                [current_state.acceleration, current_state.acceleration[:1].expand(pad, -1)], dim=0,
            )
    if seed_config is not None:
        seed_config = torch.cat(
            [seed_config, seed_config[:1].expand(pad, *[-1] * (seed_config.ndim - 1))], dim=0,
        )
    return goal_tool_poses, current_state, seed_config


def _slice_batch_result(result, batch_size: int):
    """Slice padded result tensors back to the original batch size."""
    result.success = result.success[:batch_size]
    for attr in ("solution", "position_error", "rotation_error", "goalset_index",
                 "feasible", "optimized_seeds", "total_cost_reshaped",
                 "seed_rank", "seed_cost"):
        val = getattr(result, attr, None)
        if val is not None:
            setattr(result, attr, val[:batch_size])
    if result.js_solution is not None:
        result.js_solution = result.js_solution[:batch_size]
    if result.solution_state is not None:
        result.solution_state = result.solution_state[:batch_size]
    return result


class IKSolver:
    """Inverse Kinematics solver."""

    def __init__(
        self, config: IKSolverCfg, scene_collision_checker: Optional[SceneCollision] = None
    ):
        """Initializes the IK solver."""
        self.config = config
        self.core = SolverCore(config.core_cfg, scene_collision_checker)
        self._posture_motion_enabled = getattr(config, "_posture_motion_enabled", True)
        self._quality_kinematics = (
            (create_motion_quality_kinematics(config.robot_config.kinematics)
             if self._posture_motion_enabled else
             Kinematics(config.robot_config.kinematics, compute_spheres=False))
            if config.prefer_smooth_motion else None
        )
        self._independent_tool_joints = None
        self._local_tool_solvers: list[SeedIKSolver] = []
        if config.prefer_smooth_motion and config.max_goalset == 1:
            params = config.robot_config.kinematics.kinematics_config
            tools = len(params.tool_frames)
            masks = params.joint_affects_endeffector.reshape(-1, tools).T.bool()
            if tools > 1 and not bool((masks.sum(dim=0) > 1).any()):
                self._independent_tool_joints = masks
                for tool, name in enumerate(params.tool_frames):
                    single = params.clone()
                    single.tool_frames = [name]
                    single.tool_frame_map = params.tool_frame_map[tool : tool + 1].clone()
                    single.joint_affects_endeffector = (
                        params.joint_affects_endeffector.reshape(-1, tools)[:, tool : tool + 1]
                        .flatten().clone()
                    )
                    kin = replace(
                        config.robot_config.kinematics,
                        tool_frames=[name], kinematics_config=single,
                    )
                    robot = replace(config.robot_config, kinematics=kin, dynamics=None)
                    self._local_tool_solvers.append(SeedIKSolver(SeedIKSolverCfg.create(
                        robot=robot, num_seeds=config.num_seeds,
                        max_iterations=40, inner_iterations=20,
                        use_cuda_graph=config.use_cuda_graph, device_cfg=config.device_cfg,
                        joint_limit_weight=1.0, lambda_initial=1.0,
                        path_regularization_weight=0.05,
                        path_regularization_seeds=config.num_seeds,
                    )))
        rows = torch.arange(config.num_seeds, **config.device_cfg.as_torch_dict())[:, None]
        columns = torch.arange(1, self.action_dim + 1, **config.device_cfg.as_torch_dict())[None]
        self._tool_seed_offsets = 0.2 * torch.sin(rows * columns * 2.39996323)

        self.seed_ik_solver = None

        # create an instance of seed ik solver:
        seed_solver_num_seeds = config.seed_solver_num_seeds
        max_iterations = 16
        tile_threads = 64

        if self.config.override_iters_for_multi_link_ik is not None:
            for k in self.optimizer.solver_names:
                if "lbfgs" in k:
                    index = self.optimizer.solver_names.index(k)
                    if (
                        self.config.optimizer_configs[index].num_iters
                        < self.config.override_iters_for_multi_link_ik
                    ):
                        self.optimizer.update_solver_params(
                            {k: {"num_iters": self.config.override_iters_for_multi_link_ik}}
                        )
                        log_warn(
                            f"IKSolver: increasing number of optimization iterations "
                            f"to {self.config.optimizer_configs[index].num_iters} for multi-link IK."
                        )

        # if number of goal links is more than 1, we increase number of optimization iterations
        if len(self.kinematics.tool_frames) > 1:
            seed_solver_num_seeds = 128
            max_iterations = 20
        if len(self.kinematics.tool_frames) > 2:
            seed_solver_num_seeds = 64
            max_iterations = 30
            tile_threads = 256

        if self.config.num_seeds > seed_solver_num_seeds:
            seed_solver_num_seeds = self.config.num_seeds * 2

        if self.config.use_lm_seed:
            self.seed_ik_solver = SeedIKSolver(
                SeedIKSolverCfg.create(
                    robot=self.config.robot_config,
                    num_seeds=seed_solver_num_seeds,
                    max_iterations=max_iterations,
                    inner_iterations=max_iterations,
                    joint_limit_weight=1.0,
                    lambda_initial=1.0,
                    lambda_factor=2.0,
                    lambda_max=1.0e10,
                    lambda_min=1e-5,
                    rho_min=1e-5,
                    use_backward=True,
                    use_cuda_graph=self.config.use_cuda_graph,
                    tile_threads=tile_threads,
                    device_cfg=self.config.device_cfg,
                    position_weight=self.config.seed_position_weight,
                    orientation_weight=self.config.seed_orientation_weight,
                    velocity_weight=self.config.seed_velocity_weight,
                    acceleration_weight=self.config.seed_acceleration_weight,
                    path_regularization_weight=(0.05 if config.prefer_smooth_motion else 0.0),
                    path_regularization_seeds=min(8, self.config.num_seeds // 2),
                    local_seed_count=(
                        min(8, self.config.num_seeds // 2) if config.prefer_smooth_motion else 0
                    ),
                )
            )

    # -------------------------------------------------------------------
    # Delegated properties and methods from SolverCore
    # -------------------------------------------------------------------

    @property
    def optimizer(self):
        return self.core.optimizer

    @property
    def metrics_rollout(self):
        return self.core.metrics_rollout

    @property
    def auxiliary_rollout(self):
        return self.core.auxiliary_rollout

    @property
    def kinematics(self):
        return self.core.kinematics

    @property
    def transition_model(self):
        return self.core.transition_model

    @property
    def action_dim(self) -> int:
        return self.core.action_dim

    @property
    def action_horizon(self) -> int:
        return self.core.action_horizon

    @property
    def joint_names(self):
        return self.core.joint_names

    @property
    def tool_frames(self):
        return self.core.tool_frames

    @property
    def default_joint_position(self):
        return self.core.default_joint_position

    @property
    def default_joint_state(self):
        return self.core.default_joint_state

    @property
    def device_cfg(self):
        return self.core.device_cfg

    @property
    def scene_collision_checker(self):
        return self.core.scene_collision_checker

    @property
    def goal_registry_manager(self):
        return self.core.goal_registry_manager

    @property
    def solve_state(self) -> SolveState:
        return self.core.solve_state

    @property
    def seed_manager(self):
        return self.core.seed_manager

    def get_all_rollout_instances(self, **kwargs):
        return self.core.get_all_rollout_instances(**kwargs)

    def compute_kinematics(self, state: JointState) -> KinematicsState:
        """Run forward kinematics and return tool poses and collision spheres."""
        return self.core.compute_kinematics(state)

    def get_active_js(self, full_js: JointState) -> JointState:
        return self.core.get_active_js(full_js)

    def get_full_js(self, active_js: JointState) -> JointState:
        return self.core.get_full_js(active_js)

    def sample_configs(self, num_samples: int, rejection_ratio: int = 10) -> torch.Tensor:
        return self.core.sample_configs(
            num_samples, rejection_ratio, self.config.optimizer_collision_activation_distance
        )

    def update_link_inertial(self, link_name, mass=None, com=None, inertia=None):
        return self.core.update_link_inertial(link_name, mass, com, inertia)

    def update_links_inertial(self, link_properties):
        return self.core.update_links_inertial(link_properties)

    def debug_dump(self, file_path: str):
        return self.core.debug_dump(file_path)

    def prepare_action_seeds(self, batch_size, num_seeds, seed_config=None, current_state=None, seed_traj=None):
        return self.core.prepare_action_seeds(batch_size, num_seeds, seed_config, current_state, seed_traj)

    def prepare_trajectory_seeds(self, batch_size, num_seeds, current_state, seed_config=None, seed_traj=None):
        return self.core.prepare_trajectory_seeds(batch_size, num_seeds, current_state, seed_config, seed_traj)

    def enable_tool_pose_tracking(self, tool_frames=None):
        """Enable tool-pose cost terms for the specified (or all) tool frames."""
        self.core.enable_tool_pose_tracking(
            tool_frames, self.config.non_terminal_tool_pose_weight_factor
        )

    def disable_tool_pose_tracking(self, tool_frames=None):
        self.core.disable_tool_pose_tracking(tool_frames)

    def enable_joint_position_tracking(self):
        self.core.enable_joint_position_tracking()

    def disable_joint_position_tracking(self):
        self.core.disable_joint_position_tracking()

    def reset_shape(self):
        self.core.reset_shape()

    def reset_cuda_graph(self):
        self.core.reset_cuda_graph()

    # -------------------------------------------------------------------
    # IK-specific: tool pose criteria also updates seed solver
    # -------------------------------------------------------------------

    def update_tool_pose_criteria(self, tool_pose_criteria: Dict[str, ToolPoseCriteria]):
        if self.seed_ik_solver is not None:
            self.seed_ik_solver.update_tool_pose_criteria(tool_pose_criteria)
        self.core.update_tool_pose_criteria(tool_pose_criteria)

    def reset_seed(self):
        if self.config.use_lm_seed:
            self.seed_ik_solver.reset_seed()
        self.core.reset_seed()

    def destroy(self):
        for solver in self._local_tool_solvers:
            solver.destroy()
        if self.seed_ik_solver is not None:
            self.seed_ik_solver.destroy()
        self.core.destroy()

    # -------------------------------------------------------------------
    # Goal buffer
    # -------------------------------------------------------------------

    @profiler.record_function("ik_solver/prepare_goal_buffer")
    def _prepare_goal_buffer(
        self,
        solve_state: SolveState,
        goal_tool_poses: GoalToolPose,
        current_state: Optional[JointState] = None,
        use_implicit_goal: bool = False,
        seed_goal_state: Optional[JointState] = None,
        goal_state: Optional[JointState] = None,
    ):
        goal_tool_poses = goal_tool_poses.reorder_links(self.kinematics.tool_frames)
        goal_buffer, update_reference = self.core.prepare_goal_buffer(
            solve_state=solve_state,
            goal_tool_poses=goal_tool_poses,
            current_state=current_state,
            use_implicit_goal=use_implicit_goal,
            seed_goal_state=seed_goal_state,
            goal_state=goal_state,
        )
        return goal_buffer, update_reference

    def _update_rollout_params(self, goal_buffer, include_auxiliary_rollout=True):
        self.core.update_rollout_params(goal_buffer, include_auxiliary_rollout)

    @property
    def problem_batch_size(self) -> int:
        return self.core._solve_state.get_ik_batch_size()

    # -------------------------------------------------------------------
    # IK-specific methods
    # -------------------------------------------------------------------

    def _fill_current_state_dt(
        self, current_state: Optional[JointState]
    ) -> Optional[JointState]:
        """Fill ``current_state.dt`` from ``config.optimization_dt`` when not already set."""
        if (
            current_state is not None
            and current_state.dt is None
            and self.config.optimization_dt is not None
        ):
            current_state = current_state.clone()
            current_state.dt = torch.full(
                (current_state.position.shape[0],),
                self.config.optimization_dt,
                device=current_state.device,
                dtype=current_state.dtype,
            )
        return current_state

    @profiler.record_function("ik_solver/solve_impl")
    def _solve_impl(
        self,
        solve_state: SolveState,
        goal_tool_poses: GoalToolPose,
        num_seeds: int,
        current_state: Optional[JointState] = None,
        seed_config=None,
        return_seeds: int = 1,
        run_optimizer: bool = True,
        prefer_smooth_motion: Optional[bool] = None,
    ) -> IKSolverResult:
        goal_tool_poses = goal_tool_poses.reorder_links(self.kinematics.tool_frames)
        total_timer = CudaEventTimer().start()
        if goal_tool_poses.ndim != 5:
            log_and_raise(
                f"goal_tool_poses has shape {goal_tool_poses.shape}. "
                f"Expected 5D [batch, horizon, num_links, num_goalset, 3/4]."
            )

        current_state = self._fill_current_state_dt(current_state)

        goal_buffer, update_reference = self._prepare_goal_buffer(
            solve_state=solve_state,
            goal_tool_poses=goal_tool_poses,
            current_state=current_state,
        )

        coord_position_seed = self.prepare_action_seeds(
            solve_state.batch_size, num_seeds, seed_config=seed_config
        )

        self._update_rollout_params(goal_buffer, include_auxiliary_rollout=False)

        use_quality = (
            self.config.prefer_smooth_motion
            if prefer_smooth_motion is None else prefer_smooth_motion
        )
        if use_quality and current_state is not None and run_optimizer:
            return self._solve_motion_quality(
                coord_position_seed, current_state, num_seeds, return_seeds, total_timer,
                goal_tool_poses,
            )

        if self.config.exit_early or not run_optimizer:
            metrics_result = self.metrics_rollout.compute_metrics_from_action(coord_position_seed)
            ik_result = self._get_result(
                coord_position_seed, metrics_result, num_seeds, return_seeds
            )
            success_ratio = torch.count_nonzero(ik_result.success) / ik_result.success.shape[0]
            if success_ratio >= self.config.exit_early_batch_success_threshold or not run_optimizer:
                self._compute_solution_velocity(ik_result, current_state, return_seeds)
                ik_result.total_time = total_timer.stop()
                return ik_result

        opt_timer = CudaEventTimer().start()
        self.optimizer.reinitialize(coord_position_seed)
        opt_result = self.optimizer.optimize(coord_position_seed)
        opt_time = opt_timer.stop()

        metrics_result = self.metrics_rollout.compute_metrics_from_action(opt_result)

        ik_result = self._get_result(opt_result, metrics_result, num_seeds, return_seeds)
        self._compute_solution_velocity(ik_result, current_state, return_seeds)
        ik_result.total_time = total_timer.stop()
        ik_result.solve_time = opt_time
        return ik_result

    def _endpoint_motion_cost(
        self, actions: torch.Tensor, current_state: JointState
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Evaluate nine-point joint-interpolation drafts [batch, seeds, dof]."""
        batch, seeds, dof = actions.shape
        start = self.get_active_js(current_state).position.view(batch, 1, 1, dof)
        fraction = torch.linspace(0, 1, 9, device=actions.device, dtype=actions.dtype)
        path = start + fraction[None, None, :, None] * (actions[:, :, None] - start)
        state = self._quality_kinematics.compute_kinematics(
            JointState.from_position(
                path.reshape(batch * seeds, 9, dof), joint_names=self.joint_names
            )
        )
        positions = state.tool_poses.position.view(batch, seeds, 9, -1, 3)
        count = len(self.tool_frames)
        tools = positions[..., :count, :]
        rotations = state.tool_poses.quaternion.view(batch, seeds, 9, -1, 4)[..., :count, :]
        per_tool = motion_quality_terms(tools, rotations)
        cost = motion_quality_cost(tools, path, rotations)
        large_motion = None
        if self._posture_motion_enabled:
            travel = torch.linalg.vector_norm(tools.diff(dim=2), dim=-1).sum(dim=2)
            chord = torch.linalg.vector_norm(tools[:, :, -1] - tools[:, :, 0], dim=-1)
            large_motion = ((travel > .5) & (travel - chord > .25)).any(dim=-1)
        if positions.shape[-2] > count:
            heights = positions[..., count:, 2]
            cost = cost + posture_motion_penalty(heights)
            per_tool = per_tool + 4. * (height_overshoot(heights) - .05).clamp_min(0.) / .05
        return cost, per_tool, large_motion

    def _solve_motion_quality(
        self,
        seeds: torch.Tensor,
        current_state: JointState,
        num_seeds: int,
        return_seeds: int,
        total_timer: CudaEventTimer,
        goal_tool_poses: GoalToolPose,
    ) -> IKSolverResult:
        """Improve poor feasible seeds without discarding the original alternatives."""
        batch = seeds.shape[0] // num_seeds
        metrics = self.metrics_rollout.compute_metrics_from_action(seeds)
        old = self._get_result(seeds, metrics, num_seeds, num_seeds)
        # _get_result returns native-ranked actions, not the input seed order.
        old_actions = old.solution.clone()
        old_success = old.success.clone()
        tool_success = old.debug_info["tool_success"].clone()
        old_cost, old_tool_cost, old_large = self._endpoint_motion_cost(old_actions, current_state)
        best = select_motion_candidates(old_actions, old_success, old_cost, 1)
        best_cost = torch.gather(old_cost, 1, best).squeeze(1)
        good = old_success.any(dim=1) & (best_cost <= 0.15)
        refined = not (self.config.exit_early and bool(good.all()))
        opt_time = 0.0
        actions, successful, costs = old_actions, old_success, old_cost
        tool_cost = old_tool_cost
        if refined:
            stages = [(stage, stage.config.num_iters) for stage in self.optimizer.optimizers]
            # No feasible endpoint: retain the original global optimizer budget.
            short = bool(old_success.any(dim=1).all())
            # A large draft detour or elbow rise merits the existing full IK
            # budget. Normal drafts retain bounded refinement and seed count.
            if self._posture_motion_enabled:
                large_best = torch.gather(old_large, 1, best).squeeze(1)
                short = short and not bool((large_best & (best_cost > 4.)).any())
            opt_timer = CudaEventTimer().start()
            try:
                if short:
                    for stage, iterations in stages:
                        inner = stage.config.inner_iters
                        requested = max(
                            inner, self.config.quality_refinement_iters // inner * inner
                        )
                        stage.update_niters(min(iterations, requested))
                self.optimizer.reinitialize(seeds)
                optimized = self.optimizer.optimize(seeds)
            finally:
                for stage, iterations in stages:
                    stage.update_niters(iterations)
            opt_time = opt_timer.stop()
            metrics = self.metrics_rollout.compute_metrics_from_action(optimized)
            new = self._get_result(optimized, metrics, num_seeds, num_seeds)
            new_actions = new.solution.clone()
            new_cost, new_tool_cost, _ = self._endpoint_motion_cost(new_actions, current_state)
            actions = torch.cat([old_actions, new_actions], dim=1)
            successful = torch.cat([old_success, new.success], dim=1)
            tool_success = torch.cat([tool_success, new.debug_info["tool_success"]], dim=1)
            costs = torch.cat([old_cost, new_cost], dim=1)
            tool_cost = torch.cat([old_tool_cost, new_tool_cost], dim=1)
        if self._independent_tool_joints is not None:
            if refined:
                active_start = self.get_active_js(current_state)
                for tool, solver in enumerate(self._local_tool_solvers):
                    local_seed = active_start.position[:, None] + (
                        self._tool_seed_offsets[None] * self._independent_tool_joints[tool]
                    )
                    local_seed = torch.clamp(
                        local_seed, min=solver.action_min, max=solver.action_max
                    )
                    local_goal = GoalToolPose(
                        tool_frames=solver.tool_frames,
                        position=goal_tool_poses.position[:, :, tool : tool + 1],
                        quaternion=goal_tool_poses.quaternion[:, :, tool : tool + 1],
                    )
                    local = solver.solve_batch(local_goal, current_state=active_start,
                        seed_config=local_seed, return_seeds=solver.config.num_seeds)
                    local_actions = local.solution.clone()
                    local_cost, local_tool_cost, _ = self._endpoint_motion_cost(
                        local_actions, current_state
                    )
                    converged = torch.zeros(
                        (batch, solver.config.num_seeds, len(self._local_tool_solvers)),
                        dtype=torch.bool, device=seeds.device,
                    )
                    converged[:, :, tool] = local.success
                    actions = torch.cat([actions, local_actions], dim=1)
                    successful = torch.cat([successful, torch.zeros_like(local.success)], dim=1)
                    costs = torch.cat([costs, local_cost], dim=1)
                    tool_cost = torch.cat([tool_cost, local_tool_cost], dim=1)
                    tool_success = torch.cat([tool_success, converged], dim=1)
            combined = combine_independent_tool_candidates(
                actions, tool_success, tool_cost, self._independent_tool_joints, num_seeds
            )
            combined_actions = combined.reshape_as(seeds)
            metrics = self.metrics_rollout.compute_metrics_from_action(combined_actions)
            combined_result = self._get_result(combined_actions, metrics, num_seeds, num_seeds)
            combined = combined_result.solution.clone()
            combined_cost, _, _ = self._endpoint_motion_cost(combined, current_state)
            actions = torch.cat([actions, combined], dim=1)
            successful = torch.cat([successful, combined_result.success], dim=1)
            costs = torch.cat([costs, combined_cost], dim=1)
        # Reserve half the slots for old feasible endpoints. All slots still
        # use exactly the native endpoint and full-trajectory acceptance tests.
        indices = select_motion_candidates(
            actions, successful, costs, return_seeds,
            old_count=num_seeds if refined else 0,
            reserve_old=return_seeds // 2 if refined else 0,
        )
        selected = torch.gather(actions, 1, indices[..., None].expand(-1, -1, self.action_dim))
        padded = selected[:, :1].expand(batch, num_seeds, self.action_dim).clone()
        padded[:, :return_seeds] = selected
        final_actions = padded.reshape_as(seeds)
        metrics = self.metrics_rollout.compute_metrics_from_action(final_actions)
        selection = torch.arange(return_seeds, device=seeds.device).expand(batch, -1)
        result = self._get_result(final_actions, metrics, num_seeds, return_seeds, selection)
        result.debug_info["motion_quality"] = {
            "refined": refined, "proxy_cost": torch.gather(costs, 1, indices),
            "extended_quality_refinement": (refined and self._posture_motion_enabled
                                             and bool(old_success.any(dim=1).all())
                                             and not short),
        }
        self._compute_solution_velocity(result, current_state, return_seeds)
        result.total_time = total_timer.stop()
        result.solve_time = opt_time
        return result

    def get_unique_solution(self, roundoff_decimals: int = 2) -> torch.Tensor:
        """Get unique solutions from many feasible solutions for the same problem.

        Filters :attr:`solution` by :attr:`success`, rounds to
        *roundoff_decimals* to merge near-duplicate configurations, and
        returns one representative per unique rounded configuration.

        Args:
            roundoff_decimals: Number of decimal places used when rounding
                joint values to determine uniqueness. A smaller value merges
                more solutions together.

        Returns:
            Tensor of shape ``[num_unique, dof]`` containing the unique
            feasible joint configurations.
        """
        in_solution = self.solution[self.success]
        r_sol = torch.round(in_solution, decimals=roundoff_decimals)
        if not (len(in_solution.shape) == 2):
            log_and_raise("Solution shape is not of length 2")
        s, i = torch.unique(r_sol, dim=-2, return_inverse=True)
        sol = in_solution[i[: s.shape[0]]]
        return sol

    @profiler.record_function("ik_solver/_get_result")
    @get_torch_jit_decorator(only_valid_for_compile=True, slow_to_compile=True)
    def _get_result(
        self,
        opt_result: torch.Tensor,
        metrics_result: RolloutMetrics,
        num_seeds: int,
        return_seeds: int,
        selection_indices: Optional[torch.Tensor] = None,
    ) -> IKSolverResult:
        feasible = metrics_result.costs_and_constraints.get_feasible(
            include_all_hybrid=False,
            sum_horizon=True,
        )

        converge_list = []
        cost_list = []
        goalset_index_list = []
        pose_cost_list = []
        for k in range(len(metrics_result.convergence.names)):
            metric_name = metrics_result.convergence.names[k]
            metric_values = metrics_result.convergence.values[k]

            if "position_tolerance" in metric_name:
                pose_cost_list.append(metric_values)
                converged = metric_values < self.config.position_tolerance
                converge_list.append(converged)
            elif "orientation_tolerance" in metric_name:
                pose_cost_list.append(metric_values)
                converged = metric_values < self.config.orientation_tolerance
                converge_list.append(converged)
            elif "goalset_index" in metric_name:
                goalset_index_list.append(metric_values)
            elif "start_cspace_dist_tolerance" in metric_name:
                cost_list.append(metric_values.view(-1, 1))

        converged_all_links = torch.cat(converge_list, dim=-1)
        converged = torch.all(converged_all_links, dim=-1).squeeze(-1)

        if isinstance(feasible, bool):
            success = converged
            feasible_tensor = torch.ones_like(converged) if feasible else torch.zeros_like(converged)
        else:
            if self.config.success_requires_convergence:
                success = torch.logical_and(converged, feasible)
            else:
                success = feasible
            feasible_tensor = feasible

        goalset_index = (
            torch.cat(goalset_index_list, dim=-1) if goalset_index_list else None
        )
        pose_cost = torch.cat(pose_cost_list, dim=-1).unsqueeze(dim=1)

        if len(cost_list) > 0:
            cost = torch.cat(cost_list, dim=-1).unsqueeze(dim=1) + pose_cost.sum(dim=-1)
            cost = cost.squeeze(-1)
        else:
            cost = pose_cost.sum(dim=-1).squeeze(-1)
        cost_sum = cost

        cost_sum[~success] += 1e16
        cost_sum_reshaped = cost_sum.view(-1, num_seeds)
        batch_size = cost_sum_reshaped.shape[0]

        topk_values, topk_relative_idx = stable_topk(
            cost_sum_reshaped, k=return_seeds, largest=False, dim=-1
        )
        if selection_indices is not None:
            topk_relative_idx = selection_indices

        topk_abs_idx = (
            topk_relative_idx + num_seeds * self.goal_registry_manager.batch_helper
        )

        opt_result_flat = opt_result.view(-1, self.action_dim)
        q_sol = opt_result_flat[topk_abs_idx.view(-1)].view(batch_size, return_seeds, self.action_dim)
        success_topk = success.view(-1)[topk_abs_idx.view(-1)].view(batch_size, return_seeds)
        feasible_topk = feasible_tensor.view(-1)[topk_abs_idx.view(-1)].view(
            batch_size, return_seeds
        )
        all_state = metrics_result.state
        num_links = len(all_state.cuda_robot_model_state.tool_poses.tool_frames)
        num_spheres = all_state.cuda_robot_model_state.robot_spheres.shape[-2]
        solution_state = RobotState(
            cuda_robot_model_state=KinematicsState(
                tool_poses=ToolPose(
                    tool_frames=all_state.cuda_robot_model_state.tool_poses.tool_frames,
                    position=all_state.cuda_robot_model_state.tool_poses.position.view(
                        -1, num_links, 3
                    )[topk_abs_idx.view(-1)].view(batch_size, return_seeds, num_links, 3),
                    quaternion=all_state.cuda_robot_model_state.tool_poses.quaternion.view(
                        -1, num_links, 4
                    )[topk_abs_idx.view(-1)].view(batch_size, return_seeds, num_links, 4),
                ),
                robot_spheres=all_state.cuda_robot_model_state.robot_spheres.flatten(0, 1)[
                    topk_abs_idx.view(-1)
                ].view(batch_size, return_seeds, num_spheres, 4),
                robot_com=all_state.cuda_robot_model_state.robot_com.view(-1, 4)[
                    topk_abs_idx.view(-1)
                ].view(batch_size, return_seeds, -1, 4),
            ),
            joint_state=JointState.from_position(q_sol, joint_names=self.joint_names),
        )

        pose_error_flat = pose_cost.view(-1, pose_cost.shape[-1])
        pose_error_topk = pose_error_flat[topk_abs_idx.view(-1)]
        num_links = int(pose_error_topk.shape[-1] / 2)
        pose_error_topk = pose_error_topk.view(batch_size, return_seeds, num_links, 2)

        goalset_index_topk = None
        if goalset_index is not None:
            goalset_index_flat = goalset_index.view(-1, goalset_index.shape[-1])
            goalset_index_topk = goalset_index_flat[topk_abs_idx.view(-1)].view(
                batch_size, return_seeds, num_links
            )

        position_error = torch.max(pose_error_topk[..., 0], dim=-1)[0]
        rotation_error = torch.max(pose_error_topk[..., 1], dim=-1)[0]

        js_solution = solution_state.joint_state
        js_solution = self.auxiliary_rollout.transition_model.get_full_dof_from_solution(
            js_solution
        )

        ik_result = IKSolverResult(
            success=success_topk,
            feasible=feasible_topk,
            solution=q_sol,
            js_solution=js_solution,
            position_error=position_error,
            rotation_error=rotation_error,
            goalset_index=(
                goalset_index_topk
                if goalset_index_topk is not None
                else torch.full_like(position_error, -1, dtype=torch.long)
            ),
            solve_time=0.0,
            debug_info={
                "seed_idx": topk_abs_idx,
                "tool_success": (pose_error_topk[..., 0] < self.config.position_tolerance)
                & (pose_error_topk[..., 1] < self.config.orientation_tolerance),
            },
            optimized_seeds=opt_result.view(
                batch_size, num_seeds, self.action_dim
            ),
            solution_state=solution_state,
        )

        if not bool(success.any()):
            ik_result.debug_info["failed_ik"] = {
                "seed_count": success.numel(),
                "feasible_seed_count": feasible_tensor.count_nonzero(),
                "converged_seed_count": converged.count_nonzero(),
                "successful_seed_count": success.count_nonzero(),
            }
        if self.config.detailed_failure_diagnostics and not bool(success.any()):
            manager = self.metrics_rollout.metrics_constraint_manager
            config = (
                manager.get_cost("cspace").config
                if manager is not None and manager.has_cost("cspace") else None
            )
            self_collision_config = (
                manager.get_cost("self_collision").config.self_collision_kin_config
                if manager is not None and manager.has_cost("self_collision") else None
            )
            ik_result.debug_info["failed_ik"] = ik_failure_diagnostics(
                metrics_result, feasible_tensor, converged, success, topk_abs_idx, config,
                self_collision_config=self_collision_config,
                kinematics_config=(
                    self.kinematics.config.kinematics_config
                    if self_collision_config is not None else None
                ),
                scene_collision_checker=self.scene_collision_checker,
            )

        return ik_result

    def _compute_solution_velocity(
        self,
        ik_result: IKSolverResult,
        current_state: Optional[JointState],
        return_seeds: int,
    ) -> None:
        """Set ``js_solution.velocity`` via finite difference when ``optimization_dt`` is set."""
        if (
            self.config.optimization_dt is None
            or current_state is None
            or current_state.dt is None
            or ik_result.js_solution is None
        ):
            return

        active_cur = self.get_active_js(current_state)
        active_sol = self.get_active_js(ik_result.js_solution)

        cur_pos = active_cur.position
        sol_pos = active_sol.position
        dt = current_state.dt

        cur_pos_exp = cur_pos.unsqueeze(1)
        dt_val = dt.view(-1, 1, 1) if dt.ndim == 1 else dt[:, :1].unsqueeze(-1)

        active_vel = (sol_pos - cur_pos_exp) / dt_val

        if sol_pos.shape[-1] == ik_result.js_solution.position.shape[-1]:
            ik_result.js_solution.velocity = active_vel
        else:
            full_vel = torch.zeros_like(ik_result.js_solution.position)
            active_indices = [
                ik_result.js_solution.joint_names.index(j)
                for j in self.joint_names
            ]
            full_vel[..., active_indices] = active_vel
            ik_result.js_solution.velocity = full_vel

    # -------------------------------------------------------------------
    # Public solve methods
    # -------------------------------------------------------------------

    @profiler.record_function("ik_solver/solve_pose")
    def solve_pose(
        self,
        goal_tool_poses: GoalToolPose,
        current_state: Optional[JointState] = None,
        seed_config=None,
        return_seeds: int = 1,
        run_optimizer: bool = True,
        prefer_smooth_motion: Optional[bool] = None,
    ) -> IKSolverResult:
        """Solve IK for target tool poses.

        Args:
            goal_tool_poses: Target poses for tool links.
            current_state: Optional current joint state.
            seed_config: Optional initial seeds.
            return_seeds: Number of top solutions to return per problem.
            run_optimizer: When False, skip the main optimizer.
            prefer_smooth_motion: Override endpoint ranking for this call. When False,
                LM optimization with a current state and no explicit seeds uses that
                state plus fresh global samples, without local reservations or the
                soft LM path prior. Explicit seeds retain their existing semantics.
                The override does not mutate configuration.

        Returns:
            IKSolverResult with solutions ranked by cost.
        """
        goal_tool_poses = goal_tool_poses.reorder_links(self.kinematics.tool_frames)

        max_batch = self.config.max_batch_size
        batch_size = goal_tool_poses.batch_size
        timer = CudaEventTimer().start()
        num_seeds = self.config.num_seeds
        if return_seeds > num_seeds:
            log_warn(
                f"Requested {return_seeds} solutions, increasing optimization seeds "
                f"from {num_seeds} to {return_seeds}."
            )
            num_seeds = return_seeds

        num_goalset = goal_tool_poses.num_goalset
        if num_goalset > self.config.max_goalset:
            log_and_raise(
                f"solve_pose: num_goalset={num_goalset} exceeds config.max_goalset="
                f"{self.config.max_goalset}."
            )

        if batch_size > max_batch:
            log_and_raise(
                f"solve_pose: goal_tool_poses.batch_size={batch_size} "
                f"exceeds config.max_batch_size={max_batch}."
            )

        actual_batch_size = batch_size
        needs_pad = batch_size < max_batch
        if needs_pad:
            goal_tool_poses, current_state, seed_config = _pad_batch_inputs(
                goal_tool_poses, current_state, seed_config, batch_size, max_batch,
            )
        batch_size = max_batch

        if batch_size == 1:
            solve_mode = SolveMode.SINGLE
        elif self.config.multi_env:
            solve_mode = SolveMode.MULTI_ENV
        else:
            solve_mode = SolveMode.BATCH

        solve_state = SolveState(
            solve_type=solve_mode,
            num_ik_seeds=num_seeds,
            batch_size=batch_size,
            num_envs=batch_size if self.config.multi_env else 1,
            num_goalset=num_goalset,
            tool_frames=goal_tool_poses.tool_frames,
        )

        additional_time = 0.0
        current_state = self._fill_current_state_dt(current_state)

        fresh_global_seeds = (
            prefer_smooth_motion is False
            and seed_config is None
            and current_state is not None
            and self.config.use_lm_seed
            and run_optimizer
        )
        if seed_config is None and current_state is not None:
            seed_config = current_state.position.clone().view(batch_size, 1, -1)
        if fresh_global_seeds:
            # Supplying a complete bank bypasses LM's local injection and
            # reserved slots while retaining the current state as one seed.
            lm_count = self.seed_ik_solver.config.num_seeds
            start = self.get_active_js(current_state).position.view(batch_size, 1, -1)
            if lm_count > 1:
                samples = self.seed_ik_solver.act_sample_gen.get_samples(
                    batch_size * (lm_count - 1), bounded=True
                ).view(batch_size, lm_count - 1, self.action_dim)
                seed_config = torch.cat([start, samples], dim=1)
            else:
                seed_config = start.clone()

        # Straight-chord preferences do not model obstacle homotopy. Keep the
        # original endpoint objective whenever any scene has registered geometry.
        checker = self.scene_collision_checker
        path_cost_enabled = checker is None or not any(
            checker.get_obstacle_names(env) for env in range(checker.num_envs)
        )
        for seed_solver in [self.seed_ik_solver, *self._local_tool_solvers]:
            if seed_solver is not None:
                seed_solver._path_cost_enabled = path_cost_enabled and not fresh_global_seeds

        if self.config.use_lm_seed and run_optimizer:
            if batch_size == 1:
                seed_result = self.seed_ik_solver.solve_single(
                    goal_tool_poses=goal_tool_poses,
                    seed_config=seed_config,
                    return_seeds=self.config.num_seeds,
                    current_state=current_state,
                )
            else:
                seed_result = self.seed_ik_solver.solve_batch(
                    goal_tool_poses=goal_tool_poses,
                    seed_config=seed_config,
                    return_seeds=self.config.num_seeds,
                    current_state=current_state,
                )
            seed_config = seed_result.js_solution.position.view(
                batch_size, self.config.num_seeds, self.action_dim
            )
            additional_time = seed_result.solve_time

        if seed_config is not None:
            if seed_config.shape[0] != batch_size:
                log_and_raise(
                    f"solve_pose: seed_config batch dimension ({seed_config.shape[0]}) "
                    f"should match config.max_batch_size ({batch_size})."
                )

        if current_state is not None and current_state.shape[0] != batch_size:
            log_and_raise(
                f"solve_pose: current_state batch dimension ({current_state.shape[0]}) "
                f"should match config.max_batch_size ({batch_size})."
            )

        result = self._solve_impl(
            solve_state=solve_state,
            goal_tool_poses=goal_tool_poses,
            num_seeds=num_seeds,
            seed_config=seed_config,
            current_state=current_state,
            return_seeds=return_seeds,
            run_optimizer=run_optimizer,
            prefer_smooth_motion=prefer_smooth_motion,
        )
        if self.seed_ik_solver is not None and self.seed_ik_solver.global_seed_config is not None:
            result.debug_info["global_seed_config"] = self.seed_ik_solver.global_seed_config
        result.solve_time = timer.stop() + additional_time
        result.total_time = result.solve_time

        if needs_pad:
            result = _slice_batch_result(result, actual_batch_size)
        return result

    @profiler.record_function("ik_solver/update_world")
    def update_world(self, scene_cfg: SceneCfg) -> None:
        """Reload the collision model from a new scene configuration."""
        self.scene_collision_checker.load_collision_model(scene_cfg)
