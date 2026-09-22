# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Bounded, request-local alternative seeds for partial-joint posture queries."""

from contextlib import contextmanager

import torch


def sample_posture_states(start, goals, limits, free, held, count, attempt, *, intermediate):
    """Sample limit-relative configurations without robot-specific joint names."""
    if count < 1 or attempt < 1:
        raise ValueError("posture sampling requires positive count and attempt")
    # Unscrambled Sobol avoids changing the application's global RNG state.
    engine = torch.quasirandom.SobolEngine(start.numel() + 1, scramble=False)
    engine.fast_forward(1 + (attempt - 1) * count)
    unit = engine.draw(count).to(device=start.device, dtype=start.dtype)
    indices = (torch.arange(count, device=start.device) + (attempt - 1) * count) % len(goals)
    selected = goals[indices].clone()
    if intermediate:
        progress = 0.2 + 0.6 * unit[:, :1]
        selected = start + progress * (selected - start)
        span = torch.minimum(
            0.2 * (limits[1] - limits[0]),
            0.5 * (goals[indices] - start).abs() + 0.05 * (limits[1] - limits[0]),
        )
        selected += (2 * unit[:, 1:] - 1) * span
    else:
        selected[:, free] = (
            limits[0, free] + unit[:, 1:][:, free] * (limits[1, free] - limits[0, free])
        )
    selected = torch.maximum(torch.minimum(selected, limits[1]), limits[0])
    selected[:, held] = start[held]
    return selected


def waypoint_trajectories(start, middle, ends, horizon, held):
    """Smooth two-leg initial guesses; every coordinate stays inside its convex hull."""
    if horizon < 3 or middle.shape != ends.shape:
        raise ValueError("posture paths require >=3 knots and matched waypoint batches")
    t = torch.linspace(0, 1, horizon, device=start.device, dtype=start.dtype)[None, :, None]
    u = (2 * t).clamp(0, 1)
    v = (2 * t - 1).clamp(0, 1)
    smooth = lambda x: x.pow(3) * (10 - 15 * x + 6 * x.square())
    paths = (start + smooth(u) * (middle[:, None] - start)
             + smooth(v) * (ends[:, None] - middle[:, None]))
    paths[:, 0] = start
    paths[:, -1] = ends
    paths[:, :, held] = start[held]
    return paths.unsqueeze(0)


@contextmanager
def intermediate_goals(solver, goals):
    """Temporarily retarget IK posture buffers; preserve activation and tolerances."""
    saved = []
    try:
        for rollout in [*solver.core.get_all_rollout_instances(),
                        *solver.core.additional_metrics_rollouts.values()]:
            for cost in rollout.get_cost_component_by_name("posture"):
                if cost is not None and all(cost is not item[0] for item in saved):
                    saved.append((cost, cost.goals.clone(), cost.valid.clone()))
                    cost.goals.zero_()
                    cost.goals[:len(goals)].copy_(goals)
                    cost.valid.zero_()
                    cost.valid[:len(goals)] = True
        yield
    finally:
        for cost, original, valid in saved:
            cost.goals.copy_(original)
            cost.valid.copy_(valid)


class PostureSeeds:
    """A bounded IK-projected waypoint pool owned by one plan_posture call."""

    def __init__(self, planner, current, goals, mask, held):
        """Capture immutable query inputs and a bounded sampling budget."""
        self.planner = planner
        self.current = current
        self.start = current.reorder(planner.joint_names).position[0]
        self.goals = goals
        self.free = ~mask.bool()
        self.held = held.bool()
        self.limits = planner.kinematics.get_joint_limits().position.clone()
        self.count = min(256, planner.ik_solver.config.num_seeds)
        self.records = []
        self.total_time = 0.0
        self.solve_time = 0.0

    def endpoint_seeds(self, attempt):
        """Keep the first solve unchanged, then explore free-joint branches."""
        if attempt == 0:
            return None
        return sample_posture_states(
            self.start, self.goals, self.limits, self.free, self.held,
            self.count, attempt, intermediate=False,
        ).unsqueeze(0)

    def trajectory_seeds(self, goals, endpoints, attempt):
        """Project sampled waypoints with IK and seed one continuous motion."""
        if attempt == 0:
            return None
        candidates = sample_posture_states(
            self.start, endpoints.reshape(-1, self.start.numel()), self.limits,
            self.free, self.held, self.count, attempt, intermediate=True,
        )
        # Native IK retains the captured axis reference, held joint limits,
        # self/world collision checks and bounded start-contact scope.
        with intermediate_goals(self.planner.ik_solver, candidates):
            result = self.planner.ik_solver.solve_pose(
                goals, current_state=self.current, seed_config=candidates.unsqueeze(0),
                return_seeds=self.count,
            )
        self.total_time += result.total_time
        self.solve_time += result.solve_time
        accepted = result.solution[result.success].reshape(-1, self.start.numel())
        self.records.append({"attempt": attempt + 1, "sampled": self.count,
                             "accepted": len(accepted)})
        if not len(accepted):
            # No invalid midpoint is promoted to a path. Endpoint-only seeds
            # may still succeed in the ordinary fully checked trajectory solve.
            return None
        ends = endpoints.reshape(-1, self.start.numel())
        indices = torch.arange(len(ends), device=ends.device) % len(accepted)
        return waypoint_trajectories(
            self.start, accepted[indices], ends,
            self.planner.trajopt_solver.action_horizon, self.held,
        )
