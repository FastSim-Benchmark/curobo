# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the native orientation-constrained transport recipe."""

from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
import yaml

from curobo._src.cost.cost_tool_pose import ToolPoseCost
from curobo._src.cost.cost_tool_pose_cfg import ToolPoseCostCfg
from curobo._src.rollout.metrics import CostCollection, CostsAndConstraints, RolloutMetrics
from curobo._src.solver.solver_trajopt_result import TrajOptSolverResult
from curobo._src.util.warp import init_warp
from curobo.content import get_task_configs_path
from curobo.examples.reference.tool_orientation import (
    make_planner,
    orientation_criteria,
    translation_goal,
    validate_orientation,
)
from curobo.motion_planner import MotionPlanner
from curobo.types import DeviceCfg, GoalToolPose, JointState, ToolPose


@pytest.fixture(scope="module")
def device_cfg() -> DeviceCfg:
    """Initialize the CUDA runtime used by the native pose kernel."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    init_warp(quiet=True)
    return DeviceCfg()


def pose_samples(
    device_cfg: DeviceCfg, angle: float = 0.0, step: int = 1, axis: int = 0
) -> tuple[ToolPose, GoalToolPose]:
    """Create three samples, optionally tilted at one step, with a distant XYZ goal."""
    position = device_cfg.to_device([[[[0.0, 0.0, 0.0]]] * 3])
    quaternion = device_cfg.to_device([[[[1.0, 0.0, 0.0, 0.0]]] * 3])
    quaternion[0, step, 0, 0] = math.cos(angle / 2)
    quaternion[0, step, 0, axis + 1] = math.sin(angle / 2)
    current = ToolPose(tool_frames=["tool"], position=position, quaternion=quaternion)
    goal = GoalToolPose(
        tool_frames=["tool"],
        position=device_cfg.to_device([[[[[0.5, 0.0, 0.0]]]]]),
        quaternion=device_cfg.to_device([[[[[1.0, 0.0, 0.0, 0.0]]]]]),
    )
    return current, goal


def evaluate_pose(cost: ToolPoseCost, current: ToolPose, goal: GoalToolPose) -> torch.Tensor:
    """Evaluate the native cost with an explicit batch-to-goal index."""
    indices = torch.zeros((1, 1), dtype=torch.int32, device=current.position.device)
    return cost.forward(current, goal, idxs_goal=indices)[0]


@pytest.fixture
def constraint(device_cfg: DeviceCfg) -> ToolPoseCost:
    """Build the orientation constraint from the shipped metrics preset."""
    path = Path(get_task_configs_path()) / "metrics_orientation.yml"
    data = yaml.safe_load(path.read_text())["rollout"]["constraint_cfg"]["tool_pose_cfg"]
    config = ToolPoseCostCfg(**data, tool_frames=["tool"], device_cfg=device_cfg)
    cost = ToolPoseCost(config)
    cost.setup_batch_tensors(batch_size=1, horizon=3)
    return cost


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("step", [0, 1, 2])
@pytest.mark.parametrize("angle,valid", [(0.0, True), (0.0099, True), (0.0101, False)])
def test_orientation_threshold_at_every_step(
    constraint: ToolPoseCost,
    device_cfg: DeviceCfg,
    axis: int,
    step: int,
    angle: float,
    valid: bool,
) -> None:
    """Reject tilt just beyond tolerance, including the start and intermediate steps."""
    current, goal = pose_samples(device_cfg, angle, step, axis)
    value = evaluate_pose(constraint, current, goal)
    collection = CostsAndConstraints()
    collection.constraints.add(value, "tool_pose")
    assert bool(collection.get_feasible(sum_horizon=True)) == valid
    # The distant XYZ goal must not constrain intermediate transport positions.
    assert bool((value[..., 0] == 0).all())


def test_quaternion_sign_and_runtime_tolerance(
    constraint: ToolPoseCost,
    device_cfg: DeviceCfg,
) -> None:
    """Treat q and -q identically and apply updated angular tolerances in place."""
    current, goal = pose_samples(device_cfg, angle=0.02)
    current.quaternion.neg_()
    assert float(evaluate_pose(constraint, current, goal).sum()) > 0
    constraint.update_tool_pose_criteria({"tool": orientation_criteria(0.03, device_cfg)})
    assert float(evaluate_pose(constraint, current, goal).sum()) == 0


def test_native_objective_keeps_terminal_position(
    device_cfg: DeviceCfg,
) -> None:
    """Optimize intermediate rotation and final translation without fixing the XYZ path."""
    objective = ToolPoseCost(
        ToolPoseCostCfg(
            weight=[1.0, 1.0],
            tool_frames=["tool"],
            device_cfg=device_cfg,
            use_grad_input=True,
        )
    )
    objective.setup_batch_tensors(batch_size=1, horizon=3)
    objective.update_tool_pose_criteria({"tool": orientation_criteria(0.01, device_cfg)})
    current, goal = pose_samples(device_cfg, angle=0.05)
    current.position.requires_grad_()
    current.quaternion.requires_grad_()
    value = evaluate_pose(objective, current, goal)
    value.sum().backward()
    assert bool((current.position.grad[:, :-1] == 0).all())
    assert float(current.position.grad[:, -1].abs().sum()) > 0
    assert float(current.quaternion.grad[:, 1].abs().sum()) > 0


@pytest.mark.parametrize("interpolated", [False, True])
def test_intermediate_violation_rejects_success(
    constraint: ToolPoseCost,
    device_cfg: DeviceCfg,
    interpolated: bool,
) -> None:
    """A converged endpoint cannot mask tilt in either metrics trajectory."""
    current, goal = pose_samples(device_cfg, angle=0.02)
    violation = evaluate_pose(constraint, current, goal).clone()
    convergence = CostCollection()
    for name in ("tool_pose_position_tolerance", "tool_pose_orientation_tolerance"):
        convergence.add(torch.zeros((1, 3, 1), **device_cfg.as_torch_dict()), name)

    def metrics(value: torch.Tensor) -> RolloutMetrics:
        constraints = CostsAndConstraints()
        constraints.constraints.add(value, "tool_pose")
        return RolloutMetrics(costs_and_constraints=constraints, convergence=convergence)

    result = TrajOptSolverResult(
        success=torch.ones((1, 1), dtype=torch.bool, device=device_cfg.device),
        batch_size=1,
        num_seeds=1,
        position_tolerance=0.005,
        orientation_tolerance=0.01,
        metrics=metrics(torch.zeros_like(violation) if interpolated else violation),
        interpolated_metrics=metrics(violation) if interpolated else None,
    )
    result._process_metrics()
    assert not bool(result.success.any())


@pytest.mark.parametrize("tolerance", [0.0, -0.01, math.inf, math.nan, math.pi])
def test_invalid_tolerance(tolerance: float) -> None:
    """Reject invalid user tolerances before constructing a planner."""
    with pytest.raises(ValueError, match="tolerance"):
        orientation_criteria(tolerance)


@pytest.fixture(scope="module")
def planner(device_cfg: DeviceCfg) -> Iterator[MotionPlanner]:
    """Use the reference Franka planner with CUDA graph replay enabled."""
    with make_planner() as instance:
        yield instance


def test_franka_transport_and_graph_replay(planner: MotionPlanner) -> None:
    """Reach a new XYZ goal while holding rotation, including after a criteria update."""
    start = JointState.from_position(
        planner.default_joint_state.position.unsqueeze(0), joint_names=planner.joint_names
    )
    goal = translation_goal(planner, start, [0.0, 0.12, 0.06])
    for tolerance in (0.01, 0.005):
        planner.update_tool_pose_criteria(
            {
                frame: orientation_criteria(tolerance, planner.device_cfg)
                for frame in planner.tool_frames
            }
        )
        result = planner.plan_pose(goal, start, max_attempts=2, enable_graph_attempt=2)
        assert result is not None and bool(result.success.all())
        report = validate_orientation(planner, result.get_interpolated_plan(), goal)
        assert report["max_orientation_error_rad"] < tolerance
        assert report["goal_position_error_m"] < 0.005
        for rollout in (
            planner.trajopt_solver.metrics_rollout,
            planner.trajopt_solver.additional_metrics_rollouts["interpolated_rollout"],
        ):
            assert rollout.metrics_constraint_manager.has_cost("tool_pose")


def test_incompatible_start_is_rejected(planner: MotionPlanner) -> None:
    """Refuse a fixed-orientation path when the initial orientation already disagrees."""
    start = JointState.from_position(
        planner.default_joint_state.position.unsqueeze(0).clone(), joint_names=planner.joint_names
    )
    goal = translation_goal(planner, start, [0.0, 0.12, 0.06])
    start.position[..., 0] += 0.1
    result = planner.plan_pose(goal, start, max_attempts=1)
    assert result is None or not bool(result.success.any())


def test_preset_preserves_other_constraints() -> None:
    """Keep the optional preset consistent with the ordinary collision and limit checks."""
    directory = Path(get_task_configs_path())
    base = yaml.safe_load((directory / "metrics_base.yml").read_text())
    orientation = yaml.safe_load((directory / "metrics_orientation.yml").read_text())
    del orientation["rollout"]["constraint_cfg"]["tool_pose_cfg"]
    assert orientation == base
