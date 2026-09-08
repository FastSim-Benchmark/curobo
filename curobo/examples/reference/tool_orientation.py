# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Transport a rigidly held object using native cuRobo orientation criteria.

The reference orientation comes from the measured starting joints. This example
uses Franka's rotational joints; it does not simulate a cup or liquid dynamics.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time

import torch

from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import DeviceCfg, GoalToolPose, JointState, ToolPoseCriteria


def orientation_criteria(
    tolerance: float, device_cfg: DeviceCfg = DeviceCfg()
) -> ToolPoseCriteria:
    """Hold the goal orientation throughout transport, preserving the XYZ goal."""
    if not math.isfinite(tolerance) or not 0.0 < tolerance < math.pi:
        raise ValueError("Orientation tolerance must be finite and between 0 and pi radians")
    return ToolPoseCriteria(
        terminal_pose_axes_weight_factor=[1.0] * 6,
        non_terminal_pose_axes_weight_factor=[0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        terminal_pose_convergence_tolerance=[0.0, tolerance],
        non_terminal_pose_convergence_tolerance=[0.0, tolerance],
        device_cfg=device_cfg,
    )


def make_planner(tolerance: float = 0.01) -> MotionPlanner:
    """Create a Franka planner with native optimization and dense pose validation."""
    criteria = orientation_criteria(tolerance)
    config = MotionPlannerCfg.create(
        robot="franka.yml",
        metrics_rollout="metrics_orientation.yml",
        orientation_tolerance=tolerance,
        interpolation_dt=0.01,
        num_ik_seeds=16,
        num_trajopt_seeds=4,
    )
    planner = MotionPlanner(config)
    planner.update_tool_pose_criteria({frame: criteria for frame in planner.tool_frames})
    return planner


def translation_goal(
    planner: MotionPlanner, start: JointState, displacement: list[float]
) -> GoalToolPose:
    """Offset the starting tool position in the robot base frame, keeping its quaternion."""
    tool_pose = planner.compute_kinematics(start).tool_poses
    return GoalToolPose(
        tool_frames=tool_pose.tool_frames,
        position=tool_pose.position.unsqueeze(-2) + planner.device_cfg.to_device(displacement),
        quaternion=tool_pose.quaternion.unsqueeze(-2).clone(),
    )


def validate_orientation(
    planner: MotionPlanner, trajectory: JointState, goal: GoalToolPose
) -> dict:
    """Independently check FK on eight subdivisions of each returned joint segment.

    This checks piecewise linear interpolation of the returned joint samples,
    not a continuous-time certificate for every possible execution interpolator.
    """
    positions = trajectory.reorder(planner.joint_names).position.reshape(-1, planner.action_dim)
    fractions = torch.arange(8, device=positions.device, dtype=positions.dtype) / 8
    dense = (
        positions[:-1, None]
        + fractions[None, :, None] * (positions[1:, None] - positions[:-1, None])
    ).reshape(-1, planner.action_dim)
    dense = torch.cat((dense, positions[-1:]))
    state = planner.compute_kinematics(
        JointState.from_position(dense, joint_names=planner.joint_names)
    )
    quaternion = state.tool_poses.quaternion.reshape(len(dense), -1, 4)
    quaternion = torch.nn.functional.normalize(quaternion, dim=-1)
    reference = torch.nn.functional.normalize(goal.quaternion.reshape(1, -1, 4), dim=-1)
    # The chord formula remains accurate near zero and treats q and -q equally.
    chord = torch.minimum(
        (quaternion - reference).norm(dim=-1), (quaternion + reference).norm(dim=-1)
    )
    angle = 4 * torch.asin((chord / 2).clamp(0, 1))
    endpoint = state.tool_poses.position.reshape(len(dense), -1, 3)[-1]
    return {
        "dense_samples": len(dense),
        "max_orientation_error_rad": float(angle.max()),
        "goal_position_error_m": float(
            (endpoint - goal.position.reshape(-1, 3)).norm(dim=-1).max()
        ),
    }


def main() -> None:
    """Run repeated orientation-constrained transport with CUDA graphs enabled."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orientation-tolerance", type=float, default=0.01, help="Radians")
    parser.add_argument("--displacement", type=float, nargs=3, default=[0.0, 0.12, 0.06])
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if not all(math.isfinite(value) for value in args.displacement):
        parser.error("--displacement must contain finite values")
    planner = make_planner(args.orientation_tolerance)
    start = JointState.from_position(
        planner.default_joint_state.position.unsqueeze(0), joint_names=planner.joint_names
    )
    goal = translation_goal(planner, start, args.displacement)
    rows = []
    for repeat in range(args.repeats + 1):
        torch.cuda.synchronize()
        began = time.perf_counter()
        result = planner.plan_pose(goal, start)
        torch.cuda.synchronize()
        row = {
            "warmup": repeat == 0,
            "wall_ms": 1000 * (time.perf_counter() - began),
            "success": result is not None and bool(result.success.all()),
        }
        if row["success"]:
            row.update(validate_orientation(planner, result.get_interpolated_plan(), goal))
            if row["max_orientation_error_rad"] >= args.orientation_tolerance:
                raise RuntimeError(f"Independent orientation check failed: {row}")
            position_tolerance = planner.config.trajopt_solver_config.position_tolerance
            if row["goal_position_error_m"] >= position_tolerance:
                raise RuntimeError(f"Independent goal position check failed: {row}")
        rows.append(row)
    print(
        json.dumps(
            {
                "orientation_tolerance_rad": args.orientation_tolerance,
                "warm_successes": sum(row["success"] for row in rows[1:]),
                "warm_repetitions": args.repeats,
                "warm_median_ms": statistics.median(row["wall_ms"] for row in rows[1:]),
                "runs": rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
