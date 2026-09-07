# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Jointly optimize transport and final normal landing through native cuRobo.

This internal experiment shares the departure fixtures and ordinary collision
verifier. CUDA graphs stay enabled. PRM contact integration is outside its scope.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from curobo._src.collision.contact_approach import GoalContact
from curobo.examples.reference.contact_separation import (
    box_clearance,
    make_planner,
    validate_trajectory,
)
from curobo.motion_planner import MotionPlanner
from curobo.scene import Scene
from curobo.types import GoalToolPose, JointState


def run_case(
    directory: Path,
    name: str,
    contact_enabled: bool,
    repeats: int,
    goal_position: tuple[float, float, float],
    pose_goal: bool = True,
    normal_landing: bool = True,
    support_height: float = 0.0,
    approach_clearance: float = GoalContact.approach_clearance,
) -> tuple[dict, torch.Tensor | None]:
    """Optimize the complete trajectory to the caller's world-frame tool position."""
    planner, start, goal, scene = make_planner(
        directory,
        name,
        contact_enabled,
        placement=True,
        normal_landing=normal_landing,
        placement_goal=goal_position,
        support_height=support_height,
        approach_clearance=approach_clearance,
    )
    tool_pose = planner.compute_kinematics(goal).tool_poses
    goal_pose = GoalToolPose(
        tool_frames=tool_pose.tool_frames,
        position=tool_pose.position.unsqueeze(-2),
        quaternion=tool_pose.quaternion.unsqueeze(-2),
    )
    # These ordinary, unmasked checks establish that the target is colliding.
    target_validation = validate_trajectory(
        planner, goal.position.repeat(2, 1), scene, False, placement=True
    )
    metrics = planner.trajopt_solver.config.metrics_rollout_config
    contact = metrics.constraint_cfg.scene_collision_cfg.goal_contact
    rows = []
    positions = None
    validation = None
    for repeat in range(repeats + 1):
        torch.cuda.synchronize()
        began = time.perf_counter()
        if pose_goal:
            result = planner.plan_pose(goal_pose, start, max_attempts=2, enable_graph_attempt=2)
        else:
            result = planner.plan_cspace(goal, start, max_attempts=2, enable_graph_attempt=2)
        torch.cuda.synchronize()
        elapsed = 1000 * (time.perf_counter() - began)
        success = result is not None and bool(result.success.all())
        rows.append({"success": success, "wall_ms": elapsed})
        if result is not None:
            positions = result.get_interpolated_plan().position.reshape(-1, 3)
            validation = validate_trajectory(
                planner, positions, scene, contact_enabled, True, approach_clearance
            )
            endpoint_error = float((positions[-1] - goal.position[0]).abs().max())
            validation["goal_position_error_mm"] = endpoint_error * 1000
            validation["valid"] = validation["valid"] and endpoint_error <= 1e-5
            if contact_enabled and normal_landing:
                landing = validate_landing(planner, positions, scene, contact)
                validation.update(landing)
                validation["valid"] = validation["valid"] and landing["landing_valid"]
            if success and not validation["valid"]:
                raise RuntimeError(
                    f"Independent placement validation failed: {name}: {validation}"
                )
    warm = rows[1:]
    record = {
        "scene": name,
        "contact_enabled": contact_enabled,
        "normal_landing": normal_landing,
        "joint_trajectory_optimization": True,
        "goal_position": list(goal_position),
        "support_height": support_height,
        "approach_clearance": approach_clearance,
        "planning_api": "plan_pose" if pose_goal else "plan_cspace",
        "warm_repetitions": repeats,
        "warm_successes": sum(row["success"] for row in warm),
        "cold_ms": rows[0]["wall_ms"],
        "warm_median_ms": statistics.median(row["wall_ms"] for row in warm),
        "warm_p95_ms": float(np.percentile([row["wall_ms"] for row in warm], 95)),
        "runs": rows,
        "validation": validation,
        "target_normal_collision": target_validation,
    }
    return record, positions.detach().cpu() if positions is not None else None


def validate_landing(
    planner: MotionPlanner,
    positions: torch.Tensor,
    scene: Scene,
    contact: GoalContact,
) -> dict:
    """Independently verify alignment on 8x denser FK poses in these table fixtures."""
    fractions = torch.arange(8, device=positions.device, dtype=torch.float32) / 8
    dense = (
        positions[:-1, None]
        + fractions[None, :, None] * (positions[1:, None] - positions[:-1, None])
    ).reshape(-1, 3)
    dense = torch.cat((dense, positions[-1:]))
    state = planner.compute_kinematics(
        JointState.from_position(dense, joint_names=planner.joint_names)
    )
    spheres = state.robot_spheres.reshape(len(dense), -1, 4).cpu()
    support = next(box for box in scene.cuboid if box.name == contact.obstacle_name)
    gap = box_clearance(spheres[:, contact.sphere_indices], support).amin(-1)
    landing = contact.landing
    tool = state.tool_poses.get_link_pose(landing.tool_frame)
    xyz = tool.position.reshape(-1, 3).cpu()
    quat = tool.quaternion.reshape(-1, 4).cpu()
    target_xyz = torch.tensor(landing.goal_position, dtype=torch.float32)
    target_quat = torch.tensor(landing.goal_quaternion, dtype=torch.float32)
    normal = torch.tensor(landing.outward_normal, dtype=torch.float32)
    in_band = gap < contact.approach_clearance
    touching = gap <= 0
    displacement = xyz - target_xyz
    tangent = displacement - (displacement * normal).sum(-1, keepdim=True) * normal
    lateral = tangent.norm(dim=-1)
    chord = torch.minimum((quat - target_quat).norm(dim=-1), (quat + target_quat).norm(dim=-1))
    angle = 4 * torch.asin((chord * 0.5).clamp(0, 1))
    max_lateral = float(lateral[in_band].max()) if bool(in_band.any()) else 0.0
    max_angle = float(angle[in_band].max()) if bool(in_band.any()) else 0.0
    early = touching & (
        (lateral > landing.position_tolerance) | (angle > landing.orientation_tolerance)
    )
    # Determine the phase boundary from the solved geometry, never from a fixed waypoint/time.
    indices = torch.nonzero(in_band).flatten()
    landing_start_index = int(indices[0]) if len(indices) else len(gap)
    transport_gap = gap[:landing_start_index]
    transport_minimum = float(transport_gap.min()) if len(transport_gap) else float(gap[0])
    return {
        "landing_valid": max_lateral <= landing.position_tolerance
        and max_angle <= landing.orientation_tolerance
        and transport_minimum >= contact.approach_clearance - contact.numerical_tolerance,
        "landing_start_fraction": landing_start_index / (len(gap) - 1),
        "landing_max_lateral_error_mm": max_lateral * 1000,
        "landing_max_orientation_error_rad": max_angle,
        "contact_before_alignment_count": int(early.sum()),
        "transport_min_support_clearance_mm": transport_minimum * 1000,
        "contact_before_landing_phase_count": int((transport_gap <= 0).sum()),
        "contact_lateral_span_mm": float(
            (tangent[touching].amax(0) - tangent[touching].amin(0)).norm()
        )
        * 1000
        if bool(touching.any())
        else 0.0,
    }


def main() -> None:
    """Run table, tray, low-cabinet, and infeasible sealed-cabinet placement."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--goal-position",
        type=float,
        nargs=3,
        required=True,
        metavar=("X", "Y", "Z"),
        help="Final world-frame gripper position in meters; never adjusted by the planner.",
    )
    parser.add_argument("--support-height", type=float, default=0.0)
    parser.add_argument(
        "--approach-clearance",
        type=float,
        default=GoalContact.approach_clearance,
        help="Support gap below which final alignment is required, in meters.",
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = [
        (name, enabled) for name in ("table", "tray", "cabinet") for enabled in (False, True)
    ] + [("sealed_cabinet", True)]
    records = []
    for name, enabled in cases:
        record, positions = run_case(
            args.output_dir,
            name,
            enabled,
            args.repeats,
            tuple(args.goal_position),
            support_height=args.support_height,
            approach_clearance=args.approach_clearance,
        )
        records.append(record)
        key = f"{name}_{'contact' if enabled else 'baseline'}"
        (args.output_dir / f"{key}.json").write_text(json.dumps(record, indent=2))
        if positions is not None:
            np.save(args.output_dir / f"{key}.npy", positions.numpy())
        print(
            json.dumps(
                {
                    k: record[k]
                    for k in ("scene", "contact_enabled", "warm_successes", "warm_median_ms")
                }
            ),
            flush=True,
        )
        expected = name != "sealed_cabinet" and (
            enabled or record["target_normal_collision"]["valid"]
        )
        if record["warm_successes"] != (args.repeats if expected else 0):
            raise RuntimeError(f"Unexpected placement result: {name}")
    report = {
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "cuda_graph": True,
        "cases": records,
        "notes": [
            "Native goal IK and TrajOpt; no contact-aware PRM.",
            "Each solve optimizes one full trajectory; no preplacement waypoint or concatenation.",
            "Final position is supplied by the caller; contact geometry determines landing onset.",
            "All successful repetitions receive independent dense collision validation.",
            "Fixed synthetic fixtures measure repeatability, not general success probability.",
            "No simulator, force control, grasp release, or physical execution.",
        ],
    }
    (args.output_dir / "results.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
