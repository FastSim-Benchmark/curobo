# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Keep a synthetic cup upright while departing from contact with a support.

This internal reference combines the experimental StartContact configuration
with public tool-axis criteria on Franka. The authored cup starts upright; its
up direction is transformed into the gripper frame using the grasp transform.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from curobo._src.collision.contact_separation import StartContact
from curobo.examples.reference.contact_separation import box_clearance
from curobo.examples.reference.tool_orientation import translation_goal
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.scene import Cuboid, Scene
from curobo.types import GoalToolPose, JointState, Pose, ToolPoseCriteria


def make_cup_planner(
    contact_enabled: bool = True,
    tilt_tolerance: float = 0.01,
) -> tuple[MotionPlanner, JointState, GoalToolPose, torch.Tensor, Cuboid, StartContact]:
    """Attach an upright cup with 1 mm initial support overlap and configure both constraints."""
    cfg = MotionPlannerCfg.create(
        robot="franka.yml",
        metrics_rollout="metrics_orientation.yml",
        collision_cache={"cuboid": 1},
        num_ik_seeds=16,
        num_trajopt_seeds=4,
        orientation_tolerance=tilt_tolerance,
        interpolation_dt=0.01,
    )
    # Determine the grasp geometry before constructing/capturing the configured solver.
    with MotionPlanner(cfg) as initial:
        device = initial.device_cfg
        start = JointState.from_position(
            initial.default_joint_state.position.unsqueeze(0),
            joint_names=initial.joint_names,
        )
        tool = initial.compute_kinematics(start).tool_poses.get_link_pose(initial.tool_frames[0])
        cup_pose = Pose(
            position=tool.position + device.to_device([0, 0, -0.10]),
            quaternion=device.to_device([[1, 0, 0, 0]]),
        )
        cup_in_tool = tool.inverse().multiply(cup_pose)
        axis = cup_in_tool.get_rotation_matrix()[0, :, 2].clone()
        payload = device.to_device(
            [
                [0, 0, -0.04, 0.025],
                [0, 0, 0, 0.025],
                [0, 0, 0.04, 0.025],
                [0.04, 0, 0, 0.015],
            ]
        )
        initial.attachment_manager.update(payload, start, world_objects_pose_offset=cup_pose)
        spheres = initial.compute_kinematics(start).robot_spheres.reshape(-1, 4)
        sphere_id = int(
            initial.attachment_manager.kinematics_params.get_sphere_index_from_link_name(
                "attached_object"
            )[0]
        )
        bottom = spheres[sphere_id].tolist()
        contact = StartContact("support", (sphere_id,), (tuple(bottom),))
        support = Cuboid(
            name="support",
            dims=[0.10, 0.10, 0.02],
            pose=[bottom[0], bottom[1], bottom[2] - bottom[3] + 0.001 - 0.01, 1, 0, 0, 0],
        )
    cfg.scene_collision_cfg.scene_model = Scene(cuboid=[support])
    if contact_enabled:
        core = cfg.trajopt_solver_config.core_cfg
        for rollout in [core.metrics_rollout_config, *core.optimizer_rollout_configs]:
            for manager in rollout.get_cost_manager_configs():
                if manager.scene_collision_cfg is not None:
                    manager.scene_collision_cfg.start_contact = contact
    planner = MotionPlanner(cfg)
    planner.attachment_manager.update(payload, start, world_objects_pose_offset=cup_pose)
    planner.update_tool_pose_criteria(
        {
            planner.tool_frames[0]: ToolPoseCriteria.hold_axis(axis, tilt_tolerance, device),
        }
    )
    goal = translation_goal(planner, start, [0.0, 0.12, 0.06])
    # A 180-degree twist of the reference must not demand a 180-degree cup rotation.
    origin = device.to_device([[0, 0, 0]])
    yaw = Pose(position=origin, quaternion=device.to_device([[0, 0, 0, 1]]))
    reference = yaw.multiply(Pose(position=origin, quaternion=goal.quaternion.reshape(1, 4)))
    goal.quaternion.copy_(reference.quaternion.reshape_as(goal.quaternion))
    return planner, start, goal, axis, support, contact


def validate_cup(
    planner: MotionPlanner,
    trajectory: JointState,
    axis: torch.Tensor,
    support: Cuboid,
    contact: StartContact,
) -> dict:
    """Check cup tilt and all support pairs on eight subdivisions per returned segment."""
    positions = trajectory.reorder(planner.joint_names).position.reshape(-1, planner.action_dim)
    fractions = torch.arange(8, device=positions.device, dtype=positions.dtype) / 8
    dense = (
        positions[:-1, None]
        + fractions[None, :, None] * (positions[1:, None] - positions[:-1, None])
    ).reshape(-1, planner.action_dim)
    dense = torch.cat((dense, positions[-1:]))
    state = planner.compute_kinematics(
        JointState.from_position(
            dense,
            joint_names=planner.joint_names,
        )
    )
    tool = state.tool_poses.get_link_pose(planner.tool_frames[0])
    rotation = tool.get_rotation_matrix()
    direction = rotation @ axis
    tilt = torch.atan2(direction[:, :2].norm(dim=-1), direction[:, 2])
    gaps = box_clearance(state.robot_spheres.reshape(len(dense), -1, 4).cpu(), support)
    contact_gap = gaps[:, contact.sphere_indices[0]].clone()
    gaps[:, contact.sphere_indices[0]] = torch.inf
    separated = contact_gap.clamp_max(contact.release_clearance)
    regression = float((torch.cummax(separated, dim=0).values - separated).max())
    # Track twist independently from tilt using a cup-local horizontal direction.
    local_right = torch.linalg.cross(axis, axis.new_tensor([0.0, 1.0, 0.0]))
    right = rotation @ torch.nn.functional.normalize(local_right, dim=0)
    yaw = torch.atan2(right[:, 1], right[:, 0])
    yaw_change = torch.atan2(torch.sin(yaw - yaw[0]), torch.cos(yaw - yaw[0]))
    return {
        "samples": len(dense),
        "max_tilt_rad": float(tilt.max()),
        "max_yaw_change_rad": float(yaw_change.abs().max()),
        "initial_support_gap_m": float(contact_gap[0]),
        "final_support_gap_m": float(contact_gap[-1]),
        "minimum_support_gap_m": float(contact_gap.min()),
        "separation_regression_m": regression,
        "minimum_other_pair_gap_m": float(gaps.min()),
        "contact_valid": float(gaps.min()) >= -contact.numerical_tolerance
        and float(contact_gap.min()) >= float(contact_gap[0]) - contact.numerical_tolerance
        and float(contact_gap[-1]) >= contact.release_clearance
        and regression <= contact.numerical_tolerance,
    }


def main() -> None:
    """Run repeated native contact departure while allowing free cup yaw."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--tilt-tolerance", type=float, default=0.01)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    planner, start, goal, axis, support, contact = make_cup_planner(
        tilt_tolerance=args.tilt_tolerance,
    )
    rows = []
    with planner:
        for repeat in range(args.repeats + 1):
            torch.cuda.synchronize()
            began = time.perf_counter()
            result = planner.plan_pose(goal, start, max_attempts=2, enable_graph_attempt=2)
            torch.cuda.synchronize()
            row = {
                "wall_ms": 1000 * (time.perf_counter() - began),
                "warmup": repeat == 0,
                "success": result is not None and bool(result.success.all()),
            }
            if row["success"]:
                row.update(
                    validate_cup(planner, result.get_interpolated_plan(), axis, support, contact)
                )
                if row["max_tilt_rad"] >= args.tilt_tolerance or not row["contact_valid"]:
                    raise RuntimeError(f"Independent cup validation failed: {row}")
            rows.append(row)
    print(
        json.dumps(
            {
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
