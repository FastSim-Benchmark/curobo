# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Reproduce contact departure with synthetic sphere geometry and native cuRobo.

This developer benchmark intentionally uses the experimental internal contact
configuration. It exercises a three-axis Cartesian robot, not a physical arm or
simulator. CUDA graphs stay enabled; no obstacles or robot links are disabled.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from curobo._src.collision.contact_separation import StartContact
from curobo._src.geom.collision.buffer_collision import CollisionBuffer
from curobo._src.geom.collision.collision_scene import SceneCollision, SceneCollisionCfg
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.scene import Cuboid, Scene
from curobo.types import DeviceCfg, JointState


def robot_config(directory: Path) -> dict:
    """Create an authored Cartesian gripper with four reserved payload spheres."""
    inertial = (
        '<inertial><mass value="0.1"/><inertia ixx="0.001" ixy="0" ixz="0" '
        'iyy="0.001" iyz="0" izz="0.001"/></inertial>'
    )
    links = ["base", "x_carriage", "y_carriage", "gripper"]
    xml = '<robot name="contact_test_gantry">'
    xml += "".join(f'<link name="{name}">{inertial}</link>' for name in links)
    for i, (name, axis, lower, upper) in enumerate(
        [
            ("x", "1 0 0", -0.9, 0.5),
            ("y", "0 1 0", -0.3, 0.3),
            ("z", "0 0 1", 0.05, 0.6),
        ]
    ):
        xml += (
            f'<joint name="{name}" type="prismatic"><parent link="{links[i]}"/>'
            f'<child link="{links[i + 1]}"/><axis xyz="{axis}"/>'
            f'<limit lower="{lower}" upper="{upper}" velocity="0.3" effort="100"/></joint>'
        )
    xml += "</robot>"
    path = directory / "gantry.urdf"
    path.write_text(xml)
    return {
        "robot_cfg": {
            "kinematics": {
                "format_version": 2.0,
                "urdf_path": str(path),
                "asset_root_path": str(directory),
                "base_link": "base",
                "tool_frames": ["gripper"],
                "collision_link_names": ["base", "gripper", "attached_object"],
                "collision_spheres": {
                    "base": [{"center": [-0.85, 0.0, 0.5], "radius": 0.025}],
                    "gripper": [
                        {"center": [0.0, -0.064, -0.025], "radius": 0.019},
                        {"center": [0.0, 0.064, -0.025], "radius": 0.019},
                        {"center": [0.0, 0.0, 0.06], "radius": 0.02},
                    ],
                },
                "collision_sphere_buffer": 0.0,
                "extra_collision_spheres": {"attached_object": 4},
                "extra_links": {
                    "attached_object": {
                        "link_name": "attached_object",
                        "parent_link_name": "gripper",
                        "joint_name": "payload_fixed",
                        "joint_type": "FIXED",
                        "fixed_transform": [0, 0, 0, 1, 0, 0, 0],
                    }
                },
                "self_collision_ignore": {"gripper": ["attached_object"]},
                "self_collision_buffer": {"gripper": 0.0, "attached_object": 0.0},
                "mesh_link_names": [],
                "use_global_cumul": True,
                "cspace": {
                    "joint_names": ["x", "y", "z"],
                    "default_joint_position": [0.0, 0.0, 0.12],
                    "null_space_weight": [1, 1, 1],
                    "cspace_distance_weight": [1, 1, 1],
                    "max_acceleration": 2.0,
                    "max_jerk": 20.0,
                },
            }
        }
    }


def scene_config(cabinet: bool) -> Scene:
    """Build a support and, optionally, a cabinet with 30 mm initial headroom."""
    boxes = [Cuboid(name="support", pose=[0.0, 0, -0.025, 1, 0, 0, 0], dims=[0.8, 0.5, 0.05])]
    if cabinet:
        boxes.extend(
            [
                Cuboid(name="ceiling", pose=[0, 0, 0.255, 1, 0, 0, 0], dims=[0.8, 0.5, 0.05]),
                Cuboid(name="back", pose=[0.425, 0, 0.115, 1, 0, 0, 0], dims=[0.05, 0.5, 0.23]),
                Cuboid(name="left", pose=[0, 0.265, 0.115, 1, 0, 0, 0], dims=[0.8, 0.03, 0.23]),
                Cuboid(name="right", pose=[0, -0.265, 0.115, 1, 0, 0, 0], dims=[0.8, 0.03, 0.23]),
            ]
        )
    return Scene(cuboid=boxes)


def build_scene(name: str) -> Scene:
    """Create the requested scene without external assets."""
    scene = scene_config(name in ("cabinet", "sealed_cabinet"))
    if name == "tray":
        scene.cuboid.extend(
            [
                Cuboid(
                    name="tray_front", dims=[0.025, 0.5, 0.02], pose=[-0.4125, 0, 0.01, 1, 0, 0, 0]
                ),
                Cuboid(
                    name="tray_back", dims=[0.025, 0.5, 0.02], pose=[0.4125, 0, 0.01, 1, 0, 0, 0]
                ),
                Cuboid(
                    name="tray_left", dims=[0.8, 0.025, 0.02], pose=[0, 0.2625, 0.01, 1, 0, 0, 0]
                ),
                Cuboid(
                    name="tray_right", dims=[0.8, 0.025, 0.02], pose=[0, -0.2625, 0.01, 1, 0, 0, 0]
                ),
            ]
        )
    if name == "sealed_cabinet":
        scene.cuboid.append(
            Cuboid(
                name="closed_front",
                dims=[0.05, 0.5, 0.23],
                pose=[-0.425, 0, 0.115, 1, 0, 0, 0],
            )
        )
    if name not in ("table", "tray", "cabinet", "sealed_cabinet"):
        raise ValueError(f"Unknown scene: {name}")
    return scene


def make_planner(
    directory: Path,
    name: str,
    contact_enabled: bool,
    initial_lift: float = 0.0,
    random_seed: int = 123,
) -> tuple[MotionPlanner, JointState, JointState, Scene]:
    """Bind only trajectory rollouts to the captured payload/support contact."""
    directory.mkdir(parents=True, exist_ok=True)
    scene = build_scene(name)
    device = DeviceCfg()
    cfg = MotionPlannerCfg.create(
        robot_config(directory),
        scene_model=scene,
        device_cfg=device,
        num_ik_seeds=8,
        num_trajopt_seeds=4,
        use_cuda_graph=True,
        optimizer_collision_activation_distance=0.005,
        interpolation_dt=0.01,
        interpolation_buffer_size=1200,
        random_seed=random_seed,
    )
    planner = MotionPlanner(cfg)
    start = JointState.from_position(
        device.to_device([[0, 0, 0.12 + initial_lift]]),
        joint_names=planner.joint_names,
    )
    goal = JointState.from_position(
        device.to_device([[-0.6, 0, 0.20 if name == "tray" else 0.17]]),
        joint_names=planner.joint_names,
    )
    payload = device.to_device(
        [
            [0, 0, -0.08, 0.041],
            [0, 0, -0.04, 0.041],
            [0, 0, 0, 0.041],
            [0.05, 0, -0.04, 0.016],
        ]
    )
    planner.attachment_manager.update(payload, start)
    if contact_enabled:
        spheres = planner.compute_kinematics(start).robot_spheres.reshape(-1, 4)
        ids = planner.attachment_manager.kinematics_params.get_sphere_index_from_link_name(
            "attached_object"
        ).tolist()
        # This fixture's bottom sphere is the only initially contacting sphere.
        index = ids[0]
        contact = StartContact("support", (index,), (tuple(spheres[index].tolist()),))
        core = cfg.trajopt_solver_config.core_cfg
        for rollout in [core.metrics_rollout_config, *core.optimizer_rollout_configs]:
            for field_name in ("cost_cfg", "constraint_cfg", "hybrid_cost_constraint_cfg"):
                manager = getattr(rollout, field_name)
                if manager is not None and manager.scene_collision_cfg is not None:
                    manager.scene_collision_cfg.start_contact = contact
        # Construct the configured solver before any CUDA graph is captured.
        # IK and PRM retain the original collision contract.
        del planner
        planner = MotionPlanner(cfg)
        planner.attachment_manager.update(payload, start)
    return planner, start, goal, scene


def box_clearance(spheres: torch.Tensor, box: Cuboid) -> torch.Tensor:
    """Independently evaluate these axis-aligned benchmark boxes on CPU."""
    offset = (spheres[..., :3] - torch.tensor(box.pose[:3])).abs() - torch.tensor(box.dims) / 2
    return (
        offset.clamp_min(0).square().sum(-1).sqrt()
        + offset.amax(-1).clamp_max(0)
        - spheres[..., 3]
    )


def validate_trajectory(
    planner: MotionPlanner,
    positions: torch.Tensor,
    scene: Scene,
    contact_enabled: bool,
) -> dict:
    """Check all sphere/obstacle pairs independently on 8x denser samples.

    The analytic verifier is cross-checked against unmodified cuRobo collision
    queries with one obstacle at a time. No contact replacement is used there.
    These are sampled proxy-geometry checks, not a physical execution guarantee.
    """
    positions = positions.reshape(-1, 3)
    fractions = torch.arange(8, device=positions.device, dtype=torch.float32) / 8
    dense = (
        positions[:-1, None]
        + fractions[None, :, None] * (positions[1:, None] - positions[:-1, None])
    ).reshape(-1, 3)
    dense = torch.cat((dense, positions[-1:]))
    state = JointState.from_position(dense, joint_names=planner.joint_names)
    spheres_gpu = planner.compute_kinematics(state).robot_spheres.reshape(len(dense), -1, 4)
    spheres = spheres_gpu.cpu()
    gaps = torch.stack([box_clearance(spheres, box) for box in scene.cuboid], dim=-1)
    device = DeviceCfg()
    native_costs = []
    for box in scene.cuboid:
        checker = SceneCollision.from_config(
            SceneCollisionCfg(
                scene_model=Scene(cuboid=[box]),
                device_cfg=device,
            )
        )
        query = spheres_gpu[None].contiguous()
        buffer = CollisionBuffer.from_shape(query.shape, device)
        cost = checker.get_sphere_distance_raw(
            query,
            buffer,
            device.to_device([1.0]),
            device.to_device([0.0]),
        )
        native_costs.append(cost[0].cpu())
    native = torch.stack(native_costs, dim=-1)
    native_error = float((native - (-gaps).clamp_min(0)).abs().max())
    forbidden = gaps.clone()
    native_forbidden = native.clone()
    contact_gap = gaps[:, 4, 0]
    if contact_enabled:
        forbidden[:, 4, 0] = torch.inf
        native_forbidden[:, 4, 0] = 0
    prefix = torch.cummax(contact_gap.clamp_max(0.005), dim=0).values
    regression = float((prefix - contact_gap).clamp_min(0).max())
    minimum_forbidden = float(forbidden.min())
    valid = minimum_forbidden >= -1e-5 and float(native_forbidden.max()) <= 1e-5
    if contact_enabled:
        valid = valid and (
            float(contact_gap.min()) >= float(contact_gap[0]) - 1e-5
            and float(contact_gap[-1]) >= 0.005 - 1e-5
            and regression <= 1e-5
        )
    valid = valid and native_error <= 2e-6
    lifted = positions[0:1].clone()
    lifted[0, 2] += 0.1
    lifted_spheres = (
        planner.compute_kinematics(
            JointState.from_position(lifted, joint_names=planner.joint_names)
        )
        .robot_spheres.reshape(-1, 4)
        .cpu()
    )
    lifted_gap = torch.stack([box_clearance(lifted_spheres, box) for box in scene.cuboid])
    result = {
        "valid": bool(valid),
        "samples": len(dense),
        "initial_contact_mm": 1000 * float(contact_gap[0]),
        "minimum_contact_mm": 1000 * float(contact_gap.min()),
        "terminal_support_clearance_mm": 1000 * float(contact_gap[-1]),
        "largest_separation_regression_mm": 1000 * regression,
        "minimum_forbidden_clearance_mm": 1000 * minimum_forbidden,
        "forbidden_collision_count": int((forbidden < -1e-5).sum()),
        "native_max_forbidden_penetration_mm": 1000 * float(native_forbidden.max()),
        "analytic_native_max_error_mm": 1000 * native_error,
        "ten_cm_lift_min_clearance_mm": 1000 * float(lifted_gap.min()),
    }
    if any(box.name == "ceiling" for box in scene.cuboid):
        ceiling_index = [box.name for box in scene.cuboid].index("ceiling")
        result["initial_headroom_mm"] = 1000 * float(gaps[0, :, ceiling_index].min())
        inside = dense[:, 0] >= -0.4
        result["maximum_lift_inside_cabinet_mm"] = 1000 * float(
            (dense[inside, 2] - positions[0, 2]).max()
        )
    return result


def run_case(
    directory: Path,
    name: str,
    contact_enabled: bool,
    repeats: int,
    initial_lift: float = 0.0,
) -> tuple[dict, torch.Tensor]:
    """Record cold setup separately and measure repeated CUDA-graph planning."""
    planner, start, goal, scene = make_planner(directory, name, contact_enabled, initial_lift)
    rows = []
    positions = None
    for repeat in range(repeats + 1):
        torch.cuda.synchronize()
        began = time.perf_counter()
        # Compare TrajOpt on identical settings. PRM cannot connect a colliding
        # start; graph-assisted departure remains outside this experiment.
        result = planner.plan_cspace(goal, start, max_attempts=2, enable_graph_attempt=2)
        torch.cuda.synchronize()
        elapsed = 1000 * (time.perf_counter() - began)
        success = result is not None and bool(result.success.all())
        if result is not None:
            positions = result.get_interpolated_plan().position.reshape(-1, 3)
        rows.append({"success": success, "wall_ms": elapsed})
    if positions is None:
        raise RuntimeError("cuRobo produced no diagnostic trajectory")
    validation = validate_trajectory(planner, positions, scene, contact_enabled)
    warm = rows[1:]
    record = {
        "scene": name,
        "contact_enabled": contact_enabled,
        "initial_lift_mm": initial_lift * 1000,
        "warm_repetitions": repeats,
        "warm_successes": sum(row["success"] for row in warm),
        "cold_ms": rows[0]["wall_ms"],
        "warm_median_ms": statistics.median(row["wall_ms"] for row in warm),
        "warm_p95_ms": float(np.percentile([row["wall_ms"] for row in warm], 95)),
        "runs": rows,
        "validation": validation,
        "scene_boxes": [
            {"name": box.name, "pose": box.pose, "dims": box.dims} for box in scene.cuboid
        ],
    }
    return record, positions.detach().cpu()


def plot_cases(records: list[dict], paths: dict[str, torch.Tensor], directory: Path) -> None:
    """Render side views of the verified trajectories for the two required cases."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for axis, name in zip(axes, ("table", "cabinet")):
        record = next(r for r in records if r["scene"] == name and r["contact_enabled"])
        positions = paths[f"{name}_contact"]
        for box in record["scene_boxes"]:
            if box["name"] in ("left", "right"):
                continue
            x, _, z = box["pose"][:3]
            dx, _, dz = box["dims"]
            axis.add_patch(Rectangle((x - dx / 2, z - dz / 2), dx, dz, color="#bbc5d3", alpha=0.9))
        axis.plot(
            positions[:, 0], positions[:, 2], color="#137b6e", lw=2, label="Planned gripper path"
        )
        for index, opacity in [(0, 0.7), (len(positions) // 2, 0.2), (-1, 0.7)]:
            x, _, z = positions[index].tolist()
            for dz, radius in [(-0.08, 0.041), (-0.04, 0.041), (0, 0.041), (0.06, 0.02)]:
                axis.add_patch(Circle((x, z + dz), radius, color="#248b9a", alpha=opacity))
        if name == "cabinet":
            axis.plot(
                [0, 0], [0.12, 0.22], "--", color="#bd3344", lw=2, label="Infeasible 100 mm lift"
            )
            axis.text(0.02, 0.31, "30 mm initial headroom", color="#963644", fontsize=10)
        axis.set_title("Table departure" if name == "table" else "Low cabinet extraction")
        axis.set_xlabel("World X (m)")
        axis.set_ylabel("World Z (m)")
        axis.set_aspect("equal")
        axis.set_xlim(-0.8, 0.5)
        axis.set_ylim(-0.07, 0.4)
        axis.grid(alpha=0.15)
        axis.legend(loc="upper left", fontsize=8)
    figure.savefig(directory / "contact_departure.png", dpi=180)
    plt.close(figure)


def main() -> None:
    """Run the controlled baseline, contact, and rejection experiments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records, paths = [], {}
    cases = [
        ("table", False, 0.0),
        ("table", True, 0.0),
        ("tray", False, 0.0),
        ("tray", True, 0.0),
        ("cabinet", False, 0.0),
        ("cabinet", True, 0.0),
        ("cabinet", False, 0.011),
        ("sealed_cabinet", True, 0.0),
    ]
    for name, enabled, lift in cases:
        key = f"{name}_{'contact' if enabled else 'baseline'}"
        if lift:
            key += "_clear_start"
        record, positions = run_case(args.output_dir, name, enabled, args.repeats, lift)
        records.append(record)
        paths[key] = positions
        np.save(args.output_dir / f"{key}.npy", positions.numpy())
        (args.output_dir / f"{key}.json").write_text(json.dumps(record, indent=2))
        print(
            json.dumps(
                {
                    k: record[k]
                    for k in (
                        "scene",
                        "contact_enabled",
                        "initial_lift_mm",
                        "warm_successes",
                        "warm_repetitions",
                        "warm_median_ms",
                    )
                }
            ),
            flush=True,
        )
    report = {
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "cuda_graph": True,
        "scene_type": "synthetic three-axis robot and sphere proxies",
        "world_collision_activation_mm": 5.0,
        "max_initial_overlap_mm": 2.0,
        "contact_numerical_tolerance_mm": 0.01,
        "notes": [
            "Repeated fixed scenes/seeds measure repeatability, not general success probability.",
            "Cold compilation/capture excluded from warm timing.",
            "TrajOpt only: no contact-aware PRM integration.",
            "No simulator, force model, or physical robot execution.",
        ],
        "cases": records,
    }
    (args.output_dir / "results.json").write_text(json.dumps(report, indent=2))
    plot_cases(records, paths, args.output_dir)
    for record in records:
        expected = record["scene"] != "sealed_cabinet" and (
            record["contact_enabled"] or record["initial_lift_mm"] > 0
        )
        if bool(record["warm_successes"]) != expected:
            raise RuntimeError(f"Unexpected planning result: {record['scene']}")
        if expected and (
            record["warm_successes"] != args.repeats or not record["validation"]["valid"]
        ):
            raise RuntimeError(f"Independent validation failed: {record['scene']}")


if __name__ == "__main__":
    main()
