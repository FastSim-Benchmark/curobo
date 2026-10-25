"""Finite empty-world cuRobo diagnostics; records actual plans, never simulated success."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(run: Path, robot_root: Path, source: Path) -> None:
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    from yourdfpy import URDF
    from robot_urdf import needed_joints

    if run.exists():
        raise FileExistsError(f"Use a new run: {run}")
    (run / "inputs").mkdir(parents=True)
    (run / "human-review").mkdir()
    (run / "logs").mkdir()
    spec_path = robot_root / "configs/robot.yaml"
    spec = yaml.safe_load(spec_path.read_text())
    urdf_path = (spec_path.parent / spec["urdf"]).resolve()
    if sha(urdf_path) != spec["urdf_sha256"]:
        raise ValueError("Native URDF hash mismatch")
    solver = yaml.safe_load((robot_root / "configs/solvers/curobo.yaml").read_text())
    profiles = yaml.safe_load((robot_root / "configs/profiles.yaml").read_text())
    defaults = {n: float(v["default"]) for n, v in spec["joints"].items()}
    model = URDF.load(str(urdf_path), load_meshes=False)
    model.update_cfg(defaults)
    document = ET.parse(urdf_path).getroot()
    joints = {j.get("name"): j for j in document.findall("joint")}
    configurations = {}
    for side in ("left", "right"):
        active = [f"idx{(20 if side == 'left' else 60)+i}_arm_{side[0]}_joint{i}" for i in range(1, 8)]
        kin = copy.deepcopy(solver["kinematics"])
        retained = needed_joints(document, [*kin["collision_link_names"], side + "_gripper_center"], spec["reference_link"])
        kin.update({
            "format_version": 2.0, "base_link": spec["reference_link"],
            "urdf_path": str(urdf_path), "asset_root_path": str(urdf_path.parent),
            "tool_frames": [side + "_gripper_center"],
            "lock_joints": {n: defaults[n] for n in sorted(retained - set(active))},
            "self_collision_buffer": {n: 0.0 for n in kin["collision_link_names"]},
            "cspace": {"joint_names": active, "default_joint_position": [defaults[n] for n in active],
                       "null_space_weight": [1.0]*7, "cspace_distance_weight": [1.0]*7,
                       "velocity_scale": [1.0]*7,
                       "max_acceleration": 0.8, "max_jerk": 3.2},
        })
        path = run / "inputs" / f"robot-{side}.yaml"
        path.write_text(yaml.safe_dump({"robot_cfg": {"kinematics": kin}}, sort_keys=False))
        configurations[side] = str(path)

    cases = []
    for identifier, side, delta in [
        ("right_retract_10cm", "right", [-0.10, 0, 0]),
        ("right_retract_20cm", "right", [-0.20, 0, 0]),
        ("left_retract_10cm", "left", [-0.10, 0, 0]),
        ("right_retract_down", "right", [-0.10, 0, -0.05]),
    ]:
        kin = yaml.safe_load(Path(configurations[side]).read_text())["robot_cfg"]["kinematics"]
        names = kin["cspace"]["joint_names"]
        start = np.array([defaults[n] for n in names])
        model.update_cfg(defaults)
        initial = model.get_transform(side + "_gripper_center", frame_from=spec["reference_link"]).copy()
        target = initial.copy()
        target[:3, 3] += delta
        limits = [joints[n].find("limit") for n in names]
        lower = [float(x.get("lower")) for x in limits]
        upper = [float(x.get("upper")) for x in limits]

        def residual(q: np.ndarray) -> np.ndarray:
            model.update_cfg(dict(zip(names, q)))
            actual = model.get_transform(side + "_gripper_center", frame_from=spec["reference_link"])
            return np.r_[10*(actual[:3, 3]-target[:3, 3]),
                         Rotation.from_matrix(target[:3, :3].T @ actual[:3, :3]).as_rotvec(),
                         0.0001*(q-start)]

        result = least_squares(residual, start, bounds=(lower, upper), max_nfev=500,
                               ftol=1e-12, xtol=1e-12, gtol=1e-12)
        residual(result.x)
        actual = model.get_transform(side + "_gripper_center", frame_from=spec["reference_link"])
        position_error = float(np.linalg.norm(actual[:3, 3]-target[:3, 3]))
        angle_error = float(Rotation.from_matrix(target[:3, :3].T @ actual[:3, :3]).magnitude())
        if position_error > 1e-5 or angle_error > 1e-4:
            raise ValueError(f"Unresolved endpoint: {identifier}")
        cases.append({"id": identifier, "side": side, "joint_names": names,
                      "start": start.tolist(), "joint_goal_witness": result.x.tolist(),
                      "goal_pose_base": actual.tolist(), "requested_delta_m": delta,
                      "endpoint_fk_position_error_m": position_error,
                      "endpoint_fk_angle_error_rad": angle_error,
                      "endpoint_collision_verified": False,
                      "witness_purpose": "near-start numerical IK endpoint for the joint-target control; not a trajectory"})
    sources = [spec_path, urdf_path, robot_root / "configs/solvers/curobo.yaml",
               robot_root / "configs/profiles.yaml", Path(__file__), Path.cwd() / "AGENTS.md",
               Path.cwd() / "research/docs/usd-delivery-style.md",
               Path(__file__).with_name("robot_urdf.py")]
    sources.extend((urdf_path.parent / x.get("filename")).resolve() for x in document.findall(".//mesh"))
    source_hashes = {str(p): sha(p) for p in set(sources)}
    tracked = subprocess.check_output(["git", "-C", str(source), "ls-files", "-z"]).decode().split("\0")
    digest = hashlib.sha256()
    for name in sorted(n for n in tracked if n):
        p = source / name
        if p.is_file():
            digest.update(name.encode()+b"\0"+p.read_bytes())
    manifest = {"robot_root": str(robot_root), "urdf": str(urdf_path), "defaults": defaults,
                "robot_configs": configurations, "cases": cases, "seeds": [5, 17, 29],
                "modes": ["pose", "joints"], "source": str(source),
                "source_commit": subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"]).decode().strip(),
                "source_content_sha256": digest.hexdigest(), "source_hashes": source_hashes,
                "world": "empty; native full-body self-collision remains enabled",
                "catalog": None, "api_calls": 0, "api_cost_usd": 0,
                "policy": {"max_attempts_per_plan": 3, "subprocess_timeout_seconds": 180,
                           "interpolation_dt": 1/60, "self_collision_check": True,
                           "absolute_joint_velocity_rad_s": 0.4,
                           "base_body_other_arm": "held at actual published initial joint values",
                           "collision_contract": "native robot collision spheres and native self-collision exclusions"},
                "acceptance": "planner trajectory diagnostics; physical execution and human review pending"}
    write_json(run / "inputs/study.json", manifest)
    print(json.dumps({"prepared_cases": len(cases), "run": str(run)}, ensure_ascii=False))


def plan(run: Path, side: str, seed: int) -> None:
    import torch
    import curobo
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    from curobo.kinematics import KinematicsCfg
    from curobo.types import JointState

    study = json.loads((run / "inputs/study.json").read_text())
    for name, expected in study["source_hashes"].items():
        if sha(Path(name)) != expected:
            raise RuntimeError(f"Input changed since preparation: {name}")
    if not str(Path(curobo.__file__).resolve()).startswith(study["source"] + "/"):
        raise RuntimeError("Wrong cuRobo source imported")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    robot = yaml.safe_load(Path(study["robot_configs"][side]).read_text())
    native = KinematicsCfg.from_robot_yaml_file(robot)
    bounds = native.kinematics_config.joint_limits.velocity
    bounds[0].fill_(-study["policy"]["absolute_joint_velocity_rad_s"])
    bounds[1].fill_(study["policy"]["absolute_joint_velocity_rad_s"])
    configuration = MotionPlannerCfg.create(robot={"robot_cfg": {"kinematics": native}}, scene_model=None,
                                           self_collision_check=True, use_cuda_graph=True,
                                           random_seed=seed, interpolation_dt=1/60,
                                           num_ik_seeds=32, num_trajopt_seeds=4)
    planner = MotionPlanner(configuration)
    print(f"PLANNER_READY side={side} seed={seed} source={curobo.__file__}", flush=True)
    try:
        for case in [c for c in study["cases"] if c["side"] == side]:
            ordered = list(planner.joint_names)
            start_values = dict(zip(case["joint_names"], case["start"]))
            goal_values = dict(zip(case["joint_names"], case["joint_goal_witness"]))
            state = JointState.from_position(torch.tensor([[start_values[n] for n in ordered]], device="cuda", dtype=torch.float32), joint_names=ordered)
            goal = JointState.from_position(torch.tensor([[goal_values[n] for n in ordered]], device="cuda", dtype=torch.float32), joint_names=ordered)
            target = planner.compute_kinematics(goal).tool_poses.as_goal(planner.tool_frames).clone()
            for mode in study["modes"]:
                key = f"{case['id']}_{mode}_seed{seed}"
                folder = run / "results" / key
                folder.mkdir(parents=True, exist_ok=False)
                started = time.monotonic()
                print(f"START {key}", flush=True)
                result = (planner.plan_pose(target, state, max_attempts=3) if mode == "pose"
                          else planner.plan_cspace(goal, state, max_attempts=3))
                torch.cuda.synchronize()
                report = {"id": key, "case": case["id"], "side": side, "seed": seed, "mode": mode,
                          "elapsed_s": time.monotonic()-started, "success": result is not None and bool(result.success.any()),
                          "status": str(getattr(result, "status", None)), "curobo_file": curobo.__file__,
                          "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                          "self_collision_check": True, "physical_execution": False}
                if result is not None:
                    report["native_success"] = result.success.detach().cpu().tolist()
                if report["success"]:
                    trajectory = result.get_interpolated_plan()
                    native_names = list(trajectory.joint_names)
                    raw = trajectory.position.detach().cpu().numpy()
                    print(f"TRAJECTORY shape={raw.shape} native_dof={len(native_names)} active_dof={len(ordered)}", flush=True)
                    if len(set(native_names)) != len(native_names) or raw.shape[-1] != len(native_names):
                        raise RuntimeError("Native trajectory joint-name/column mismatch")
                    if any(n not in native_names for n in ordered) or any(d != 1 for d in raw.shape[:-2]):
                        raise RuntimeError("Native trajectory omitted active joints or returned multiple plans")
                    q = raw.reshape(-1, len(native_names))
                    locks = robot["robot_cfg"]["kinematics"]["lock_joints"]
                    for i, name in enumerate(native_names):
                        if name not in ordered and (name not in locks or not np.allclose(q[:, i], locks[name], atol=1e-6, rtol=0)):
                            raise RuntimeError(f"Native trajectory moved uncontrolled joint: {name}")
                    if not np.isfinite(q).all():
                        raise RuntimeError("Nonfinite native trajectory")
                    arrays = {"position": q, "joint_names": np.array(native_names), "dt": np.array(1/60)}
                    for attribute in ("velocity", "acceleration", "jerk"):
                        tensor = getattr(trajectory, attribute, None)
                        if tensor is not None:
                            arrays[attribute] = tensor.detach().cpu().numpy().reshape(-1, len(native_names))
                    np.savez_compressed(folder / "trajectory.npz", **arrays)
                    report["samples"] = len(q)
                    report["duration_s"] = (len(q)-1)/60
                    report["native_joint_names"] = native_names
                    report["active_joint_names"] = ordered
                    report["locked_joint_values_verified"] = True
                write_json(folder / "result.json", report)
                print(f"DONE {key} success={report['success']} seconds={report['elapsed_s']:.3f}", flush=True)
    finally:
        planner.destroy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "plan"])
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--robot-root", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--side", choices=["left", "right"])
    parser.add_argument("--seed", type=int, default=5)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.run.resolve(), args.robot_root.resolve(), args.source.resolve())
    else:
        plan(args.run.resolve(), args.side, args.seed)


if __name__ == "__main__":
    main()
