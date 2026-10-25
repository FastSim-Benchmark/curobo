"""Bounded single/dual-arm probes using the organization's native cuRobo planner."""
from __future__ import annotations

import argparse
import copy
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml

from study_empty_arm_motion import prepare as prepare_baseline
from study_empty_arm_motion import sha, write_json


def prepare(run: Path, robot_root: Path, source: Path) -> None:
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    from yourdfpy import URDF

    prepare_baseline(run, robot_root, source)
    study = json.loads((run / "inputs/study.json").read_text())
    model = URDF.load(study["urdf"], load_meshes=False)
    joints = {j.get("name"): j for j in ET.parse(study["urdf"]).getroot().findall("joint")}
    left = yaml.safe_load(Path(study["robot_configs"]["left"]).read_text())["robot_cfg"]["kinematics"]
    right = yaml.safe_load(Path(study["robot_configs"]["right"]).read_text())["robot_cfg"]["kinematics"]
    both = copy.deepcopy(right)
    both_names = left["cspace"]["joint_names"]+right["cspace"]["joint_names"]
    both["lock_joints"] = {n: v for n, v in both["lock_joints"].items() if n not in both_names}
    both["tool_frames"] = ["left_gripper_center", "right_gripper_center"]
    both["cspace"].update(joint_names=both_names,
                          default_joint_position=[study["defaults"][n] for n in both_names],
                          null_space_weight=[1.0]*14, cspace_distance_weight=[1.0]*14,
                          velocity_scale=[1.0]*14)
    path = run / "inputs/robot-both.yaml"
    path.write_text(yaml.safe_dump({"robot_cfg": {"kinematics": both}}, sort_keys=False))
    study["robot_configs"]["both"] = str(path)
    # A CPU sphere check selects collision-free varied starts. Native checking remains enabled.
    sphere_links, local_centers, radii = [], [], []
    for link, spheres in both["collision_spheres"].items():
        for sphere in spheres:
            sphere_links.append(link); local_centers.append(sphere["center"]); radii.append(sphere["radius"])
    local_centers, radii = np.array(local_centers), np.array(radii)
    mask = np.triu(np.ones((len(radii), len(radii)), dtype=bool), 1)
    ignored = both["self_collision_ignore"]
    for i, a in enumerate(sphere_links):
        for j, b in enumerate(sphere_links):
            if a == b or b in ignored.get(a, []) or a in ignored.get(b, []):
                mask[i, j] = False

    def clearance(values: dict) -> float:
        model.update_cfg(values)
        transforms = {n: model.get_transform(n, frame_from="mobile_base_link") for n in set(sphere_links)}
        centers = np.array([transforms[n][:3, :3]@p+transforms[n][:3, 3] for n, p in zip(sphere_links, local_centers)])
        distances = np.linalg.norm(centers[:, None]-centers[None, :], axis=-1)-radii[:, None]-radii[None, :]
        return float(distances[mask].min())

    default_clearance = clearance(study["defaults"])
    if default_clearance < -1e-5:
        raise RuntimeError("Published initial state fails native sphere-pair precheck")
    requested = [
        ("retract", "回撤20cm", [-.20, 0, 0], [0, 0, 0]),
        ("extend", "前伸8cm", [.08, 0, 0], [0, 0, 0]),
        ("outward", "向外侧移18cm", [0, .18, 0], [0, 0, 0]),
        ("inward", "向身体中线移动18cm", [0, -.18, 0], [0, 0, 0]),
        ("raise", "抬手25cm", [0, 0, .25], [0, 0, 0]),
        ("lower", "落手25cm", [0, 0, -.25], [0, 0, 0]),
        ("diagonal_up", "回撤并向外上方移动", [-.12, .16, .20], [0, 0, 0]),
        ("diagonal_down", "回撤并向内下方移动", [-.12, -.14, -.18], [0, 0, 0]),
        ("roll_positive", "TCP原地正向翻腕90度", [0, 0, 0], [90, 0, 0]),
        ("roll_negative", "TCP原地反向翻腕90度", [0, 0, 0], [-90, 0, 0]),
        ("pitch", "TCP原地俯仰45度", [0, 0, 0], [0, 45, 0]),
        ("yaw", "TCP原地转向60度", [0, 0, 0], [0, 0, 60]),
    ]
    cases = []
    for group in ["right", "left", "both"]:
        kin = {"right": right, "left": left, "both": both}[group]
        names, tools = kin["cspace"]["joint_names"], kin["tool_frames"]
        default = np.array([study["defaults"][n] for n in names])
        lo = np.array([float(joints[n].find("limit").get("lower")) for n in names])+1e-6
        hi = np.array([float(joints[n].find("limit").get("upper")) for n in names])-1e-6
        prefix = {"right": "右臂", "left": "左臂", "both": "双臂同时"}[group]
        for key, label, offset, angles in requested:
            model.update_cfg(study["defaults"])
            goals = {}
            for tool in tools:
                t = model.get_transform(tool, frame_from="mobile_base_link").copy()
                sign = 1 if tool.startswith("left") else -1
                t[:3, 3] += np.array(offset)*[1, sign, 1]
                t[:3, :3] = t[:3, :3]@Rotation.from_euler("xyz", np.array(angles)*[1, 1, sign], degrees=True).as_matrix()
                goals[tool] = t

            def residual(q: np.ndarray) -> np.ndarray:
                model.update_cfg(dict(zip(names, q)))
                terms = []
                for tool in tools:
                    actual = model.get_transform(tool, frame_from="mobile_base_link")
                    terms.extend([10*(actual[:3, 3]-goals[tool][:3, 3]),
                                  Rotation.from_matrix(goals[tool][:3, :3].T@actual[:3, :3]).as_rotvec()])
                return np.concatenate([*terms, .0001*(q-default)])

            result = least_squares(residual, default, bounds=(lo, hi), max_nfev=350,
                                   ftol=1e-10, xtol=1e-10, gtol=1e-10)
            error = residual(result.x)[:6*len(tools)].reshape(-1, 6)
            precise = bool(np.max(np.linalg.norm(error[:, :3], axis=1))/10 < 1e-4 and
                           np.max(np.linalg.norm(error[:, 3:], axis=1)) < 1e-3)
            primary = tools[-1]
            cases.append({"id": group+"_"+key, "group": group, "side": "left" if group == "left" else "right",
                          "label": prefix+label, "joint_names": names, "tool_frames": tools,
                          "start": default.tolist(), "joint_goal_witness": result.x.tolist() if precise else None,
                          "goal_pose_base": goals[primary].tolist(), "tool_goal_poses": {t: g.tolist() for t, g in goals.items()},
                          "local_endpoint_ik_precise": precise, "start_clearance_m": default_clearance,
                          "primary_tool": primary})

        rng = np.random.default_rng({"right": 301, "left": 302, "both": 303}[group])
        accepted = []
        for attempt in range(150):
            q = np.clip(default+rng.uniform(-.65, .65, len(names)), lo, hi)
            full = dict(study["defaults"]); full.update(zip(names, q))
            gap = clearance(full)
            if gap >= .002 and np.linalg.norm(q-default) >= .5:
                accepted.append((q, gap))
                if len(accepted) == 2:
                    break
        for i, (initial, gap) in enumerate(accepted, 1):
            for kind in ["nearby", "restore"]:
                endpoint = np.clip(initial+rng.uniform(-.08, .08, len(names)), lo, hi) if kind == "nearby" else default
                full = dict(study["defaults"]); full.update(zip(names, endpoint))
                if clearance(full) < -.00001:
                    continue
                goals = {t: model.get_transform(t, frame_from="mobile_base_link").copy() for t in tools}
                primary = tools[-1]
                cases.append({"id": f"{group}_{kind}_{i}", "group": group,
                              "side": "left" if group == "left" else "right",
                              "label": prefix+("不同起始姿态附近小幅动作" if kind == "nearby" else "从不同姿态回到默认位姿")+str(i),
                              "joint_names": names, "tool_frames": tools, "start": initial.tolist(),
                              "joint_goal_witness": endpoint.tolist(), "goal_pose_base": goals[primary].tolist(),
                              "tool_goal_poses": {t: g.tolist() for t, g in goals.items()},
                              "local_endpoint_ik_precise": True, "start_clearance_m": gap, "primary_tool": primary})

    study.update(cases=cases, seeds=[5], modes=["pose", "joints", "same_endpoint"],
                 experiment="expanded single/dual-arm free-space motion probes",
                 planned_case_count=len(cases), requested_slot_count=3*len(cases),
                 dual_arm_semantics="14 active joints and two simultaneous tool poses in one native plan_pose solve",
                 detection="flags are screening heuristics, not proof of unnatural motion")
    study["policy"]["subprocess_timeout_seconds"] = 240
    study["policy"]["base_body_other_arm"] = "base, body and head held; opposite arm held only for single-arm groups"
    for p in [Path(__file__).resolve(), Path(__file__).with_name("render_empty_arm_motion.py").resolve(),
              *map(Path, study["robot_configs"].values())]:
        study["source_hashes"][str(p)] = sha(p)
    write_json(run / "inputs/study.json", study)
    print(f"SUITE_PREPARED cases={len(cases)} slots={3*len(cases)} groups=right,left,both", flush=True)


def plan(run: Path, group: str) -> None:
    import torch
    import curobo
    from curobo.kinematics import KinematicsCfg
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    from curobo.types import GoalToolPose, JointState, Pose

    study = json.loads((run / "inputs/study.json").read_text())
    for name, expected in study["source_hashes"].items():
        if sha(Path(name)) != expected:
            raise RuntimeError("Input drift: "+name)
    if not str(Path(curobo.__file__).resolve()).startswith(study["source"]+"/"):
        raise RuntimeError("Wrong native cuRobo source")
    robot = yaml.safe_load(Path(study["robot_configs"][group]).read_text())
    native = KinematicsCfg.from_robot_yaml_file(robot)
    native.kinematics_config.joint_limits.velocity[0].fill_(-.4)
    native.kinematics_config.joint_limits.velocity[1].fill_(.4)
    planner = MotionPlanner(MotionPlannerCfg.create(robot={"robot_cfg": {"kinematics": native}},
                                                    scene_model=None, self_collision_check=True,
                                                    use_cuda_graph=True, random_seed=5, interpolation_dt=1/60,
                                                    num_ik_seeds=32, num_trajopt_seeds=4))
    names = list(planner.joint_names)
    print(f"READY group={group} dof={len(names)} tools={planner.tool_frames}", flush=True)
    try:
        for case in [c for c in study["cases"] if c["group"] == group]:
            current_map = dict(zip(case["joint_names"], case["start"]))
            current = JointState.from_position(torch.tensor([[current_map[n] for n in names]], device="cuda", dtype=torch.float32), joint_names=names)
            goal_poses = {t: Pose.from_matrix(torch.tensor(case["tool_goal_poses"][t], device="cuda", dtype=torch.float32).unsqueeze(0)) for t in planner.tool_frames}
            goal = GoalToolPose.from_poses(goal_poses, ordered_tool_frames=planner.tool_frames)
            pose_endpoint = None
            for mode in study["modes"]:
                key = f"{case['id']}_{mode}_seed5"
                folder = run / "results" / key; folder.mkdir(parents=True, exist_ok=False)
                report = {"id": key, "case": case["id"], "group": group, "side": case["side"],
                          "seed": 5, "mode": mode, "success": False, "planning_called": False,
                          "tool_frames": list(planner.tool_frames), "active_joint_names": names,
                          "active_dof": len(names), "physical_execution": False, "curobo_file": curobo.__file__,
                          "self_collision_check": True, "gpu": torch.cuda.get_device_name(0)}
                if mode == "joints" and case["joint_goal_witness"] is None:
                    report["status"] = "not_attempted_no_precise_local_joint_witness"
                    write_json(folder / "result.json", report); continue
                if mode == "same_endpoint" and pose_endpoint is None:
                    report["status"] = "not_attempted_pose_plan_failed"
                    write_json(folder / "result.json", report); continue
                if mode == "joints":
                    goal_map = dict(zip(case["joint_names"], case["joint_goal_witness"]))
                    joint_goal = JointState.from_position(torch.tensor([[goal_map[n] for n in names]], device="cuda", dtype=torch.float32), joint_names=names)
                elif mode == "same_endpoint":
                    joint_goal = JointState.from_position(torch.tensor([pose_endpoint], device="cuda", dtype=torch.float32), joint_names=names)
                started = time.monotonic(); print("START", key, flush=True)
                report["planning_called"] = True
                try:
                    result = planner.plan_pose(goal, current, max_attempts=3, timeout_s=60) if mode == "pose" else planner.plan_cspace(joint_goal, current, max_attempts=3, timeout_s=60)
                    torch.cuda.synchronize()
                    report.update(success=result is not None and bool(result.success.any()), status=str(getattr(result, "status", None)))
                    if result is not None:
                        report["native_success"] = result.success.detach().cpu().tolist()
                    if report["success"]:
                        trajectory = result.get_interpolated_plan(); native_names = list(trajectory.joint_names)
                        raw = trajectory.position.detach().cpu().numpy()
                        if raw.shape[-1] != len(native_names) or any(d != 1 for d in raw.shape[:-2]):
                            raise RuntimeError("Ambiguous native trajectory shape")
                        q = raw.reshape(-1, len(native_names)); locks = robot["robot_cfg"]["kinematics"]["lock_joints"]
                        for i, n in enumerate(native_names):
                            if n not in names and (n not in locks or not np.allclose(q[:, i], locks[n], atol=1e-6, rtol=0)):
                                raise RuntimeError("Uncontrolled joint moved: "+n)
                        if not np.isfinite(q).all():
                            raise RuntimeError("Nonfinite trajectory")
                        arrays = {"position": q, "joint_names": np.array(native_names), "dt": np.array(1/60)}
                        for attr in ["velocity", "acceleration", "jerk"]:
                            value = getattr(trajectory, attr, None)
                            if value is not None:
                                arrays[attr] = value.detach().cpu().numpy().reshape(-1, len(native_names))
                        np.savez_compressed(folder / "trajectory.npz", **arrays)
                        report.update(samples=len(q), duration_s=(len(q)-1)/60, native_joint_names=native_names,
                                      locked_joint_values_verified=True)
                        if mode == "pose":
                            pose_endpoint = [float(q[-1, native_names.index(n)]) for n in names]
                except Exception as error:
                    report.update(success=False, status="exception", exception_type=type(error).__name__, reason=str(error))
                report["elapsed_s"] = time.monotonic()-started
                write_json(folder / "result.json", report)
                print(f"DONE {key} success={report['success']} status={report['status']} seconds={report['elapsed_s']:.2f}", flush=True)
                write_json(run / f"progress-{group}.json", {"case": case["id"], "mode": mode, "time": time.time()})
    finally:
        planner.destroy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "plan"])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--robot-root", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--group", choices=["left", "right", "both"])
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.run.resolve(), args.robot_root.resolve(), args.source.resolve())
    else:
        plan(args.run.resolve(), args.group)
