"""Screen actual arm paths for detours and redundant end-posture changes."""
from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from yourdfpy import URDF


def main(run: Path) -> None:
    study = json.loads((run / "inputs/study.json").read_text())
    model = URDF.load(study["urdf"], load_meshes=False)
    limits = {j.get('name'): j.find('limit') for j in ET.parse(study['urdf']).getroot().findall('joint')}
    cases = {c["id"]: c for c in study["cases"]}
    rows, identities = [], {}
    for path in sorted((run / "results").glob("*/result.json")):
        row = json.loads(path.read_text())
        case = cases[row["case"]]
        if row["success"]:
            data = np.load(path.parent / "trajectory.npz")
            q, names = data["position"], data["joint_names"].tolist()
            active = q[:, [names.index(n) for n in case["joint_names"]]]
            difference = np.diff(active, axis=0)
            path_length = float(np.linalg.norm(difference, axis=1).sum())
            chord = float(np.linalg.norm(active[-1]-active[0]))
            variation = np.abs(difference).sum(axis=0)
            backtrack = .5*(variation-np.abs(active[-1]-active[0]))
            tools = {t: {"tcp": [], "elbow": [], "rotation": []}
                     for t in ['left_gripper_center', 'right_gripper_center']}
            model.update_cfg(study["defaults"])
            for positions in q:
                model.update_cfg(dict(zip(names, positions)))
                for tool, values in tools.items():
                    side = "l" if tool.startswith("left") else "r"
                    transform = model.get_transform(tool, frame_from="mobile_base_link")
                    values["tcp"].append(transform[:3, 3].copy())
                    values["rotation"].append(transform[:3, :3].copy())
                    values["elbow"].append(model.get_transform(f"arm_{side}_link4", frame_from="mobile_base_link")[:3, 3].copy())
            tool_metrics = {}
            fk_arrays = {}
            for tool, values in tools.items():
                tcp, elbow, rotation = (np.array(values[k]) for k in ["tcp", "elbow", "rotation"])
                goal = np.eye(4)
                goal[:3, :3], goal[:3, 3] = rotation[0], tcp[0]
                if tool in case['tool_goal_poses']:
                    goal = np.array(case['tool_goal_poses'][tool])
                length = float(np.linalg.norm(np.diff(tcp, axis=0), axis=1).sum())
                tcp_chord = float(np.linalg.norm(tcp[-1]-tcp[0]))
                line = tcp[-1]-tcp[0]
                if tcp_chord > 1e-6:
                    fraction = np.clip(((tcp-tcp[0])@line)/(line@line), 0, 1)
                    deviation = np.linalg.norm(tcp-(tcp[0]+fraction[:, None]*line), axis=1)
                else:
                    deviation = np.linalg.norm(tcp-tcp[0], axis=1)
                angular_length = float(Rotation.from_matrix(np.einsum("nji,njk->nik", rotation[:-1], rotation[1:])).magnitude().sum())
                tool_metrics[tool] = {
                    "tcp_path_m": length, "tcp_chord_m": tcp_chord,
                    "tcp_path_ratio": length/tcp_chord if tcp_chord > .01 else None,
                    "tcp_max_line_deviation_m": float(deviation.max()),
                    "elbow_rise_m": float(elbow[:, 2].max()-elbow[0, 2]),
                    "elbow_overshoot_m": float(elbow[:, 2].max()-max(elbow[0, 2], elbow[-1, 2])),
                    "endpoint_position_error_m": float(np.linalg.norm(tcp[-1]-goal[:3, 3])),
                    "endpoint_angle_error_rad": float(Rotation.from_matrix(goal[:3, :3].T@rotation[-1]).magnitude()),
                    "orientation_path_rad": angular_length,
                    "orientation_chord_rad": float(Rotation.from_matrix(rotation[0].T @ rotation[-1]).magnitude()),
                    "tcp_max_start_offset_m": float(np.linalg.norm(tcp-tcp[0], axis=1).max()),
                    "requested_translation_m": float(np.linalg.norm(goal[:3, 3]-tcp[0])),
                    "requested_rotation_rad": float(Rotation.from_matrix(rotation[0].T @ goal[:3, :3]).magnitude()),
                }
                fk_arrays[tool+"_tcp"] = tcp; fk_arrays[tool+"_elbow"] = elbow
            np.savez_compressed(path.parent / "suite-fk.npz", **fk_arrays)
            violation = 0.0
            for i, name in enumerate(names):
                limit = limits.get(name)
                if limit is not None and limit.get('lower') is not None:
                    violation = max(violation, float(float(limit.get('lower'))-q[:, i].min()),
                                    float(q[:, i].max()-float(limit.get('upper'))))
            row["screening"] = {"joint_path_rad": path_length, "joint_chord_rad": chord,
                                 "joint_path_ratio": path_length/chord if chord > .001 else None,
                                 "max_joint_backtrack_rad": float(backtrack.max()),
                                 "joint_variation_by_joint_rad": dict(zip(case['joint_names'], map(float, variation))),
                                 "joint_limit_violation_rad": violation,
                                 "observed_max_velocity_rad_s": float(np.abs(difference).max()/float(data['dt'])),
                                 "backtrack_by_joint_rad": dict(zip(case["joint_names"], map(float, backtrack))),
                                 "tools": tool_metrics}
            key = hashlib.sha256(data["joint_names"].tobytes()+q.tobytes()+data["dt"].tobytes()).hexdigest()
            identities.setdefault(key, []).append(row["id"])
        rows.append(row)
        if len(rows) % 50 == 0:
            print(f'ANALYZED {len(rows)} latest={row["id"]}', flush=True)
    grouped = {}
    for row in rows:
        grouped.setdefault(row["case"], {})[row["mode"]] = row
    findings = []
    for case, modes in grouped.items():
        pose = modes.get("pose", {})
        if not pose.get("success"):
            continue
        m = pose["screening"]
        flags, score = [], 0.0
        if m["joint_path_ratio"] is not None and m["joint_path_ratio"] > 1.5 and m["joint_path_rad"]-m["joint_chord_rad"] > .4:
            flags.append("joint_detour"); score += m["joint_path_ratio"]
        if m["max_joint_backtrack_rad"] > .3:
            flags.append("joint_reversal"); score += m["max_joint_backtrack_rad"]*3
        for tool, t in m["tools"].items():
            if t["elbow_overshoot_m"] > .06:
                flags.append(tool+":elbow_overshoot"); score += t["elbow_overshoot_m"]*15
            if t["tcp_path_ratio"] is not None and t["tcp_path_ratio"] > 1.5 and t["tcp_path_m"]-t["tcp_chord_m"] > .05:
                flags.append(tool+":tcp_detour"); score += t["tcp_path_ratio"]
        comparisons = {}
        for control in ["joints", "same_endpoint"]:
            other = modes.get(control)
            if other and other["success"]:
                denominator = other["screening"]["joint_path_rad"]
                ratio = m["joint_path_rad"]/denominator if denominator > .001 else None
                comparisons[control] = {"pose_joint_path_rad": m["joint_path_rad"],
                                        "control_joint_path_rad": denominator, "ratio": ratio}
                if ratio and ratio > (2.0 if control == "joints" else 1.25) and m["joint_path_rad"]-denominator > .5:
                    flags.append("endpoint_choice" if control == "joints" else "same_endpoint_longer_path")
                    score += min(ratio, 12)
        findings.append({"case": case, "label": cases[case]["label"], "group": cases[case]["group"],
                         "flags": flags, "score": score, "comparisons": comparisons,
                         "interpretation": "screening candidate; visual review and constraint-aware controls required"})
    findings.sort(key=lambda x: x["score"], reverse=True)
    summary = {"requested_cases": len(study["cases"]), "requested_slots": study["requested_slot_count"],
               "recorded_slots": len(rows), "actual_planning_calls": sum(r["planning_called"] for r in rows),
               "successful_calls": sum(r["success"] for r in rows),
               "skipped_slots": sum(not r["planning_called"] for r in rows),
               "failed_planning_calls": sum(r["planning_called"] and not r["success"] for r in rows),
               "unique_trajectories": len(identities), "flagged_pose_cases": sum(bool(f["flags"]) for f in findings),
               "by_group": {g: {"pose_success": sum(r["group"] == g and r["mode"] == "pose" and r["success"] for r in rows),
                                "pose_failed": sum(r["group"] == g and r["mode"] == "pose" and not r["success"] for r in rows)} for g in ["left", "right", "both"]},
               "api_calls": 0, "api_cost_usd": 0, "physical_execution": False}
    (run / "analysis.json").write_text(json.dumps({"summary": summary, "findings": findings, "results": rows,
                                                  "identical_trajectory_groups": identities}, ensure_ascii=False, indent=2, allow_nan=False))
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    print(json.dumps(findings[:8], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    main(parser.parse_args().run.resolve())
