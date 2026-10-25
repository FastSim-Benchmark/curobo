"""Render actual cuRobo trajectories with the selected robot's complete URDF meshes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import imageio.v2 as imageio
import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont
from pxr import Gf, Sdf, Usd, UsdGeom, Vt
from scipy.spatial.transform import Rotation
from yourdfpy import URDF


LABELS = {"right_retract_10cm": "右臂回撤 10 cm", "right_retract_20cm": "右臂回撤 20 cm",
          "left_retract_10cm": "左臂回撤 10 cm", "right_retract_down": "右臂回撤 10 cm 并下降 5 cm"}


def save_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False))


def look_at(eye: list[float], target: list[float]) -> np.ndarray:
    z = np.asarray(eye, dtype=float)-np.asarray(target, dtype=float)
    z /= np.linalg.norm(z)
    x = np.cross([0, 0, 1], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    result = np.eye(4)
    result[:3, :3] = np.stack([x, y, z], axis=1)
    result[:3, 3] = eye
    return result


def metrics(model: URDF, study: dict, case: dict, data: dict) -> tuple[dict, np.ndarray, np.ndarray]:
    names = data["joint_names"].tolist()
    q = data["position"]
    model.update_cfg(study["defaults"])
    tcp, elbows, rotations = [], [], []
    for row in q:
        model.update_cfg(dict(zip(names, row)))
        t = model.get_transform(case["side"]+"_gripper_center", frame_from="mobile_base_link")
        tcp.append(t[:3, 3].copy())
        rotations.append(t[:3, :3].copy())
        elbows.append(model.get_transform("arm_"+case["side"][0]+"_link4", frame_from="mobile_base_link")[:3, 3].copy())
    tcp, elbows = np.array(tcp), np.array(elbows)
    goal = np.array(case["goal_pose_base"])
    active = q[:, [names.index(n) for n in case["joint_names"]]]
    steps = np.linalg.norm(np.diff(active, axis=0), axis=1)
    chord = np.linalg.norm(active[-1]-active[0])
    angular = Rotation.from_matrix(np.einsum("ij,njk->nik", goal[:3, :3].T, rotations)).magnitude()
    tcp_length = np.linalg.norm(np.diff(tcp, axis=0), axis=1).sum()
    start_to_end = np.linalg.norm(tcp[-1]-tcp[0])
    line = tcp[-1]-tcp[0]
    if start_to_end > 1e-6:
        t = np.clip(((tcp-tcp[0]) @ line)/(line @ line), 0, 1)
        deviations = np.linalg.norm(tcp-(tcp[0]+t[:, None]*line), axis=1)
    else:
        deviations = np.linalg.norm(tcp-tcp[0], axis=1)
    out = {"tcp_path_length_m": float(tcp_length), "tcp_chord_m": float(start_to_end),
           "tcp_path_ratio": float(tcp_length/start_to_end) if start_to_end > 1e-5 else None, "tcp_max_line_deviation_m": float(deviations.max()),
           "elbow_rise_from_start_m": float(elbows[:, 2].max()-elbows[0, 2]),
           "elbow_overshoot_above_endpoints_m": float(elbows[:, 2].max()-max(elbows[0, 2], elbows[-1, 2])),
           "elbow_endpoint_rise_m": float(elbows[-1, 2]-elbows[0, 2]),
           "joint_path_length_rad": float(steps.sum()), "joint_chord_rad": float(chord),
           "joint_path_ratio": float(steps.sum()/chord) if chord > 1e-5 else None,
           "joint_total_variation_rad": np.abs(np.diff(active, axis=0)).sum(axis=0).tolist(),
           "tcp_endpoint_error_m": float(np.linalg.norm(tcp[-1]-goal[:3, 3])),
           "tcp_endpoint_angle_error_rad": float(angular[-1]), "tcp_max_angle_error_rad": float(angular.max()),
           "elbow_measurement_link": "arm_"+case["side"][0]+"_link4",
           "frame": "mobile_base_link; SI; +Z up", "physical_execution": False}
    return out, tcp, elbows


def main(run: Path, videos_only: bool = False) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pyrender

    study = json.loads((run / "inputs/study.json").read_text())
    LABELS.update({c["id"]: c.get("label", c["id"]) for c in study["cases"]})
    review = run / "human-review"
    deps = review / "deps"
    deps.mkdir(parents=True, exist_ok=True)
    model = URDF.load(study["urdf"], load_meshes=True)
    model.update_cfg(study["defaults"])
    nodes = list(model.scene.graph.nodes_geometry)
    geometry_ids = {n: f"part_{i:03d}" for i, n in enumerate(nodes)}
    if videos_only:
        results = json.loads((run / "metrics.json").read_text())
        cache = {r["id"]: dict(np.load(run / "results" / r["id"] / "trajectory.npz")) for r in results if r["success"]}
    else:
        # One binary geometry layer shared by every animation. No simplified robot proxies.
        library = Usd.Stage.CreateNew(str(deps / "robot-visuals.usdc"))
        root = UsdGeom.Xform.Define(library, "/Robot")
        library.SetDefaultPrim(root.GetPrim())
        UsdGeom.SetStageMetersPerUnit(library, 1.0)
        UsdGeom.SetStageUpAxis(library, UsdGeom.Tokens.z)
        for node in nodes:
            transform, key = model.scene.graph[node]
            mesh = model.scene.geometry[key]
            path = "/Robot/"+geometry_ids[node]
            part = UsdGeom.Xform.Define(library, path)
            part.GetPrim().SetCustomDataByKey("sourceGeometryNode", str(node))
            part.AddTransformOp().Set(Gf.Matrix4d(transform.T))
            obj = UsdGeom.Mesh.Define(library, path+"/Mesh")
            obj.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(mesh.vertices, dtype=np.float32)))
            obj.CreateFaceVertexCountsAttr([3]*len(mesh.faces))
            obj.CreateFaceVertexIndicesAttr(np.asarray(mesh.faces, dtype=np.int32).ravel().tolist())
            obj.CreateSubdivisionSchemeAttr("none")
            visual = mesh.visual.to_color() if hasattr(mesh.visual, "to_color") else mesh.visual
            colors = np.asarray(visual.vertex_colors)
            color = colors[:, :3].mean(axis=0)/255 if colors.ndim == 2 and colors.shape[1] >= 3 else np.array([0.65, 0.69, 0.74])
            obj.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        library.GetRootLayer().Save()

        results = []
        cache = {}
        for result_path in sorted((run / "results").glob("*/result.json")):
            result = json.loads(result_path.read_text())
            if not result["success"]:
                results.append(result)
                continue
            case = next(c for c in study["cases"] if c["id"] == result["case"])
            data = dict(np.load(result_path.parent / "trajectory.npz"))
            values, tcp, elbows = metrics(model, study, case, data)
            result["metrics"] = values
            cache[result["id"]] = data
            np.savez_compressed(result_path.parent / "fk-diagnostics.npz", tcp=tcp, elbow=elbows)
            stage = Usd.Stage.CreateNew(str(review / (result["id"]+"_review.usda")))
            base = UsdGeom.Xform.Define(stage, "/Root")
            stage.SetDefaultPrim(base.GetPrim())
            base.GetPrim().SetCustomDataByKey("reviewOnly", True)
            base.GetPrim().SetCustomDataByKey("physicalExecution", False)
            base.GetPrim().SetCustomDataByKey("taskId", result["id"])
            UsdGeom.SetStageMetersPerUnit(stage, 1.0)
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            stage.SetStartTimeCode(0)
            stage.SetEndTimeCode(len(data["position"])-1)
            stage.SetTimeCodesPerSecond(60)
            stage.SetFramesPerSecond(60)
            robot = UsdGeom.Xform.Define(stage, "/Root/Robot").GetPrim()
            robot.GetReferences().AddReference("./deps/robot-visuals.usdc", "/Robot")
            model.update_cfg(study["defaults"])
            previous_transforms = {}
            for frame, row in enumerate(data["position"]):
                model.update_cfg(dict(zip(data["joint_names"].tolist(), row)))
                for node in nodes:
                    transform = model.scene.graph[node][0]
                    if node in previous_transforms and np.allclose(transform, previous_transforms[node], atol=1e-12, rtol=0):
                        continue
                    attr = stage.GetPrimAtPath("/Root/Robot/"+geometry_ids[node]).GetAttribute("xformOp:transform")
                    attr.Set(Gf.Matrix4d(transform.T), frame)
                    previous_transforms[node] = transform.copy()
            stage.GetRootLayer().Save()
            reopened = Usd.Stage.Open(str(review / (result["id"]+"_review.usda")))
            if reopened.GetCompositionErrors() or len([p for p in reopened.Traverse() if p.IsA(UsdGeom.Mesh)]) != len(nodes):
                raise RuntimeError("Review composition incomplete: "+result["id"])
            result["review"] = result["id"]+"_review.usda"
            result["human_review"] = "pending"
            result["visual_geometry_nodes"] = len(nodes)
            time_axis = np.arange(len(tcp))/60
            fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))
            axes[0].plot(time_axis, 100*(elbows[:, 2]-elbows[0, 2]), color="#d97706")
            axes[0].set(xlabel="Time (s)", ylabel="Elbow rise (cm)")
            for i, label in enumerate(["x", "y", "z"]):
                axes[1].plot(time_axis, 100*(tcp[:, i]-tcp[0, i]), label=label)
            axes[1].set(xlabel="Time (s)", ylabel="TCP displacement (cm)")
            axes[1].legend()
            for name in case["joint_names"]:
                index = data["joint_names"].tolist().index(name)
                axes[2].plot(time_axis, data["position"][:, index]-data["position"][0, index], label=name.split("_")[-1])
            axes[2].set(xlabel="Time (s)", ylabel="Joint displacement (rad)")
            axes[2].legend(fontsize=6, ncol=2)
            fig.suptitle(result["id"])
            fig.tight_layout()
            fig.savefig(review / (result["id"]+"_curves.png"), dpi=120)
            plt.close(fig)
            results.append(result)
            print("EXPORTED", result["id"], flush=True)

        save_json(run / "metrics.json", results)
        save_json(review / "INDEX.json", {"entries": results, "geometry_source": study["urdf"],
                                         "rendered_full_robot": True, "human_review": "pending",
                                         "scale": 1.0, "workspace": "not applicable: free-space arm trajectory study"})
    scene = pyrender.Scene(bg_color=[0.94, 0.96, 0.98, 1], ambient_light=[0.20, 0.20, 0.20])
    render_nodes = {}
    for node in nodes:
        transform, key = model.scene.graph[node]
        mesh = model.scene.geometry[key].copy()
        # CAD mesh material is used where available; no collision geometry replacement.
        material = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.64, 0.70, 0.77, 1.0], metallicFactor=0.25, roughnessFactor=0.65)
        render_nodes[node] = scene.add(pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False), pose=transform)
    floor = trimesh.creation.box(extents=[5, 5, 0.02])
    floor.apply_translation([0, 0, -0.01])
    floor.visual.face_colors = [210, 218, 229, 255]
    scene.add(pyrender.Mesh.from_trimesh(floor, smooth=False))
    camera = pyrender.PerspectiveCamera(yfov=np.deg2rad(36))
    camera_node = scene.add(camera)
    light_node = scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=1.8))
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=0.6), pose=look_at([-3, 2, 3], [0, 0, 1]))
    renderer = pyrender.OffscreenRenderer(640, 720)
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    font = ImageFont.truetype(font_path, 19)
    small_font = ImageFont.truetype(font_path, 16)
    videos = []
    try:
        for case in study["cases"]:
            if study.get("video_case_ids") and case["id"] not in study["video_case_ids"]:
                continue
            control_mode = study.get("video_control_mode_by_case", {}).get(case["id"], "joints")
            pair = [next(r for r in results if r["case"] == case["id"] and r["mode"] == mode and r["seed"] == 5) for mode in ["pose", control_mode]]
            if not all(r["success"] for r in pair):
                continue
            datasets = [cache[r["id"]] for r in pair]
            max_duration = max(r["duration_s"] for r in pair)
            eye = [3.8, -.8, 2.0] if case.get("group") == "both" else [2.5, -3.2 if case["side"] == "right" else 3.2, 2.0]
            view = look_at(eye, [0.15, 0, 0.92])
            scene.set_pose(camera_node, view)
            scene.set_pose(light_node, view)
            video = review / (case["id"]+"_comparison.webm")
            snapshot_frames = []
            with imageio.get_writer(video, format="FFMPEG", fps=20, codec="libvpx", quality=None, macro_block_size=16,
                                    ffmpeg_params=["-crf", "10", "-b:v", "2M", "-deadline", "realtime", "-cpu-used", "4"]) as writer:
                count = int(np.ceil(max_duration*20))+21
                for i in range(count):
                    seconds = max(0, i/20-0.4)
                    images = []
                    for result, data in zip(pair, datasets):
                        frame = min(int(round(seconds*60)), len(data["position"])-1)
                        model.update_cfg(study["defaults"])
                        model.update_cfg(dict(zip(data["joint_names"].tolist(), data["position"][frame])))
                        for node in nodes:
                            scene.set_pose(render_nodes[node], model.scene.graph[node][0])
                        color, _ = renderer.render(scene)
                        im = Image.fromarray(color)
                        draw = ImageDraw.Draw(im)
                        draw.rectangle([0, 0, 640, 100], fill="#101b2a")
                        method = "TCP目标：规划器自行选择关节终点" if result["mode"] == "pose" else ("同一TCP目标：指定另一关节终点" if result["mode"] == "joints" else "相同关节终点：仅改变规划接口")
                        method = study.get("video_method_labels", {}).get(result["mode"], method)
                        draw.text((18, 8), LABELS[case["id"]], font=font, fill="white")
                        draw.text((18, 37), method, font=small_font, fill="#bac8da")
                        measured_side = "左肘" if case["side"] == "left" else "右肘"
                        draw.text((18, 65), f"t = {frame/60:.2f} s  |  {measured_side}最大上升 {100*result['metrics']['elbow_rise_from_start_m']:.1f} cm", font=small_font, fill="#ffcc80")
                        draw.text((18, 694), "实际规划轨迹的运动学回放 · 非物理执行", font=small_font, fill="#172435")
                        images.append(np.asarray(im))
                    combined = np.concatenate(images, axis=1)
                    writer.append_data(combined)
                    if i in [8, int(max_duration*10)+8, count-1]:
                        snapshot_frames.append(combined)
                imageio.imwrite(review / (case["id"]+"_preview.png"), np.concatenate(snapshot_frames, axis=0))
            videos.append({"case": case["id"], "filename": video.name, "seed": 5})
            print("VIDEO", video.name, flush=True)
    finally:
        renderer.delete()

    save_json(review / "videos.json", videos)
    rows = []
    for r in results:
        if not r["success"]:
            status = "本轮未得解" if r.get("planning_called", True) else "对照未调用"
            rows.append(f"| {r['id']} | {status} | | | |")
            continue
        m = r["metrics"]
        rows.append(f"| {r['id']} | {100*m['elbow_rise_from_start_m']:.2f} cm | {100*m['elbow_overshoot_above_endpoints_m']:.2f} cm | [USD]({r['review']}) | [曲线]({r['id']}_curves.png) |")
    video_links = "\n".join(f"- [{LABELS[v['case']]}]({v['filename']})" for v in videos)
    index = ("# 空场景机器人运动复现\n\n真实规划轨迹的运动学回放；未运行物理仿真。视频使用 seed 5。\n\n"
             + video_links + "\n\n## 完整候选索引\n\n"
             + "| 候选 | 肘部相对起点上升 | 超过两端的额外抬高 | USD | 曲线 |\n|---|---:|---:|---|---|\n"
             + "\n".join(rows) + "\n\n人工审核 pending；USD 时间轴为60fps，查看器额外scale保持1.0。\n")
    (review / "INDEX.md").write_text(index)
    print("REVIEW_READY", review, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--videos-only", action="store_true")
    args = parser.parse_args()
    main(args.run.resolve(), videos_only=args.videos_only)
