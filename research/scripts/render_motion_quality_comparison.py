"""Render every saved request in a four-version benchmark as a local WebM."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

SOURCES = ("original", "commit20133", "current", "final")
LABELS = ("原始版本 · 30dd119", "历史版本 · 20133a7", "修改前 · 当前基础", "本轮优化 · LM + L-BFGS")
RUNS = ("compact-sequence01", "compact-obstacle01", "compact-independent01")
FPS = 15
WIDTH, BODY_HEIGHT, PANEL_HEIGHT = 640, 416, 544
RENDERER: Any = None


def digest(path: Path) -> str:
    """Hash one frozen input or generated artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, data: Any) -> None:
    """Atomically save a readable record."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def build_specs(root: Path, run_names: tuple = RUNS, expected_count: int = 300) -> list[dict]:
    """Match all four versions by run, group, case and repeat, including failures."""
    specs = []
    for run_name in run_names:
        runs = [root / "variants" / source / "runs" / run_name for source in SOURCES]
        studies = [json.loads((run / "inputs/study.json").read_text()) for run in runs]
        results = [json.loads((run / "analysis.json").read_text())["results"] for run in runs]
        row_maps = [{(r["case"], r["repeat"]): r for r in rows} for rows in results]
        assert all(set(rows) == set(row_maps[0]) for rows in row_maps)
        case_maps = [{c["id"]: c for c in study["cases"]} for study in studies]
        order = {case["id"]: i for i, case in enumerate(studies[0]["cases"])}
        for key in sorted(row_maps[0], key=lambda k: (order[k[0]], k[1])):
            cases = [mapping[key[0]] for mapping in case_maps]
            assert all(case == cases[0] for case in cases), key
            assert all(s["defaults"] == studies[0]["defaults"] for s in studies)
            rows = [mapping[key] for mapping in row_maps]
            assert all(row["planning_called"] for row in rows)
            number = len(specs) + 1
            stem = f"{number:03d}_{key[0]}" + ("_fresh" if "independent" in run_name else "")
            specs.append(dict(number=number, stem=stem, case=cases[0], rows=rows,
                              runs=[str(r) for r in runs], run_name=run_name,
                              study=studies[0], repeat=key[1]))
    assert len(specs) == expected_count and len({s["stem"] for s in specs}) == expected_count
    for spec in specs:
        spec["total"] = expected_count
    return specs


def look_at(eye: list, target: list) -> Any:
    """Build a Z-up OpenGL camera pose."""
    import numpy as np

    z = np.asarray(eye, dtype=float) - np.asarray(target, dtype=float)
    z /= np.linalg.norm(z)
    x = np.cross([0, 0, 1], z)
    x /= np.linalg.norm(x)
    pose = np.eye(4)
    pose[:3, :3] = np.stack([x, np.cross(z, x), z], axis=1)
    pose[:3, 3] = eye
    return pose


class MeshRenderer:
    """Reuse native robot meshes and one EGL context within each render worker."""

    def __init__(self, study: dict) -> None:
        import numpy as np
        import pyrender
        import trimesh
        from PIL import ImageFont
        from yourdfpy import URDF

        self.model = URDF.load(study["urdf"], load_meshes=True)
        self.model.update_cfg(study["defaults"])
        self.scene = pyrender.Scene(bg_color=[.94, .96, .98, 1], ambient_light=[.25] * 3)
        self.nodes = {}
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[.64, .70, .77, 1], metallicFactor=.25, roughnessFactor=.65)
        for node in self.model.scene.graph.nodes_geometry:
            transform, key = self.model.scene.graph[node]
            mesh = pyrender.Mesh.from_trimesh(self.model.scene.geometry[key].copy(),
                                             material=material, smooth=False)
            self.nodes[node] = self.scene.add(mesh, pose=transform)
        assert len(self.nodes) == 86
        floor = trimesh.creation.box(extents=[5, 5, .02])
        floor.apply_translation([0, 0, -.01])
        self.scene.add(pyrender.Mesh.from_trimesh(floor, smooth=False))
        self.camera = self.scene.add(pyrender.PerspectiveCamera(yfov=np.deg2rad(36)))
        self.light = self.scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=1.8))
        self.scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=.6),
                       pose=look_at([-3, 2, 3], [0, 0, 1]))
        self.renderer = pyrender.OffscreenRenderer(WIDTH, BODY_HEIGHT)
        font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
        self.font = ImageFont.truetype(font_path, 19)
        self.small = ImageFont.truetype(font_path, 15)
        self.obstacles: list = []

    def setup(self, case: dict) -> None:
        """Use identical views and exact saved obstacles in all four panels."""
        import pyrender
        import trimesh

        for node in self.obstacles:
            self.scene.remove_node(node)
        self.obstacles = []
        world = case.get("scene") or {}
        assert not (set(world) - {"cuboid"}), "Unsupported obstacle geometry"
        for obstacle in world.get("cuboid", {}).values():
            pose = obstacle["pose"]
            transform = trimesh.transformations.quaternion_matrix(pose[3:7])
            transform[:3, 3] = pose[:3]
            mesh = trimesh.creation.box(extents=obstacle["dims"])
            material = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[.9, .42, .12, 1], metallicFactor=0, roughnessFactor=.8)
            self.obstacles.append(self.scene.add(
                pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False), pose=transform))
        eye = ([3.7, -.5, 2.0] if case["group"] == "both" else
               [2.8, -2.5 if case["group"] == "right" else 2.5, 2.0])
        view = look_at(eye, [.05, 0, .98])
        self.scene.set_pose(self.camera, view)
        self.scene.set_pose(self.light, view)

    def image(self, configuration: dict) -> Any:
        """Render one saved joint configuration without changing its scale."""
        self.model.update_cfg(configuration)
        for node, render_node in self.nodes.items():
            self.scene.set_pose(render_node, self.model.scene.graph[node][0])
        color, _ = self.renderer.render(self.scene)
        return color


def render_one(spec: dict, output: str) -> dict:
    """Encode a complete four-panel replay, never inventing unsuccessful motion."""
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image, ImageDraw

    global RENDERER
    out = Path(output)
    case, rows, study = spec["case"], spec["rows"], spec["study"]
    file = out / (spec["stem"] + ".webm")
    record_file = file.with_suffix(".json")
    inputs = {}
    data = []
    for run, row in zip(spec["runs"], rows):
        folder = Path(run) / "results" / row["id"]
        for path in (Path(run) / "inputs/study.json", Path(run) / "analysis.json",
                     folder / "result.json"):
            inputs[str(path)] = digest(path)
        if row["success"]:
            path = folder / "trajectory.npz"
            inputs[str(path)] = digest(path)
            with np.load(path) as saved:
                trajectory = {k: saved[k] for k in saved.files}
            assert np.isfinite(trajectory["position"]).all()
            duration = (len(trajectory["position"]) - 1) * float(trajectory["dt"])
            assert abs(duration - row["duration_s"]) < 1e-6
            data.append(trajectory)
        else:
            data.append(None)
    if record_file.exists():
        record = json.loads(record_file.read_text())
        assert record["input_hashes"] == inputs
        assert file.exists() and record["video_sha256"] == digest(file)
        assert record["renderer_sha256"] == digest(Path(__file__))
        return record
    if RENDERER is None:
        RENDERER = MeshRenderer(study)
    RENDERER.setup(case)
    maximum = max((r.get("duration_s", 0) for r in rows), default=0)
    frames = max(45, int(np.ceil((maximum + 1.0) * FPS)) + 1)
    static = dict(study["defaults"])
    static.update(dict(zip(case["joint_names"], case["start"])))
    body_cache: dict = {}
    preview_frames = []
    preview_indices = {6, frames // 2, frames - 1}
    temporary = file.with_name(file.stem + ".partial.webm")
    with imageio.get_writer(
        temporary, format="FFMPEG", fps=FPS, codec="libvpx", quality=None,
        macro_block_size=16, ffmpeg_params=["-crf", "10", "-b:v", "2M",
                                           "-deadline", "realtime", "-cpu-used", "4",
                                           "-threads", "1"],
    ) as writer:
        for i in range(frames):
            seconds = max(0., i / FPS - .4)
            panels = []
            for panel, (source, label, row, trajectory) in enumerate(zip(SOURCES, LABELS, rows, data)):
                if trajectory is None:
                    frame = 0
                    configuration = static
                else:
                    frame = min(round(seconds / float(trajectory["dt"])),
                                len(trajectory["position"]) - 1)
                    configuration = dict(study["defaults"])
                    configuration.update(dict(zip(trajectory["joint_names"].tolist(),
                                                  trajectory["position"][frame])))
                if panel not in body_cache or body_cache[panel][0] != frame:
                    body_cache[panel] = (frame, RENDERER.image(configuration))
                image = Image.new("RGB", (WIDTH, PANEL_HEIGHT), "#101b2a")
                image.paste(Image.fromarray(body_cache[panel][1]), (0, 104))
                draw = ImageDraw.Draw(image)
                draw.text((10, 3), label, font=RENDERER.font, fill="white")
                timing = f"规划 {row['elapsed_s'] * 1000:.2f} ms"
                state = "成功" if row["success"] else "未得解 · 无可播放轨迹"
                color = "#a3e6b7" if row["success"] else "#ff9c9c"
                draw.text((10, 29), f"{state}  |  {timing}", font=RENDERER.small, fill=color)
                if row["success"]:
                    tools = row["screening"]["tools"]
                    left, right = tools["left_gripper_center"], tools["right_gripper_center"]
                    draw.text((10, 52),
                              f"TCP 左 {left['tcp_path_m'] * 100:.2f} / 右 {right['tcp_path_m'] * 100:.2f} cm"
                              f"  |  关节 {row['screening']['joint_path_rad']:.3f} rad",
                              font=RENDERER.small, fill="#ffcc80")
                    draw.text((10, 76),
                              f"姿态转动总量 左 {np.rad2deg(left['orientation_path_rad']):.1f} / 右 {np.rad2deg(right['orientation_path_rad']):.1f}°",
                              font=RENDERER.small, fill="#ffcc80")
                    shown_time = min(seconds, row["duration_s"])
                    bottom = f"同步 t={shown_time:.2f} s  |  真实网格 · 运动学回放"
                else:
                    draw.text((10, 55), "仅显示该次请求的起始姿态，不代表可执行解", font=RENDERER.small, fill="#bac8da")
                    bottom = "未得解  |  静态起始姿态 · 非物理执行"
                draw.text((10, 522), bottom, font=RENDERER.small, fill="#bac8da")
                draw.text((10, 109), f"{spec['number']:03d}/{spec['total']}  {case['id']}",
                          font=RENDERER.small, fill="#172435")
                if "independent" in spec["run_name"]:
                    draw.text((10, 131), "全新规划器独立调用", font=RENDERER.small, fill="#172435")
                elif "goal_rotation_deg" in case:
                    changes = [
                        f"{'左' if tool.startswith('left') else '右'} "
                        f"{distance * 100:.0f} cm / {case['goal_rotation_deg'][tool]:.0f}°"
                        for tool, distance in case["goal_translation_m"].items()
                    ]
                    draw.text((10, 131), "目标变化：" + "；".join(changes),
                              font=RENDERER.small, fill="#172435")
                panels.append(np.asarray(image))
            combined = np.concatenate((np.concatenate(panels[:2], axis=1),
                                       np.concatenate(panels[2:], axis=1)), axis=0)
            writer.append_data(combined)
            if i in preview_indices:
                preview_frames.append(combined)
    temporary.replace(file)
    preview = file.with_suffix(".jpg")
    imageio.imwrite(preview, np.concatenate(preview_frames, axis=1), quality=85)
    assert all(digest(Path(path)) == checksum for path, checksum in inputs.items())
    record = dict(number=spec["number"], case=case["id"], group=case["group"],
                  stratum=rows[0]["stratum"], evaluation_run=spec["run_name"],
                  repeat=spec["repeat"], video=str(file), preview=str(preview),
                  frames=frames, fps=FPS, resolution=[2 * WIDTH, 2 * PANEL_HEIGHT],
                  video_sha256=digest(file), renderer_sha256=digest(Path(__file__)),
                  input_hashes=inputs, source_runs=spec["runs"],
                  panels=[dict(source=s, label=label, status=r["status"],
                               elapsed_s=r["elapsed_s"], metrics=r.get("screening"),
                               duration_s=r.get("duration_s"))
                          for s, label, r in zip(SOURCES, LABELS, rows)],
                  scene=case.get("scene"), native_mesh_count=86,
                  timing="same wall-clock seconds, 0.4 s initial hold; completed paths hold endpoint",
                  physical_execution=False, human_naturalness_review="pending")
    write_json(record_file, record)
    return record


def write_index(output: Path, entries: list[dict], complete: bool, expected_count: int) -> None:
    """Keep every request, including independent repeats and no-solution rows."""
    entries = sorted(entries, key=lambda e: e["number"])
    write_json(output / "INDEX.json", dict(entries=entries, complete=complete,
                                           expected_count=expected_count, count=len(entries)))
    lines = [f"# {expected_count} 次请求：四版本完整对照视频", "",
             "左上：原始 30dd119；右上：20133a7；左下：修改前；右下：本轮 LM + L-BFGS 优化。", "",
             "每次请求一个视频；相同秒数同步播放，较早完成的轨迹停在终点，不变速对齐。"
             "仅规划计算耗时参与时间预算，轨迹执行时长不参与验收。", "",
             "使用真实尺寸的 86 个机器人网格，障碍案例显示原测试障碍物；灰色地面仅作视觉参照。"
             "未得解显示静态起始姿态，不补造轨迹。所有内容均为运动学回放，未进行物理执行，人工自然性审核待完成。", "",
             f"当前文件数：**{len(entries)} / {expected_count}**。每个视频有同名 JSON（来源、哈希、四版本指标）和 JPG（起点/中间/终点预览）。", "",
             "| 序号 | 请求 / 视频 | 测试组 | 原始 / 20133a7 / 修改前 / 本轮 |", "| --- | --- | --- | --- |"]
    for entry in entries:
        statuses = " / ".join("成功" if p["status"] == "success" else "未得解" for p in entry["panels"])
        category = "独立调用" if "independent" in entry["evaluation_run"] else entry["stratum"]
        lines.append(f"| {entry['number']:03d} | [{entry['case']}]({entry['video']}) | {category} | {statuses} |")
    (output / "INDEX.md").write_text("\n".join(lines) + "\n")
    (output / f"ALL_{expected_count}.m3u").write_text("#EXTM3U\n" + "\n".join(e["video"] for e in entries) + "\n")


def main() -> None:
    """Run bounded render workers inside the user's 20-CPU allowance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--numbers", type=int, nargs="*")
    parser.add_argument("--runs", nargs="+", default=list(RUNS))
    parser.add_argument("--expected-count", type=int, default=300)
    parser.add_argument("--summary-file", type=Path)
    args = parser.parse_args()
    assert 1 <= args.workers <= 10
    root, output = args.root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cpus = sorted(os.sched_getaffinity(0))[:20]
    os.sched_setaffinity(0, cpus)
    specs = build_specs(root, tuple(args.runs), args.expected_count)
    if args.numbers:
        specs = [s for s in specs if s["number"] in args.numbers]
    write_json(output / "render-contract.json", dict(
        requested=args.expected_count, selected_for_this_invocation=[s["number"] for s in specs],
        sources=SOURCES, workers=args.workers, cpu_affinity=cpus,
        renderer=str(Path(__file__).resolve()), renderer_sha256=digest(Path(__file__)),
        planning_rerun=False, source_summary_sha256=digest(
            args.summary_file or root / "compact-comparison.json")))
    entries = []
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(render_one, spec, str(output)): spec["number"] for spec in specs}
        for future in as_completed(futures):
            record = future.result()
            entries.append(record)
            write_index(output, entries, complete=len(entries) == args.expected_count,
                        expected_count=args.expected_count)
            print(f"RENDER_DONE {record['number']:03d} {record['case']} ({len(entries)}/{len(specs)})", flush=True)
    print(f"RENDER_COMPLETE {len(entries)}", flush=True)


if __name__ == "__main__":
    main()
