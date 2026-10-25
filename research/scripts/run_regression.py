"""Prepare and supervise standalone, finite planner regression runs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "research/scripts"
FIXTURES = ROOT / "research/fixtures/empty_world_v1"
ROBOT = ROOT / "research/assets/g2_crsb_op3_e"
GPU_PYTHON = Path("/home/ubuntu/miniforge3/envs/fastsim_vnext/bin/python")
USD_PYTHON = Path("/home/ubuntu/miniforge3/envs/isaac5/bin/python")


def digest(path: Path) -> str:
    """Hash one saved input or source file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    """Save a diagnostic record without changing historical evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def code_files() -> list[Path]:
    """Collect native and local experiment sources, including uncommitted files."""
    suffixes = {".py", ".cu", ".cpp", ".cuh", ".h", ".yaml", ".yml", ".json", ".toml"}
    paths = {p for p in (ROOT / "curobo").rglob("*")
             if p.is_file() and p.suffix in suffixes and "__pycache__" not in p.parts}
    paths.update(SCRIPTS.glob("*.py"))
    paths.update([ROOT / "AGENTS.md", ROOT / "README.md", ROOT / "pyproject.toml"])
    return sorted(paths)


def prepare(args: argparse.Namespace) -> None:
    """Rebase frozen numeric inputs onto this independent source and asset copy."""
    run = args.run.resolve()
    if run.exists():
        raise FileExistsError(f"Use a new run directory: {run}")
    if not run.is_relative_to(ROOT / "runs"):
        raise ValueError("New runs must be inside this project's runs/ directory")
    study = json.loads((FIXTURES / "study.json").read_text())
    study["historical_source"] = {key: study[key] for key in
                                  ("source", "source_commit", "source_content_sha256")}
    selected = [case for case in study["cases"] if case["group"] == args.group]
    if args.case:
        selected = [case for case in selected if case["id"] == args.case]
    if not selected:
        raise ValueError("No matching regression cases")
    study["cases"] = selected
    study["planned_case_count"] = len(selected)
    study["requested_slot_count"] = len(selected) * len(study["modes"])
    study["robot_root"] = str(ROBOT)
    study["urdf"] = str(ROBOT / "assets/urdf/g2_crsB_op3_e.urdf")
    study["source"] = str(ROOT)
    study["source_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    (run / "inputs").mkdir(parents=True)
    for group in ["left", "right", "both"]:
        robot = yaml.safe_load((FIXTURES / f"robot-{group}.yaml").read_text())
        kin = robot["robot_cfg"]["kinematics"]
        kin["urdf_path"] = study["urdf"]
        kin["asset_root_path"] = str(Path(study["urdf"]).parent)
        target = run / "inputs" / f"robot-{group}.yaml"
        target.write_text(yaml.safe_dump(robot, sort_keys=False))
        study["robot_configs"][group] = str(target)
    study["diagnostic"] = {"policy": args.policy, "advance_lm": 0, "lm_state": None,
                           "purpose": "Independent planner research project regression",
                           "per_request_timeout_s": 60, "per_request_max_attempts": 3}
    study["policy"]["subprocess_timeout_seconds"] = 240
    files = code_files()
    source_digest = hashlib.sha256()
    for path in files:
        source_digest.update(str(path.relative_to(ROOT)).encode() + b"\0" + path.read_bytes())
    study["source_content_sha256"] = source_digest.hexdigest()
    files += [path for path in ROBOT.rglob("*") if path.is_file()]
    files += [path for path in FIXTURES.iterdir() if path.is_file()]
    files += list((run / "inputs").glob("*.yaml"))
    study["source_hashes"] = {str(path): digest(path) for path in sorted(set(files))}
    write_json(run / "inputs/study.json", study)
    write_json(run / "start-hashes.json", study["source_hashes"])
    for path in code_files():
        target = run / "inputs/source-snapshot" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    print(f"PREPARED {run} cases={len(selected)} policy={args.policy}", flush=True)


def supervise(command: list[str], run: Path, name: str, timeout_s: int) -> None:
    """Wait for a finite child while reporting liveness at most every 30 seconds."""
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    log = run / f"{name}.log"
    started = time.monotonic()
    with log.open("w") as output:
        with subprocess.Popen(command, cwd=ROOT, env=env, stdout=output,
                              stderr=subprocess.STDOUT) as process:
            while True:
                remaining = timeout_s - (time.monotonic() - started)
                if remaining <= 0:
                    process.kill()
                    process.wait()
                    write_json(run / f"{name}-process.json", {"timeout_s": timeout_s,
                               "exit_code": process.returncode, "status": "predeclared_timeout"})
                    raise TimeoutError(f"{name} exceeded the declared {timeout_s}s budget")
                try:
                    code = process.wait(timeout=min(30, remaining))
                    break
                except subprocess.TimeoutExpired:
                    progress = sorted(run.glob("progress-*.json"))
                    print(f"RUNNING {name} pid={process.pid} log_mtime={log.stat().st_mtime} "
                          f"progress={[(p.name, p.stat().st_mtime) for p in progress]}", flush=True)
    write_json(run / f"{name}-process.json", {"timeout_s": timeout_s, "exit_code": code,
               "elapsed_s": time.monotonic() - started})
    if code:
        raise RuntimeError(f"{name} failed with exit {code}; read {log}")


def plan(run: Path, group: str) -> None:
    """Run the local trace/refinement policy with the new project import path."""
    supervise([str(GPU_PYTHON), str(SCRIPTS / "diagnose_arm_detours.py"), "plan",
               "--run", str(run), "--group", group], run, "planning", 240)


def analyze(run: Path) -> None:
    """Analyze saved native trajectories using the independent asset copy."""
    supervise([str(USD_PYTHON), str(SCRIPTS / "analyze_arm_motion_suite.py"),
               "--run", str(run)], run, "analysis", 120)


def main() -> None:
    """Dispatch a new run or an explicit phase of a prepared run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "plan", "analyze", "run"])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--group", choices=["left", "right", "both"], default="both")
    parser.add_argument("--case")
    parser.add_argument("--policy", choices=["trace", "refine", "reset_lm", "nearest"], default="trace")
    args = parser.parse_args()
    if args.action in ["prepare", "run"]:
        prepare(args)
    run = args.run.resolve()
    if args.action in ["plan", "run"]:
        plan(run, args.group)
    if args.action in ["analyze", "run"]:
        analyze(run)
    print(f"DONE {args.action} {run}", flush=True)


if __name__ == "__main__":
    main()
