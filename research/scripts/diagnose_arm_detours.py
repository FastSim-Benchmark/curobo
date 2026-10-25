"""Trace native IK selection and test bounded, local ranking experiments.

This diagnostic wraps live solver instances; it never edits the dependency.
The nearest policy uses actual joint displacement, with native feasibility and
pose convergence checked before ranking. No joint angles are wrapped modulo 2pi.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from study_empty_arm_motion import sha, write_json


def prepare(args: argparse.Namespace) -> None:
    run = args.run.resolve()
    (run / "inputs").mkdir(parents=True, exist_ok=False)
    study = json.loads((args.baseline / "inputs/study.json").read_text())
    study["cases"] = [c for c in study["cases"] if c["group"] == args.group
                      and (not args.case or c["id"] == args.case)]
    if not study["cases"]:
        raise ValueError("No cases selected")
    study["requested_slot_count"] = len(study["cases"])*len(study["modes"])
    for group, original in study["robot_configs"].items():
        target = run / "inputs" / Path(original).name
        target.write_bytes(Path(original).read_bytes())
        study["robot_configs"][group] = str(target)
    if args.lm_state:
        frozen_state = run / "inputs/lm-generator-state.npz"
        frozen_state.write_bytes(args.lm_state.read_bytes())
    study["diagnostic"] = {"policy": args.policy, "advance_lm": args.advance_lm,
                           "lm_state": str(frozen_state) if args.lm_state else None,
                           "purpose": "Controlled local instrumentation; dependency unchanged",
                           "per_request_timeout_s": 60, "per_request_max_attempts": 3}
    files = [Path(p) for p in study["source_hashes"] if Path(p).exists()]
    files += [Path(__file__).resolve(), *map(Path, study["robot_configs"].values())]
    if args.lm_state:
        files.append(frozen_state)
    study["source_hashes"] = {str(p): sha(p) for p in files}
    write_json(run / "inputs/study.json", study)
    snapshot = run / "inputs/source-snapshot"
    snapshot.mkdir()
    for p in files:
        if p.suffix == ".py":
            (snapshot / p.name).write_bytes(p.read_bytes())
    write_json(run / "start-hashes.json", study["source_hashes"])
    print("PREPARED", run, len(study["cases"]), args.policy, flush=True)


def plan(run: Path, group: str) -> None:
    import torch
    import curobo.motion_planner as public
    from study_arm_motion_suite import plan as native_suite_plan

    study = json.loads((run / "inputs/study.json").read_text())
    policy = study["diagnostic"]["policy"]
    native_class = public.MotionPlanner
    calls = []
    trace_dir = run / "traces"
    trace_dir.mkdir(exist_ok=False)

    def construct(config):
        planner = native_class(config)
        ik = planner.ik_solver
        lm = ik.seed_ik_solver
        native_select = lm._select_top_solutions
        native_get = ik._get_result
        native_solve = ik.solve_pose
        native_plan = planner.plan_pose
        context = {"request": -1, "case": None, "current": None, "ik_call": None}
        if policy == "refine":
            # Let the native optimizer refine promising seeds even when another
            # distant seed has already passed the endpoint tolerance.
            ik.config.exit_early = False
        if study["diagnostic"].get("lm_state"):
            state = np.load(study["diagnostic"]["lm_state"])["state"]
            lm.act_sample_gen._int_gen.set_state(torch.from_numpy(state.copy()))
        if study["diagnostic"]["advance_lm"]:
            lm.act_sample_gen.sequencer.fast_forward(study["diagnostic"]["advance_lm"])

        def array(value):
            return value.detach().cpu().numpy().copy()

        def select(solutions, successes, position_errors, orientation_errors,
                   start_joint_position, batch_size, num_seeds, return_seeds):
            record = context["ik_call"]
            record["lm_num_seeds"] = num_seeds
            record["lm_start_distance_weight"] = lm.config.start_cspace_dist_weight
            payload = dict(solution=array(solutions), success=array(successes),
                           position_error=array(position_errors), rotation_error=array(orientation_errors),
                           start=array(start_joint_position))
            result = native_select(solutions, successes, position_errors, orientation_errors,
                                   start_joint_position, batch_size, num_seeds, return_seeds)
            payload["selected_solution"] = array(result[1])
            np.savez_compressed(trace_dir / (record["id"]+"-lm.npz"), **payload)
            return result

        def get_result(opt_result, metrics, num_seeds, return_seeds):
            result = native_get(opt_result, metrics, num_seeds, return_seeds)
            all_result = native_get(opt_result, metrics, num_seeds, num_seeds)
            record = context["ik_call"]
            distance = torch.linalg.vector_norm(all_result.solution-context["current"][:, None], dim=-1)
            native_indices = result.debug_info["seed_idx"].clone()
            payload = dict(solution=array(all_result.solution), success=array(all_result.success),
                           position_error=array(all_result.position_error), rotation_error=array(all_result.rotation_error),
                           distance=array(distance), native_selected=array(result.solution),
                           native_indices=array(native_indices), all_indices=array(all_result.debug_info["seed_idx"]))
            record["convergence_names"] = list(metrics.convergence.names)
            record["native_selected_distance"] = array(torch.linalg.vector_norm(result.solution-context["current"][:, None], dim=-1)).tolist()
            good = distance[all_result.success]
            record["nearest_feasible_distance"] = float(good.min()) if good.numel() else None
            if policy in ("nearest", "nearest_reset_lm"):
                # Stable lexicographic selection: native success first, then distance.
                order = torch.argsort(distance, dim=-1, stable=True)
                valid = torch.gather(all_result.success, 1, order)
                order = torch.gather(order, 1, torch.argsort(valid.to(torch.int32), dim=-1,
                                                          descending=True, stable=True))
                pick = order[0, :return_seeds]
                for name in ("solution", "success", "feasible", "position_error", "rotation_error", "goalset_index"):
                    value = getattr(all_result, name, None)
                    if value is not None:
                        setattr(result, name, value[:, pick].clone())
                result.js_solution = all_result.js_solution[:, pick].clone()
                result.solution_state = all_result.solution_state[:, pick]
                result.debug_info["seed_idx"] = all_result.debug_info["seed_idx"][:, pick].clone()
                payload["modified_selected"] = array(result.solution)
            record["returned_distance"] = array(torch.linalg.vector_norm(result.solution-context["current"][:, None], dim=-1)).tolist()
            record["feasible_count"] = int(all_result.success.sum())
            np.savez_compressed(trace_dir / (record["id"]+f"-ik-{record.get('metrics_calls', 0)}.npz"), **payload)
            record["metrics_calls"] = record.get("metrics_calls", 0)+1
            return result

        def solve(*args, **kwargs):
            current = kwargs["current_state"]
            context["current"] = current.position.clone()
            record = {"id": f"{len(calls):03d}-{context['case']}", "case": context["case"],
                      "lm_offset_before": lm.act_sample_gen.sequencer.qmc.num_generated,
                      "ik_num_seeds": ik.config.num_seeds}
            state = array(lm.act_sample_gen._int_gen.get_state())
            record["lm_rng_before_sha256"] = hashlib.sha256(state.tobytes()).hexdigest()
            np.savez_compressed(trace_dir / (record["id"]+"-rng-before.npz"), state=state)
            calls.append(record)
            context["ik_call"] = record
            result = native_solve(*args, **kwargs)
            record["lm_offset_after"] = lm.act_sample_gen.sequencer.qmc.num_generated
            record["lm_rng_after_sha256"] = hashlib.sha256(array(lm.act_sample_gen._int_gen.get_state()).tobytes()).hexdigest()
            write_json(run / "trace-index.json", calls)
            return result

        def pose(*args, **kwargs):
            context["request"] += 1
            context["case"] = study["cases"][context["request"]]["id"]
            if policy in ("reset_lm", "nearest_reset_lm"):
                # Reset only the LM sampler. Keep CUDA graphs and optimizer caches.
                lm.act_sample_gen.reset()
            return native_plan(*args, **kwargs)

        lm._select_top_solutions = select
        ik._get_result = get_result
        ik.solve_pose = solve
        planner.plan_pose = pose
        return planner

    public.MotionPlanner = construct
    native_suite_plan(run, group)
    expected = study["source_hashes"]
    observed = {p: sha(Path(p)) for p in expected}
    write_json(run / "end-validation.json", {"input_and_code_unchanged": expected == observed,
               "hashes": observed, "ik_calls": len(calls), "native_git_status": subprocess.check_output(
                   ["git", "status", "--porcelain"], cwd=study["source"], text=True)})
    if expected != observed:
        raise RuntimeError("Inputs changed during run")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "plan"])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--group", choices=["left", "right", "both"], required=True)
    parser.add_argument("--case")
    parser.add_argument("--policy", choices=["trace", "reset_lm", "nearest", "nearest_reset_lm", "refine"], default="trace")
    parser.add_argument("--advance-lm", type=int, default=0)
    parser.add_argument("--lm-state", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args)
    else:
        plan(args.run.resolve(), args.group)
