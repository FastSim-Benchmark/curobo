"""Finite pose-only quality/latency comparisons from frozen TCP requests."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
import yaml

from run_regression import ROOT, ROBOT, FIXTURES, code_files, digest, write_json


def main(args: argparse.Namespace) -> None:
    """Run paired planner policies, preserving complete group order and evidence."""
    import torch
    import curobo
    from curobo.kinematics import KinematicsCfg
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    from curobo.types import GoalToolPose, JointState, Pose

    run = args.run.resolve()
    if run.exists() or not run.is_relative_to(ROOT / 'runs'):
        raise ValueError('Use a new directory under this project runs/')
    assert Path(curobo.__file__).resolve() == ROOT / 'curobo/__init__.py'
    study = json.loads((FIXTURES / 'study.json').read_text())
    selected = [c for c in study['cases'] if c['group'] in args.groups]
    if args.only_extra_cases:
        if args.extra_cases is None:
            raise ValueError('--only-extra-cases requires --extra-cases')
        selected = []
    if args.cases:
        selected = [c for c in selected if c['id'] in args.cases]
    if args.extra_cases:
        selected += [c for c in json.loads(args.extra_cases.read_text()) if c['group'] in args.groups]
    if not args.independent:
        for group in args.groups:
            present = [c.get('scene') is not None for c in selected if c['group'] == group]
            if any(present) and not all(present):
                raise ValueError('Mixed empty/obstacle scenes require --independent')
    study.update(cases=selected, planned_case_count=len(selected), source=str(ROOT),
                 source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                 urdf=str(ROBOT / 'assets/urdf/g2_crsB_op3_e.urdf'), robot_root=str(ROBOT),
                 modes=args.policies, requested_slot_count=len(selected)*len(args.policies)*args.repeats,
                 benchmark={'independent': args.independent, 'repeats': args.repeats,
                            'max_attempts': args.max_attempts, 'timeout_s': 10, 'policies': args.policies,
                            'shuffle_repeats': args.shuffle_repeats,
                            'per_request_scene_updates': not args.independent and any(c.get('scene') is not None for c in selected),
                            'sequence': [c['id'] for c in selected], 'warmup': True},
                 runtime={'torch': torch.__version__, 'cuda': torch.version.cuda,
                          'gpu': torch.cuda.get_device_name(0)})
    (run / 'inputs').mkdir(parents=True)
    for group in args.groups:
        robot = yaml.safe_load((FIXTURES / f'robot-{group}.yaml').read_text())
        kin = robot['robot_cfg']['kinematics']
        kin.update(urdf_path=study['urdf'], asset_root_path=str(Path(study['urdf']).parent))
        path = run / 'inputs' / f'robot-{group}.yaml'
        path.write_text(yaml.safe_dump(robot, sort_keys=False))
        study['robot_configs'][group] = str(path)
    paths = code_files() + [p for p in ROBOT.rglob('*') if p.is_file()] + list(FIXTURES.glob('*'))
    paths += list((run / 'inputs').glob('*.yaml'))
    acceptance = ROOT/'research/fixtures/motion_quality_acceptance_v1.json'
    if acceptance.exists():
        paths.append(acceptance)
    if args.extra_cases:
        paths.append(args.extra_cases.resolve())
    hashes = {str(p): digest(p) for p in paths if p.is_file()}
    study['source_hashes'] = hashes
    study['source_content_sha256'] = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    write_json(run / 'inputs/study.json', study)
    study_hash = digest(run / 'inputs/study.json')
    write_json(run / 'start-hashes.json', hashes)
    snapshot = run / 'inputs/source-snapshot'
    for path in code_files():
        target = snapshot / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())

    def construct(group: str, policy: str, scene: dict | None = None) -> MotionPlanner:
        initialization_started = time.perf_counter()
        if scene is None and not args.independent:
            scene = next((c.get('scene') for c in selected if c['group'] == group), None)
        robot = yaml.safe_load(Path(study['robot_configs'][group]).read_text())
        native = KinematicsCfg.from_robot_yaml_file(robot)
        native.kinematics_config.joint_limits.velocity[0].fill_(-0.4)
        native.kinematics_config.joint_limits.velocity[1].fill_(0.4)
        cfg = MotionPlannerCfg.create(robot={'robot_cfg': {'kinematics': native}},
            scene_model=scene, self_collision_check=True, use_cuda_graph=True,
            random_seed=5, interpolation_dt=1/60, num_ik_seeds=32, num_trajopt_seeds=4)
        ik = cfg.ik_solver_config
        ik.prefer_smooth_motion = policy.startswith(('quality', 'previous', 'balanced'))
        ik.quality_refinement_iters = int(policy.removeprefix('quality').removeprefix('previous').removeprefix('balanced')) if ik.prefer_smooth_motion else 40
        ik._posture_motion_enabled = policy.startswith('quality')
        ik.detailed_failure_diagnostics = policy == 'detailed'
        cfg.trajopt_solver_config.prefer_smooth_motion = ik.prefer_smooth_motion
        cfg.trajopt_solver_config._posture_motion_enabled = ik._posture_motion_enabled
        if policy.startswith('refine'):
            ik.exit_early = False
            cfg.ik_solver_config.optimizer_configs[-1].num_iters = int(policy.removeprefix('refine'))
        planner = MotionPlanner(cfg)
        planner.trajopt_solver._balanced_motion_enabled = not policy.startswith('previous')
        warmup_started = time.perf_counter()
        planner.warmup(enable_graph=True)
        planner.reset_seed()
        planner.ik_solver.seed_ik_solver.act_sample_gen.reset()
        torch.cuda.synchronize()
        planner._research_warmup_s = time.perf_counter()-warmup_started
        planner._research_initialization_s = time.perf_counter()-initialization_started
        return planner

    rows = []
    for group in args.groups:
        cases = [c for c in selected if c['group'] == group]
        order = [(repeat, policy) for repeat in range(args.repeats)
                 for policy in (args.policies if repeat % 2 == 0 else args.policies[::-1])]
        for round_index, policy in order:
            ordered_cases = list(cases)
            if args.shuffle_repeats and round_index:
                np.random.default_rng(1025 + round_index).shuffle(ordered_cases)
            planner = None
            try:
                for repeat in [round_index]:
                    if not args.independent:
                        planner = construct(group, policy)
                    for sequence_index, case in enumerate(ordered_cases):
                        if args.independent:
                            planner = construct(group, policy, case.get('scene'))
                        scene_update_s = 0.0
                        if not args.independent and case.get('scene') is not None:
                            from curobo._src.geom.types import SceneCfg
                            scene_started = time.perf_counter()
                            planner.update_world(SceneCfg.create(case['scene']))
                            torch.cuda.synchronize()
                            scene_update_s = time.perf_counter()-scene_started
                        names = list(planner.joint_names)
                        current_map = dict(zip(case['joint_names'], case['start']))
                        current = JointState.from_position(torch.tensor([[current_map[n] for n in names]],
                            device='cuda', dtype=torch.float32), joint_names=names)
                        poses = {t: Pose.from_matrix(torch.tensor(case['tool_goal_poses'][t],
                            device='cuda', dtype=torch.float32).unsqueeze(0)) for t in planner.tool_frames}
                        goal = GoalToolPose.from_poses(poses, ordered_tool_frames=planner.tool_frames)
                        sampler = planner.ik_solver.seed_ik_solver.act_sample_gen
                        if case.get('lm_advance') is not None:
                            sampler.reset()
                            for _ in range(case['lm_advance']):
                                sampler.get_samples(planner.ik_solver.seed_ik_solver.config.num_seeds-1, bounded=True)
                        rng_before = sampler._int_gen.get_state().cpu().numpy().copy()
                        counts = {'ik': 0, 'trajopt': 0, 'prm': 0}
                        quality_events = []
                        originals = []
                        candidate_records = []
                        trajectory_records = []
                        if args.trace_candidates:
                            native_get = planner.ik_solver._get_result
                            def capture(actions, metrics, num_seeds, return_seeds, *extra):
                                answer = native_get(actions, metrics, num_seeds, return_seeds, *extra)
                                candidate_records.append(dict(solution=answer.solution.clone(),
                                    success=answer.success.clone(), position_error=answer.position_error.clone(),
                                    rotation_error=answer.rotation_error.clone(), actions=actions.clone()))
                                return answer
                            originals.append((planner.ik_solver, '_get_result', native_get))
                            planner.ik_solver._get_result = capture
                            native_rank = planner.trajopt_solver._get_best_result
                            def capture_trajectories(pool, count):
                                trajectory_records.append(dict(
                                    stage='ranked_result',
                                    quality=planner.trajopt_solver.config.prefer_smooth_motion,
                                    success=pool.success.clone(), position=pool.interpolated_trajectory.position.clone(),
                                    last_step=pool.interpolated_last_tstep.clone(),
                                    joint_names=pool.interpolated_trajectory.joint_names))
                                return native_rank(pool, count)
                            originals.append((planner.trajopt_solver, '_get_best_result', native_rank))
                            planner.trajopt_solver._get_best_result = capture_trajectories
                            native_validate = planner.trajopt_solver._validate_retimed_initial_seed
                            def capture_initial(*a, _fn=native_validate, **kw):
                                pool = _fn(*a, **kw)
                                trajectory_records.append(dict(stage='retimed_seed',
                                    quality=planner.trajopt_solver.config.prefer_smooth_motion,
                                    success=pool.success.clone(), position=pool.interpolated_trajectory.position.clone(),
                                    last_step=pool.interpolated_last_tstep.clone(),
                                    joint_names=pool.interpolated_trajectory.joint_names))
                                return pool
                            originals.append((planner.trajopt_solver, '_validate_retimed_initial_seed', native_validate))
                            planner.trajopt_solver._validate_retimed_initial_seed = capture_initial
                        for obj, method, key in [(planner.ik_solver, 'solve_pose', 'ik'),
                                (planner.trajopt_solver, 'solve_pose', 'trajopt'),
                                (planner.graph_planner, 'find_path', 'prm')]:
                            if obj is None:
                                continue
                            original = getattr(obj, method)
                            def call(*a, _fn=original, _key=key, **kw):
                                counts[_key] += 1
                                answer = _fn(*a, **kw)
                                if _key == 'ik' and answer is not None:
                                    info = (getattr(answer, 'debug_info', None) or {}).get('motion_quality')
                                    if info is not None:
                                        quality_events.append({k:info[k] for k in
                                            ['refined', 'extended_quality_refinement',
                                             'reused_global_seeds'] if k in info})
                                return answer
                            originals.append((obj, method, original))
                            setattr(obj, method, call)
                        key = f'{case["id"]}_{policy}_r{repeat}'
                        folder = run / 'results' / key
                        folder.mkdir(parents=True)
                        report = dict(id=key, case=case['id'], group=group, side=case['side'], seed=5,
                            mode=policy, policy=policy, repeat=repeat, planning_called=True, success=False,
                            sequence_index=sequence_index, stratum=case.get('stratum', 'canonical'),
                            tool_frames=list(planner.tool_frames), active_joint_names=names,
                            quality_elbow_frames=(planner._motion_quality.elbow_frames
                                if getattr(planner, '_motion_quality', None) is not None else
                                (planner.ik_solver._quality_kinematics.tool_frames[len(planner.tool_frames):]
                                 if planner.ik_solver._quality_kinematics is not None else [])),
                            physical_execution=False, curobo_file=curobo.__file__,
                            ik_quality_events=quality_events,
                            scene=case.get('scene'),
                            scene_update_s=scene_update_s,
                            construction_warmup_s=planner._research_initialization_s,
                            warmup_s=planner._research_warmup_s,
                            new_planner_for_request=args.independent or sequence_index == 0,
                            constraints=dict(self_collision=True, position_m=0.005, orientation_rad=0.05,
                                             velocity_rad_s=0.4), calls=counts,
                            lm_rng_before_sha256=hashlib.sha256(rng_before.tobytes()).hexdigest(),
                            lm_buffer_sha256=hashlib.sha256(sampler._sample_buffer.detach().cpu().numpy().tobytes()).hexdigest(),
                            lm_qmc_generated=int(sampler.sequencer.qmc.num_generated))
                        np.savez_compressed(folder/'rng-before.npz', state=rng_before)
                        torch.cuda.synchronize()
                        started = time.perf_counter()
                        try:
                            result = planner.plan_pose(goal, current, max_attempts=args.max_attempts, timeout_s=10)
                            torch.cuda.synchronize()
                            report['elapsed_s'] = time.perf_counter()-started
                            report['success'] = result is not None and bool(result.success.any())
                            report['status'] = 'success' if report['success'] else 'no_solution'
                            if report['success']:
                                balance = (result.debug_info or {}).get('motion_quality_balance')
                                if balance is not None:
                                    report['motion_quality_balance'] = {
                                        k: v.detach().cpu().tolist() for k, v in balance.items()}
                                guard = (result.debug_info or {}).get('quality_guard')
                                if guard:
                                    def serial(value):
                                        if isinstance(value, torch.Tensor):
                                            return value.detach().cpu().tolist()
                                        if isinstance(value, dict):
                                            return {k:serial(v) for k,v in value.items()}
                                        return value
                                    report['quality_guard'] = serial({k:v for k,v in guard.items()
                                        if k not in ['baseline_trajectory', 'baseline_joint_names', 'baseline_last_step']})
                                    if 'baseline_trajectory' in guard:
                                        np.savez_compressed(folder/'baseline-trajectory.npz',
                                            position=guard['baseline_trajectory'].detach().cpu().numpy(),
                                            joint_names=np.array(guard['baseline_joint_names']),
                                            last_step=guard['baseline_last_step'].detach().cpu().numpy(), dt=np.array(1/60))
                                trajectory = result.get_interpolated_plan()
                                native_names = list(trajectory.joint_names)
                                q = trajectory.position.detach().cpu().numpy().reshape(-1, len(native_names))
                                locks = yaml.safe_load(Path(study['robot_configs'][group]).read_text())['robot_cfg']['kinematics']['lock_joints']
                                assert np.isfinite(q).all()
                                for i, n in enumerate(native_names):
                                    if n not in names:
                                        assert np.allclose(q[:, i], locks[n], atol=1e-6, rtol=0)
                                np.savez_compressed(folder/'trajectory.npz', position=q,
                                    joint_names=np.array(native_names), dt=np.array(1/60))
                                report.update(samples=len(q), duration_s=(len(q)-1)/60,
                                    locked_joint_values_verified=True,
                                    endpoint_position_error_m=float(result.position_error.max()),
                                    endpoint_orientation_error_rad=float(result.rotation_error.max()))
                        except Exception as error:
                            report.update(elapsed_s=time.perf_counter()-started,
                                status='timeout' if isinstance(error, TimeoutError) else 'exception',
                                exception_type=type(error).__name__, reason=str(error),
                                traceback=traceback.format_exc())
                            write_json(folder/'result.json', report)
                            if isinstance(error, torch.AcceleratorError):
                                # A poisoned CUDA context cannot safely serialize GPU traces.
                                raise
                        finally:
                            for obj, method, original in originals:
                                setattr(obj, method, original)
                        if candidate_records:
                            np.savez_compressed(folder/'ik-candidates.npz', **{
                                f'{i}_{k}': v.detach().cpu().numpy() for i, item in enumerate(candidate_records)
                                for k, v in item.items()})
                        if trajectory_records:
                            np.savez_compressed(folder/'trajectory-candidates.npz', **{
                                f'{i}_{k}': v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.array(v)
                                for i,item in enumerate(trajectory_records) for k,v in item.items()})
                        report['lm_rng_after_sha256'] = hashlib.sha256(sampler._int_gen.get_state().cpu().numpy().tobytes()).hexdigest()
                        write_json(folder/'result.json', report)
                        rows.append(report)
                        write_json(run/'progress.json', dict(completed=len(rows), last=key, updated=time.time()))
                        print(f'DONE {key} {report["status"]} {1000*report["elapsed_s"]:.2f}ms {counts}', flush=True)
                        if args.independent:
                            planner.destroy()
                            planner = None
                    if planner is not None:
                        planner.destroy()
                        planner = None
            finally:
                if planner is not None:
                    planner.destroy()
    observed = {p: digest(Path(p)) for p in hashes}
    unchanged = hashes == observed and digest(run / 'inputs/study.json') == study_hash
    write_json(run/'end-validation.json', dict(input_and_code_unchanged=unchanged,
        study_sha256=study_hash, hashes=observed))
    assert unchanged
    write_json(run/'requests.json', rows)
    print(f'COMPLETE {run} requests={len(rows)}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--groups', nargs='+', default=['right', 'left', 'both'])
    parser.add_argument('--policies', nargs='+', default=['legacy', 'quality40'])
    parser.add_argument('--cases', nargs='+')
    parser.add_argument('--extra-cases', type=Path)
    parser.add_argument('--only-extra-cases', action='store_true')
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--independent', action='store_true')
    parser.add_argument('--trace-candidates', action='store_true')
    parser.add_argument('--max-attempts', type=int, default=3)
    parser.add_argument('--shuffle-repeats', action='store_true')
    main(parser.parse_args())
