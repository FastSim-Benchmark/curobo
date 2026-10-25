"""Finite pose-only quality/latency comparisons from frozen TCP requests."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
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
    if args.cases:
        selected = [c for c in selected if c['id'] in args.cases]
    if args.extra_cases:
        selected += [c for c in json.loads(args.extra_cases.read_text()) if c['group'] in args.groups]
    study.update(cases=selected, planned_case_count=len(selected), source=str(ROOT),
                 source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                 urdf=str(ROBOT / 'assets/urdf/g2_crsB_op3_e.urdf'), robot_root=str(ROBOT),
                 modes=args.policies, requested_slot_count=len(selected)*len(args.policies)*args.repeats,
                 benchmark={'independent': args.independent, 'repeats': args.repeats,
                            'max_attempts': args.max_attempts, 'timeout_s': 10, 'policies': args.policies,
                            'shuffle_repeats': args.shuffle_repeats,
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

    def construct(group: str, policy: str) -> MotionPlanner:
        robot = yaml.safe_load(Path(study['robot_configs'][group]).read_text())
        native = KinematicsCfg.from_robot_yaml_file(robot)
        native.kinematics_config.joint_limits.velocity[0].fill_(-0.4)
        native.kinematics_config.joint_limits.velocity[1].fill_(0.4)
        cfg = MotionPlannerCfg.create(robot={'robot_cfg': {'kinematics': native}},
            scene_model=None, self_collision_check=True, use_cuda_graph=True,
            random_seed=5, interpolation_dt=1/60, num_ik_seeds=32, num_trajopt_seeds=4)
        ik = cfg.ik_solver_config
        ik.prefer_smooth_motion = policy.startswith('quality')
        ik.quality_refinement_iters = int(policy.removeprefix('quality')) if ik.prefer_smooth_motion else 40
        ik.detailed_failure_diagnostics = policy == 'detailed'
        cfg.trajopt_solver_config.prefer_smooth_motion = ik.prefer_smooth_motion
        if policy.startswith('refine'):
            ik.exit_early = False
            cfg.ik_solver_config.optimizer_configs[-1].num_iters = int(policy.removeprefix('refine'))
        planner = MotionPlanner(cfg)
        planner.warmup(enable_graph=True)
        planner.reset_seed()
        planner.ik_solver.seed_ik_solver.act_sample_gen.reset()
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
                            planner = construct(group, policy)
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
                        originals = []
                        candidate_records = []
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
                        for obj, method, key in [(planner.ik_solver, 'solve_pose', 'ik'),
                                (planner.trajopt_solver, 'solve_pose', 'trajopt'),
                                (planner.graph_planner, 'find_path', 'prm')]:
                            if obj is None:
                                continue
                            original = getattr(obj, method)
                            def call(*a, _fn=original, _key=key, **kw):
                                counts[_key] += 1
                                return _fn(*a, **kw)
                            originals.append((obj, method, original))
                            setattr(obj, method, call)
                        key = f'{case["id"]}_{policy}_r{repeat}'
                        folder = run / 'results' / key
                        folder.mkdir(parents=True)
                        report = dict(id=key, case=case['id'], group=group, side=case['side'], seed=5,
                            mode=policy, policy=policy, repeat=repeat, planning_called=True, success=False,
                            sequence_index=sequence_index, stratum=case.get('stratum', 'canonical'),
                            tool_frames=list(planner.tool_frames), active_joint_names=names,
                            physical_execution=False, curobo_file=curobo.__file__,
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
                                exception_type=type(error).__name__, reason=str(error))
                        finally:
                            for obj, method, original in originals:
                                setattr(obj, method, original)
                        if candidate_records:
                            np.savez_compressed(folder/'ik-candidates.npz', **{
                                f'{i}_{k}': v.detach().cpu().numpy() for i, item in enumerate(candidate_records)
                                for k, v in item.items()})
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
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--independent', action='store_true')
    parser.add_argument('--trace-candidates', action='store_true')
    parser.add_argument('--max-attempts', type=int, default=3)
    parser.add_argument('--shuffle-repeats', action='store_true')
    main(parser.parse_args())
