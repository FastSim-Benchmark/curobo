# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Read-only LM/L-BFGS numerical probes; these do not certify motion quality."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from run_regression import FIXTURES, ROBOT, ROOT, code_files, digest, write_json
from curobo._src.curobolib.cuda_ops.optimization import LBFGScu
from curobo._src.optim.components.quasi_newton_buffers import QuasiNewtonBuffers
from curobo._src.optim.gradient.lbfgs_jit_helpers import jit_lbfgs_compute_step_direction
from curobo._src.optim.gradient.line_search_context import LineSearchContext
from curobo._src.optim.gradient.line_search_strategy import ApproxWolfeLineSearchStrategy
from curobo._src.optim.optimization_iteration_state import OptimizationIterationState
from curobo._src.solver.seed_ik.seed_ik_error_calculator import SeedIKErrorCalculator
from curobo._src.solver.seed_ik.seed_ik_solver import SeedIKSolver
from curobo._src.solver.seed_ik.seed_ik_solver_cfg import SeedIKSolverCfg
from curobo._src.solver.seed_ik.seed_ik_state import SeedIKState
from curobo._src.solver.seed_ik.seed_iteration_state_manager import SeedIterationStateManager
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import Pose
from curobo._src.types.tool_pose import GoalToolPose


def lm_rejection_probe(device: str) -> dict:
    """Exercise a rejection followed by a still-worse trial, with a normal control."""
    manager = SeedIterationStateManager(
        torch.full((1,), -10., device=device), torch.full((1,), 10., device=device),
        1e-3, 2., 1e-5, 1e10, .005, .05, 1.,
    )

    def state(q: float, cost: float) -> SeedIKState:
        return SeedIKState(
            joint_position=torch.tensor([[q]], device=device),
            error_norm=torch.tensor([cost], device=device),
            jTerror=torch.ones(1, 1, device=device),
            jacobian=torch.ones(1, 1, 1, device=device),
            lambda_damping=torch.ones(1, 1, 1, device=device),
            position_errors=torch.tensor([.1], device=device),
            orientation_errors=torch.tensor([.1], device=device),
        )

    pred = torch.ones(1, device=device)
    original = state(0., 1.)
    rejected = manager.update_iteration_state(original, state(1., 2.), pred, 1)
    follow = manager.update_iteration_state(rejected, state(.5, 1.5), pred, 1)
    control = manager.update_iteration_state(original, state(.1, .5), pred, 1)
    return dict(
        original_cost=1., rejected_trial_cost=2.,
        retained_q_after_rejection=rejected.joint_position.item(),
        stored_cost_after_rejection=rejected.error_norm.item(),
        next_trial_cost=1.5, next_trial_accepted=follow.improvement.item(),
        decreasing_control_accepted=control.improvement.item(),
    )


def limit_residual_probe(device: str) -> dict:
    """Compare the limit merit's derivative with the LM right-hand side."""
    calculator = SimpleNamespace(
        action_min=torch.tensor([-1.], device=device),
        action_max=torch.tensor([1.], device=device),
        config=SimpleNamespace(joint_limit_weight=1.),
    )
    q = torch.tensor([[1.1]], device=device, requires_grad=True)
    g, jacobian, merit = SeedIKErrorCalculator._compute_joint_limit_errors(calculator, q, 1)
    actual_gradient = torch.autograd.grad(merit.sum(), q)[0]
    return dict(q=q.item(), reported_cost=merit.item(),
                derivative_of_reported_cost=actual_gradient.item(),
                lm_jTerror=g.item(), jacobian=jacobian.item())


def lbfgs_curvature_probe(device: str) -> list[dict]:
    """Compare actual eager and fused recursions on positive/negative curvature."""
    rows = []
    for sign in [1., -1.]:
        for backend in ["torch", "cuda_shared", "cuda_global"]:
            if device == "cpu" and backend != "torch":
                continue
            buffers = QuasiNewtonBuffers(DeviceCfg(device=device), 7)
            buffers.resize(1, 14)
            q = torch.zeros(1, 14, device=device)
            q[:, 0] = 1.
            gradient = sign * q[:, None]
            if backend == "torch":
                buffers.update(q, gradient)
                step = jit_lbfgs_compute_step_direction(
                    buffers.alpha, buffers.rho, buffers.y, buffers.s,
                    gradient, 7, .01, True,
                ).view(1, 14)
            else:
                step = LBFGScu.apply(
                    buffers.step_q_buffer, buffers.rho, buffers.y, buffers.s,
                    q, gradient, buffers.x_0, buffers.grad_0, .01, True,
                    backend == "cuda_shared",
                )
            rows.append(dict(backend=backend, sTy=sign,
                             stored_rho=buffers.rho[-1].item(),
                             first_step=step[0, 0].item(),
                             gradient_dot_step=(gradient.flatten()*step.flatten()).sum().item()))
    return rows


def line_search_probe(device: str) -> list[dict]:
    """A steep descent direction needs a step below the fixed search grid."""
    rows = []

    def cost_grad(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return .5*x.square().sum(dim=-1, keepdim=True), x.clone()

    for fused in ([False, True] if device != "cpu" else [False]):
        context = LineSearchContext(
            DeviceCfg(device=device), [0., .1, .5, 1.], 1e-5, .9,
            1, 14, 1, 14, 1., False, torch.ones(14, device=device),
            fused, cost_grad, 10, 0., 0.,
        )
        strategy = ApproxWolfeLineSearchStrategy()
        strategy.update_num_problems(1, context)
        x = torch.zeros(1, 1, 14, device=device)
        x[..., 0] = 1.
        state = OptimizationIterationState(
            action=x.clone(), cost=torch.tensor([.5], device=device), gradient=x.clone(),
            exploration_action=x.clone(), exploration_gradient=x.clone(),
            exploration_cost=torch.tensor([.5], device=device), step_direction=-100*x,
            best_action=x.clone(), best_cost=torch.tensor([.5], device=device),
            best_iteration=torch.zeros(1, dtype=torch.int32, device=device),
            current_iteration=torch.zeros(1, dtype=torch.int32, device=device),
            converged=torch.zeros(1, dtype=torch.uint8, device=device),
        )
        result = strategy.search(state, context)
        rows.append(dict(backend="cuda" if fused else "torch", starting_cost=.5,
                         selected_cost=result.cost.item(), best_cost=result.best_cost.item(),
                         exploration_cost=result.exploration_cost.item(),
                         exploration_first_coordinate=result.exploration_action[0, 0, 0].item(),
                         cost_at_untried_alpha_001=0.))
    return rows


def pose_gradient_probe(solver: SeedIKSolver, start: torch.Tensor, goal: GoalToolPose) -> list[dict]:
    """Compare native pose gradients against float32 central differences at the start."""
    calculator = solver.error_calculator
    weight = calculator.pose_cost._weight
    original_weight = weight.clone()
    q = start.expand(solver._num_problems, -1).clone()
    rows = []
    try:
        for name, values in [("position", [1., 0.]), ("orientation", [0., 1.])]:
            weight.copy_(weight.new_tensor(values))
            base = calculator._compute_pose_errors(q, goal, solver._idxs_goal, q.shape[0])
            analytical = base[0][0].detach().clone()
            for h in [1e-3, 3e-4]:
                finite_difference = torch.zeros_like(analytical)
                for joint in range(solver.dof):
                    offset = torch.zeros_like(q)
                    offset[:, joint] = h
                    plus = calculator._compute_pose_errors(
                        q + offset, goal, solver._idxs_goal, q.shape[0],
                    )[-1][0].detach().clone()
                    minus = calculator._compute_pose_errors(
                        q - offset, goal, solver._idxs_goal, q.shape[0],
                    )[-1][0].detach().clone()
                    finite_difference[joint] = (plus - minus)/(2*h)
                norm = finite_difference.norm().clamp_min(1e-8)
                rows.append(dict(component=name, difference_step_rad=h,
                    analytical=analytical.tolist(), finite_difference=finite_difference.tolist(),
                    relative_error=((analytical-finite_difference).norm()/norm).item(),
                    gradient_norm_ratio=(analytical.norm()/norm).item()))
    finally:
        weight.copy_(original_weight)
    return rows


def trace_robot_lm(run: Path, case: dict, repair_cost: bool) -> dict:
    """Trace the native LM steps from identical seeds, without executing trajectories.

    The diagnostic intervention restores the accepted state's merit after rejection.
    It only affects this process and is not a planner change or feasibility claim.
    """
    group = case["group"]
    robot = yaml.safe_load((FIXTURES / f"robot-{group}.yaml").read_text())
    urdf = ROBOT / "assets/urdf/g2_crsB_op3_e.urdf"
    robot["robot_cfg"]["kinematics"].update(
        urdf_path=str(urdf), asset_root_path=str(urdf.parent),
    )
    robot_file = run / "inputs" / f"robot-{group}.yaml"
    if not robot_file.exists():
        robot_file.write_text(yaml.safe_dump(robot, sort_keys=False))
    iterations = 20 if group == "both" else 16
    solver = SeedIKSolver(SeedIKSolverCfg.create(
        robot=robot, num_seeds=128 if group == "both" else 32,
        max_iterations=iterations, inner_iterations=iterations,
        lambda_initial=1., rho_min=1e-5, local_seed_count=8,
        use_cuda_graph=False, tile_threads=64,
    ))
    start_map = dict(zip(case["joint_names"], case["start"]))
    start = torch.tensor([[start_map[n] for n in solver.joint_names]], device="cuda")
    goal = GoalToolPose.from_poses({
        name: Pose.from_matrix(torch.tensor(case["tool_goal_poses"][name], device="cuda").unsqueeze(0))
        for name in solver.tool_frames
    }, ordered_tool_frames=solver.tool_frames)
    rng = solver.act_sample_gen._int_gen.get_state().cpu().numpy().copy()
    seeds = solver._generate_seed_configs(1, start[:, None]).clone()
    solver._setup_batch_size(1, seeds.shape[1])
    gradients = pose_gradient_probe(solver, start, goal) if not repair_cost else []
    state = solver._compute_initial_iteration_state(seeds.flatten(0, 1), goal).clone()
    state.lambda_damping = torch.ones(seeds.shape[1], 1, 1, device="cuda")
    arrays = {"seeds": seeds.cpu().numpy(), "rng_before": rng}
    rows = []
    for index in range(iterations):
        true_old = solver._compute_pose_error_and_jacobian(state.joint_position, goal).error_norm.clone()
        next_state = solver._levenberg_marquardt_step_impl(state, goal).clone()
        true_new = solver._compute_pose_error_and_jacobian(next_state.joint_position, goal).error_norm.clone()
        if repair_cost:
            next_state.error_norm = torch.where(next_state.improvement,
                                                next_state.error_norm, state.error_norm)
        uphill = next_state.improvement & (true_new > true_old + 1e-6)
        mismatch = (next_state.error_norm - true_new).abs() > 1e-5
        rows.append(dict(iteration=index, accepted=int(next_state.improvement.sum()),
                         accepted_uphill=int(uphill.sum()), stale_cost=int(mismatch.sum()),
                         local_accepted_uphill=int(uphill[:8].sum()),
                         local_stale_cost=int(mismatch[:8].sum())))
        arrays[f"q_{index:02d}"] = next_state.joint_position.detach().cpu().numpy()
        arrays[f"true_cost_{index:02d}"] = true_new.detach().cpu().numpy()
        arrays[f"stored_cost_{index:02d}"] = next_state.error_norm.detach().cpu().numpy()
        arrays[f"accepted_{index:02d}"] = next_state.improvement.cpu().numpy()
        state = next_state
    success = solver._check_convergence(state, 1)
    distances = (state.joint_position - start).norm(dim=-1)
    label = "diagnostic_cost_repair" if repair_cost else "native"
    np.savez_compressed(run / f"{case['id']}_{label}.npz", **arrays)
    row = dict(case=case["id"], variant=label, mode="fresh_solver_eager_LM_only",
               pose_gradient_checks=gradients,
               iterations=rows, native_lm_success_count=int(success.sum()),
               local_lm_success_count=int(success[:8].sum()),
               local_joint_distances_rad=distances[:8].tolist(),
               local_position_error_m=state.position_errors[:8].tolist(),
               local_orientation_error_rad=state.orientation_errors[:8].tolist(),
               joint_names=solver.joint_names, tool_frames=solver.tool_frames,
               collision_validation=False, physical_execution=False,
               note="LM-only; joint distances are endpoints, not executed joint travel")
    solver.destroy()
    return row


def main(run: Path, robot_probes: bool) -> None:
    """Save inputs, source snapshots and bounded read-only numerical diagnostics."""
    import curobo

    run = run.resolve()
    if run.exists() or not run.is_relative_to(ROOT / "runs"):
        raise ValueError("Use a new runs/ directory")
    assert Path(curobo.__file__).resolve() == ROOT / "curobo/__init__.py"
    assert torch.cuda.is_available()
    (run / "inputs").mkdir(parents=True)
    source_paths = code_files()
    paths = source_paths + list(FIXTURES.glob("*"))
    if robot_probes:
        paths += [p for p in ROBOT.rglob("*") if p.is_file()]
    hashes = {str(p): digest(p) for p in paths if p.is_file()}
    write_json(run / "start-hashes.json", hashes)
    for path in source_paths:
        copy = run / "inputs/source-snapshot" / path.relative_to(ROOT)
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy.write_bytes(path.read_bytes())
    study = json.loads((FIXTURES / "study.json").read_text())
    selected = [c for c in study["cases"] if c["id"] in
                ["both_nearby_2", "both_roll_positive", "right_restore_2", "both_nearby_1"]]
    write_json(run / "inputs/cases.json", selected)
    results = dict(
        scope="Numerical diagnostics only; no trajectory or motion-quality acceptance",
        curobo_file=curobo.__file__, torch=torch.__version__, cuda_runtime=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        git_status=subprocess.check_output(["git", "status", "--short"], text=True),
        case_order=[c["id"] for c in selected] if robot_probes else [],
    )
    started = time.perf_counter()
    results["completed"] = False
    try:
        for device in ["cpu", "cuda"]:
            results[device] = dict(lm_rejection=lm_rejection_probe(device),
                                   joint_limit=limit_residual_probe(device),
                                   lbfgs=lbfgs_curvature_probe(device),
                                   line_search=line_search_probe(device))
            write_json(run / "results.json", results)
            print(f"{device}: numerical probes complete", flush=True)
        if robot_probes:
            results["robot_lm"] = []
            for case in selected:
                for repair_cost in [False, True]:
                    row = trace_robot_lm(run, case, repair_cost)
                    results["robot_lm"].append(row)
                    write_json(run / "results.json", results)
                    print(f"{case['id']} {row['variant']}: complete", flush=True)
        results["completed"] = True
    except Exception as error:
        results["error"] = repr(error)
        raise
    finally:
        results["elapsed_s"] = time.perf_counter() - started
        changed = [p for p, before in hashes.items() if not Path(p).exists() or digest(Path(p)) != before]
        added_sources = sorted(set(map(str, code_files())) - set(map(str, source_paths)))
        write_json(run / "end-validation.json", dict(
            unchanged=not changed and not added_sources, changed=changed, added_sources=added_sources,
            input_hashes={str(p): digest(p) for p in (run / "inputs").glob("*") if p.is_file()},
        ))
        write_json(run / "results.json", results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--robot-probes", action="store_true")
    arguments = parser.parse_args()
    main(arguments.run, arguments.robot_probes)
