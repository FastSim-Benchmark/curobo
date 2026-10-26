"""Diagnostic-only CPU contracts for isolated contact-continuation candidates.

Set CONTACT_AUDIT_SOURCE_ROOT to the unmodified cuRobo checkout and
CONTACT_AUDIT_CANDIDATE_SOURCES to an isolated directory containing the candidate
contact_continuation.py and axis_seeds.py. Only their function ASTs are executed.
Native planners, collision queries, simulator SDKs, and GPU kernels are not run.
The planar scene, IK, deadline, contact scope, and final native results are
explicit test substitutes. These tests establish control flow and arithmetic,
not physical feasibility or U15 runtime causality.
"""

from __future__ import annotations

import ast
import os
from contextlib import contextmanager
from pathlib import Path
from types import MethodType
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation, Slerp

from curobo._src.motion import axis_seeds as source_axis_module
from curobo._src.motion.motion_goalset import goal_group
from curobo._src.state.state_joint import JointState
from curobo._src.types.tool_pose import GoalToolPose, ToolPose

SOURCE_ROOT = Path(
    os.environ.get(
        "CONTACT_AUDIT_SOURCE_ROOT", str(Path(source_axis_module.__file__).resolve().parents[3])
    )
)
CANDIDATE_ROOT = Path(
    os.environ.get("CONTACT_AUDIT_CANDIDATE_SOURCES", str(SOURCE_ROOT / "curobo/_src/motion"))
)


def load_functions(path, names, namespace, class_name=None):
    tree = ast.parse(path.read_text(), filename=str(path))
    nodes = tree.body
    if class_name is not None:
        nodes = next(
            node for node in nodes if isinstance(node, ast.ClassDef) and node.name == class_name
        ).body
    selected = [node for node in nodes if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in selected} == set(names)
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            *selected,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def fail(message):
    raise ValueError(message)


def harness(descent=0.015, gap=0.0, tolerance=1e-5, failure=None):
    names = ["z_translation", "held_joint"]
    physical = torch.tensor([[0.0, 0.0], [0.2, 0.0]])
    active = physical.clone()
    endpoint_z = 0.05 + gap
    current = JointState.from_position(torch.tensor([[endpoint_z + descent, 0.0]]), names)
    endpoint = JointState.from_position(torch.tensor([[endpoint_z, 0.0]]), names)
    # A real proposal obeys its native local limits. At an exact 20 mm request,
    # independently rounded float32 Cartesian values can differ by one ULP.
    # Bound the proposal substitute, while retaining the authored Cartesian goal.
    endpoint.position[0, 0] = torch.maximum(
        endpoint.position[0, 0], current.position[0, 0] - torch.tensor(0.02)
    )
    target = GoalToolPose(
        ["tool"],
        torch.tensor([[[[[0.0, 0.0, endpoint_z]]]]]),
        torch.tensor([[[[[1.0, 0.0, 0.0, 0.0]]]]]),
    )
    flags = NS(
        scope=False,
        entered=0,
        closed=False,
        prefix=[],
        endpoint_metrics=0,
        trajectories=0,
        proposal_count=0,
        deadline_checks=0,
    )

    def fk(state):
        positions = torch.zeros(1, 1, 1, 3)
        positions[..., 2] = state.position[0, 0]
        return NS(
            tool_poses=ToolPose(["tool"], positions, torch.tensor([[[[1.0, 0.0, 0.0, 0.0]]]]))
        )

    def update(*, position):
        assert bool((position[0] >= physical[0]).all())
        assert bool((position[1] <= physical[1]).all())
        active.copy_(position)

    def budget():
        flags.deadline_checks += 1
        if failure == "prefix_deadline" and len(flags.prefix) == 4:
            raise TimeoutError("prefix deadline")
        if failure == "scoped_deadline" and flags.scope:
            raise TimeoutError("scoped deadline")

    cost_namespace = {"torch": torch, "log_and_raise": fail}
    load_functions(
        SOURCE_ROOT / "curobo/_src/collision/contact_separation.py",
        ["captured_state_cost"],
        cost_namespace,
        "ContactSeparation",
    )
    load_functions(
        SOURCE_ROOT / "curobo/_src/collision/contact_approach.py",
        ["cost"],
        cost_namespace,
        "ContactApproach",
    )
    sphere = torch.tensor([[0.0, 0.0, endpoint.position[0, 0].item(), 0.05]])
    separation = NS(
        indices=torch.tensor([0]),
        initial_spheres=sphere,
        initial_clearance=sphere[:, 2] - sphere[:, 3],
        declaration=NS(numerical_tolerance=tolerance),
        output_mask=torch.ones(1),
    )
    separation.clearance = lambda spheres: spheres[..., 2] - spheres[..., 3]
    separation.geometry_cost = lambda spheres: torch.zeros(spheres.shape[:2])
    separation.captured_state_cost = MethodType(cost_namespace["captured_state_cost"], separation)
    contact = NS(declaration=NS(terminal_only=True), separation=separation, landing=None)
    contact.cost = MethodType(cost_namespace["cost"], contact)

    def candidates(*args):
        flags.proposal_count += 1
        try:
            assert not flags.scope
            assert (active[1, 0] - current.position[0, 0]).item() == pytest.approx(0.02)
            if failure == "proposal_exception":
                raise RuntimeError("proposal exception")
            if failure != "proposal_none":
                yield endpoint
        finally:
            flags.closed = True
            if failure == "iterator_close_exception":
                raise RuntimeError("iterator close exception")

    @contextmanager
    def scope(planner, start, captured, mode):
        assert flags.closed and torch.equal(active, physical)
        assert start is current and mode == "end"
        assert torch.equal(captured.position, endpoint.position)
        if failure == "scope_enter_exception":
            raise RuntimeError("scope entry exception")
        flags.entered += 1
        flags.scope = True
        try:
            yield
        finally:
            flags.scope = False

    def ik(goal, **options):
        if options.get("run_optimizer") is False:
            flags.endpoint_metrics += 1
            assert flags.scope and torch.equal(active, physical)
            assert goal is target
            assert torch.equal(options["seed_config"][0, 0], endpoint.position[0])
            if failure == "endpoint_exception":
                raise RuntimeError("endpoint exception")
            return NS(success=torch.tensor([failure != "endpoint_reject"]))
        z = goal.position.reshape(-1, 3)[0, 2]
        query_spheres = sphere[None, None].clone()
        query_spheres[..., 2] = z
        ordinary_cost = max(0.0, float(0.05 - z))
        actual_cost = contact.cost(query_spheres).item() if flags.scope else ordinary_cost
        flags.prefix.append(
            {
                "terminal_only_scope": flags.scope,
                "cost": actual_cost,
                "ordinary_cost": ordinary_cost,
            }
        )
        if failure == "prefix_exception" and len(flags.prefix) == 4:
            raise RuntimeError("prefix exception")
        solution = torch.zeros(1, 2, 2)
        solution[..., 0] = z
        assert bool(((solution >= active[0]) & (solution <= active[1])).all())
        return NS(
            solution=solution,
            success=torch.full((1, 2), actual_cost == 0),
            total_time=0.0,
            solve_time=0.0,
        )

    def trajectory(goal, start, **options):
        flags.trajectories += 1
        assert goal is target and start is current
        assert flags.scope and torch.equal(active, physical)
        assert torch.equal(options["seed_config"][0, 0], endpoint.position[0])
        assert torch.equal(options["seed_traj"][0, 0, -1], endpoint.position[0])
        assert bool(
            ((options["seed_traj"] >= physical[0]) & (options["seed_traj"] <= physical[1])).all()
        )
        if failure == "trajectory_exception":
            raise RuntimeError("trajectory exception")
        return NS(success=torch.tensor([failure != "trajectory_reject"]))

    planner = NS(
        joint_names=names,
        kinematics=NS(
            get_joint_limits=lambda: NS(position=active),
            config=NS(
                kinematics_config=NS(
                    joint_map=torch.tensor([0, 1]), joint_map_type=torch.tensor([2, 5])
                )
            ),
        ),
        update_joint_limits=update,
        compute_kinematics=fk,
        ik_solver=NS(config=NS(num_seeds=2), solve_pose=ik),
        trajopt_solver=NS(config=NS(num_seeds=2), action_horizon=16, solve_pose=trajectory),
        config=NS(trajopt_finetune_attempts=None),
    )
    axis_namespace = {
        "torch": torch,
        "np": np,
        "Rotation": Rotation,
        "Slerp": Slerp,
        "JointState": JointState,
        "GoalToolPose": GoalToolPose,
        "goal_group": goal_group,
        "check_time_budget": budget,
    }
    load_functions(
        CANDIDATE_ROOT / "axis_seeds.py",
        ["local_search_radius", "resample_seed", "continuous_axis_seed"],
        axis_namespace,
    )
    namespace = {
        "torch": torch,
        "JointState": JointState,
        "GoalToolPose": GoalToolPose,
        "check_time_budget": budget,
        "boundary_contact_scope": scope,
        "contact_goal_candidates": candidates,
        "continuous_axis_seed": axis_namespace["continuous_axis_seed"],
        "local_search_radius": axis_namespace["local_search_radius"],
    }
    load_functions(
        CANDIDATE_ROOT / "contact_continuation.py", ["plan_contact_continuation"], namespace
    )
    diagnostics = {}
    return NS(
        planner=planner,
        target=target,
        current=current,
        endpoint=endpoint,
        physical=physical,
        active=active,
        flags=flags,
        diagnostics=diagnostics,
        plan=lambda: namespace["plan_contact_continuation"](
            planner, target, current, {"tool": object()}, diagnostics=diagnostics
        ),
        axis=axis_namespace["continuous_axis_seed"],
        axis_namespace=axis_namespace,
    )


def restored(case):
    assert case.flags.closed
    assert not case.flags.scope
    assert torch.equal(case.active, case.physical)


@pytest.mark.parametrize("descent", [0.001, 0.015, 0.020])
@pytest.mark.parametrize("gap", [0.0, -5e-6, -2e-5])
@pytest.mark.parametrize("tolerance", [1e-6, 1e-5, 5e-5])
def test_intermediate_ik_ordinary_scene_final_checks_exact_scope(descent, gap, tolerance):
    case = harness(descent, gap, tolerance)
    result = case.plan()
    assert result is not None and bool(result.success.any()), case.flags.prefix
    assert len(case.flags.prefix) == 15
    assert all(not sample["terminal_only_scope"] for sample in case.flags.prefix)
    assert case.flags.endpoint_metrics == 1 and case.flags.trajectories == 1
    assert case.flags.entered == 1
    assert "terminal_tail_seed_proposal" not in case.diagnostics["seed_record"]
    restored(case)


@pytest.mark.parametrize("descent,gap", [(0.001, -0.0002), (0.015, -0.001), (0.020, -0.0015)])
def test_shallow_tail_is_seed_only_and_final_checks_keep_scope(descent, gap):
    case = harness(descent, gap)
    result = case.plan()
    assert result is not None and bool(result.success.any())
    assert any(sample["ordinary_cost"] > 0 for sample in case.flags.prefix)
    assert all(not sample["terminal_only_scope"] for sample in case.flags.prefix)
    record = case.diagnostics["seed_record"]["terminal_tail_seed_proposal"]
    assert record == {
        "from_ik_step": len(case.flags.prefix),
        "reason": "native_ik_no_valid_seed",
        "qualification": "seed_proposal_only_not_native_trajectory_acceptance",
    }
    assert case.flags.endpoint_metrics == 1 and case.flags.trajectories == 1
    restored(case)


@pytest.mark.parametrize("failure", ["endpoint_reject", "trajectory_reject"])
def test_final_native_rejection_is_never_overwritten(failure):
    case = harness(0.015, -0.001, failure=failure)
    result = case.plan()
    assert case.flags.endpoint_metrics == 1
    assert result is None if failure == "endpoint_reject" else not bool(result.success.any())
    assert case.flags.trajectories == (0 if failure == "endpoint_reject" else 1)
    assert "terminal_tail_seed_proposal" in case.diagnostics["seed_record"]
    restored(case)


@pytest.mark.parametrize(
    "failure",
    [
        "proposal_exception",
        "iterator_close_exception",
        "prefix_exception",
        "prefix_deadline",
        "endpoint_exception",
        "trajectory_exception",
        "scope_enter_exception",
        "scoped_deadline",
    ],
)
def test_failure_exception_deadline_cleanup(failure):
    case = harness(failure=failure)
    error = TimeoutError if "deadline" in failure else RuntimeError
    with pytest.raises(error):
        case.plan()
    if failure.startswith("prefix"):
        assert len(case.flags.prefix) == 4
        assert case.flags.entered == 0
    restored(case)


def test_missing_endpoint_never_enters_scope():
    case = harness(failure="proposal_none")
    assert case.plan() is None
    assert not case.flags.prefix and case.flags.entered == 0
    restored(case)


def axis_failure_case(
    hint_value,
    *,
    hint_enabled=True,
    proposal_enabled=True,
    proposal_outcome="same",
    exception=None,
):
    case = harness()
    # Direct helper contract: a deliberately distant prismatic endpoint tests the
    # current local box. This fixture bypasses caller endpoint proposal on purpose.
    case.current.position.zero_()
    case.planner.compute_kinematics = lambda state: NS(
        tool_poses=ToolPose(
            ["tool"], torch.zeros(1, 1, 1, 3), torch.tensor([[[[1.0, 0.0, 0.0, 0.0]]]])
        )
    )
    goal = GoalToolPose(
        ["tool"], torch.tensor([[[[[0.0, 0.0, 0.1]]]]]), torch.tensor([[[[[1.0, 0.0, 0.0, 0.0]]]]])
    )
    hint = JointState.from_position(torch.tensor([[hint_value, 0.0]]), case.planner.joint_names)
    observations = []

    def no_valid(target, **options):
        if exception is not None:
            raise exception
        return NS(
            solution=torch.zeros(1, 2, 2),
            success=torch.zeros(1, 2, dtype=torch.bool),
            total_time=0.0,
            solve_time=0.0,
        )

    def proposal(planner, target, previous):
        observations.append(target.position.clone())
        # The tail proposal receives the final paired target, not step 1/16.
        assert torch.equal(target.position, goal.position)
        if proposal_outcome == "raise":
            raise RuntimeError("tail proposal failure")
        if proposal_outcome == "none":
            return None
        position = hint.position.clone()
        if proposal_outcome == "different":
            position[0, 0] = torch.nextafter(position[0, 0], torch.tensor(float("inf")))
        elif proposal_outcome == "nonfinite":
            position[0, 0] = float("nan")
        elif proposal_outcome == "outside":
            position[0, 0] = 0.1
        elif proposal_outcome == "wrong_shape":
            return NS(reorder=lambda names: NS(position=torch.zeros(1, 3)))
        return JointState.from_position(position, case.planner.joint_names)

    case.planner.ik_solver.solve_pose = no_valid
    options = {}
    if hint_enabled:
        options["terminal_hint"] = hint
    if proposal_enabled:
        options["terminal_proposal"] = proposal
    return case, goal, hint, observations, options


@pytest.mark.parametrize("direction", ["inside", "exact", "outside"])
def test_tail_current_local_box_is_inclusive_without_expansion(direction):
    boundary = torch.tensor(0.02)
    value = (
        boundary
        if direction == "exact"
        else torch.nextafter(
            boundary, torch.tensor(0.0 if direction == "inside" else float("inf"))
        )
    )
    case, goal, hint, calls, options = axis_failure_case(value.item())
    result = case.axis(case.planner, goal, case.current, {"tool": object()}, **options)
    if direction == "outside":
        assert result is None and not calls
    else:
        assert result is not None and len(calls) == 1
        assert torch.equal(result[1], hint.position[0])
        assert result[3]["terminal_tail_seed_proposal"]["from_ik_step"] == 1
    assert torch.equal(case.active, case.physical)


@pytest.mark.parametrize(
    "hint_enabled,proposal_enabled", [(False, False), (True, False), (False, True)]
)
def test_tail_requires_both_hint_and_proposal(hint_enabled, proposal_enabled):
    case, goal, _, calls, options = axis_failure_case(
        0.01, hint_enabled=hint_enabled, proposal_enabled=proposal_enabled
    )
    assert case.axis(case.planner, goal, case.current, {"tool": object()}, **options) is None
    assert not calls and torch.equal(case.active, case.physical)


@pytest.mark.parametrize(
    "outcome", ["none", "different", "nonfinite", "outside", "wrong_shape", "raise"]
)
def test_tail_proposal_is_once_exact_finite_local_or_rejected(outcome):
    case, goal, _, calls, options = axis_failure_case(0.01, proposal_outcome=outcome)
    if outcome == "none":
        assert case.axis(case.planner, goal, case.current, {"tool": object()}, **options) is None
    else:
        with pytest.raises(RuntimeError if outcome == "raise" else ValueError):
            case.axis(case.planner, goal, case.current, {"tool": object()}, **options)
    assert len(calls) == 1 and torch.equal(case.active, case.physical)


@pytest.mark.parametrize(
    "error", [RuntimeError("native failure"), TimeoutError("native deadline")]
)
def test_native_exception_is_not_replaced_with_tail(error):
    case, goal, _, calls, options = axis_failure_case(0.01, exception=error)
    with pytest.raises(type(error)):
        case.axis(case.planner, goal, case.current, {"tool": object()}, **options)
    assert not calls and torch.equal(case.active, case.physical)


def test_expired_budget_after_native_failure_does_not_call_tail():
    case, goal, _, calls, options = axis_failure_case(0.01)
    checks = []

    def budget():
        checks.append(True)
        if len(checks) == 2:
            raise TimeoutError("expired after native return")

    case.axis_namespace["check_time_budget"] = budget
    with pytest.raises(TimeoutError):
        case.axis(case.planner, goal, case.current, {"tool": object()}, **options)
    assert len(checks) == 2 and not calls and torch.equal(case.active, case.physical)
