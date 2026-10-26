"""CPU-only AST execution of the isolated candidate; no cuRobo/SDK imports."""

from __future__ import annotations

import ast
import json
import os
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np

MOTION = (
    Path(os.environ["CONTACT_DIAGNOSTIC_CANDIDATE_SOURCES"])
    if "CONTACT_DIAGNOSTIC_CANDIDATE_SOURCES" in os.environ
    else Path(__file__).resolve().parents[3] / "_src" / "motion"
)


class Array:
    def __init__(self, value):
        self.value = np.asarray(value)

    def __getitem__(self, index):
        return Array(self.value[index])

    def reshape(self, *shape):
        return Array(self.value.reshape(*shape))

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class Success:
    def __init__(self, value):
        self.value = value

    def any(self):
        return self.value


def native_result(success=False):
    return NS(
        success=Success(success),
        goalset_index=0,
        total_time=0.01,
        solve_time=0.02,
        debug_info={
            "solver": {"large_native": "x" * 6000},
            "continuous_contact_seed": {"stale": True},
        },
    )


def fixture():
    poses = NS(
        tool_frames=["tool"],
        position=Array([[[[0.0, 0.0, 0.0]]]]),
        quaternion=Array([[[[1.0, 0.0, 0.0, 0.0]]]]),
    )
    goals = NS(
        num_goalset=2,
        position=Array([[[[[1.0, 0.0, 0.0], [0.1, 0.0, 0.0]]]]]),
        quaternion=Array([[[[[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]]]]),
    )
    goals.reorder_links = lambda frames: goals
    return NS(compute_kinematics=lambda state: NS(tool_poses=poses)), goals, object()


def load_seed(continuation):
    path = MOTION / "contact_axis_seeds.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    tree.body = [
        node
        for node in tree.body
        if not (
            isinstance(node, ast.ImportFrom)
            and (node.level or (node.module or "").startswith("curobo"))
        )
    ]
    scope = {
        "goal_group": lambda goals, index: (goals, index),
        "remap_goalset_index": lambda result, index, count: setattr(
            result, "goalset_index", index
        ),
        "plan_contact_continuation": continuation,
    }
    exec(compile(tree, str(path), "exec"), scope)
    return NS(**scope)


class ReplaceCandidateImports(ast.NodeTransformer):
    def visit_ImportFrom(self, node):
        if (node.module or "").startswith("curobo"):
            return [
                ast.copy_location(
                    ast.Assign(
                        targets=[ast.Name(id=item.asname or item.name, ctx=ast.Store())],
                        value=ast.Subscript(
                            value=ast.Name(id="test_imports", ctx=ast.Load()),
                            slice=ast.Constant(item.name),
                            ctx=ast.Load(),
                        ),
                    ),
                    node,
                )
                for item in node.names
            ]
        return node


def load_plan_pose(seed, terminal, logs):
    path = MOTION / "motion_planner.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MotionPlanner"
    )
    method = next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "plan_pose"
    )
    method.decorator_list = []
    method = ReplaceCandidateImports().visit(method)
    code = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            method,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(code)
    scope = {
        "log_warn": logs.append,
        "test_imports": {
            "boundary_contact_scope": lambda *args: nullcontext(),
            "validate_boundary_mode": lambda mode: None,
            "plan_continuous_axis_seed": lambda *args: None,
            "plan_contact_axis_seed": seed,
            "plan_terminal_pose": terminal,
        },
    }
    exec(compile(code, str(path), "exec"), scope)
    return scope["plan_pose"]


def fill_record(record):
    record.update(
        endpoint_proposed=True,
        continuous_seed_present=False,
        endpoint_metrics_success=False,
        continuation={"last_step": 5, "remaining_endpoint_delta": [-0.2, 0.1]},
        seed_record={
            "ik_solve_time": 0.03,
            "terminal_tail_seed_proposal": {
                "from_ik_step": 5,
                "reason": "native_ik_no_valid_seed",
                "qualification": "seed_proposal_only_not_native_trajectory_acceptance",
            },
        },
        giant_unrelated="x" * 100_000,
    )


class DiagnosticPreservationTests(unittest.TestCase):
    def test_native_none_rejected_and_success_keep_scalar_evidence(self):
        for outcome in ("none", "rejected", "success"):
            with self.subTest(outcome=outcome):
                returned = None if outcome == "none" else native_result(outcome == "success")

                def continuation(*args, diagnostics):
                    fill_record(diagnostics)
                    return returned

                module = load_seed(continuation)
                planner, goals, current = fixture()
                evidence = {"must_be_removed": "x" * 100_000}
                result = module.plan_contact_axis_seed(
                    planner, goals, current, {"tool": object()}, False, diagnostics=evidence
                )
                self.assertIs(result, returned)
                self.assertEqual(evidence["outcome"], "no_seed" if outcome == "none" else outcome)
                self.assertEqual(evidence["original_goal_index"], 1)
                self.assertEqual(evidence["remaining_endpoint_delta_absmax"], 0.2)
                self.assertEqual(evidence["tail_seed_proposal_from_ik_step"], 5)
                self.assertNotIn("must_be_removed", evidence)
                self.assertTrue(
                    all(type(value) in (bool, int, float, str) for value in evidence.values())
                )
                self.assertLess(len(repr(evidence).encode()), 1024)
                if returned is not None:
                    self.assertEqual(returned.goalset_index, 1)
                    self.assertAlmostEqual(returned.solve_time, 0.05)
                    self.assertEqual(next(iter(returned.debug_info)), "continuous_contact_seed")
                    self.assertEqual(returned.debug_info["continuous_contact_seed"], evidence)

    def test_timeout_and_error_propagate_with_partial_evidence(self):
        for error in (TimeoutError("deadline"), RuntimeError("native failure")):

            def continuation(*args, diagnostics):
                fill_record(diagnostics)
                raise error

            module = load_seed(continuation)
            planner, goals, current = fixture()
            evidence = {}
            with self.assertRaises(type(error)) as caught:
                module.plan_contact_axis_seed(
                    planner, goals, current, {"tool": object()}, diagnostics=evidence
                )
            self.assertIs(caught.exception, error)
            self.assertEqual(evidence["outcome"], "interrupted")
            self.assertEqual(evidence["continuation_last_step"], 5)

    def test_summary_bounds_reject_arbitrary_objects_arrays_and_nonfinite_numbers(self):
        class NoString:
            def __str__(self):
                raise AssertionError("diagnostics must not stringify arbitrary objects")

        module = load_seed(lambda *args, **kwargs: None)
        raw = {
            "original_goal_index": 1 << 100_000,
            "endpoint_proposed": NoString(),
            "continuation": {
                "last_step": 1 << 100_000,
                "remaining_endpoint_delta": [1 << 100_000],
            },
            "seed_record": {
                "terminal_tail_seed_proposal": {
                    "from_ik_step": NoString(),
                    "reason": NoString(),
                    "qualification": "x" * 100_000,
                }
            },
            "unknown": "x" * 100_000,
        }
        for delta in ([float("nan")], [float("inf")], [NoString()], [0.0] * 4097):
            raw["continuation"]["remaining_endpoint_delta"] = delta
            evidence = module._contact_seed_summary(raw, "x" * 100_000)
            self.assertNotIn("remaining_endpoint_delta_absmax", evidence)
            self.assertTrue(
                all(type(value) in (bool, int, float, str) for value in evidence.values())
            )
            self.assertLess(len(json.dumps(evidence, allow_nan=False).encode()), 1024)
        raw["continuation"]["remaining_endpoint_delta"] = [1 << 100_000]
        self.assertNotIn(
            "remaining_endpoint_delta_absmax", module._contact_seed_summary(raw, "no_seed")
        )

    def test_actual_candidate_routing_success_rejection_none_and_fallback_none(self):
        for outcome, fallback_none in (
            ("success", False),
            ("rejected", False),
            ("none", False),
            ("rejected", True),
            ("none", True),
        ):
            with self.subTest(outcome=outcome, fallback_none=fallback_none):
                candidate = None if outcome == "none" else native_result(outcome == "success")
                fallback = None if fallback_none else native_result(False)
                calls, logs = [], []

                def continuation(*args, diagnostics):
                    fill_record(diagnostics)
                    calls.append("continuous")
                    return candidate

                def terminal(*args):
                    calls.append("terminal")
                    return fallback

                module = load_seed(continuation)
                plan = load_plan_pose(module.plan_contact_axis_seed, terminal, logs)
                planner, goals, current = fixture()
                planner._hold_axis_scope = lambda *args: nullcontext()
                returned = plan(
                    planner,
                    goals,
                    current,
                    hold_axis={"tool": object()},
                    allow_boundary_collision="end",
                    max_attempts=1,
                )
                self.assertIs(returned, candidate if outcome == "success" else fallback)
                self.assertEqual(
                    calls, ["continuous"] if outcome == "success" else ["continuous", "terminal"]
                )
                self.assertEqual(len(logs), 1)
                self.assertIn("Continuous terminal-contact seed summary:", logs[0])
                self.assertNotIn("giant_unrelated", logs[0])
                self.assertLess(len(logs[0].encode()), 1200)
                if returned is not None:
                    self.assertEqual(next(iter(returned.debug_info)), "continuous_contact_seed")
                    evidence = returned.debug_info["continuous_contact_seed"]
                    self.assertIn(repr(evidence), logs[0])
                    self.assertIn(repr(evidence), str(returned.debug_info)[:2048])

    def test_planner_preserves_original_exception_and_logs_before_fallback(self):
        for source in ("continuous", "fallback"):
            for error in (TimeoutError("deadline"), RuntimeError("native failure")):
                with self.subTest(source=source, error=type(error)):
                    logs = []

                    def continuation(*args, diagnostics):
                        fill_record(diagnostics)
                        if source == "continuous":
                            raise error
                        return None

                    def terminal(*args):
                        self.assertEqual(len(logs), 1)
                        raise error

                    module = load_seed(continuation)
                    plan = load_plan_pose(module.plan_contact_axis_seed, terminal, logs)
                    planner, goals, current = fixture()
                    planner._hold_axis_scope = lambda *args: nullcontext()
                    with self.assertRaises(type(error)) as caught:
                        plan(
                            planner,
                            goals,
                            current,
                            hold_axis={"tool": object()},
                            allow_boundary_collision="end",
                        )
                    self.assertIs(caught.exception, error)
                    self.assertEqual(len(logs), 1)

    def test_noneligible_routes_do_not_add_diagnostics(self):
        for mode, holds, attempts in (
            ("end", None, 1),
            ("end", {"tool": object()}, 0),
            ("both", {"tool": object()}, 1),
            ("none", None, 1),
            ("start", {"tool": object()}, 1),
        ):

            def must_not_run(*args, **kwargs):
                self.fail("continuous contact seed must retain its original eligibility")

            fallback, logs = native_result(False), []
            original_debug = fallback.debug_info
            plan = load_plan_pose(must_not_run, lambda *args: fallback, logs)
            planner = NS(
                _hold_axis_scope=lambda *args: nullcontext(), _plan_pose=lambda *args: fallback
            )
            self.assertIs(
                plan(
                    planner,
                    NS(),
                    NS(),
                    hold_axis=holds,
                    allow_boundary_collision=mode,
                    max_attempts=attempts,
                ),
                fallback,
            )
            self.assertIs(fallback.debug_info, original_debug)
            self.assertEqual(logs, [])

    def test_original_timeout_survives_logging_handler_failure(self):
        class LegacyTimeoutError(TimeoutError):
            def __getattribute__(self, name):
                if name == "add_note":
                    raise AttributeError(name)
                return super().__getattribute__(name)

        original = LegacyTimeoutError("original deadline")
        attempted_messages = []

        def continuation(*args, diagnostics):
            fill_record(diagnostics)
            raise original

        def broken_logger(message):
            attempted_messages.append(message)
            raise RuntimeError("handler failed")

        module = load_seed(continuation)
        plan = load_plan_pose(
            module.plan_contact_axis_seed,
            lambda *args: self.fail("no fallback"),
            NS(append=broken_logger),
        )
        planner, goals, current = fixture()
        planner._hold_axis_scope = lambda *args: nullcontext()
        with self.assertRaises(TimeoutError) as caught:
            plan(
                planner,
                goals,
                current,
                hold_axis={"tool": object()},
                allow_boundary_collision="end",
            )
        self.assertIs(caught.exception, original)
        self.assertEqual(len(attempted_messages), 1)
        self.assertIn("'outcome': 'interrupted'", attempted_messages[0])
        self.assertFalse(
            any(
                isinstance(node, ast.Attribute) and node.attr == "add_note"
                for node in ast.walk(ast.parse((MOTION / "motion_planner.py").read_text()))
            )
        )

    def test_logging_failure_without_planning_error_keeps_existing_semantics(self):
        original = RuntimeError("handler failed")

        def broken_logger(message):
            raise original

        module = load_seed(lambda *args, **kwargs: None)
        plan = load_plan_pose(
            module.plan_contact_axis_seed,
            lambda *args: self.fail("no fallback"),
            NS(append=broken_logger),
        )
        planner, goals, current = fixture()
        planner._hold_axis_scope = lambda *args: nullcontext()
        with self.assertRaises(RuntimeError) as caught:
            plan(
                planner,
                goals,
                current,
                hold_axis={"tool": object()},
                allow_boundary_collision="end",
            )
        self.assertIs(caught.exception, original)


if __name__ == "__main__":
    unittest.main(verbosity=2)
