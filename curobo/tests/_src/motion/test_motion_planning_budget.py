"""Whole-call planning time budgets preserve results, cleanup and nested deadlines."""

from contextlib import contextmanager

import pytest

from curobo._src.motion.motion_planner import MotionPlanner


@pytest.mark.parametrize("method", ["plan_pose", "plan_cspace", "plan_posture"])
@pytest.mark.parametrize("value", [0, -1, True, float("inf"), float("nan"), "1"])
def test_invalid_budget_rejected_before_planning(method, value):
    planner = MotionPlanner.__new__(MotionPlanner)
    planner._destroyed = True
    with pytest.raises(ValueError, match="timeout_s"):
        getattr(planner, method)(None, None, timeout_s=value)


@pytest.fixture
def clock(monkeypatch):
    from curobo._src.util import time_budget as module

    now = [100.0]
    monkeypatch.setattr(module, "monotonic", lambda: now[0])
    return now


def test_nested_budget_cannot_extend_parent_and_resets(clock):
    from curobo._src.util.time_budget import check_time_budget, time_budget

    with pytest.raises(TimeoutError), time_budget(5):
        clock[0] += 4
        with pytest.raises(TimeoutError):
            with time_budget(100):
                clock[0] += 1
                check_time_budget()
    clock[0] += 1000
    check_time_budget()
    with time_budget(1):
        check_time_budget()


def test_child_timeout_restores_live_parent(clock):
    from curobo._src.util.time_budget import check_time_budget, time_budget

    with time_budget(10):
        with pytest.raises(TimeoutError):
            with time_budget(1):
                clock[0] += 2
        check_time_budget()


def test_none_inherits_and_exception_does_not_leak(clock):
    from curobo._src.util.time_budget import check_time_budget, time_budget

    with pytest.raises(TimeoutError):
        with time_budget(1), time_budget(None):
            clock[0] += 2
    check_time_budget()
    with pytest.raises(RuntimeError, match="original"):
        with time_budget(1):
            raise RuntimeError("original")
    check_time_budget()


def test_pending_chunk_is_finished_before_check_only_with_budget(clock):
    from curobo._src.util.time_budget import check_time_budget, time_budget

    calls = []

    def finish():
        calls.append(True)
        clock[0] += 2

    check_time_budget(finish)
    assert calls == []
    with pytest.raises(TimeoutError), time_budget(1):
        check_time_budget(finish)
    assert calls == [True]


@pytest.mark.parametrize("method", ["plan_pose", "plan_cspace"])
@pytest.mark.parametrize("expire", [False, True])
def test_public_entry_preserves_results_and_unwinds_constraints(clock, method, expire):
    from curobo._src.util.time_budget import check_time_budget

    planner = MotionPlanner.__new__(MotionPlanner)
    planner._destroyed = True
    changes = []

    @contextmanager
    def constraints(*args):
        changes.append("installed")
        try:
            yield
        finally:
            changes.append("restored")

    planner._hold_axis_scope = constraints
    result = object()

    def solve(*args):
        clock[0] += 2 if expire else 0.1
        check_time_budget()
        return result

    planner._plan_pose = solve
    planner._plan_cspace = solve
    if expire:
        with pytest.raises(TimeoutError):
            getattr(planner, method)(None, None, timeout_s=1)
    else:
        assert getattr(planner, method)(None, None, timeout_s=1) is result
    assert changes == ["installed", "restored"]
    check_time_budget()


def test_timeout_restores_temporary_optimizer_settings():
    from types import SimpleNamespace

    from curobo._src.solver.solver_trajopt import _restore_optimizer_settings_on_error

    class Optimizer:
        def __init__(self):
            self.enabled = False
            self.config = SimpleNamespace(num_iters=100)

        def enable(self):
            self.enabled = True

        def disable(self):
            self.enabled = False

        def update_niters(self, value):
            self.config.num_iters = value

    optimizer = Optimizer()
    solver = SimpleNamespace(optimizer=SimpleNamespace(optimizers=[optimizer]))

    @_restore_optimizer_settings_on_error
    def solve(self):
        optimizer.enable()
        optimizer.update_niters(7)
        raise TimeoutError("budget")

    with pytest.raises(TimeoutError):
        solve(solver)
    assert optimizer.config.num_iters == 100
    assert optimizer.enabled is False


@pytest.mark.parametrize('use_graph', [False, True])
def test_native_optimizer_timeout_then_reuse(monkeypatch, clock, use_graph):
    import torch

    from curobo._src.motion.motion_planner_cfg import MotionPlannerCfg
    from curobo._src.optim.components.gradient_opt_core import GradientOptCore
    from curobo._src.types.device_cfg import DeviceCfg
    planner = MotionPlanner(MotionPlannerCfg.create(
        robot='franka.yml', device_cfg=DeviceCfg(device='cuda:0'),
        num_ik_seeds=2, num_trajopt_seeds=1, use_cuda_graph=use_graph,
    ))
    original = GradientOptCore._dispatch
    completed = []
    def dispatch(self, name, *args):
        result = original(self, name, *args)
        if name == '_opt_iters':
            completed.append(name)
            clock[0] += 20
        return result
    from curobo._src.state.state_joint import JointState
    start = JointState.from_position(planner.default_joint_state.position.unsqueeze(0),
                                     joint_names=planner.joint_names)
    try:
        monkeypatch.setattr(GradientOptCore, '_dispatch', dispatch)
        with pytest.raises(TimeoutError):
            planner.plan_cspace(start, start, max_attempts=1, timeout_s=10)
        assert completed
        monkeypatch.setattr(GradientOptCore, '_dispatch', original)
        result = planner.plan_cspace(start, start, max_attempts=1, timeout_s=10)
        assert result is not None and bool(result.success.any())
        assert torch.isfinite(result.js_solution.position).all()
    finally:
        planner.destroy()


def test_timeout_keeps_chunk_context(clock):
    from curobo._src.util.time_budget import check_time_budget, time_budget

    with pytest.raises(TimeoutError, match="trajectory chunk 7"), time_budget(1):
        clock[0] += 2
        check_time_budget(context="trajectory chunk 7")
