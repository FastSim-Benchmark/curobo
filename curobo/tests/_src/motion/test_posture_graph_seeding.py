# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""A failed direct posture path can reach graph seeding without losing checks."""
from contextlib import contextmanager, nullcontext
from types import MethodType, SimpleNamespace

import pytest
import torch

from curobo._src.motion.motion_planner import MotionPlanner
from curobo.types import JointState


@pytest.mark.parametrize(
    ("graph_enabled", "graph_found", "reject_native", "attempts"),
    [(True, True, False, 2), (True, False, False, 2),
     (False, False, False, 2), (True, True, True, 2), (True, True, False, 1)],
)
def test_posture_detour_reaches_native_validation(
    monkeypatch, graph_enabled, graph_found, reject_native, attempts,
):
    import curobo._src.motion.motion_failure_diagnostics as diagnostics
    import curobo._src.motion.motion_posture as posture

    names = ["x", "y", "held"]
    initial = torch.tensor([[-1., 0., .3]])
    targets = torch.tensor([[1., 0., .3], [1., .2, .3]])
    start = JointState.from_position(initial, joint_names=names)
    goals = JointState.from_position(targets, joint_names=names)
    direct = torch.lerp(initial[0], targets[0], torch.linspace(0, 1, 31)[:, None])
    detour = direct.clone()
    detour[:, 1] = .8 * (1 - direct[:, 0].abs())
    posture_detour = detour.clone()
    posture_detour[:, 1].neg_()
    endpoints = targets.unsqueeze(0)
    graph_calls, native_checks = [], []

    @contextmanager
    def scope(*args, **kwargs):
        yield targets, torch.ones(3), torch.tensor([0., 0., 1.]), initial

    monkeypatch.setattr(posture, "posture_scope", scope)
    monkeypatch.setattr(posture, "boundary_contact_scope", lambda *a, **kw: nullcontext())
    monkeypatch.setattr(posture, "GoalToolPose", lambda *args: SimpleNamespace(num_goalset=1))
    monkeypatch.setattr(diagnostics, "terminal_failure_summary", lambda *a, **kw: {})
    monkeypatch.setattr(posture, "PostureSeeds", lambda *args: SimpleNamespace(
        goals=targets, total_time=0., solve_time=0., records=[],
        endpoint_seeds=lambda attempt: endpoints.clone(),
        trajectory_seeds=lambda tool, q, attempt: (
            None if not attempt else posture_detour[None, None].repeat(1, 2, 1, 1)
        ),
    ))

    def graph_seed(current, candidates):
        assert torch.equal(current.position, initial)
        assert torch.equal(candidates, endpoints)
        graph_calls.append(True)
        return detour[None, None].repeat(1, 2, 1, 1) if graph_found else None

    def solve_pose(tool, current, **kwargs):
        seed = kwargs["seed_traj"]
        path = direct if seed is None else seed[0, 0]
        # A radius .4 barrier rejects the direct path. Both detours clear it;
        # held coordinates and the original joint goal remain authoritative.
        accepted = (
            bool(torch.linalg.vector_norm(path[:, :2], dim=-1).min() > .4)
            and bool(torch.equal(path[:, 2], initial[0, 2].expand(len(path))))
            and bool(torch.equal(path[-1], targets[0]))
            and not reject_native
        )
        native_checks.append({"accepted": accepted, "path": path.clone()})
        trajectory = JointState.from_position(path.unsqueeze(0), joint_names=names)
        return SimpleNamespace(
            success=torch.tensor([accepted]), total_time=.2, solve_time=.1,
            debug_info={"native_checks_retained": True}, goalset_index=torch.tensor([0]),
            get_interpolated_plan=lambda: trajectory,
        )

    planner = SimpleNamespace(
        joint_names=names, action_dim=3, graph_planner=object() if graph_enabled else None,
        config=SimpleNamespace(trajopt_finetune_attempts=None),
        ik_solver=SimpleNamespace(solve_pose=lambda *a, **kw: SimpleNamespace(
            success=torch.ones((1, 2), dtype=torch.bool), solution=endpoints.clone(),
            total_time=.1, solve_time=.05,
        )),
        trajopt_solver=SimpleNamespace(config=SimpleNamespace(num_seeds=2), solve_pose=solve_pose),
        compute_kinematics=lambda q: SimpleNamespace(tool_poses=SimpleNamespace(
            tool_frames=("tcp",), position=torch.zeros((1, 1, 3)),
            quaternion=torch.zeros((1, 1, 4)),
        )),
        _hold_axis_scope=lambda *a, **kw: nullcontext(),
        _get_graph_seed_trajectories=graph_seed,
    )
    planner._plan_pose_single = MethodType(MotionPlanner._plan_pose_single, planner)
    result = posture.plan_posture(
        planner, goals, start, held_joints=("held",), max_attempts=attempts,
    )
    assert len(native_checks) == attempts
    assert not native_checks[0]["accepted"]
    assert len(graph_calls) == int(graph_enabled and attempts > 1)
    assert bool(result.success.any()) == (attempts > 1 and not reject_native)
    assert result.debug_info["native_checks_retained"]
    assert result.debug_info["posture"]["held_error"] == 0.
    assert result.debug_info["posture"]["candidate_count"] == 2
    assert torch.equal(goals.position, targets)
    if attempts > 1:
        expected = detour if graph_enabled and graph_found else posture_detour
        assert torch.equal(native_checks[-1]["path"], expected)
