# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Public per-call boundary contact parameters on complete native trajectories."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from curobo._src.solver.solver_trajopt_result import TrajOptSolverResult
from curobo.examples.reference.contact_separation import make_planner
from curobo.types import JointState


@pytest.mark.parametrize("kind", ["pose", "joints"])
@pytest.mark.parametrize("mode", ["start", "end", "both"])
def test_native_boundary_contacts_are_request_scoped(tmp_path: Path, kind: str, mode: str) -> None:
    """Plan directly between caller endpoints and restore ordinary collision checks."""
    planner, contact, clear, scene = make_planner(tmp_path, "table", False)
    start = clear if mode == "end" else contact
    goal = clear if mode == "start" else contact.clone()
    if mode == "both":
        goal.position[..., 0] = -0.2
    try:

        def plan(value: str) -> TrajOptSolverResult | None:
            options = dict(max_attempts=2, allow_boundary_collision=value)
            if kind == "joints":
                return planner.plan_cspace(goal, start, **options)
            return planner.plan_pose(
                planner.compute_kinematics(goal).tool_poses.as_goal(), start, **options
            )

        result = plan(mode)
        assert result is not None and result.success.all()
        positions = result.get_interpolated_plan().position.reshape(-1, 3)
        torch.testing.assert_close(positions[0], start.position[0], atol=1e-5, rtol=0)
        torch.testing.assert_close(positions[-1], goal.position[0], atol=2e-4, rtol=0)
        # Independent dense FK and analytic clearance for the payload/support pair.
        fractions = torch.arange(8, device=positions.device) / 8
        dense = (
            positions[:-1, None] + fractions[None, :, None] * positions.diff(dim=0)[:, None]
        ).reshape(-1, 3)
        dense = torch.cat((dense, positions[-1:]))
        spheres = planner.compute_kinematics(
            JointState.from_position(dense, joint_names=planner.joint_names)
        ).robot_spheres.reshape(len(dense), -1, 4)
        support = scene.cuboid[0]
        local = (
            spheres[..., :3] - spheres.new_tensor(support.pose[:3])
        ).abs() - spheres.new_tensor(support.dims) / 2
        gaps = local.clamp_min(0).norm(dim=-1) + local.amax(-1).clamp_max(0) - spheres[..., 3]
        enabled = spheres[0, :, 3] > 0
        assert gaps[:, enabled].min() >= -0.00102
        if mode == "both":
            assert gaps[:, enabled].amin(-1).max() >= 0.00498
        ordinary = plan("none")
        assert ordinary is None or not ordinary.success.any()
    finally:
        planner.destroy()


def test_invalid_boundary_mode_is_rejected_before_planning(tmp_path: Path) -> None:
    """A malformed policy cannot turn into a permissive default."""
    planner, start, goal, _ = make_planner(tmp_path, "table", False)
    try:
        with pytest.raises(ValueError, match="allow_boundary_collision"):
            planner.plan_cspace(goal, start, allow_boundary_collision="all")
    finally:
        planner.destroy()


def test_boundary_scope_restores_after_exception_and_rejects_deep_contact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Failed calls must restore every rollout and may not accept deep penetration."""
    from curobo._src.motion.motion_contact import scene_costs

    planner, start, goal, _ = make_planner(tmp_path, "table", False)
    try:
        with monkeypatch.context() as patch:

            def fail(*args: object, **kwargs: object) -> None:
                raise RuntimeError("injected trajectory failure")

            patch.setattr(planner, "_plan_cspace", fail)
            with pytest.raises(RuntimeError, match="injected trajectory"):
                planner.plan_cspace(goal, start, allow_boundary_collision="start")
        for solver in (planner.ik_solver, planner.trajopt_solver):
            assert all(cost._contact is None for cost in scene_costs(solver))
        deep = start.clone()
        deep.position[..., 2] -= 0.02
        with pytest.raises(ValueError, match="penetration"):
            planner.plan_cspace(goal, deep, allow_boundary_collision="start")
        result = planner.plan_cspace(goal, start, max_attempts=2, allow_boundary_collision="start")
        assert result is not None and result.success.all()
    finally:
        planner.destroy()


def test_goal_seed_exception_restores_exact_scene_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Temporary evidence IK preserves runtime weights and disabled cost components."""
    from curobo._src.motion.motion_contact import scene_costs

    planner, contact, clear, _ = make_planner(tmp_path, "table", False)
    try:
        costs = scene_costs(planner.ik_solver)
        assert len(costs) > 1
        costs[0].disable_cost()
        for cost in costs[1:]:
            if cost.enabled:
                cost._weight.mul_(0.75)
        saved = [(cost.enabled, cost._weight.clone()) for cost in costs]
        assert any(enabled for enabled, _ in saved)

        def fail(*args: object, **kwargs: object) -> None:
            assert all(not cost.enabled for cost in costs)
            raise RuntimeError("injected evidence IK failure")

        monkeypatch.setattr(planner.ik_solver, "solve_pose", fail)
        with pytest.raises(RuntimeError, match="injected evidence IK"):
            planner.plan_pose(
                planner.compute_kinematics(contact).tool_poses.as_goal(),
                clear,
                allow_boundary_collision="end",
            )
        for cost, (enabled, weight) in zip(costs, saved):
            assert cost.enabled is enabled
            torch.testing.assert_close(cost._weight, weight, atol=0, rtol=0)
            assert cost._contact is None
    finally:
        planner.destroy()
