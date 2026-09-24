# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Kinematic endpoint candidates must pass real bounded contact validation."""

from types import SimpleNamespace

import pytest
import torch

from curobo._src.cost.cost_scene_collision import SceneCollisionCost
from curobo._src.cost.cost_scene_collision_cfg import SceneCollisionCostCfg
from curobo._src.geom.collision.collision_scene import SceneCollision, SceneCollisionCfg
from curobo._src.geom.types import Cuboid, SceneCfg
from curobo._src.motion.motion_contact import contact_goal_seed
from curobo.types import JointState


@pytest.fixture(params=["mesh", "cuboid"])
def scene(request):
    """Keep a support and an unrelated ceiling active in the native collision scene."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    table = Cuboid(name="table", dims=[1, 1, 0.1], pose=[0, 0, -0.05, 1, 0, 0, 0])
    ceiling = Cuboid(name="ceiling", dims=[1, 1, 0.1], pose=[0, 0, 0.2, 1, 0, 0, 0])
    return SceneCollision.from_config(
        SceneCollisionCfg(
            scene_model=SceneCfg(
                cuboid=[ceiling] + ([table] if request.param == "cuboid" else []),
                mesh=[table.get_mesh()] if request.param == "mesh" else [],
            )
        )
    )


def planner_fixture(scene, heights, successful=None):
    """Inject deterministic IK endpoints but use real mesh/cuboid clearance queries."""
    config = SceneCollisionCostCfg(weight=5.0, num_spheres=1)
    config.scene_collision_checker = scene
    cost = SceneCollisionCost(config)
    calls = []

    def solve(*args, **kwargs):
        assert not cost.enabled
        assert kwargs["return_seeds"] == len(heights)
        calls.append(kwargs)
        return SimpleNamespace(
            solution=torch.tensor(heights, device="cuda").reshape(1, -1, 1),
            success=torch.tensor([successful or [True] * len(heights)], device="cuda"),
        )

    def forward(state):
        assert cost.enabled
        sphere = torch.zeros((1, 4), device="cuda")
        sphere[0, 2] = state.position[0, 0]
        sphere[0, 3] = 0.05
        return SimpleNamespace(robot_spheres=sphere)

    core = SimpleNamespace(
        get_all_rollout_instances=lambda: [
            SimpleNamespace(get_cost_component_by_name=lambda name: [cost])
        ],
        additional_metrics_rollouts={},
        invalidate_parameter_graphs=lambda: None,
    )
    planner = SimpleNamespace(
        ik_solver=SimpleNamespace(
            core=core, config=SimpleNamespace(num_seeds=len(heights)), solve_pose=solve
        ),
        compute_kinematics=forward,
        scene_collision_checker=scene,
        joint_names=["height"],
    )
    return planner, cost, calls


@pytest.mark.parametrize("invalid_height", [0.03, 0.2, 0.047999])
def test_reject_deep_candidate_then_keep_bounded_candidate(scene, invalid_height):
    """Reject deep support/other-obstacle overlap before selecting a sub-2mm endpoint."""
    planner, cost, calls = planner_fixture(scene, [invalid_height, 0.048001])
    before = cost._weight.clone()
    result = contact_goal_seed(planner, object(), JointState.from_position(torch.zeros((1, 1))))
    assert float(result.position[0, 0]) == pytest.approx(0.048001)
    assert len(calls) == 1
    assert cost.enabled
    torch.testing.assert_close(cost._weight, before)


def test_all_deep_candidates_return_none_with_summary(scene, caplog):
    """Candidate geometry rejection is ordinary failure, never increased contact tolerance."""
    planner, cost, _ = planner_fixture(scene, [0.03, 0.2])
    assert (
        contact_goal_seed(planner, object(), JointState.from_position(torch.zeros((1, 1)))) is None
    )
    assert "All 2 kinematic endpoint candidates failed" in caplog.text
    assert cost.enabled


def test_ik_rejected_candidates_remain_rejected(scene):
    """Self-collision/limit failure from the IK result cannot become a contact candidate."""
    planner, _, _ = planner_fixture(scene, [0.049, 0.055], successful=[False, True])
    result = contact_goal_seed(planner, object(), JointState.from_position(torch.zeros((1, 1))))
    assert float(result.position[0, 0]) == pytest.approx(0.055)


def test_all_ik_failures_preserve_evidence_and_restore_scene_checks(scene, caplog):
    """Report existing bounded IK evidence without accepting an invalid endpoint."""
    planner, cost, calls = planner_fixture(scene, [0.049], successful=[False])
    solve = planner.ik_solver.solve_pose
    evidence = {"seed_count": 1, "self_collision": {"maximum": 0.003}}

    def rejected(*args, **kwargs):
        result = solve(*args, **kwargs)
        result.debug_info = {"failed_ik": evidence}
        return result

    planner.ik_solver.solve_pose = rejected
    before = cost._weight.clone()
    current = JointState.from_position(torch.zeros((1, 1)))
    assert contact_goal_seed(planner, object(), current) is None
    assert "Terminal endpoint IK rejected all seeds" in caplog.text
    assert str(evidence) in caplog.text
    assert len(calls) == 1 and cost.enabled
    torch.testing.assert_close(cost._weight, before)


def test_unrelated_value_error_propagates(scene):
    """Unexpected model failures are not swallowed as inadmissible geometry."""
    planner, cost, _ = planner_fixture(scene, [0.049])

    def broken_model(state):
        raise ValueError("broken collision model")

    planner.compute_kinematics = broken_model
    with pytest.raises(ValueError, match="broken collision model"):
        contact_goal_seed(planner, object(), JointState.from_position(torch.zeros((1, 1))))
    assert cost.enabled
