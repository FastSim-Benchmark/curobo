# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""IK failures retain the distinction between convergence and constraints."""

from types import SimpleNamespace

import pytest
import torch

from curobo._src.cost.cost_cspace_type import CSpaceCostType
from curobo._src.solver.ik_diagnostics import (
    ik_failure_diagnostics,
    self_collision_pair_diagnostics,
)
from curobo._src.state.state_joint import JointState
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.types import GoalToolPose, Pose


def test_failed_ik_counts_constraints_without_changing_success():
    """Collision-free but unconverged seeds differ from converged colliding seeds."""
    positions = torch.tensor([[[0.0]], [[1.5]], [[0.5]], [[0.75]]])
    feasible = torch.tensor([True, False, False, True])
    converged = torch.tensor([False, True, True, False])
    success = feasible & converged
    before = success.clone()
    metrics = SimpleNamespace(
        state=SimpleNamespace(joint_state=JointState.from_position(positions)),
        costs_and_constraints=SimpleNamespace(
            constraints=SimpleNamespace(
                names=["cspace", "scene_collision", "self_collision"],
                values=[
                    torch.tensor([0.0, 2.0, 0.0, 0.0]).reshape(4, 1, 1),
                    torch.tensor([0.0, 0.0, 3.0, 0.0]).reshape(4, 1, 1),
                    torch.zeros(4, 1, 1),
                ],
            )
        ),
    )
    config = SimpleNamespace(
        cost_type=CSpaceCostType.POSITION,
        joint_limits=SimpleNamespace(joint_names=["arm"], position=torch.tensor([[-1.0], [1.0]])),
        weight=torch.tensor([5000.0, 0.0]),
        activation_distance=torch.zeros(2),
    )
    result = ik_failure_diagnostics(
        metrics, feasible, converged, success, torch.tensor([[1, 3]]), config
    )
    assert result["feasible_seed_count"] == 2
    assert result["converged_seed_count"] == 2
    assert result["successful_seed_count"] == 0
    assert result["constraints"]["scene_collision"] == {
        "positive_cost_seed_count": 1, "maximum": 3.0, "selected_maximum": 0.0
    }
    assert result["constraints"]["self_collision"]["positive_cost_seed_count"] == 0
    assert result["selected_joint_bounds"]["terms"]["position"]["maximum_excess"] == 0.5
    assert result["selected_joint_bounds"]["source"] == "ik_endpoint_joint_state"
    assert result["converged_rejected_seeds"] == {
        "count": 2,
        "sample_truncated": False,
        "seeds": [
            {"seed_index": 1, "feasible": False,
             "constraint_maxima": {"cspace": 2.0, "scene_collision": 0.0, "self_collision": 0.0}},
            {"seed_index": 2, "feasible": False,
             "constraint_maxima": {"cspace": 0.0, "scene_collision": 3.0, "self_collision": 0.0}},
        ],
    }
    torch.testing.assert_close(success, before)


def test_missing_cspace_config_and_selected_indices_stay_bounded():
    """Optional bounds remain absent and only eight selected indices are emitted."""
    metrics = SimpleNamespace(
        costs_and_constraints=SimpleNamespace(constraints=SimpleNamespace(names=[], values=[]))
    )
    zero = torch.zeros(40, dtype=torch.bool)
    result = ik_failure_diagnostics(metrics, zero, zero, zero, torch.arange(40), None)
    assert result["selected_seed_indices"] == list(range(8))
    assert result["selected_seed_count"] == 40
    assert "selected_joint_bounds" not in result
    assert result["converged_rejected_seeds"] == {
        "count": 0, "sample_truncated": False, "seeds": []
    }


def test_converged_rejections_include_unselected_seeds_and_are_bounded():
    metrics = SimpleNamespace(
        costs_and_constraints=SimpleNamespace(constraints=SimpleNamespace(
            names=["scene_collision"], values=[torch.arange(20, dtype=torch.float32)]
        ))
    )
    converged = torch.arange(20) >= 5
    feasible = torch.arange(20) == 19
    success = feasible & converged
    before = [tensor.clone() for tensor in (feasible, converged, success)]
    result = ik_failure_diagnostics(
        metrics, feasible, converged, success, torch.tensor([0]), None
    )["converged_rejected_seeds"]
    assert result["count"] == 14 and result["sample_truncated"]
    assert [s["seed_index"] for s in result["seeds"]] == list(range(5, 13))
    assert all(s["constraint_maxima"]["scene_collision"] == s["seed_index"]
               for s in result["seeds"])
    for tensor, previous in zip((feasible, converged, success), before):
        torch.testing.assert_close(tensor, previous)


def pair_inputs(spheres, pairs, padding=None):
    spheres = torch.tensor(spheres, dtype=torch.float32)
    count = spheres.shape[-2]
    seed_count = spheres.shape[0]
    state = SimpleNamespace(
        robot_spheres=spheres,
        joint_state=JointState.from_position(torch.arange(seed_count).reshape(seed_count, 1, 1)),
    )
    collision = SimpleNamespace(
        num_spheres=count,
        collision_pairs=torch.tensor(pairs, dtype=torch.int16).reshape(-1, 2),
        sphere_padding=torch.tensor(padding) if padding is not None else torch.zeros(count),
    )
    kinematics = SimpleNamespace(
        link_sphere_idx_map=torch.arange(count),
        link_name_to_idx_map={f"link_{index}": index for index in range(count)},
        joint_names=["arm"],
    )
    return state, collision, kinematics


def test_pairs_use_same_fk_enabled_mask_padding_and_selected_seed():
    """An ignored overlapping pair and an inactive sphere must not be reported."""
    state, collision, kinematics = pair_inputs(
        [[[[0., 0., 0., .25], [.75, 0., 0., .25], [0., 0., 0., 5.], [0., 0., 0., -1.]]],
         [[[0., 0., 0., .25], [.5, 0., 0., .25], [0., 0., 0., 5.], [0., 0., 0., -1.]]]],
        [[0, 1], [0, 3]], padding=[.125, .125, 0., 0.],
    )
    before = state.robot_spheres.clone()
    result = self_collision_pair_diagnostics(
        state, torch.tensor([[1, 0]]), 2, collision, kinematics
    )
    first, tangent = result["seeds"]
    assert first["seed_index"] == 1 and first["joint_positions"] == [1]
    assert first["positive_pair_count"] == 1
    pair, = first["pairs"]
    assert [s["native_index"] for s in pair["spheres"]] == [0, 1]
    assert [s["link"] for s in pair["spheres"]] == ["link_0", "link_1"]
    assert pair["spheres"][1]["center_m"] == [.5, 0., 0.]
    assert pair["spheres"][0]["padding_m"] == .125
    assert pair["overlap_m"] == .25
    assert pair["squared_overlap_m2"] == .3125
    assert tangent["positive_pair_count"] == 0 and tangent["pairs"] == []
    torch.testing.assert_close(before, state.robot_spheres)


def test_pair_details_are_bounded_and_ranked_by_native_squared_overlap():
    """Bound logging to four selected seeds and eight active pairs per seed."""
    one = [[0., 0., 0., float(index+1)] for index in range(12)]
    state, collision, kinematics = pair_inputs(
        [[one] for _ in range(6)], [[0, j] for j in range(1, 12)]
    )
    # The metrics FK may retain batch and seed axes rather than a merged axis.
    state.robot_spheres = state.robot_spheres.reshape(2, 3, 1, 12, 4)
    result = self_collision_pair_diagnostics(
        state, torch.arange(5, -1, -1), 6, collision, kinematics
    )
    assert result["selected_seed_count"] == 6 and result["reported_seed_count"] == 4
    assert result["seed_sample_truncated"]
    assert [s["seed_index"] for s in result["seeds"]] == [5, 4, 3, 2]
    for seed in result["seeds"]:
        assert seed["positive_pair_count"] == 11 and seed["pair_sample_truncated"]
        assert len(seed["pairs"]) == 8
        assert [p["spheres"][1]["native_index"] for p in seed["pairs"]] == list(range(11, 3, -1))


def test_missing_fk_is_explicit_and_invalid_layout_is_not_silently_ignored():
    state, collision, kinematics = pair_inputs([[[[0., 0., 0., .25]]]], [])
    result = self_collision_pair_diagnostics(state, torch.tensor([0]), 1, collision, kinematics)
    assert result["enabled_pair_count"] == 0 and result["seeds"][0]["pairs"] == []
    state.robot_spheres = None
    assert self_collision_pair_diagnostics(state, torch.tensor([0]), 1, collision, kinematics) == {
        "available": False, "reason": "metrics_state_has_no_robot_spheres"
    }
    state.robot_spheres = torch.zeros(1, 1, 2, 4)
    with pytest.raises(ValueError, match="sphere shape"):
        self_collision_pair_diagnostics(state, torch.tensor([0]), 1, collision, kinematics)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_native_unreachable_ik_retains_failed_constraints():
    """The native failed IK result contains diagnostics without accepting the goal."""
    config = InverseKinematicsCfg.create("franka.yml", num_seeds=4, use_cuda_graph=False)
    solver = InverseKinematics(config)
    try:
        frames = solver.kinematics.tool_frames
        pose = Pose(
            position=torch.tensor([[[20.0, 0.0, 0.0]]], device="cuda"),
            quaternion=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], device="cuda"),
        )
        goals = GoalToolPose.from_poses({frames[0]: pose}, ordered_tool_frames=frames)
        result = solver.solve_pose(goals, return_seeds=1)
        assert not result.success.any()
        detail = result.debug_info["failed_ik"]
        assert detail["successful_seed_count"] == 0
        assert detail["seed_count"] == 4
        assert "self_collision" in detail["constraints"]
        assert detail["selected_joint_bounds"]["trajectory_count"] == 1
        pairs = detail["selected_self_collision_pairs"]
        assert pairs["available"] and pairs["reported_seed_count"] == 1
        assert pairs["source"] == "same_ik_metrics_robot_spheres_and_enabled_collision_pairs"
    finally:
        solver.destroy()
