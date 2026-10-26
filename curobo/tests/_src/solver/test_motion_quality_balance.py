# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Large per-arm exchanges must not displace balanced feasible trajectories."""
import pytest
import torch

from curobo._src.solver.solver_motion_quality import (
    height_overshoot,
    posture_motion_penalty,
    rank_balanced_motion_candidates,
)


def paths(lengths: list[list[list[float]]]) -> torch.Tensor:
    """Construct out-and-back TCP paths with shared endpoints [B,S,3,L,3]."""
    length = torch.tensor(lengths, dtype=torch.float32)
    position = torch.zeros(*length.shape[:2], 3, length.shape[-1], 3)
    position[:, :, 1, :, 0] = length / 2
    return position


def test_large_single_arm_exchange_cannot_win_by_soft_score() -> None:
    position = paths([[[1.40, 1.70], [.10, 2.30]]])
    success = torch.tensor([[True, True]])
    order, details = rank_balanced_motion_candidates(
        position, torch.full((1, 2), 3), success, torch.tensor([[0., 1.]])
    )
    assert order.tolist() == [[1, 0]]
    assert details['within_limits'].tolist() == [[False, True]]
    torch.testing.assert_close(success, torch.tensor([[True, True]]))


def test_small_absolute_exchange_preserves_original_soft_preference() -> None:
    order, _ = rank_balanced_motion_candidates(
        paths([[[.05, .30], [.02, .40]]]), torch.full((1, 2), 3),
        torch.tensor([[True, True]]), torch.tensor([[0., 1.]]),
    )
    assert order.tolist() == [[0, 1]]


def test_a_good_turning_path_survives_without_joint_motion_veto() -> None:
    order, _ = rank_balanced_motion_candidates(
        paths([[[.365, .016], [3.06, .019]]]), torch.full((1, 2), 3),
        torch.tensor([[True, True]]), torch.tensor([[0., 5.]]),
    )
    assert order.tolist() == [[0, 1]]


def test_failed_seed_cannot_set_reference_or_displace_success() -> None:
    order, details = rank_balanced_motion_candidates(
        paths([[[0., 0.], [.40, .50], [.45, .55]]]), torch.full((1, 3), 3),
        torch.tensor([[False, True, True]]), torch.tensor([[-10., 1., 0.]]),
    )
    assert order.tolist() == [[2, 1, 0]]
    torch.testing.assert_close(details['reference_m'], torch.tensor([[.40, .50]]))


def test_conflicting_arm_minima_choose_smallest_overrun_without_failing() -> None:
    success = torch.tensor([[True, True]])
    order, details = rank_balanced_motion_candidates(
        paths([[[.10, 1.], [.60, .10]]]), torch.full((1, 2), 3),
        success, torch.tensor([[0., 1.]]),
    )
    assert not bool(details['within_limits'].any())
    assert order.tolist() == [[1, 0]]
    assert bool(torch.gather(success, 1, order).all())


def test_padding_nan_is_ignored_and_batches_rank_independently() -> None:
    position = paths([[[.10], [.90]], [[.90], [.10]]])
    position[:, :, 2] = torch.nan
    order, details = rank_balanced_motion_candidates(
        position, torch.full((2, 2), 2), torch.ones(2, 2, dtype=torch.bool),
        torch.tensor([[0., -1.], [-1., 0.]]),
    )
    assert order.tolist() == [[0, 1], [1, 0]]
    assert bool(torch.isfinite(details['tcp_path_m']).all())


def test_all_failed_preserves_old_cost_order_and_does_not_create_success() -> None:
    success = torch.zeros(1, 2, dtype=torch.bool)
    order, _ = rank_balanced_motion_candidates(
        paths([[[10.], [0.]]]), torch.full((1, 2), 3), success,
        torch.tensor([[0., 1.]]),
    )
    assert order.tolist() == [[0, 1]]
    assert not bool(success.any())


def test_posture_penalty_allows_small_changes_and_necessary_endpoint_rise() -> None:
    height = torch.tensor([[[[0.], [.02], [0.]], [[0.], [.05], [0.]],
                            [[0.], [.10], [.20]]]])
    torch.testing.assert_close(posture_motion_penalty(height), torch.zeros(1, 3))


def test_posture_overshoot_uses_actual_endpoint_and_ignores_nan_padding() -> None:
    height = torch.tensor([[[[.10], [.30], [.20], [float('nan')]]]])
    actual = height_overshoot(height, torch.tensor([[3]]))
    torch.testing.assert_close(actual, torch.tensor([[[.10]]]))
    torch.testing.assert_close(posture_motion_penalty(height, torch.tensor([[3]])),
                               torch.tensor([[4.]]))


def test_balanced_rank_rejects_large_elbow_trade_within_tcp_allowance() -> None:
    heights = torch.tensor([[[[0.], [.20], [0.]], [[0.], [0.], [0.]]]])
    order, details = rank_balanced_motion_candidates(
        paths([[[.88], [1.10]]]), torch.full((1, 2), 3),
        torch.ones(1, 2, dtype=torch.bool), torch.tensor([[0., 1.]]), heights,
    )
    assert order.tolist() == [[1, 0]]
    assert details['within_limits'].tolist() == [[False, True]]


def test_failed_low_elbow_candidate_cannot_set_posture_reference() -> None:
    heights = torch.tensor([[[[0.], [0.], [0.]], [[0.], [.20], [0.]]]])
    order, details = rank_balanced_motion_candidates(
        paths([[[.80], [.90]]]), torch.full((1, 2), 3),
        torch.tensor([[False, True]]), torch.tensor([[0., 1.]]), heights,
    )
    assert order.tolist() == [[1, 0]]
    torch.testing.assert_close(details['elbow_reference_m'], torch.tensor([[.20]]))


def test_elbow_improvement_cannot_introduce_a_large_tcp_loss() -> None:
    heights = torch.tensor([[[[0.], [.20], [0.]], [[0.], [0.], [0.]]]])
    order, details = rank_balanced_motion_candidates(
        paths([[[.88], [1.15]]]), torch.full((1, 2), 3),
        torch.ones(1, 2, dtype=torch.bool), torch.tensor([[12., 1.]]), heights,
        tcp_baseline_quality=torch.tensor([[0., 1.]]),
    )
    assert order.tolist() == [[0, 1]]
    assert details['tcp_protection_eligible'].tolist() == [[True, False]]


def test_small_tcp_exchange_can_still_remove_large_elbow_overshoot() -> None:
    heights = torch.tensor([[[[0.], [.20], [0.]], [[0.], [0.], [0.]]]])
    order, details = rank_balanced_motion_candidates(
        paths([[[.88], [.90]]]), torch.full((1, 2), 3),
        torch.ones(1, 2, dtype=torch.bool), torch.tensor([[12., 1.]]), heights,
        tcp_baseline_quality=torch.tensor([[0., 1.]]),
    )
    assert order.tolist() == [[1, 0]]
    assert details['tcp_protection_eligible'].tolist() == [[True, True]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_quality_landmarks_preserve_native_topology_and_tool_poses() -> None:
    from curobo._src.motion.motion_planner_cfg import MotionPlannerCfg
    from curobo._src.robot.kinematics.kinematics import Kinematics
    from curobo._src.solver.solver_motion_quality import create_motion_quality_kinematics
    from curobo._src.state.state_joint import JointState

    config = MotionPlannerCfg.create("franka.yml").ik_solver_config.robot_config.kinematics
    params = config.kinematics_config
    native_frames = list(params.tool_frames)
    native_mask = params.joint_affects_endeffector.clone()
    native_spheres = params.link_spheres.clone()
    native = Kinematics(config, compute_spheres=False)
    quality = create_motion_quality_kinematics(config)
    assert len(quality.tool_frames) == 2 * len(native_frames)
    assert params.tool_frames == native_frames
    torch.testing.assert_close(params.joint_affects_endeffector, native_mask, atol=0, rtol=0)
    torch.testing.assert_close(params.link_spheres, native_spheres, atol=0, rtol=0)
    q = native.default_joint_position.reshape(1, 1, -1)
    state = JointState.from_position(q, joint_names=native.joint_names)
    a = native.compute_kinematics(state).tool_poses
    b = quality.compute_kinematics(state).tool_poses
    torch.testing.assert_close(a.position, b.position[..., :len(native_frames), :])
    torch.testing.assert_close(a.quaternion, b.quaternion[..., :len(native_frames), :])
