# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""CPU checks for request-scoped posture activation and graph invalidation."""

from types import SimpleNamespace

import pytest
import torch

import curobo.runtime as runtime
from curobo._src.cost.cost_base_cfg import BaseCostCfg
from curobo._src.cost.cost_posture import PostureCost
from curobo._src.motion.motion_posture import posture_scope
from curobo._src.rollout.cost_manager.cost_manager_robot import RobotCostManager
from curobo._src.rollout.metrics import CostCollection
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg


@pytest.fixture
def device_cfg(monkeypatch):
    monkeypatch.setattr(runtime, "cuda_streams", False)
    return DeviceCfg(device=torch.device("cpu"))


def test_unconfigured_posture_is_omitted_from_ordinary_rollout(device_cfg):
    cost = PostureCost(BaseCostCfg(weight=1.0, device_cfg=device_cfg), 3)
    manager = RobotCostManager(device_cfg)
    manager.register_cost("posture", cost)
    state = SimpleNamespace(joint_state=JointState.from_position(torch.zeros((1, 2, 3))))
    assert not cost.enabled
    assert manager.compute_costs(state).is_empty()


def test_inactive_and_preexisting_cost_events_are_not_joined(device_cfg, monkeypatch):
    from curobo._src.rollout.cost_manager import cost_manager_robot

    cost = PostureCost(BaseCostCfg(weight=1.0, device_cfg=device_cfg), 3)
    manager = RobotCostManager(device_cfg)
    manager.register_cost("posture", cost)
    joined = []
    monkeypatch.setattr(
        cost_manager_robot, "synchronize_cuda_streams",
        lambda events, device: joined.append(tuple(events)),
    )
    state = SimpleNamespace(joint_state=JointState.from_position(torch.zeros((1, 2, 3))))
    existing = CostCollection()
    existing.add(torch.zeros((1, 2, 1)), "posture")
    manager.compute_costs(state, existing)
    assert joined == [()]
    cost.configure(
        torch.ones((1, 3)), torch.ones(3), torch.zeros(3), torch.zeros(3), torch.zeros(3)
    )
    manager.compute_costs(state)
    assert joined == [(), ("posture",)]
    cost.deactivate()
    manager.compute_costs(state)
    assert joined == [(), ("posture",), ()]
    goals = torch.tensor([[0.0, 1.0, 0.3], [1.0, 0.0, 0.3]])
    cost.configure(goals, torch.ones(3), torch.tensor([0.0, 0.0, 1.0]), goals[0], torch.zeros(3))
    assert cost.enabled
    assert manager.compute_costs(state).names == ["posture"]
    cost.deactivate()
    assert not cost.enabled
    assert manager.compute_costs(state).is_empty()


def test_reactivated_posture_preserves_goal_selection_and_gradients(device_cfg):
    cost = PostureCost(BaseCostCfg(weight=1.0, device_cfg=device_cfg), 3)
    goals = torch.tensor([[0.0, 1.0, 0.3], [1.0, 0.0, 0.3]])
    for _ in range(2):
        cost.configure(
            goals, torch.ones(3), torch.tensor([0.0, 0.0, 1.0]), goals[0], torch.zeros(3)
        )
        assert cost.forward(goals[0].reshape(1, 1, 3)).sum() == 0
        mixed = torch.tensor([[[0.0, 0.0, 0.3]]], requires_grad=True)
        residual = cost.forward(mixed).sum()
        assert residual == 1
        residual.backward()
        assert torch.equal(mixed.grad, torch.tensor([[[0.0, -2.0, 0.0]]]))
        path = goals[0].repeat(1, 3, 1)
        path[0, 1, 2] += 0.1
        assert cost.forward(path).sum() == pytest.approx(0.01)
        cost.deactivate()
        assert cost.forward(path).sum() == 0


@pytest.mark.parametrize("abort", [False, True])
def test_scope_invalidates_graphs_and_restores_after_exit(device_cfg, abort):
    cost = PostureCost(BaseCostCfg(weight=1.0, device_cfg=device_cfg), 3)
    axes = SimpleNamespace(
        terminal_pose_axes_weight_factor=torch.ones(6),
        non_terminal_pose_axes_weight_factor=torch.ones(6),
    )
    tool = SimpleNamespace(_stacked_tool_pose_criteria=axes)
    rollout = SimpleNamespace(
        get_cost_component_by_name=lambda name: {
            "posture": [cost], "tool_pose": [tool], "axis_hold": []
        }[name]
    )
    invalidations = []

    def solver(name):
        return SimpleNamespace(core=SimpleNamespace(
            get_all_rollout_instances=lambda: [rollout],
            additional_metrics_rollouts={},
            invalidate_parameter_graphs=lambda: invalidations.append((name, cost.enabled)),
        ))

    planner = SimpleNamespace(
        joint_names=["a", "b", "c"], ik_solver=solver("ik"), trajopt_solver=solver("trajopt"),
        kinematics=SimpleNamespace(get_joint_limits=lambda: SimpleNamespace(
            position=torch.tensor([[-1.0] * 3, [1.0] * 3])
        )),
    )
    start = JointState.from_position(torch.zeros((1, 3)), joint_names=planner.joint_names)
    goals = JointState.from_position(torch.ones((1, 3)) * 0.1, joint_names=planner.joint_names)
    try:
        with posture_scope(planner, goals, start, (), (), 0.01):
            assert cost.enabled
            assert not axes.terminal_pose_axes_weight_factor.any()
            assert invalidations == [("ik", False), ("trajopt", False)]
            if abort:
                raise RuntimeError("cancel query")
    except RuntimeError as error:
        assert abort and str(error) == "cancel query"
    assert not cost.enabled
    assert not cost.active.any()
    assert axes.terminal_pose_axes_weight_factor.all()
    assert axes.non_terminal_pose_axes_weight_factor.all()
    assert invalidations == [("ik", False), ("trajopt", False)] * 2


def test_zero_weight_posture_remains_disabled(device_cfg):
    cost = PostureCost(BaseCostCfg(weight=0.0, device_cfg=device_cfg), 3)
    cost.configure(
        torch.ones((1, 3)), torch.ones(3), torch.zeros(3), torch.zeros(3), torch.zeros(3)
    )
    assert not cost.enabled
