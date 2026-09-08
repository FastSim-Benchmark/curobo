# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Geometric axis alignment, independent of twist about the held object's axis."""

from __future__ import annotations

import math

import pytest
import torch

from curobo._src.cost.cost_tool_pose import ToolPoseCost
from curobo._src.cost.cost_tool_pose_cfg import ToolPoseCostCfg
from curobo._src.geom.transform import quaternion_rate_to_axis_angle_rate
from curobo._src.util.warp import init_warp
from curobo.examples.reference.tool_orientation import make_planner
from curobo.types import DeviceCfg, GoalToolPose, JointState, Pose, ToolPose, ToolPoseCriteria


@pytest.fixture(scope="module")
def device_cfg() -> DeviceCfg:
    """Initialize CUDA for the native pose cost."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    init_warp(quiet=True)
    return DeviceCfg()


def rotation(axis: list[float], angle: float, device: DeviceCfg) -> torch.Tensor:
    """Construct a unit wxyz quaternion."""
    vector = torch.nn.functional.normalize(device.to_device(axis), dim=0)
    return torch.cat((device.to_device([math.cos(angle / 2)]), vector * math.sin(angle / 2)))


def multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Multiply wxyz quaternions without converting to Euler angles."""
    scalar = left[0] * right[0] - (left[1:] * right[1:]).sum()
    vector = left[0] * right[1:] + right[0] * left[1:] + torch.linalg.cross(left[1:], right[1:])
    return torch.cat((scalar.reshape(1), vector))


def make_cost(criteria: ToolPoseCriteria, lie_group: bool = False) -> ToolPoseCost:
    """Use the ordinary pose cost and its existing runtime criteria update."""
    cost = ToolPoseCost(
        ToolPoseCostCfg(
            weight=[0.0, 1.0],
            tool_frames=["cup"],
            device_cfg=criteria.device_cfg,
            use_grad_input=True,
            use_lie_group=lie_group,
        )
    )
    cost.setup_batch_tensors(1, 1)
    cost.update_tool_pose_criteria({"cup": criteria})
    return cost


def evaluate(
    cost: ToolPoseCost,
    current: torch.Tensor,
    goal: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate a single pair of orientations using the real pose kernel."""
    current_pose = ToolPose(
        tool_frames=["cup"],
        position=torch.zeros((1, 1, 1, 3), device=current.device),
        quaternion=current.reshape(1, 1, 1, 4),
    )
    goal_pose = GoalToolPose(
        tool_frames=["cup"],
        position=torch.zeros((1, 1, 1, 1, 3), device=goal.device),
        quaternion=goal.reshape(1, 1, 1, 1, 4),
    )
    values, _, angle, _ = cost.forward(
        current_pose,
        goal_pose,
        idxs_goal=torch.zeros((1, 1), device=current.device, dtype=torch.int32),
    )
    return values.sum(), angle


@pytest.mark.parametrize("yaw", [0.0, math.pi / 2, math.pi, -math.pi, 3 * math.pi / 2])
@pytest.mark.parametrize(
    "tilt,valid", [(0.0, True), (0.0099, True), (0.0101, False), (math.pi, False)]
)
def test_tilt_threshold_independent_of_yaw(
    device_cfg: DeviceCfg,
    yaw: float,
    tilt: float,
    valid: bool,
) -> None:
    """Allow every yaw, but reject excessive tilt and an upside-down cup."""
    cost = make_cost(ToolPoseCriteria.hold_axis([0, 0, 1], 0.01, device_cfg))
    goal = rotation([0, 0, 1], 0, device_cfg)
    current = multiply(rotation([0, 0, 1], yaw, device_cfg), rotation([1, 0, 0], tilt, device_cfg))
    value, angle = evaluate(cost, current, goal)
    assert (float(value) == 0.0) == valid
    assert float(angle) == pytest.approx(tilt, abs=2e-6)


@pytest.mark.parametrize("project", [False, True])
@pytest.mark.parametrize("lie_group", [False, True])
def test_axis_in_gripper_frame(
    device_cfg: DeviceCfg,
    project: bool,
    lie_group: bool,
) -> None:
    """Handle a cup axis that is not aligned with any gripper axis."""
    axis = [1.0, 2.0, 3.0]
    criteria = ToolPoseCriteria.hold_axis(axis, 0.01, device_cfg)
    criteria.project_distance_to_goal.fill_(project)
    cost = make_cost(criteria, lie_group)
    goal = rotation([1, -1, 0], 0.7, device_cfg)
    for yaw in [0.0, 1.5, math.pi]:
        current = multiply(goal, rotation(axis, yaw, device_cfg))
        value, angle = evaluate(cost, -current, goal)
        assert float(value) == 0
        assert float(angle) < 1e-6


@pytest.mark.parametrize("project", [False, True])
def test_axis_gradient(device_cfg: DeviceCfg, project: bool) -> None:
    """Match independent autodiff through the geometric axis angle in float32."""
    axis = [1.0, 2.0, 3.0]
    criteria = ToolPoseCriteria.hold_axis(axis, 0.01, device_cfg)
    criteria.project_distance_to_goal.fill_(project)
    cost = make_cost(criteria)
    raw = device_cfg.to_device([0.2, 0.3, -0.4, 0.5]).requires_grad_()
    goal = rotation([0, 1, 1], 1.2, device_cfg)
    current = torch.nn.functional.normalize(raw, dim=0)
    value, _ = evaluate(cost, current, goal)
    actual = torch.autograd.grad(value, raw, retain_graph=True)[0]
    matrix = Pose(quaternion=current.reshape(1, 4)).get_rotation_matrix()[0]
    goal_matrix = Pose(quaternion=goal.reshape(1, 4)).get_rotation_matrix()[0]
    local_axis = torch.nn.functional.normalize(device_cfg.to_device(axis), dim=0)
    a, b = matrix @ local_axis, goal_matrix @ local_axis
    tilt = torch.atan2(torch.linalg.cross(a, b).norm(), (a * b).sum())
    expected = torch.autograd.grad(tilt.square(), raw)[0]
    torch.testing.assert_close(value, tilt.square(), atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("axis", [[0, 0, 0], [1, 2], [math.nan, 0, 1], [math.inf, 0, 1]])
def test_invalid_axis(device_cfg: DeviceCfg, axis: list[float]) -> None:
    """Reject undefined axes before updating any captured solver buffers."""
    with pytest.raises(ValueError, match="orientation_axis"):
        ToolPoseCriteria.hold_axis(axis, 0.01, device_cfg)


def test_incompatible_weights(device_cfg: DeviceCfg) -> None:
    """Axis alignment uses one isotropic rotation weight, not an Euler-axis mask."""
    with pytest.raises(ValueError, match="rotation weights"):
        ToolPoseCriteria(
            orientation_axis=[0, 0, 1],
            terminal_pose_axes_weight_factor=[1, 1, 1, 1, 1, 0],
            device_cfg=device_cfg,
        )


@pytest.mark.parametrize("tolerance", [0, -0.1, math.nan, math.inf, math.pi])
def test_invalid_axis_tolerance(device_cfg: DeviceCfg, tolerance: float) -> None:
    """Reject undefined or unbounded angular tolerances."""
    with pytest.raises(ValueError, match="tolerance"):
        ToolPoseCriteria.hold_axis([0, 0, 1], tolerance, device_cfg)


def test_criteria_copy_and_graph_replay(device_cfg: DeviceCfg) -> None:
    """Switch between full orientation and axis alignment without recapturing the graph."""
    full = ToolPoseCriteria(device_cfg=device_cfg)
    axis = ToolPoseCriteria.hold_axis([0, 0, 2], device_cfg=device_cfg)
    copied = axis.clone()
    assert copied.orientation_axis.data_ptr() != axis.orientation_axis.data_ptr()
    torch.testing.assert_close(copied.orientation_axis, device_cfg.to_device([0, 0, 1]))
    copied.copy_(full)
    assert copied.orientation_axis is None
    copied.copy_(axis)
    torch.testing.assert_close(copied.orientation_axis, axis.orientation_axis)
    cost = make_cost(full)
    current = rotation([0, 0, 1], math.pi, device_cfg)
    goal = rotation([0, 0, 1], 0, device_cfg)
    for _ in range(3):
        evaluate(cost, current, goal)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        value, _ = evaluate(cost, current, goal)
    graph.replay()
    assert float(value) > 9
    pointer = cost._stacked_tool_pose_criteria.orientation_axis.data_ptr()
    cost.update_tool_pose_criteria({"cup": axis})
    graph.replay()
    assert float(value) == 0
    cost.update_tool_pose_criteria({"cup": full})
    graph.replay()
    assert float(value) > 9
    assert cost._stacked_tool_pose_criteria.orientation_axis.data_ptr() == pointer


def test_mixed_tools_and_goalsets(device_cfg: DeviceCfg) -> None:
    """Apply independent axis/full-orientation modes with batched alternative goals."""
    config = ToolPoseCostCfg(weight=[0, 1], tool_frames=["cup", "gripper"], device_cfg=device_cfg)
    cost = ToolPoseCost(config)
    cost.setup_batch_tensors(2, 3)
    cost.update_tool_pose_criteria(
        {
            "cup": ToolPoseCriteria.hold_axis([0, 0, 1], device_cfg=device_cfg),
            "gripper": ToolPoseCriteria.hold_axis([1, 0, 0], device_cfg=device_cfg),
        }
    )
    quaternion = device_cfg.to_device([0, 0, 0, 1]).repeat(2, 3, 2, 1)
    goal_quaternion = device_cfg.to_device([1, 0, 0, 0]).repeat(2, 1, 2, 2, 1)
    # Only the second environment has a matching alternative for the second tool.
    goal_quaternion[1, :, 1, 1] = device_cfg.to_device([0, 0, 0, 1])
    current = ToolPose(["cup", "gripper"], torch.zeros_like(quaternion[..., :3]), quaternion)
    goal = GoalToolPose(
        ["cup", "gripper"],
        torch.zeros_like(goal_quaternion[..., :3]),
        goal_quaternion,
    )
    value, _, angle, indices = cost.forward(
        current,
        goal,
        idxs_goal=torch.tensor([[0], [1]], device=device_cfg.device, dtype=torch.int32),
    )
    assert bool((value[0, :, 1] == 0).all())
    assert bool((angle[0, :, 1] > 3).all())
    assert bool((indices[1, :, 1] == 1).all())
    assert bool((value[1] == 0).all())


@pytest.mark.parametrize("offset", [0.0, 0.2])
def test_axis_joint_gradient(device_cfg: DeviceCfg, offset: float) -> None:
    """Check the full quaternion-to-joint gradient against FK finite differences."""
    with make_planner() as planner:
        cost = make_cost(ToolPoseCriteria.hold_axis([0, 0, 1], device_cfg=device_cfg))
        goal = rotation([0, 1, 0], 0.4, device_cfg)
        joints = planner.default_joint_state.position.detach().clone().reshape(1, -1)
        joints += offset * device_cfg.to_device([1, -0.3, 0.5, 0.1, 0.4, -0.2, 0.6])
        joints.requires_grad_()

        def loss(q: torch.Tensor) -> torch.Tensor:
            state = planner.kinematics.compute_kinematics(
                JointState.from_position(
                    q,
                    joint_names=planner.joint_names,
                )
            )
            return evaluate(cost, state.tool_poses.quaternion.reshape(4), goal)[0]

        gradient = torch.autograd.grad(loss(joints), joints)[0]
        numeric = torch.zeros_like(joints)
        for index in range(planner.action_dim):
            plus, minus = joints.detach().clone(), joints.detach().clone()
            plus[0, index] += 0.001
            minus[0, index] -= 0.001
            high, low = float(loss(plus)), float(loss(minus))
            numeric[0, index] = (high - low) / 0.002
        torch.testing.assert_close(gradient, numeric, atol=0.002, rtol=0.002)


def test_spatial_gradient_adjoint(device_cfg: DeviceCfg) -> None:
    """The analytical seed-IK path uses the same spatial convention as FK backward."""
    q = rotation([1, 2, -1], 1.3, device_cfg)
    spatial = device_cfg.to_device([0.3, -0.7, 0.2])
    gradient = 2 * multiply(torch.cat((spatial.new_zeros(1), spatial)), q)
    actual = quaternion_rate_to_axis_angle_rate(gradient, q)
    torch.testing.assert_close(actual, spatial, atol=2e-7, rtol=2e-7)
