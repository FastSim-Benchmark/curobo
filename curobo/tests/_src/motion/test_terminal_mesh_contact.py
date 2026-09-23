# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Terminal mesh contact uses selective replacement and the bounded approach rule."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from curobo._src.collision.contact_approach import GoalContact
from curobo._src.cost.cost_scene_collision import SceneCollisionCost
from curobo._src.cost.cost_scene_collision_cfg import SceneCollisionCostCfg
from curobo._src.geom.collision.collision_scene import SceneCollision, SceneCollisionCfg
from curobo._src.geom.types import Cuboid, SceneCfg
from curobo._src.motion.motion_contact import capture_contact, scene_costs
from curobo.examples.reference.contact_separation import box_clearance, robot_config
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import DeviceCfg, JointState


@pytest.fixture(params=["cuboid", "mesh"])
def support_scene(request) -> SceneCollision:
    """Represent the same physical support in either native collision bucket."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
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


def capture(scene, height):
    """Run production contact discovery against the native scene and real BVH."""
    spheres = torch.tensor([[0, 0, height, 0.05]], device="cuda")
    planner = SimpleNamespace(
        scene_collision_checker=scene,
        compute_kinematics=lambda state: SimpleNamespace(robot_spheres=spheres),
    )
    return capture_contact(planner, torch.zeros((1, 1), device="cuda"), terminal=True)


@pytest.mark.parametrize("penetration", [0.0005, 0.001999])
def test_capture_terminal_support_below_two_mm(support_scene, penetration):
    """A mesh endpoint must create the same bounded goal declaration as a cuboid."""
    declaration = capture(support_scene, 0.05 - penetration)
    assert isinstance(declaration, GoalContact)
    assert declaration.obstacle_name == "table"
    assert declaration.sphere_indices == (0,)
    assert declaration.max_goal_penetration == 0.002


@pytest.mark.parametrize("penetration", [0.002001, 0.01])
def test_capture_terminal_support_above_two_mm_is_rejected(support_scene, penetration):
    """Contact discovery cannot turn excessive model overlap into an exemption."""
    with pytest.raises(ValueError, match="penetration"):
        capture(support_scene, 0.05 - penetration)


def test_clear_or_disabled_support_is_not_captured(support_scene):
    """Only a currently enabled touching support is eligible."""
    assert capture(support_scene, 0.07) is None
    support_scene.enable_obstacle("table", False)
    assert capture(support_scene, 0.049) is None


def collision_cost(scene, horizon, *, declaration=None, sweep=False, count=1):
    """Evaluate the actual collision manager rather than the replacement alone."""
    cfg = SceneCollisionCostCfg(
        weight=1.0,
        num_spheres=count,
        use_sweep=sweep,
        goal_contact=declaration,
    )
    cfg.scene_collision_checker = scene
    cost = SceneCollisionCost(cfg)
    cost.setup_batch_tensors(1, horizon)
    return cost


@pytest.mark.parametrize("sweep", [False, True])
@pytest.mark.parametrize(
    "heights, accepted",
    [
        ([0.07, 0.055, 0.049], True),
        ([0.07, 0.047, 0.049], False),
        ([0.07, 0.05, 0.06, 0.049], False),
        ([0.049, 0.055, 0.049], False),
    ],
)
def test_terminal_approach_replaces_only_monotonic_bounded_contact(
    support_scene,
    sweep,
    heights,
    accepted,
):
    """Preserve rejection of overshoot, rebound and contact at the start."""
    declaration = capture(support_scene, 0.049)
    assert isinstance(declaration, GoalContact)
    spheres = torch.tensor([[[[0, 0, z, 0.05]] for z in heights]], device="cuda")
    state = SimpleNamespace(robot_spheres=spheres)
    dt = torch.tensor([0.01], device="cuda")
    baseline = collision_cost(support_scene, len(heights), sweep=sweep)
    assert baseline.forward(state, trajectory_dt=dt)[0, -1, 0] > 0
    cost = collision_cost(support_scene, len(heights), declaration=declaration, sweep=sweep)
    result = cost.forward(state, trajectory_dt=dt)
    assert bool((result == 0).all()) is accepted


@pytest.mark.parametrize("sweep", [False, True])
def test_terminal_contact_keeps_other_spheres_and_obstacles_checked(support_scene, sweep):
    """The selected payload still hits the ceiling; an unselected finger hits the table."""
    declaration = GoalContact("table", (0,), ((0, 0, 0.049, 0.05),))
    cost = collision_cost(support_scene, 3, declaration=declaration, sweep=sweep, count=2)
    spheres = torch.tensor(
        [
            [
                [[0, 0, 0.16, 0.05], [0.2, 0, 0.07, 0.05]],
                [[0, 0, 0.06, 0.05], [0.2, 0, 0.06, 0.05]],
                [[0, 0, 0.049, 0.05], [0.2, 0, 0.04, 0.05]],
            ]
        ],
        device="cuda",
    )
    result = cost.forward(
        SimpleNamespace(robot_spheres=spheres),
        trajectory_dt=torch.tensor([0.01], device="cuda"),
    )
    assert result[0, 0, 0] > 0
    assert result[0, -1, 0] == 0
    assert result[0, -1, 1] > 0
    for data in (support_scene.data.cuboids, support_scene.data.meshes):
        if data is not None:
            assert all(data.enable[0, i] == 1 for i, name in enumerate(data.names[0]) if name)


@pytest.mark.parametrize("height, accepted", [(0.049, True), (0.048, False), (0.06, False)])
def test_terminal_only_ik_keeps_exact_goal_geometry(support_scene, height, accepted):
    """Single-state IK contact accepts only its captured terminal geometry."""
    declaration = GoalContact("table", (0,), ((0, 0, 0.049, 0.05),), terminal_only=True)
    cost = collision_cost(support_scene, 1, declaration=declaration)
    spheres = torch.tensor([[[[0, 0, height, 0.05]]]], device="cuda")
    assert bool((cost.forward(SimpleNamespace(robot_spheres=spheres)) == 0).all()) is accepted


@pytest.mark.parametrize("kind", ["pose", "joints"])
def test_native_end_mesh_contact_is_scoped_and_bounded(tmp_path: Path, kind: str):
    """The public per-call end policy works through mesh IK and native TrajOpt."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    table = Cuboid(name="support", dims=[1, 1, 0.1], pose=[0, 0, -0.05, 1, 0, 0, 0])
    device = DeviceCfg()
    config = MotionPlannerCfg.create(
        robot_config(tmp_path),
        scene_model=SceneCfg(mesh=[table.get_mesh()]),
        device_cfg=device,
        num_ik_seeds=8,
        num_trajopt_seeds=4,
        use_cuda_graph=True,
        position_tolerance=1e-5,
        interpolation_dt=0.01,
        interpolation_buffer_size=1200,
        random_seed=123,
    )
    planner = MotionPlanner(config)
    start = JointState.from_position(device.to_device([[0, 0, 0.2]]), planner.joint_names)
    goal = JointState.from_position(device.to_device([[0, 0, 0.1205]]), planner.joint_names)
    planner.attachment_manager.update(device.to_device([[0, 0, -0.08, 0.041]]), start)
    try:

        def plan(mode):
            options = dict(max_attempts=2, allow_boundary_collision=mode)
            if kind == "joints":
                return planner.plan_cspace(goal, start, **options)
            return planner.plan_pose(
                planner.compute_kinematics(goal).tool_poses.as_goal(),
                start,
                **options,
            )

        result = plan("end")
        assert result is not None and result.success.all()
        positions = result.get_interpolated_plan().position.reshape(-1, 3)
        torch.testing.assert_close(positions[-1], goal.position[0], atol=1e-5, rtol=0)
        fractions = torch.arange(8, device="cuda") / 8
        dense = (
            positions[:-1, None] + fractions[None, :, None] * positions.diff(dim=0)[:, None]
        ).reshape(-1, 3)
        dense = torch.cat((dense, positions[-1:]))
        spheres = planner.compute_kinematics(
            JointState.from_position(dense, joint_names=planner.joint_names),
        ).robot_spheres.reshape(len(dense), -1, 4)
        gaps = box_clearance(spheres.cpu(), table)
        enabled = spheres[0, :, 3].cpu() > 0
        assert gaps[:, enabled].min() >= -0.00052
        assert gaps[-1, enabled].min() == pytest.approx(-0.0005, abs=2e-5)
        for solver in (planner.ik_solver, planner.trajopt_solver):
            assert all(cost._contact is None for cost in scene_costs(solver))
        assert planner.scene_collision_checker.data.meshes.enable[0, 0] == 1
        ordinary = plan("none")
        assert ordinary is None or not ordinary.success.any()
    finally:
        planner.destroy()
