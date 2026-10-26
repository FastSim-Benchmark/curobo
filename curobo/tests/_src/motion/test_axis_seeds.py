"""CPU contracts for continuous seeds; native geometry is checked separately."""
from contextlib import nullcontext
from types import SimpleNamespace as NS

import pytest
import torch

from curobo._src.motion import axis_seeds as module
from curobo._src.state.state_joint import JointState
from curobo._src.types.tool_pose import GoalToolPose


def fixture(fail_at=None, raises_at=None):
    """Make a deterministic two-joint solver with one held coordinate."""
    bounds = torch.tensor([[0., 0.], [.3, 0.]])
    active = bounds.clone()
    calls = []
    names = ["translation_without_robot_specific_name", "held_joint"]
    start = JointState.from_position(torch.zeros(1, 2), names)
    poses = GoalToolPose(["tool"], torch.zeros(1, 1, 1, 1, 3),
                         torch.tensor([[[[[1., 0., 0., 0.]]]]]))
    positions = torch.tensor([[[[[.25, 0., 0.], [.16, 0., 0.]]]]])
    goals = GoalToolPose(["tool"], positions, poses.quaternion.repeat(1, 1, 1, 2, 1))

    def update(*, position):
        active.copy_(position)
        assert bool((active[0] >= bounds[0]).all())
        assert bool((active[1] <= bounds[1]).all())

    def solve(target, **kwargs):
        calls.append(active.clone())
        if len(calls) == raises_at:
            raise RuntimeError("injected native failure")
        solution = torch.zeros(1, 2, 2)
        solution[:, :, 0] = target.position.flatten()[0]
        assert bool((solution >= active[0]).all() & (solution <= active[1]).all())
        return NS(solution=solution, success=torch.full((1, 2), len(calls) != fail_at),
                  total_time=.01, solve_time=.005)

    kin = NS(get_joint_limits=lambda: NS(position=active), config=NS(
        kinematics_config=NS(joint_map=torch.tensor([0, 1]),
                             joint_map_type=torch.tensor([6, 5]))))
    planner = NS(joint_names=names, kinematics=kin, update_joint_limits=update,
                 compute_kinematics=lambda state: NS(tool_poses=poses),
                 ik_solver=NS(config=NS(num_seeds=2), solve_pose=solve),
                 trajopt_solver=NS(config=NS(num_seeds=2), action_horizon=16),
                 config=NS(trajopt_finetune_attempts=None))
    return planner, goals, start, active, bounds, calls


def test_continuation_keeps_physical_bounds_locked_joints_and_goal_pairing():
    """Verify the named continuous-path contract without a GPU."""
    planner, goals, start, active, bounds, calls = fixture()
    target, endpoint, seeds, record = module.continuous_axis_seed(
        planner, goals, start, {"tool": object()},
    )
    assert len(calls) == 16
    assert record["original_goal_index"] == 1
    assert torch.equal(target.position, goals.position[..., 1:2, :])
    assert torch.equal(endpoint, torch.tensor([.16, 0.]))
    assert seeds.shape == (1, 2, 16, 2)
    assert torch.equal(active, bounds)
    assert not bool(seeds[..., 1].any())


@pytest.mark.parametrize("fail_at", [1, 8, 16])
def test_no_ik_solution_returns_no_seed_and_restores_limits(fail_at):
    """Verify the named continuous-path contract without a GPU."""
    planner, goals, start, active, bounds, calls = fixture(fail_at=fail_at)
    assert module.continuous_axis_seed(planner, goals, start, {"tool": object()}) is None
    assert len(calls) == fail_at
    assert torch.equal(active, bounds)


def test_native_exception_is_not_hidden_and_restores_limits():
    """Verify the named continuous-path contract without a GPU."""
    planner, goals, start, active, bounds, _ = fixture(raises_at=5)
    with pytest.raises(RuntimeError, match="injected native failure"):
        module.continuous_axis_seed(planner, goals, start, {"tool": object()})
    assert torch.equal(active, bounds)


def test_expired_budget_is_not_hidden_and_restores_limits(monkeypatch):
    """Verify the named continuous-path contract without a GPU."""
    planner, goals, start, active, bounds, calls = fixture()

    def budget():
        if len(calls) == 4:
            raise TimeoutError("expired")

    monkeypatch.setattr(module, "check_time_budget", budget)
    with pytest.raises(TimeoutError, match="expired"):
        module.continuous_axis_seed(planner, goals, start, {"tool": object()})
    assert torch.equal(active, bounds)


@pytest.mark.parametrize("success", [True, False])
def test_native_trajectory_acceptance_preserved_and_goal_index_remapped(success):
    """Verify the named continuous-path contract without a GPU."""
    planner, goals, start, active, bounds, _ = fixture()

    def solve(target, current, **kwargs):
        assert torch.equal(active, bounds)
        assert kwargs["seed_traj"].shape == (1, 2, 16, 2)
        return NS(success=torch.tensor([[success]]), goalset_index=torch.tensor([[[0]]]),
                  total_time=.1, solve_time=.05, debug_info={})

    planner.trajopt_solver.solve_pose = solve
    result = module.plan_continuous_axis_seed(planner, goals, start, {"tool": object()}, True)
    assert result.success.item() is success
    assert result.goalset_index.item() == 1
    assert result.debug_info["continuous_axis_seed"]["ik_steps"] == 16


def test_resampling_preserves_endpoints_and_segment_bounds():
    """Verify the named continuous-path contract without a GPU."""
    path = torch.tensor([[0., 1.], [.1, .8], [.2, .4]])
    seed = module.resample_seed(path, 18, 3)
    assert torch.equal(seed[0, 0, 0], path[0])
    assert torch.equal(seed[0, 0, -1], path[-1])
    assert bool((seed >= path.min(0).values).all() & (seed <= path.max(0).values).all())


@pytest.mark.parametrize("kind", [0, 1, 2, 6, 7, 8])
def test_all_prismatic_axes_have_translation_radius(kind):
    """Verify the named continuous-path contract without a GPU."""
    kin = NS(config=NS(kinematics_config=NS(joint_map=torch.tensor([0]),
                                            joint_map_type=torch.tensor([kind]))))
    assert module.local_search_radius(kin, torch.zeros(1)).item() == pytest.approx(.02)


@pytest.mark.parametrize("held", [False, True])
@pytest.mark.parametrize("success", [False, True])
def test_public_dispatch_preserves_ordinary_search_after_seed_rejection(
    monkeypatch, held, success,
):
    """Only a native successful result can replace the ordinary search."""
    from curobo._src.motion.motion_planner import MotionPlanner

    calls = []
    planned = NS(success=torch.tensor([success]))
    ordinary = object()
    monkeypatch.setattr(module, "plan_continuous_axis_seed",
                        lambda *args: calls.append("continuous") or planned)
    planner = MotionPlanner.__new__(MotionPlanner)
    planner._destroyed = True
    planner._hold_axis_scope = lambda *args: nullcontext()
    planner._plan_pose = lambda *args: calls.append("ordinary") or ordinary
    result = planner.plan_pose(NS(), NS(), hold_axis={"tool": object()} if held else None)
    assert result is (planned if held and success else ordinary)
    assert calls == (["continuous"] if held and success else
                     ["continuous", "ordinary"] if held else ["ordinary"])


@pytest.mark.parametrize("mode", ["end", "both"])
def test_terminal_contacts_keep_their_existing_planner(monkeypatch, mode):
    """The non-contact seed does not bypass the endpoint-contact scope."""
    from curobo._src.motion.motion_planner import MotionPlanner

    monkeypatch.setattr(module, "plan_continuous_axis_seed",
                        lambda *args: pytest.fail("contact query must not use this seed"))
    monkeypatch.setattr("curobo._src.motion.contact_axis_seeds.plan_contact_axis_seed",
                        lambda *args, **kwargs: None)
    expected = NS(debug_info={})
    monkeypatch.setattr("curobo._src.motion.motion_contact_goalset.plan_terminal_pose",
                        lambda *args: expected)
    planner = MotionPlanner.__new__(MotionPlanner)
    planner._destroyed = True
    planner._hold_axis_scope = lambda *args: nullcontext()
    assert planner.plan_pose(NS(), NS(), hold_axis={"tool": object()},
                             allow_boundary_collision=mode) is expected
