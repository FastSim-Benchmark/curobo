# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Request lifecycle tests; these do not establish physical insertion success."""

from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from curobo._src.motion import motion_contact as module


@pytest.mark.parametrize("links", [["finger"], ([],), ("",), (" finger",), ("a", "a")])
def test_invalid_link_declarations_fail_explicitly(links: object) -> None:
    """Reject malformed input before accessing the planner."""
    with pytest.raises(ValueError, match="stationary_contact_links"):
        with module.stationary_contact_scope(None, None, links, "none"):
            pytest.fail("invalid declaration was accepted")


def test_empty_policy_leaves_existing_planner_untouched() -> None:
    """The default does not access or mutate any solver state."""
    with module.stationary_contact_scope(None, None, (), "none"):
        pass


@pytest.mark.parametrize("raise_inside", [False, True])
def test_contact_scope_restores_both_solvers(monkeypatch, raise_inside: bool) -> None:
    """Restore contacts and invalidate graphs on success and planning exception."""
    costs = [NS(_contact=None, config=NS(scene_collision_checker=None, num_spheres=1))
             for _ in range(2)]
    solvers = [NS(core=NS(invalidate_parameter_graphs=Mock()), cost=cost) for cost in costs]
    planner = NS(ik_solver=solvers[0], trajopt_solver=solvers[1])
    capture = Mock(return_value=object())
    constraint = object()
    monkeypatch.setattr(module, "capture_contact", capture)
    monkeypatch.setattr(module, "scene_costs", lambda solver: [solver.cost])
    monkeypatch.setattr(module, "StationaryContactSet", lambda *args: constraint)
    try:
        with module.stationary_contact_scope(planner, None, ("finger",), "none"):
            assert all(cost._contact is constraint for cost in costs)
            if raise_inside:
                raise RuntimeError("planning failed")
    except RuntimeError as error:
        assert raise_inside and str(error) == "planning failed"
    assert all(cost._contact is None for cost in costs)
    assert all(solver.core.invalidate_parameter_graphs.call_count == 2 for solver in solvers)
    capture.assert_called_once_with(planner, None, terminal=False, contact_links=("finger",))


@pytest.mark.parametrize('mode', ['end', 'both'])
def test_boundary_contact_combination_is_not_silently_ignored(monkeypatch, mode: str) -> None:
    """Unsupported combinations must fail before installing any exemption."""
    monkeypatch.setattr(module, "capture_contact", lambda *args, **kwargs: object())
    with pytest.raises(ValueError, match="cannot yet combine"):
        with module.stationary_contact_scope(None, None, ("finger",), mode):
            pytest.fail("unsupported combination was accepted")


@pytest.mark.parametrize('raise_inside', [False, True])
def test_stationary_and_departure_scopes_are_partitioned_and_restored(monkeypatch, raise_inside: bool) -> None:
    """IK retains holding contacts; TrajOpt enforces both disjoint contact rules."""
    class Holding:
        def __init__(self, *args):
            self.contacts = (NS(declaration=NS(sphere_indices=(0,))),)
    costs = [NS(_contact=None, config=NS(scene_collision_checker=NS(data=NS(meshes=None)),
                                       num_spheres=2)) for _ in range(2)]
    solvers = [NS(core=NS(invalidate_parameter_graphs=Mock()), cost=c) for c in costs]
    planner = NS(ik_solver=solvers[0], trajopt_solver=solvers[1],
                 scene_collision_checker=NS(data=NS(meshes=None)))
    capture = Mock(side_effect=[object(), object()])
    combined = object()
    combine = Mock(return_value=combined)
    monkeypatch.setattr(module, 'capture_contact', capture)
    monkeypatch.setattr(module, 'scene_costs', lambda solver: [solver.cost])
    monkeypatch.setattr(module, 'StationaryContactSet', Holding)
    monkeypatch.setattr(module, 'ContactDepartureSet', combine)
    try:
        with module.stationary_contact_scope(planner, None, ('holding',), 'start'):
            stationary = [c._contact for c in costs]
            with module.boundary_contact_scope(planner, None, None, 'start'):
                assert costs[0]._contact is stationary[0]
                assert costs[1]._contact is combined
                if raise_inside:
                    raise RuntimeError('abort combined query')
            assert [c._contact for c in costs] == stationary
    except RuntimeError as error:
        assert raise_inside and str(error) == 'abort combined query'
    assert all(c._contact is None for c in costs)
    assert capture.call_args_list[1].kwargs['excluded_sphere_indices'] == (0,)
    assert combine.call_args.kwargs['stationary'] is stationary[1]
    assert all(s.core.invalidate_parameter_graphs.call_count == 4 for s in solvers)


def test_no_captured_contact_preserves_normal_collision_checks(monkeypatch) -> None:
    """A named link alone does not install an exemption."""
    monkeypatch.setattr(module, "capture_contact", lambda *args, **kwargs: None)
    with module.stationary_contact_scope(None, None, ("finger",), "none"):
        pass


def test_preexisting_contact_is_preserved_after_partial_setup(monkeypatch) -> None:
    """Do not replace another contact policy, including on partial setup failure."""
    existing = object()
    costs = [NS(_contact=None, config=NS(scene_collision_checker=None, num_spheres=1)),
             NS(_contact=existing)]
    solvers = [NS(core=NS(invalidate_parameter_graphs=Mock()), cost=cost) for cost in costs]
    planner = NS(ik_solver=solvers[0], trajopt_solver=solvers[1])
    monkeypatch.setattr(module, "capture_contact", lambda *args, **kwargs: object())
    monkeypatch.setattr(module, "scene_costs", lambda solver: [solver.cost])
    monkeypatch.setattr(module, "StationaryContactSet", lambda *args: object())
    with pytest.raises(ValueError, match="cannot override"):
        with module.stationary_contact_scope(planner, None, ("finger",), "none"):
            pytest.fail("existing contact was overwritten")
    assert costs[0]._contact is None
    assert costs[1]._contact is existing


@pytest.mark.parametrize('use_cuda_graph', [False, True])
def test_native_planner_holds_one_link_and_departs_with_another(tmp_path, use_cuda_graph: bool) -> None:
    """Exercise the real capture, IK, TrajOpt and graph paths, retaining defaults."""
    import torch
    from curobo.examples.reference.contact_separation import robot_config, scene_config
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    from curobo.types import GoalToolPose, JointState
    robot = robot_config(tmp_path)
    robot['robot_cfg']['kinematics']['collision_spheres']['base'] = [
        {'center': [-.2, 0., .049], 'radius': .05}]
    with MotionPlanner(MotionPlannerCfg.create(robot, scene_model=scene_config(False),
        use_cuda_graph=use_cuda_graph, num_ik_seeds=8, num_trajopt_seeds=4)) as planner:
        start = JointState.from_position(planner.device_cfg.to_device([[0., 0., .12]]),
                                         joint_names=planner.joint_names)
        planner.attachment_manager.update(planner.device_cfg.to_device([[0., 0., -.08, .041]]), start)
        end = JointState.from_position(planner.device_cfg.to_device([[0., 0., .2]]),
                                       joint_names=planner.joint_names)
        pose = planner.compute_kinematics(end).tool_poses.get_link_pose('gripper')
        result = planner.plan_pose(GoalToolPose.from_poses({'gripper': pose}, num_goalset=1),
            start, max_attempts=2, allow_boundary_collision='start', stationary_contact_links=('base',))
        assert result is not None and bool(result.success.any())
        assert all(c._contact is None for solver in (planner.ik_solver, planner.trajopt_solver)
                   for c in module.scene_costs(solver))
        ordinary = planner.plan_pose(GoalToolPose.from_poses({'gripper': pose}, num_goalset=1),
                                      start, max_attempts=1)
        assert ordinary is None or not bool(ordinary.success.any())
