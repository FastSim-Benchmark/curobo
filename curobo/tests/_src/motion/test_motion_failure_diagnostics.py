"""Diagnostic mesh samples retain joint, sphere and obstacle identity."""

import os
from types import SimpleNamespace as NS

import pytest
import torch

from curobo._src.collision.contact_mesh import MeshClearance
from curobo._src.motion.motion_failure_diagnostics import (
    _approach_gap_summary,
    _departure_gap_summary,
    terminal_failure_summary,
)
from curobo._src.state.state_joint import JointState


@pytest.mark.parametrize(
    "gaps, rebound, undershoot",
    [([.02, .006, -.001], 0., 0.),
     ([.02, .001, .004, -.001], .003, 0.),
     ([.02, -.003, -.001], .002, .002)],
)
def test_approach_diagnostic_separates_rebound_from_penetration(
    gaps: list[float], rebound: float, undershoot: float
) -> None:
    """A clear but nonmonotonic approach must not be reported as penetration."""
    separation = NS(
        indices=torch.tensor([1]),
        initial_clearance=torch.tensor([-.001]),
        initial_spheres=torch.tensor([[0., 0., -.001, .05]]),
        declaration=NS(release_clearance=.005, numerical_tolerance=.00001),
        interpolate_samples=lambda value: value,
        clearance=lambda value: value[..., 2],
    )
    contact = NS(separation=separation, declaration=NS(obstacle_name="support"))
    spheres = torch.zeros(1, len(gaps), 2, 4)
    spheres[0, :, 1, 2] = torch.tensor(gaps)
    spheres[..., 3] = .05
    before = spheres.clone()
    row = _approach_gap_summary(contact, spheres, 0)["spheres"][0]
    assert row["maximum_approach_rebound_m"] == pytest.approx(rebound, abs=1e-7)
    assert row["maximum_below_goal_gap_m"] == pytest.approx(undershoot, abs=1e-7)
    assert row["minimum_gap_m"] == pytest.approx(min(gaps))
    assert row["goal_gap_m"] == pytest.approx(-.001)
    assert row["sphere"] == 1
    torch.testing.assert_close(before, spheres)


@pytest.mark.parametrize("kind", ["optimized", "interpolated"])
def test_mesh_sample_uses_selected_peak_and_masks_disabled_spheres(monkeypatch, kind):
    state = JointState.from_position(torch.arange(6.0).reshape(2, 3, 1), joint_names=["base"])
    dense = JointState.from_position(
        torch.arange(10.0, 16.0).reshape(2, 3, 1), joint_names=["base"]
    )
    spheres = torch.tensor([[0.0, 0, 0, 0.1], [1.0, 0, 0, -1.0]])
    meshes = NS(
        enable=torch.tensor([[True, False, True]]), names=[["table", "disabled", "machine"]],
        inv_pose=torch.tensor([[[1., 2., 3., 1., 0., 0., 0., 99.],
                                [4., 5., 6., 1., 0., 0., 0., 99.],
                                [7., 8., 9., .5, .5, .5, .5, 99.]]]),
        use_signed_distance=torch.tensor([[False, True, True]]),
        dims=torch.tensor([[[1., 2., 3., 99.], [4., 5., 6., 99.], [7., 8., 9., 99.]]]),
    )

    def compute(selected):
        assert selected.position.tolist() == [[4.0 if kind == "optimized" else 14.0]]
        return NS(robot_spheres=spheres)

    def gaps(selected, scene, indices):
        assert scene is meshes and indices.tolist() == [0, 2]
        return torch.tensor([[0.02, -0.003], [-10.0, -10.0]])

    monkeypatch.setattr(MeshClearance, "apply", gaps)
    planner = NS(
        kinematics=NS(get_active_js=lambda value: value),
        scene_collision_checker=NS(data=NS(meshes=meshes)),
        compute_kinematics=compute,
        trajopt_solver=NS(metrics_rollout=NS(metrics_constraint_manager=None),
                          additional_metrics_rollouts={
                              "interpolated_rollout": NS(metrics_constraint_manager=None)}),
        attachment_manager=NS(
            kinematics_params=NS(
                link_name_to_idx_map={"finger": 0, "unused": 1},
                link_sphere_idx_map=torch.tensor([0, 1]),
            )
        ),
    )
    result = NS(
        js_solution=state,
        interpolated_trajectory=dense,
        debug_info={
            "selected_constraint_maxima": {
                kind: {"scene_collision": {"maximum_step": 1, "maximum_trajectory": 1}}
            }
        },
    )
    before = state.position.clone()
    dense_before = dense.position.clone()
    spheres_before = spheres.clone()
    pose_before = meshes.inv_pose.clone()
    sample = terminal_failure_summary(planner, result, trajectory_kind=kind)["raw_mesh_gap_sample"]
    assert sample["trajectory"] == sample["step"] == 1
    assert sample["joint_names"] == ["base"]
    assert sample["joint_position"] == [4.0 if kind == "optimized" else 14.0]
    assert sample["selected_endpoint"] == {
        "step": 2, "joint_position": [5.0 if kind == "optimized" else 15.0]
    }
    assert sample["sphere_frame"] == "kinematics_model"
    assert sample["pairs"][0]["obstacle"] == "machine"
    assert sample["pairs"][0]["link"] == "finger"
    assert sample["pairs"][0]["raw_gap_m"] < 0
    first = sample["pairs"][0]
    torch.testing.assert_close(torch.tensor(first["sphere_xyzr"]), spheres[0])
    assert first["mesh_environment_index"] == 0
    assert first["mesh_index"] == 2
    assert first["mesh_inverse_pose_xyz_wxyz"] == [7., 8., 9., .5, .5, .5, .5]
    assert first["mesh_use_signed_distance"] is True
    assert first["mesh_dims_m"] == [7., 8., 9.]
    assert sample["pairs"][1]["mesh_index"] == 0
    assert sample["pairs"][1]["mesh_use_signed_distance"] is False
    assert sample["contact_allowance_applied"] is False
    assert len(sample["pairs"]) == 2
    torch.testing.assert_close(before, state.position)
    torch.testing.assert_close(dense_before, dense.position)
    torch.testing.assert_close(spheres_before, spheres)
    torch.testing.assert_close(pose_before, meshes.inv_pose)


def test_missing_dense_state_is_not_replaced_with_optimized_state():
    result = NS(js_solution=JointState.from_position(torch.zeros(1, 2, 1)),
                debug_info={"selected_constraint_maxima": {"interpolated": {
                    "scene_collision": {"maximum_step": 1, "maximum_trajectory": 0}}}})
    evidence = terminal_failure_summary(None, result, trajectory_kind="interpolated")
    assert evidence["trajectory_state_available"] is False
    assert "raw_mesh_gap_sample" not in evidence
    with pytest.raises(ValueError, match="trajectory_kind"):
        terminal_failure_summary(None, result, trajectory_kind="unknown")


@pytest.mark.parametrize(
    "gaps,rebound,below",
    [([-.001, 0., .006], 0., 0.),
     ([-.001, .00004, .00001, .006], .00003, 0.),
     ([-.001, -.00103, .006], .00003, .00003),
     ([-.001, .02, .007], 0., 0.)],
)
def test_departure_reports_raw_rebound_without_relaxing_tolerance(gaps, rebound, below):
    contact = NS(
        indices=torch.tensor([1]), initial_clearance=torch.tensor([-.001]),
        initial_spheres=torch.tensor([[0., 0., -.001, .05]]),
        declaration=NS(obstacle_name="support", release_clearance=.005,
                       numerical_tolerance=.00001),
        interpolate_samples=lambda x: x, clearance=lambda x: x[..., 2],
    )
    spheres = torch.zeros(1, len(gaps), 2, 4)
    spheres[..., 3] = .05
    spheres[0, :, 1, 2] = torch.tensor(gaps)
    before = spheres.clone()
    report = _departure_gap_summary(contact, spheres, 0)
    row = report["spheres"][0]
    assert row["maximum_departure_rebound_m"] == pytest.approx(rebound, abs=1e-9)
    assert row["maximum_below_initial_gap_m"] == pytest.approx(below, abs=1e-9)
    assert row["minimum_gap_m"] == pytest.approx(min(gaps))
    assert report["numerical_tolerance_m"] == .00001
    assert report["acceptance_modified"] is False
    assert report["start_sphere_mismatch_m"] == 0.
    torch.testing.assert_close(spheres, before)


def test_missing_selected_metric_is_explicit_without_geometry_query():
    assert terminal_failure_summary(None, NS(debug_info={})) == {
        "constraints": {},
        "joint_bounds": {},
    }


@pytest.mark.parametrize("padding, expected", [(0.0, 0), (0.02, 1)])
def test_self_collision_uses_enabled_pairs_and_distinguishes_padding(padding, expected):
    state = JointState.from_position(torch.arange(6.0).reshape(2, 3, 1), joint_names=["joint"])
    spheres = torch.tensor([[0., 0, 0, .1], [.21, 0, 0, .1], [0., 0, 0, -1.]])
    config = NS(collision_pairs=torch.tensor([[0, 1], [0, 2]]),
                sphere_padding=torch.tensor([padding, padding, 10.]))
    metrics = NS(has_cost=lambda name: name == "self_collision",
                 get_cost=lambda name: NS(config=NS(self_collision_kin_config=config)))
    sampled = []

    def compute(selected):
        sampled.append(selected.position.item())
        return NS(robot_spheres=spheres)

    planner = NS(
        kinematics=NS(get_active_js=lambda value: value),
        compute_kinematics=compute,
        trajopt_solver=NS(metrics_rollout=NS(metrics_constraint_manager=metrics)),
        attachment_manager=NS(kinematics_params=NS(
            link_name_to_idx_map={"finger": 0, "attached_object": 1, "disabled": 2},
            link_sphere_idx_map=torch.tensor([0, 1, 2]))),
    )
    result = NS(js_solution=state, debug_info={"selected_constraint_maxima": {
        "optimized": {"self_collision": {"maximum_trajectory": 1, "maximum_step": 2}}}})
    before = spheres.clone()
    summary = terminal_failure_summary(planner, result)["raw_self_collision_sample"]
    assert sampled == [3., 5.]
    assert summary["acceptance_modified"] is False
    assert summary["joint_names"] == ["joint"]
    assert summary["sphere_frame"] == "kinematics_model"
    assert [sample["joint_position"] for sample in summary["samples"]] == [[3.], [5.]]
    for sample in summary["samples"]:
        assert sample["collision_pair_count"] == expected
        assert len(sample["pairs"]) == expected
        if expected:
            pair = sample["pairs"][0]
            assert pair["links"] == ["finger", "attached_object"]
            assert pair["raw_overlap_m"] == pytest.approx(-.01)
            assert pair["padded_overlap_m"] == pytest.approx(.03)
            torch.testing.assert_close(torch.tensor(pair["sphere_xyzr"]), spheres[:2])
    torch.testing.assert_close(before, spheres)


@pytest.mark.parametrize("kind", ["optimized", "interpolated"])
@pytest.mark.parametrize("layout", ["full", "permuted_full", "native"])
def test_reduced_diagnostics_project_restored_columns_without_mutating_results(kind, layout):
    """Real SDK augmentation and active-state projection precede diagnostic FK."""
    from curobo._src.robot.kinematics.kinematics import Kinematics
    from curobo._src.state.state_joint_ops import augment_joint_state

    names = [f"active_{index}" for index in range(9)]
    held = [f"held_{index}" for index in range(34)]
    native = JointState.from_position(torch.arange(54.0).reshape(2, 3, 9), joint_names=names)
    locked = JointState.from_position(torch.arange(34.0) / 100.0, joint_names=held)
    full = augment_joint_state(native, names + held, locked)
    state = native if layout == "native" else full
    if layout == "permuted_full":
        state = full.reorder(list(reversed(full.joint_names)))
    state_names, state_position = list(state.joint_names), state.position.clone()
    receiver = NS(joint_names=names)
    receiver.get_active_js = lambda value: Kinematics.get_active_js(receiver, value)
    observed = []

    def compute(selected):
        assert selected.joint_names == names
        observed.append(selected.position.clone())
        return NS(robot_spheres=torch.tensor([[0.0, 0.0, 0.0, 0.1], [1.0, 0.0, 0.0, 0.1]]))

    config = NS(collision_pairs=torch.tensor([[0, 1]]), sphere_padding=torch.zeros(2))
    metrics = NS(
        has_cost=lambda name: name == "self_collision",
        get_cost=lambda name: NS(config=NS(self_collision_kin_config=config)),
    )
    rollout = NS(metrics_constraint_manager=metrics)
    planner = NS(
        kinematics=receiver,
        compute_kinematics=compute,
        trajopt_solver=NS(
            metrics_rollout=rollout, additional_metrics_rollouts={"interpolated_rollout": rollout}
        ),
        attachment_manager=NS(
            kinematics_params=NS(
                link_name_to_idx_map={"custom_tcp": 0, "payload": 1},
                link_sphere_idx_map=torch.tensor([0, 1]),
            )
        ),
    )
    maxima = {kind: {"self_collision": {"maximum_trajectory": 1, "maximum_step": 2}}}
    result = NS(
        js_solution=state,
        interpolated_trajectory=state,
        debug_info={"selected_constraint_maxima": maxima},
    )
    report = terminal_failure_summary(planner, result, trajectory_kind=kind)
    assert report["constraints"] is maxima
    sample = report["raw_self_collision_sample"]
    assert sample["joint_names"] == names
    assert sample["acceptance_modified"] is False
    assert [value.tolist() for value in observed] == [
        native.position[1, :1].tolist(),
        native.position[1, 2:3].tolist(),
    ]
    assert result.js_solution is state and result.interpolated_trajectory is state
    assert state.joint_names == state_names
    torch.testing.assert_close(state.position, state_position, rtol=0.0, atol=0.0)
    assert full.joint_names == names + held
    torch.testing.assert_close(full.position[..., 9:], locked.position.expand(2, 3, -1))


@pytest.mark.skipif(
    not os.environ.get("CUROBO_TEST_TERMINAL_LOCKED_NATIVE"),
    reason="opt-in real CUDA kinematics and dynamic custom TCP attachment regression",
)
@pytest.mark.parametrize("kind", ["optimized", "interpolated"])
def test_native_locked_tcp_failure_diagnostics_preserve_full_model(tmp_path, kind):
    """Use actual 43/9 SDK FK and dynamic object-to-custom-TCP attachment geometry."""
    import copy
    import json

    from curobo._src.collision.attachment_manager import AttachmentManager
    from curobo.kinematics import Kinematics, KinematicsCfg
    from curobo.types import DeviceCfg, Pose

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    config, active, held, locks = _locked_tcp_configuration(tmp_path)
    device = DeviceCfg(device="cuda:0")
    reduced = Kinematics(
        KinematicsCfg.from_robot_yaml_file(
            {"robot_cfg": {"kinematics": copy.deepcopy(config)}}, device_cfg=device
        )
    )
    full_config = copy.deepcopy(config)
    full_config["lock_joints"] = None
    complete = Kinematics(
        KinematicsCfg.from_robot_yaml_file(
            {"robot_cfg": {"kinematics": full_config}}, device_cfg=device
        )
    )
    assert len(reduced.joint_names) == 9 and set(reduced.joint_names) == set(active)
    assert len(complete.joint_names) == 43 and set(complete.joint_names) == set(active + held)
    managers = [AttachmentManager(model, device_cfg=device) for model in (reduced, complete)]
    pair_config = reduced.get_self_collision_config()
    metrics = NS(
        has_cost=lambda name: name == "self_collision",
        get_cost=lambda name: NS(config=NS(self_collision_kin_config=pair_config)),
    )
    rollout = NS(metrics_constraint_manager=metrics)
    planner = NS(
        kinematics=reduced,
        compute_kinematics=reduced.compute_kinematics,
        trajopt_solver=NS(
            metrics_rollout=rollout, additional_metrics_rollouts={"interpolated_rollout": rollout}
        ),
        attachment_manager=managers[0],
        scene_collision_checker=None,
    )
    expected_fixed = bool(os.environ.get("CUROBO_TEST_DIAGNOSTIC_EXPECT_FIXED"))
    fitted = torch.tensor([[0.01, 0.02, 0.03, 0.015], [0.04, -0.02, 0.01, 0.012]], device="cuda:0")
    phase_evidence = []
    for phase in range(2):
        values = {name: 0.01 * (i + phase + 1) for i, name in enumerate(active)}
        native = JointState.from_position(
            torch.tensor([[values[name] for name in reduced.joint_names]], device="cuda:0"),
            joint_names=list(reduced.joint_names),
        )
        restored = reduced.get_full_js(native)
        full_state = restored.reorder(complete.joint_names)
        offset = Pose(
            position=torch.tensor([[0.4 + phase * 0.02, 0.2, 0.5]], device="cuda:0"),
            quaternion=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda:0"),
        )
        for manager, state in zip(managers, (native, full_state)):
            manager.update(
                fitted, state, link_name="payload_slot", world_objects_pose_offset=offset
            )
        reduced_fk = reduced.compute_kinematics(native)
        complete_fk = complete.compute_kinematics(full_state)
        torch.testing.assert_close(
            reduced_fk.robot_spheres, complete_fk.robot_spheres, rtol=0.0, atol=1e-5
        )
        slots = managers[0].kinematics_params.get_sphere_index_from_link_name("payload_slot")
        actual_spheres = reduced_fk.robot_spheres.reshape(-1, 4)[slots.long()]
        torch.testing.assert_close(
            actual_spheres[:, :3], offset.position + fitted[:, :3], rtol=0.0, atol=1e-5
        )
        torch.testing.assert_close(actual_spheres[:, 3], fitted[:, 3], rtol=0.0, atol=0.0)
        phase_evidence.append(
            {
                "phase": phase,
                "native_names": list(reduced.joint_names),
                "full_names": list(complete.joint_names),
                "held_count": len(locks),
                "custom_tcp": "custom_tcp",
                "dynamic_payload_spheres": actual_spheres.tolist(),
                "max_full_reduced_sphere_error": float(
                    (reduced_fk.robot_spheres - complete_fk.robot_spheres).abs().max()
                ),
            }
        )
        sequence = restored if kind == "optimized" else native
        sequence = JointState.from_position(
            sequence.position[:, None, :].repeat(1, 3, 1),
            joint_names=list(sequence.joint_names),
        )
        before_names, before_position = list(sequence.joint_names), sequence.position.clone()
        maxima = {kind: {"self_collision": {"maximum_trajectory": 0, "maximum_step": 2}}}
        result = NS(
            js_solution=sequence,
            interpolated_trajectory=sequence,
            debug_info={"selected_constraint_maxima": maxima},
            success=torch.tensor([False]),
        )
        if not expected_fixed and kind == "optimized":
            with pytest.raises(ValueError, match="Joint names do not match"):
                terminal_failure_summary(planner, result, trajectory_kind=kind)
        else:
            report = terminal_failure_summary(planner, result, trajectory_kind=kind)
            assert report["raw_self_collision_sample"]["acceptance_modified"] is False
            assert report["raw_self_collision_sample"]["joint_names"] == reduced.joint_names
        assert not bool(result.success.any())
        assert sequence.joint_names == before_names
        torch.testing.assert_close(sequence.position, before_position, rtol=0.0, atol=0.0)
        for name, value in locks.items():
            assert restored.position[0, restored.joint_names.index(name)].item() == pytest.approx(
                value
            )

    (tmp_path / "native-evidence.json").write_text(
        json.dumps(
            {
                "trajectory_kind": kind,
                "expected_fixed": expected_fixed,
                "robot_model": "synthetic 43 public / 9 numerical / 34 compiled holds",
                "actual_fk_geometry": True,
                "phases": phase_evidence,
                "trajectory_optimization_run": False,
                "acceptance_changed": False,
            },
            indent=2,
        )
        + "\n"
    )


def _locked_tcp_configuration(tmp_path):
    """Author one local 43-joint fixture with 34 locks and a named offset TCP."""
    active = ["base_x", "base_y", *[f"right_{i}" for i in range(7)]]
    held = [f"held_{i}" for i in range(34)]
    parts = ['<robot name="locked_tcp"><link name="world"/>']
    for branch, joints in (("right", active), ("left", held)):
        parent = "world"
        for index, name in enumerate(joints):
            link = f"{name}_link"
            axis = "0 1 0" if name == "base_y" else "1 0 0"
            offset = "0 1 0" if branch == "left" and index == 0 else "0 0 .01"
            parts.append(
                f'<link name="{link}"/><joint name="{name}" type="prismatic">'
                f'<parent link="{parent}"/><child link="{link}"/>'
                f'<origin xyz="{offset}"/><axis xyz="{axis}"/>'
                '<limit lower="-1" upper="1" effort="100" velocity="1"/>'
                "</joint>"
            )
            parent = link
    parts.append("</robot>")
    urdf = tmp_path / "locked_tcp.urdf"
    urdf.write_text("".join(parts))
    locks = {name: 0.001 * index for index, name in enumerate(held)}
    config = {
        "urdf_path": str(urdf),
        "base_link": "world",
        "tool_frames": ["custom_tcp", "held_33_link"],
        "collision_link_names": ["custom_tcp", "held_33_link", "payload_slot"],
        "collision_spheres": {
            "custom_tcp": [{"center": [0.0, 0.0, 0.0], "radius": 0.02}],
            "held_33_link": [{"center": [0.0, 0.0, 0.0], "radius": 0.02}],
            "payload_slot": [
                {"center": [0.0, 0.0, 0.0], "radius": -100.0},
                {"center": [0.0, 0.0, 0.0], "radius": -100.0},
            ],
        },
        "self_collision_ignore": {},
        "self_collision_buffer": {},
        "lock_joints": locks,
        "extra_links": {
            "custom_tcp": {
                "parent_link_name": "right_6_link",
                "link_name": "custom_tcp",
                "joint_name": "custom_tcp_joint",
                "joint_type": "FIXED",
                "fixed_transform": [0.1, 0.03, 0.04, 1.0, 0.0, 0.0, 0.0],
            },
            "payload_slot": {
                "parent_link_name": "custom_tcp",
                "link_name": "payload_slot",
                "joint_name": "payload_slot_joint",
                "joint_type": "FIXED",
                "fixed_transform": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            },
        },
        "cspace": {
            "joint_names": active + held,
            "default_joint_position": [0.0] * 43,
            "null_space_weight": [1.0] * 43,
            "cspace_distance_weight": [1.0] * 43,
            "max_acceleration": [1.0] * 43,
            "max_jerk": [4.0] * 43,
        },
    }
    return config, active, held, locks
