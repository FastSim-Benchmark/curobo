"""Diagnostic mesh samples retain joint, sphere and obstacle identity."""

from types import SimpleNamespace as NS

import pytest
import torch

from curobo._src.collision.contact_mesh import MeshClearance
from curobo._src.motion.motion_failure_diagnostics import terminal_failure_summary
from curobo._src.state.state_joint import JointState


def test_mesh_sample_uses_selected_peak_and_masks_disabled_spheres(monkeypatch):
    state = JointState.from_position(torch.arange(6.0).reshape(2, 3, 1), joint_names=["base"])
    spheres = torch.tensor([[0.0, 0, 0, 0.1], [1.0, 0, 0, -1.0]])
    meshes = NS(
        enable=torch.tensor([[True, False, True]]), names=[["table", "disabled", "machine"]]
    )

    def compute(selected):
        assert selected.position.tolist() == [[4.0]]
        return NS(robot_spheres=spheres)

    def gaps(selected, scene, indices):
        assert scene is meshes and indices.tolist() == [0, 2]
        return torch.tensor([[0.02, -0.003], [-10.0, -10.0]])

    monkeypatch.setattr(MeshClearance, "apply", gaps)
    planner = NS(
        scene_collision_checker=NS(data=NS(meshes=meshes)),
        compute_kinematics=compute,
        trajopt_solver=NS(metrics_rollout=NS(metrics_constraint_manager=None)),
        attachment_manager=NS(
            kinematics_params=NS(
                link_name_to_idx_map={"finger": 0, "unused": 1},
                link_sphere_idx_map=torch.tensor([0, 1]),
            )
        ),
    )
    result = NS(
        js_solution=state,
        debug_info={
            "selected_constraint_maxima": {
                "optimized": {"scene_collision": {"maximum_step": 1, "maximum_trajectory": 1}}
            }
        },
    )
    before = state.position.clone()
    sample = terminal_failure_summary(planner, result)["raw_mesh_gap_sample"]
    assert sample["trajectory"] == sample["step"] == 1
    assert sample["pairs"][0]["obstacle"] == "machine"
    assert sample["pairs"][0]["link"] == "finger"
    assert sample["pairs"][0]["raw_gap_m"] < 0
    assert sample["contact_allowance_applied"] is False
    assert len(sample["pairs"]) == 2
    torch.testing.assert_close(before, state.position)


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
    for sample in summary["samples"]:
        assert sample["collision_pair_count"] == expected
        assert len(sample["pairs"]) == expected
        if expected:
            pair = sample["pairs"][0]
            assert pair["links"] == ["finger", "attached_object"]
            assert pair["raw_overlap_m"] == pytest.approx(-.01)
            assert pair["padded_overlap_m"] == pytest.approx(.03)
    torch.testing.assert_close(before, spheres)
