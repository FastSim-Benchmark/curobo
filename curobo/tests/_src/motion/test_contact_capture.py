# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for fresh, sparse boundary-contact capture."""

from types import SimpleNamespace

import pytest
import torch

from curobo._src.collision.contact_approach import GoalContact
from curobo._src.collision.contact_separation import StartContact
from curobo._src.motion.motion_contact import capture_contact


@pytest.fixture
def planner() -> SimpleNamespace:
    """Reserve unused slots around one support and one distant obstacle."""
    inverse = torch.zeros((1, 128, 8))
    inverse[:, :, 3] = 1.0
    inverse[0, 127, 0] = -10.0
    names = [None] * 128
    names[1], names[127] = "support", "distant"
    enable = torch.zeros((1, 128), dtype=torch.uint8)
    enable[0, 1] = enable[0, 127] = 1
    cuboids = SimpleNamespace(
        names=[names], enable=enable, dims=torch.ones((1, 128, 4)), inv_pose=inverse
    )
    spheres = torch.tensor([[0.599, 0, 0, 0.1], [0.2, 0, 0, -1.0]])
    return SimpleNamespace(
        scene_collision_checker=SimpleNamespace(
            data=SimpleNamespace(num_envs=1, cuboids=cuboids, meshes=None),
            device_cfg=SimpleNamespace(device=torch.device("cpu")),
        ),
        spheres=spheres,
        compute_kinematics=lambda state: SimpleNamespace(robot_spheres=spheres),
        attachment_manager=SimpleNamespace(
            kinematics_params=SimpleNamespace(
                get_sphere_index_from_link_name=lambda name: [0] if name == "tool" else [1]
            )
        ),
    )


@pytest.mark.parametrize("terminal", [False, True])
def test_sparse_capture_preserves_support_and_sphere_identity(planner, terminal) -> None:
    contact = capture_contact(planner, torch.zeros((1, 2)), terminal=terminal)
    assert isinstance(contact, GoalContact if terminal else StartContact)
    assert contact.obstacle_name == "support"
    assert contact.sphere_indices == (0,)
    captured = contact.goal_spheres if terminal else contact.initial_spheres
    assert captured == (tuple(planner.spheres[0].tolist()),)


def test_enable_changes_are_observed_on_the_next_capture(planner) -> None:
    state = torch.zeros((1, 2))
    assert capture_contact(planner, state, terminal=False) is not None
    planner.scene_collision_checker.data.cuboids.enable[0, 1] = 0
    assert capture_contact(planner, state, terminal=False) is None
    planner.scene_collision_checker.data.cuboids.enable[0, 1] = 1
    assert capture_contact(planner, state, terminal=False) is not None


def test_contact_link_filter_preserves_ineligible_spheres(planner) -> None:
    state = torch.zeros((1, 2))
    assert capture_contact(planner, state, terminal=False, contact_links=("tool",)) is not None
    assert capture_contact(planner, state, terminal=False, contact_links=("disabled",)) is None


def test_excess_penetration_still_rejects_capture(planner) -> None:
    planner.spheres[0, 0] = 0.590
    with pytest.raises(ValueError, match="initial penetration"):
        capture_contact(planner, torch.zeros((1, 2)), terminal=False)
