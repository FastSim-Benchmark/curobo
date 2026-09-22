# SPDX-License-Identifier: Apache-2.0

"""Public FAST fitting contracts and shared clipping behavior."""

import inspect

import numpy as np
import pytest
import torch
import trimesh

from curobo._src.collision.attachment_manager import AttachmentManager
from curobo._src.geom.sphere_fit._fast_refine import SilhouetteHull
from curobo._src.geom.types import Obstacle
from curobo._src.robot.builder.builder_robot import RobotBuilder
from curobo.sphere_fit import SphereFitType, fit_spheres_to_mesh
from curobo.types import DeviceCfg


@pytest.mark.parametrize("density,budget", [(0.5, 16), (1.0, 32), (1e308, 256)])
def test_fast_automatic_budget_and_large_density(monkeypatch, density, budget):
    """Automatic budgets scale with density and saturate without overflow."""
    import curobo._src.geom.sphere_fit.fit_fast as backend

    def fake_fit(mesh, num_spheres):
        assert num_spheres == budget
        return np.zeros((1, 3)), np.array([0.1]), {}

    monkeypatch.setattr(backend, "fast_fit_mesh", fake_fit)
    result = fit_spheres_to_mesh(
        trimesh.creation.box(), sphere_density=density, device_cfg=DeviceCfg(device="cpu")
    )
    assert result.debug_info["resolved_n_spheres"] == budget


def test_all_public_defaults_select_fast():
    """Changing the leaf default must also update forwarding interfaces."""
    methods = [
        fit_spheres_to_mesh,
        Obstacle.get_bounding_spheres,
        RobotBuilder.fit_collision_spheres,
        RobotBuilder.refit_link_spheres,
    ]
    for method in methods:
        assert inspect.signature(method).parameters["fit_type"].default is SphereFitType.FAST
    assert (
        inspect.signature(AttachmentManager.fit_spheres).parameters["sphere_fit_type"].default
        is SphereFitType.FAST
    )


def test_fast_default_is_original_mesh_cpu_budgeted_and_audited():
    """The default returns finite typed spheres and exposes quality diagnostics."""
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.1)
    result = fit_spheres_to_mesh(mesh, device_cfg=DeviceCfg(device="cpu", dtype=torch.float64))
    assert 0 < result.num_spheres <= 32
    assert result.centers.device.type == "cpu"
    assert result.centers.dtype == result.radii.dtype == torch.float64
    assert result.centers.is_contiguous() and result.radii.is_contiguous()
    assert torch.isfinite(result.centers).all() and (result.radii > 0).all()
    assert result.used_mesh is mesh
    assert result.debug_info["fit_type"] == "fast"
    assert not result.debug_info["used_convex_hull"]
    assert "pre_clip_audit" in result.debug_info["fast"]


@pytest.mark.parametrize("budget", [0, -1, 257, 1.5, True])
def test_fast_rejects_invalid_budget(budget):
    """Unsupported budgets fail explicitly instead of changing methods."""
    with pytest.raises(ValueError):
        fit_spheres_to_mesh(
            trimesh.creation.box(), num_spheres=budget, device_cfg=DeviceCfg(device="cpu")
        )


@pytest.mark.parametrize("density", [0, -1, np.nan])
def test_fast_rejects_invalid_automatic_density(density):
    """Automatic density has an explicit finite positive contract."""
    with pytest.raises(ValueError):
        fit_spheres_to_mesh(
            trimesh.creation.box(), sphere_density=density, device_cfg=DeviceCfg(device="cpu")
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fast_uses_shared_clip_plane_precision(dtype):
    """The new fitter must obey the existing post-conversion plane bound."""
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=0.1)
    mesh.apply_translation([0, 0, 0.15])
    result = fit_spheres_to_mesh(
        mesh,
        num_spheres=1,
        clip_plane=([0, 0, 1], 0.12),
        device_cfg=DeviceCfg(device="cpu", dtype=dtype),
    )
    assert result.num_spheres == 1
    assert bool((result.radii.double() <= result.centers[:, 2].double() - (0.12 + 0.02)).all())


def test_fast_visual_hull_fills_cup_but_keeps_visible_gap():
    """A blind cavity is allowed; a gap between disconnected rods is not filled."""
    wall = trimesh.creation.annulus(r_min=0.38, r_max=0.5, height=1, sections=24)
    bottom = trimesh.creation.cylinder(radius=0.5, height=0.08, sections=24)
    bottom.apply_translation([0, 0, -0.46])
    cup = trimesh.util.concatenate([wall, bottom])
    hull = SilhouetteHull.from_mesh(cup.vertices, cup.faces, resolution=32, view_count=8)
    assert hull.distance(np.zeros((1, 3)))[0] < 0
    rod = trimesh.creation.box([0.12, 0.12, 1])
    other = rod.copy()
    rod.apply_translation([-0.4, 0, 0])
    other.apply_translation([0.4, 0, 0])
    mesh = trimesh.util.concatenate([rod, other])
    hull = SilhouetteHull.from_mesh(mesh.vertices, mesh.faces, resolution=32, view_count=8)
    assert hull.distance(np.zeros((1, 3)))[0] > 0.2


def test_fast_empty_result_never_falls_back_to_legacy(monkeypatch):
    import curobo._src.geom.sphere_fit.fit_fast as backend
    monkeypatch.setattr(backend, "fast_fit_mesh", lambda *_: (None, None, {}))
    with pytest.raises(ValueError, match="legacy fitting fallback is disabled"):
        fit_spheres_to_mesh(
            trimesh.creation.box(), num_spheres=32, device_cfg=DeviceCfg(device="cpu")
        )
