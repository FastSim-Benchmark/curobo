# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The returned sphere dtype must preserve the requested half-plane bound."""

import importlib

import numpy as np
import pytest
import torch
import trimesh

from curobo.sphere_fit import SphereFitType
from curobo.types import DeviceCfg

fit_module = importlib.import_module('curobo._src.geom.sphere_fit.fit_spheres')


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('normal,center,boundary', [
    ([0., 0., 1.], [0., 0., .00002], 0.),
    ([0., 0., 1.], [0., 0., 1.], 2. / 3.),
    ([1., 2., 3.], [.0123456789, .0234567891, .0345678912], .015),
])
def test_clipped_output_is_inside_plane_at_requested_precision(
    monkeypatch, dtype, normal, center, boundary,
):
    monkeypatch.setattr(fit_module, 'sample_even_fit_mesh',
                        lambda *_: (np.array([center]), np.array([1.])))
    offset = boundary - .02
    result = fit_module.fit_spheres_to_mesh(
        trimesh.creation.box(), num_spheres=1, fit_type=SphereFitType.SURFACE,
        clip_plane=(normal, offset), device_cfg=DeviceCfg(device='cpu', dtype=dtype),
    )
    n = torch.tensor(normal, dtype=torch.float64)
    n /= torch.linalg.vector_norm(n)
    distance = result.centers.double() @ n - (offset + .02)
    assert result.num_spheres == 1
    assert result.radii.dtype == dtype
    assert bool((result.radii.double() <= distance).all())
    larger = torch.nextafter(result.radii, torch.full_like(result.radii, float('inf')))
    assert bool((larger.double() > distance).all())


def test_clip_discards_behind_plane_and_preserves_interior_sphere(monkeypatch):
    monkeypatch.setattr(fit_module, 'sample_even_fit_mesh', lambda *_: (
        np.array([[0., 0., -.1], [0., 0., 0.], [0., 0., .5]]),
        np.array([.01, .01, .125]),
    ))
    result = fit_module.fit_spheres_to_mesh(
        trimesh.creation.box(), num_spheres=3, fit_type=SphereFitType.SURFACE,
        clip_plane=([0., 0., 1.], -.02), device_cfg=DeviceCfg(device='cpu'),
    )
    assert result.num_spheres == 1
    assert result.radii.tolist() == [.125]
    assert result.centers.tolist() == [[0., 0., .5]]


def test_center_rounding_onto_plane_removes_sphere(monkeypatch):
    monkeypatch.setattr(fit_module, 'sample_even_fit_mesh', lambda *_: (
        np.array([[0., 0., 1. + 2e-9]]), np.array([.01]),
    ))
    result = fit_module.fit_spheres_to_mesh(
        trimesh.creation.box(), num_spheres=1, fit_type=SphereFitType.SURFACE,
        clip_plane=([0., 0., 1.], .98), device_cfg=DeviceCfg(device='cpu'),
    )
    assert result.num_spheres == 0
    assert result.centers.shape == (0, 3)
    assert result.radii.shape == (0,)
