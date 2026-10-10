# SPDX-License-Identifier: Apache-2.0
"""Small-sphere coverage metrics must remain accurate with global TF32 enabled."""

import numpy as np
import pytest
import torch
import trimesh

from curobo._src.geom.sphere_fit.metrics import compute_sphere_fit_metrics


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("covered", [True, False])
def test_translated_millimeter_geometry_has_correct_coverage(device, covered, monkeypatch):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    center = np.array([.13, .11, -.10])
    mesh = trimesh.creation.box([.002] * 3)
    mesh.apply_translation(center)
    centers = np.repeat(center[None], 32, axis=0)
    radii = np.full(32, .002 if covered else .0002)
    state = np.random.get_state()
    try:
        np.random.seed(537)
        metrics = compute_sphere_fit_metrics(
            mesh, centers, radii, n_interior=512, n_surface=512,
            n_sphere_surface=8, device=torch.device(device),
        )
    finally:
        np.random.set_state(state)
    if covered:
        assert metrics.coverage == 1
        assert metrics.max_uncovered_gap == 0
    else:
        assert metrics.coverage < .1
        assert metrics.max_uncovered_gap > .001
    assert torch.backends.cuda.matmul.allow_tf32 is True
