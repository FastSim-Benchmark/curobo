# SPDX-License-Identifier: Apache-2.0

"""Metrics for compound collision solids must use union containment."""

import numpy as np
import pytest
import torch
import trimesh

from curobo._src.geom.sphere_fit.metrics import (
    _solid_components,
    _union_sdf,
    compute_sphere_fit_metrics,
)
from curobo._src.geom.sphere_fit.wp_mesh_query import WarpMeshQuery


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_overlapping_components_keep_internal_faces_inside(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    left, right = trimesh.creation.box([2, 2, 2]), trimesh.creation.box([2, 2, 2])
    right.apply_translation([1, 0, 0])
    parts = _solid_components(trimesh.util.concatenate([left, right]))
    query = [WarpMeshQuery(part, torch.device(device)) for part in parts]
    points = torch.tensor([[0.1, 0, 0], [1.1, 0, 0], [2.3, 0, 0]], device=device)
    distances = _union_sdf(query, points)
    assert distances[:2].max() < 0
    assert distances[2].item() == pytest.approx(.3, abs=1e-6)


def test_union_coverage_is_measured_on_solid_components(monkeypatch):
    left, right = trimesh.creation.box([.01]*3), trimesh.creation.box([.01]*3)
    right.apply_translation([.02, 0, 0])
    mesh = trimesh.util.concatenate([left, right])
    # A combined parity query must never provide interior samples for a union.
    original = trimesh.Trimesh.contains

    def contains(self, points):
        assert self is not mesh
        return original(self, points)

    monkeypatch.setattr(trimesh.Trimesh, "contains", contains)
    metrics = compute_sphere_fit_metrics(
        mesh, np.array([[0, 0, 0], [.02, 0, 0]]), np.array([.009, .009]),
        n_interior=512, n_surface=128, n_sphere_surface=32, device=torch.device("cpu"),
    )
    assert metrics.coverage == 1
    assert metrics.max_uncovered_gap == 0


def test_open_components_are_not_silently_repaired_for_metrics():
    box = trimesh.creation.box()
    box.update_faces(np.arange(len(box.faces)-1))
    other = trimesh.creation.box()
    other.apply_translation([3, 0, 0])
    mesh = trimesh.util.concatenate([box, other])
    assert _solid_components(mesh) == [mesh]
