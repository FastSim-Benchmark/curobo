# SPDX-License-Identifier: Apache-2.0

"""Sphere-fit diagnostics retain the requested CPU or CUDA device."""

import numpy as np
import pytest
import torch
import trimesh

from curobo._src.geom.sphere_fit.wp_mesh_query import WarpMeshQuery
from curobo.sphere_fit import fit_spheres_to_mesh
from curobo.types import DeviceCfg


@pytest.mark.parametrize("device", ["cpu", "cuda", "cuda:0"])
def test_queries_keep_device_and_geometric_results(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    query = WarpMeshQuery(trimesh.creation.box(extents=[2, 2, 2]), torch.device(device))
    points = torch.tensor([[0.25, 0, 0], [1.25, 0, 0]], device=device)
    assert query.device == points.device
    outside = query.query_outside_mask(points)
    assert outside.device == points.device
    assert outside.tolist() == [False, True]
    sdf, gradient = query.query_sdf(points)
    assert sdf.device == gradient.device == points.device
    torch.testing.assert_close(sdf, torch.tensor([-0.75, 0.25], device=device))
    closest, distances = query.query_closest_point(points)
    assert closest.device == distances.device == points.device
    torch.testing.assert_close(closest, torch.tensor([[1., 0, 0], [1., 0, 0]], device=device))


def test_public_cpu_fit_with_metrics():
    result = fit_spheres_to_mesh(
        trimesh.creation.box(extents=[0.1, 0.1, 0.1]),
        num_spheres=8,
        compute_metrics=True,
        device_cfg=DeviceCfg(device="cpu"),
    )
    assert result.centers.device.type == result.radii.device.type == "cpu"
    assert result.metrics is not None
    assert np.isfinite(result.metrics.surface_gap_mean)
    assert 0 <= result.metrics.coverage <= 1
