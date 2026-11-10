# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Higher precision capacity retains the bounded fit safety contract."""

import pytest
import trimesh

from curobo.sphere_fit import SphereFitType, fit_spheres_to_mesh
from curobo.types import DeviceCfg


@pytest.mark.parametrize("budget", [16384, 24576])
def test_precision_capacity_preserves_certificate(budget: int) -> None:
    result = fit_spheres_to_mesh(
        trimesh.creation.box([0.01, 0.014, 0.012]),
        num_spheres=budget, fit_type=SphereFitType.BOUNDED,
        max_protrusion_m=0.002, max_bottom_protrusion_m=0.002,
        device_cfg=DeviceCfg(device="cpu"), compute_metrics=False,
    )
    proof = result.debug_info["bounded"]
    assert proof["full_volume_covered"]
    assert proof["verified_max_protrusion_m"] <= 0.002
    assert result.num_spheres <= budget
    assert not result.debug_info["fallback_used"]


def test_precision_capacity_still_bounded() -> None:
    with pytest.raises(ValueError, match="budget"):
        fit_spheres_to_mesh(
            trimesh.creation.box([0.01] * 3), num_spheres=24577,
            fit_type=SphereFitType.BOUNDED, max_protrusion_m=0.002,
            device_cfg=DeviceCfg(device="cpu"), compute_metrics=False,
        )
