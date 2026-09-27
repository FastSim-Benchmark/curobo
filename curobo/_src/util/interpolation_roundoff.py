# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Remove single-ULP interpolation excursions without relaxing trajectory limits."""

from dataclasses import replace

import torch

from curobo._src.state.state_joint import JointState


def project_interpolation_position_roundoff(
    interpolated: JointState, reference: JointState, position_limits: torch.Tensor
) -> JointState:
    """Project one representable step beyond a limit for in-bound references.

    The eligibility mask is per trajectory and coordinate. An already violating
    reference, larger excursion, or nonfinite sample remains available to normal
    constraint rejection. Derivatives and the input states are not modified.
    """
    lower, upper = position_limits.unbind(0)
    eligible = (
        torch.isfinite(reference.position)
        & (reference.position >= lower)
        & (reference.position <= upper)
    ).all(dim=-2, keepdim=True)
    below = torch.nextafter(lower, torch.full_like(lower, -float("inf")))
    above = torch.nextafter(upper, torch.full_like(upper, float("inf")))
    position = interpolated.position
    finite = torch.isfinite(position)
    position = torch.where(
        eligible & finite & (position < lower) & (position >= below), lower, position
    )
    position = torch.where(
        eligible & finite & (position > upper) & (position <= above), upper, position
    )
    return replace(interpolated, position=position)
