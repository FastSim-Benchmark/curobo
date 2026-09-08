# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Per-motion preservation of a direction rigidly attached to a tool."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class AxisHold:
    """Hold a tool-local axis in its starting direction, allowing free twist.

    The reference is captured once from the planning start state. This does
    not straighten an initially tilted object or change the requested endpoint.
    ``axis`` can describe an object axis expressed in tool coordinates.
    """

    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    tolerance_rad: float = 0.01

    def __post_init__(self):
        """Validate and normalize the caller's tool-local direction."""
        values = tuple(self.axis)
        if len(values) != 3 or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
            for v in values
        ):
            raise ValueError("axis must contain three finite numbers")
        norm = math.hypot(*values)
        if norm == 0.0 or not math.isfinite(norm):
            raise ValueError("axis must have finite non-zero length")
        if (
            isinstance(self.tolerance_rad, bool)
            or not isinstance(self.tolerance_rad, (int, float))
            or not 0.0 < self.tolerance_rad < math.pi
        ):
            raise ValueError("tolerance_rad must be finite and between zero and pi")
        object.__setattr__(self, "axis", tuple(v / norm for v in values))
