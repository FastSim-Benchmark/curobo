# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Geometric tool-axis error for the existing pose cost kernel."""

import warp as wp


@wp.func
def compute_tool_axis_error(
    current_quat: wp.quat,
    goal_quat: wp.quat,
    local_axis: wp.vec3,
    weight: wp.float32,
    squared_tolerance: wp.float32,
):
    """Return weighted tilt squared, its spatial rotation gradient, and tilt in radians."""
    current_axis = wp.quat_rotate(current_quat, local_axis)
    goal_axis = wp.quat_rotate(goal_quat, local_axis)
    cross = wp.cross(current_axis, goal_axis)
    sine = wp.length(cross)
    cosine = wp.clamp(wp.dot(current_axis, goal_axis), -1.0, 1.0)
    angle = wp.atan2(sine, cosine)
    if weight == 0.0:
        angle = 0.0
    distance = weight * angle * angle
    gradient = wp.vec3(0.0, 0.0, 0.0)
    if distance < squared_tolerance:
        distance = 0.0
    elif sine > 1.0e-7:
        gradient = (-2.0 * weight * angle / sine) * cross
    elif cosine >= 0.0:
        # angle / sin(angle) tends to one near alignment.
        gradient = -2.0 * weight * cross
    else:
        # At the antipode there is no unique descent direction. Choose a
        # deterministic perpendicular direction; the reported error stays pi.
        basis = wp.vec3(1.0, 0.0, 0.0)
        if wp.abs(current_axis[0]) > 0.9:
            basis = wp.vec3(0.0, 1.0, 0.0)
        gradient = -2.0 * weight * angle * wp.normalize(wp.cross(current_axis, basis))
    return distance, gradient, angle
