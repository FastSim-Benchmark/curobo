# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Generic Warp kernels for swept-sphere obstacle collision detection.

This module provides type-generic collision kernels that work with any obstacle
type via Warp function overloading. The kernels use `typing.Any` for the obstacle
set parameter, and Warp dispatches to the correct overloaded functions based on
the struct type at runtime.

Key features:
- Single kernel for all obstacle types (cuboid, mesh, voxel)
- 2D parallelization: bounded obstacle lanes per sphere, dynamic populated-slot traversal
- Sparse atomic accumulation: only non-zero costs write to output
- Optimized frame transforms: load once, accumulate in local frame, transform once

Performance optimization:
- Frame transform is loaded once per (sphere, obstacle) pair
- Gradients are accumulated in obstacle local frame
- Single gradient-to-world transform at the end
- Reduces quaternion operations by ~N for N sweep queries

Kernels:
- swept_sphere_obstacle_collision_kernel: Swept collision (trajectory interpolation)
"""

# Standard Library
# =============================================================================
# Automatic SDF Function Overload Registration
# =============================================================================
#
# Registers SDF function overloads from all obstacle data modules.
# Module list is centralized in curobo._src.geom.data.registry.OBSTACLE_SDF_MODULES.
#
# How it works:
# - wp.func uses scope_locals.get(func.func.__name__) to find existing Functions
# - By keeping variables named "is_obs_enabled" and "compute_sdf_with_grad" in scope,
#   subsequent wp.func calls detect the existing Function and add overloads
from importlib import import_module
from typing import Any

# Third Party
import warp as wp

from curobo._src.geom.collision.wp_collision_common import (
    accumulate_collision,
    apply_collision_activation,
    load_sphere_query,
)
from curobo._src.geom.data.registry import OBSTACLE_SDF_MODULES

# Initialize to None - will be set by first iteration
is_obs_enabled = None
query_aabb_overlaps = None
load_obstacle_transform = None
compute_local_sdf = None
compute_local_sdf_with_grad = None


for _module_path in OBSTACLE_SDF_MODULES:
    _data_module = import_module(_module_path)
    _obs_fn = getattr(_data_module, "is_obs_enabled")
    _bounds_fn = getattr(_data_module, "query_aabb_overlaps")
    _transform_fn = getattr(_data_module, "load_obstacle_transform")
    _sdf_only_fn = getattr(_data_module, "compute_local_sdf")
    _sdf_fn = getattr(_data_module, "compute_local_sdf_with_grad")

    # Register with wp.func - adds overloads on subsequent iterations
    is_obs_enabled = wp.func(_obs_fn, module=__name__)
    query_aabb_overlaps = wp.func(_bounds_fn, module=__name__)
    load_obstacle_transform = wp.func(_transform_fn, module=__name__)
    compute_local_sdf = wp.func(_sdf_only_fn, module=__name__)
    compute_local_sdf_with_grad = wp.func(_sdf_fn, module=__name__)

del _module_path, _data_module, _obs_fn, _bounds_fn, _transform_fn, _sdf_fn


SWEEP_STEPS = wp.constant(3)
SWEEP_GROUP_SIZE = wp.constant(32)


@wp.kernel(enable_backward=False)
def obstacle_world_bounds(
    obs_set: Any,
    local_bounds: wp.array2d(dtype=wp.float32),
    lower_bounds: wp.array(dtype=wp.vec4),
    upper_bounds: wp.array(dtype=wp.vec4),
):
    """Transform each populated obstacle once per query, never cache stale poses."""
    index = wp.tid()
    env = index / obs_set.max_n
    local = index % obs_set.max_n
    lower = wp.vec3(1.0e30)
    upper = wp.vec3(-1.0e30)
    state = wp.int32(0)
    if is_obs_enabled(obs_set, env, local):
        state = 1
        inverse = load_obstacle_transform(obs_set, env, local)
        quaternion = wp.transform_get_rotation(inverse)
        translation = wp.transform_get_translation(inverse)
        norm = wp.dot(quaternion, quaternion)
        if not wp.isfinite(norm) or wp.abs(norm - 1.0) > 1.0e-6:
            state = 2
        scale = wp.vec3(1.0) + wp.abs(translation)
        for axis in range(3):
            lo = local_bounds[index, axis]
            hi = local_bounds[index, axis + 3]
            scale[axis] = scale[axis] + wp.max(wp.abs(lo), wp.abs(hi))
            if not wp.isfinite(lo) or not wp.isfinite(hi) or lo > hi:
                state = 2
        world = wp.transform_inverse(inverse)
        for corner in range(8):
            point = wp.vec3(local_bounds[index, 0], local_bounds[index, 1], local_bounds[index, 2])
            if corner % 2 == 1:
                point[0] = local_bounds[index, 3]
            if (corner / 2) % 2 == 1:
                point[1] = local_bounds[index, 4]
            if (corner / 4) % 2 == 1:
                point[2] = local_bounds[index, 5]
            point = wp.transform_point(world, point)
            for axis in range(3):
                if not wp.isfinite(point[axis]):
                    state = 2
            lower = wp.min(lower, point)
            upper = wp.max(upper, point)
        # Include transform, quaternion-normalization and interval-rounding error.
        # Unusable bounds retain the original narrow phase instead of excluding it.
        scale = wp.max(scale, wp.max(wp.abs(lower), wp.abs(upper)))
        margin = wp.vec3(3.0517578125e-5 * (scale[0] + scale[1] + scale[2]))
        lower = lower - margin
        upper = upper + margin
    lower_bounds[index] = wp.vec4(lower[0], lower[1], lower[2], wp.float32(state))
    upper_bounds[index] = wp.vec4(upper[0], upper[1], upper[2], 0.0)


@wp.kernel(enable_backward=False)
def swept_sphere_group_bounds(
    spheres: wp.array(dtype=wp.vec4),
    activation_distance: wp.array(dtype=wp.float32),
    lower_bounds: wp.array(dtype=wp.vec4),
    upper_bounds: wp.array(dtype=wp.vec4),
    horizon: wp.int32,
    num_spheres: wp.int32,
    groups_per_step: wp.int32,
):
    """Enclose each adjacent sweep, without crossing batch or timestep boundaries."""
    group = wp.tid()
    step = group / groups_per_step
    h_idx = step % horizon
    begin = (group % groups_per_step) * SWEEP_GROUP_SIZE
    lower = wp.vec3(1.0e30)
    upper = wp.vec3(-1.0e30)
    state = wp.int32(0)  # empty, finite, or nonfinite (must retain all obstacles)
    for offset in range(SWEEP_GROUP_SIZE):
        sphere = begin + offset
        if sphere >= num_spheres:
            break
        index = step * num_spheres + sphere
        current = spheres[index]
        if current[3] < 0.0:
            continue
        state = wp.max(state, 1)
        radius = wp.max(current[3] + activation_distance[0], 0.0)
        if not wp.isfinite(current[3]) or not wp.isfinite(radius):
            state = 2
        for neighbor in range(3):
            shift = neighbor - 1
            if h_idx + shift < 0 or h_idx + shift >= horizon:
                continue
            value = spheres[index + shift * num_spheres]
            point = wp.vec3(value[0], value[1], value[2])
            if not wp.isfinite(point[0]) or not wp.isfinite(point[1]) or not wp.isfinite(point[2]):
                state = 2
            lower = wp.min(lower, point - wp.vec3(radius))
            upper = wp.max(upper, point + wp.vec3(radius))
    # Account for float32 expansion and the subsequent rigid transform.
    scale = wp.max(wp.abs(lower), wp.abs(upper))
    scale = wp.max(scale, wp.vec3(1.0))
    margin = 3.814697265625e-6 * scale
    lower = lower - margin
    upper = upper + margin
    lower_bounds[group] = wp.vec4(lower[0], lower[1], lower[2], wp.float32(state))
    upper_bounds[group] = wp.vec4(upper[0], upper[1], upper[2], 0.0)


# =============================================================================
# Generic Swept-Sphere Collision Kernel
# =============================================================================


@wp.func
def swept_sphere_obstacle_collision_kernel_pair(
    # Obstacle set (generic - uses Warp function overloading)
    obs_set: Any,
    # Sphere data
    spheres: wp.array(dtype=wp.vec4),  # [batch * horizon * num_spheres]
    # Parameters
    weight: wp.array(dtype=wp.float32),
    activation_distance: wp.array(dtype=wp.float32),
    env_query_idx: wp.array(dtype=wp.int32),
    # Outputs (pre-zeroed)
    distance: wp.array(dtype=wp.float32),  # [batch * horizon * num_spheres]
    gradient: wp.array(dtype=wp.float32),  # [batch * horizon * num_spheres * 4]
    # Dimensions
    batch_size: wp.int32,
    horizon: wp.int32,
    num_spheres: wp.int32,
    max_n_obs: wp.int32,
    use_multi_env: wp.uint8,
    replacement_cuboid_ids: wp.array(dtype=wp.int32),
    replacement_pair_count: wp.int32,
    sph_flat_idx: wp.int32,
    obs_local_idx: wp.int32,
):
    """Compute swept-sphere collision with 2D parallelization.

    Sweeps toward previous and next timesteps in the trajectory.
    Uses adaptive stepping based on penetration distance.
    The number of sweep steps is fixed at compile time via SWEEP_STEPS constant.

    Thread layout: (sphere, obstacle), without a signed 32-bit pair index.

    Args:
        obs_set: Obstacle set (CuboidDataWarp, MeshDataWarp, or VoxelDataWarp).
        spheres: Sphere positions and radii for all timesteps.
        weight: Collision cost weight.
        activation_distance: Distance threshold for collision activation.
        env_query_idx: Environment index per batch element.
        distance: Output distance buffer (pre-zeroed).
        gradient: Output gradient buffer (pre-zeroed).
        batch_size: Number of batch elements.
        horizon: Trajectory horizon.
        num_spheres: Number of spheres per timestep.
        max_n_obs: Number of parallel obstacle lanes per sphere.
        use_multi_env: Whether to use batch-specific environments.
        replacement_cuboid_ids: Flattened (num_envs, num_spheres, replacement_pair_count) map.
        replacement_pair_count: Number of replacement slots per sphere for this obstacle set.
        sph_flat_idx: Flattened batch, timestep and sphere coordinate.
        obs_local_idx: Original obstacle slot in the selected environment.
    """
    # Keep each coordinate representable when their product exceeds int32.

    total_spheres = batch_size * horizon * num_spheres
    if sph_flat_idx >= total_spheres:
        return



    # Decode batch/horizon indices
    b_idx = sph_flat_idx / (horizon * num_spheres)
    h_idx = (sph_flat_idx - b_idx * horizon * num_spheres) / num_spheres

    # Get environment index
    env_idx = wp.int32(0)
    if use_multi_env == wp.uint8(1):
        env_idx = env_query_idx[b_idx]


    # Check if obstacle exists and is enabled
    if not is_obs_enabled(obs_set, env_idx, obs_local_idx):
        return

    if replacement_pair_count > 0:
        sphere_idx = sph_flat_idx % num_spheres
        offset = (env_idx * num_spheres + sphere_idx) * replacement_pair_count
        for slot in range(replacement_pair_count):
            if replacement_cuboid_ids[offset + slot] == obs_local_idx:
                return

    # Copy struct from param space to local stack to prevent nvcc
    # from generating cvta.to.local on the param pointer
    obs = obs_set

    # Load parameters
    eta = activation_distance[0]
    w = weight[0]

    # Load sphere - extract radius_adjusted as scalar so query struct can die early
    query = load_sphere_query(spheres, sph_flat_idx, eta)
    if query.radius < 0.0:
        return
    radius_adjusted = query.radius_adjusted

    # Load obstacle transform ONCE for all sweep queries
    inv_t = load_obstacle_transform(obs, env_idx, obs_local_idx)

    # Accumulators in local frame (apply weight and transform at end)
    cost_sum = wp.float32(0.0)
    grad_sum_local = wp.vec3(0.0, 0.0, 0.0)

    # Transform current sphere center to local frame.
    # After this, query.center is no longer needed - sweep distances
    # are computed in local frame (rigid transforms preserve distances).
    local_current = wp.transform_point(inv_t, query.center)

    # AABB of the full adjacent segments contains every adaptive sweep sample.
    # Checking endpoints alone would incorrectly discard a crossing trajectory.
    query_lower = local_current
    query_upper = local_current
    if h_idx > 0:
        previous = spheres[sph_flat_idx - num_spheres]
        point = wp.transform_point(inv_t, wp.vec3(previous[0], previous[1], previous[2]))
        query_lower = wp.min(query_lower, point)
        query_upper = wp.max(query_upper, point)
    if h_idx < horizon - 1:
        following = spheres[sph_flat_idx + num_spheres]
        point = wp.transform_point(inv_t, wp.vec3(following[0], following[1], following[2]))
        query_lower = wp.min(query_lower, point)
        query_upper = wp.max(query_upper, point)
    if not query_aabb_overlaps(
        obs, env_idx, obs_local_idx, query_lower, query_upper, radius_adjusted
    ):
        return

    # Process current timestep
    sdf_result = compute_local_sdf_with_grad(
        obs, env_idx, obs_local_idx, local_current, radius_adjusted
    )
    penetration = -sdf_result[0] + radius_adjusted

    if penetration > 0.0:
        activation_result = apply_collision_activation(penetration, eta)
        cost_sum += activation_result[0]
        grad_sum_local += activation_result[1] * wp.vec3(
            sdf_result[1], sdf_result[2], sdf_result[3]
        )

    # Sweep toward h-1 (previous timestep)
    # Interpolation and distance are computed in local frame to:
    # 1. Avoid per-step quaternion rotations (lerp in local space)
    # 2. Free query.center registers (rigid transforms preserve distances)
    if h_idx > 0 and SWEEP_STEPS > 0:
        prev_sphere = spheres[sph_flat_idx - num_spheres]
        local_prev = wp.transform_point(
            inv_t, wp.vec3(prev_sphere[0], prev_sphere[1], prev_sphere[2])
        )
        half_dist = wp.length(local_prev - local_current) * 0.5
        # Both adaptive half sweeps exclude jump == half_dist. Always inspect
        # their shared boundary, including when a minimum step jumps over it.
        # Keep every original adaptive sample as well.
        if half_dist > 0.0:
            midpoint = 0.5 * (local_current + local_prev)
            midpoint_sdf = compute_local_sdf_with_grad(
                obs, env_idx, obs_local_idx, midpoint, radius_adjusted
            )
            midpoint_penetration = radius_adjusted - midpoint_sdf[0]
            if midpoint_penetration > 0.0:
                midpoint_activation = apply_collision_activation(midpoint_penetration, eta)
                cost_sum += midpoint_activation[0]
                grad_sum_local += midpoint_activation[1] * wp.vec3(
                    midpoint_sdf[1], midpoint_sdf[2], midpoint_sdf[3]
                )
        inv_half_dist = 1.0 / wp.max(half_dist, 0.001)
        jump = wp.float32(0.0)
        for _ in range(SWEEP_STEPS):
            if jump >= half_dist:
                break
            t = 1.0 - 0.5 * jump * inv_half_dist
            local_pt = t * local_current + (1.0 - t) * local_prev

            # Use one SDF overload in the sweep loops to avoid invalid Warp
            # codegen for VoxelDataWarp when both overloads are mixed here.
            sdf_result = compute_local_sdf_with_grad(
                obs, env_idx, obs_local_idx, local_pt, radius_adjusted
            )
            penetration = -sdf_result[0] + radius_adjusted

            if penetration > 0.0:
                activation_result = apply_collision_activation(penetration, eta)
                cost_sum += activation_result[0]
                grad_sum_local += activation_result[1] * wp.vec3(
                    sdf_result[1], sdf_result[2], sdf_result[3]
                )
                jump += penetration
            else:
                if -penetration >= 1000.0:
                    jump += radius_adjusted
                else:
                    jump += wp.max(-penetration, radius_adjusted)

    # Sweep toward h+1 (next timestep)
    if h_idx < horizon - 1 and SWEEP_STEPS > 0:
        next_sphere = spheres[sph_flat_idx + num_spheres]
        local_next = wp.transform_point(
            inv_t, wp.vec3(next_sphere[0], next_sphere[1], next_sphere[2])
        )
        half_dist = wp.length(local_next - local_current) * 0.5
        # Both adaptive half sweeps exclude jump == half_dist. Always inspect
        # their shared boundary, including when a minimum step jumps over it.
        # Keep every original adaptive sample as well.
        if half_dist > 0.0:
            midpoint = 0.5 * (local_current + local_next)
            midpoint_sdf = compute_local_sdf_with_grad(
                obs, env_idx, obs_local_idx, midpoint, radius_adjusted
            )
            midpoint_penetration = radius_adjusted - midpoint_sdf[0]
            if midpoint_penetration > 0.0:
                midpoint_activation = apply_collision_activation(midpoint_penetration, eta)
                cost_sum += midpoint_activation[0]
                grad_sum_local += midpoint_activation[1] * wp.vec3(
                    midpoint_sdf[1], midpoint_sdf[2], midpoint_sdf[3]
                )
        inv_half_dist = 1.0 / wp.max(half_dist, 0.001)
        jump = wp.float32(0.0)
        for _ in range(SWEEP_STEPS):
            if jump >= half_dist:
                break
            t = 1.0 - 0.5 * jump * inv_half_dist
            local_pt = t * local_current + (1.0 - t) * local_next

            # See previous sweep loop for why this uses the with-grad overload.
            sdf_result = compute_local_sdf_with_grad(
                obs, env_idx, obs_local_idx, local_pt, radius_adjusted
            )
            penetration = -sdf_result[0] + radius_adjusted

            if penetration > 0.0:
                activation_result = apply_collision_activation(penetration, eta)
                cost_sum += activation_result[0]
                grad_sum_local += activation_result[1] * wp.vec3(
                    sdf_result[1], sdf_result[2], sdf_result[3]
                )
                jump += penetration
            else:
                if -penetration >= 1000.0:
                    jump += radius_adjusted
                else:
                    jump += wp.max(-penetration, radius_adjusted)

    # Transform accumulated gradient to world frame ONCE and apply weight
    if cost_sum > 0.0:
        fwd_t = wp.transform_inverse(inv_t)
        grad_world = wp.transform_vector(fwd_t, grad_sum_local)
        accumulate_collision(sph_flat_idx, w * cost_sum, w * grad_world, distance, gradient)


@wp.kernel(enable_backward=False)
def swept_sphere_obstacle_collision_kernel(
    # Obstacle set (generic - uses Warp function overloading)
    obs_set: Any,
    # Sphere data
    spheres: wp.array(dtype=wp.vec4),  # [batch * horizon * num_spheres]
    # Parameters
    weight: wp.array(dtype=wp.float32),
    activation_distance: wp.array(dtype=wp.float32),
    env_query_idx: wp.array(dtype=wp.int32),
    # Outputs (pre-zeroed)
    distance: wp.array(dtype=wp.float32),  # [batch * horizon * num_spheres]
    gradient: wp.array(dtype=wp.float32),  # [batch * horizon * num_spheres * 4]
    # Dimensions
    batch_size: wp.int32,
    horizon: wp.int32,
    num_spheres: wp.int32,
    max_n_obs: wp.int32,
    use_multi_env: wp.uint8,
    replacement_cuboid_ids: wp.array(dtype=wp.int32),
    replacement_pair_count: wp.int32,
):
    """Visit populated slots with a bounded lane count, including during graph replay."""
    sph_flat_idx, lane = wp.tid()
    if sph_flat_idx >= batch_size * horizon * num_spheres:
        return
    if spheres[sph_flat_idx][3] < 0.0:
        return
    b_idx = sph_flat_idx / (horizon * num_spheres)
    env_idx = wp.int32(0)
    if use_multi_env == wp.uint8(1):
        env_idx = env_query_idx[b_idx]
    for obs_local_idx in range(lane, wp.min(obs_set.n_per_env[env_idx], obs_set.max_n), max_n_obs):
        swept_sphere_obstacle_collision_kernel_pair(
            obs_set, spheres, weight, activation_distance, env_query_idx,
            distance, gradient, batch_size, horizon, num_spheres, max_n_obs,
            use_multi_env, replacement_cuboid_ids, replacement_pair_count,
            sph_flat_idx, obs_local_idx,
        )


@wp.kernel(enable_backward=False)
def collect_swept_group_candidates(
    obs_set: Any,
    env_query_idx: wp.array(dtype=wp.int32),
    batch_size: wp.int32,
    horizon: wp.int32,
    max_n_obs: wp.int32,
    use_multi_env: wp.uint8,
    lower_bounds: wp.array(dtype=wp.vec4),
    upper_bounds: wp.array(dtype=wp.vec4),
    groups_per_step: wp.int32,
    obstacle_lower: wp.array(dtype=wp.vec4),
    obstacle_upper: wp.array(dtype=wp.vec4),
    candidate_counts: wp.array(dtype=wp.int32),
    candidates: wp.array2d(dtype=wp.int32),
):
    """Prune whole sphere groups before calling the unchanged pair evaluation."""
    group, lane = wp.tid()
    if group >= batch_size * horizon * groups_per_step:
        return
    lower = lower_bounds[group]
    if lower[3] == 0.0:
        return
    upper = upper_bounds[group]
    step = group / groups_per_step
    b_idx = step / horizon
    env_idx = wp.int32(0)
    if use_multi_env == wp.uint8(1):
        env_idx = env_query_idx[b_idx]
    for obs_local_idx in range(lane, wp.min(obs_set.n_per_env[env_idx], obs_set.max_n), max_n_obs):
        if not is_obs_enabled(obs_set, env_idx, obs_local_idx):
            continue
        obstacle_index = env_idx * obs_set.max_n + obs_local_idx
        world_lower = obstacle_lower[obstacle_index]
        world_upper = obstacle_upper[obstacle_index]
        if lower[3] == 1.0 and world_lower[3] == 1.0:
            separated = False
            for axis in range(3):
                if upper[axis] < world_lower[axis] or lower[axis] > world_upper[axis]:
                    separated = True
            if separated:
                continue
        if lower[3] == 1.0:
            inverse = load_obstacle_transform(obs_set, env_idx, obs_local_idx)
            local_lower = wp.vec3(1.0e30)
            local_upper = wp.vec3(-1.0e30)
            finite = True
            for corner in range(8):
                point = wp.vec3(lower[0], lower[1], lower[2])
                if corner % 2 == 1:
                    point[0] = upper[0]
                if (corner / 2) % 2 == 1:
                    point[1] = upper[1]
                if (corner / 4) % 2 == 1:
                    point[2] = upper[2]
                point = wp.transform_point(inverse, point)
                if (
                    not wp.isfinite(point[0])
                    or not wp.isfinite(point[1])
                    or not wp.isfinite(point[2])
                ):
                    finite = False
                local_lower = wp.min(local_lower, point)
                local_upper = wp.max(local_upper, point)
            if finite and not query_aabb_overlaps(
                obs_set, env_idx, obs_local_idx, local_lower, local_upper, 0.0
            ):
                continue
        slot = wp.atomic_add(candidate_counts, group, 1)
        if slot < candidates.shape[1]:
            candidates[group, slot] = obs_local_idx


@wp.kernel(enable_backward=False)
def grouped_swept_sphere_obstacle_collision_kernel(
    obs_set: Any,
    spheres: wp.array(dtype=wp.vec4),
    weight: wp.array(dtype=wp.float32),
    activation_distance: wp.array(dtype=wp.float32),
    env_query_idx: wp.array(dtype=wp.int32),
    distance: wp.array(dtype=wp.float32),
    gradient: wp.array(dtype=wp.float32),
    batch_size: wp.int32,
    horizon: wp.int32,
    num_spheres: wp.int32,
    max_n_obs: wp.int32,
    use_multi_env: wp.uint8,
    replacement_cuboid_ids: wp.array(dtype=wp.int32),
    replacement_pair_count: wp.int32,
    groups_per_step: wp.int32,
    candidate_counts: wp.array(dtype=wp.int32),
    candidates: wp.array2d(dtype=wp.int32),
):
    """Evaluate each sphere independently; overflow checks all populated obstacles."""
    group, offset = wp.tid()
    count = candidate_counts[group]
    if count == 0:
        return
    step = group / groups_per_step
    sphere = (group % groups_per_step) * SWEEP_GROUP_SIZE + offset
    if sphere >= num_spheres:
        return
    env = wp.int32(0)
    if use_multi_env == wp.uint8(1):
        env = env_query_idx[step / horizon]
    overflow = count > candidates.shape[1]
    end = count
    if overflow:
        end = wp.min(obs_set.n_per_env[env], obs_set.max_n)
    for slot in range(end):
        obstacle = slot
        if not overflow:
            obstacle = candidates[group, slot]
        swept_sphere_obstacle_collision_kernel_pair(
            obs_set, spheres, weight, activation_distance, env_query_idx,
            distance, gradient, batch_size, horizon, num_spheres, max_n_obs,
            use_multi_env, replacement_cuboid_ids, replacement_pair_count,
            step * num_spheres + sphere, obstacle,
        )
