# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Differentiable sphere/mesh clearances for captured support contacts."""

import torch
import warp as wp

from curobo._src.geom.collision.wp_collision_kernel import (
    compute_local_sdf_with_grad,
    is_obs_enabled,
    load_obstacle_transform,
)
from curobo._src.geom.data.data_mesh import MeshData, MeshDataWarp
from curobo._src.util.warp import get_warp_device_stream


@wp.kernel
def mesh_clearance_kernel(
    meshes: MeshDataWarp,
    spheres: wp.array(dtype=wp.vec4),
    indices: wp.array(dtype=wp.int32),
    count: int,
    gaps: wp.array(dtype=wp.float32),
    gradients: wp.array(dtype=wp.vec4),
):
    """Evaluate every requested sphere/support pair in environment zero."""
    tid = wp.tid()
    sphere = spheres[tid // count]
    index = indices[tid % count]
    gaps[tid] = 1.0e6
    gradients[tid] = wp.vec4(0.0)
    if sphere[3] <= 0.0 or not is_obs_enabled(meshes, 0, index):
        return
    transform = load_obstacle_transform(meshes, 0, index)
    center = wp.vec3(sphere[0], sphere[1], sphere[2])
    local = wp.transform_point(transform, center)
    result = compute_local_sdf_with_grad(meshes, 0, index, local, sphere[3] + 0.01)
    # The collision helper returns the negative SDF gradient.
    gradient = wp.transform_vector(
        wp.transform_inverse(transform), -wp.vec3(result[1], result[2], result[3])
    )
    gaps[tid] = result[0] - sphere[3]
    gradients[tid] = wp.vec4(gradient[0], gradient[1], gradient[2], -1.0)


class MeshClearance(torch.autograd.Function):
    """Reuse the collision scene's BVHs and preserve signed/surface distance mode."""

    @staticmethod
    def forward(ctx, spheres: torch.Tensor, meshes: MeshData, indices: torch.Tensor):
        """Return gaps shaped (*sphere_shape, support_count)."""
        count = indices.numel()
        flat = spheres.contiguous().reshape(-1, 4)
        gaps = torch.empty((len(flat), count), device=spheres.device)
        gradients = torch.empty((len(flat), count, 4), device=spheres.device)
        device, stream = get_warp_device_stream(spheres)
        wp.launch(
            mesh_clearance_kernel,
            dim=len(flat) * count,
            inputs=[
                meshes.to_warp(),
                wp.from_torch(flat.detach(), dtype=wp.vec4),
                wp.from_torch(indices),
                count,
                wp.from_torch(gaps.reshape(-1)),
                wp.from_torch(gradients.reshape(-1, 4), dtype=wp.vec4),
            ],
            device=device,
            stream=stream,
        )
        ctx.save_for_backward(gradients)
        ctx.input_shape = spheres.shape
        return gaps.reshape(*spheres.shape[:-1], count)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        """Accumulate support gradients for each input sphere."""
        (derivative,) = ctx.saved_tensors
        result = (gradient.reshape(derivative.shape[:-1])[..., None] * derivative).sum(1)
        return result.reshape(ctx.input_shape), None, None
