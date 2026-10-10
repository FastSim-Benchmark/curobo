# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""cuda.core backend for geometry/collision kernels.

Only self-collision uses CUDA kernels. World collision (OBB, voxel) uses Warp.
"""

# Standard Library
from functools import lru_cache

# Third Party
import torch
from cuda.core import LaunchConfig

# CuRobo
from curobo._src.context import get_runtime
from curobo._src.curobolib.backends.cuda_core_backend.geometry_config import GeometryKernelCfg
from curobo._src.curobolib.backends.cuda_core_backend.launch_helper import launch_kernel
from curobo._src.curobolib.backends.cuda_core_backend.util import ceil_div
from curobo._src.util.logging import log_and_raise, log_info

COLLISION_PAIR_SIZE = 8  # sizeof(CollisionPair): float(4) + int16(2) + int16(2)
STATIC_SMEM_OVERHEAD = COLLISION_PAIR_SIZE * 33  # block_reduce_shared_data[32] + reduced_max_d


@lru_cache(maxsize=None)
def _device_max_shared_memory(device_id: int) -> int:
    import cuda.bindings.runtime as cudart

    (err, device_max_smem) = cudart.cudaDeviceGetAttribute(
        cudart.cudaDeviceAttr.cudaDevAttrMaxSharedMemoryPerBlockOptin, device_id
    )
    if err != cudart.cudaError_t.cudaSuccess:
        log_and_raise(f"Failed to query device max shared memory: {err}")
    return device_max_smem


@lru_cache(maxsize=64)
def _report_global_sphere_reads(device_id: int, nspheres: int) -> None:
    log_info(
        f"Self-collision uses global-memory sphere reads for {nspheres} spheres "
        f"on CUDA device {device_id}; all configured collision pairs remain active."
    )


def _sphere_shared_memory_size(nspheres: int) -> int:
    device_id = torch.cuda.current_device()
    size = 16 * nspheres
    if size + STATIC_SMEM_OVERHEAD <= _device_max_shared_memory(device_id):
        return size
    _report_global_sphere_reads(device_id, nspheres)
    return 0


def _validate_and_configure_shared_memory(
    kernel, dynamic_smemsize: int, nspheres: int, kernel_name: str
):
    import cuda.bindings.runtime as cudart

    total_smem = dynamic_smemsize + STATIC_SMEM_OVERHEAD
    device_max_smem = _device_max_shared_memory(torch.cuda.current_device())

    if total_smem > device_max_smem:
        max_nspheres = (device_max_smem - STATIC_SMEM_OVERHEAD) // 16  # sizeof(float4)
        log_and_raise(
            f"Self-collision kernel {kernel_name} requires {total_smem} bytes of shared "
            f"memory ({dynamic_smemsize} dynamic + {STATIC_SMEM_OVERHEAD} static) for "
            f"{nspheres} spheres, but device supports at most {device_max_smem} bytes. "
            f"Reduce nspheres to at most {max_nspheres}"
        )

    if dynamic_smemsize > 48000:
        kernel_ptr = int(kernel._handle)
        err = cudart.cudaFuncSetAttribute(
            kernel_ptr,
            cudart.cudaFuncAttribute.cudaFuncAttributeMaxDynamicSharedMemorySize,
            dynamic_smemsize,
        )
        if err[0] != cudart.cudaError_t.cudaSuccess:
            log_and_raise(
                f"cudaFuncSetAttribute failed for self-collision kernel {kernel_name}: {err}"
            )


def self_collision_distance(
    out_distance: torch.Tensor,
    out_vec: torch.Tensor,
    pair_distance: torch.Tensor,
    sparse_index: torch.Tensor,
    robot_spheres: torch.Tensor,
    sphere_padding: torch.Tensor,
    weight: torch.Tensor,
    pair_locations: torch.Tensor,
    block_batch_max_value: torch.Tensor,
    block_batch_max_index: torch.Tensor,
    num_blocks_per_batch: int,
    max_threads_per_block: int,
    batch_size: int,
    horizon: int,
    nspheres: int,
    num_collision_pairs: int,
    store_pair_distance: bool,
    compute_grad: bool,
):
    """Launch self-collision distance kernel using cuda.core runtime compilation.

    This function has the EXACT same signature as the PyBind11 version
    to ensure seamless backend swapping.

    Note:
        Modifies output tensors in-place
    """
    runtime = get_runtime()
    cache = runtime.get_cuda_core_cache()
    kernel_config = GeometryKernelCfg()
    smemsize = _sphere_shared_memory_size(nspheres)
    shared = str(smemsize != 0).lower()

    if num_blocks_per_batch == 1:
        num_blocks = batch_size * horizon

        num_threads_per_block = min(max_threads_per_block, num_collision_pairs)

        warp_size = 32
        num_threads_per_block = ceil_div(num_threads_per_block, warp_size) * warp_size
        num_threads_per_block = min(num_threads_per_block, max_threads_per_block)

        config = LaunchConfig(grid=num_blocks, block=num_threads_per_block, shmem_size=smemsize)

        kernel_name = (
            "curobo::geometry::self_collision::self_collision_max_distance_kernel"
            f"<{str(store_pair_distance).lower()}, {shared}>"
        )

        kernel_files = [
            kernel_config.kernel_dir / f for f in kernel_config.get_kernel_files("self_collision")
        ]
        kernel = cache.get_or_compile_kernel(
            source_files=kernel_files,
            kernel_name=kernel_name,
            include_dirs=kernel_config.get_include_dirs(),
            compile_flags=kernel_config.get_compile_flags(),
        )
        _validate_and_configure_shared_memory(kernel, smemsize, nspheres, kernel_name)

        pt_stream = torch.cuda.current_stream(out_distance.device)
        stream = cache.get_stream_wrapper(pt_stream)

        kernel_args = (
            out_distance.data_ptr(),
            out_vec.data_ptr(),
            pair_distance.data_ptr(),
            sparse_index.data_ptr(),
            robot_spheres.data_ptr(),
            sphere_padding.data_ptr(),
            weight.data_ptr(),
            pair_locations.data_ptr(),
            batch_size,
            horizon,
            nspheres,
            num_collision_pairs,
            compute_grad,
        )

        launch_kernel(kernel_name, stream, config, kernel, *kernel_args)

    else:
        num_blocks_first_kernel = batch_size * horizon * num_blocks_per_batch
        num_threads_per_block_first_kernel = max_threads_per_block

        config_first = LaunchConfig(
            grid=num_blocks_first_kernel,
            block=num_threads_per_block_first_kernel,
            shmem_size=smemsize,
        )

        kernel_name_first = (
            "curobo::geometry::self_collision::self_collision_max_block_kernel"
            f"<{str(store_pair_distance).lower()}, {shared}>"
        )

        kernel_files = [
            kernel_config.kernel_dir / f for f in kernel_config.get_kernel_files("self_collision")
        ]
        kernel_first = cache.get_or_compile_kernel(
            source_files=kernel_files,
            kernel_name=kernel_name_first,
            include_dirs=kernel_config.get_include_dirs(),
            compile_flags=kernel_config.get_compile_flags(),
        )
        _validate_and_configure_shared_memory(kernel_first, smemsize, nspheres, kernel_name_first)

        pt_stream = torch.cuda.current_stream(out_distance.device)
        stream = cache.get_stream_wrapper(pt_stream)

        kernel_args_first = (
            out_vec.data_ptr(),
            pair_distance.data_ptr(),
            sparse_index.data_ptr(),
            robot_spheres.data_ptr(),
            sphere_padding.data_ptr(),
            pair_locations.data_ptr(),
            block_batch_max_value.data_ptr(),
            block_batch_max_index.data_ptr(),
            num_blocks_per_batch,
            batch_size,
            horizon,
            nspheres,
            num_collision_pairs,
        )

        launch_kernel(kernel_name_first, stream, config_first, kernel_first, *kernel_args_first)

        num_blocks_second_kernel = batch_size * horizon
        num_threads_per_block_second_kernel = min(512, num_blocks_per_batch)

        config_second = LaunchConfig(
            grid=num_blocks_second_kernel, block=num_threads_per_block_second_kernel, shmem_size=0
        )

        kernel_name_second = "curobo::geometry::self_collision::self_collision_max_reduce_kernel"

        kernel_second = cache.get_or_compile_kernel(
            source_files=kernel_files,
            kernel_name=kernel_name_second,
            include_dirs=kernel_config.get_include_dirs(),
            compile_flags=kernel_config.get_compile_flags(),
        )

        kernel_args_second = (
            out_distance.data_ptr(),
            out_vec.data_ptr(),
            pair_distance.data_ptr(),
            sparse_index.data_ptr(),
            robot_spheres.data_ptr(),
            sphere_padding.data_ptr(),
            weight.data_ptr(),
            pair_locations.data_ptr(),
            block_batch_max_value.data_ptr(),
            block_batch_max_index.data_ptr(),
            num_blocks_per_batch,
            batch_size,
            horizon,
            nspheres,
            num_collision_pairs,
            compute_grad,
        )

        launch_kernel(
            kernel_name_second, stream, config_second, kernel_second, *kernel_args_second
        )
