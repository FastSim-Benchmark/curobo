# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

# Standard Library
from dataclasses import replace
from typing import TYPE_CHECKING, Optional

# Third Party
import torch

from curobo._src.cost.cost_base import BaseCost

# CuRobo
from curobo._src.curobolib.cuda_ops.geometry import SelfCollisionDistance
from curobo._src.util.logging import log_and_raise

if TYPE_CHECKING:
    # CuRobo
    from curobo._src.cost.cost_self_collision_cfg import SelfCollisionCostCfg


class SelfCollisionCost(BaseCost):
    def __init__(self, config: SelfCollisionCostCfg):
        if config.self_collision_kin_config is None:
            log_and_raise("SelfCollisionCostConfig must contain self_collision_kin_config")
        super().__init__(config)

        self._batch_size = None
        self._active_pair_cache = None
        self._active_pair_buffers = None

    def _query_pair_config(self, robot_spheres):
        """Prune only pairs disabled in every query, outside graph capture.

        Captured graphs and pair-distance diagnostics retain the full table so
        replay can activate arbitrary spheres and diagnostic columns stay stable.
        Radius support is checked on every eager call, including every environment
        and trajectory sample. No position-based collision exclusion is cached.
        """
        config = self.config.self_collision_kin_config
        pairs = config.collision_pairs
        padding = config.sphere_padding
        if (
            self.config.store_pair_distance
            or pairs.shape[0] < 4096
            or not robot_spheres.is_cuda
            or torch.cuda.is_current_stream_capturing()
            or torch.compiler.is_compiling()
            or torch.is_inference(pairs)
            or torch.is_inference(padding)
        ):
            return config
        with torch.no_grad():
            # Preserve the original kernel's behavior for nonfinite inputs.
            if not bool(torch.isfinite(robot_spheres).all()):
                return config
            effective = robot_spheres[..., 3] + padding
            # Retain nonfinite padding rather than using it to exclude a pair.
            active = (~(effective < 0)).any(dim=(0, 1))
            key = (id(config), id(pairs), pairs._version, id(padding), padding._version)
            cached = self._active_pair_cache
            if cached is not None and cached[0] == key and torch.equal(cached[1], active):
                return cached[2]
            retained = []
            for chunk in pairs.split(1_000_000):
                indices = chunk.long()
                retained.append(chunk[active[indices[:, 0]] & active[indices[:, 1]]])
            filtered = torch.cat(retained).contiguous()
            # The native reduction expects at least one block. Empty tables use
            # the original disabled-pair path, which also clears stale gradients.
            selected = (
                replace(config, collision_pairs=filtered)
                if 0 < filtered.shape[0] < pairs.shape[0]
                else config
            )
            self._active_pair_cache = (key, active, selected, config, pairs, padding)
            return selected

    def setup_batch_tensors(self, batch_size: int, horizon: int):
        # Assuming n stays constant

        num_spheres = self.config.self_collision_kin_config.num_spheres
        sphere_dim = 4

        if self._batch_size is None or self._batch_size != batch_size or self._horizon != horizon:
            # b, h, n, k = robot_spheres.shape
            self._out_distance = torch.zeros(
                (batch_size, horizon, 1),
                device=self.device_cfg.device,
                dtype=self.device_cfg.dtype,
            )
            self._out_grad = torch.zeros(
                (batch_size, horizon, num_spheres, sphere_dim),
                device=self.device_cfg.device,
                dtype=self.device_cfg.dtype,
            )

            # Sparse index stores index of gradients that are non-zero.
            self._sparse_sphere_idx = torch.zeros(
                (batch_size, horizon, num_spheres),
                device=self.device_cfg.device,
                dtype=torch.uint8,
            )
            if self.config.store_pair_distance:
                num_collision_pairs = self.config.self_collision_kin_config.collision_pairs.shape[0]
                self._pair_distance = torch.zeros(
                    (batch_size, horizon, num_collision_pairs),
                    device=self.device_cfg.device,
                    dtype=self.device_cfg.dtype,
                )
            else:
                self._pair_distance = torch.zeros(
                    (1),
                    device=self.device_cfg.device,
                    dtype=self.device_cfg.dtype,
                )

            # calculate block parallelization parameters:

            self._block_batch_max_index = torch.full(
                (
                    batch_size,
                    horizon,
                    self.config.self_collision_kin_config.num_blocks_per_batch,
                    2,
                ),
                fill_value=0,
                device=self.device_cfg.device,
                dtype=torch.int16,
            )
            self._block_batch_max_value = torch.full(
                (batch_size, horizon, self.config.self_collision_kin_config.num_blocks_per_batch),
                fill_value=0,
                device=self.device_cfg.device,
                dtype=self.device_cfg.dtype,
            )
            super().setup_batch_tensors(batch_size, horizon)

    def forward(
        self,
        robot_spheres: torch.Tensor,
    ):
        """Compute the self collision cost.

        This will return the largest penetration distance between spheres across the batch and
        horizon. The sphere pairs are chosen using
        self.config.self_collision_kin_config (SelfCollisionKinematicsCfg).

        Args:
            robot_spheres: The robot spheres. This is of shape (batch, horizon, num_spheres, 4),
            with last dimension being (position_x, position_y, position_z, radius) in meters.

        Returns:
            The self collision cost.
        """
        self.validate_input(robot_spheres)

        pair_config = self._query_pair_config(robot_spheres)
        block_values = self._block_batch_max_value
        block_indices = self._block_batch_max_index
        if pair_config is not self.config.self_collision_kin_config:
            shape = (self._batch_size, self._horizon, pair_config.num_blocks_per_batch)
            if self._active_pair_buffers is None or self._active_pair_buffers[0].shape != shape:
                self._active_pair_buffers = (
                    torch.zeros(shape, device=robot_spheres.device, dtype=robot_spheres.dtype),
                    torch.zeros((*shape, 2), device=robot_spheres.device, dtype=torch.int16),
                )
            block_values, block_indices = self._active_pair_buffers

        dist = SelfCollisionDistance.apply(
            robot_spheres,
            self._out_distance,
            self._out_grad,
            self._pair_distance,
            self._sparse_sphere_idx,
            self._weight,
            self.config.self_collision_kin_config.sphere_padding,
            pair_config.collision_pairs,
            block_values,
            block_indices,
            pair_config.num_blocks_per_batch,
            pair_config.max_threads_per_block,
            self.config.store_pair_distance,
            self.config.use_grad_input,
        )
        if self.config.convert_to_binary:
            dist = torch.clamp(dist, max=1.0)
            dist = torch.where(dist > 0, dist + 1.0, dist)

        return dist

    def reset(
        self,
        reset_problem_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ):


        super().reset(
            reset_problem_ids=reset_problem_ids,
            **kwargs,
        )

    def validate_input(
        self, robot_spheres: torch.Tensor
    ):
        if robot_spheres.ndim != 4:
            log_and_raise("robot_spheres must be of shape (batch, horizon, num_spheres, 4)")
        if robot_spheres.shape[3] != 4:
            log_and_raise("robot_spheres must have 4 dimensions (x, y, z, radius)")
        if robot_spheres.shape[1] != self._horizon:
            log_and_raise("robot_spheres.shape[1] must be equal to self._horizon")
        if robot_spheres.shape[2] != self.config.self_collision_kin_config.num_spheres:
            log_and_raise(
                "robot_spheres.shape[2] must be equal to self.config.self_collision_kin_config.num_spheres"
            )
        if robot_spheres.shape[0] != self._batch_size:
            log_and_raise("robot_spheres.shape[0] must be equal to self._batch_size")
        if self.config.store_pair_distance:
            if self._pair_distance.ndim != 3:
                log_and_raise(
                    "self._pair_distance must be of shape (batch, horizon, num_collision_pairs)"
                )
            if self._pair_distance.shape[0] != self._batch_size:
                log_and_raise("self._pair_distance.shape[0] must be equal to self._batch_size")
            if self._pair_distance.shape[1] != self._horizon:
                log_and_raise("self._pair_distance.shape[1] must be equal to self._horizon")
            if (
                self._pair_distance.shape[2]
                != self.config.self_collision_kin_config.collision_pairs.shape[0]
            ):
                log_and_raise(
                    "self._pair_distance.shape[2] must be equal to self.config.self_collision_kin_config.collision_pairs.shape[0]"
                )
