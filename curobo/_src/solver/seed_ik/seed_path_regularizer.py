# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Local LM residual for unnecessary TCP excursion during joint interpolation."""
from __future__ import annotations

import torch

from curobo._src.robot.kinematics.kinematics import Kinematics
from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg


class SeedPathRegularizer:
    """Soft geometric prior on selected seeds, with unmodified global candidates.

    At three interior joint-space fractions, penalize TCP displacement from the
    endpoint chord. Both interior and endpoint positions participate in the
    derivative. No waypoint is a hard constraint, and native feasibility checks
    still decide whether a candidate can be returned.
    """

    def __init__(
        self, robot: KinematicsCfg, weight: float, seed_count: int, device_cfg: DeviceCfg
    ) -> None:
        self.kinematics = Kinematics(robot, compute_spheres=False, compute_jacobian=True)
        self.dof = self.kinematics.get_dof()
        self.num_links = len(self.kinematics.tool_frames)
        self.weight = weight
        self.seed_count = seed_count
        self.device_cfg = device_cfg
        self.reference = None
        self.active = torch.zeros(1, **device_cfg.as_torch_dict())
        self.fractions = torch.tensor([0., .25, .5, .75, 1.],
                                      **device_cfg.as_torch_dict()).view(1, 5, 1)

    def setup_batch_tensors(self, batch_size: int, num_seeds: int) -> None:
        """Reserve per-seed weights outside CUDA graph capture."""
        if self.reference is None or self.reference.shape[0] < batch_size * num_seeds:
            self.reference = torch.zeros(batch_size * num_seeds, self.dof,
                                         **self.device_cfg.as_torch_dict())
        self.batch_size, self.num_seeds = batch_size, num_seeds
        self.count = min(self.seed_count, num_seeds)
        indices = torch.arange(self.count, device=self.device_cfg.device)
        levels = torch.tensor([0., .1, 1., 10.], **self.device_cfg.as_torch_dict())
        self.weights = (self.weight * levels[indices % 4]).sqrt().repeat(batch_size)
        self.weights = self.weights.view(-1, 1, 1, 1)

    def compute(
        self, position: torch.Tensor, start: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return gradient (N,D), residual Jacobian (N,9L,D), and cost (N).

        N is batch times seeds. Only the selected seed prefix requires extra FK;
        other candidates receive exactly zero cost, gradient, and Jacobian rows.
        """
        batch, seeds, dof = self.batch_size, self.num_seeds, self.dof
        gradient = torch.zeros_like(position).view(batch, seeds, dof)
        jacobian = position.new_zeros(batch, seeds, 9 * self.num_links, dof)
        cost = position.new_zeros(batch, seeds)
        if self.count == 0:
            return gradient.flatten(0, 1), jacobian.flatten(0, 1), cost.flatten()
        q = position.view(batch, seeds, dof)[:, :self.count].reshape(-1, dof)
        q0 = start.view(batch, seeds, dof)[:, :self.count].reshape(-1, dof)
        path_q = q0[:, None] + self.fractions * (q - q0)[:, None]
        state = self.kinematics.compute_kinematics(JointState.from_position(
            path_q.reshape(-1, dof), joint_names=self.kinematics.joint_names))
        n = batch * self.count
        p = state.tool_poses.position.view(n, 5, self.num_links, 3)
        j = state.tool_jacobians.view(n, 5, self.num_links, 6, dof)[..., :3, :]
        alpha = self.fractions[:, 1:-1, :, None]
        weights = self.weights * self.active
        residual = weights * (p[:, 1:-1] - p[:, :1] - alpha * (p[:, -1:] - p[:, :1]))
        local_jacobian = (weights[..., None] * alpha[..., None]
                          * (j[:, 1:-1] - j[:, -1:])).reshape(n, -1, dof)
        residual = residual.reshape(n, -1, 1)
        local_gradient = (local_jacobian.transpose(-2, -1) @ residual).squeeze(-1)
        gradient[:, :self.count] = local_gradient.view(batch, self.count, dof)
        jacobian[:, :self.count] = local_jacobian.view(batch, self.count, -1, dof)
        cost[:, :self.count] = (0.5 * residual.square().sum((-2, -1))).view(batch, self.count)
        return gradient.flatten(0, 1), jacobian.flatten(0, 1), cost.flatten()
