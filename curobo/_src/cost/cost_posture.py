# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Joint goal-set residual with explicit free and start-held coordinates."""

import torch

from curobo._src.cost.cost_base import BaseCost
from curobo._src.cost.cost_base_cfg import BaseCostCfg

PATH_CONSTRAINT_PRIORITY = 100.0


class PostureCost(BaseCost):
    """One goal index is selected jointly across all terminal coordinates."""

    def __init__(self, config: BaseCostCfg, dof: int) -> None:
        """Reserve bounded goal buffers before CUDA graph capture."""
        super().__init__(config)
        self.goals = torch.zeros((256, dof), device=self.device_cfg.device)
        self.valid = torch.zeros(256, device=self.device_cfg.device, dtype=torch.bool)
        self.mask = torch.zeros(dof, device=self.device_cfg.device)
        self.held = torch.zeros(dof, device=self.device_cfg.device)
        self.start = torch.zeros(dof, device=self.device_cfg.device)
        self.tolerance = torch.ones(dof, device=self.device_cfg.device)
        self.active = torch.zeros(1, device=self.device_cfg.device)
        self.hold_weight = PATH_CONSTRAINT_PRIORITY if float(config.weight.max()) > 1 else 1.0
        self.disable_cost()

    def configure(
        self,
        goals: torch.Tensor,
        mask: torch.Tensor,
        held: torch.Tensor,
        start: torch.Tensor,
        tolerance: torch.Tensor,
    ) -> None:
        """Copy one query into stable buffers."""
        self.goals.zero_()
        self.goals[: len(goals)].copy_(goals)
        self.valid.zero_()
        self.valid[: len(goals)] = True
        self.mask.copy_(mask)
        self.held.copy_(held)
        self.start.copy_(start)
        self.tolerance.copy_(tolerance)
        self.active.fill_(1)
        self.enable_cost()

    def deactivate(self) -> None:
        """Remove this request's residual from ordinary Cartesian rollouts."""
        self.active.zero_()
        self.disable_cost()

    def forward(self, position: torch.Tensor) -> torch.Tensor:
        """Evaluate terminal goal-set and all-waypoint hold residuals."""
        # Hold undeclared coordinates throughout; apply the goal set only at the
        # endpoint (IK has horizon=1). No averaging of incompatible goal members.
        held = ((position - self.start).abs() - self.tolerance).clamp_min(0) * self.held
        delta = (position[:, -1:, :] - self.goals).abs()
        residual = (delta - self.tolerance).clamp_min(0) * self.mask
        candidates = residual.square().sum(-1).masked_fill(~self.valid, 1e6)
        terminal = candidates.min(-1).values
        endpoint = torch.zeros_like(position[..., 0])
        endpoint[:, -1] = terminal
        return (
            (self.hold_weight * held.square().sum(-1) + endpoint).unsqueeze(-1)
            * self._weight
            * self.active
        )
