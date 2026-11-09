# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""A start-referenced axis constraint reusing the native tool-pose kernel."""

import torch

from curobo._src.cost.cost_tool_pose import ToolPoseCost
from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.types.tool_pose import GoalToolPose


class AxisHoldCost(ToolPoseCost):
    """Keep fixed GPU buffers so changing motion parameters survives graph replay."""

    def __init__(self, config):
        """Allocate one immutable-shape reference shared by all optimization seeds."""
        super().__init__(config)
        position = torch.zeros(
            (1, 1, self.num_links, 1, 3), device=self.device_cfg.device, dtype=torch.float32
        )
        quaternion = torch.zeros(
            (1, 1, self.num_links, 1, 4), device=self.device_cfg.device, dtype=torch.float32
        )
        quaternion[..., 0] = 1.0
        self.reference = GoalToolPose(self.tool_frames, position, quaternion)
        self.set_hold({}, None)

    def set_hold(self, holds, starting_poses):
        """Copy per-query axes and reference into buffers, or clear their weights."""
        if not holds:
            criteria = self._stacked_tool_pose_criteria
            criteria.terminal_pose_axes_weight_factor.zero_()
            criteria.non_terminal_pose_axes_weight_factor.zero_()
            criteria.orientation_axis.zero_()
            return
        if starting_poses is not None:
            self.reference.quaternion.copy_(starting_poses.quaternion[:1, :1].unsqueeze(3))
        criteria = {}
        for frame in self.tool_frames:
            hold = holds.get(frame)
            if hold is None:
                criteria[frame] = ToolPoseCriteria(
                    terminal_pose_axes_weight_factor=[0.0] * 6,
                    non_terminal_pose_axes_weight_factor=[0.0] * 6,
                    device_cfg=self.device_cfg,
                )
            else:
                item = ToolPoseCriteria.hold_axis(
                    list(hold.axis), hold.tolerance_rad, self.device_cfg
                )
                item.terminal_pose_axes_weight_factor[:3] = 0.0
                criteria[frame] = item
        self.update_tool_pose_criteria(criteria)

    def forward(self, current_tool_poses):
        """Evaluate the axis residual independently of the motion's goal registry."""
        return super().forward(current_tool_poses, self.reference, self.reference_indices)[0]

    def setup_batch_tensors(self, batch_size, horizon, **kwargs):
        """Index every seed's samples into the same captured reference pose."""
        if batch_size != self._batch_size:
            self.reference_indices = torch.zeros(
                (batch_size, 1), dtype=torch.int32, device=self.device_cfg.device
            )
        super().setup_batch_tensors(batch_size, horizon, **kwargs)
