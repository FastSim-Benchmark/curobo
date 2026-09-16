# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Equality coordinates survive every step policy and CUDA graph replay."""

from types import SimpleNamespace

import pytest
import torch

from curobo._src.optim.gradient.line_search_strategy import GreedyLineSearchStrategy


@pytest.mark.parametrize("step_scale", [0.0, 1.0, 0.5])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_search_projects_old_seed_and_holds_directions(step_scale: float, device: str) -> None:
    """Project held coordinates before evaluating seeds or replaying a captured step."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    lower = torch.tensor([0.2, -1.0], device=device)
    upper = torch.tensor([0.2, 1.0], device=device)
    context = SimpleNamespace(
        action_horizon=1, action_dim=2, step_scale=step_scale,
        fix_terminal_action=False, action_lower_bounds=lower, action_upper_bounds=upper,
        action_horizon_step_max=step_scale * (upper - lower),
        line_search_scale=torch.tensor([0.0, 0.5, 1.0], device=device).view(1, 3, 1, 1),
    )
    strategy = GreedyLineSearchStrategy()
    seed = torch.zeros(1, 1, 2, device=device)
    direction = torch.ones_like(seed) * 0.1
    points, step = strategy._prepare_search_points(seed, direction, context)
    assert torch.all(points[..., 0] == lower[0])
    assert torch.all(step[..., 0] == 0)
    assert torch.any(points[..., 1] != 0)
    if device == "cuda":
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            replay_points, replay_step = strategy._prepare_search_points(seed, direction, context)
        # Release one coordinate and hold the other using the same buffers.
        lower.copy_(torch.tensor([-1.0, 0.3], device=device))
        upper.copy_(torch.tensor([1.0, 0.3], device=device))
        context.action_horizon_step_max.copy_(step_scale * (upper - lower))
        graph.replay()
        assert torch.all(replay_points[..., 1] == lower[1])
        assert torch.all(replay_step[..., 1] == 0)
        assert torch.any(replay_points[..., 0] != 0)
