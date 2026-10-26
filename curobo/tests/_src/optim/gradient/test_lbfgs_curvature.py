# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from curobo._src.curobolib.cuda_ops.optimization import LBFGScu
from curobo._src.optim.components.quasi_newton_buffers import QuasiNewtonBuffers
from curobo._src.optim.gradient.lbfgs_jit_helpers import jit_lbfgs_compute_step_direction
from curobo._src.types.device_cfg import DeviceCfg

@pytest.mark.parametrize("backend", ["torch", "cuda_shared", "cuda_global"])
@pytest.mark.parametrize("sign", [1., -1.])
def test_lbfgs_curvature_produces_descent(backend: str, sign: float) -> None:
    """Negative curvature must not introduce an uphill quasi-Newton direction."""
    buffers = QuasiNewtonBuffers(DeviceCfg(), 7)
    buffers.resize(1, 14)
    q = torch.zeros(1, 14, device="cuda")
    q[:, 0] = 1.
    gradient = sign*q[:, None]
    if backend == "torch":
        buffers.update(q, gradient)
        step = jit_lbfgs_compute_step_direction(buffers.alpha, buffers.rho, buffers.y,
            buffers.s, gradient, 7, .01, True).view(1, 14)
    else:
        step = LBFGScu.apply(buffers.step_q_buffer, buffers.rho, buffers.y, buffers.s,
            q, gradient, buffers.x_0, buffers.grad_0, .01, True, backend == "cuda_shared")
    assert (step.flatten()*gradient.flatten()).sum() < 0.
    assert torch.all(buffers.rho >= 0.)
