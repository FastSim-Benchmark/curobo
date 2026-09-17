# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Kernel selection must distinguish history storage from kernel capacity."""

from types import SimpleNamespace

import pytest
import torch

from curobo._src.curobolib.backends.cuda_core_backend.optimization_config import LBFGSLaunchCfg
from curobo._src.curobolib.cuda_ops.optimization import LBFGScu
from curobo._src.optim.components.quasi_newton_buffers import QuasiNewtonBuffers
from curobo._src.optim.gradient import lbfgs
from curobo._src.optim.gradient.lbfgs_jit_helpers import jit_lbfgs_compute_step_direction
from curobo._src.types.device_cfg import DeviceCfg


@pytest.mark.parametrize(
    "dimension,history,requested,shared,expected_kernel,expected_shared",
    [
        (196, 27, True, True, True, True),
        (301, 27, True, True, True, True),
        (302, 27, True, True, True, False),
        (320, 27, True, True, True, False),
        (832, 27, True, True, True, False),
        (1023, 31, True, True, True, False),
        (1024, 27, True, True, False, True),
        (320, 32, True, True, False, True),
        (832, 27, False, True, False, False),
        (196, 27, True, False, True, False),
    ],
)
def test_selects_global_memory_before_disabling_kernel(
    monkeypatch, dimension, history, requested, shared, expected_kernel, expected_shared
):
    """Shared-memory overflow retains a supported native global-memory route."""
    class CoreStub:
        def __init__(self, config, *args, **kwargs):
            self.config = config

        def update_num_problems(self, count):
            pass

        def finish_init(self):
            pass

    monkeypatch.setattr(lbfgs, "GradientOptCore", CoreStub)
    config = lbfgs.LBFGSOptCfg(
        device_cfg=DeviceCfg(device=torch.device("cpu")), history=history,
        use_cuda_kernel_step_direction=requested, use_cuda_kernel_shared_buffers=shared,
    )
    rollout = SimpleNamespace(action_horizon=1, action_dim=dimension)
    lbfgs.LBFGSOpt(config, [rollout, rollout])
    assert config.use_cuda_kernel_step_direction is expected_kernel
    assert config.use_cuda_kernel_shared_buffers is expected_shared
    assert config.history == history
    assert config.num_iters == 100 and config.stable_mode


@pytest.mark.parametrize("dimension", [302, 320, 832, 1023])
@pytest.mark.parametrize("request_shared", [False, True])
def test_global_kernel_launch_only_reserves_alpha(dimension, request_shared):
    """Fallback launches reserve only the dynamic alpha vector."""
    config, shared, _ = LBFGSLaunchCfg.calculate_config(2, dimension, 27, request_shared)
    assert not shared
    assert config.shmem_size == 27 * 4


@pytest.mark.parametrize("dimension", [320, 832, 1023])
@pytest.mark.parametrize("request_shared", [False, True])
def test_global_kernel_matches_eager_two_loop_and_updates_history(dimension, request_shared):
    """Compare every history buffer and direction beyond a full history cycle."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for native kernel comparison")
    device = DeviceCfg(device=torch.device("cuda:0"))
    reference = QuasiNewtonBuffers(device, 27)
    native = QuasiNewtonBuffers(device, 27)
    for buffers in (reference, native):
        buffers.resize(2, dimension)
    generator = torch.Generator(device="cuda").manual_seed(72819)
    positions = torch.zeros((2, dimension), device="cuda")
    weights = torch.linspace(0.1, 2.0, dimension, device="cuda").unsqueeze(0)
    for iteration in range(35):
        if iteration:
            positions = positions + 0.01 * torch.randn(
                positions.shape, device="cuda", generator=generator
            )
        gradient = (positions * weights).unsqueeze(1)
        reference.update(positions, gradient)
        expected = jit_lbfgs_compute_step_direction(
            reference.alpha, reference.rho, reference.y, reference.s,
            gradient, 27, 0.01, True,
        ).squeeze(-1)
        actual = LBFGScu.apply(
            native.step_q_buffer, native.rho, native.y, native.s,
            positions, gradient, native.x_0, native.grad_0, 0.01, True, request_shared,
        )
        # Parallel reductions differ in rounding; both paths retain float32,
        # stable mode, the same history length and identical captured inputs.
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
        torch.testing.assert_close(native.rho, reference.rho, rtol=2e-4, atol=2e-5)
        for name in ("s", "y", "x_0", "grad_0"):
            assert torch.equal(getattr(native, name), getattr(reference, name))


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("case", ["negative_curvature", "zero_curvature", "tiny_curvature"])
def test_stable_kernel_matches_finite_rho_and_gamma_rules(shared, case):
    """Native stable mode uses the eager finite-rho and epsilon-gamma rules."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = DeviceCfg(device=torch.device("cuda:0"))
    reference = QuasiNewtonBuffers(device, 7)
    native = QuasiNewtonBuffers(device, 7)
    for buffers in (reference, native):
        buffers.resize(2, 64)
    q = torch.ones((2, 64), device="cuda")
    if case == "negative_curvature":
        gradient = -q.unsqueeze(1)
    elif case == "zero_curvature":
        gradient = q.unsqueeze(1)
        reference.grad_0.fill_(1)
        native.grad_0.fill_(1)
    else:
        # Products underflow in both IEEE float32 and the native FTZ kernel.
        q.fill_(1e-25)
        gradient = q.unsqueeze(1)
    reference.update(q, gradient)
    expected = jit_lbfgs_compute_step_direction(
        reference.alpha, reference.rho, reference.y, reference.s,
        gradient, 7, 0.01, True,
    ).squeeze(-1)
    actual = LBFGScu.apply(
        native.step_q_buffer, native.rho, native.y, native.s,
        q, gradient, native.x_0, native.grad_0, 0.01, True, shared,
    )
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(native.rho, reference.rho, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize("dof", [10, 26])
@pytest.mark.parametrize("use_graph", [False, True])
def test_wide_optimizer_converges_with_kernel_and_eager(dof, use_graph):
    """Both routes converge from multiple seeds, including captured replay."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    from curobo.tests._src.optim.gradient.test_lbfgs import MockRollout, cost_fn

    device = DeviceCfg(device=torch.device("cuda:0"))
    solvers = []
    for native in (False, True):
        config = lbfgs.LBFGSOptCfg(
            device_cfg=device, num_problems=2, num_iters=50, history=27,
            use_cuda_kernel_step_direction=native, stable_mode=True,
            line_search_scale=[0.0, 0.1, 0.5, 1.0],
        )
        rollout = MockRollout(num_dof=dof, action_horizon=32, batch_size=2)
        optimizer = lbfgs.LBFGSOpt(config, [rollout, rollout], use_cuda_graph=use_graph)
        assert config.use_cuda_kernel_step_direction is native
        assert not config.use_cuda_kernel_shared_buffers
        solvers.append(optimizer)
    generator = torch.Generator(device="cuda").manual_seed(28974)
    for _ in range(3):
        initial = torch.randn((2, 32, dof), device="cuda", generator=generator)
        results = []
        for optimizer in solvers:
            optimizer.reinitialize(initial.clone())
            result = optimizer.optimize(initial.clone()).clone()
            assert torch.isfinite(result).all()
            assert cost_fn(result).max() < 1e-5
            results.append(result)
        torch.testing.assert_close(results[0], results[1], rtol=1e-4, atol=1e-4)


def test_unstable_optimizer_mode_remains_rejected():
    """The public optimizer continues to require numerical stability mode."""
    with pytest.raises(ValueError, match="stable_mode must be true"):
        lbfgs.LBFGSOptCfg(stable_mode=False)
