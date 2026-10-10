"""Check identical collision semantics above the per-block shared-memory limit."""

import pytest
import torch

from curobo._src.curobolib.backends.cuda_core_backend import geometry


def _buffers(spheres, pairs, blocks):
    b, h, n, _ = spheres.shape
    return dict(
        out_distance=torch.zeros((b, h, 1), device="cuda"),
        out_vec=torch.zeros_like(spheres),
        pair_distance=torch.zeros((b, h, len(pairs)), device="cuda"),
        sparse_index=torch.zeros((b, h, n), device="cuda", dtype=torch.uint8),
        block_batch_max_value=torch.zeros((b, h, blocks), device="cuda"),
        block_batch_max_index=torch.zeros((b, h, blocks, 2), device="cuda", dtype=torch.int16),
    )


def _run(spheres, padding, pairs, blocks, store, buffers):
    b, h, n, _ = spheres.shape
    geometry.self_collision_distance(
        **buffers,
        robot_spheres=spheres,
        sphere_padding=padding,
        weight=torch.tensor([2.0], device="cuda"),
        pair_locations=pairs,
        num_blocks_per_batch=blocks,
        max_threads_per_block=128,
        batch_size=b,
        horizon=h,
        nspheres=n,
        num_collision_pairs=len(pairs),
        store_pair_distance=store,
        compute_grad=True,
    )
    torch.cuda.synchronize()


@pytest.mark.parametrize("blocks", [1, 32])
@pytest.mark.parametrize("store", [False, True])
def test_shared_and_global_reads_match(monkeypatch, blocks, store):
    """Padding, disabled spheres, batch indexing and sparse reuse stay identical."""
    spheres = torch.zeros((2, 3, 48, 4), device="cuda")
    spheres[..., 0] = torch.arange(48, device="cuda") * 0.15
    spheres[..., 3] = 0.1
    spheres[1, :, :, 0] *= 2
    spheres[:, 1, :, 1] = torch.arange(48, device="cuda") * 0.03
    spheres[..., 8, 3] = -1
    spheres[..., 21, 3] = 0.35
    padding = torch.linspace(0, 0.015, 48, device="cuda")
    pairs = torch.combinations(torch.arange(48, device="cuda", dtype=torch.int16))
    shared = _buffers(spheres, pairs, blocks)
    global_reads = _buffers(spheres, pairs, blocks)
    _run(spheres, padding, pairs, blocks, store, shared)
    monkeypatch.setattr(geometry, "_device_max_shared_memory", lambda device: 512)
    _run(spheres, padding, pairs, blocks, store, global_reads)
    for key in ("out_distance", "out_vec", "sparse_index", "pair_distance"):
        torch.testing.assert_close(global_reads[key], shared[key], rtol=0, atol=0)
    assert torch.all(global_reads["out_distance"] > 0)
    # The same output allocations must lose stale gradients when collision disappears.
    spheres[..., 0] *= 100
    _run(spheres, padding, pairs, blocks, store, global_reads)
    for key in ("out_distance", "out_vec", "sparse_index"):
        assert torch.count_nonzero(global_reads[key]) == 0


@pytest.mark.parametrize("blocks", [1, 32])
def test_model_exceeding_actual_shared_memory(blocks):
    """High-index pairs remain enabled without requiring all spheres in shared memory."""
    limit = geometry._device_max_shared_memory(torch.cuda.current_device())
    n = (limit - geometry.STATIC_SMEM_OVERHEAD) // 16 + 19
    assert n < 32767
    spheres = torch.zeros((2, 3, n, 4), device="cuda")
    spheres[..., 3] = -1
    spheres[..., -2:, 3] = 0.2
    spheres[..., -1, 0] = 0.1
    padding = torch.zeros(n, device="cuda")
    padding[-1] = 0.03
    pairs = torch.tensor([[n - 2, n - 1]], device="cuda", dtype=torch.int16)
    buffers = _buffers(spheres, pairs, blocks)
    _run(spheres, padding, pairs, blocks, True, buffers)
    torch.testing.assert_close(
        buffers["out_distance"], torch.full((2, 3, 1), 0.1749, device="cuda")
    )
    expected = torch.zeros_like(spheres)
    expected[..., -2, 0] = 0.2
    expected[..., -1, 0] = -0.2
    expected[..., -2:, 3] = -2
    torch.testing.assert_close(buffers["out_vec"], expected)
    torch.testing.assert_close(buffers["pair_distance"], buffers["out_distance"])


def test_shared_memory_boundary(monkeypatch):
    """Account for static reduction storage at the exact device boundary."""
    monkeypatch.setattr(geometry, "_device_max_shared_memory", lambda device: 1288)
    assert geometry._sphere_shared_memory_size(64) == 1024
    assert geometry._sphere_shared_memory_size(65) == 0


def test_global_reads_cuda_graph_replay(monkeypatch):
    """Graph replay reads changed spheres and clears the previous collision gradient."""
    monkeypatch.setattr(geometry, "_device_max_shared_memory", lambda device: 512)
    spheres = torch.zeros((1, 1, 32, 4), device="cuda")
    spheres[..., 3] = -1
    spheres[..., :2, 3] = 0.2
    spheres[..., 1, 0] = 0.1
    padding = torch.zeros(32, device="cuda")
    pairs = torch.tensor([[0, 1]], device="cuda", dtype=torch.int16)
    buffers = _buffers(spheres, pairs, 1)
    weight = torch.tensor([2.0], device="cuda")

    def launch():
        geometry.self_collision_distance(
            **buffers,
            robot_spheres=spheres,
            sphere_padding=padding,
            weight=weight,
            pair_locations=pairs,
            num_blocks_per_batch=1,
            max_threads_per_block=128,
            batch_size=1,
            horizon=1,
            nspheres=32,
            num_collision_pairs=1,
            store_pair_distance=True,
            compute_grad=True,
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        launch()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    graph.replay()
    torch.cuda.synchronize()
    assert buffers["out_distance"].item() == pytest.approx(0.15)
    spheres[..., 1, 0] = 2
    graph.replay()
    torch.cuda.synchronize()
    assert torch.count_nonzero(buffers["out_vec"]) == 0
    assert buffers["out_distance"].item() == 0


@pytest.mark.parametrize("batch,horizon", [(2, 3), (2, 1), (1, 3)])
@pytest.mark.parametrize("global_reads", [False, True])
def test_autograd_scales_each_batch_and_timestep(monkeypatch, batch, horizon, global_reads):
    """Incoming scalar cost gradients broadcast over spheres and coordinates only."""
    from curobo._src.curobolib.cuda_ops.geometry import SelfCollisionDistance

    if global_reads:
        monkeypatch.setattr(geometry, "_device_max_shared_memory", lambda device: 512)
    spheres = torch.zeros((batch, horizon, 32, 4), device="cuda")
    spheres[..., 3] = -1
    spheres[..., :2, 3] = 0.2
    spheres[..., 1, 0] = 0.1
    spheres.requires_grad_(True)
    padding = torch.zeros(32, device="cuda")
    pairs = torch.tensor([[0, 1]], device="cuda", dtype=torch.int16)
    buffers = _buffers(spheres, pairs, 1)
    result = SelfCollisionDistance.apply(
        spheres,
        buffers["out_distance"],
        buffers["out_vec"],
        buffers["pair_distance"],
        buffers["sparse_index"],
        torch.tensor([2.0], device="cuda"),
        padding,
        pairs,
        buffers["block_batch_max_value"],
        buffers["block_batch_max_index"],
        1,
        128,
        True,
        True,
    )
    coefficients = torch.arange(1, batch * horizon + 1, device="cuda").reshape(batch, horizon, 1)
    (result * coefficients).sum().backward()
    expected = torch.zeros_like(spheres)
    expected[..., 0, 0] = 0.2
    expected[..., 1, 0] = -0.2
    expected[..., :2, 3] = -2
    expected *= coefficients.unsqueeze(-1)
    torch.testing.assert_close(spheres.grad, expected)
