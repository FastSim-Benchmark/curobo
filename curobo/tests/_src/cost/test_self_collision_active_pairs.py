"""Inactive allocation slots must not change collision costs or lifecycle safety."""
import pytest
import torch

from curobo._src.cost.cost_self_collision import SelfCollisionCost
from curobo._src.cost.cost_self_collision_cfg import SelfCollisionCostCfg
from curobo._src.robot.types.self_collision_params import SelfCollisionKinematicsCfg
from curobo.types import DeviceCfg

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def fixture():
    n = 160
    spheres = torch.zeros((2, 3, n, 4), device="cuda")
    spheres[..., 0] = torch.arange(n, device="cuda") * .5
    spheres[..., 3] = -100
    spheres[..., :8, 3] = .01
    spheres[..., 1, 0] = .005
    padding = torch.zeros(n, device="cuda")
    pairs = torch.combinations(torch.arange(n, device="cuda", dtype=torch.int16))
    params = SelfCollisionKinematicsCfg(n, padding, pairs)
    def make(store):
        cost = SelfCollisionCost(SelfCollisionCostCfg(
            weight=torch.tensor([1.0], device="cuda"), device_cfg=DeviceCfg(device="cuda:0"),
            self_collision_kin_config=params, use_grad_input=True, store_pair_distance=store))
        cost.setup_batch_tensors(2, 3)
        return cost
    return spheres, params, make(False), make(True)


def evaluate(cost, spheres):
    x = spheres.clone().requires_grad_(True)
    value = cost.forward(x)
    value.sum().backward()
    return value.detach().clone(), x.grad.detach().clone()


def equal(cost, reference, spheres):
    actual = evaluate(cost, spheres)
    expected = evaluate(reference, spheres)
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    return actual


def test_support_changes_across_environment_and_time():
    spheres, params, cost, reference = fixture()
    equal(cost, reference, spheres)
    selected = cost._active_pair_cache[2]
    assert len(selected.collision_pairs) == 28
    equal(cost, reference, spheres)
    assert cost._active_pair_cache[2] is selected
    # The last environment/time activates a previously omitted high-index pair.
    spheres[1, 2, -2:, 3] = .03
    spheres[1, 2, -1, 0] = spheres[1, 2, -2, 0] + .003
    values, _ = equal(cost, reference, spheres)
    assert values[1, 2] > values[0, 0]
    assert cost._active_pair_cache[2] is not selected
    spheres[..., 3] = -100
    values, gradients = equal(cost, reference, spheres)
    assert torch.count_nonzero(values) == torch.count_nonzero(gradients) == 0


def test_padding_and_pair_mutation_invalidate_cache():
    spheres, params, cost, reference = fixture()
    original_values, _ = equal(cost, reference, spheres)
    old = cost._active_pair_cache[2]
    spheres[..., -2:, 3] = -.01
    spheres[..., -1, 0] = spheres[..., -2, 0] + .005
    params.sphere_padding[-2:] = .05
    values, _ = equal(cost, reference, spheres)
    assert cost._active_pair_cache[2] is not old
    assert torch.all(values > original_values)
    # Same tensor identity, changed eligibility table.
    params.collision_pairs[:] = torch.tensor([0, 1], dtype=torch.int16, device="cuda")
    equal(cost, reference, spheres)
    assert cost._active_pair_cache[2] is params


def test_diagnostic_columns_and_nonfinite_inputs_keep_full_table():
    spheres, params, cost, reference = fixture()
    assert reference._query_pair_config(spheres) is params
    spheres[..., -1, 0] = float('nan')
    assert cost._query_pair_config(spheres) is params


def test_capture_replay_can_activate_previously_disabled_spheres():
    spheres, params, cost, reference = fixture()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            cost.forward(spheres)
    torch.cuda.current_stream().wait_stream(stream)
    assert cost._active_pair_cache is not None
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = cost.forward(spheres)
    spheres[..., -2:, 3] = .03
    spheres[..., -1, 0] = spheres[..., -2, 0] + .003
    graph.replay()
    torch.cuda.synchronize()
    actual = captured.clone()
    expected = reference.forward(spheres).detach().clone()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
