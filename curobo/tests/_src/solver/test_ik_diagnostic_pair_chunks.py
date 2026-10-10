import pytest
import torch

from curobo._src.solver.ik_diagnostics import _bounded_pair_overlaps as bounded_pairs


@pytest.mark.parametrize("chunk_size", [1, 7, 128, 4096])
@pytest.mark.parametrize("inactive", [False, True])
def test_chunk_reduction_matches_full_arithmetic_and_stable_order(chunk_size, inactive):
    generator = torch.Generator().manual_seed(731)
    sample = torch.randn((24, 4), generator=generator, dtype=torch.float64)
    sample[:, 3] = 2.0
    sample[:5, :3] = 0.0  # Equal overlaps across multiple chunks.
    if inactive:
        sample[::3, 3] = -1.0
    padding = torch.zeros(24, dtype=torch.float64)
    padding[0] = 1.0  # Padding reactivates a zero-effective-radius sphere.
    pairs = torch.triu_indices(24, 24, offset=1).T.to(torch.int16)
    first, second = pairs.long().T
    radii = sample[:, 3] + padding
    distance = ((sample[first, :3] - sample[second, :3]) ** 2).sum(-1)
    sums = radii[first] + radii[second]
    overlap = sums**2 - distance
    positive = torch.nonzero((radii[first] >= 0) & (radii[second] >= 0) & (overlap > 0)).flatten()
    expected = positive[
        torch.argsort(overlap[positive], descending=True, stable=True)[:8]
    ].tolist()
    before = sample.clone()
    count, top, distances, radius_sums, overlaps = bounded_pairs(
        sample, pairs, padding, chunk_size
    )
    assert count == len(positive) and top == expected
    for index in top:
        assert distances[index].item() == distance[index].item()
        assert radius_sums[index].item() == sums[index].item()
        assert overlaps[index].item() == overlap[index].item()
    assert torch.equal(sample, before)


def test_empty_disabled_and_tangent_pairs():
    sample = torch.tensor([[0, 0, 0, 1], [2, 0, 0, 1], [0, 0, 0, -1]], dtype=torch.float64)
    for pairs in [
        torch.empty((0, 2), dtype=torch.int16),
        torch.tensor([[0, 1], [0, 2]], dtype=torch.int16),
    ]:
        assert bounded_pairs(sample, pairs, torch.zeros(3), 1) == (0, [], {}, {}, {})
    with pytest.raises(ValueError):
        bounded_pairs(sample, pairs, torch.zeros(3), 0)
