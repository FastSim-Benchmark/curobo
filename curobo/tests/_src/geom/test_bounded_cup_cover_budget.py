# SPDX-License-Identifier: Apache-2.0
"""A rotated physical cup cover must not depend on a greedy dead end."""

import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import curobo._src.geom.sphere_fit.fit_bounded as bounded


def fixture():
    path = Path(__file__).parent / "fixtures/bounded-cup-rotation-cover.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        case = json.load(stream)
    return [int(mask, 16) for mask in case["masks_hex"]], case["count"], case["capacity"]


def test_relaxation_recovers_certified_cup_without_branching(monkeypatch):
    masks, count, capacity = fixture()
    monkeypatch.setattr(bounded, "_MAX_UNION_SEARCH_NODES", 0)
    selected = bounded._select_union_cover(masks, count, capacity)
    coverage = 0
    for index in selected:
        coverage |= masks[index]
    assert len(selected) <= capacity
    assert coverage == (1 << count) - 1


def test_default_budget_finds_complete_40_sphere_cover():
    masks, count, capacity = fixture()
    selected = bounded._select_union_cover(masks, count, capacity)
    coverage = 0
    for index in selected:
        coverage |= masks[index]
    assert len(selected) <= 40
    assert coverage == (1 << count) - 1


@pytest.mark.parametrize("shift", [1, 39, 101])
def test_reordered_cup_candidates_keep_capacity(shift):
    masks, count, capacity = fixture()
    masks = masks[shift:] + masks[:shift]
    selected = bounded._select_union_cover(masks, count, capacity)
    coverage = 0
    for index in selected:
        coverage |= masks[index]
    assert len(selected) <= capacity
    assert coverage == (1 << count) - 1


def test_fractional_relaxation_does_not_certify_an_impossible_capacity():
    with pytest.raises(ValueError, match="capacity"):
        bounded._select_union_cover([0b011, 0b110, 0b101], 3, 1)


def test_relaxation_failure_is_explicit(monkeypatch):
    monkeypatch.setattr(bounded, "linprog", lambda *a, **kw: SimpleNamespace(success=False))
    with pytest.raises(ValueError, match="relaxation failed or exhausted"):
        bounded._select_union_cover([0b001111, 0b010011, 0b101100], 6, 2)
