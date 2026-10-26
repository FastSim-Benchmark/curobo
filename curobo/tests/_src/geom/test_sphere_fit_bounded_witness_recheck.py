# SPDX-License-Identifier: Apache-2.0

"""Witness insertion must use the current proposal set, without weakening proof."""

import numpy as np
import pytest
import trimesh

from curobo._src.geom.sphere_fit import fit_bounded as module


def _thin_tetrahedron():
    return trimesh.convex.convex_hull(np.array([
        [0., 0., 0.], [.04, 0., 0.], [0., .002, 0.], [0., 0., .002],
    ]))


def test_current_witness_recheck_avoids_exhausting_proposal_capacity():
    """The same true tetrahedron needs three proposals instead of five."""
    mesh = _thin_tetrahedron()
    centers, radii = module._cover(mesh, .0149, 3)
    assert len(radii) == 3
    diagnostics = {
        "sphere_components": [0] * len(radii),
        "max_protrusion_m": .0149,
    }
    module.verify_bounded_result(mesh, centers, radii, diagnostics)
    assert diagnostics["full_volume_covered"]
    assert diagnostics["verified_max_protrusion_m"] <= .0149


def test_insufficient_rechecked_capacity_still_rejected():
    """Witness reuse cannot turn an insufficient capacity into truncation."""
    with pytest.raises(ValueError, match="proposal budget exhausted"):
        module._cover(_thin_tetrahedron(), .0149, 2)


@pytest.mark.parametrize("offset,skip", [(0., True), (-1e-8, True), (1e-8, False)])
def test_recheck_retains_strict_containment_margin(monkeypatch, offset, skip):
    """A point beyond r-margin still spends a new proposal, even inside r."""
    threshold = 1. - module._MARGIN
    point = np.array([threshold + offset, 0., 0.])
    cell = np.array([[0., 0., 0.], point, [5., 0., 0.], [0., 5., 0.]])
    monkeypatch.setattr(module, "_volume_cells", lambda _: cell[None])
    monkeypatch.setattr(module, "_cell_ball", lambda *_: None)
    calls = []

    class NextWitness(Exception):
        pass

    def witness(value, *_):
        calls.append(value.copy())
        if len(calls) == 1:
            return np.zeros(3), 1.
        raise NextWitness

    monkeypatch.setattr(module, "_witness_ball", witness)
    with pytest.raises(NextWitness):
        module._cover(trimesh.creation.box([.1]*3), .0149, 4)
    np.testing.assert_array_equal(calls[1], cell[2] if skip else point)
