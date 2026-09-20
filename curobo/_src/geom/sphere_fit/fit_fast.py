# SPDX-License-Identifier: Apache-2.0

"""Budgeted CPU fitting against finite-view obstacle silhouettes."""

import numpy as np

from curobo._src.geom.sphere_fit._fast_core import Config, fit_spheres
from curobo._src.geom.sphere_fit._fast_partition import propose_parts
from curobo._src.geom.sphere_fit._fast_refine import refine
from curobo._src.util.logging import log_warn


def fast_fit_mesh(mesh, num_spheres):
    """Fit original geometry; partition only when the basic audit fails.

    The count is a budget, not a promise to output exactly that many spheres.
    Audits describe the fit before shared clipping and dtype conversion.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces)
    config = Config(max_spheres=num_spheres)
    initial = fit_spheres(vertices, faces, config)
    partition = {"status": "skipped_basic_targets_met"}
    if not initial["target_met"]:
        parts, partition = propose_parts(vertices, faces, min(8, num_spheres), config.seed, 30)
        if parts is not None:
            initial = fit_spheres(vertices, faces, config, convex_parts=parts)
        else:
            log_warn(f"FAST partition unavailable: {partition}; retaining the audited basic fit")
    final = refine(vertices, faces, initial["centers"], initial["radii"])
    diagnostics = {
        "partition": partition,
        "initial_route": initial["route"],
        "refinement_route": final["route"],
        "pre_clip_audit": final["after"],
        "pre_clip_envelope_relative": final["envelope_relative"],
        "pre_clip_target_met": bool(
            final["after"]["gap_p95_relative"] <= config.sample_gap_tolerance
            and final["after"]["gap_max_relative"] <= 4 * config.sample_gap_tolerance
        ),
    }
    return np.asarray(final["centers"]), np.asarray(final["radii"]), diagnostics
