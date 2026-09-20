# SPDX-License-Identifier: Apache-2.0

"""Deterministic, budgeted collision proxies; no object-category classifiers.

The proxy is an approximate obstacle representation, not a coverage certificate.
All fitting coordinates are normalized internally. Spheres may overlap.
"""

import time
from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import LinearConstraint, minimize
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist


@dataclass(frozen=True)
class Config:
    max_spheres: int = 32
    envelope_tolerance: float = 0.025
    min_sample_coverage: float = 0.97
    fit_samples: int = 4096
    validation_samples: int = 2048
    audit_samples: int = 8192
    vertex_budget: int = 8192
    seed: int = 17
    tight_envelope: bool = True
    sample_gap_tolerance: float = 0.035

    def __post_init__(self):
        if not 1 <= self.max_spheres <= 256:
            raise ValueError("max_spheres must be in [1, 256]")
        if not np.isfinite(self.envelope_tolerance) or self.envelope_tolerance <= 0:
            raise ValueError("envelope_tolerance must be positive and finite")
        if not 0 < self.min_sample_coverage <= 1:
            raise ValueError("min_sample_coverage must be in (0, 1]")
        if not np.isfinite(self.sample_gap_tolerance) or self.sample_gap_tolerance <= 0:
            raise ValueError("sample_gap_tolerance must be finite and positive")
        for name in ("fit_samples", "validation_samples", "audit_samples", "vertex_budget"):
            if getattr(self, name) < 64:
                raise ValueError(f"{name} must be >= 64")


def directions(count=128):
    """Deterministic unit directions, including coordinate diagonals and axes."""
    z = 1 - 2 * (np.arange(count) + 0.5) / count
    phi = np.arange(count) * (np.pi * (3 - np.sqrt(5)))
    fib = np.c_[np.sqrt(1 - z * z) * np.cos(phi), np.sqrt(1 - z * z) * np.sin(phi), z]
    grid = np.array(
        [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if a or b or c]
    )
    return np.concatenate((fib, grid / np.linalg.norm(grid, axis=1)[:, None]))


def sample_surface(vertices, faces, count, seed):
    triangles = vertices[faces]
    area = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
    )
    if not np.isfinite(area).all() or area.sum() <= 0:
        raise ValueError("mesh has no finite, nondegenerate surface")
    rng = np.random.default_rng(seed)
    tri = triangles[rng.choice(len(faces), count, p=area / area.sum())]
    uv = rng.random((count, 2))
    u = np.sqrt(uv[:, 0])
    return (
        (1 - u)[:, None] * tri[:, 0]
        + (u * (1 - uv[:, 1]))[:, None] * tri[:, 1]
        + (u * uv[:, 1])[:, None] * tri[:, 2]
    )


def signed_gaps(points, centers, radii):
    return (cdist(points, centers) - radii).min(axis=1)


def enclosing_ball(points):
    """Choose the tighter of bbox, mean and farthest-pair centered balls.

    Every assigned point is enclosed, but this is not a minimum-ball solver.
    """
    box_center = (points.min(axis=0) + points.max(axis=0)) / 2
    a = points[np.argmax(np.sum((points - box_center) ** 2, axis=1))]
    b = points[np.argmax(np.sum((points - a) ** 2, axis=1))]
    centers = np.array([box_center, points.mean(axis=0), (a + b) / 2])
    radii = cdist(centers, points).max(axis=1)
    k = np.argmin(radii)
    return centers[k], max(float(radii[k]), 1e-9)


def best_split(points, dirs, support):
    if len(points) < 4:
        return None
    center = points.mean(axis=0)
    _, vectors = np.linalg.eigh((points - center).T @ (points - center))
    best = None
    for axis in vectors.T[::-1]:
        coord = (points - center) @ axis
        if np.ptp(coord) < 1e-9:
            continue
        for threshold in (np.median(coord), (coord.min() + coord.max()) / 2):
            mask = coord <= threshold
            if not mask.any() or mask.all():
                continue
            groups = (points[mask], points[~mask])
            balls = [enclosing_ball(p) for p in groups]
            # Prioritize the worst envelope error, then radius. Pure summed
            # volume gain can reject the first necessary split of a flat surface.
            excess = max(float(np.max(c @ dirs.T + r - support)) for c, r in balls)
            cost = excess + 0.05 * max(r for _, r in balls) + 0.01 * sum(r**3 for _, r in balls)
            if best is None or cost < best[0]:
                best = (cost, groups, balls)
    return best


def tighten_balls(points, centers, radii, dirs, support, tolerance):
    """Move centers and fit radii under linear directional-envelope constraints.

    Minimize squared positive gaps to the assigned points. Coverage is soft;
    finite-direction envelope constraints are hard. This is NOT uniform scaling.
    """
    centers, radii = np.array(centers, copy=True), np.array(radii, copy=True)
    labels = (cdist(points, centers) / np.maximum(radii, 1e-12)).argmin(axis=1)
    matrix = np.c_[dirs, np.ones(len(dirs))]
    ceiling = support + tolerance
    outcomes = []
    for i in range(len(radii)):
        if np.max(centers[i] @ dirs.T + radii[i] - support) <= tolerance:
            outcomes.append("already_feasible")
            continue
        part = points[labels == i]
        if not len(part):
            part = points[[np.argmin(np.linalg.norm(points - centers[i], axis=1))]]
        if len(part) > 384:
            # Deterministic subsampling plus coordinate extrema.
            selected = np.random.default_rng(731).choice(len(part), 384, replace=False)
            selected = np.unique(np.r_[selected, part.argmin(axis=0), part.argmax(axis=0)])
            part = part[selected]
        center = part.mean(axis=0)
        cap = float(np.min(ceiling - center @ dirs.T))
        initial = np.r_[center, max(1e-9, min(radii[i], cap - 1e-10))]

        def objective(params, part=part):
            delta = params[:3] - part
            dist = np.maximum(np.linalg.norm(delta, axis=1), 1e-12)
            gaps = np.maximum(dist - params[3], 0)
            loss = np.mean(gaps**2) - 1e-8 * params[3]
            grad = np.r_[
                np.mean(2 * gaps[:, None] * delta / dist[:, None], axis=0), -2 * gaps.mean() - 1e-8
            ]
            return loss, grad

        optimum = minimize(
            objective,
            initial,
            jac=True,
            method="SLSQP",
            constraints=[LinearConstraint(matrix, -np.inf, ceiling)],
            bounds=[(None, None)] * 3 + [(1e-9, None)],
            options={"maxiter": 80, "ftol": 1e-10},
        )
        if not np.isfinite(optimum.x).all():
            raise ValueError("non-finite constrained sphere optimization")
        candidate = optimum.x
        cap = float(np.min(ceiling - candidate[:3] @ dirs.T))
        if cap <= 1e-9:
            # Feasible initialization is explicit and recorded, never inflate.
            candidate = initial
            cap = float(np.min(ceiling - candidate[:3] @ dirs.T))
            outcomes.append("feasible_initialization_used")
        else:
            outcomes.append("converged" if optimum.success else "feasible_iteration_limit")
        centers[i] = candidate[:3]
        radii[i] = min(candidate[3], cap)
    return centers, radii, outcomes


def fit_spheres(vertices, faces, config=None, convex_parts=None):
    """Return centers/radii and an auditable decision record, in input units.

    envelope_tolerance is relative to the longest PCA-aligned extent. The
    measured envelope excess is the exact support excess in a finite set of
    directions; it is NOT a Hausdorff bound. Concavities are intentionally free.
    The independent audit is never used to select or repair a candidate.
    """
    cfg = config or Config()
    started = time.perf_counter()
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces)
    if v.ndim != 2 or v.shape[1] != 3 or len(v) < 3 or not np.isfinite(v).all():
        raise ValueError("vertices must be a finite (N,3) array with N >= 3")
    if (
        f.ndim != 2
        or f.shape[1] != 3
        or len(f) == 0
        or not np.issubdtype(f.dtype, np.integer)
        or f.min() < 0
        or f.max() >= len(v)
    ):
        raise ValueError("faces must be valid integer triangle indices")
    # Remove unused vertices: they are not part of the collision surface.
    used, inverse = np.unique(f.ravel(), return_inverse=True)
    v, f = v[used], inverse.reshape(-1, 3)
    shift = (v.min(axis=0) + v.max(axis=0)) / 2
    rough_scale = np.linalg.norm(np.ptp(v, axis=0))
    if rough_scale <= 0:
        raise ValueError("mesh extent must be positive")
    raw = (v - shift) / rough_scale
    train = sample_surface(raw, f, cfg.fit_samples, cfg.seed)
    covariance = np.cov(train.T)
    eigenvalues, basis = np.linalg.eigh(covariance)
    basis = basis[:, ::-1]
    # Stable signs make serialized results reproducible across eigenvector signs.
    for i in range(3):
        if basis[np.argmax(np.abs(basis[:, i])), i] < 0:
            basis[:, i] *= -1
    x = raw @ basis
    scale = np.ptp(x, axis=0).max()
    x, train = x / scale, train @ basis / scale
    physical_scale = rough_scale * scale
    dirs = directions()
    support = np.full(len(dirs), -np.inf)
    extrema = []
    for chunk in np.array_split(np.arange(len(x)), max(1, int(np.ceil(len(x) / 4096)))):
        dots = x[chunk] @ dirs.T
        support = np.maximum(support, dots.max(axis=0))
        extrema.extend(chunk[dots.argmax(axis=0)].tolist())
    rng = np.random.default_rng(cfg.seed)
    selected = (
        np.arange(len(x))
        if len(x) <= cfg.vertex_budget
        else rng.choice(len(x), cfg.vertex_budget, replace=False)
    )
    points = np.unique(np.concatenate((train, x[np.unique(np.r_[selected, extrema])])), axis=0)
    validation = sample_surface(x, f, cfg.validation_samples, cfg.seed + 1)
    records, candidates = [], []
    partition_diagnostics = None

    def evaluate(route, centers, radii):
        centers, radii = np.asarray(centers), np.asarray(radii)
        optimization = None
        if cfg.tight_envelope:
            centers, radii, optimization = tighten_balls(
                points, centers, radii, dirs, support, cfg.envelope_tolerance
            )
        gaps = signed_gaps(validation, centers, radii)
        excess = np.maximum((centers @ dirs.T + radii[:, None]).max(axis=0) - support, 0)
        coverage = float(np.mean(gaps <= 1e-7))
        envelope = float(excess.max())
        gap95 = float(np.quantile(np.maximum(gaps, 0), 0.95))
        record = {
            "route": route,
            "num_spheres": len(radii),
            "validation_coverage": coverage,
            "validation_gap_p99_relative": float(np.quantile(np.maximum(gaps, 0), 0.99)),
            "validation_gap_p95_relative": gap95,
            "validation_gap_max_relative": float(max(0, gaps.max())),
            "envelope_excess_relative": envelope,
            "summed_sphere_volume_relative": float(4 * np.pi / 3 * np.sum(radii**3)),
            "meets_targets": bool(
                (
                    (
                        gap95 <= cfg.sample_gap_tolerance
                        and max(0, gaps.max()) <= 4 * cfg.sample_gap_tolerance
                    )
                    if cfg.tight_envelope
                    else coverage >= cfg.min_sample_coverage
                )
                and envelope <= cfg.envelope_tolerance + 1e-9
            ),
            "optimization": optimization,
        }
        records.append(record)
        candidates.append((centers.copy(), radii.copy()))
        return record

    c, r = enclosing_ball(points)
    initial = evaluate("single", [c], [r])
    if not initial["meets_targets"]:
        checkpoints = sorted(
            {n for n in (2, 3, 4, 6, 8, 12, 16, 24, cfg.max_spheres) if n <= cfg.max_spheres}
        )
        # Slabs are a candidate, not a category decision. Branches can make them bad.
        for n in checkpoints:
            boundaries = np.linspace(points[:, 0].min(), points[:, 0].max(), n + 1)[1:-1]
            labels = np.searchsorted(boundaries, points[:, 0])
            balls = [enclosing_ball(points[labels == i]) for i in range(n) if np.any(labels == i)]
            evaluate("principal_axis_slabs", [b[0] for b in balls], [b[1] for b in balls])

        # Spatial splitting handles slabs, branches and irregular shapes uniformly.
        leaves = [(points, c, r, best_split(points, dirs, support))]
        for n in range(2, cfg.max_spheres + 1):
            choices = [
                (float(np.max(leaf[1] @ dirs.T + leaf[2] - support)) + 0.01 * leaf[2], i)
                for i, leaf in enumerate(leaves)
                if leaf[3] is not None
            ]
            if not choices:
                break
            _, index = max(choices)
            _, _, _, split = leaves.pop(index)
            for part, (pc, pr) in zip(split[1], split[2], strict=True):
                leaves.append((part, pc, pr, best_split(part, dirs, support)))
            if n in checkpoints:
                evaluate("adaptive_spatial_split", [p[1] for p in leaves], [p[2] for p in leaves])

        if convex_parts is not None:
            if not 1 <= len(convex_parts) <= cfg.max_spheres:
                raise ValueError("convex partition count must be within the sphere budget")
            # CoACD partitions are only proposals. Assign the ORIGINAL fitting
            # points to their closest convex part; never fit a repaired mesh as
            # if it were the source. This also covers points lost in preprocessing.
            distances = []
            for part in convex_parts:
                part = (np.asarray(part) - shift) / rough_scale @ basis / scale
                hull = ConvexHull(part)
                equations = hull.equations
                distances.append((points @ equations[:, :3].T + equations[:, 3]).max(axis=1))
            labels = np.argmin(np.stack(distances, axis=1), axis=1)
            leaves = []
            for i in range(len(convex_parts)):
                cluster = points[labels == i]
                if not len(cluster):
                    continue
                pc, pr = enclosing_ball(cluster)
                leaves.append((cluster, pc, pr, best_split(cluster, dirs, support)))
            partition_diagnostics = {
                "provided_parts": len(convex_parts),
                "nonempty_parts": len(leaves),
            }
            evaluate("coacd_seeded_split", [p[1] for p in leaves], [p[2] for p in leaves])
            for n in range(len(leaves) + 1, cfg.max_spheres + 1):
                choices = [
                    (float(np.max(leaf[1] @ dirs.T + leaf[2] - support)) + 0.01 * leaf[2], i)
                    for i, leaf in enumerate(leaves)
                    if leaf[3] is not None
                ]
                if not choices:
                    break
                _, index = max(choices)
                _, _, _, split = leaves.pop(index)
                for part, (pc, pr) in zip(split[1], split[2], strict=True):
                    leaves.append((part, pc, pr, best_split(part, dirs, support)))
                if n in checkpoints:
                    evaluate("coacd_seeded_split", [p[1] for p in leaves], [p[2] for p in leaves])

    passing = [i for i, rec in enumerate(records) if rec["meets_targets"]]
    if passing:
        index = min(
            passing,
            key=lambda i: (
                records[i]["num_spheres"],
                records[i]["envelope_excess_relative"],
                records[i]["summed_sphere_volume_relative"],
            ),
        )
        selection = (
            "fewest_spheres_meeting_gap_and_envelope_targets"
            if cfg.tight_envelope
            else "fewest_spheres_meeting_sample_and_envelope_targets"
        )
    else:
        # Count is capped; return the least-violating candidate, visibly flagged.
        index = (
            min(
                range(len(records)),
                key=lambda i: (
                    records[i]["validation_gap_p95_relative"]
                    + 0.1 * records[i]["validation_gap_max_relative"],
                    records[i]["num_spheres"],
                ),
            )
            if cfg.tight_envelope
            else min(
                range(len(records)),
                key=lambda i: (
                    max(0, cfg.min_sample_coverage - records[i]["validation_coverage"]) * 10
                    + max(0, records[i]["envelope_excess_relative"] / cfg.envelope_tolerance - 1),
                    records[i]["num_spheres"],
                ),
            )
        )
        selection = "budget_limited_best_effort"
    centers, radii = candidates[index]
    # Remove spheres contained in another sphere. This preserves the sphere union.
    keep = np.ones(len(radii), dtype=bool)
    for i in np.argsort(radii, kind="stable"):
        for j in np.flatnonzero(keep):
            if i != j and np.linalg.norm(centers[i] - centers[j]) + radii[i] <= radii[j] + 1e-12:
                keep[i] = False
                break
    centers, radii = centers[keep], radii[keep]
    audit = sample_surface(x, f, cfg.audit_samples, cfg.seed + 2)
    gaps = signed_gaps(audit, centers, radii)
    vertex_gaps = np.concatenate(
        [
            signed_gaps(part, centers, radii)
            for part in np.array_split(x, max(1, int(np.ceil(len(x) / 8192))))
        ]
    )
    audit_coverage = float(np.mean(gaps <= 1e-7))
    audit_gap95 = float(np.quantile(np.maximum(gaps, 0), 0.95))
    return {
        "centers": ((centers * scale) @ basis.T * rough_scale + shift).tolist(),
        "radii": (radii * physical_scale).tolist(),
        "num_spheres": len(radii),
        "route": records[index]["route"],
        "selection": selection,
        "config": asdict(cfg),
        "target_met": bool(
            records[index]["meets_targets"]
            and (
                (
                    audit_gap95 <= cfg.sample_gap_tolerance
                    and max(0, gaps.max()) <= 4 * cfg.sample_gap_tolerance
                )
                if cfg.tight_envelope
                else audit_coverage >= cfg.min_sample_coverage
            )
        ),
        "metrics": {
            "audit_surface_coverage": audit_coverage,
            "audit_gap_p95": float(np.quantile(np.maximum(gaps, 0), 0.95) * physical_scale),
            "audit_gap_p95_relative": audit_gap95,
            "audit_gap_max": float(max(0, gaps.max()) * physical_scale),
            "all_vertex_coverage": float(np.mean(vertex_gaps <= 1e-7)),
            "all_vertex_gap_max": float(max(0, vertex_gaps.max()) * physical_scale),
            "directional_envelope_excess": records[index]["envelope_excess_relative"]
            * physical_scale,
            "directional_envelope_excess_relative": records[index]["envelope_excess_relative"],
        },
        "features": {
            "pca_extents": (np.ptp(x, axis=0) * physical_scale).tolist(),
            "pca_eigenvalues": eigenvalues[::-1].tolist(),
        },
        "candidate_decisions": records,
        "selection_candidate_index": index,
        "partition_diagnostics": partition_diagnostics,
        "fit_time_s": time.perf_counter() - started,
        "semantics": (
            "approximate obstacle proxy; concavities may be filled; "
            "no continuous coverage guarantee"
        ),
    }
