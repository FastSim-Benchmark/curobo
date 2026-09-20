# SPDX-License-Identifier: Apache-2.0

"""Fixed-budget sphere refinement against a finite orthographic visual hull.

Silhouettes use triangle rasterization, not semantic labels. A blind cavity
may be filled while a gap visible in at least one view remains constrained.
This finite-resolution engineering approximation is not a safety certificate.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

from curobo._src.geom.sphere_fit._fast_core import directions, sample_surface, signed_gaps


@dataclass
class SilhouetteHull:
    axes: np.ndarray
    origins: np.ndarray
    fields: list
    gradients: list
    pitch: float

    @classmethod
    def from_mesh(cls, vertices, faces, resolution=128, view_count=20):
        v, f = np.asarray(vertices), np.asarray(faces)
        if resolution < 24 or view_count < 6:
            raise ValueError("resolution >= 24 and view_count >= 6 required")
        if v.ndim != 2 or v.shape[1] != 3 or not np.isfinite(v).all():
            raise ValueError("finite (N, 3) vertices required")
        if f.ndim != 2 or f.shape[1] != 3 or not np.issubdtype(f.dtype, np.integer):
            raise ValueError("integer triangular faces required")
        if not len(f) or f.min() < 0 or f.max() >= len(v) or np.ptp(v, axis=0).max() <= 0:
            raise ValueError("nonempty valid geometry required")
        pitch = np.ptp(v, axis=0).max() / resolution
        normals = directions(view_count)
        # Opposite orthographic views have identical silhouettes.
        unique = []
        for normal in normals:
            if not any(abs(normal @ other) > 1 - 1e-10 for other in unique):
                unique.append(normal)
        axes, origins, fields, gradients = [], [], [], []
        for normal in unique:
            helper = np.eye(3)[np.argmin(abs(normal))]
            u = np.cross(normal, helper)
            u /= np.linalg.norm(u)
            axis = np.stack([u, np.cross(normal, u)], axis=1)
            projected = v @ axis
            origin = projected.min(axis=0) - 8 * pitch
            pixels = np.rint((projected - origin) / pitch).astype(np.int32)
            shape = pixels.max(axis=0) + 9
            mask = np.zeros(tuple(shape[::-1]), dtype=np.uint8)
            flat = pixels[f].reshape(-1, 6)
            if pixels.max() < 256:
                # Collision-free six-byte keys avoid NumPy's slow row sort.
                keys = flat.astype(np.uint64) @ (
                    np.uint64(1) << (8 * np.arange(6, dtype=np.uint64))
                )
                _, indices = np.unique(keys, return_index=True)
                triangles = flat[indices].reshape(-1, 3, 2)
            else:
                triangles = np.unique(flat, axis=0).reshape(-1, 3, 2)
            # Separate convex fills union overlapping triangles (fillPoly's
            # multi-contour even/odd rule would incorrectly introduce holes).
            for tri in triangles:
                cv2.fillConvexPoly(mask, tri, 1)
            field = (distance_transform_edt(mask == 0) - distance_transform_edt(mask != 0)) * pitch
            gy, gx = np.gradient(field, pitch)
            axes.append(axis)
            origins.append(origin)
            fields.append(field)
            gradients.append((gx, gy))
        return cls(np.array(axes), np.array(origins), fields, gradients, pitch)

    def distance(self, points, gradient=False):
        """Max projected signed distance, a surrogate, not 3-D Euclidean SDF."""
        points = np.asarray(points)
        value = np.full(len(points), -np.inf)
        winners = np.zeros(len(points), dtype=int)
        best_uv = np.zeros((len(points), 2))
        for index, (axis, origin, field) in enumerate(
            zip(self.axes, self.origins, self.fields, strict=True)
        ):
            uv = (points @ axis - origin) / self.pitch
            clipped = np.clip(uv, 0, np.array(field.shape[::-1]) - 1)
            outside = uv - clipped
            coords = clipped[:, ::-1].T
            dist = map_coordinates(field, coords, order=1, mode="nearest")
            dist += np.linalg.norm(outside, axis=1) * self.pitch
            take = dist > value
            value[take] = dist[take]
            if gradient:
                winners[take] = index
                best_uv[take] = uv[take]
        if gradient:
            grad = np.zeros_like(points)
            # Only the maximizing view contributes a gradient. Interpolating
            # gradients in every other view was redundant, not extra accuracy.
            for index, (axis, field, (gx, gy)) in enumerate(
                zip(self.axes, self.fields, self.gradients, strict=True)
            ):
                take = winners == index
                if not take.any():
                    continue
                uv = best_uv[take]
                clipped = np.clip(uv, 0, np.array(field.shape[::-1]) - 1)
                outside = uv - clipped
                coords = clipped[:, ::-1].T
                g2 = np.c_[
                    map_coordinates(gx, coords, order=1, mode="nearest"),
                    map_coordinates(gy, coords, order=1, mode="nearest"),
                ]
                far = np.linalg.norm(outside, axis=1) > 0
                g2[far] = outside[far] / np.linalg.norm(outside[far], axis=1)[:, None]
                grad[take] = g2 @ axis.T
        return (value, grad) if gradient else value


def measure(hull, points, centers, radii, sphere_directions):
    surface = (centers[:, None] + radii[:, None, None] * sphere_directions).reshape(-1, 3)
    outside = np.maximum(hull.distance(surface), 0)
    gaps = np.maximum(signed_gaps(points, centers, radii), 0)
    return {
        "local_excess_p95_relative": float(np.quantile(outside, 0.95)),
        "local_excess_rms_relative": float(np.sqrt(np.mean(outside**2))),
        "local_excess_max_relative": float(outside.max()),
        "gap_p95_relative": float(np.quantile(gaps, 0.95)),
        "gap_max_relative": float(gaps.max()),
        "coverage": float(np.mean(gaps <= 1e-7)),
    }


def interior_seed(hull, vertices, points, weights, count):
    """Sample local thickness in the visual hull, then greedily distribute balls.

    This is an alternative initialization, not an exact medial-axis algorithm.
    Large interior clearances propose thick-body balls; narrow branches propose
    smaller balls. All proposals compete using the same geometric objective.
    """
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    pitch = np.ptp(vertices, axis=0).max() / 48
    axes = [
        np.linspace(a, b, max(3, int(np.ceil((b - a) / pitch)) + 1))
        for a, b in zip(low, high, strict=True)
    ]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    clearance = -hull.distance(grid)
    inside = np.flatnonzero(clearance > 0)
    if len(inside) < count:
        return None
    rng = np.random.default_rng(271)
    selected = np.unique(
        np.r_[
            rng.choice(inside, min(2048, len(inside)), replace=False),
            inside[np.argsort(clearance[inside])[-256:]],
        ]
    )
    pool = grid[selected]
    radii = clearance[selected] + hull.pitch
    gaps = np.maximum(cdist(pool, points) - radii[:, None], 0)
    scores = np.exp(-((gaps / 0.035) ** 2))
    scores *= weights
    choices = []
    for _ in range(count):
        gain = scores.sum(axis=1)
        gain[choices] = -np.inf
        index = int(np.argmax(gain))
        choices.append(index)
        # Maintain weighted marginal gains in place instead of allocating and
        # weighting the full candidate-by-point matrix on every greedy step.
        scores -= scores[index].copy()
        np.maximum(scores, 0, out=scores)
    return np.c_[pool[choices], radii[choices]].ravel()


def refine(vertices, faces, centers, radii, resolution=128, iterations=100):
    v, f = np.asarray(vertices), np.asarray(faces)
    centers, radii = np.asarray(centers, dtype=float), np.asarray(radii, dtype=float)
    if centers.shape != (len(radii), 3) or not len(radii) or not np.isfinite(centers).all():
        raise ValueError("nonempty finite centers/radii required")
    if not np.isfinite(radii).all() or (radii <= 0).any() or iterations < 1:
        raise ValueError("positive radii and iterations required")
    shift = (v.min(axis=0) + v.max(axis=0)) / 2
    train_world = sample_surface(v, f, 4096, 17)
    _, basis = np.linalg.eigh(np.cov(train_world.T))
    basis = basis[:, ::-1]
    for axis in range(3):
        if basis[np.argmax(abs(basis[:, axis])), axis] < 0:
            basis[:, axis] *= -1
    scale = np.ptp((v - shift) @ basis, axis=0).max()
    x = (v - shift) @ basis / scale
    c0, r0 = (centers - shift) @ basis / scale, radii / scale
    hull = SilhouetteHull.from_mesh(x, f, resolution)
    train = (train_world - shift) @ basis / scale
    envelope_dirs = directions()
    support = np.full(len(envelope_dirs), -np.inf)
    extrema = []
    for chunk in np.array_split(x, max(1, int(np.ceil(len(x) / 4096)))):
        dots = chunk @ envelope_dirs.T
        support = np.maximum(support, dots.max(axis=0))
        extrema.extend(chunk[dots.argmax(axis=0)])
    critical = np.unique(np.array(extrema), axis=0)
    if len(critical) > 512:
        critical = critical[np.random.default_rng(19).choice(len(critical), 512, replace=False)]
    points = np.r_[train, critical]
    weights = np.r_[
        np.full(len(train), 0.75 / len(train)), np.full(len(critical), 0.25 / len(critical))
    ]
    dirs = directions(48)
    n = len(r0)

    def objective(params, exterior_weight):
        spheres = params.reshape(n, 4)
        c, r = spheres[:, :3], spheres[:, 3]
        dist = cdist(points, c)
        labels = (dist - r).argmin(axis=1)
        near = np.maximum(dist[np.arange(len(points)), labels], 1e-12)
        gap = np.maximum(near - r[labels], 0)
        factor = 2 * gap * weights
        grad = np.zeros((n, 4))
        np.add.at(grad[:, :3], labels, factor[:, None] * (c[labels] - points) / near[:, None])
        np.add.at(grad[:, 3], labels, -factor)
        loss = np.sum(weights * gap**2)
        surface = (c[:, None] + r[:, None, None] * dirs).reshape(-1, 3)
        outside, derivative = hull.distance(surface, gradient=True)
        excess = np.maximum(outside - hull.pitch, 0)
        factor = 2 * exterior_weight * excess / len(surface)
        g = (factor[:, None] * derivative).reshape(n, len(dirs), 3)
        loss += exterior_weight * np.mean(excess**2)
        grad[:, :3] += g.sum(axis=1)
        grad[:, 3] += np.sum(g * dirs, axis=(1, 2))
        over = np.maximum(c @ envelope_dirs.T + r[:, None] - support - 0.025, 0)
        loss += 10 * np.mean(over**2)
        grad[:, :3] += 20 / over.size * over @ envelope_dirs
        grad[:, 3] += 20 / over.size * over.sum(axis=1)
        return float(loss), grad.ravel()

    initial = np.c_[c0, r0].ravel()
    candidates = [(c0, r0, "unchanged")]
    optimizers = []
    seed = interior_seed(hull, x, points, weights, n)
    starts = [("joint_visual_hull", initial, weight) for weight in (0.15, 0.5, 2.0)]
    if seed is not None:
        starts.extend(("thickness_seed", seed, weight) for weight in (0.15, 0.5))
    # Joint optimization permits assignment changes. Compare several tradeoffs,
    # including conservative partial steps, with the identical ball count.
    for source, start, exterior_weight in starts:
        solved = minimize(
            objective,
            start,
            args=(exterior_weight,),
            jac=True,
            method="L-BFGS-B",
            bounds=[
                bound
                for _ in range(n)
                for bound in [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (1e-5, 1.0)]
            ],
            options={"maxiter": iterations, "ftol": 1e-11, "maxls": 30},
        )
        optimizers.append(
            {
                "success": bool(solved.success),
                "message": str(solved.message),
                "iterations": int(solved.nit),
                "exterior_weight": exterior_weight,
                "initialization": source,
            }
        )
        if not np.isfinite(solved.x).all():
            raise ValueError("non-finite visual hull refinement")
        for fraction in (0.25, 0.5, 0.75, 1.0) if source == "joint_visual_hull" else (1.0,):
            result = (initial + fraction * (solved.x - initial)).reshape(n, 4)
            c, r = result[:, :3], result[:, 3]
            caps = np.min(support + 0.025 - c @ envelope_dirs.T, axis=1)
            if (caps > 1e-5).all():
                r = np.minimum(r, caps)
                candidates.append((c, r, f"{source}_w{exterior_weight}_step{fraction}"))
    validation = sample_surface(x, f, 4096, 18)
    validation_dirs = directions(192)
    before = measure(hull, validation, c0, r0, validation_dirs)
    records = []
    accepted = 0
    for index, (c, r, label) in enumerate(candidates):
        metrics = measure(hull, validation, c, r, validation_dirs)
        # Do not trade arbitrary missed geometry for attractive smaller spheres.
        eligible = (
            metrics["gap_p95_relative"] <= before["gap_p95_relative"] + 0.008
            and metrics["gap_max_relative"] <= before["gap_max_relative"] + 0.025
            and metrics["coverage"] >= before["coverage"] - 0.08
            and before["local_excess_rms_relative"] - metrics["local_excess_rms_relative"]
            >= hull.pitch / 4
            and metrics["local_excess_rms_relative"] < before["local_excess_rms_relative"] * 0.98
        )
        records.append({"candidate": label, "eligible": bool(eligible), **metrics})
        if (
            eligible
            and metrics["local_excess_rms_relative"]
            < records[accepted]["local_excess_rms_relative"]
        ):
            accepted = index
    c, r, label = candidates[accepted]
    audit = sample_surface(x, f, 8192, 19)
    audit_dirs = directions(384)
    return {
        "centers": ((c * scale) @ basis.T + shift).tolist(),
        "radii": (r * scale).tolist(),
        "num_spheres": n,
        "route": label,
        "before": measure(hull, audit, c0, r0, audit_dirs),
        "after": measure(hull, audit, c, r, audit_dirs),
        "scale": float(scale),
        "pitch_relative": hull.pitch,
        "views": len(hull.axes),
        "validation_candidates": records,
        "optimizers": optimizers,
        "envelope_relative": float(
            np.maximum(c @ envelope_dirs.T + r[:, None] - support, 0).max()
        ),
        "all_vertex_coverage": float(np.mean(signed_gaps(x, c, r) <= 1e-7)),
        "all_vertex_gap_max": float(max(0, signed_gaps(x, c, r).max()) * scale),
    }
