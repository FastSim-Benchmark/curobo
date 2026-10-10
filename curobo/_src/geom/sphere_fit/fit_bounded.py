# SPDX-License-Identifier: Apache-2.0

"""Budgeted full-volume covers with a source-relative outward distance bound.

Closed components must have an interior polyhedral kernel. Their convex hulls
are covered by tetrahedral subdivision, while verified kernel witnesses bound
any hull excess. Unsupported geometry and exhausted budgets raise explicitly.
This is offline preprocessing; no random samples establish either guarantee.
"""

import numpy as np
from scipy.optimize import LinearConstraint, linprog, minimize
from scipy.spatial import ConvexHull, HalfspaceIntersection

_MARGIN = 1e-7
_MAX_CELLS = 500000
_MAX_DEPTH = 32


def _components(mesh):
    if not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
        raise ValueError("BOUNDED requires finite nonempty geometry")
    parts = list(mesh.split(only_watertight=False, repair=False))
    for part in parts:
        if not part.is_watertight or not part.is_winding_consistent or part.volume <= 0:
            raise ValueError("BOUNDED requires closed outward-oriented components")
    return parts


def _radial_hull_excess(part, origin):
    """Bound each hull/face-cone partition by feasible source tetrahedron witnesses."""
    origin = np.asarray(origin, dtype=float)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("Radial partition requires a finite origin")
    if not part.is_watertight or not part.is_winding_consistent or part.volume <= 0:
        raise ValueError("Radial partition requires a closed outward source")
    clearance = np.einsum("ij,ij->i", part.face_normals, part.triangles[:, 0] - origin)
    if not np.isfinite(clearance).all() or clearance.min() <= 1e-9:
        raise ValueError("Radial partition origin is outside the strict kernel")
    vertices = np.asarray(part.vertices) - origin
    triangles = np.asarray(part.triangles) - origin
    hull_equations = ConvexHull(vertices).equations
    tolerance = 1e-9
    maximum = 0.0
    for triangle in triangles:
        matrix = triangle.T
        if not np.isfinite(matrix).all() or np.linalg.cond(matrix) > 1e8:
            raise ValueError('Radial partition source tetrahedron is ill-conditioned')
        inverse = np.linalg.inv(matrix)
        # Cone membership is exactly nonnegative barycentric ray weights.
        normals = -inverse / np.linalg.norm(inverse, axis=1)[:, None]
        equations = np.vstack([hull_equations, np.c_[normals, np.zeros(3)]])
        solved = linprog(
            [0, 0, 0, -1], A_ub=np.c_[equations[:, :3], np.ones(len(equations))],
            b_ub=-equations[:, 3], bounds=[(None, None)] * 4, method='highs',
        )
        if not solved.success or solved.x[3] <= tolerance:
            raise ValueError('Radial partition has no numerically strict interior')
        points = HalfspaceIntersection(equations, solved.x[:3]).intersections
        if (not len(points) or not np.isfinite(points).all()
                or np.max(points @ equations[:, :3].T + equations[:, 3]) > tolerance):
            raise ValueError('Radial partition vertices violate their halfspaces')
        weights = points @ inverse.T
        # Construct feasible source tetrahedron witnesses explicitly rather
        # than accepting slightly infeasible QHull vertices as source points.
        weights = np.maximum(weights, 0)
        weights /= np.maximum(weights.sum(axis=1), 1)[:, None]
        witnesses = weights @ triangle
        bound = float(np.linalg.norm(points - witnesses, axis=1).max())
        maximum = max(maximum, bound)
    # Explicit arithmetic reserve, still checked against the requested bound.
    return maximum + 1e-7


def _hull_excess(part):
    """Upper distance bound from the whole hull to a subset of the source."""
    normals = np.asarray(part.face_normals)
    offsets = -np.einsum("ij,ij->i", normals, part.triangles[:, 0])
    solved = linprog(
        [0, 0, 0, -1], A_ub=np.c_[normals, np.ones(len(normals))],
        b_ub=-offsets, bounds=[(None, None)] * 4, method="highs",
    )
    if not solved.success:
        raise ValueError("BOUNDED component has no verified interior kernel")
    origin = solved.x[:3]
    clearance = -(normals @ origin + offsets)
    if not np.isfinite(clearance).all() or clearance.min() <= _MARGIN:
        raise ValueError("BOUNDED component kernel is empty or numerically too thin")
    hull = part.convex_hull
    vertices = np.asarray(hull.vertices)
    excess = vertices @ normals.T + offsets
    fraction = np.max(
        np.maximum(excess + 1e-9, 0) / np.maximum(excess + clearance[None], 1e-9), axis=1,
    )
    witnesses = vertices + (origin - vertices) * np.clip(fraction, 0, 1)[:, None]
    for i in np.flatnonzero(np.max(excess, axis=1) > 1e-9):
        point = vertices[i]
        candidate = minimize(
            lambda q: 0.5 * np.sum((q - point) ** 2), witnesses[i],
            jac=lambda q: q - point, method="SLSQP",
            constraints=LinearConstraint(normals, -np.inf, -offsets - 1e-9),
            options={"ftol": 1e-15, "maxiter": 200},
        )
        if candidate.success and np.max(normals @ candidate.x + offsets) <= 0:
            witnesses[i] = candidate.x
    if np.max(witnesses @ normals.T + offsets) > 0:
        raise ValueError("BOUNDED kernel witnesses are not feasible")
    # Distance to a convex set is convex. Feasible witnesses at all hull
    # vertices therefore bound the full hull, not just its sampled surface.
    bound = float(np.linalg.norm(witnesses - vertices, axis=1).max())
    # Refining below the arithmetic reserve cannot improve the usable budget.
    if bound > 1e-6:
        bound = min(bound, _radial_hull_excess(part, origin))
    return hull, bound + 1e-9


def _tetrahedra(hull):
    origin = hull.vertices.mean(axis=0)
    cells = np.concatenate(
        [np.repeat(origin[None, None], len(hull.faces), axis=0), hull.triangles], axis=1,
    )
    volume = abs(np.linalg.det(cells[:, 1:] - cells[:, :1])).sum() / 6
    if not np.isclose(volume, abs(hull.volume), rtol=1e-8, atol=1e-14):
        raise ValueError("BOUNDED tetrahedra do not reproduce component volume")
    return cells


def _split(cell):
    i, j = max(
        ((i, j) for i in range(4) for j in range(i + 1, 4)),
        key=lambda pair: np.linalg.norm(cell[pair[0]] - cell[pair[1]]),
    )
    midpoint = (cell[i] + cell[j]) / 2
    left, right = cell.copy(), cell.copy()
    left[i], right[j] = midpoint, midpoint
    return left, right


def _witness_ball(point, equations, centroid, outward):
    normals, offsets = equations[:, :3], equations[:, 3]
    distances = -(normals @ point + offsets)
    if distances.min() < -_MARGIN:
        raise ValueError("BOUNDED witness lies outside its component hull")
    near = np.argsort(distances)[:min(8, len(distances))]
    directions = [*normals]
    for count in (2, 3, 4, 8):
        direction = normals[near[:count]].sum(axis=0)
        if np.linalg.norm(direction) > 1e-9:
            directions.append(direction / np.linalg.norm(direction))
    direction = point - centroid
    if np.linalg.norm(direction) > 1e-9:
        directions.append(direction / np.linalg.norm(direction))
    directions = np.asarray(directions)
    denominator = np.maximum(1 - normals @ directions.T, 0)
    ratios = np.full_like(denominator, np.inf)
    np.divide(
        distances[:, None] + outward * 0.95, denominator,
        out=ratios, where=denominator > 1e-12,
    )
    depths = np.min(ratios, axis=0)
    directions = np.repeat(directions, 6, axis=0)
    depths = (depths[:, None] * np.array([0.25, 0.5, 0.75, 0.95, 0.999, 1])).reshape(-1)
    centers = (point[None] - depths[:, None] * directions).astype(np.float32).astype(float)
    inside = np.min(-(normals @ centers.T + offsets[:, None]), axis=0)
    radii = np.nextafter((inside + outward * 0.99).astype(np.float32), np.float32(np.inf))
    radii = radii.astype(float)
    clearance = radii - np.linalg.norm(centers - point, axis=1)
    valid = (inside >= 0) & (radii > 0) & (clearance > _MARGIN) & (radii - inside <= outward)
    if not valid.any():
        raise ValueError("BOUNDED cannot cover a witness within the outward bound")
    score = np.where(valid, clearance * (2 * radii - clearance), -np.inf)
    best = int(score.argmax())
    return centers[best], radii[best]


def _cover(hull, outward, proposal_budget):
    equations = ConvexHull(hull.vertices).equations
    centers, radii = np.empty((0, 3)), np.empty(0)
    centroid = hull.vertices.mean(axis=0)
    stack = [(cell, 0) for cell in _tetrahedra(hull)]
    visited = 0

    def add(point):
        nonlocal centers, radii
        if len(radii) >= proposal_budget:
            raise ValueError("BOUNDED sphere proposal budget exhausted")
        center, radius = _witness_ball(point, equations, centroid, outward)
        centers = np.concatenate([centers, center[None]])
        radii = np.append(radii, radius)

    while stack:
        cell, depth = stack.pop()
        visited += 1
        if visited > _MAX_CELLS:
            raise ValueError("BOUNDED cell budget exhausted during fitting")
        distances = np.linalg.norm(cell[:, None] - centers[None], axis=-1)
        if len(radii) and np.any(np.all(distances <= radii[None] - _MARGIN, axis=0)):
            continue
        gaps = np.min(distances - radii[None], axis=1) if len(radii) else np.full(4, np.inf)
        if np.any(gaps > _MARGIN):
            for point in cell[gaps > _MARGIN]:
                add(point)
            stack.append((cell, depth))
        elif depth >= _MAX_DEPTH:
            if len(radii) >= proposal_budget:
                raise ValueError("BOUNDED sphere proposal budget exhausted")
            # Finish an unresolved small tetrahedron with one enclosing ball.
            # The same outward bound still applies; depth never bypasses it.
            center = (cell.mean(axis=0) * 0.999999 + centroid * 0.000001)
            center = center.astype(np.float32).astype(float)
            radius = float(np.nextafter(
                np.float32(np.linalg.norm(cell - center, axis=1).max() + 1e-6),
                np.float32(np.inf),
            ))
            inside = float(-(equations[:, :3] @ center + equations[:, 3]).max())
            if inside < 0 or radius - inside > outward:
                raise ValueError("BOUNDED subdivision depth cannot satisfy the outward bound")
            centers = np.concatenate([centers, center[None]])
            radii = np.append(radii, radius)
            stack.append((cell, depth))
        else:
            stack.extend((child, depth + 1) for child in _split(cell))
    return centers, radii


def _cover_masks(hull, centers, radii):
    """Record every sphere containing each terminal tetrahedron."""
    stack = [(cell, 0) for cell in _tetrahedra(hull)]
    masks = [0] * len(radii)
    visited = accepted = 0
    while stack:
        cell, depth = stack.pop()
        visited += 1
        if visited > _MAX_CELLS or depth > _MAX_DEPTH:
            raise ValueError("BOUNDED full-volume verification budget exhausted")
        distances = np.linalg.norm(cell[:, None] - centers[None], axis=-1)
        containing = np.flatnonzero(np.all(distances <= radii[None] - _MARGIN, axis=0))
        if len(containing):
            bit = 1 << accepted
            for index in containing:
                masks[index] |= bit
            accepted += 1
        elif np.any(np.all(distances > radii[None] + _MARGIN, axis=1)):
            raise ValueError("BOUNDED result has uncovered volume")
        else:
            stack.extend((child, depth + 1) for child in _split(cell))
    return masks, accepted


def _compress(hull, centers, radii):
    masks, count = _cover_masks(hull, centers, radii)
    uncovered = (1 << count) - 1
    selected = []
    while uncovered:
        index = max(range(len(radii)), key=lambda i: (masks[i] & uncovered).bit_count())
        if not masks[index] & uncovered:
            raise ValueError("BOUNDED compression lost a certified cell")
        selected.append(index)
        uncovered &= ~masks[index]
    return centers[selected], radii[selected]


def bounded_fit_mesh(mesh, num_spheres, max_protrusion_m):
    """Fit within explicit capacity; never truncate spheres to meet the budget."""
    if type(num_spheres) not in (int, np.int64, np.int32) or not 1 <= num_spheres <= 16384:
        raise ValueError("BOUNDED num_spheres must be an integer budget in [1, 16384]")
    if (isinstance(max_protrusion_m, bool) or max_protrusion_m is None
            or not np.isfinite(max_protrusion_m) or max_protrusion_m <= 1e-5):
        raise ValueError("BOUNDED max_protrusion_m must be finite and greater than 1e-5")
    parts = _components(mesh)
    centers, radii, owners, diagnostics = [], [], [], []
    remaining = int(num_spheres)
    for index, part in enumerate(parts):
        hull, hull_excess = _hull_excess(part)
        outward = float(max_protrusion_m) - hull_excess - 1e-6
        if outward <= 1e-5:
            raise ValueError("BOUNDED component hull excess consumes the outward budget")
        if remaining <= 0:
            raise ValueError("BOUNDED sphere capacity exhausted")
        c, r = _cover(hull, outward, min(32767, 2 * remaining))
        c, r = _compress(hull, c, r)
        if len(r) > remaining:
            raise ValueError("BOUNDED sphere capacity cannot retain full-volume coverage")
        remaining -= len(r)
        centers.extend(c.tolist())
        radii.extend(r.tolist())
        owners.extend([index] * len(r))
        diagnostics.append({"hull_excess_m": hull_excess, "sphere_count": len(r)})
    return np.asarray(centers), np.asarray(radii), {
        "max_protrusion_m": float(max_protrusion_m), "sphere_components": owners,
        "containment_margin_m": _MARGIN,
        "components": diagnostics, "full_volume_covered": False,
    }


def verify_bounded_result(mesh, centers, radii, diagnostics):
    """Verify the returned, converted and clipped spheres, not an earlier fit."""
    c, r = np.asarray(centers, dtype=float), np.asarray(radii, dtype=float)
    owners = np.asarray(diagnostics["sphere_components"])
    if c.shape != (len(r), 3) or owners.shape != r.shape or not len(r):
        raise ValueError("BOUNDED clipping removed spheres required by the volume certificate")
    if not np.isfinite(c).all() or not np.isfinite(r).all() or np.any(r <= 0):
        raise ValueError("BOUNDED result must contain finite positive spheres")
    parts = _components(mesh)
    if np.any(owners < 0) or np.any(owners >= len(parts)):
        raise ValueError("BOUNDED sphere component ownership is invalid")
    maximum = 0.0
    for index, part in enumerate(parts):
        hull, excess = _hull_excess(part)
        cc, rr = c[owners == index], r[owners == index]
        if not len(rr):
            raise ValueError("BOUNDED result omits a source component")
        eq = ConvexHull(hull.vertices).equations
        inside = -(cc @ eq[:, :3].T + eq[:, 3]).max(axis=1)
        outward = rr - inside + excess
        if np.any(inside < 0) or np.any(outward > diagnostics["max_protrusion_m"]):
            raise ValueError("BOUNDED result exceeds the source-relative outward bound")
        maximum = max(maximum, float(outward.max()))
        _cover_masks(hull, cc, rr)
    diagnostics["full_volume_covered"] = True
    diagnostics["verified_max_protrusion_m"] = maximum
