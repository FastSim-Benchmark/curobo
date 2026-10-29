# SPDX-License-Identifier: Apache-2.0

"""Budgeted full-volume covers with a source-relative outward distance bound.

Closed components must have an interior polyhedral kernel. Their convex hulls
are covered by convex cell subdivision, while verified kernel witnesses bound
any hull excess. Unsupported geometry and exhausted budgets raise explicitly.
This is offline preprocessing; no random samples establish either guarantee.
"""

import heapq

import numpy as np
from scipy.optimize import LinearConstraint, NonlinearConstraint, linprog, minimize
from scipy.sparse import csr_matrix
from scipy.spatial import ConvexHull, HalfspaceIntersection, cKDTree
from trimesh import Trimesh

_MARGIN = 1e-7
_MAX_CELLS = 500000
_MAX_DEPTH = 32
_MAX_UNION_SEARCH_NODES = 1000000


def _canonical_mesh(mesh):
    """Keep exact coordinates/winding while removing serialization order."""
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    order = np.lexsort((vertices[:, 2], vertices[:, 1], vertices[:, 0]))
    inverse = np.empty(len(order), dtype=np.int64)
    inverse[order] = np.arange(len(order))
    faces = inverse[np.asarray(mesh.faces)]
    starts = faces.argmin(axis=1)
    faces = np.take_along_axis(faces, (np.arange(3)[None]+starts[:, None]) % 3, axis=1)
    faces = faces[np.lexsort((faces[:, 2], faces[:, 1], faces[:, 0]))]
    return Trimesh(vertices[order], faces, process=False)


def _components(mesh):
    if not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
        raise ValueError("BOUNDED requires finite nonempty geometry")
    parts = list(mesh.split(only_watertight=False, repair=False))
    for part in parts:
        if not part.is_watertight or not part.is_winding_consistent or part.volume <= 0:
            raise ValueError("BOUNDED requires closed outward-oriented components")
    canonical = [_canonical_mesh(part) for part in parts]
    return sorted(canonical, key=lambda part: (len(part.vertices),
        np.asarray(part.vertices).tobytes(), np.asarray(part.faces).tobytes()))


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
    hull = _canonical_mesh(part.convex_hull)
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


def _split(cell: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bisect a convex cell while preserving its entire volume."""
    if len(cell) == 8:
        axis = max(range(3), key=lambda i: np.linalg.norm(cell[1 << i] - cell[0]))
        low = np.array([i for i in range(8) if not i & (1 << axis)])
        high = low | (1 << axis)
        midpoint = (cell[low] + cell[high]) / 2
        left, right = cell.copy(), cell.copy()
        left[high], right[low] = midpoint, midpoint
        return left, right
    i, j = max(
        ((i, j) for i in range(4) for j in range(i + 1, 4)),
        key=lambda pair: np.linalg.norm(cell[pair[0]] - cell[pair[1]]),
    )
    midpoint = (cell[i] + cell[j]) / 2
    left, right = cell.copy(), cell.copy()
    left[i], right[j] = midpoint, midpoint
    return left, right


def _volume_cells(hull: Trimesh) -> np.ndarray:
    """Use an enclosing orthogonal box only for verified rectangular hulls."""
    vertices = np.asarray(hull.vertices)
    if len(vertices) == 8:
        normals = ConvexHull(vertices).equations[:, :3]
        x = normals[0]
        candidates = normals[np.abs(normals @ x) < 1e-10]
        if len(candidates):
            y = candidates[0] - (candidates[0] @ x) * x
            y /= np.linalg.norm(y)
            axes = np.array([x, y, np.cross(x, y)])
            origin = vertices.mean(axis=0)
            local = (vertices - origin) @ axes.T
            lower, upper = local.min(axis=0), local.max(axis=0)
            tolerance = np.finfo(float).eps * max(1., np.abs(vertices).max()) * 128
            if np.all(np.minimum(abs(local - lower), abs(local - upper)) <= tolerance):
                corners = np.array([
                    [upper[j] if i & (1 << j) else lower[j] for j in range(3)]
                    for i in range(8)
                ])
                # Rounding reserve makes this an enclosure, never a shrunken OBB.
                corners += np.where(corners == upper, tolerance, -tolerance)
                return (corners @ axes + origin)[None]
    return _tetrahedra(hull)


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


def _cell_ball(
    cell: np.ndarray, equations: np.ndarray, centroid: np.ndarray, outward: float,
    box: np.ndarray | None = None,
) -> tuple[np.ndarray, float] | None:
    """Return a maximal admissible ball if it certifies the entire cell."""
    center = (cell.mean(axis=0) * 0.999999 + centroid * 0.000001)
    center = center.astype(np.float32).astype(float)
    inside = float(-(equations[:, :3] @ center + equations[:, 3]).max())
    radius = float(np.nextafter(np.float32(inside + outward * 0.99), np.float32(-np.inf)))
    if (inside >= 0 and radius > 0 and radius - inside <= outward
            and np.linalg.norm(cell - center, axis=1).max() <= radius - _MARGIN):
        return center, radius
    center, radius = _witness_ball(cell.mean(axis=0), equations, centroid, outward)
    if np.linalg.norm(cell - center, axis=1).max() <= radius - _MARGIN:
        return center, radius
    if box is not None:
        edges = np.array([box[1 << i] - box[0] for i in range(3)])
        axes = edges / np.linalg.norm(edges, axis=1)[:, None]
        source_half = np.linalg.norm(edges, axis=1) / 2
        origin = box.mean(axis=0)
        points = (cell - origin) @ axes.T
        middle = (points.max(axis=0) + points.min(axis=0)) / 2
        half = (points.max(axis=0) - points.min(axis=0)) / 2
        # At radius r, the best box-constrained center is the cell midpoint
        # projected onto the source box eroded by r - outward. The clearance
        # is concave in r; its derivative locates the global maximum.
        lower, upper = 0., float(source_half.min() + outward * 0.98)
        for _ in range(32):
            trial = (lower + upper) / 2
            shift = np.abs(middle) - source_half + trial - outward * 0.98
            distance = half + np.maximum(shift, 0)
            if distance[shift > 0].sum() <= np.linalg.norm(distance):
                lower = trial
            else:
                upper = trial
        trial = (lower + upper) / 2
        extent = np.maximum(source_half - trial + outward * 0.98, 0)
        local_center = np.clip(middle, -extent, extent)
        center = (local_center @ axes + origin).astype(np.float32).astype(float)
        inside = float(-(equations[:, :3] @ center + equations[:, 3]).max())
        radius = float(np.nextafter(np.float32(inside + outward * 0.99), np.float32(-np.inf)))
        if (inside >= 0 and radius > 0 and radius - inside <= outward
                and np.linalg.norm(cell - center, axis=1).max() <= radius - _MARGIN):
            return center, radius
        return None
    # Solve the convex enclosing-ball feasibility problem before refining.
    # Surface witnesses maximize a local patch, not containment of a volume cell.
    normals, offsets = equations[:, :3], equations[:, 3]

    def constraint(x: np.ndarray) -> np.ndarray:
        return x[3] - np.linalg.norm(cell - x[:3], axis=1)

    def jacobian(x: np.ndarray) -> np.ndarray:
        delta = cell - x[:3]
        distance = np.linalg.norm(delta, axis=1)
        return np.c_[delta / np.maximum(distance[:, None], 1e-15), np.ones(len(cell))]

    solved = minimize(
        lambda x: x[3], np.r_[center, radius], jac=lambda x: np.array([0., 0., 0., 1.]),
        method="SLSQP", constraints=[
            LinearConstraint(np.c_[normals, np.ones(len(normals))],
                             -np.inf, outward * 0.98 - offsets),
            NonlinearConstraint(constraint, 2 * _MARGIN, np.inf, jac=jacobian),
        ], options={"ftol": 1e-12, "maxiter": 30},
    )
    if solved.success:
        center = solved.x[:3].astype(np.float32).astype(float)
        inside = float(-(normals @ center + offsets).max())
        radius = float(np.nextafter(np.float32(inside + outward * 0.99), np.float32(-np.inf)))
        if (inside >= 0 and radius > 0 and radius - inside <= outward
                and np.linalg.norm(cell - center, axis=1).max() <= radius - _MARGIN):
            return center, radius
    return None


def _cover(hull: Trimesh, outward: float, proposal_budget: int) -> tuple[np.ndarray, np.ndarray]:
    equations = ConvexHull(hull.vertices).equations
    centers, radii = np.empty((0, 3)), np.empty(0)
    centroid = hull.vertices.mean(axis=0)
    cells = _volume_cells(hull)
    box = cells[0] if cells.shape[1] == 8 else None
    stack = [(cell, 0) for cell in cells]
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
        if box is not None:
            nearby = np.flatnonzero(np.linalg.norm(centers - cell[0], axis=1) <= radii - _MARGIN)
        else:
            nearby = np.arange(len(radii))
        distances = np.linalg.norm(cell[:, None] - centers[None, nearby], axis=-1)
        if len(nearby) and np.any(np.all(distances <= radii[None, nearby] - _MARGIN, axis=0)):
            continue
        ball = _cell_ball(cell, equations, centroid, outward, box)
        if ball is not None:
            if len(radii) >= proposal_budget:
                raise ValueError("BOUNDED sphere proposal budget exhausted")
            center, radius = ball
            centers = np.concatenate([centers, center[None]])
            radii = np.append(radii, radius)
            continue
        gaps = (np.min(distances - radii[None, nearby], axis=1)
                if len(nearby) else np.full(len(cell), np.inf))
        if box is None and np.any(gaps > _MARGIN):
            for point in cell[gaps > _MARGIN]:
                # Earlier vertices of this cell may have added a ball that
                # already contains this witness. Recheck the current proposal
                # set before spending capacity; volume cells are still checked
                # and subdivided by the ordinary certification path below.
                if len(radii) and np.any(
                    np.linalg.norm(centers - point, axis=1) <= radii - _MARGIN
                ):
                    continue
                add(point)
            stack.append((cell, depth))
        elif depth >= _MAX_DEPTH:
            raise ValueError("BOUNDED subdivision depth cannot satisfy the outward bound")
        else:
            stack.extend((child, depth + 1) for child in _split(cell))
    return centers, radii


def _cover_masks(
    hull: Trimesh, centers: np.ndarray, radii: np.ndarray, *, refinement_levels: int = 0,
) -> tuple[list[int], int]:
    """Record every sphere containing each terminal convex volume cell."""
    stack = [(cell, 0, None) for cell in _volume_cells(hull)]
    masks = [0] * len(radii)
    tree = cKDTree(centers)
    search_radius = float(radii.max() + _MARGIN)
    visited = accepted = 0
    while stack:
        cell, depth, remaining = stack.pop()
        visited += 1
        if visited > _MAX_CELLS or depth > _MAX_DEPTH:
            raise ValueError("BOUNDED full-volume verification budget exhausted")
        # A sphere containing the whole cell must contain its first vertex.
        candidates = np.asarray(tree.query_ball_point(cell[0], search_radius), dtype=int)
        distances = np.linalg.norm(cell[:, None] - centers[None, candidates], axis=-1)
        containing = candidates[np.all(distances <= radii[None, candidates] - _MARGIN, axis=0)]
        if len(containing):
            # A coarse cell contained by one proposal can hide a different
            # combination covering its children. Refine the exact partition
            # before capacity selection; no samples replace volume coverage.
            remaining = refinement_levels if remaining is None else remaining
            if remaining:
                stack.extend((child, depth + 1, remaining - 1) for child in _split(cell))
                continue
            bit = 1 << accepted
            for index in containing:
                masks[index] |= bit
            accepted += 1
        else:
            for point in cell:
                nearby = np.asarray(tree.query_ball_point(point, search_radius), dtype=int)
                if not np.any(np.linalg.norm(centers[nearby] - point, axis=1)
                              <= radii[nearby] + _MARGIN):
                    raise ValueError("BOUNDED result has uncovered volume")
            stack.extend((child, depth + 1, remaining) for child in _split(cell))
    return masks, accepted


def _compress(
    hull: Trimesh, centers: np.ndarray, radii: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    masks, count = _cover_masks(hull, centers, radii)
    uncovered = (1 << count) - 1
    selected = []
    candidates = [(-mask.bit_count(), index) for index, mask in enumerate(masks)]
    heapq.heapify(candidates)
    while uncovered:
        if not candidates:
            raise ValueError("BOUNDED compression lost a certified cell")
        _, index = heapq.heappop(candidates)
        score = (masks[index] & uncovered).bit_count()
        if not score:
            continue
        # Stale scores are upper bounds because uncovered only shrinks.
        # The index tie-break preserves the original deterministic greedy order.
        if candidates and (-score, index) > candidates[0]:
            heapq.heappush(candidates, (-score, index))
            continue
        selected.append(index)
        uncovered &= ~masks[index]
    return centers[selected], radii[selected]


def _select_union_cover(masks, count, capacity):
    """Guide bounded certified-cell search with a continuous cover relaxation."""
    full, available = (1 << count) - 1, 0
    for mask in masks:
        available |= mask
    if available & full != full:
        raise ValueError("BOUNDED union proposals lost a certified volume cell")
    uncovered, selected = full, []
    while uncovered and len(selected) < capacity:
        index = max(range(len(masks)), key=lambda i: (masks[i] & uncovered).bit_count())
        selected.append(index)
        uncovered &= ~masks[index]
    if not uncovered:
        return selected
    incidence = [[] for _ in range(count)]
    for index, mask in enumerate(masks):
        while mask:
            bit = mask & -mask
            incidence[bit.bit_length() - 1].append(index)
            mask ^= bit
    # This relaxation only orders actual sphere choices. Fractional values do
    # not certify coverage: every returned selection is checked with exact cell
    # masks, and the independent geometry verifier still runs afterwards.
    constraints = sorted(set(map(tuple, incidence)))
    lengths = np.asarray([len(row) for row in constraints])
    matrix = csr_matrix((np.ones(int(lengths.sum())),
                         (np.repeat(np.arange(len(constraints)), lengths),
                          np.asarray([i for row in constraints for i in row]))),
                        shape=(len(constraints), len(masks)))
    relaxation = linprog(np.ones(len(masks)), A_ub=-matrix,
                         b_ub=-np.ones(len(constraints)), bounds=(0., 1.),
                         method="highs", options={"maxiter": 10000, "time_limit": 5.})
    if not relaxation.success or not np.isfinite(relaxation.x).all():
        raise ValueError("BOUNDED union cover relaxation failed or exhausted its budget")
    weights = relaxation.x
    uncovered, selected = full, []
    while uncovered and len(selected) < capacity:
        index = max(range(len(masks)), key=lambda i:
                    ((masks[i] & uncovered).bit_count() * (weights[i] + 1e-6), -i))
        selected.append(index)
        uncovered &= ~masks[index]
    if not uncovered:
        return selected
    witnesses = sorted(range(count), key=lambda i: (len(incidence[i]), i))
    failed, visited = set(), 0

    def search(remaining, slots):
        nonlocal visited
        if not remaining:
            return []
        if not slots or (remaining, slots) in failed:
            return None
        visited += 1
        if visited > _MAX_UNION_SEARCH_NODES:
            raise ValueError("BOUNDED union cover search budget exhausted")
        gains = [(mask & remaining).bit_count() for mask in masks]
        maximum = max(gains)
        # A sphere cannot cover more remaining cells than its current gain.
        if not maximum or maximum * slots < remaining.bit_count():
            failed.add((remaining, slots))
            return None
        witness = next(i for i in witnesses if remaining & (1 << i))
        candidates = sorted(incidence[witness], key=lambda i: (-weights[i], -gains[i], i))
        for index in candidates:
            tail = search(remaining & ~masks[index], slots - 1)
            if tail is not None:
                return [index, *tail]
        failed.add((remaining, slots))
        return None

    selected = search((1 << count) - 1, capacity)
    if selected is not None:
        return selected
    raise ValueError("BOUNDED sphere capacity cannot retain full-volume union coverage")


def _union_cover(hulls, outward_bounds, capacity):
    """Share balls across overlapping source components within a small budget.

    Each ball keeps one component as its outward-bound witness. Coverage uses
    every ball across every source volume cell; it never fills the convex hull
    of the union. Proposal capacity is independent of the returned ball budget.
    """
    centers, radii, owners = [], [], []
    for index, (hull, outward) in enumerate(zip(hulls, outward_bounds, strict=True)):
        c, r = _cover(hull, outward, max(256, 4 * capacity))
        # A locally redundant ball can cover another component better than a
        # locally selected one. Spend the budget only after global selection.
        centers.extend(c)
        radii.extend(r)
        owners.extend([index] * len(r))
    centers, radii, owners = np.asarray(centers), np.asarray(radii), np.asarray(owners)
    masks = [0] * len(radii)
    count = 0
    for hull in hulls:
        local, size = _cover_masks(hull, centers, radii, refinement_levels=2)
        masks = [whole | (part << count) for whole, part in zip(masks, local, strict=True)]
        count += size
    selected = _select_union_cover(masks, count, capacity)
    return centers[selected], radii[selected], owners[selected].tolist()


def bounded_fit_mesh(mesh, num_spheres, max_protrusion_m):
    """Fit within explicit capacity; never truncate spheres to meet the budget."""
    if type(num_spheres) not in (int, np.int64, np.int32) or not 1 <= num_spheres <= 16384:
        raise ValueError("BOUNDED num_spheres must be an integer budget in [1, 16384]")
    if (isinstance(max_protrusion_m, bool) or max_protrusion_m is None
            or not np.isfinite(max_protrusion_m) or max_protrusion_m <= 1e-5):
        raise ValueError("BOUNDED max_protrusion_m must be finite and greater than 1e-5")
    parts = _components(mesh)
    if num_spheres <= 50 and len(parts) > 1:
        hulls, diagnostics, outward_bounds = [], [], []
        for part in parts:
            hull, excess = _hull_excess(part)
            outward = float(max_protrusion_m) - excess - 1e-6
            if outward <= 1e-5:
                raise ValueError("BOUNDED component hull excess consumes the outward budget")
            hulls.append(hull)
            outward_bounds.append(outward)
            diagnostics.append({"hull_excess_m": excess})
        c, r, owners = _union_cover(hulls, outward_bounds, int(num_spheres))
        for index, diagnostic in enumerate(diagnostics):
            diagnostic["sphere_count"] = owners.count(index)
        return c, r, {
            "max_protrusion_m": float(max_protrusion_m), "sphere_components": owners,
            "containment_margin_m": _MARGIN, "components": diagnostics,
            "coverage_scope": "source_union", "full_volume_covered": False,
        }
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
    scope = diagnostics.get("coverage_scope", "component")
    if scope not in ("component", "source_union"):
        raise ValueError("BOUNDED coverage scope is invalid")
    if (not np.issubdtype(owners.dtype, np.integer)
            or np.any(owners < 0) or np.any(owners >= len(parts))):
        raise ValueError("BOUNDED sphere component ownership is invalid")
    maximum = 0.0
    for index, part in enumerate(parts):
        hull, excess = _hull_excess(part)
        cc, rr = c[owners == index], r[owners == index]
        if not len(rr) and scope == "component":
            raise ValueError("BOUNDED result omits a source component")
        if len(rr):
            eq = ConvexHull(hull.vertices).equations
            inside = -(cc @ eq[:, :3].T + eq[:, 3]).max(axis=1)
            outward = rr - inside + excess
            if np.any(inside < 0) or np.any(outward > diagnostics["max_protrusion_m"]):
                raise ValueError("BOUNDED result exceeds the source-relative outward bound")
            maximum = max(maximum, float(outward.max()))
        _cover_masks(hull, c if scope == "source_union" else cc,
                     r if scope == "source_union" else rr)
    diagnostics["full_volume_covered"] = True
    diagnostics["verified_max_protrusion_m"] = maximum
