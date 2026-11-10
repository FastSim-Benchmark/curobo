# Authored collision pair filters

## `MotionPlanner.set_collision_pair_filters(filters)`

`filters` maps loaded cuboid or mesh names to tuples of global robot collision-sphere
indices. Only those sphere/obstacle pairs are excluded. Other spheres, obstacles,
self-collision checks, and boundary-contact penetration limits are unchanged.

The declaration replaces the previous declaration. Empty input restores ordinary
collision checking. Unknown obstacles, duplicate or out-of-range indices, and
multiple scene environments are rejected. Voxel pair exclusions are unsupported.
Indices refer to the current robot model, including its configured payload slots.

World replacement, scene-cache clearing, and obstacle updates clear the declaration.
Filtering must be reapplied to the current obstacle identities after these operations.
A changed declaration invalidates solver graphs and graph-planner buffers; subsequent
queries recapture with CUDA graph execution still enabled. Identical declarations do
not invalidate captured programs. These operations must run between planning calls.

Authored filters are independent of bounded departure/arrival contact constraints.
Filtered pairs are omitted from boundary-contact discovery. Every unfiltered pair
retains the ordinary contact and collision checks.

## `SceneCollision.set_collision_pair_filters(filters, *, num_spheres)`

The lower-level declaration returns whether map storage changed. Callers owning
captured solver graphs must invalidate them when the result is true. The
`MotionPlanner` entry point performs that lifecycle operation.

## `SceneCollision.filtered_sphere_indices(obstacle)`

Returns the currently excluded sphere indices for a loaded obstacle name, or an
empty tuple when no pair is excluded.

## Explicit stationary contacts

`MotionPlanner.plan_pose(..., stationary_contact_links=())` keeps the existing
planning path by default. A nonempty tuple captures existing contacts on those
robot links with the normal 2 mm initial penetration bound. Captured sphere
centers and radii must remain fixed within the existing numerical tolerance;
all other collision pairs remain checked. The scope is cleared on success and
failure. It supports `allow_boundary_collision="none"` and disjoint `"start"`
departures; captured stationary contacts with `"end"` or `"both"` are rejected.

These extensions preserve the existing weighted axis-hold tolerance rule.
Bounded sphere fitting accepts an explicit budget up to 24,576 spheres; budgets
through 16,384 retain the existing fitting search and coverage validation.
