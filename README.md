<!-- SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# cuRobo

## Swept obstacle bounds

Swept mesh and cuboid queries compute conservative world bounds once per query
from the current obstacle poses and local bounds. Sphere groups reject distant
obstacles before repeating local-frame transforms. Surviving pairs retain the
existing local broad phase, sweep samples, collision costs and gradients.
Nonfinite or invalid bounds retain the original checks. Query-local scratch also
reads updated poses and dimensions during CUDA graph replay.

Surviving group-obstacle pairs are collected separately from exact sweep
evaluation, allowing the spheres in each group to run in parallel. Each group
stores at most 16 candidates; if more survive, evaluation checks every populated
obstacle instead of truncating the collision query. Contact replacements and
speed scaling retain their existing semantics. Parallel accumulation can change
floating-point rounding, particularly after speed-gradient projection.

## Sphere-fit diagnostic precision

Sphere-fit coverage and surface-gap metrics use direct Euclidean distances so
global TF32 matrix settings cannot introduce false millimeter-scale gaps or
false coverage. This changes diagnostic measurements only; fitted spheres and
the independent bounded-volume certificate remain unchanged.

## Continuous seeds for axis-constrained motion

Ordinary non-contact pose queries with an axis hold first try one continuous
Cartesian seed. Sixteen IK steps search near the previous configuration, using
joint types to distinguish translation from rotation and intersecting each local
search box with the original physical limits. Limits are restored on success,
failure and timeout. Matching targets across tool frames stay paired, and returned
goal indices refer to the caller's original goal set.

The seed still passes the native endpoint, joint, derivative, self/world collision
and dense interpolation checks. An unsuccessful seed proceeds to the existing
bounded search; the original request deadline remains in force. Queries without
an axis hold, start/both-boundary contact queries and joint-goal planning retain
their existing paths.

Axis-held end-contact queries first try a nearby physical-limit-bounded endpoint
and a continuous seed ending at that exact contact configuration. The continuation
uses at most sixteen steps bounded by local joint search boxes. Intermediate IK
uses ordinary complete collision checks. Only the captured endpoint metrics and
native trajectory validation use the end-contact scope. If intermediate IK returns
no valid candidate, the
seed may propose a joint-space tail to that same endpoint only when it lies in the
current local box and exactly matches the terminal hint. This tail is a seed
proposal; it does not certify a Cartesian path or an accepted trajectory.
The authored goal and all native endpoint, axis, joint, derivative, self/world
collision, contact and dense interpolation checks remain authoritative. Contact
cleanup and physical-limit restoration run on failure too; an unavailable or
rejected seed retains the existing terminal-contact search and shared deadline.
A bounded continuation summary records the proposal and outcome even when the
terminal-contact search supplies the returned result.

## Recovering an initial trajectory seed

When every optimized trajectory is rejected, TrajOpt also checks its original
seed after recomputing timing with the configured velocity, acceleration and
jerk limits. Recovery requires the ordinary endpoint, joint, axis, self-collision,
scene/contact and dense interpolated checks to pass. It does not change geometry,
contact tolerances, or planning deadlines. A recovered result records
`initial_seed_recovery` in its debug information. Successful optimized results
keep their existing selection and refinement behavior.

## Trajectory interpolation at joint limits

Dense trajectory validation projects a position excursion of at most one
representable floating-point step back onto its joint limit, only when that
coordinate's optimized trajectory is finite and entirely within the limits.
This prevents spline rounding at an exact limit from falsely rejecting a valid
trajectory. Larger excursions, violating reference trajectories, derivative
limits, and collision checks retain their existing rejection behavior.

Failed posture and terminal-contact queries log bounded collision diagnostics
for both the optimized and interpolated trajectories while the request's contact
scope is still active. Dense collision peaks use the dense state and its own
metrics rollout. Departure diagnostics distinguish actual support penetration
from a nonmonotonic clearance change; neither measurement changes acceptance.
Each bounded raw mesh sample includes its exact active joint state, the selected
trajectory's endpoint joints, and sphere centers/radii in the kinematics model
frame. Its at-most-eight pairs also retain the environment-zero mesh slot,
query-frame-to-mesh-local inverse pose (XYZ plus WXYZ), local dimensions and
distance mode. These identify the same query sample; they do not export a full
trajectory or establish an independent physical collision certificate.

Rejected trajectory candidates also log their selected optimized and interpolated
constraint maxima immediately, before the enclosing search returns. This retains
the rejection evidence if an end-to-end timeout later discards the search result;
it does not alter candidate ranking, planning budgets, or acceptance thresholds.

Terminal-contact pose search uses `max_attempts` to bound both fresh endpoint
proposal batches and captured candidates per target. If a proposal batch contains
no acceptable endpoint, remaining batches are sampled before moving to another
target. Endpoint contact validation, exact target/seed binding, and trajectory
collision and limit checks remain mandatory. Exhausted searches can take longer
than a single proposal batch; no additional unbounded retry loop is introduced.

## Bounded IK diagnostic memory

Failed IK self-collision diagnostics scan the enabled pair table in bounded
chunks. Inactive radii are filtered before distance vectors are allocated.
The report retains the same full positive-pair count and stable top-eight order,
including ties, while pair indices retain their original width until each chunk
is indexed. Float64 diagnostic distances and native planning acceptance remain
unchanged.

## Self-collision initialization

Forward kinematics uses full-width sphere tensor offsets. The former signed
16-bit float offset wrapped at sphere index 8192, corrupting or rejecting large
robot/payload models. Forward sphere output, its Jacobian variant, and backward
tensor addressing preserve complete sphere positions and radii across batches.

Self-collision initialization enumerates eligible sphere pairs with tensor
operations, preserving pair order and exclusions for detailed robot models.
Models with no eligible pair produce an empty pair table; oversized sphere
indices are rejected before conversion to the native int16 representation.
Collision radii, padding, and runtime acceptance thresholds are unchanged.

Eager self-collision queries compact large pair tables when allocation slots are
disabled throughout the complete batch and trajectory. Eligibility includes
radius padding and is rechecked on every call, so attaching, detaching, or
changing an object's sphere support invalidates the cached selection. Only
disabled pairs are omitted; active sphere geometry and collision rules remain
unchanged. CUDA graph capture/replay and per-pair distance diagnostics use the
complete table. At tied maxima the native reduction may select a different
valid maximum-pair subgradient after compaction.

Scene collision queries use separate sphere and obstacle thread coordinates.
Large trajectory batches whose sphere/obstacle pair count exceeds the signed
32-bit range retain collision costs and gradients for the final query spheres,
including swept queries. Geometry, contact policy, and collision thresholds are
unchanged.

Mesh collision queries reject pairs only when the sphere, or the complete adjacent
swept-segment bounds, lie outside the mesh's local bounds with a rounding guard.
Interior points and segment crossings remain eligible for the existing narrow
phase. Bounds follow mesh replacement, removal and reload in stable GPU buffers,
including CUDA Graph replay. This optimization preserves the existing collision
cost and gradient; it does not turn the bounded adaptive sweep into an exact
continuous collision certificate.

For unsigned surface meshes, the broad phase also checks whether any triangle
bound overlaps the expanded query bounds. This excludes empty space inside a
room-sized mesh bound without removing geometry or reducing collision radii.
Signed solid interiors remain eligible even when no surface is nearby. Mesh
replacement and captured graph replay read the current mesh handle.

Cuboid sweeps also use conservative bounds for groups of adjacent spheres before
running the unchanged analytic collision query. The bounds include the full
adjacent sweep, activation radius, and a floating-point guard; solid interiors,
thin-wall crossings, tangencies, and nonfinite bounds remain eligible. Captured
graph replay reads current box dimensions and poses rather than cached bounds.

Swept collision also checks the midpoint shared by adjacent half sweeps. The
adaptive loops previously excluded that boundary, allowing a thin wall crossing
to return zero cost even though both endpoint queries were correctly clear.
The midpoint check supplements every original adaptive sample; finite clear
surfaces, disabled spheres and stationary queries retain their existing meaning.

## Default sphere fitting in this fork

Fork version `0.8.0.post1.dev58` introduces the FAST default and clipping
precision fixes below. Package versions continue to follow Git via setuptools-scm.

New sphere fits use `SphereFitType.FAST`: a CPU preprocessing pipeline with a
default budget of 32 spheres, finite-view silhouette constraints, and conditional
CoACD decomposition. It allows filled cavities and local coverage gaps; it is
not a conservative enclosure. Existing saved sphere configurations are unchanged.
Select `SphereFitType.MORPHIT`, `VOXEL`, or `SURFACE` explicitly to retain a legacy
method. Shared clipping and output device/dtype conversion still apply.
See [sphere fitting](docs/reference/sphere_fitting.rst) for budgets and diagnostics.

Opt-in `SphereFitType.AUTO` routes small-budget rectangular plates to RIM,
explicit outward-bound requests to BOUNDED, and other geometry to FAST.
For a phone-sized rectangular plate, `num_spheres=40` selects a 40-sphere
outer contour proxy. This does not certify the filled volume. Routing decisions
are recorded in `debug_info["routing"]`; existing defaults remain unchanged.

For compound BOUNDED requests with capacity at most 50, source components share
coverage balls instead of spending capacity independently on overlapping pieces.
When greedy selection exceeds capacity, a sparse continuous cover relaxation
guides actual sphere choices and branching over certified volume cells.
Fractional weights never certify coverage: the selected integer set must cover
every cell within capacity. The relaxation is bounded to 10,000 iterations and
five seconds, and branching to 1,000,000 nodes. Each retained ball still has its own
source-component outward witness, and every component's complete volume is checked
again after conversion and clipping. An exhausted budget raises; it never returns
a truncated fit or substitutes a filled convex hull for the source union.
BOUNDED canonicalizes component, vertex and triangle ordering before subdivision.
This retains exact coordinates and triangle winding so mesh serialization order
cannot change the selected fitting route or spend a different sphere budget.

An explicit `SphereFitType.RIM` selection fits a plate's outer convex
projected contour and its observed rim-height band with 4--50 spheres, without
a height-to-width cutoff. It uses the complete compound to avoid treating
internal convex-part seams as rims.
Returned float32/float64 geometry must retain every assigned contour section
and the original outer-band vertex witnesses after clipping. The result records
convex bridging of concavities and does not certify the plate's filled face or
volume. With an object-local bottom bound, the contour band is intersected with
the source's real local-Z extent and sphere centers are fitted to the bound;
post-fit clipping cannot substitute for the coverage proof. This preserves the
requested sphere capacity. FAST remains the default and retains its existing
fitting pipeline.

*CUDA Accelerated Robot Library*

**[Documentation](https://nvlabs.github.io/curobo) | [Paper](https://arxiv.org/abs/2603.05493)**

> [!NOTE]
> cuRoboV2 is a significant rewrite and the public API has changed from cuRobo v1.
> If you depend on the v1 API, pin to the [`v0.7.8`](https://github.com/NVlabs/curobo/tree/v0.7.8) tag.

cuRobo is a CUDA-accelerated library for robot motion generation, built on
PyTorch, CUDA, and Warp. It provides GPU-parallel algorithms for
forward/inverse kinematics, collision checking, trajectory optimization,
geometric planning, GPU-native perception, and whole-body motion generation,
scaling from single-arm manipulators to high-DoF humanoids.

Key capabilities:
- **Dynamics-aware trajectory optimization** with B-spline representation enforcing smoothness and torque limits
- **GPU-native ESDF perception** that generates dense signed distance fields from depth images, up to 10x faster than state-of-the-art
- **Scalable whole-body computation** including topology-aware kinematics, differentiable inverse dynamics, and map-reduce self-collision for high-DoF robots
- **Collision-free motion generation** combining IK, geometric planning, and trajectory optimization.

## FastSim fork: incremental collision worlds

`SceneCollision.apply_obstacle_updates(obstacles, removed, env_idx=0)` updates
only named mesh/cuboid obstacles within preallocated capacities. Existing names
replace geometry, new names add slots, and removed names recycle slots. Unmentioned
meshes retain their Warp acceleration structures and device buffers retain their
addresses. When geometry is unchanged, use
`SceneCollision.update_obstacle_poses(names, poses, env_idx=0)` with one `Pose`
containing position `(N, 3)` and quaternion `(N, 4)` tensors in name order. The
batch API also covers voxel poses, preserves enable flags and performs one batch
inverse followed by in-place writes. It checks every name, environment, shape,
device, dtype and finite/unit-quaternion value before any pose write. Names must
be unique and present; empty batches are allowed. Quaternion squared norms must
be within `1e-5` of one. No broadcasting or implicit device conversion is applied.

Pose updates must be serialized with queries and other updates. Validation reads
a device scalar, so call outside CUDA graph capture; previously captured query
graphs continue reading the same buffers. Runtime/device failures during writes
require discarding the owning planner. The scalar and batch pose APIs update
device storage only; the CPU `scene_model` reference is unchanged.

The method validates names, types and capacity before mutation. Replacing a mesh
name shared with another environment is rejected. A device or geometry-load error
during mutation can leave a partial update: discard the owning planner on such an
error before further planning. This API does not promise transactional rollback.

## Passive joint effort

Complete robot models may retain passive joints with a URDF effort interval of
`[0, 0]`. This bound remains zero through loading and cloning; it does not grant
actuation. Kinematic motion profiles must hold any passive coordinate that the
controller cannot command. Reversed, non-finite and nonzero equal effort bounds
remain invalid.

## Runtime motion limits

Pose goalset planning honors `max_attempts` when an IK seed batch has no feasible
solution: it continues with the remaining attempts instead of rejecting the whole
request immediately. If the joint goalset search finds no successful trajectory,
it additionally searches individual target groups, at most once per group and
at most `max_attempts * num_ik_seeds` groups. Each group preserves all tool targets
at the original goalset index; returned indices still refer to that original set.
Only successful IK endpoints seed trajectory optimization, and every trajectory
must pass the normal collision, held-joint and motion-limit checks. Exhausting
both bounded searches without any IK solution returns no trajectory.

`MotionPlanner.update_joint_limits(position=..., velocity=..., acceleration=...,
jerk=...)` updates named limit tensors of shape `(2, dof)` in place, in the
planner's joint order. Omitted quantities retain their current values. All inputs
are validated before mutation; derivative intervals must strictly bracket zero.
Equal position bounds hold a coordinate, and restoring the original bounds
releases it. The caller must supply start/goal states compatible with the holds
and physical robot limits.

The update preserves the planner, IK, trajectory optimizer, graph planner and
captured CUDA graphs. Sampling bounds and constraint validators update together;
old seeds and roadmaps are cleared. Spline roundoff on a held coordinate is
projected to its constant position and zero derivatives before validation and
output, only when every sample is within four float32 ULPs at unit scale. Larger
violations remain subject to the ordinary constraint checks. Topology, tools and
collision capacity must be admitted at construction. Calls must be serialized
with planning; this method does not coordinate concurrent callers.

## Optional trajectory time refinement

`MotionPlannerCfg.create(..., trajopt_finetune_attempts=0)` runs the initial
trajectory optimization and skips additional time-optimal refinement passes for
`MotionPlanner` pose, joint and posture queries, including goalsets and graph-seeded
queries. Initial optimization, velocity/acceleration/jerk retiming, collision and
feasibility metrics, interpolated-trajectory validation and result ranking remain
active. Planning can finish sooner, while the resulting motion may take longer
to execute; compare total planning and execution time for the intended task.

The default `None` preserves each branch's existing refinement policy. Explicit
values must be nonnegative integers; booleans, negative values and nonintegers are
rejected before solver construction. `BatchMotionPlanner` retains its own
per-query refinement parameters.

## Citation

If you found this work useful, please cite cuRoboV2,

```
@misc{curobo_v2,
      title={cuRoboV2: Dynamics-Aware Motion Generation with Depth-Fused Distance Fields for High-DoF Robots},
      author={Balakumar Sundaralingam and Adithyavairavan Murali and Stan Birchfield},
      year={2026},
      eprint={2603.05493},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```

## Contributing

Contributions are welcome. Bugs: [open an issue](https://github.com/NVlabs/curobo/issues). General usage questions: [GitHub Discussions](https://github.com/NVlabs/curobo/discussions). For pull requests, please read [`CONTRIBUTING.md`](CONTRIBUTING.md). All commits must include a DCO sign-off (`git commit -s`).

## License

cuRobo is released under the [Apache 2.0 license](LICENSE).

The example robot assets bundled in this repository are provided under their respective licenses. See [LICENSE_ASSETS](LICENSE_ASSETS) for details.

## Third-Party Software

This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.

## YAML configuration compatibility

Configuration loading rejects Python-specific YAML tags. Writers emit portable
YAML data, converting supported robot tensors and parameter objects to plain
values. `RobotCfg.write_config` preserves the runtime robot object and requires
its original `generator_config` to produce a reloadable configuration. Convert
legacy Python-tagged configuration files to plain YAML before loading them.

## Partial joint posture goal sets

`MotionPlanner.plan_posture(goal_states, current_state, free_joints=(),
held_joints=(), tolerance=0.01, max_attempts=5, hold_axis=None,
allow_boundary_collision="none", max_initial_penetration=0.002, contact_links=None)` solves one joint goal set (1–256 rows).
All terminal coordinates except explicitly free joints must match one member;
held joints additionally preserve the current value throughout the trajectory.
They use equal position bounds during the call, restored on every exit, and a
separate 1e-5 numerical acceptance threshold independent of terminal tolerance.
The caller must resolve omitted task variables into held joints. The planner
uses the native IK/TrajOpt pipeline with a joint goal-set residual; Cartesian
tracking and Cartesian LM seed projection are disabled for the call and restored
afterwards. The LM seeder does not understand joint posture goals; projecting onto
the dummy current-tool pose would overwrite the sampled posture configurations.
Axis hold, physical
limits and collision checks remain active. Start contact supports up to eight static
cuboids or meshes, with a configurable initial sphere-proxy penetration bound in
metres. Captured contact cannot deepen, must clear the support by the endpoint,
and cannot recur after release. Other pairs retain ordinary collision checks;
other contact policies are rejected. Mesh queries reuse the current scene BVH.
The interpolated result is also checked against the declared joint tolerances.

The first attempt seeds exact authored goal members, retaining captured values
for free and held coordinates. Native endpoint checks can accept a feasible
goal directly before optimization; invalid goals retain all rejection checks.
Authored coordinates are not clipped or mixed across goal members. Subsequent attempts within
`max_attempts` explore deterministic limit-relative free-joint configurations and
intermediate posture candidates. Native IK checks intermediate candidates with
the same axis, held-joint, collision and start-contact constraints. Accepted
waypoints seed smooth two-leg trajectories for the full native trajectory solve;
they do not add task actions or relax final acceptance. Sampling is request-local,
bounded to at most 256 intermediate IK seeds per retry, and does not alter the
global random generator. Rejected intermediate batches retain endpoint-only
seeding; all-failed planning still reports failure. The result debug field
`posture_seed_attempts` records sampled and accepted intermediate counts.

When the first trajectory attempt fails, posture goal sets also use the native
graph planner for detour seeds within the same attempt limit and deadline. If
the roadmap query finds no route, the current posture seed remains available
for native optimization. Graph output is only a seed; collision, dense-path,
held-joint and terminal-goal acceptance stay unchanged.

`contact_links` restricts initial-contact capture to spheres on named robot links;
`None` considers all active spheres. Collision checks for other pairs stay active.

## Terminal support contact

`MotionPlanner.plan_pose` and `plan_cspace` accept `allow_boundary_collision="end"`
for one static cuboid or mesh support. The captured terminal sphere-model overlap
must not exceed 2 mm. The trajectory must begin clear and approach the captured
contact monotonically, without overshoot or rebound; goal IK accepts only the
captured terminal geometry. Only captured sphere/support pairs use this bounded
contact rule. Self collision and every other sphere/obstacle pair retain ordinary
checks, and request-scoped contact declarations are restored after the call.
Mesh checks use the current scene BVH, including the terminal mesh of an existing
cuboid-to-mesh `"both"` transfer. Multiple terminal supports remain ambiguous and
are rejected.

Pose queries with `"end"` or `"both"` bind each target group and kinematic endpoint
to its own terminal declaration. Deep or ambiguous endpoint candidates are
rejected before attempting a trajectory. The captured joint state is validated
again by the complete native IK metrics without re-optimizing it, then seeds
TrajOpt inside that same contact scope. A failed trajectory continues with the
remaining candidates and target groups. At most `max_attempts` captured
candidates are tried per group and at most `max_attempts * num_ik_seeds` groups
are visited; result diagnostics record the actual attempts and original selected
goal index. Unrelated solver/model errors still propagate.

Rejected terminal trajectories log bounded geometry diagnostics alongside the
optimized and interpolated constraint summaries. Raw mesh samples identify the
selected optimized peak's sphere, link and obstacle. Self-collision samples list
up to eight enabled pairs at the optimized start and peak, separating geometric
overlap from padding, in meters. Disabled spheres are omitted. These samples do
not replace contact-aware feasibility checks or explain unsampled interpolated
states. Diagnostic collection does not change collision masks or acceptance.
Boundary-contact penetration errors also identify the worst global sphere index
and its captured center and radius, so rejected endpoints can be traced back to
the robot collision model without relaxing contact tolerances.

Failed IK batches also report constraint maxima for up to eight converged but
rejected seeds, including seeds outside the optimizer's selected subset. The
total count and truncation flag distinguish a bounded sample from the full batch.
For a single collision environment, up to four such seeds also report eight raw
mesh gap pairs each using their existing FK spheres. These samples exclude
disabled spheres and meshes, do not apply contact allowance, and do not describe
non-mesh obstacles. Unsupported diagnostic environments are marked unavailable.

Self-collision in the CUDA Core backend keeps the shared-memory sphere cache for
models that fit the device's per-block capacity. Larger models read spheres from
global memory with the same padding, enabled pairs, reduction, and gradient
semantics. This retains collision coverage without reducing the robot geometry;
the existing sphere-index representation limit still applies. Global reads can
have a different performance cost and do not imply a planning speedup.

World-collision queries traverse populated obstacle slots with a bounded number
of parallel lanes per sphere. Reserved but unused cache capacity no longer
requires one GPU thread per sphere-slot pair. Slot IDs, enable flags, contact
replacements and per-environment counts remain live during CUDA Graph replay;
this optimization does not remove obstacles or narrow the query geometry.
Self-collision autograd also scales incoming gradients independently for each
batch and timestep when gradient-input scaling is enabled.

`MotionPlanner.plan_pose`, `plan_cspace`, and `plan_posture` accept an optional
positive, finite `timeout_s`. The default `None` retains unbounded wall-clock
behavior. A budget covers nested proposal, optimization, refinement, and graph
search stages without restarting on each attempt. Expiry raises `TimeoutError`
and unwinds temporary constraints and optimizer settings. Checks are cooperative:
queued optimization chunks finish before the deadline is checked, so this is not
a hard preemption of a CUDA kernel or native operation. Call-local budget state
is cleared even on failure, allowing a later request to reuse the planner.
