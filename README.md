<!-- SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# cuRobo

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

The first attempt retains ordinary endpoint seeding. Subsequent attempts within
`max_attempts` explore deterministic limit-relative free-joint configurations and
intermediate posture candidates. Native IK checks intermediate candidates with
the same axis, held-joint, collision and start-contact constraints. Accepted
waypoints seed smooth two-leg trajectories for the full native trajectory solve;
they do not add task actions or relax final acceptance. Sampling is request-local,
bounded to at most 256 intermediate IK seeds per retry, and does not alter the
global random generator. Rejected intermediate batches retain endpoint-only
seeding; all-failed planning still reports failure. The result debug field
`posture_seed_attempts` records sampled and accepted intermediate counts.

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

Failed IK batches also report constraint maxima for up to eight converged but
rejected seeds, including seeds outside the optimizer's selected subset. The
total count and truncation flag distinguish a bounded sample from the full batch.
For a single collision environment, up to four such seeds also report eight raw
mesh gap pairs each using their existing FK spheres. These samples exclude
disabled spheres and meshes, do not apply contact allowance, and do not describe
non-mesh obstacles. Unsupported diagnostic environments are marked unavailable.
