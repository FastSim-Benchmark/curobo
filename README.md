<!-- SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# cuRobo

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
tracking is disabled for the call and restored afterwards. Axis hold, physical
limits and collision checks remain active. Start contact supports up to eight static
cuboids or meshes, with a configurable initial sphere-proxy penetration bound in
metres. Captured contact cannot deepen, must clear the support by the endpoint,
and cannot recur after release. Other pairs retain ordinary collision checks;
other contact policies are rejected. Mesh queries reuse the current scene BVH.
The interpolated result is also checked against the declared joint tolerances.

`contact_links` restricts initial-contact capture to spheres on named robot links;
`None` considers all active spheres. Collision checks for other pairs stay active.
