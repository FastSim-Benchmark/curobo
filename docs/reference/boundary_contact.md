# Per-call boundary contact

`MotionPlanner.plan_pose` and `MotionPlanner.plan_cspace` accept one optional
keyword, `allow_boundary_collision`:

| Value | Endpoint behavior |
| --- | --- |
| `none` | Ordinary collision checking; the default. |
| `start` | Capture a shallow initial sphere/cuboid contact and require departure. |
| `end` | Capture a shallow terminal sphere/cuboid contact and require final approach. |
| `both` | Permit departure followed by final approach in one optimized trajectory. |

The planner captures endpoint spheres from its own kinematics and current scene.
Each endpoint supports one enabled static cuboid in one environment. Maximum model
overlap is 2 mm, release/approach clearance is 5 mm, and numerical tolerance is
0.01 mm. Deeper overlap and ambiguous multiple supports are rejected. An endpoint
without cuboid contact retains ordinary checks. Mesh contact is not relaxed.
Payload geometry must already be attached to the planning collision model.

The selected sphere/cuboid pairs receive bounded contact constraints. Other scene
pairs, robot self collisions, and joint limits keep ordinary checks. Start and end
supports may be the same or different. Shared pairs must achieve release clearance
between endpoints; sliding throughout or repeated middle contact is rejected.
The optimizer chooses the departure and approach timing, without an intermediate
waypoint or concatenating independently planned trajectories.

For pose goals, preliminary IK obtains kinematic endpoint evidence with only its
scene costs disabled. That seed is never executed or accepted as a trajectory.
The subsequent goal IK checks the captured contact and all other pairs; trajectory
costs, constraints, original metrics, and interpolated metrics enforce the complete
contact contract. A failed candidate returns planning failure or a declaration error.

All contact state is local to the call and is cleared after success, failure, or
exception. Captured CUDA programs are invalidated when contact objects change;
CUDA graph execution remains enabled and the next call recaptures its own program.
This entails graph capture overhead for contact requests. Contact queries use native
TrajOpt without PRM, whose endpoint connections retain ordinary collision semantics.
The parameter composes with `hold_axis`; neither relaxes the other constraint.

`SceneCollisionCostCfg` also accepts simultaneous explicit `StartContact` and
`GoalContact` declarations. `GoalContact.terminal_only` remains exclusive to
single-state goal IK and cannot be combined with a departure declaration.
Per-call automatic capture rejects preconfigured explicit contact declarations.

These constraints describe collision-sphere geometry and sampled path checks. They
do not establish physical grasp stability, friction, contact force, or exact mesh
collision guarantees. The [departure experiment](../guides/contact_separation_experiment.md)
and [placement experiment](../guides/contact_placement_experiment.md) describe the
geometry model and independent validation used by the development fixtures.
