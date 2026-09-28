# Per-call boundary contact

`MotionPlanner.plan_pose` and `MotionPlanner.plan_cspace` accept one optional
keyword, `allow_boundary_collision`:

| Value | Endpoint behavior |
| --- | --- |
| `none` | Ordinary collision checking; the default. |
| `start` | Capture shallow initial sphere/support contact and require departure. |
| `end` | Capture shallow terminal sphere/support contact and require final approach. |
| `both` | Permit departure followed by final approach in one optimized trajectory. |

The planner captures endpoint spheres from its own kinematics and current scene.
Each terminal endpoint supports one enabled static cuboid or mesh in one environment. Maximum model
overlap is 2 mm, release/approach clearance is 5 mm, and numerical tolerance is
0.01 mm. Deeper overlap and ambiguous multiple supports are rejected. An endpoint
without support contact retains ordinary checks. Mesh contacts use the same
bounded clearance contract; mesh departure currently requires start-only queries.
Payload geometry must already be attached to the planning collision model.

Endpoint admission uses the declared numerical tolerance when comparing the
captured gap with the model-overlap limit, just as the path constraints do.
For the defaults, a computed overlap above 2.01 mm is rejected. This accounts
for rounding at the 2 mm boundary without changing sphere geometry, the declared
2 mm model-overlap limit, or the required 5 mm departure clearance. The tolerance
does not accumulate across trajectory samples; deeper penetration and recontact
remain constrained against the captured endpoint and the best previous clearance.

The selected sphere/support pairs receive bounded contact constraints. Other scene
pairs, robot self collisions, and joint limits keep ordinary checks. Start and end
supports may be the same or different. Shared pairs must achieve release clearance
between endpoints; sliding throughout or repeated middle contact is rejected.
The optimizer chooses the departure and approach timing, without an intermediate
waypoint or concatenating independently planned trajectories.

For pose goals with an attached payload, preliminary IK first keeps the body and
all scene obstacles active while temporarily suspending only the attached link's
collision spheres. This lets the body avoid furniture before the payload's
terminal support contact is known. The complete sphere buffer is restored before
any candidate is captured or returned, including on solver exceptions and when a
consumer stops after its candidate budget. Restored payload penetrations above
the existing contact bound are rejected. The subsequent endpoint metrics check
payload self collision and every non-contact scene pair with the full geometry.

The existing kinematic proposal pass, with only scene costs disabled, follows
when more candidates are needed or no payload is attached. Each pass runs one
bounded IK batch. Terminal pose queries try up to `max_attempts` fresh pairs of
proposal passes per target when previous passes exhaust their candidates. The
total captured endpoint budget per target remains `max_attempts`, across all
proposal batches; successful planning stops immediately. An empty first batch
therefore does not terminate a target's remaining search budget. Exhausted
queries may take longer, but both native proposal calls and endpoint/trajectory
checks stay bounded. Search diagnostics include the proposal batch count and
the batch index of each captured endpoint.
Neither kind of seed is executed or accepted as a trajectory.
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
