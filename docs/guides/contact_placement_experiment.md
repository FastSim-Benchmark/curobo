# Run placement with a final normal landing

Use this developer experiment to plan a held payload onto a known static support
when collision-sphere overlap makes the final state fail ordinary collision checks.
The experiment optimizes one complete B-spline trajectory from the start state to
the caller's final pose, including transport and the final normal landing. There is
no preplacement waypoint, fixed lift offset, fixed landing time, or concatenation
of separately solved trajectories. The support gap along the solved trajectory
determines when final alignment is required.
It shares the synthetic three-axis robot, sphere geometry, and table/tray/cabinet
fixtures from the [departure experiment](contact_separation_experiment.md).

`SceneCollisionCostCfg.goal_contact` is an opt-in special-purpose parameter. Its
default is `None`, so ordinary collision checking remains the default. Physical
placement requires contact; the permitted sphere penetration is a bounded model
tolerance, not a request to drive the physical payload through the support.

1. Capture the selected payload spheres at the intended world-frame placement pose.
   Declare their indices, the support cuboid name, and a `GoalContact` containing
   those `goal_spheres`. The default maximum terminal overlap is 2 mm, approach
   clearance is 5 mm, and numerical geometry tolerance is 0.01 mm.
2. Set `GoalContact.landing` to a `NormalLanding` with a tool frame rigidly connected
   to the payload, the final world position and quaternion (wxyz), and the outward
   unit support normal. Inside the approach band, the tool must remain on the final
   normal line within 0.1 mm and keep the final orientation within 0.001 rad by
   default. The normal is checked against the captured support geometry.
3. Install the declaration in the trajectory optimizer's cost, constraint, and
   original/interpolated trajectory metrics configurations before CUDA graph capture.
4. For pose planning, also install a copy with `terminal_only=True` in the dedicated
   goal IK cost and metrics configurations. Ordinary collision-aware IK would otherwise
   reject the placement pose before trajectory optimization starts. This mode accepts
   only single-state queries; it must not be used for trajectory rollouts or general
   state validation.
5. Call `plan_pose(final_pose, current_state)` or `plan_cspace(final_state, current_state)`
   on the configured planner. Each attempt optimizes all trajectory control points
   together. Native iterations, retiming, and retries still operate on that complete
   trajectory; the example never solves for an intermediate target. No separate
   global linear-motion criterion is used, so transport can move laterally and
   change orientation outside the required landing band.
6. Keep the payload attached while planning and executing placement. Grasp release,
   detachment, and updating the placed object's world representation belong to the
   subsequent task phase and are not performed by this experiment.

The declaration replaces only the named sphere/support pairs. The trajectory must
start outside the approach clearance. Once it enters that band, its signed gap may
only decrease toward the captured goal contact, within numerical tolerance. It may
not overshoot the declared goal penetration, bounce away and recontact, or finish
at a different sphere pose. With `NormalLanding`, contact is allowed only after
lateral position and orientation alignment, during the final normal approach;
sliding or rotating on the support beyond the declared tolerances is rejected.
All other sphere/obstacle pairs retain ordinary collision checks.

`GoalContact.landing` defaults to `None` for the earlier bounded-contact behavior.
That mode constrains support gap and final sphere geometry but allows lateral
motion or rotation during contact. Enable `NormalLanding` when placement must
finish with a normal landing.

Choose `approach_clearance` as a safety margin appropriate to the scene. It is
configurable and defaults to 5 mm; it is not an absolute height or a waypoint. The
optimizer can change the path and the time at which it enters this band. Validity
checks require alignment throughout the band, within the declared pose tolerances.
With positive `SceneCollisionCostCfg.activation_distance`, the optimizer starts
alignment guidance that much earlier and reduces its alignment dead zone toward
zero. Zero-activation validity checks retain the declared tolerances. Negative
activation distances cannot weaken the landing alignment requirement.
The example sets optimizer activation distance to the larger of its normal 5 mm
avoidance setting and the requested approach clearance, so increasing the hard
alignment band also increases the optimizer's advance guidance.

The contact penalty allows either clear transport or aligned landing. Taking the
minimum of the clearance and alignment violations provides a correction near the
band boundary without the vanishing gradient of their product. Native scene costs
for all other collision pairs retain their configured activation distances.

"Final landing" denotes the last motion phase, not a promise that only the last
sample can touch. A continuous path to a target with sphere-model overlap must
cross zero gap before the endpoint. The user's final height is preserved, and no
additional penetration beyond that captured target and numerical tolerance is
permitted. A zero-gap target can instead finish at first contact in an exact model.

The implementation evaluates the existing departure constraint with reversed time,
sharing its support checks, interpolation, and geometry bounds. A moved, resized, or
disabled support invalidates the declaration. Simultaneous `start_contact` and
`goal_contact` now support departure and final approach in one trajectory, including
a shared support. The public [boundary-contact parameter](../reference/boundary_contact.md)
captures declarations for each planning call without manual cost configuration.
Contact distances use float32 component arithmetic so global TF32 matmul settings
cannot round away the declared geometry tolerance. This also applies to departure.

Run the complete pose-planning comparison, including goal IK:

```bash
python -m curobo.examples.reference.contact_placement \
  --output-dir /path/to/placement-output --repeats 10 \
  --goal-position 0 0 0.12
```

`--goal-position X Y Z` is required and supplies the final world-frame gripper
position in meters. The three-axis fixture has fixed orientation. The example
never changes the supplied height. Use `--support-height` to translate the fixture
support/cabinet, and `--approach-clearance` to set the alignment band. For example,
`--support-height 0.06 --goal-position -0.05 0 0.181 --approach-clearance 0.008`
places the fixture payload at zero model gap on an elevated support.

The output includes per-case JSON results, available trajectories, and `results.json`.
The ordinary baseline must reject the contacting goals for the table, tray, and low
cabinet. Explicit goal contact must permit those three fixtures. A sealed cabinet
must remain infeasible. Every successful repetition receives independent dense
analytic and ordinary unmasked cuRobo sphere/box collision checks, outside the timed
planning interval. Alignment checks independently recompute FK at 8x the returned
trajectory density, check transport for early support contact, and report lateral
and orientation errors in the approach band. The reported landing-start fraction
is measured from the optimized trajectory, not prescribed as an input. Warm timings
include goal IK and full-trajectory optimization, excluding initial compilation,
graph capture, and validation.

Run the focused checks with:

```bash
python -m pytest -o addopts= \
  curobo/tests/_src/collision/test_contact_approach.py \
  curobo/tests/_src/collision/test_contact_landing.py \
  curobo/tests/_src/motion/test_contact_placement.py
```

This is an experimental configuration binding, not a reusable request-scoped contact
API. Each planner is bound to one immutable goal/scene and robot sphere layout.
Only one environment and one static cuboid support are supported. The example
configures native IK and TrajOpt; contact-aware PRM connections are not implemented.
The sphere model and sampled checks do not certify physical mesh clearance or model
friction, contact forces, compliance, grasp stability, or a real arm's performance.

## Observed experiment

On an RTX 5090 with PyTorch 2.11.0+cu128 and CUDA graphs enabled, ten warm
`plan_pose` repetitions per fixture produced the following results with normal
landing enabled. Timings include goal IK and the full-trajectory optimization;
compilation/capture is excluded.

| Placement scene | Normal checking | Explicit goal contact | Median planning time |
| --- | ---: | ---: | ---: |
| Table | 0/10 | 10/10 | 131.0 ms |
| Tray | 0/10 | 10/10 | 131.6 ms |
| Low cabinet | 0/10 | 10/10 | 132.9 ms |
| Sealed cabinet | Not measured | 0/10 | 138.4 ms |

Every successful repetition passed independent dense checks, with no contact
before final alignment, no forbidden collision, and the supplied final position
preserved. The target retained its declared 1 mm terminal sphere overlap.
The three-axis fixture cannot rotate, so orientation rejection is covered by
separate tests, including rotation with a stationary contact sphere.

The cabinet provided 30 mm headroom at the placement pose. Earlier two-plan
normal-landing runs took about 153–158 ms and required a fixed preplacement offset.
Joint optimization removes that imposed waypoint and the required intermediate stop; it
does not promise lower solve time or a globally optimal trajectory. Separate
regressions exercise caller-supplied heights, a translated support, configurable
approach clearance, and the absence of intermediate planning requests. These
fixed-fixture repetitions do not measure general planning success rates or
physical execution reliability.
