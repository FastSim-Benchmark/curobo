# Keep a held object upright during transport

## Per-motion axis parameter

Use `AxisHold` on a single `MotionPlanner.plan_pose` or `plan_cspace` call to
preserve a tool-local direction from the supplied start state:

```python
from curobo.types import AxisHold

hold = {"panda_hand": AxisHold(axis=(0.0, 0.0, 1.0), tolerance_rad=0.01)}
result = planner.plan_pose(goal, start, hold_axis=hold)
result = planner.plan_cspace(joint_goal, start, hold_axis=hold)
```

This parameter needs no orientation metrics preset or global criteria update.
Its independent native axis cost guides IK and trajectory optimization, and its
feasibility cost checks the optimization and interpolated trajectories. XYZ
translation and twist are free subject to the ordinary endpoint and collision
constraints. Pose endpoints keep their requested orientation; joint endpoints
keep their exact joint values. Conflicting endpoints are infeasible. The starting
reference is independent of the goal quaternion and is recaptured for each call.
Buffers are reserved before CUDA graph capture; changing or omitting the parameter
does not leave a constraint on subsequent calls, including after exceptions.

For a rigidly held cup, express the cup's local opening axis in EE coordinates
using the grasp transform. The EE itself need not point up. An initially upright
cup remains upright within the angular tolerance; an initially tilted cup is not
automatically leveled. Native `StartContact`, `GoalContact`, and `NormalLanding`
remain active and are tested together with this parameter. This API targets one
planning problem per call; the batch planner does not expose it.

The following lower-level criteria recipes remain available when a caller wants
to configure pose tracking itself, including changing terminal orientation DOFs.

Keep a cup's up axis vertical while allowing arbitrary yaw and a freely planned
position path. Use `ToolPoseCriteria.hold_axis` with the existing
`MotionPlanner.update_tool_pose_criteria` API. Full orientation locking is also
available when the task requires it.
The complete B-spline trajectory is optimized together; no intermediate height,
fixed Cartesian path, or extra motion phase is required.

## Keep the cup upright with free yaw

Create the planner with `metrics_rollout="metrics_orientation.yml"` to include
pose constraints in trajectory validation. Supply the cup's up direction in the
EE frame, using the rigid grasp transform:

```python
from curobo.types import ToolPoseCriteria

# ee_pose and cup_pose are measured poses in the same robot base frame.
cup_in_ee = ee_pose.inverse().multiply(cup_pose)
cup_axis_in_ee = cup_in_ee.get_rotation_matrix()[0, :, 2]
criteria = ToolPoseCriteria.hold_axis(
    axis=cup_axis_in_ee,
    tolerance=0.01,  # radians of cup tilt, approximately 0.57 degrees
    device_cfg=planner.device_cfg,
)
planner.update_tool_pose_criteria({tool_frame: criteria})
```

`orientation_axis` is the local vector that the pose cost rotates into the
reference frame. The error is the angle between `R(current) @ axis` and
`R(goal) @ axis`, independent of twist. This is the actual directional tilt,
including near 180 degrees of yaw; an upside-down cup has pi radians of error.
Setting a yaw weight to zero in the original quaternion error is not equivalent.

Choose a goal quaternion that maps this axis to world up. If the grasped cup
starts exactly upright, the initial EE quaternion is one such reference. Retain
this grasp transform throughout transport. Do not infer the cup axis by assuming
the EE's local Z axis points through the cup opening. In a tilted robot base,
express world up in the robot base frame first.

With `hold_axis`, yaw is free at the endpoint too; rotating the goal quaternion
about the desired up direction does not command a yaw change. Position still
converges to the requested XYZ goal. For a goalset, every alternative must imply
the same desired up direction to keep the reference constant along the path.
The three rotation factors must be equal and nonnegative in each terminal or
nonterminal group; partial Euler-axis weighting is rejected in this mode.

Run the attached-cup example, which also enables initial contact separation:

```bash
python -m curobo.examples.reference.cup_upright --repeats 5
```

It uses a seven-axis Franka arm, an authored upright cup represented by four
collision spheres, and 1 mm initial overlap with a cuboid support. It checks tilt,
support separation, and all other sphere/support pairs on eight subdivisions of
each returned joint segment. This is a geometric planning experiment, not a
simulation of grasp compliance or liquid dynamics.

## Configure optimization and validation

Create the planner with `metrics_orientation.yml` to reject orientation violations
over both the optimization samples and the interpolated trajectory. This opt-in
preset preserves the collision and joint-limit checks from `metrics_base.yml`.
The ordinary preset checks orientation convergence only at the endpoint.

For **full orientation locking**, including yaw, use the original criteria:

```python
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import GoalToolPose, JointState, ToolPoseCriteria

orientation_tolerance = 0.01  # radians, about 0.57 degrees
config = MotionPlannerCfg.create(
    robot="franka.yml",
    metrics_rollout="metrics_orientation.yml",
    orientation_tolerance=orientation_tolerance,
    interpolation_dt=0.01,
)
planner = MotionPlanner(config)
criteria = ToolPoseCriteria(
    terminal_pose_axes_weight_factor=[1, 1, 1, 1, 1, 1],
    non_terminal_pose_axes_weight_factor=[0, 0, 0, 1, 1, 1],
    terminal_pose_convergence_tolerance=[0, orientation_tolerance],
    non_terminal_pose_convergence_tolerance=[0, orientation_tolerance],
    device_cfg=planner.device_cfg,
)
planner.update_tool_pose_criteria({frame: criteria for frame in planner.tool_frames})
```

The intermediate XYZ weights are zero, so transport can go around obstacles.
The terminal XYZ weights remain one, so the destination remains an optimization
goal. Do not replace this with `ToolPoseCriteria.track_orientation()`: that helper
also zeros the terminal XYZ weights.

The validation preset uses the existing axis-angle pose cost with unit rotation
weight and unit rotation-axis factors, making the validation threshold an angle
in radians. Keep those weights unchanged when using an angular tolerance. Set a
finite positive tolerance; zero error cannot be required numerically.

## Use the grasp orientation as the reference

Read `start` from the robot's current joint state after grasping. The following
default joints and relative displacement are only a runnable Franka example:

```python
start = JointState.from_position(
    planner.default_joint_state.position.unsqueeze(0),
    joint_names=planner.joint_names,
)
initial_pose = planner.compute_kinematics(start).tool_poses
goal = GoalToolPose(
    tool_frames=initial_pose.tool_frames,
    position=initial_pose.position.unsqueeze(-2)
    + planner.device_cfg.to_device([0.0, 0.12, 0.06]),
    quaternion=initial_pose.quaternion.unsqueeze(-2).clone(),
)
result = planner.plan_pose(goal, start)
if result is not None and bool(result.success.all()):
    trajectory = result.get_interpolated_plan()
```

Supply your actual destination in the robot base frame. Keep the reference
quaternion constant across calls that belong to the same transport operation;
recapturing it on every replan can accumulate orientation drift. For a goalset,
all alternatives must share the same reference quaternion for each held tool.
For multiple tools, supply criteria explicitly for every frame; use
`ToolPoseCriteria.disabled()` for frames that should be unconstrained.

The object must already be upright and rigidly attached to the end effector.
This full-orientation variant also fixes yaw. It preserves the object's
initial orientation; it does not infer a cup's up axis or correct an initially
tilted grasp. A path that requires tilting through a narrow opening can become
infeasible. Liquid sloshing additionally depends on acceleration.

## Combine with pick and place

For full orientation locking, keep the grasp quaternion in the transport and
placement goals. When configuring `NormalLanding`, use the same quaternion as
its `goal_quaternion`. The transport orientation constraint then applies before
the final landing band as well. `hold_axis` also composes with `StartContact`:
the contact condition and tilt condition must both pass. `NormalLanding` retains
its own full-orientation condition inside the landing band, so using it with
`hold_axis` makes yaw free during transport and restricted during final landing.
Continue to configure `StartContact` or `GoalContact` on the scene collision cost
as in the [departure](contact_separation_experiment.md) and
[placement](contact_placement_experiment.md) guides. If you already customize the
metrics configuration, add the preset's `tool_pose_cfg` to your existing
`constraint_cfg` instead of replacing your contact configuration.

This recipe uses `plan_pose`. Do not assume that joint-space planning or grasp
helpers that change pose criteria preserve the setting. PRM supplies seeds;
TrajOpt must still optimize and validate the final result with these criteria.

## Verify the result

```bash
python -m curobo.examples.reference.tool_orientation --repeats 5
```

The example keeps CUDA graphs enabled and independently checks FK on eight
subdivisions of every returned joint segment. It reports the maximum angular
error and final position error for every successful plan.

Success certifies the configured sampled constraints, including interpolated
samples. It is not a mathematical continuous-time guarantee or a guarantee about
hardware tracking error. Match the interpolation interval to the executor and
leave margin between the planning tolerance and the task's allowed tilt.
