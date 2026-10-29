.. _tutorial_motion_planning:

Motion Planning
===============

.. automodule:: curobo.examples.getting_started.motion_planning
   :no-members:

Complete joint posture targets
------------------------------

``MotionPlanner.plan_posture`` uses joint-space planning for a single target
that specifies every active coordinate, with no free or held coordinates and
ordinary collision checks. It preserves the requested joint endpoint instead
of first projecting that endpoint through Cartesian IK. Tool-axis constraints,
the planning deadline, native collision and derivative checks, and the caller's
posture endpoint tolerance still apply. Partial goals, held coordinates,
multiple target candidates and boundary-contact requests retain posture
planning.

Posture goal sets can use native graph seeds after the first unsuccessful
trajectory attempt. The attempt limit and shared planning deadline still bound
the request. When a graph query finds no seed, the posture seed remains eligible
for native optimization; graph output alone never certifies a trajectory.
Every accepted path retains the original joint, collision, dense interpolation,
held-coordinate and terminal-goal checks.
