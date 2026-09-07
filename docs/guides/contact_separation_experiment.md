# Run the contact departure experiment

This developer experiment plans from an attached payload touching a static support.
It uses an authored three-axis Cartesian gripper, four payload spheres, and cuboid
obstacles. It requires CUDA but no simulator, external robot assets, or credentials.

This is an opt-in special-purpose parameter: `SceneCollisionCostCfg.start_contact`
defaults to `None`. Enable it explicitly with a captured `StartContact` declaration
for a known initial support contact. Attaching a payload does not enable it automatically.

Run from an environment with cuRobo and its test dependencies installed:

```bash
python -m curobo.examples.reference.contact_separation \
  --output-dir /path/to/experiment-output --repeats 10
```

The output includes `results.json`, individual case results and trajectories, and
`contact_departure.png`. Cold compilation and graph capture are recorded separately
from repeated planning. Repeated fixed scenes and seeds measure repeatability, not
a general planning success probability.

The experiment compares normal collision checking and an experimental
`SceneCollisionCostCfg.start_contact` declaration in these cases:

- A payload whose bottom sphere overlaps a tabletop by 1 mm.
- The same initial contact inside a tray with a 20 mm rim.
- A low cabinet with 30 mm of initial gripper headroom. A 100 mm vertical lift
  intersects the cabinet ceiling; extraction must respect the available clearance.
- A closed cabinet, which must remain infeasible.
- A cabinet control with an initially collision-free state, to distinguish contact
  failure from robot/configuration or trajectory optimization failures.

The declaration captures specific sphere/support pairs and a bounded geometry-error
tolerance. Their signed gaps may not decrease beyond numerical tolerance, and the
terminal state must achieve the release clearance. Once release clearance has been
achieved, it must be maintained. All other robot/world and robot self-collision pairs
keep their normal checks. The experiment never disables a world obstacle or robot
link. An invalid initial declaration or changed support geometry is rejected.

The declaration is installed in trajectory optimization and both original and
interpolated trajectory validation before CUDA graph capture. IK and PRM retain
normal collision semantics. This experiment uses TrajOpt without PRM fallback;
contact-aware graph connections are not implemented. The fixture binds a separate
planner to each immutable start/scene. It is not a reusable request-scoped contact
API, and it must not be reused after scene storage or robot topology is replaced.

The current prototype supports one environment and one static cuboid support per
declaration. The selected contact pairs use differentiable box distances with four
linear sphere samples per interval. Independent validation samples the returned
trajectory eight times more densely and checks every sphere/box pair against both
an analytic distance calculation and ordinary cuRobo queries with no contact
replacement. These are sampled checks of the stated sphere geometry, not exact
continuous collision certification, physical mesh validation, or contact-force
simulation. A cup's hollow interior, friction, deformation, and grasp stability are
outside this experiment.

Run the focused regressions with:

```bash
python -m pytest -o addopts= \
  curobo/tests/_src/collision/test_contact_separation.py \
  curobo/tests/_src/motion/test_contact_departure.py
```

The explicit `addopts` override allows serial execution when pytest-xdist is not
installed; it does not change which tests run.

## Observed experiment

A run on an NVIDIA GeForce RTX 5090 with PyTorch 2.11.0+cu128, CUDA graphs enabled,
and ten warm repetitions per fixed case produced these results:

| Scene | Normal checking successes | Contact departure successes | Contact median planning time |
| --- | ---: | ---: | ---: |
| Table, 1 mm initial overlap | 0/10 | 10/10 | 66.4 ms |
| Tray, 1 mm initial overlap | 0/10 | 10/10 | 66.0 ms |
| Low cabinet, 1 mm initial overlap | 0/10 | 10/10 | 65.9 ms |
| Sealed cabinet | Not measured | 0/10 | 69.0 ms |

The low-cabinet path rose at most 24.7 mm while its gripper origin remained inside
the cabinet footprint. Independent checks at 2,401 samples found no forbidden
collision, with 5.1 mm minimum clearance to the other sphere/obstacle pairs. A
100 mm initial vertical lift would penetrate an obstacle by 31.0 mm.

The normal low-cabinet control, starting 11 mm higher and therefore already clear
of the support, succeeded 10/10 with a 31.3 ms median. Its different initial state
means this comparison is not an isolated measurement of contact-kernel overhead.
These measurements establish feasibility on the fixtures, not general success
rates, physical-arm performance, or a production latency guarantee.
