.. _sphere_fitting_note:

Fitting Spheres to Geometry
==================================

cuRobo represents robots and grasped objects as sets of spheres for collision checking.
This page describes the available sphere fitting techniques.

An axis narrower than the nominal voxel pitch receives a seed at its
midpoint, so thin geometry is not rejected merely because its
initial seeds lie outside the bounding box. This also applies to MorphIt's
voxel initialization; sampling on other axes, the mesh and collision checks
are unchanged.

.. _attach_object_note:

Use Cases
----------

- **Grasped objects**: During pick-and-place, the grasped object must be checked for collisions
  with the world. cuRobo approximates it as spheres and attaches them to the robot's kinematic
  model.
- **Robot links**: Robot geometry is approximated with spheres for self-collision and world
  collision checking. See :ref:`tutorial_build_robot_model` for configuring robot spheres.

Entry Point
-----------

The main function is :func:`curobo.sphere_fit.fit_spheres_to_mesh`:

.. code-block:: python

   from curobo.sphere_fit import fit_spheres_to_mesh, SphereFitType
   import trimesh

   mesh = trimesh.load("my_object.obj")

   # Automatic sphere count with default density
   result = fit_spheres_to_mesh(mesh)

   # Explicit sphere count
   result = fit_spheres_to_mesh(mesh, num_spheres=50)

   # With quality metrics
   result = fit_spheres_to_mesh(mesh, compute_metrics=True)
   print(f"Coverage: {result.metrics.coverage:.2%}, Protrusion: {result.metrics.protrusion:.2%}")

.. list-table:: Parameters
   :header-rows: 1
   :widths: 25 75

   * - Parameter
     - Description
   * - ``num_spheres``
     - Sphere budget. FAST defaults to ``ceil(32 * sphere_density)``, clamped
       to 1--256. Legacy methods estimate from bounding-box volume and density.
   * - ``sphere_density``
     - Density multiplier for auto sphere count (default ``1.0``).
       ``2.0`` doubles the count, ``0.5`` halves it. Range: ``0.1`` -- ``10.0``.
   * - ``fit_type``
     - Fitting algorithm (default ``FAST``). See below.
   * - ``surface_radius``
     - Radius for surface-sampled spheres. Only affects ``SURFACE`` fit type.
   * - ``iterations``
     - Optimization iterations for ``MORPHIT`` (default ``200``). FAST uses its
       bounded internal optimizer and does not use this parameter.
   * - ``compute_metrics``
     - When ``True``, populates quality metrics on the result using the requested
       ``device_cfg`` device, including CPU. CPU queries are not redirected to CUDA.
   * - ``clip_plane``
     - Half-plane constraint ``((nx, ny, nz), offset)`` in mesh-local coordinates.
       Spheres that cross the plane are penalised during MorphIt optimization and
       hard-clamped afterwards.  Useful for keeping base-link spheres from
       protruding into a mounting surface.


Fit Types
----------

cuRobo provides seven fitting selections via :class:`curobo.sphere_fit.SphereFitType`:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Type
     - Description
   * - ``FAST`` (default)
     - Fits a budgeted sphere proxy on the CPU. Uses original geometry, a
       finite-view visual hull, and joint center/radius optimization. CoACD
       decomposition is attempted only if the basic fit fails its audit.
   * - ``BOUNDED``
     - Requires explicit ``num_spheres`` capacity (1--16384) and
       ``max_protrusion_m > 1e-5``. Verifies complete component volume coverage
       and a source-relative outward distance bound after output conversion
       and clipping. Unsupported geometry and insufficient capacity raise.
   * - ``AUTO``
     - Routes explicit ``max_protrusion_m`` requests to BOUNDED. Otherwise,
       routes a single closed rectangular plate with an explicit 4--50 sphere
       budget to RIM when its thickness is at most 20% of its intermediate
       extent. All other inputs use FAST, including compounds and requests
       without an explicit budget. Does not retry another method after failure.
       ``debug_info["routing"]`` records the selected backend and reason;
       ``debug_info["fit_type"]`` identifies the actual backend. Automatic RIM
       selection certifies the outer contour band, not the filled volume or
       a source-relative outward bound. Explicit selections and defaults remain
       unchanged. For a phone-sized rectangular plate, use
       ``fit_type=SphereFitType.AUTO, num_spheres=40``.
   * - ``RIM``
     - Requires an explicit budget in 4--50. Covers a plate's complete
       outer convex projected contour at every height in its observed rim band.
       Does not certify filled-face or volume coverage; clipping that loses
       contour or boundary-vertex coverage raises.
   * - ``SURFACE``
     - Samples the mesh surface evenly with fixed-radius spheres. Fast fallback
       for thin or degenerate meshes.
   * - ``VOXEL``
     - Voxelizes the bounding box, filters by SDF to keep interior voxels, and
       assigns inscribed radii. Good for convex shapes.
   * - ``MORPHIT``
     - Initialises with ``VOXEL``, then runs Adam optimization to minimise
       coverage gaps and protrusion. Remains available by explicit selection.

FAST uses up to 32 spheres at the default density. Automatic budgets are
``ceil(32 * sphere_density)``, clamped to 1--256; an explicit ``num_spheres``
is a budget in 1--256. Unlike the legacy volume heuristic, this budget is
independent of mesh units. The output may use fewer spheres. FAST permits
filled cavities and local uncovered gaps; it is not a conservative enclosing
volume or a no-missed-collision certificate. It limits global support excess
in 154 sampled directions to 2.5% of the longest PCA extent before clipping.
This is not a Hausdorff bound.

BOUNDED is an opt-in alternative for close-clearance payload collision. Inputs
must contain closed, consistently outward-oriented components with verifiable
interior polyhedral kernels. Components are split without hole repair. A bound
on each component hull's excess is deducted from the outward budget before
sphere fitting. For nontrivial excess, the bound is tightened by partitioning
the hull into source-face cones and constructing feasible witnesses inside
each source tetrahedron. This avoids extrapolating one shallow local dent
across the entire component; excessive concavity is still rejected. Convex cell containment checks
cover complete volumes, with a 0.1 micrometre numerical containment margin,
rather than relying on surface samples. Sphere deletion retains coverage of
every certified terminal cell. Verified rectangular hulls use orthogonal box
subdivision and a bounded radius search for a sphere covering each whole cell.
Other hulls retain tetrahedral subdivision, with enclosing-ball feasibility
checked before further refinement. Every candidate is checked after numerical
conversion. Compression uses spatial sphere queries and deterministic lazy
greedy selection; final output coverage is independently checked again.
Refinement is bounded by 32 levels and 500000
cell visits per component per verification pass; exhaustion raises explicitly.
The output supports float32 and float64. BOUNDED never truncates its sphere set
or falls back to another algorithm. Clipping that invalidates coverage raises.
When a bottom-protrusion bound is supplied, the tighter of it and
``max_protrusion_m`` applies to all directions. Results expose
``debug_info["bounded"]["full_volume_covered"]`` and
``verified_max_protrusion_m`` for the final returned geometry. For BOUNDED,
``fit_time_s`` includes final converted-geometry coverage verification.

This preprocessing can need thousands of spheres and substantial CPU time for
detailed thin geometry. Reserve sufficient attached-object capacity and measure
runtime collision cost before deployment. Existing FAST and legacy fits are
unchanged; only BOUNDED accepts ``max_protrusion_m``.

RIM is an explicit CPU preprocessing strategy for plate outer contours. It uses
all source collision components together, removes duplicate vertices for PCA,
and requires two nondegenerate in-plane spans and a nondegenerate height band,
without a height-to-width cutoff. It builds one two-dimensional outer convex
contour; internal component seams and interior holes do not become separate rims.
Concavities and separated pieces may be bridged by this convex approximation.
The source mesh remains unchanged; the top-level ``used_convex_hull`` flag
stays false because no three-dimensional source-hull substitution occurs.

Source vertices within 1.5 mm of the projected outer contour define a rim-height
band in the PCA normal coordinate. Equal arc-length contour sections receive
one sphere each. A sphere contains its section's upper and lower corners and
assigned original boundary-vertex witnesses, with a 1 micrometre containment
reserve before outward radius rounding. Convexity then covers each intervening
piecewise-linear contour segment throughout that height band. The returned
float32/float64 spheres are rechecked after shared clipping and conversion;
lost sections, missing spheres and uncovered witnesses fail explicitly. RIM
never truncates or changes methods, and does not accept ``max_protrusion_m``.
With ``max_bottom_protrusion_m``, each contour prism is first intersected with
the original mesh's local-Z lower extent. PCA height extrusion can otherwise
introduce fictitious corners below the source bottom. The clipped prism still
contains every source rim witness; the source mesh is unchanged. Sphere centers
are then raised as needed to contain their assigned prism corners and boundary
vertices while obeying the bottom plane, with the same requested sphere count.
Final clipping and dtype conversion still require positive coverage margins.
A plane with no positive clearance for a required witness remains infeasible;
zero bottom protrusion does not permit a strict enclosure of bottom witnesses.
Arbitrary ``clip_plane`` constraints continue to use the shared final clipping.

``debug_info["rim"]`` uses schema ``curobo-rim-fit/1`` and mesh-local witness
coordinates. It reports ``section_corners_m`` per sphere,
``boundary_vertices_m``, positive ``containment_margin_m``, exact final count,
PCA extents, perimeter, rim heights, and final bottom protrusion. The flags
``contour_band_source_bottom_clipped`` and ``bottom_constrained_fit`` identify
the bottom-constrained geometry and fitting path. It also reports
source-unbacked intervals on the convex contour as
``convex_contour_source_gap_length_m`` and
``maximum_convex_contour_bridge_m``. These are interval lengths along the
convex boundary, not Hausdorff distances. Both ``full_volume_covered`` and
``whole_plate_volume_covered`` remain false. ``compute_metrics=True`` evaluates
the original complete mesh; its sampled face/volume metrics do not expand the
rim certificate into a whole-object coverage guarantee.

FAST is an offline preprocessing operation, not a faster GPU collision query.
It uses OpenCV for projected triangle rasterization, SciPy for optimization,
and CoACD in an isolated subprocess (30-second timeout) when escalation is
needed. Failed or timed-out decomposition is logged and recorded in
``result.debug_info["fast"]["partition"]``. Quality diagnostics under
``pre_clip_audit`` describe the fit before shared clipping and dtype conversion;
``compute_metrics=True`` separately evaluates the returned spheres.

For a compound mesh whose components are all closed and outward-oriented,
quality metrics use the union of those solids. Interior sampling and outward
distances query individual components; combined-mesh parity and internal
overlapping faces do not define the solid boundary. No hole repair is performed
for this classification. Surface gaps still sample the authored surfaces, and
``volume_ratio`` compares summed sphere volumes to the source's summed component
volume, which can count overlaps more than once.

The default changes for direct fitting, obstacle bounding spheres, robot
building/refitting, and attachment fitting. Explicit ``MORPHIT``, ``VOXEL``,
and ``SURFACE`` selections retain their algorithms. Existing saved sphere
configurations are not regenerated. MorphIt-specific weights and
``iterations`` continue to affect only MorphIt. All methods share the existing
``clip_plane`` postprocessing and requested output device/dtype.

.. figure:: ../images/sphere_approx.png
   :width: 690
   :align: center

   Comparison of the three fit types on robot link meshes. From left to right:
   MORPHIT (pink), VOXEL (green), SURFACE (blue).

To visually compare the fit types interactively, run the comparison demo:

.. code-block:: bash

   python -m curobo.examples.reference.sphere_fit_comparison

This launches a `Viser <https://viser.studio>`_ viewer showing each fit type as a column of
coloured spheres next to the original mesh. You can customise the robot, links, and methods:

.. code-block:: bash

   python -m curobo.examples.reference.sphere_fit_comparison --robot franka.yml
   python -m curobo.examples.reference.sphere_fit_comparison --links panda_link0 panda_link7
   python -m curobo.examples.reference.sphere_fit_comparison --methods surface voxel morphit

The fitting pipeline automatically handles degenerate cases:

- **Hollow/thin meshes**: Watertight meshes with extremely low fill ratio are replaced
  with their convex hull before fitting.
- **Fallback chain**: If the primary method produces no spheres, falls back to ``VOXEL``,
  then ``SURFACE``.


Result
-------

:class:`curobo.sphere_fit.SphereFitResult` contains:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Description
   * - ``centers``
     - Sphere centre positions, shape ``(N, 3)``.
   * - ``radii``
     - Sphere radii, shape ``(N,)``.
   * - ``num_spheres``
     - Number of fitted spheres.
   * - ``fit_time_s``
     - Wall-clock fitting time in seconds.

Quality Metrics
----------------

When ``compute_metrics=True``, the following fields are populated:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Metric
     - Description
   * - ``coverage``
     - Fraction of interior sample points covered by at least one sphere.
   * - ``protrusion``
     - Fraction of sphere-surface sample points outside the mesh.
   * - ``protrusion_dist_mean``
     - Mean distance (m) of protruding points to the mesh surface.
   * - ``protrusion_dist_p95``
     - 95th-percentile protrusion distance (m).
   * - ``surface_gap_mean``
     - Mean gap (m) from mesh surface samples to nearest sphere surface.
   * - ``surface_gap_p95``
     - 95th-percentile surface gap (m).
   * - ``max_uncovered_gap``
     - Maximum gap (m) from mesh surface to nearest sphere.
   * - ``volume_ratio``
     - Total sphere volume divided by mesh volume.

Inspecting Fit Quality
-----------------------

Pass ``--compute-metrics`` to the robot builder script to print a per-link
quality report after sphere fitting:

.. code-block:: bash

   python -m curobo.examples.getting_started.build_robot_model \
       --urdf robot.urdf --asset-path meshes/ --output robot.yml \
       --compute-metrics

The same metrics are available programmatically via
``builder.link_metrics`` (a dict of :class:`SphereFitMetrics`).


Clip Planes
------------

When a robot is mounted on a stand or bolted to the floor, base-link spheres
may protrude into the mounting surface.  Pass ``--clip-link`` to prevent this:

.. code-block:: bash

   python -m curobo.examples.getting_started.build_robot_model \
       --urdf robot.urdf --asset-path meshes/ --output robot.yml \
       --clip-link base_link z 0.0

This adds a half-plane constraint during MorphIt optimization (as a
differentiable loss term) and applies a hard clamp after fitting, so no sphere
on ``base_link`` extends below ``z=0`` in link-local coordinates.  The flag can
be repeated for multiple links.

The clamp also applies after conversion to the requested output dtype. If
nearest rounding would move a sphere across the plane, its radius uses the
adjacent representable value inside the boundary. No additional geometric
clearance is introduced. A minimum-radius floor cannot override the plane;
spheres whose centers round onto or behind it are removed. This is a local
fitting guarantee, not a tolerance for later collision queries or transforms.

Programmatically, pass ``clip_links`` to :meth:`RobotBuilder.fit_collision_spheres`:

.. code-block:: python

   builder.fit_collision_spheres(clip_links={"base_link": ("z", 0.0)})


Geometry Helper
----------------

Individual geometry objects also provide a convenience method for sphere fitting:

.. code-block:: python

   from curobo.geom.types import Capsule, WorldCfg

   capsule = Capsule(
      name="capsule",
      radius=0.2,
      base=[0, 0, 0],
      tip=[0, 0, 0.5],
      pose=[0.0, 5, 0.0, 0.043, -0.471, 0.284, 0.834],
   )

   sph = capsule.get_bounding_spheres(num_spheres=128)
   WorldCfg(spheres=sph).save_world_as_mesh("bounding_spheres.obj")

Object-local bottom bound
-------------------------

``fit_spheres_to_mesh(..., max_bottom_protrusion_m=0.002)`` optionally limits
every sphere's lowest local Z to the source mesh minimum Z minus 2 mm. The
default ``None`` preserves existing behavior. The bound must be finite and
nonnegative. FAST includes this plane in its candidate fitting and sphere-count
selection, then evaluates refinement candidates after applying the same bound.
This avoids selecting an apparently adequate fit whose coverage is lost only
when the bottom is clipped afterward. The plane remains in object coordinates
through the internal PCA transforms. Final radius clipping and invalid-sphere
removal remain as a precision safeguard, including after output dtype conversion.
Other fitters retain their existing post-fit clipping behavior. Quality metrics
describe the final bounded result.
This is not a coverage guarantee or a world-gravity constraint after rotation.
FAST returns an error for an empty fit; it never falls back to a legacy fitter.
