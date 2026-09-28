"""The hard bottom bound must participate in fitting, not erase coverage afterward."""
import numpy as np
import pytest
import torch
import trimesh
from scipy.spatial.distance import cdist

from curobo._src.geom.sphere_fit._fast_core import fit_spheres
from curobo.sphere_fit import SphereFitType, fit_spheres_to_mesh
from curobo.types import DeviceCfg


def test_thin_panel_bound_preserves_more_surface_than_post_fit_clipping():
    mesh = trimesh.creation.box([.3, .2, .012])
    mesh.apply_translation([.11, -.07, .43])
    low = mesh.bounds[0, 2] - .002
    kwargs = dict(num_spheres=32, fit_type=SphereFitType.FAST,
                  device_cfg=DeviceCfg(device='cpu', dtype=torch.float64))
    unbounded = fit_spheres_to_mesh(mesh, **kwargs)
    bounded = fit_spheres_to_mesh(mesh, max_bottom_protrusion_m=.002, **kwargs)
    centers = unbounded.centers.numpy()
    radii = np.minimum(unbounded.radii.numpy(), centers[:, 2] - low)
    valid = radii > 0
    xy = np.stack(
        np.meshgrid(np.linspace(-.15, .15, 31), np.linspace(-.1, .1, 29)), -1
    ).reshape(-1, 2)
    points = np.r_[np.c_[xy, np.full(len(xy), -.006)], np.c_[xy, np.full(len(xy), .006)]]
    points += [.11, -.07, .43]
    old_gaps = np.maximum((cdist(points, centers[valid]) - radii[valid]).min(axis=1), 0)
    new_gaps = np.maximum(
        (cdist(points, bounded.centers.numpy()) - bounded.radii.numpy()).min(axis=1), 0
    )
    # Require a millimetre-scale improvement, not float rounding in the shared
    # final clip. This fixture does not certify enclosure of arbitrary meshes.
    assert np.quantile(new_gaps, .95) < np.quantile(old_gaps, .95) - .001
    assert np.all(bounded.centers.numpy()[:, 2] - bounded.radii.numpy() >= low)
    assert 0 < bounded.num_spheres <= 32


@pytest.mark.parametrize('minimum_z', [float('nan'), float('inf'), 1.])
def test_core_rejects_invalid_or_mesh_cutting_bottom(minimum_z):
    mesh = trimesh.creation.box()
    with pytest.raises(ValueError, match='minimum_z'):
        fit_spheres(mesh.vertices, mesh.faces, minimum_z=minimum_z)
