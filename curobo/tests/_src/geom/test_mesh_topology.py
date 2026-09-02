# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for collision-mesh topology admission."""

# Third Party
import numpy as np
import pytest
import trimesh

# CuRobo
from curobo._src.geom.data.data_mesh import _validate_solid_mesh


def _box() -> trimesh.Trimesh:
    return trimesh.creation.box(extents=[1.0, 1.0, 1.0])


def test_valid_solid_mesh_is_accepted():
    """A closed outward box defines a reliable signed distance."""
    mesh = _box()

    _validate_solid_mesh("box", mesh.vertices, mesh.faces)


def test_open_mesh_declared_solid_is_rejected():
    """A boundary edge makes solid inside/outside semantics undefined."""
    mesh = _box()
    faces = mesh.faces[:-1]

    with pytest.raises(ValueError, match="not a watertight two-manifold"):
        _validate_solid_mesh("open_box", mesh.vertices, faces)


def test_non_manifold_mesh_declared_solid_is_rejected():
    """More than two faces sharing one edge is not a solid two-manifold."""
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
        ]
    )
    faces = np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4]])

    with pytest.raises(ValueError, match="not a watertight two-manifold"):
        _validate_solid_mesh("non_manifold", vertices, faces)


def test_non_manifold_solid_vertex_is_rejected():
    """Closed shells that only meet at one vertex do not form a manifold solid."""
    first = trimesh.creation.icosphere(subdivisions=0, radius=1.0)
    second = first.copy()
    shared_point = first.vertices[0]
    second.apply_translation(shared_point - second.vertices[0])
    vertices = np.vstack([first.vertices, second.vertices[1:]])
    second_faces = second.faces.copy()
    second_faces[second_faces == 0] = -1
    second_faces[second_faces > 0] += len(first.vertices) - 1
    second_faces[second_faces == -1] = 0
    faces = np.vstack([first.faces, second_faces])

    with pytest.raises(ValueError, match="non-manifold vertex"):
        _validate_solid_mesh("touching_shells", vertices, faces)


def test_inward_solid_mesh_is_rejected():
    """A closed mesh with inward winding has the opposite signed-distance sign."""
    mesh = _box()

    with pytest.raises(ValueError, match="outward positive volume"):
        _validate_solid_mesh("inside_out", mesh.vertices, mesh.faces[:, ::-1])


def test_self_intersecting_solid_mesh_is_rejected():
    """Intersecting closed components do not define one unambiguous solid boundary."""
    first = _box()
    second = _box().copy()
    second.apply_translation([0.5, 0.0, 0.0])
    mesh = trimesh.util.concatenate([first, second])

    with pytest.raises(ValueError, match="self-intersects"):
        _validate_solid_mesh("intersecting_boxes", mesh.vertices, mesh.faces)


def test_disjoint_solid_components_are_accepted():
    """Multiple closed components are valid when their surfaces do not intersect."""
    first = _box()
    second = _box().copy()
    second.apply_translation([2.0, 0.0, 0.0])
    mesh = trimesh.util.concatenate([first, second])

    _validate_solid_mesh("disjoint_boxes", mesh.vertices, mesh.faces)


def test_degenerate_solid_triangle_is_rejected():
    """Zero-area faces cannot participate in a solid boundary."""
    mesh = _box()
    faces = np.vstack([mesh.faces, [0, 0, 1]])

    with pytest.raises(ValueError, match="degenerate triangles"):
        _validate_solid_mesh("degenerate", mesh.vertices, faces)
