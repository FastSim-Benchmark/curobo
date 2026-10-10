"""Regression for conservative rejection inside a large surface-mesh AABB."""

import pytest
import torch
import warp as wp

from curobo._src.geom.collision.wp_collision_kernel import query_aabb_overlaps
from curobo._src.geom.data.data_mesh import MeshDataWarp
from curobo._src.geom.data.data_scene import SceneData
from curobo._src.geom.types import Cuboid, MeshDistanceMode
from curobo._src.types.device_cfg import DeviceCfg


@wp.kernel
def bounds_queries(data: MeshDataWarp, output: wp.array(dtype=wp.int32)):
    output[0] = wp.int32(query_aabb_overlaps(data, 0, 0, wp.vec3(0.0), wp.vec3(0.0), 0.01))
    output[1] = wp.int32(
        query_aabb_overlaps(data, 0, 0, wp.vec3(-6.0, 0.0, 0.0), wp.vec3(6.0, 0.0, 0.0), 0.01)
    )
    output[2] = wp.int32(
        query_aabb_overlaps(data, 0, 0, wp.vec3(4.99, 0.0, 0.0), wp.vec3(4.99, 0.0, 0.0), 0.02)
    )
    output[3] = wp.int32(
        query_aabb_overlaps(data, 0, 0, wp.vec3(6.0, 0.0, 0.0), wp.vec3(6.0, 0.0, 0.0), 0.01)
    )
    output[4] = wp.int32(
        query_aabb_overlaps(data, 0, 0, wp.vec3(5.01, 0.0, 0.0), wp.vec3(5.01, 0.0, 0.0), 0.01)
    )


@pytest.mark.parametrize(
    "mode,inside", [(MeshDistanceMode.SURFACE, 0), (MeshDistanceMode.SOLID, 1)]
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_surface_interior_is_clear_but_solid_interior_is_retained(mode, inside, device):
    cfg = DeviceCfg(device=device)
    scene = SceneData.create_cache(1, cfg, mesh_cache=1)
    mesh = Cuboid(name="room", dims=[10, 10, 10], pose=[0, 0, 0, 1, 0, 0, 0]).get_mesh()
    mesh.distance_mode = mode
    scene.meshes.load_batch([mesh], 0)
    output = torch.empty(5, device=device, dtype=torch.int32)
    wp.launch(
        bounds_queries,
        dim=1,
        inputs=[scene.meshes.to_warp(), wp.from_torch(output)],
        device=device,
    )
    assert output.tolist() == [inside, 1, 1, 0, 1]


def test_graph_replay_uses_replaced_surface_geometry():
    cfg = DeviceCfg(device="cuda:0")
    scene = SceneData.create_cache(1, cfg, mesh_cache=1)

    def box(size):
        mesh = Cuboid(name="room", dims=[size] * 3, pose=[0, 0, 0, 1, 0, 0, 0]).get_mesh()
        mesh.distance_mode = MeshDistanceMode.SURFACE
        return mesh

    scene.meshes.load_batch([box(10)], 0)
    output = torch.empty(5, device="cuda:0", dtype=torch.int32)

    def query():
        wp.launch(
            bounds_queries,
            dim=1,
            inputs=[scene.meshes.to_warp(), wp.from_torch(output)],
            device="cuda:0",
            stream=wp.stream_from_torch(torch.cuda.current_stream()),
        )

    query()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        query()
    for size, expected in [(0.01, 1), (10, 0)]:
        scene.meshes.replace(box(size))
        graph.replay()
        assert output[0].item() == expected
