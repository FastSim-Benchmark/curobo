# SPDX-License-Identifier: Apache-2.0

"""Sphere addresses must not wrap at the signed-16-bit float offset boundary."""

import pytest
import torch

from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.types import JointState


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("count", [8191, 8192, 8193, 10525])
@pytest.mark.parametrize("jacobian", [False, True])
def test_all_sphere_positions_radii_and_gradients_above_8192(tmp_path, count, jacobian):
    urdf = tmp_path / "slider.urdf"
    urdf.write_text('''<robot name="slider"><link name="base"/><link name="tip"/>
<joint name="slide" type="prismatic"><parent link="base"/><child link="tip"/>
<axis xyz="1 0 0"/><limit lower="-1" upper="1" effort="10" velocity="1"/>
</joint></robot>''')
    balls = [{"center": [i * 1e-6, .01, -.02], "radius": .003 + i * 1e-7}
             for i in range(count)]
    config = KinematicsCfg.from_robot_yaml_file({"robot_cfg": {"kinematics": {
        "urdf_path": str(urdf), "base_link": "base", "tool_frames": ["tip"],
        "collision_link_names": ["tip"], "collision_spheres": {"tip": balls},
        "collision_sphere_buffer": 0., "self_collision_ignore": {"tip": []},
        "self_collision_buffer": {"tip": 0.},
        "cspace": {"joint_names": ["slide"], "default_joint_position": [0.],
                   "null_space_weight": [1.], "cspace_distance_weight": [1.],
                   "max_acceleration": [10.], "max_jerk": [100.]},
    }}}, ["tip"])
    model = Kinematics(config, compute_jacobian=jacobian)
    local = config.kinematics_config.link_spheres[0].clone()
    q = torch.tensor([[[.1], [.2], [.3]], [[-.1], [-.2], [-.3]]],
                     device="cuda", dtype=torch.float32, requires_grad=True)
    state = model.compute_kinematics(JointState.from_position(q, joint_names=["slide"]))
    expected = local.expand(2, 3, count, 4).clone()
    expected[..., 0] += q.detach()[..., 0, None]
    torch.testing.assert_close(state.robot_spheres, expected, atol=1e-7, rtol=1e-6)
    state.robot_spheres[..., -1, 0].sum().backward()
    torch.testing.assert_close(q.grad, torch.ones_like(q), atol=1e-6, rtol=1e-6)
