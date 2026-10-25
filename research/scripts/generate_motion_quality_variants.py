# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0
"""Freeze diverse empty-scene pose requests without consulting either planner."""
from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation
from yourdfpy import URDF

from run_regression import FIXTURES, ROBOT, digest, write_json


def generate(output: Path) -> None:
    """Generate 300 requests and separate, unused geometric witnesses."""
    if output.exists():
        raise FileExistsError(output)
    urdf = ROBOT / 'assets/urdf/g2_crsB_op3_e.urdf'
    study = json.loads((FIXTURES / 'study.json').read_text())
    model = URDF.load(str(urdf), load_meshes=False)
    joints = {j.get('name'): j for j in ET.parse(urdf).getroot().findall('joint')}
    config = yaml.safe_load((FIXTURES / 'robot-both.yaml').read_text())['robot_cfg']['kinematics']
    links, centers, radii = [], [], []
    for link, spheres in config['collision_spheres'].items():
        for sphere in spheres:
            links.append(link)
            centers.append(sphere['center'])
            radii.append(sphere['radius'])
    centers, radii = np.array(centers), np.array(radii)
    mask = np.triu(np.ones((len(radii), len(radii)), dtype=bool), 1)
    ignored = config['self_collision_ignore']
    for i, a in enumerate(links):
        for j, b in enumerate(links):
            if a == b or b in ignored.get(a, []) or a in ignored.get(b, []):
                mask[i, j] = False
    pair_i, pair_j = np.where(mask)

    def poses(names: list[str], q: np.ndarray, tools: list[str]) -> dict[str, np.ndarray]:
        model.update_cfg(study['defaults'])
        model.update_cfg(dict(zip(names, q)))
        return {t: model.get_transform(t, frame_from='mobile_base_link').copy() for t in tools}

    def clearance(names: list[str], q: np.ndarray) -> float:
        poses(names, q, [])
        transforms = {n: model.get_transform(n, frame_from='mobile_base_link') for n in set(links)}
        world = np.array([transforms[n][:3, :3] @ p + transforms[n][:3, 3]
                          for n, p in zip(links, centers)])
        gap = np.linalg.norm(world[pair_i] - world[pair_j], axis=1) - radii[pair_i] - radii[pair_j]
        return float(gap.min())

    cases, witnesses = [], {}
    attempts = {}
    rng = np.random.default_rng(20261025)
    for group in ['right', 'left', 'both']:
        kin = yaml.safe_load((FIXTURES / f'robot-{group}.yaml').read_text())['robot_cfg']['kinematics']
        names, tools = kin['cspace']['joint_names'], kin['tool_frames']
        default = np.array([study['defaults'][n] for n in names])
        low = np.array([float(joints[n].find('limit').get('lower')) for n in names]) + .02
        high = np.array([float(joints[n].find('limit').get('upper')) for n in names]) - .02
        for family, count in [('small', 40), ('restore', 30), ('rotate', 30)]:
            accepted, trials = 0, 0
            while accepted < count and trials < 20000:
                trials += 1
                amplitude = [.25, .5, .8, 1.1][accepted % 4]
                start = np.clip(default + rng.uniform(-amplitude, amplitude, len(names)), low, high)
                gap = clearance(names, start)
                if gap < .002:
                    continue
                initial = poses(names, start, tools)
                held = tools[(accepted // 2) % 2] if group == 'both' and accepted % 2 == 0 else None
                endpoint = None
                if family != 'rotate':
                    if family == 'small':
                        delta = rng.uniform(-1, 1, len(names)) * [.03, .06, .10, .14][accepted % 4]
                        if held:
                            side = 'l' if held.startswith('left') else 'r'
                            delta[[i for i, n in enumerate(names) if f'_arm_{side}_' in n]] = 0
                        endpoint = np.clip(start + delta, low, high)
                    else:
                        endpoint = default.copy()
                        held = None
                    # These easy controls have a collision-free nine-point joint interpolation.
                    # They remain pose requests; this witness is never passed to the planner.
                    gaps = [clearance(names, q) for q in np.linspace(start, endpoint, 9)]
                    if min(gaps) < .002:
                        continue
                    goals = poses(names, endpoint, tools)
                    angle = None
                else:
                    angle = [-150, -120, -90, -60, -30, 30, 60, 90, 120, 150][accepted % 10]
                    axis = np.eye(3)[(accepted // 10) % 3]
                    goals = {t: p.copy() for t, p in initial.items()}
                    for tool in tools:
                        if tool != held:
                            goals[tool][:3, :3] = goals[tool][:3, :3] @ Rotation.from_rotvec(
                                axis * np.deg2rad(angle)).as_matrix()
                key = f'variants_{group}_{family}_{accepted:03d}'
                if held:
                    assert np.allclose(goals[held], initial[held], atol=1e-12, rtol=0)
                stratum = family + ('_one_hand_held' if held else '')
                cases.append(dict(id=key, group=group, side='left' if group == 'left' else 'right',
                    label=f'{group} {family} {accepted}', stratum=stratum, joint_names=names,
                    tool_frames=tools, start=start.tolist(), primary_tool=tools[-1],
                    goal_pose_base=goals[tools[-1]].tolist(),
                    tool_goal_poses={t: g.tolist() for t, g in goals.items()},
                    start_clearance_m=gap, start_joint_margin_rad=float(np.minimum(start-low+.02, high-start+.02).min()),
                    held_tool=held, requested_rotation_degrees=angle,
                    goal_translation_m={t: float(np.linalg.norm(goals[t][:3, 3]-initial[t][:3, 3])) for t in tools},
                    geometric_joint_interpolation_precheck=endpoint is not None,
                    joint_goal_witness=None))
                witnesses[key] = endpoint.tolist() if endpoint is not None else None
                accepted += 1
            attempts[f'{group}_{family}'] = dict(accepted=accepted, trials=trials)
            if accepted != count:
                raise RuntimeError(f'Insufficient accepted cases: {group} {family}')
            print(f'GENERATED {group} {family} {accepted} trials={trials}', flush=True)
    assert len({json.dumps([c['group'], c['start'], c['tool_goal_poses']], sort_keys=True) for c in cases}) == 300
    write_json(output, cases)
    write_json(output.with_suffix('.provenance.json'), dict(seed=20261025, cases=300,
        attempts=attempts, source_hashes={str(p): digest(p) for p in
            [Path(__file__), urdf, FIXTURES/'study.json', *FIXTURES.glob('robot-*.yaml')]},
        native_sphere_count=len(radii), fixture_sha256=digest(output),
        selection='No planner run or output used to generate or filter requests',
        scope='empty world; native sphere geometry precheck; final native validity remains mandatory',
        witnesses_not_passed_to_pose_planner=witnesses,
        rotation_goals='arbitrary pose targets, no full reachability claim'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    generate(parser.parse_args().output.resolve())
