# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0
"""Compare all paired successes, keeping mixed outcomes and failed requests."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from run_regression import write_json


def flags(metrics: dict) -> list[str]:
    """Screen large excess motion; flags alone do not establish necessity."""
    out = []
    for tool, values in metrics['tools'].items():
        if values['tcp_path_m']-values['tcp_chord_m'] > .15:
            out.append(tool+':tcp_excess')
        if values['elbow_overshoot_m'] > .06:
            out.append(tool+':elbow_overshoot')
        if values['orientation_path_rad']-values['orientation_chord_rad'] > 1:
            out.append(tool+':rotation_excess')
    if metrics['max_joint_backtrack_rad'] > .3:
        out.append('joint_backtrack')
    return out


def classify(old: dict, new: dict) -> tuple[str, dict]:
    """Compare practical metric changes without hiding a worse other hand."""
    changes = {'joint_path_rad': new['joint_path_rad']-old['joint_path_rad']}
    improvements, regressions = [], []
    # These review thresholds are frozen in source before running the planners.
    for tool, values in old['tools'].items():
        changes[tool] = {k: new['tools'][tool][k]-values[k] for k in
            ['tcp_path_m', 'tcp_max_line_deviation_m', 'elbow_overshoot_m',
             'elbow_rise_m', 'orientation_path_rad', 'tcp_max_start_offset_m']}
        for metric, threshold in [('tcp_path_m', .03), ('elbow_overshoot_m', .03),
                                  ('orientation_path_rad', .3)]:
            delta = changes[tool][metric]
            if delta < -threshold:
                improvements.append(tool+':'+metric)
            if delta > threshold:
                regressions.append(tool+':'+metric)
    if changes['joint_path_rad'] < -.3:
        improvements.append('joint_path')
    if changes['joint_path_rad'] > .3:
        regressions.append('joint_path')
    category = ('mixed' if improvements and regressions else 'improved' if improvements else
                'regressed' if regressions else 'similar')
    return category, dict(changes=changes, improvements=improvements, regressions=regressions)


def main(run: Path) -> None:
    """Write aggregate and complete request-level comparisons for one run."""
    study = json.loads((run/'inputs/study.json').read_text())
    data = json.loads((run/'analysis.json').read_text())['results']
    cases = {c['id']: c for c in study['cases']}
    paired = collections.defaultdict(dict)
    for row in data:
        paired[(row['case'], row['repeat'])][row['policy']] = row
    comparisons = []
    for (case, repeat), pair in sorted(paired.items()):
        old, new = pair['legacy'], pair['quality20']
        row = dict(case=case, repeat=repeat, group=cases[case]['group'],
            stratum=cases[case].get('stratum', 'canonical'),
            legacy_success=old['success'], quality_success=new['success'],
            old_result=old['id'], new_result=new['id'])
        if old['success'] and new['success']:
            category, changes = classify(old['screening'], new['screening'])
            row.update(category=category, **changes, legacy_flags=flags(old['screening']),
                       quality_flags=flags(new['screening']),
                       legacy=old['screening'], quality=new['screening'])
        else:
            row['category'] = ('gained_success' if new['success'] else 'lost_success'
                               if old['success'] else 'both_no_solution')
        comparisons.append(row)
    latency = {}
    for policy in ['legacy', 'quality20']:
        rows = [r for r in data if r['policy'] == policy]
        latency[policy] = dict(status_counts=dict(collections.Counter(r['status'] for r in rows)))
        for label, selected in [('all', rows), ('success', [r for r in rows if r['success']])]:
            times = np.array([r['elapsed_s']*1000 for r in selected])
            latency[policy][label] = dict(count=len(times), mean_ms=float(times.mean()),
                p50_ms=float(np.percentile(times, 50)), p95_ms=float(np.percentile(times, 95)),
                max_ms=float(times.max()))
    counts = lambda rows: dict(collections.Counter(r['category'] for r in rows))
    affected = [r for r in comparisons if r.get('legacy_flags')]
    unique = {}
    for case in cases:
        rows = [r for r in comparisons if r['case'] == case]
        unique[case] = dict(categories=[r['category'] for r in rows],
            success_consistent=len({(r['legacy_success'], r['quality_success']) for r in rows}) == 1,
            quality_screen_flags=[r.get('quality_flags') for r in rows])
    successful = [r['screening'] for r in data if r['success']]
    out = dict(unique_cases=len(cases), generated_cases=sum(c.startswith('variants_') for c in cases),
        paired_requests=len(comparisons), planner_calls=len(data), physical_execution=False,
        human_review='pending', thresholds='Practical change: TCP 3cm, elbow overshoot 3cm, rotation .3rad, joints .3rad; heuristic, not naturalness acceptance',
        outcomes=counts(comparisons), baseline_screened_pairs=len(affected),
        baseline_screened_outcomes=counts(affected),
        by_group={g: counts([r for r in comparisons if r['group'] == g]) for g in ['right','left','both']},
        by_stratum={s: counts([r for r in comparisons if r['stratum'] == s])
                    for s in sorted({r['stratum'] for r in comparisons})},
        latency=latency, latency_increase_percent={k: 100*(latency['quality20'][k]['mean_ms']/latency['legacy'][k]['mean_ms']-1)
                                                 for k in ['all', 'success']},
        validation=json.loads((run/'end-validation.json').read_text())['input_and_code_unchanged'],
        maximum_position_error_m=max(t['endpoint_position_error_m'] for r in successful for t in r['tools'].values()),
        maximum_angle_error_rad=max(t['endpoint_angle_error_rad'] for r in successful for t in r['tools'].values()),
        maximum_joint_limit_violation_rad=max(r['joint_limit_violation_rad'] for r in successful),
        maximum_observed_velocity_rad_s=max(r['observed_max_velocity_rad_s'] for r in successful),
        unique_case_outcomes=unique, comparisons=comparisons)
    write_json(run/'comparison.json', out)
    print(json.dumps({k:v for k,v in out.items() if k not in ['comparisons','unique_case_outcomes']}, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    main(parser.parse_args().run.resolve())
