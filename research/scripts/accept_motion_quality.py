"""Apply the frozen five-percent contract using unique requests and physical metrics."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np

from run_regression import ROOT, digest, write_json


def distribution(values: list[float]) -> dict:
    """Describe magnitudes without merging unlike physical units."""
    if not values:
        return dict(count=0, median=None, p95=None, maximum=None)
    return dict(count=len(values), median=float(np.median(values)),
                p95=float(np.percentile(values, 95)), maximum=float(max(values)))


def measurements(row: dict) -> dict[str, float]:
    """Map independently computed dense FK metrics into the runtime guard contract."""
    m = row['screening']
    result = dict(joint_path_rad=m['joint_path_rad'], joint_backtrack_rad=m['max_joint_backtrack_rad'],
                  tcp_path_sum_m=sum(t['tcp_path_m'] for t in m['tools'].values()))
    for tool, values in m['tools'].items():
        side = 'left' if tool.startswith('left') else 'right'
        result.update({f'{side}:tcp_path_m': values['tcp_path_m'],
            f'{side}:rotation_excess_rad': max(0, values['orientation_path_rad']-values['orientation_chord_rad']),
            f'{side}:elbow_overshoot_m': values['elbow_overshoot_m'],
            f'{side}:joint_path_by_arm_rad': values['joint_path_rad'],
            f'{side}:joint_backtrack_by_arm_rad': values['joint_backtrack_rad'],
            f'{side}:elbow_rise_m': values['elbow_rise_m'],
            f'{side}:tcp_max_line_deviation_m': values['tcp_max_line_deviation_m']})
    return result


def evaluate(runs: list[Path], policy: str, reference: str, rules: dict) -> dict:
    """Pair within conditions, then give each unique request one vote."""
    comparisons, raw = [], []
    for run in runs:
        study = json.loads((run/'inputs/study.json').read_text())
        cases = {c['id']: c for c in study['cases']}
        rows = json.loads((run/'analysis.json').read_text())['results']
        pairs = collections.defaultdict(dict)
        for row in rows:
            pairs[(row['case'], row['repeat'])][row['policy']] = row
        for (case, repeat), pair in pairs.items():
            if policy not in pair or reference not in pair:
                continue
            old, new = pair[reference], pair[policy]
            c = cases[case]
            identity = hashlib.sha256(json.dumps([c['group'],c['joint_names'],c['start'],
                c['tool_goal_poses']], sort_keys=True).encode()).hexdigest()
            row = dict(case=case, identity=identity, group=c['group'],
                condition=f'{run.name}:r{repeat}', old_success=old['success'], new_success=new['success'],
                old_status=old['status'], new_status=new['status'])
            if old['success'] and new['success']:
                a,b = measurements(old),measurements(new)
                differences, regressed, violations = {}, [], []
                for metric, value in a.items():
                    kind = metric.split(':')[-1]
                    delta = b[metric]-value
                    tolerance = rules['absolute_tolerances'].get(kind)
                    entry = dict(old=value, new=b[metric], delta=delta,
                        raw_relative_percent=100*delta/value if value>1e-8 else None)
                    if tolerance is not None:
                        entry.update(absolute_tolerance=tolerance,
                            relative_comparison_well_scaled=value >= tolerance/.05,
                            allowed_increase=max(.05*value,tolerance))
                        if delta > tolerance+1e-6:
                            regressed.append(metric)
                        if delta > max(.05*value,tolerance)+1e-6:
                            violations.append(metric)
                    differences[metric] = entry
                row.update(metrics=differences, regressed=regressed, amplitude_violations=violations)
            comparisons.append(row)
        raw += [r for r in rows if r['policy'] in [reference,policy]]
    grouped = collections.defaultdict(list)
    for row in comparisons:
        grouped[row['identity']].append(row)
    common = {key:rows for key,rows in grouped.items()
              if all(r['old_success'] and r['new_success'] for r in rows)}
    regressed = {k for k,rows in common.items() if any(r['regressed'] for r in rows)}
    violations = {k for k,rows in common.items() if any(r['amplitude_violations'] for r in rows)}
    lost = {k for k,rows in grouped.items() if any(r['old_success'] and not r['new_success'] for r in rows)}
    gained = {k for k,rows in grouped.items() if any(r['new_success'] and not r['old_success'] for r in rows)}
    failed = {k for k,rows in grouped.items() if any(not r['new_success'] and not r['old_success'] for r in rows)}
    magnitude, overall = {}, {}
    metrics = next(iter(common.values()))[0]['metrics']
    for metric in metrics:
        values = []
        for identity, rows in common.items():
            old = float(np.mean([r['metrics'][metric]['old'] for r in rows]))
            new = float(np.mean([r['metrics'][metric]['new'] for r in rows]))
            values.append((old,new))
        old,new = np.array(values).T
        overall[metric] = dict(old_mean=float(old.mean()), new_mean=float(new.mean()),
            old_median=float(np.median(old)), new_median=float(np.median(new)),
            mean_change_percent=float(100*(new.mean()/old.mean()-1)) if old.mean()>1e-8 else None)
        worst = [max((r['metrics'][metric] for r in rows), key=lambda v:v['delta']) for rows in common.values()]
        positive = [v for v in worst if v['delta']>v.get('absolute_tolerance',float('inf'))+1e-6]
        magnitude[metric] = dict(absolute_increase=distribution([v['delta'] for v in positive]),
            relative_increase_percent=distribution([v['raw_relative_percent'] for v in positive
                if v.get('relative_comparison_well_scaled')]),
            near_zero_absolute_increase=distribution([v['delta'] for v in positive
                if not v.get('relative_comparison_well_scaled')]),
            unique_requests_with_increase=len(positive))
    per_condition = {}
    for condition in sorted({r['condition'] for r in comparisons}):
        rows = [r for r in comparisons if r['condition']==condition]
        successful = [r for r in rows if 'metrics' in r]
        per_condition[condition] = dict(requests=len(rows), common_success=len(successful),
            regressed=sum(bool(r['regressed']) for r in successful),
            amplitude_violations=sum(bool(r['amplitude_violations']) for r in successful),
            old_success=sum(r['old_success'] for r in rows), new_success=sum(r['new_success'] for r in rows))
    latency = {}
    for method in [reference,policy]:
        # Candidate-trace/independent construction runs are diagnostic, not latency evidence.
        data = [r for r in json.loads((runs[0]/'requests.json').read_text()) if r['policy']==method]
        successful = [1000*r['elapsed_s'] for r in data if r['success']]
        latency[method] = dict(success_mean_ms=float(np.mean(successful)),
            success_p95_ms=float(np.percentile(successful,95)),
            all_mean_ms=float(np.mean([1000*r['elapsed_s'] for r in data])),
            status_counts=dict(collections.Counter(r['status'] for r in data)))
    reliability = dict(lost_success=[grouped[k][0]['case'] for k in sorted(lost)],
        gained_success=[grouped[k][0]['case'] for k in sorted(gained)],
        both_failed=[grouped[k][0]['case'] for k in sorted(failed)])
    rate = len(regressed)/len(common)
    overall_pass = (overall['tcp_path_sum_m']['new_mean'] <= overall['tcp_path_sum_m']['old_mean']+1e-7
        and overall['joint_path_rad']['new_mean'] <= overall['joint_path_rad']['old_mean']+1e-7
        and (overall['tcp_path_sum_m']['new_mean'] < overall['tcp_path_sum_m']['old_mean']-1e-7
             or overall['joint_path_rad']['new_mean'] < overall['joint_path_rad']['old_mean']-1e-7))
    validation = dict(regression_rate_pass=rate <= .05, amplitude_pass=not violations,
        overall_improvement_pass=overall_pass, success_preservation_pass=not lost)
    return dict(policy=policy,reference=reference, unique_requests=len(grouped),
        common_success_unique_requests=len(common), regressed_requests=len(regressed),
        regression_rate_percent=100*rate, amplitude_violation_requests=len(violations),
        regressed_case_ids=[grouped[k][0]['case'] for k in sorted(regressed)],
        amplitude_violation_case_ids=[grouped[k][0]['case'] for k in sorted(violations)],
        reliability=reliability, magnitude=magnitude, overall=overall, per_condition=per_condition,
        latency=latency, success_latency_increase_percent=100*(latency[policy]['success_mean_ms']/latency[reference]['success_mean_ms']-1),
        validation=validation, numerical_contract_pass=all(validation.values()),
        comparisons=comparisons, physical_execution=False, human_naturalness_acceptance='pending')


def main(args: argparse.Namespace) -> None:
    """Write one complete, reproducible contract report for all supplied conditions."""
    rules_path = ROOT/'research/fixtures/motion_quality_acceptance_v1.json'
    rules = json.loads(rules_path.read_text())
    result = evaluate(args.runs, args.policy, 'legacy', rules)
    result.update(rules=rules, rules_sha256=digest(rules_path))
    write_json(args.runs[0]/f'acceptance-{args.policy}.json', result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['comparisons','magnitude','rules']},
                      ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--policy', default='quality20')
    main(parser.parse_args())
