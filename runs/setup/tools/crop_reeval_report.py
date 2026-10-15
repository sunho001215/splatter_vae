"""Crop follow-up step 1 report (user request 2026-10-15; report only, no decision): crop vs base at 100k, re-evaluated
with the current evaluator. M2D-era single-seed results. Each difference is compared with the later measured seed
margins: the Stage 2 reporting margins (fixed M3D seed 0/1 pair, all metrics) and the S1 margins (C seed 0/1 pair,
four metrics; docs/EXPERIMENT_LOG.md "S1 margins"). Readers are the margins file's own (reader, column) per metric.
Requires all four re-evaluations verified by the Stage 2 local driver.

    python -I crop_reeval_report.py
"""
import hashlib
import json
from datetime import datetime
from pathlib import Path

ROOT = Path('/home/ws/ws/splatter4d')
SETUP = ROOT / 'runs/setup'
MARGINS = SETUP / 'stage2-reporting-margins-20261013.json'
MARGINS_SHA = '5fadd04fd5050a440f13d5adad540e424a0f78257ed3f82eacec36df31db5677'
METRICS = ('retrieval_traj', 'retrieval_near', 'retrieval_train', 'r2_pos_train', 'r2_pos_near', 'r2_pos_traj',
           'r2_obj_pos_traj', 'r2_vel_train', 'r2_vel_near', 'r2_vel_traj', 'cd_render_traj')
# S1 margins (R = C seed 0/1, 100k, full split), recorded 2026-10-10 before any S1 variant was compared
S1 = {'hammer': {'retrieval_traj': 0.030, 'r2_pos_traj': 0.050, 'r2_vel_traj': 0.055, 'cd_render_traj': 0.147},
      'pick-place': {'retrieval_traj': 0.030, 'r2_pos_traj': 0.050, 'r2_vel_traj': 0.050, 'cd_render_traj': 0.197}}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def value(s, column, stride):
    return float(s[f'metric/{column}@s{stride}'])


def read(s, reader, column):
    if reader == 'mean':
        return 0.5 * (value(s, column, 2) + value(s, column, 6))
    if reader == 's6':
        return value(s, column, 6)
    p90 = {k: 0.5 * (value(s, f'cd_{column}_p2g_p90', k) + value(s, f'cd_{column}_g2p_p90', k)) for k in (2, 6)}
    return 0.5 * (p90[2] + p90[6]) if reader == 'sym_mean' else p90[6]


assert sha(MARGINS) == MARGINS_SHA
margins = json.loads(MARGINS.read_text())
report = {'recorded_local': datetime.now().astimezone().isoformat(),
          'label': 'M2D-era single-seed results (seed 0, base 200k schedule at 100k); report only',
          'stage2_margins': str(MARGINS.relative_to(ROOT)), 'stage2_margins_sha256': MARGINS_SHA, 'tasks': {}}
lines = []
for task in ('hammer', 'pick-place'):
    runs = {arm: f'creval-{arm}-{task}-100k' for arm in ('base', 'crop')}
    summaries = {}
    for arm, job in runs.items():
        proof = json.loads((SETUP / f's2-{job}-completion.json').read_text())
        path = ROOT / 'runs' / job / 'eval/step_0100000/summary.json'
        assert sha(path) == proof['summary_sha256'] and proof['checkpoint_unchanged'] is True
        summaries[arm] = json.loads(path.read_text())
    rows = {}
    for metric in METRICS:
        spec = margins['tasks'][task]['metrics'][metric]
        base, crop = (read(summaries[a], spec['reader'], spec['column']) for a in ('base', 'crop'))
        delta = crop / base - 1.0 if spec['relative'] else crop - base
        signed = delta if spec['higher_is_better'] else -delta
        s2_margin = spec['primary']['margin']
        s1_margin = S1[task].get(metric)
        rows[metric] = {'base': base, 'crop': crop, 'delta': delta, 'relative': spec['relative'],
                        'higher_is_better': spec['higher_is_better'], 'stage2_margin': s2_margin,
                        'exceeds_stage2_margin': abs(delta) > s2_margin, 's1_margin': s1_margin,
                        'exceeds_s1_margin': None if s1_margin is None else abs(delta) > s1_margin,
                        'direction': 'better' if signed > 0 else 'worse'}
    report['tasks'][task] = {'runs': runs, 'metrics': rows}
    lines.append(f'\n{task} (crop - base; CD-render as relative change; lower CD is better)')
    lines.append(f"{'metric':17s} {'base':>8s} {'crop':>8s} {'delta':>8s} {'S2 marg':>8s} {'S1 marg':>8s}  exceeds")
    for metric, r in rows.items():
        s1 = '-' if r['s1_margin'] is None else f"{r['s1_margin']:.3f}"
        flag = ('S2' if r['exceeds_stage2_margin'] else '') + ('+S1' if r['exceeds_s1_margin'] else '')
        unit = '%' if r['relative'] else ''
        delta = f"{100 * r['delta']:+.1f}{unit}" if r['relative'] else f"{r['delta']:+.3f}"
        lines.append(f"{metric:17s} {r['base']:8.3f} {r['crop']:8.3f} {delta:>8s} {r['stage2_margin']:8.3f} "
                     f"{s1:>8s}  {flag or 'within noise'} ({r['direction']})")
out = SETUP / 'crop-reeval-report-20261015.json'
out.write_text(json.dumps(report, indent=2) + '\n')
print('\n'.join(lines))
print('\nREPORT', out, sha(out))
