"""Compact Markdown tables for the consolidated follow-up report, rendered from a followup_report.py JSON.

    python -I followup_tables.py runs/setup/followup-report-<time>.json > tables.md
"""
import json
import sys

r = json.load(open(sys.argv[1]))
KINDS = (('train', 'training'), ('heldout', 'held-out'), ('traj', 'trajectory'))


def ci(x, f='{:+.3f}'):
    return f'[{f.format(x[0])}, {f.format(x[1])}]'


def g(m):
    return 'NA' if m['hedges_g'] is None else f"{m['hedges_g']:+.2f}"


def success_table(cells, arm, control):
    out = [f'| task | camera | {control} | {arm} | Δ last-5 [paired 95% CI] | g | historical rule | Δ AUC [CI] | Δ early AUC [CI] |',
           '|---|---|---|---|---|---|---|---|---|']
    for task, c in cells.items():
        for key, label in KINDS:
            m = c['metrics']
            l, a, e = m[f'{key}/success/last5'], m[f'{key}/success/auc'], m[f'{key}/success/early_auc']
            rule = c['margin_rule_last5_success'][key]
            out.append(f"| {task} | {label} | {l['control_mean']:.3f} | {l['arm_mean']:.3f} | {l['delta']:+.3f} "
                       f"{ci(l['paired_ci95'])} | {g(l)} | {rule['verdict']} ({rule['margin']:.2f}) | "
                       f"{a['delta']:+.3f} {ci(a['paired_ci95'])} | {e['delta']:+.3f} {ci(e['paired_ci95'])} |")
    return out


def efficiency_table(cells, arm, control):
    out = [f'| task | time to 0.5, {control} → {arm} (k steps; reach) | time to 0.8 (k steps; reach) | '
           f'Δ return last-5 [CI] | Δ return AUC [CI] |', '|---|---|---|---|---|']
    for task, c in cells.items():
        m = c['metrics']
        cells_ = []
        for t in ('0.5', '0.8'):
            x = m[f'time_to_{t}/steps']
            rr = c['reach_rate'][t]
            cells_.append(f"{x['control_mean'] / 1e3:.0f} → {x['arm_mean'] / 1e3:.0f} "
                          f"({rr['control']} → {rr['arm']}; Δ {x['delta'] / 1e3:+.0f} "
                          f"{ci([v / 1e3 for v in x['paired_ci95']], '{:+.0f}')})")
        rl, ra = m['train/return/last5'], m['train/return/auc']
        out.append(f"| {task} | {cells_[0]} | {cells_[1]} | {rl['delta']:+.0f} {ci(rl['paired_ci95'], '{:+.0f}')} | "
                   f"{ra['delta']:+.0f} {ci(ra['paired_ci95'], '{:+.0f}')} |")
    return out


def aggregate_table(agg, arm, control):
    out = [f'| metric | mean Δ over tasks [task-stratified 95% CI] | IQM {arm} [CI] | IQM {control} [CI] |',
           '|---|---|---|---|']
    keys = [f'{k}/success/{s}' for s in ('last5', 'auc', 'early_auc') for k, _ in KINDS] + \
        ['time_to_0.5/steps', 'time_to_0.8/steps', 'train/return/last5', 'train/return/auc']
    for key in keys:
        a = agg[key]
        if key.startswith('time_to'):
            out.append(f"| {key} (k steps) | {a['mean_delta'] / 1e3:+.0f} {ci([v / 1e3 for v in a['ci95']], '{:+.0f}')} | | |")
        elif '/return/' in key:
            out.append(f"| {key} | {a['mean_delta']:+.0f} {ci(a['ci95'], '{:+.0f}')} | | |")
        else:
            out.append(f"| {key} | {a['mean_delta']:+.3f} {ci(a['ci95'])} | {a['iqm_arm']:.3f} "
                       f"{ci(a['iqm_arm_ci95'], '{:.3f}')} | {a['iqm_control']:.3f} {ci(a['iqm_control_ci95'], '{:.3f}')} |")
    return out


def representation_lines(rep, yardsticks):
    rows = []
    for name in yardsticks:
        flagged = {k: v for k, v in rep['metrics'].items() if v[f'verdict_{name}'] != 'within margin'}
        shown = ', '.join(f"{k} {'+' if v['delta'] >= 0 else ''}{100 * v['delta'] if v['relative'] else v['delta']:.3g}"
                          f"{'%' if v['relative'] else ''} ({v[f'verdict_{name}']})" for k, v in flagged.items())
        c = rep['beyond_margin_counts'][name]
        rows.append(f"{name} margins: {c['better']} better / {c['worse']} worse of {len(rep['metrics'])}"
                    + (f" — {shown}" if shown else ''))
    return rows


L = []
if r['E1']['rl']:
    L += ['### E1 success', ''] + success_table(r['E1']['rl'], 'crop', 'full') + ['']
    L += ['### E1 learning efficiency and return (training cameras)', ''] + \
        efficiency_table(r['E1']['rl'], 'crop', 'full') + ['']
    for name, agg in r['E1']['aggregate'].items():
        L += [f'### E1 aggregate, {name.replace("_", " ")}', ''] + aggregate_table(agg, 'crop', 'full') + ['']
    L += ['### E1 representation (crop vs full, 200k full-split evaluation)', '']
    for task, rep in r['E1']['representation'].items():
        L.append(f"- **{task}**: " + '; '.join(representation_lines(rep, rep['yardsticks'])))
    L.append('')
if r['E2']['rl']:
    for task, cells in r['E2']['rl'].items():
        for name, c in cells.items():
            arm, control = name.split(' vs ')
            L += [f'### E2 {task}: {name}', ''] + success_table({task: c}, arm, control) + [''] + \
                efficiency_table({task: c}, arm, control) + ['']
    L += ['### E2 representation', '']
    for task, reps in r['E2']['representation'].items():
        for name, rep in reps.items():
            L.append(f"- **{task}, {name}**: " + '; '.join(representation_lines(rep, rep['yardsticks'])))
    L.append('')
if r['E3']['rl']:
    L += ['### E3 success', ''] + success_table(r['E3']['rl'], 'no shift', 'CNN') + ['']
    L += ['### E3 learning efficiency and return (training cameras)', ''] + \
        efficiency_table(r['E3']['rl'], 'no shift', 'CNN') + ['']
    L += ['### E3 viewpoint: Δ(novel) − Δ(training), raw success units', '',
          '| task | held-out last-5 | trajectory last-5 | held-out AUC | trajectory AUC |', '|---|---|---|---|---|']
    for task, v in r['E3']['viewpoint'].items():
        d = v['did']
        L.append(f"| {task} | " + ' | '.join(f"{d[k]['did']:+.3f} {ci(d[k]['paired_ci95'])}"
                                             for k in ('heldout/last5', 'traj/last5', 'heldout/auc', 'traj/auc')) + ' |')
    pooled = r['E3']['aggregate'].get('eight_tasks_did')
    if pooled:
        L.append('| **pooled (8 tasks)** | ' + ' | '.join(f"{pooled[k]['mean_did']:+.3f} {ci(pooled[k]['ci95'])}"
                                                      for k in ('heldout/last5', 'traj/last5', 'heldout/auc', 'traj/auc')) + ' |')
    L.append('')
    for name, agg in r['E3']['aggregate'].items():
        if name == 'eight_tasks':
            L += ['### E3 aggregate, eight tasks', ''] + aggregate_table(agg, 'no shift', 'CNN') + ['']
    na = {t: v['ratios']['heldout/last5']['na_runs'] for t, v in r['E3']['viewpoint'].items()
          if v['ratios']['heldout/last5']['na_runs']}
    L.append(f"Novel/training ratio NA (training last-5 success 0): {na or 'none'}")
print('\n'.join(L))
