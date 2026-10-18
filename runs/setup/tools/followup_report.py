"""Follow-up report: E1 pretraining crop on eight tasks, E2 pretraining random shift (dev tasks), E3 DrM CNN without
RL random shift. Analysis plan registered in runs/setup/followup-registration-20261017.json (analysis_plan); the
implementation details are logged in docs/EXPERIMENT_LOG.md (2026-10-18 22:20). Attribution only: no method decision.

A cell (experiment, comparison, task) is reported once all three seeds of both arms count (trainer and evaluator done;
Stage 2 driver runs also need their closeout proof with a matching eval.jsonl hash); other cells are listed as pending.
Outputs: runs/setup/followup-report-<time>.json, a Markdown table digest next to it, and SVG + PNG learning curves under
docs/figures/followup/.

    python -I followup_report.py
"""
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

ROOT = Path('/home/ws/ws/splatter4d')
sys.path.insert(0, str(ROOT))
from scripts import jobs  # noqa: E402
sys.path.insert(0, str(ROOT / "runs/setup/tools"))
from report_figures import png_curves, svg_curves  # noqa: E402

RUNS, SETUP = ROOT / 'runs', ROOT / 'runs/setup'
FIGURES = ROOT / 'docs/figures/followup'
REGISTRATION = SETUP / 'followup-registration-20261017.json'
MARGINS = SETUP / 'stage2-reporting-margins-20261013.json'
MARGINS_SHA = '5fadd04fd5050a440f13d5adad540e424a0f78257ed3f82eacec36df31db5677'
STEPS = list(range(10000, 1000001, 10000))
PATHS = {'train': [f'train{i}' for i in range(6)], 'heldout': [f'eval{i}' for i in range(4)],
         'traj': ['traj_lateral', 'traj_circular']}
TASKS = ('bin-picking', 'coffee-push', 'door-open', 'hammer', 'peg-insert-side', 'peg-unplug-side', 'pick-place',
         'stick-push')
DEV = ('hammer', 'pick-place')
NONDEV = tuple(t for t in TASKS if t not in DEV)
STATS = ('last5', 'final', 'auc', 'early_auc')
THRESHOLDS = (0.5, 0.8)
B = 10000

registration = json.loads(REGISTRATION.read_text())
E3_CONTROLS = registration['analysis_plan']['reuse']['E3_controls']
STATE = jobs.read_registry()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def eval_path(rid):
    path = RUNS / f'{rid}-eval' / 'eval.jsonl'
    return path if path.is_file() else RUNS / rid / 'eval.jsonl'


def status(job):
    return STATE.get(job, {}).get('status')


def ready(rid):
    jobs = [rid] + ([f'{rid}-eval'] if f'{rid}-eval' in STATE or (RUNS / f'{rid}-eval').is_dir() else [])
    if not all(status(j) == 'done' for j in jobs):
        return False
    if rid.startswith('s2rl-'):
        proof = SETUP / f's2-{rid}-completion.json'
        if not proof.is_file():
            return False
        assert json.loads(proof.read_text())['eval_jsonl_sha256'] == sha(eval_path(rid)), rid
    synced = STATE.get(jobs[-1], {}).get('sync', {}).get('files', {}).get('eval.jsonl')
    assert synced in (None, sha(eval_path(rid))), rid
    return True


RUN_CACHE = {}


def run(rid):
    if rid in RUN_CACHE:
        return RUN_CACHE[rid]
    rows = [json.loads(line) for line in eval_path(rid).read_text().splitlines() if line.strip()]
    assert [r['step'] for r in rows] == STEPS, rid
    assert {r['episodes'] for r in rows} == {240}, rid
    final = json.loads((RUNS / rid / 'final.json').read_text())
    assert final['completed'] is True and final['steps'] == 1_000_000, rid
    prov = json.loads((RUNS / rid / 'provenance.json').read_text())
    out = {'eval_jsonl_sha256': sha(eval_path(rid)), 'host': prov['host'], 'git_commit': prov['git_commit'],
           'curves': {}, 'scores': {}}
    for kind, paths in PATHS.items():
        for measure in ('success', 'return'):
            series = np.array([np.mean([r[f'{p}_{measure}'] for p in paths]) for r in rows])
            out['scores'][f'{kind}/{measure}'] = {'last5': float(series[-5:].mean()), 'final': float(series[-1]),
                                                  'auc': float(series.mean()), 'early_auc': float(series[:30].mean())}
            if measure == 'success':
                out['curves'][kind] = series
    stored = np.array([r['train_cameras_success'] for r in rows])
    assert np.allclose(stored, out['curves']['train']), rid
    for threshold in THRESHOLDS:
        hit = np.nonzero(out['curves']['train'] >= threshold - 1e-9)[0]
        out['scores'][f'time_to_{threshold}'] = {'steps': float(STEPS[hit[0]] if hit.size else 1_000_000),
                                                 'reached': bool(hit.size)}
    RUN_CACHE[rid] = out
    return out


def flat(tree, prefix=''):
    out = {}
    for key, value in tree.items():
        if isinstance(value, dict):
            out.update(flat(value, f'{prefix}{key}.'))
        else:
            out[prefix + key] = value
    return out


def config_diff(a, b):
    ca, cb = (flat(yaml.safe_load((RUNS / r / 'config.yaml').read_text())) for r in (a, b))
    return sorted(k for k in set(ca) | set(cb) if ca.get(k) != cb.get(k) and not k.startswith('run.'))


def hedges_g(a, b):
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    if pooled == 0:
        return None
    return float((np.mean(a) - np.mean(b)) / pooled * (1 - 3 / (4 * (na + nb) - 9)))


def pct(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def difference(a, b):
    """a, b: per-seed values matched by seed. Paired and unpaired percentile bootstrap of the difference of means."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(0)
    ia, ib = rng.integers(0, len(a), (B, len(a))), rng.integers(0, len(b), (B, len(b)))
    return {'arm_mean': float(a.mean()), 'control_mean': float(b.mean()), 'delta': float(a.mean() - b.mean()),
            'arm_seeds': a.tolist(), 'control_seeds': b.tolist(),
            'paired_ci95': pct(a[ia].mean(1) - b[ia].mean(1)), 'unpaired_ci95': pct(a[ia].mean(1) - b[ib].mean(1)),
            'hedges_g': hedges_g(a, b)}


def margin_rule(a, b):
    """Historical Stage 2 RL rule: seed-mean last-5 success, margin max(0.10, 2 x SE of the difference)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    se = float(np.sqrt(np.var(a, ddof=1) / len(a) + np.var(b, ddof=1) / len(b)))
    margin, delta = max(0.10, 2 * se), float(a.mean() - b.mean())
    return {'delta': delta, 'se_difference': se, 'margin': margin,
            'verdict': 'better' if delta > margin else 'worse' if delta < -margin else 'within margin'}


def metric_values(rids, metric, stat):
    if metric.startswith('time_to_'):
        return [run(r)['scores'][metric]['steps'] for r in rids]
    return [run(r)['scores'][metric][stat] for r in rids]


METRICS = [(f'{k}/{m}', s) for k in PATHS for m in ('success', 'return') for s in STATS] + \
    [(f'time_to_{t}', 'steps') for t in THRESHOLDS]


def compare_cell(arm_ids, control_ids):
    cell = {'arm_runs': arm_ids, 'control_runs': control_ids, 'metrics': {}, 'margin_rule_last5_success': {},
            'reach_rate': {}, 'hosts': {r: run(r)['host'] for r in arm_ids + control_ids},
            'config_differences': {a: config_diff(a, c) for a, c in zip(arm_ids, control_ids)}}
    for metric, stat in METRICS:
        cell['metrics'][f'{metric}/{stat}'] = difference(metric_values(arm_ids, metric, stat),
                                                         metric_values(control_ids, metric, stat))
    for kind in PATHS:
        cell['margin_rule_last5_success'][kind] = margin_rule(metric_values(arm_ids, f'{kind}/success', 'last5'),
                                                              metric_values(control_ids, f'{kind}/success', 'last5'))
    for t in THRESHOLDS:
        cell['reach_rate'][str(t)] = {
            'arm': f"{sum(run(r)['scores'][f'time_to_{t}']['reached'] for r in arm_ids)}/{len(arm_ids)}",
            'control': f"{sum(run(r)['scores'][f'time_to_{t}']['reached'] for r in control_ids)}/{len(control_ids)}"}
    return cell


def aggregate(cells, tasks):
    """Mean over tasks of per-task differences; task-stratified bootstrap (seeds resampled within task and arm)."""
    out = {}
    for metric, stat in METRICS:
        key = f'{metric}/{stat}'
        a = np.array([cells[t]['metrics'][key]['arm_seeds'] for t in tasks])
        b = np.array([cells[t]['metrics'][key]['control_seeds'] for t in tasks])
        rng = np.random.default_rng(0)
        ia = rng.integers(0, a.shape[1], (B, *a.shape))
        ib = rng.integers(0, b.shape[1], (B, *b.shape))
        boots = (np.take_along_axis(np.broadcast_to(a, (B, *a.shape)), ia, 2).mean(2)
                 - np.take_along_axis(np.broadcast_to(b, (B, *b.shape)), ib, 2).mean(2)).mean(1)
        out[key] = {'mean_delta': float((a.mean(1) - b.mean(1)).mean()), 'ci95': pct(boots)}
        if metric.endswith('/success'):
            out[key]['iqm_arm'], out[key]['iqm_arm_ci95'] = iqm_with_ci(a)
            out[key]['iqm_control'], out[key]['iqm_control_ci95'] = iqm_with_ci(b)
    return out


def iqm_with_ci(x):
    """As the final comparison: IQM over tasks x seeds, stratified bootstrap (seeds resampled within tasks)."""
    def iqm(v):
        v = np.sort(v, axis=-1)
        k = int(0.25 * v.shape[-1])
        return v[..., k:v.shape[-1] - k].mean(axis=-1)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, x.shape[1], size=(B, *x.shape))
    boots = iqm(np.take_along_axis(np.broadcast_to(x, (B, *x.shape)), idx, axis=2).reshape(B, -1))
    return float(iqm(x.ravel())), pct(boots)


def viewpoint(arm_ids, control_ids):
    """E3: novel/training success ratio per run and difference-in-differences (delta novel - delta training)."""
    out = {'ratios': {}, 'did': {}}
    for stat in ('last5', 'final', 'auc'):
        for novel in ('heldout', 'traj'):
            ratios, na_runs = {}, []
            for rid in arm_ids + control_ids:
                tr = run(rid)['scores']['train/success'][stat]
                nv = run(rid)['scores'][f'{novel}/success'][stat]
                ratios[rid] = None if tr == 0 else nv / tr
                if tr == 0:
                    na_runs.append(rid)
            out['ratios'][f'{novel}/{stat}'] = {'per_run': ratios, 'na_runs': na_runs}
            a = np.array([run(r)['scores'][f'{novel}/success'][stat] - run(r)['scores']['train/success'][stat]
                          for r in arm_ids])
            b = np.array([run(r)['scores'][f'{novel}/success'][stat] - run(r)['scores']['train/success'][stat]
                          for r in control_ids])
            d = difference(a, b)
            out['did'][f'{novel}/{stat}'] = {'did': d['delta'], 'paired_ci95': d['paired_ci95'],
                                             'arm_gap_seeds': a.tolist(), 'control_gap_seeds': b.tolist()}
    return out


def pooled_did(views, cells, tasks):
    out = {}
    for key in views[tasks[0]]['did']:
        x = np.array([np.array(views[t]['did'][key]['arm_gap_seeds']) - np.array(views[t]['did'][key]['control_gap_seeds'])
                      for t in tasks])
        idx = np.random.default_rng(0).integers(0, x.shape[1], (B, *x.shape))
        boots = np.take_along_axis(np.broadcast_to(x, (B, *x.shape)), idx, 2).mean(2).mean(1)
        out[key] = {'mean_did': float(x.mean()), 'ci95': pct(boots)}
    return out


# representation (E1, E2)
def summary(job):
    return json.loads((RUNS / job / 'eval/step_0200000/summary.json').read_text())


def read_metric(s, reader, column):
    value = lambda c, k: float(s[f'metric/{c}@s{k}'])  # noqa: E731
    if reader == 'mean':
        return 0.5 * (value(column, 2) + value(column, 6))
    if reader == 's6':
        return value(column, 6)
    p90 = {k: 0.5 * (value(f'cd_{column}_p2g_p90', k) + value(f'cd_{column}_g2p_p90', k)) for k in (2, 6)}
    return 0.5 * (p90[2] + p90[6]) if reader == 'sym_mean' else p90[6]


def representation(variant_eval, reference_eval, yardsticks):
    # the variant needs its Stage 2 closeout proof; the full-method references predate the Stage 2 driver
    if not (SETUP / f's2-{variant_eval}-completion.json').is_file() or status(reference_eval) != 'done':
        return None
    sv, sr = summary(variant_eval), summary(reference_eval)
    rows = {}
    for metric, spec in margins['tasks'][DEV[0]]['metrics'].items():
        v, r = read_metric(sv, spec['reader'], spec['column']), read_metric(sr, spec['reader'], spec['column'])
        delta = v / r - 1.0 if spec['relative'] else v - r
        signed = delta if spec['higher_is_better'] else -delta
        row = {'variant': v, 'reference': r, 'delta': delta, 'relative': spec['relative'],
               'higher_is_better': spec['higher_is_better']}
        for name in yardsticks:
            m = margins['tasks'][name]['metrics'][metric]['primary']['margin']
            row[f'margin_{name}'] = m
            row[f'verdict_{name}'] = 'better' if signed > m else 'worse' if signed < -m else 'within margin'
        rows[metric] = row
    counts = {name: {v: sum(r[f'verdict_{name}'] == v for r in rows.values()) for v in ('better', 'worse')}
              for name in yardsticks}
    return {'variant_eval': variant_eval, 'reference_eval': reference_eval, 'yardsticks': list(yardsticks),
            'beyond_margin_counts': counts, 'metrics': rows}


def curve_panels(groups, tasks):
    """groups: {arm: {task: [run ids]}} -> panels per task and camera condition plus a task-mean row."""
    panels = []
    for task in tasks + ('mean over tasks',):
        for kind in PATHS:
            series = {}
            for arm, by_task in groups.items():
                if task == 'mean over tasks':
                    if all(t in by_task for t in tasks):
                        series[arm] = np.mean([[run(r)['curves'][kind] for r in by_task[t]] for t in tasks], axis=0)
                elif task in by_task:
                    series[arm] = np.array([run(r)['curves'][kind] for r in by_task[task]])
            panels.append((f'{task} / {kind}', series))
    return panels


assert sha(MARGINS) == MARGINS_SHA
margins = json.loads(MARGINS.read_text())
report = {'recorded_local': datetime.now().astimezone().isoformat(), 'registration': str(REGISTRATION.relative_to(ROOT)),
          'registration_sha256': sha(REGISTRATION), 'analysis_plan': registration['analysis_plan'],
          'E1': {'rl': {}, 'representation': {}, 'aggregate': {}}, 'E2': {'rl': {}, 'representation': {}},
          'E3': {'rl': {}, 'viewpoint': {}, 'aggregate': {}}, 'pending': []}
expected_diff = {'E1': ['vision.export_path'], 'E2': ['vision.export_path'], 'E3': ['agent.random_shift']}


def cell(experiment, name, task, arm_ids, control_ids):
    missing = [r for r in arm_ids + control_ids if not ready(r)]
    if missing:
        report['pending'].append({'experiment': experiment, 'comparison': name, 'task': task, 'waiting_for': missing})
        return None
    c = compare_cell(arm_ids, control_ids)
    c['unexpected_config_differences'] = {k: [d for d in v if d not in expected_diff[experiment]]
                                          for k, v in c['config_differences'].items()
                                          if any(d not in expected_diff[experiment] for d in v)}
    return c


S_FINAL, S_DEV = (2000, 2001, 2002), (1000, 1001, 1002)
curves = {'E1': {}, 'E2': {}, 'E3': {}}
# E1: crop vs full, eight tasks, seeds 2000-2002
for task in TASKS:
    arm = [f's2rl-crop-{task}-s{s}' for s in S_FINAL]
    ctl = [f's3-drm-splatter4d-{task}-s{s}' for s in S_FINAL]
    c = cell('E1', 'crop vs full', task, arm, ctl)
    if c:
        report['E1']['rl'][task] = c
        curves['E1'].setdefault('crop', {})[task] = arm
        curves['E1'].setdefault('full', {})[task] = ctl
    reference = f'fulleval-ls-a200-m3d-seed0-{task}-200k' if task in DEV else f'fulleval-s3-m3d-seed0-{task}-200k'
    rep = representation(f'fulleval-s2ab-crop-seed0-{task}-200k', reference, (task,) if task in DEV else DEV)
    if rep:
        report['E1']['representation'][task] = rep
for name, tasks in (('eight_tasks', TASKS), ('six_nondev_tasks', NONDEV)):
    if all(t in report['E1']['rl'] for t in tasks):
        report['E1']['aggregate'][name] = aggregate(report['E1']['rl'], tasks)
# E2: shift vs full and vs crop (and crop vs full, already reported), dev tasks, seeds 1000-1002
for task in DEV:
    ids = {a: [f's2rl-{a}-{task}-s{s}' for s in S_DEV] for a in ('full', 'crop', 'shift')}
    for arm, ctl in (('shift', 'full'), ('shift', 'crop'), ('crop', 'full')):
        c = cell('E2', f'{arm} vs {ctl}', task, ids[arm], ids[ctl])
        if c:
            report['E2']['rl'].setdefault(task, {})[f'{arm} vs {ctl}'] = c
            for a in (arm, ctl):
                curves['E2'].setdefault(a, {})[task] = ids[a]
    for ctl, ref in (('full', f'fulleval-ls-a200-m3d-seed0-{task}-200k'),
                     ('crop', f'fulleval-s2ab-crop-seed0-{task}-200k')):
        rep = representation(f'fulleval-s2ab-shift-seed0-{task}-200k', ref, (task,))
        if rep:
            report['E2']['representation'].setdefault(task, {})[f'shift vs {ctl}'] = rep
# E3: CNN without RL shift vs CNN, eight tasks, seeds 2000-2002
for task in TASKS:
    arm = [f's4-drm-cnnnoshift-{task}-s{s}' for s in S_FINAL]
    ctl = [E3_CONTROLS[a] for a in arm]
    c = cell('E3', 'cnnnoshift vs cnn', task, arm, ctl)
    if c:
        report['E3']['rl'][task] = c
        report['E3']['viewpoint'][task] = viewpoint(arm, ctl)
        curves['E3'].setdefault('cnnnoshift', {})[task] = arm
        curves['E3'].setdefault('cnn', {})[task] = ctl
if all(t in report['E3']['rl'] for t in TASKS):
    report['E3']['aggregate']['eight_tasks'] = aggregate(report['E3']['rl'], TASKS)
    report['E3']['aggregate']['eight_tasks_did'] = pooled_did(report['E3']['viewpoint'], report['E3']['rl'], TASKS)

FIGURES.mkdir(parents=True, exist_ok=True)
for exp, arms, tasks, title in (('E1', ('full', 'crop'), TASKS, 'E1 pretraining crop vs full (success, seeds 2000-2002)'),
                                ('E2', ('full', 'crop', 'shift'), DEV, 'E2 pretraining shift / crop / full (seeds 1000-1002)'),
                                ('E3', ('cnn', 'cnnnoshift'), TASKS, 'E3 DrM CNN with / without RL random shift (2000-2002)')):
    done = tuple(t for t in tasks if all(t in curves[exp].get(a, {}) for a in arms))
    if done:
        groups = {a: curves[exp][a] for a in arms}
        label, panels = title + ('' if done == tasks else ' [partial]'), curve_panels(groups, done)
        svg_curves(FIGURES / f'{exp.lower()}_success_curves.svg', label, panels, arms)
        png_curves(FIGURES / f'{exp.lower()}_success_curves.png', label, panels, arms)

stamp = f'{datetime.now():%Y%m%d-%H%M%S}'
out = SETUP / f'followup-report-{stamp}.json'
out.write_text(json.dumps(report, indent=1, default=str) + '\n')

# Markdown digest
L = [f'# Follow-up report digest ({report["recorded_local"]})', '', f'JSON: `{out.relative_to(ROOT)}`', '']


def ci(x, f='{:+.3f}'):
    return f'[{f.format(x[0])}, {f.format(x[1])}]'


def rl_table(title, cells):
    L.extend([f'## {title}', '', '| task | camera | stat | arm | control | delta | paired 95% CI | g | rule (last-5) |',
              '|---|---|---|---|---|---|---|---|---|'])
    for task, c in cells.items():
        for kind in PATHS:
            for stat in ('last5', 'final', 'auc', 'early_auc'):
                m = c['metrics'][f'{kind}/success/{stat}']
                rule = c['margin_rule_last5_success'][kind]
                g = 'NA' if m['hedges_g'] is None else f"{m['hedges_g']:+.2f}"
                L.append(f"| {task} | {kind} | {stat} | {m['arm_mean']:.3f} | {m['control_mean']:.3f} | "
                         f"{m['delta']:+.3f} | {ci(m['paired_ci95'])} | {g} | "
                         f"{rule['verdict'] + ' (' + format(rule['margin'], '.3f') + ')' if stat == 'last5' else ''} |")
        for t in THRESHOLDS:
            m = c['metrics'][f'time_to_{t}/steps']
            L.append(f"| {task} | train | time to {t} (k steps) | {m['arm_mean'] / 1e3:.0f} | {m['control_mean'] / 1e3:.0f} | "
                     f"{m['delta'] / 1e3:+.0f} | {ci(np.array(m['paired_ci95']) / 1e3, '{:+.0f}')} | | "
                     f"reach {c['reach_rate'][str(t)]['arm']} vs {c['reach_rate'][str(t)]['control']} |")
        for stat in ('last5', 'final', 'auc'):
            m = c['metrics'][f'train/return/{stat}']
            L.append(f"| {task} | train | return {stat} | {m['arm_mean']:.0f} | {m['control_mean']:.0f} | "
                     f"{m['delta']:+.0f} | {ci(m['paired_ci95'], '{:+.0f}')} | | |")
    L.append('')


rl_table('E1 crop vs full (seeds 2000-2002)', report['E1']['rl'])
for name, agg in report['E1']['aggregate'].items():
    L.extend([f'### E1 aggregate, {name}', '', '| metric | mean delta | 95% CI | IQM crop | IQM full |', '|---|---|---|---|---|'])
    for key, a in agg.items():
        if '/success/' in key or key.startswith('time_to'):
            iq = f"{a['iqm_arm']:.3f} {ci(a['iqm_arm_ci95'], '{:.3f}')} | {a['iqm_control']:.3f} " \
                 f"{ci(a['iqm_control_ci95'], '{:.3f}')}" if 'iqm_arm' in a else ' | '
            L.append(f"| {key} | {a['mean_delta']:+.3f} | {ci(a['ci95'])} | {iq} |")
    L.append('')
for task, cells in report['E2']['rl'].items():
    rl_table(f'E2 {task}', {f'{task}: {k}': v for k, v in cells.items()})
rl_table('E3 CNN without RL shift vs CNN (seeds 2000-2002)', report['E3']['rl'])
if report['E3']['viewpoint']:
    L.extend(['### E3 viewpoint: delta(novel) - delta(train) (raw success units)', '',
              '| task | condition | DiD | paired 95% CI |', '|---|---|---|---|'])
    for task, v in report['E3']['viewpoint'].items():
        for key, d in v['did'].items():
            L.append(f"| {task} | {key} | {d['did']:+.3f} | {ci(d['paired_ci95'])} |")
    for key, d in report['E3']['aggregate'].get('eight_tasks_did', {}).items():
        L.append(f"| pooled | {key} | {d['mean_did']:+.3f} | {ci(d['ci95'])} |")
    L.append('')
for exp in ('E1', 'E2'):
    for task, rep in report[exp]['representation'].items():
        for label, r in (rep.items() if exp == 'E2' else [('crop vs full', rep)]):
            L.append(f"- {exp} representation {task} {label}: beyond-margin counts {r['beyond_margin_counts']}")
L.extend(['', f"Pending cells: {len(report['pending'])}"])
for p in report['pending']:
    L.append(f"- {p['experiment']} {p['comparison']} {p['task']}: {len(p['waiting_for'])} runs")
md = out.with_suffix('.md')
md.write_text('\n'.join(L) + '\n')
print('\n'.join(L[-len(report['pending']) - 2:]))
print('REPORT', out, sha(out))
print('DIGEST', md)
