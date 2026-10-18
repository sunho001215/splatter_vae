"""Eight-task final comparison with the new baselines (registration baselines-eight-task-registration-20261017.json).

Reporting standard of rl_reporting.py, unchanged: success and return as last-5 / final / AUC for training cameras,
held-out cameras and trajectories; per task (seed mean, per-seed values, seed bootstrap 95 % CI) and IQM over
8 tasks x seeds 2000-2002 with the stratified bootstrap CI (success only; returns are task-scaled). Rows: ours, SinCro,
ReViWo, CNN (the completed final comparison) and DynaMo, 3D-MVP A "multi-view RGB-D + extrinsics (privileged)",
3D-MVP B "single RGB-D + extrinsics (privileged)" as separate rows with their input labels. Descriptive: no new
decision rule. Fairness: RL config keys differing from ours outside the encoder/input fields, hosts and commits,
evaluation episodes per point.

    python -I baselines_eight_task_report.py
"""
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

ROOT = Path('/home/ws/ws/splatter4d')
sys.path.insert(0, str(ROOT))
from scripts import jobs  # noqa: E402
sys.path.insert(0, str(ROOT / 'runs/setup/tools'))
from report_figures import png_curves, svg_curves  # noqa: E402

RUNS, SETUP = ROOT / 'runs', ROOT / 'runs/setup'
STEPS = list(range(10000, 1000001, 10000))
PATHS = {'train': [f'train{i}' for i in range(6)], 'heldout': [f'eval{i}' for i in range(4)],
         'traj': ['traj_lateral', 'traj_circular']}
STATS = ('last5', 'final', 'auc')
TASKS = ('bin-picking', 'coffee-push', 'door-open', 'hammer', 'peg-insert-side', 'peg-unplug-side', 'pick-place',
         'stick-push')
SEEDS = (2000, 2001, 2002)
METHODS = {'ours': 'Splatter4D (ours)', 'sincro': 'SinCro', 'reviwo': 'ReViWo', 'cnn': 'DrM CNN',
           'dynamo': 'DynaMo', 'mvp3d-multiview': '3D-MVP A, multi-view RGB-D + extrinsics (privileged)',
           'mvp3d-single': '3D-MVP B, single RGB-D + extrinsics (privileged)'}
ENCODER_FIELDS = ('encoder_name', 'vision.', 'env.cameras', 'env.depth', 'env.views', 'agent.feature_dim')
B = 10000
STATE = jobs.read_registry()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_id(method, task, seed):
    if method == 'ours':
        return f's3-drm-splatter4d-{task}-s{seed}'
    if method == 'cnn' and (task, seed) == ('bin-picking', 2000):
        return 's4-drm-cnn-bin-picking-s2000-r4'
    return f's4-drm-{method}-{task}-s{seed}'


def eval_path(rid):
    path = RUNS / f'{rid}-eval' / 'eval.jsonl'
    return path if path.is_file() else RUNS / rid / 'eval.jsonl'


def ready(rid):
    jobs_ = [rid] + ([f'{rid}-eval'] if f'{rid}-eval' in STATE else [])
    return all(STATE.get(j, {}).get('status') == 'done' for j in jobs_)


def flat(tree, prefix=''):
    out = {}
    for key, value in tree.items():
        if isinstance(value, dict):
            out.update(flat(value, f'{prefix}{key}.'))
        else:
            out[prefix + key] = value
    return out


def load(rid):
    rows = [json.loads(line) for line in eval_path(rid).read_text().splitlines() if line.strip()]
    assert [r['step'] for r in rows] == STEPS, rid
    final = json.loads((RUNS / rid / 'final.json').read_text())
    assert final['completed'] is True and final['steps'] == 1_000_000, rid
    synced = STATE.get(f'{rid}-eval', STATE.get(rid, {})).get('sync', {}).get('files', {}).get('eval.jsonl')
    assert synced in (None, sha(eval_path(rid))), rid
    out, curves = {}, {}
    for kind, paths in PATHS.items():
        for measure in ('success', 'return'):
            series = np.array([np.mean([r[f'{p}_{measure}'] for p in paths]) for r in rows])
            out[f'{kind}/{measure}'] = {'last5': float(series[-5:].mean()), 'final': float(series[-1]),
                                        'auc': float(series.mean())}
            if measure == 'success':
                curves[kind] = series
    assert np.allclose([r['train_cameras_success'] for r in rows], curves['train']), rid
    prov = json.loads((RUNS / rid / 'provenance.json').read_text())
    return {'scores': out, 'curves': curves, 'episodes': sorted({r['episodes'] for r in rows}),
            'host': prov['host'], 'git_commit': prov['git_commit'], 'eval_jsonl_sha256': sha(eval_path(rid))}


def iqm_with_ci(x):
    def iqm(v):
        v = np.sort(v, axis=-1)
        k = int(0.25 * v.shape[-1])
        return v[..., k:v.shape[-1] - k].mean(axis=-1)
    idx = np.random.default_rng(0).integers(0, x.shape[1], size=(B, *x.shape))
    boots = iqm(np.take_along_axis(np.broadcast_to(x, (B, *x.shape)), idx, axis=2).reshape(B, -1))
    return float(iqm(x.ravel())), [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def mean_ci95(values):
    values = np.asarray(values, dtype=float)
    boots = values[np.random.default_rng(0).integers(0, len(values), size=(B, len(values)))].mean(axis=1)
    return [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


missing = [run_id(m, t, s) for m in METHODS for t in TASKS for s in SEEDS if not ready(run_id(m, t, s))]
if missing:
    print('WAITING', len(missing), missing[:6])
    raise SystemExit(0)
runs = {run_id(m, t, s): load(run_id(m, t, s)) for m in METHODS for t in TASKS for s in SEEDS}
report = {'recorded_local': datetime.now().astimezone().isoformat(), 'standard': __doc__.split('\n\n')[1],
          'methods': METHODS, 'runs': {k: {kk: vv for kk, vv in v.items() if kk != 'curves'} for k, v in runs.items()},
          'comparison': {}, 'fairness': {}}
for method in METHODS:
    for metric in runs[run_id(method, TASKS[0], SEEDS[0])]['scores']:
        for stat in STATS:
            x = np.array([[runs[run_id(method, t, s)]['scores'][metric][stat] for s in SEEDS] for t in TASKS])
            entry = report['comparison'].setdefault(metric, {}).setdefault(stat, {}).setdefault(method, {})
            entry['per_task_mean'] = {t: float(x[i].mean()) for i, t in enumerate(TASKS)}
            entry['per_task_seeds'] = {t: x[i].tolist() for i, t in enumerate(TASKS)}
            entry['per_task_mean_ci95'] = {t: mean_ci95(x[i]) for i, t in enumerate(TASKS)}
            if metric.endswith('/success'):
                entry['iqm'], entry['iqm_ci95'] = iqm_with_ci(x)
fair = report['fairness']
diffs, hosts, episodes = {}, Counter(), Counter()
for method in METHODS:
    for t in TASKS:
        for s in SEEDS:
            rid = run_id(method, t, s)
            ref = flat(yaml.safe_load((RUNS / run_id('ours', t, s) / 'config.yaml').read_text()))
            cfg = flat(yaml.safe_load((RUNS / rid / 'config.yaml').read_text()))
            assert cfg['seed'] == s and cfg['task'] == t, rid
            other = sorted(k for k in set(cfg) | set(ref) if cfg.get(k) != ref.get(k) and not k.startswith('run.')
                           and not k.startswith(ENCODER_FIELDS))
            if other:
                diffs[rid] = other
            hosts[f"{method} @ {runs[rid]['host']} ({runs[rid]['git_commit'][:7]})"] += 1
            episodes[str(runs[rid]['episodes'])] += 1
fair['rl_config_differences_outside_encoder_fields'] = diffs
fair['allowed_differences'] = list(ENCODER_FIELDS)
fair['method_host_commit_counts'] = dict(hosts)
fair['evaluation_episodes_per_point'] = dict(episodes)

figures = ROOT / 'docs/figures/baselines/eight_task'
figures.mkdir(parents=True, exist_ok=True)
panels = []
for task in TASKS + ('mean over tasks',):
    for kind in PATHS:
        series = {}
        for m in METHODS:
            if task == 'mean over tasks':
                series[m] = np.mean([[runs[run_id(m, t, s)]['curves'][kind] for s in SEEDS] for t in TASKS], axis=0)
            else:
                series[m] = np.array([runs[run_id(m, task, s)]['curves'][kind] for s in SEEDS])
        panels.append((f'{task} / {kind}', series))
title = 'Eight-task final comparison incl. DynaMo and 3D-MVP (success, seeds 2000-2002)'
svg_curves(figures / 'success_curves.svg', title, panels, tuple(METHODS))
png_curves(figures / 'success_curves.png', title, panels, tuple(METHODS))

out = SETUP / f'baselines-eight-task-report-{datetime.now():%Y%m%d-%H%M%S}.json'
out.write_text(json.dumps(report, indent=1, default=str) + '\n')
L = [f'# Eight-task final comparison with DynaMo and 3D-MVP ({report["recorded_local"]})', '',
     f'JSON: `{out.relative_to(ROOT)}`; figure: `docs/figures/baselines/eight_task/success_curves.png`', '',
     '## IQM of success over 8 tasks x 3 seeds [stratified bootstrap 95% CI]', '',
     '| method | camera | last-5 | final | AUC |', '|---|---|---|---|---|']
for m, label in METHODS.items():
    for kind in PATHS:
        cells = [f"{report['comparison'][f'{kind}/success'][st][m]['iqm']:.3f} "
                 f"[{report['comparison'][f'{kind}/success'][st][m]['iqm_ci95'][0]:.3f}, "
                 f"{report['comparison'][f'{kind}/success'][st][m]['iqm_ci95'][1]:.3f}]" for st in STATS]
        L.append(f'| {label} | {kind} | ' + ' | '.join(cells) + ' |')
for kind in PATHS:
    L.extend(['', f'## Per-task seed-mean success, {kind} cameras (last-5 / final / AUC)', '',
              '| task | ' + ' | '.join(METHODS) + ' |', '|---' * (len(METHODS) + 1) + '|'])
    for t in TASKS:
        L.append(f'| {t} | ' + ' | '.join(
            '/'.join(f"{report['comparison'][f'{kind}/success'][st][m]['per_task_mean'][t]:.2f}" for st in STATS)
            for m in METHODS) + ' |')
L.extend(['', '## Per-task seed-mean return, training cameras (last-5 / final / AUC)', '',
          '| task | ' + ' | '.join(METHODS) + ' |', '|---' * (len(METHODS) + 1) + '|'])
for t in TASKS:
    L.append(f'| {t} | ' + ' | '.join(
        '/'.join(f"{report['comparison']['train/return'][st][m]['per_task_mean'][t]:.0f}" for st in STATS)
        for m in METHODS) + ' |')
L.extend(['', '## Fairness', '', f"- RL config differences outside encoder/input fields: {diffs or 'none'}",
          f"- evaluation episodes per point: {dict(episodes)}", f"- method @ host (commit): {dict(hosts)}"])
md = out.with_suffix('.md')
md.write_text('\n'.join(L) + '\n')
print('\n'.join(L[:5 + 3 * len(METHODS) + 2]))
print('REPORT', out, sha(out))
print('DIGEST', md)
