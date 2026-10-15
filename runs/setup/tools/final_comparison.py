"""Final in-domain comparison (Stages 3/4), as pre-registered in docs/RL_PROTOCOL.md "Reported metrics":
final success = the last evaluation (1M agent steps); success AUC = mean success over all 100 evaluation points.
Training cameras, held-out cameras and trajectories are reported separately. Per task: mean and 95 % bootstrap CI
over seeds 2000-2002. Aggregate over tasks x seeds: interquartile mean (IQM) and probability of improvement of ours
over each baseline (rliable definitions, implemented here without new dependencies), with stratified bootstrap CIs.

Methods with complete runs: ours (frozen C + D0 + M3D, 200k), SinCro, ReViWo, CNN. SNeRL, DynaMo and 3D-MVP are not
run (upstream execution permission-blocked). Every run must have final.json at 1M and evaluations at exactly
10k, 20k, ..., 1M, and a driver closeout record.

    python -I final_comparison.py
"""
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path('/home/ws/ws/splatter4d')
RUNS = ROOT / 'runs'
TASKS = ('bin-picking', 'coffee-push', 'door-open', 'hammer', 'peg-insert-side', 'peg-unplug-side', 'pick-place',
         'stick-push')
SEEDS = (2000, 2001, 2002)
KINDS = {'train': 'train_cameras_success', 'heldout': 'heldout_cameras_success', 'traj': 'trajectories_success'}
METHODS = ('ours', 'sincro', 'reviwo', 'cnn')
B = 10000  # bootstrap resamples
STEPS = list(range(10000, 1000001, 10000))


def run_id(method, task, seed):
    if method == 'ours':
        return f's3-drm-splatter4d-{task}-s{seed}'
    if method == 'cnn' and (task, seed) == ('bin-picking', 2000):
        return 's4-drm-cnn-bin-picking-s2000-r4'  # attempts r1-r3 failed (infrastructure); RESULTS.md Stage 4 CNN
    return f's4-drm-{method}-{task}-s{seed}'


def closeout(method, rid):
    if method == 'cnn':
        if rid == 's4-drm-cnn-pick-place-s2000':  # the local CNN run (OOM-resumed at 650k; RESULTS.md caveat)
            return {'local': True}
        path = RUNS / 'remote/stage4-finished' / f'{rid}.json'
    else:
        path = RUNS / 'remote/finished' / f'{rid}.json'
    record = json.loads(path.read_text())
    assert record['final_steps'] == 1_000_000 and record['eval_points'] == 100, rid
    return record


def load(method, task, seed):
    rid = run_id(method, task, seed)
    closeout(method, rid)
    final = json.loads((RUNS / rid / 'final.json').read_text())
    assert final['completed'] is True and final['steps'] == 1_000_000, rid
    path = RUNS / f'{rid}-eval' / 'eval.jsonl'
    if not path.is_file():
        path = RUNS / rid / 'eval.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert [r['step'] for r in rows] == STEPS, rid
    return rid, hashlib.sha256(path.read_bytes()).hexdigest(), {
        k: {'final': float(rows[-1][c]), 'auc': float(np.mean([r[c] for r in rows]))} for k, c in KINDS.items()}


def iqm(x):
    x = np.sort(np.asarray(x).ravel())
    k = int(0.25 * x.size)
    return float(x[k:x.size - k].mean())


def prob_improvement(x, y):
    """x, y: (tasks, seeds). Mean over tasks of P(X > Y) with ties counted 1/2 (Mann-Whitney)."""
    p = [((xt[:, None] > yt[None, :]) + 0.5 * (xt[:, None] == yt[None, :])).mean() for xt, yt in zip(x, y)]
    return float(np.mean(p))


rng = np.random.default_rng(0)
scores, sources = {}, {}
for method in METHODS:
    for task in TASKS:
        for seed in SEEDS:
            rid, digest, s = load(method, task, seed)
            sources[rid] = digest
            for kind in KINDS:
                for stat in ('final', 'auc'):
                    scores.setdefault((method, kind, stat), np.zeros((len(TASKS), len(SEEDS))))[
                        TASKS.index(task), SEEDS.index(seed)] = s[kind][stat]


def strat(x, idx):
    return np.take_along_axis(x, idx, axis=1)


report = {'recorded_local': datetime.now().astimezone().isoformat(), 'protocol': 'docs/RL_PROTOCOL.md',
          'bootstrap_resamples': B, 'rng_seed': 0, 'methods': METHODS, 'tasks': TASKS, 'seeds': SEEDS,
          'eval_jsonl_sha256': sources, 'per_task': {}, 'aggregate': {}}
for (method, kind, stat), x in scores.items():
    for t, task in enumerate(TASKS):
        boots = x[t][rng.integers(0, len(SEEDS), size=(B, len(SEEDS)))].mean(axis=1)
        report['per_task'].setdefault(f'{kind}/{stat}', {}).setdefault(task, {})[method] = {
            'seeds': x[t].tolist(), 'mean': float(x[t].mean()),
            'ci95': [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}
    idx = rng.integers(0, len(SEEDS), size=(B, len(TASKS), len(SEEDS)))
    boots = [iqm(strat(x, i)) for i in idx]
    report['aggregate'].setdefault(f'{kind}/{stat}', {})[method] = {
        'iqm': iqm(x), 'iqm_ci95': [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        'mean': float(x.mean())}
for kind in KINDS:
    for stat in ('final', 'auc'):
        ours = scores[('ours', kind, stat)]
        for base in METHODS[1:]:
            other = scores[(base, kind, stat)]
            ix = rng.integers(0, len(SEEDS), size=(B, len(TASKS), len(SEEDS)))
            iy = rng.integers(0, len(SEEDS), size=(B, len(TASKS), len(SEEDS)))
            boots = [prob_improvement(strat(ours, a), strat(other, b)) for a, b in zip(ix, iy)]
            report['aggregate'][f'{kind}/{stat}'].setdefault('prob_improvement_ours_over', {})[base] = {
                'p': prob_improvement(ours, other),
                'ci95': [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}

out = ROOT / 'runs/setup' / f"final-comparison-{datetime.now():%Y%m%d-%H%M%S}.json"
out.write_text(json.dumps(report, indent=2) + '\n')
fmt = lambda r: f"{r['mean']:.2f} [{r['ci95'][0]:.2f},{r['ci95'][1]:.2f}]"  # noqa: E731
for kind in KINDS:
    for stat in ('final', 'auc'):
        key = f'{kind}/{stat}'
        print(f'\n== {key}: per task mean [95% CI over seeds]')
        print(f"{'task':16s} " + ' '.join(f'{m:>19s}' for m in METHODS))
        for task in TASKS:
            print(f'{task:16s} ' + ' '.join(f"{fmt(report['per_task'][key][task][m]):>19s}" for m in METHODS))
        agg = report['aggregate'][key]
        cells = [f"{agg[m]['iqm']:.2f} [{agg[m]['iqm_ci95'][0]:.2f},{agg[m]['iqm_ci95'][1]:.2f}]" for m in METHODS]
        print(f"{'IQM':16s} " + ' '.join(c.rjust(19) for c in cells))
        poi = agg['prob_improvement_ours_over']
        print('P(ours > baseline): ' + ', '.join(
            f"{b} {v['p']:.2f} [{v['ci95'][0]:.2f},{v['ci95'][1]:.2f}]" for b, v in poi.items()))
print('\nREPORT', out, hashlib.sha256(out.read_bytes()).hexdigest())
