"""Apply the future_k2 adoption rule (runs/setup/afk2-adoption-rule-20261022.json, recorded before any future_k2 RL
result was inspected) at decision point D1 (hammer, pick-place) or D2 (eight tasks).

Primary endpoint: last-5 success (mean of the final five evaluation points) on the training cameras (train0..train5),
per run; IQM over runs (tasks x seeds 2000-2002) of future_k2 (afk2-drm-<task>-s<seed>) minus that of the original
(s3-drm-splatter4d-<task>-s<seed>); task-stratified bootstrap (B = 2000, seeds resampled within task, both arms
independently) 95 % interval of the difference. Adopt if difference >= 0.05 and interval lower bound > 0.
Secondary values (descriptive): held-out and moving cameras, return, final success, AUC, per-task means.
Refuses unless every required run's evaluation reached 1M agent steps (final.json present, last step >= 1,000,000).

    python -I afk2_adoption_decision.py --point D1 [--self-test]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path('/home/ws/ws/splatter4d')
RUNS = ROOT / 'runs'
KINDS = {'train': [f'train{i}' for i in range(6)], 'heldout': [f'eval{i}' for i in range(4)],
         'traj': ['traj_lateral', 'traj_circular']}
SEEDS = (2000, 2001, 2002)
POINTS = {'D1': ('hammer', 'pick-place'),
          'D2': ('bin-picking', 'coffee-push', 'door-open', 'hammer', 'peg-insert-side', 'peg-unplug-side', 'pick-place',
                 'stick-push')}
ARMS = {'future_k2': 'afk2-drm-{task}-s{seed}', 'original': 's3-drm-splatter4d-{task}-s{seed}'}
B = 2000


def load(rid):
    path = RUNS / f'{rid}-eval' / 'eval.jsonl'
    if not path.is_file():
        path = RUNS / rid / 'eval.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not (RUNS / rid / 'final.json').is_file() or rows[-1]['step'] < 1_000_000:
        raise SystemExit(f'{rid}: evaluation incomplete (last step {rows[-1]["step"]})')
    out = {}
    for kind, names in KINDS.items():
        for measure in ('success', 'return'):
            s = np.array([np.mean([r[f'{n}_{measure}'] for n in names]) for r in rows])
            out[f'{kind}/{measure}'] = {'last5': float(s[-5:].mean()), 'final': float(s[-1]), 'auc': float(s.mean())}
    return out


def iqm(v):
    v = np.sort(np.asarray(v, float).ravel())
    k = int(0.25 * len(v))
    return float(v[k:len(v) - k].mean())


def boot_iqm(x, rng):
    x = np.asarray(x, float)  # (tasks, seeds)
    idx = rng.integers(0, x.shape[1], size=(B, *x.shape))
    boots = np.sort(np.take_along_axis(np.broadcast_to(x, (B, *x.shape)), idx, axis=2).reshape(B, -1), axis=1)
    k = int(0.25 * boots.shape[1])
    return boots[:, k:boots.shape[1] - k].mean(axis=1)


def compare(tasks, arms):
    data = {arm: {t: [load(pat.format(task=t, seed=s)) for s in SEEDS] for t in tasks} for arm, pat in arms.items()}
    rng = np.random.default_rng(0)
    out = {}
    names = list(arms)
    for key in ('train/success', 'heldout/success', 'traj/success', 'train/return'):
        for stat in ('last5', 'final', 'auc'):
            x = {arm: np.array([[r[key][stat] for r in data[arm][t]] for t in tasks]) for arm in arms}
            ba, bb = boot_iqm(x[names[0]], rng), boot_iqm(x[names[1]], rng)
            d = iqm(x[names[0]]) - iqm(x[names[1]])
            out[f'{key}/{stat}'] = {'iqm': {arm: iqm(x[arm]) for arm in arms}, 'difference': d,
                                    'ci95': [float(np.percentile(ba - bb, 2.5)), float(np.percentile(ba - bb, 97.5))],
                                    'per_task_mean': {t: {arm: float(x[arm][i].mean()) for arm in arms}
                                                      for i, t in enumerate(tasks)}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--point', choices=sorted(POINTS), required=True)
    ap.add_argument('--self-test', action='store_true', help='original vs itself (no future_k2 data read)')
    args = ap.parse_args()
    tasks = POINTS[args.point]
    arms = {'original_a': ARMS['original'], 'original_b': ARMS['original']} if args.self_test else ARMS
    result = compare(tasks, arms)
    primary = result['train/success/last5']
    if args.self_test:
        assert abs(primary['difference']) < 1e-12 and primary['ci95'][0] <= 0 <= primary['ci95'][1]
        print(json.dumps({'self_test': 'ok', 'iqm_original': primary['iqm']}, indent=1))
        return
    adopt = primary['difference'] >= 0.05 and primary['ci95'][0] > 0
    verdict = 'ADOPT' if adopt else ('DEFER to D2' if args.point == 'D1' else 'NOT ADOPTED')
    record = {'decided_at': time.strftime('%F %T %z'), 'point': args.point, 'tasks': list(tasks),
              'rule': 'runs/setup/afk2-adoption-rule-20261022.json', 'primary': primary, 'verdict': verdict,
              'all': result}
    out = ROOT / f'runs/setup/afk2-adoption-{args.point}-20261022.json'
    out.write_text(json.dumps(record, indent=1) + '\n')
    print(json.dumps({'verdict': verdict, 'primary_difference': primary['difference'], 'ci95': primary['ci95'],
                      'iqm': primary['iqm']}, indent=1))
    print(out)


if __name__ == '__main__':
    main()
