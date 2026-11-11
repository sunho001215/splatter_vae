"""arm4 final: SUPPLEMENTARY, DESCRIPTIVE measurements that the registered plan lists under cost / stability but the frozen
scripts do not compute (training stability, computational cost, encoder size). Read-only; CPU; no new runs.

Nothing here is a registered endpoint or a decision input: the registered readings come only from the frozen
arm4_rl.py / arm4_render_metrics.py outputs. Unavailable quantities are written as null with a reason.

    .venv/bin/python -I -B supplementary.py <out-dir>
"""
import json
import math
import statistics as st
import sys
from pathlib import Path

import numpy as np

ROOT = Path('/home/ws/ws/splatter4d')
RUNS = ROOT / 'runs'
TASKS = ('bin-picking', 'coffee-push', 'door-open', 'hammer', 'peg-insert-side', 'peg-unplug-side', 'pick-place',
         'stick-push')
SEEDS = (2000, 2001, 2002)
GRID = list(range(10000, 1000001, 10000))
MODELS = {
    'original': {'drm': 's3-drm-splatter4d-{t}-s{s}', 'pretrain': None, 'export': None},
    'cst50': {'drm': 'cst50-drm-{t}-s{s}', 'pretrain': 'cst50-seed0-{t}-200k', 'export': 'cst50-export-seed0-{t}-200k'},
    'chl': {'drm': 'chl-drm-{t}-s{s}', 'pretrain': 'chl-seed0-{t}-200k', 'export': 'chl-export-seed0-{t}-200k'},
    'A': {'drm': 'ocsw4-drm-{t}-s{s}', 'pretrain': 'ocsw4-seed0-{t}-200k', 'export': 'ocsw4-export-seed0-{t}-200k'},
    'B': {'drm': 'ochf-drm-{t}-s{s}', 'pretrain': 'ochf-seed0-{t}-200k', 'export': 'ochf-export-seed0-{t}-200k'},
    'C': {'drm': 'c50dg-drm-{t}-s{s}', 'pretrain': 'c50dg-seed0-{t}-200k', 'export': 'c50dg-export-seed0-{t}-200k'},
    'D': {'drm': 'c50vis-drm-{t}-s{s}', 'pretrain': 'c50vis-seed0-{t}-200k', 'export': 'c50vis-export-seed0-{t}-200k'},
}


def original_pretrain_dir(t):
    return RUNS / (f'pretrain/i2-m3d-seed0-{t}' if t in ('hammer', 'pick-place') else f's3-m3d-seed0-{t}-200k')


def original_export(t):
    return RUNS / (f'exports/ls-a200-m3d-seed0-{t}-200k.pt' if t in ('hammer', 'pick-place')
                   else f's3-export-m3d-seed0-{t}-200k/encoder.pt')


def pretrain_dir(model, t):
    if model == 'original':
        return original_pretrain_dir(t)
    rid = MODELS[model]['pretrain'].format(t=t)
    for d in (RUNS / 'pretrain' / rid, RUNS / rid):
        if (d / 'metrics.jsonl').is_file():
            return d
    return None


def registry_durations():
    launched, exited = {}, {}
    for line in (ROOT / 'experiments/registry.jsonl').open():
        if '"event"' not in line:
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        i, ev = e.get('id'), e.get('event')
        if ev == 'launched' and e.get('time'):
            launched.setdefault(i, []).append(e['time'])
        elif ev in ('exited', 'remote_exited') and e.get('code') == 0 and e.get('time') and i not in exited:
            exited[i] = e['time']
    return launched, exited


def wall_hours(rid, launched, exited):
    """Wall-clock of the successful attempt (last launch before the first successful exit); None if unknown."""
    if rid not in exited or rid not in launched:
        return None
    starts = [t for t in launched[rid] if t <= exited[rid]]
    return (exited[rid] - max(starts)) / 3600 if starts else None


def pretrain_stability(d):
    """Non-finite losses, spikes of loss/total (> 3x the median of the previous 200 logged values), step time."""
    tot, step_time, gpu_time, nonfinite, n = [], [], [], 0, 0
    final_val = None
    for line in (d / 'metrics.jsonl').open():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if 'loss/total' in r:
            n += 1
            v = r['loss/total']
            if v is None or not math.isfinite(v):
                nonfinite += 1
            else:
                tot.append(v)
        for key, arr in (('train/step_time', step_time), ('train/gpu_time', gpu_time)):
            if isinstance(r.get(key), (int, float)) and math.isfinite(r[key]):
                arr.append(r[key])
        if r.get('step') == 200000 and 'val/total@s2' in r:
            final_val = r['val/total@s2']
    spikes = 0
    a = np.asarray(tot)
    for i in range(200, len(a)):
        if a[i] > 3 * np.median(a[i - 200:i]):
            spikes += 1
    return {'logged_train_records': n, 'nonfinite_total_loss': nonfinite, 'loss_spikes_gt3x_median200': spikes,
            'median_step_time_s': st.median(step_time) if step_time else None,
            'median_gpu_time_s': st.median(gpu_time) if gpu_time else None, 'val_total_s2_at_200k': final_val}


def eval_curve(rid):
    p = RUNS / f'{rid}-eval' / 'eval.jsonl'
    if not p.is_file():
        return None
    last = {}
    for line in p.open():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        last[r['step']] = r
    if sorted(last) != GRID:
        return None
    return np.array([np.mean([last[s][f'train{i}_success'] for i in range(6)]) for s in GRID])


def drm_stability(curve):
    """Descriptive: peak of the 5-point moving average vs the last-5 mean (late degradation), curve roughness."""
    ma = np.convolve(curve, np.ones(5) / 5, mode='valid')
    last5 = float(curve[-5:].mean())
    return {'auc': float(curve.mean()), 'last5': last5, 'peak_ma5': float(ma.max()),
            'late_drop': float(ma.max() - last5), 'roughness': float(np.mean(np.abs(np.diff(curve))))}


def encoder_params(path):
    import torch
    d = torch.load(path, map_location='cpu', weights_only=False)
    sd = d.get('state_dict', d)
    return int(sum(v.numel() for v in sd.values() if hasattr(v, 'numel'))), d.get('encoder_config')


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    launched, exited = registry_durations()
    res = {'meta': {'label': 'SUPPLEMENTARY, descriptive; not registered endpoints, never decision inputs',
                    'definitions': {
                        'late_drop': 'max of the 5-evaluation moving average of training success minus the last-5 mean',
                        'roughness': 'mean |difference| between consecutive evaluations of training success',
                        'loss_spikes': 'logged loss/total values > 3x the median of the previous 200 logged values',
                        'wall_hours': 'registry launch-to-successful-exit of the counted attempt (shared MIG; '
                                      'co-residence and host load confound speed)'}},
           'pretrain': {}, 'drm': {}, 'cost': {}, 'encoder': {}}
    for m, spec in MODELS.items():
        res['pretrain'][m], res['drm'][m], res['encoder'][m] = {}, {}, {}
        cost = {'pretrain_wall_h': [], 'drm_wall_h': [], 'eval_wall_h': []}
        for t in TASKS:
            d = pretrain_dir(m, t)
            res['pretrain'][m][t] = pretrain_stability(d) if d else None
            if spec['pretrain']:
                h = wall_hours(spec['pretrain'].format(t=t), launched, exited)
                if h is not None:
                    cost['pretrain_wall_h'].append(h)
            ex = original_export(t) if m == 'original' else RUNS / spec['export'].format(t=t) / 'encoder.pt'
            if ex.is_file():
                n, cfg = encoder_params(ex)
                res['encoder'][m][t] = {'params': n, 'config': cfg}
            res['drm'][m][t] = {}
            for s in SEEDS:
                rid = spec['drm'].format(t=t, s=s)
                c = eval_curve(rid)
                res['drm'][m][t][s] = drm_stability(c) if c is not None else None
                for key, r in (('drm_wall_h', rid), ('eval_wall_h', rid + '-eval')):
                    h = wall_hours(r, launched, exited)
                    if h is not None:
                        cost[key].append(h)
        res['cost'][m] = {k: {'n': len(v), 'median': st.median(v) if v else None, 'sum': sum(v) if v else None}
                          for k, v in cost.items()}
        steps = [res['pretrain'][m][t]['median_step_time_s'] for t in TASKS
                 if res['pretrain'][m][t] and res['pretrain'][m][t]['median_step_time_s']]
        res['cost'][m]['pretrain_median_step_time_s'] = st.median(steps) if steps else None
    # per-model summaries
    summ = {}
    for m in MODELS:
        runs = [v for t in TASKS for v in res['drm'][m][t].values() if v]
        cell_sd = [float(np.std([v['auc'] for v in res['drm'][m][t].values() if v], ddof=1))
                   for t in TASKS if sum(1 for v in res['drm'][m][t].values() if v) >= 2]
        pre = [res['pretrain'][m][t] for t in TASKS if res['pretrain'][m][t]]
        summ[m] = {'complete_runs': len(runs),
                   'median_late_drop': st.median([v['late_drop'] for v in runs]) if runs else None,
                   'runs_late_drop_gt_0.2': sum(v['late_drop'] > 0.2 for v in runs),
                   'median_roughness': st.median([v['roughness'] for v in runs]) if runs else None,
                   'mean_within_cell_seed_sd_auc': float(np.mean(cell_sd)) if cell_sd else None,
                   'pretrain_nonfinite_total': sum(p['nonfinite_total_loss'] for p in pre) if pre else None,
                   'pretrain_spikes_total': sum(p['loss_spikes_gt3x_median200'] for p in pre) if pre else None,
                   'encoder_params': sorted({v['params'] for v in res['encoder'][m].values()})}
    res['summary'] = summ
    (out / 'supplementary.json').write_text(json.dumps(res, indent=1, default=str) + '\n')
    for m, s in summ.items():
        print(m, json.dumps(s), json.dumps(res['cost'][m]))


if __name__ == '__main__':
    main()
