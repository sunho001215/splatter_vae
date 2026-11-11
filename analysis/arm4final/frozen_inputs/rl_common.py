"""Read-only loaders for the architecture comparison (DrM, original eight Meta-World tasks).

Mirrors work/g6/rl_data.py and work/g4/scripts/data.py (themselves mirrors of the results-artifact loaders):
- a run is COMPLETE when runs/<id>/final.json says completed at 1M agent steps AND the evaluation grid 10k..1M (every
  10k, 100 evaluations) is present; otherwise PARTIAL (evaluation rows exist, last step reported) or MISSING.
- evaluation file: runs/<id>-eval/eval.jsonl, else runs/<id>/eval.jsonl; the last record per step is kept.
- per evaluation: return = per-episode reward sum averaged over the camera's 20 episodes, then over cameras;
  success = fraction of successful episodes, averaged over cameras.
  camera groups: train = train0..5 (6 training cameras, 120 episodes), heldout = eval0..3 (4 held-out static cameras,
  80 episodes), traj = traj_lateral + traj_circular (in-training moving-camera protocol, 1x speed, 40 episodes).
Nothing here imports or executes campaign drivers, job scripts or sync tools.
"""
import json
from pathlib import Path

import numpy as np

RUNS = Path('/home/ws/ws/splatter4d/runs')
ART = Path('/home/ws/ws/splatter4d/reports/results-artifact')
TASKS = ('bin-picking', 'coffee-push', 'door-open', 'hammer', 'peg-insert-side', 'peg-unplug-side', 'pick-place',
         'stick-push')
PILOT = ('bin-picking', 'hammer', 'pick-place')
SEEDS = (2000, 2001, 2002)
GRID = list(range(10_000, 1_000_001, 10_000))
KINDS = {'train': [f'train{i}' for i in range(6)], 'heldout': [f'eval{i}' for i in range(4)],
         'traj': ['traj_lateral', 'traj_circular']}
NORM = json.load(open(ART / 'campaigns' / 'meta.json'))['norm_refs']

OURS = ('full', 'chl', 'chlrm', 'chlhm', 'cs', 'w4', 'csw4', 'w8', 'h2', 'hn', 'cst50', 'cst75')
BASELINES = ('cnn', 'cnnnoshift', 'sincro', 'reviwo', 'dynamo', 'mvp3d-multiview', 'mvp3d-single', 'snerl')
ARMS = OURS + BASELINES
LABEL = {
    'full': 'original (full)', 'chl': 'chl', 'chlrm': 'chlrm', 'chlhm': 'chlhm', 'cs': 'cs', 'w4': 'w4',
    'csw4': 'csw4 (Ours)', 'w8': 'w8 (pilot)', 'h2': 'h2 (pilot)', 'hn': 'hn (pilot)',
    'cst50': 'cst50 (csw4 + tube 50 %; stick-push 2/3 seeds)', 'cst75': 'cst75 (csw4 + tube 75 %)', 'cnn': 'CNN end-to-end', 'cnnnoshift': 'CNN end-to-end, no shift',
    'sincro': 'SinCro', 'reviwo': 'ReViWo', 'dynamo': 'DynaMo', 'mvp3d-multiview': '3D-MVP A (multi-view RGB-D)',
    'mvp3d-single': '3D-MVP B (single RGB-D)', 'snerl': 'SNeRL (3 posed views)'}
TASKS_OF = {a: (PILOT if a in ('w8', 'h2', 'hn') else TASKS) for a in ARMS}
# Known gap (2026-11-06 snapshot): cst50-drm-stick-push-s2002 trained to 1M but its evaluation never ran (launch refused,
# awaiting the user's reconciliation) -> MISSING under the completeness rule; cst50 stick-push has 2 complete seeds.
# Nothing is imputed: a task cell with 2 complete seeds is reported with n = 2 (seed bootstrap over those 2) and every
# pooled cst50 value is given twice, over all 8 tasks (stick-push n = 2) and over the 7 tasks without stick-push.
MIN_SEEDS_REPORTED = 2
NO_STICK = tuple(t for t in TASKS if t != 'stick-push')


def run_id(arm, task, seed):
    if arm == 'full':
        return f's3-drm-splatter4d-{task}-s{seed}'
    if arm in ('cs', 'w4', 'csw4', 'w8', 'h2', 'hn'):
        return f'enc-{arm}-drm-{task}-s{seed}'
    if arm in ('chl', 'chlrm', 'chlhm', 'cst50', 'cst75'):
        return f'{arm}-drm-{task}-s{seed}'
    if arm == 'cnn' and (task, seed) == ('bin-picking', 2000):
        return 's4-drm-cnn-bin-picking-s2000-r4'  # as section 07 (earlier attempts r2/r3 superseded)
    return f's4-drm-{arm}-{task}-s{seed}'


def read_run(rid):
    path = RUNS / f'{rid}-eval' / 'eval.jsonl'
    if not path.is_file():
        path = RUNS / rid / 'eval.jsonl'
    if not path.is_file():
        return {'rid': rid, 'status': 'missing', 'last_step': 0, 'n_evals': 0, 'eval_path': ''}
    by_step = {}
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get('step', 0) <= 1_000_000:
                by_step[r['step']] = r
    rows = [by_step[s] for s in sorted(by_step)]
    if not rows:
        return {'rid': rid, 'status': 'missing', 'last_step': 0, 'n_evals': 0, 'eval_path': str(path)}
    out = {'rid': rid, 'eval_path': str(path), 'steps': [r['step'] for r in rows], 'last_step': rows[-1]['step'],
           'n_evals': len(rows), 'episodes': sorted({r.get('episodes') for r in rows})}
    for kind, names in KINDS.items():
        for m in ('success', 'return'):
            out[f'{kind}/{m}'] = np.array([np.mean([r[f'{n}_{m}'] for n in names]) for r in rows])
    fin = RUNS / rid / 'final.json'
    done = False
    if fin.is_file():
        f = json.loads(fin.read_text())
        done = f.get('completed') is True and f.get('steps') == 1_000_000
    out['final_json'] = done
    out['status'] = 'complete' if (done and out['steps'] == GRID) else 'partial'
    return out


def nr(task, R):
    a = NORM[task]
    return (np.asarray(R, float) - a['random']) / (a['expert'] - a['random'])


def summary(run, key, s, task=None):
    """last5 = mean of the last five evaluations (960k..1M for a complete run); auc = mean over all evaluations
    (10k..1M, 100 points, i.e. the normalised area under the learning curve)."""
    if key.endswith('/nR'):
        c = nr(task, run[key.replace('/nR', '/return')])
    else:
        c = run[key]
    return {'last5': float(c[-5:].mean()), 'auc': float(c.mean()), 'final': float(c[-1])}[s]


def get_nR(run, kind, task):
    return nr(task, run[f'{kind}/return'])


# ---- camera suite v2 (eval-only, 1M checkpoint, paired v2 episode seeds); exists for csw4 + baselines only ----
CV2_GROUPS = {
    'cv2_fixed_train': [f'ref_train{i}' for i in range(6)],
    'cv2_moving_1x': ['ref_traj_lateral', 'ref_traj_circular'],
    'cv2_moving_2x': ['traj_lateral_2x', 'traj_circular_2x'],
    'cv2_moving_3x': ['traj_lateral_3x', 'traj_circular_3x'],
    'cv2_heldout_v2': ['vis0', 'vis1', 'vis2', 'vis3'],
}


def cv2_run(arm, task, seed):
    if arm == 'csw4':
        return f'ds-cv2-enc-csw4-drm-{task}-s{seed}-eval'
    if arm == 'cnn' and (task, seed) == ('bin-picking', 2000):
        return 'ds-cv2-s4-drm-cnn-bin-picking-s2000-r4-eval'
    if arm in BASELINES:
        return f'ds-cv2-s4-drm-{arm}-{task}-s{seed}-eval'
    if arm in ('cs', 'w4', 'w8', 'h2', 'hn'):
        return f'ds-cv2-enc-{arm}-drm-{task}-s{seed}-eval'
    if arm == 'full':
        return f'ds-cv2-s3-drm-splatter4d-{task}-s{seed}-eval'
    return f'ds-cv2-{arm}-drm-{task}-s{seed}-eval'


def load_cv2(arm, task, seed):
    p = RUNS / cv2_run(arm, task, seed) / 'eval_cameras_v2.jsonl'
    if not p.is_file():
        return None
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    rr = [x for x in rows if x['step'] == 1_000_000]
    if not rr:
        return None
    r = rr[-1]
    out = {'path': str(p), 'episodes': r['episodes']}
    for g, names in CV2_GROUPS.items():
        for m in ('success', 'return'):
            out[f'{g}/{m}'] = float(np.mean([r[f'{n}_{m}'] for n in names]))
        out[f'{g}/nR'] = float(nr(task, out[f'{g}/return']))
    return out
