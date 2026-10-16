"""RL reporting standard and Stage 2 diagnosis (report only; no decision follows from it).

Reporting standard: success AND return, each as last-5 (mean of the last five evaluations), final (the 1M evaluation)
and AUC (mean over all 100 evaluation points), for training cameras, held-out cameras and trajectories. Aggregate
returns are plain means over the camera paths, as the stored aggregate successes are (s4d/rl/evaluate.py). Reported
per run, per task and method/arm (seed mean and per-seed values), and as IQM over tasks x seeds with stratified
bootstrap 95 % CIs for success. Returns are task-scaled, so they are reported per task only.

Groups: final comparison (ours, SinCro, ReViWo, CNN; 8 tasks; seeds 2000-2002) and Stage 2 RL (full, T=1,
no-motion, pure no-InfoNCE, single-group; hammer, pick-place; seeds 1000-1002).

Stage 2 fairness checks (same host, same code, same RL configuration except the encoder, same evaluation protocol):
resolved RL config diffs against the full-method run of the same task and seed, provenance commits, evaluation
cadence and episode counts, encoder-export configs, GPU placement, throughput, and checkpoint pauses/resumes.

    python -I rl_reporting.py
"""
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

ROOT = Path('/home/ws/ws/splatter4d')
RUNS = ROOT / 'runs'
STEPS = list(range(10000, 1000001, 10000))
PATHS = {'train': [f'train{i}' for i in range(6)], 'heldout': [f'eval{i}' for i in range(4)],
         'traj': ['traj_lateral', 'traj_circular']}
STATS = ('last5', 'final', 'auc')
FINAL_TASKS = ('bin-picking', 'coffee-push', 'door-open', 'hammer', 'peg-insert-side', 'peg-unplug-side', 'pick-place',
               'stick-push')
FINAL_SEEDS = (2000, 2001, 2002)
S2_TASKS, S2_SEEDS = ('hammer', 'pick-place'), (1000, 1001, 1002)
S2_ARMS = ('full', 't1', 'nomotion', 'noinfonce', 'singlegroup')
# Later Stage 2 RL arms (crop follow-up 2026-10-15; no-synth arm 2026-10-16), reported per task once all seeds closed out.
EXTRA_ARMS = ('crop', 'cropnosynth', 'nosynth')
B = 10000


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def final_run(method, task, seed):
    if method == 'ours':
        return f's3-drm-splatter4d-{task}-s{seed}'
    if method == 'cnn' and (task, seed) == ('bin-picking', 2000):
        return 's4-drm-cnn-bin-picking-s2000-r4'
    return f's4-drm-{method}-{task}-s{seed}'


def eval_path(rid):
    path = RUNS / f'{rid}-eval' / 'eval.jsonl'
    return path if path.is_file() else RUNS / rid / 'eval.jsonl'


def closed_out(rid):
    """A run counts only after its trainer/companion closeout proof (Stage 2 driver) exists."""
    return (ROOT / 'runs/setup' / f's2-{rid}-completion.json').is_file()


def eval_rows(rid):
    path = eval_path(rid)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert [r['step'] for r in rows] == STEPS, rid
    final = json.loads((RUNS / rid / 'final.json').read_text())
    assert final['completed'] is True and final['steps'] == 1_000_000, rid
    return rows, sha(path)


def scores(rows):
    out = {}
    for kind, paths in PATHS.items():
        for measure in ('success', 'return'):
            series = np.array([np.mean([r[f'{p}_{measure}'] for p in paths]) for r in rows])
            out[f'{kind}/{measure}'] = {'last5': float(series[-5:].mean()), 'final': float(series[-1]),
                                        'auc': float(series.mean())}
    stored = np.array([r['train_cameras_success'] for r in rows])
    assert np.allclose(stored, [np.mean([r[f'{p}_success'] for p in PATHS['train']]) for r in rows])
    return out


def iqm_with_ci(x, rng):
    """x: (tasks, seeds) -> IQM over all runs and a stratified bootstrap 95 % CI (seeds resampled within tasks)."""
    def iqm(flat):
        flat = np.sort(flat, axis=-1)
        k = int(0.25 * flat.shape[-1])
        return flat[..., k:flat.shape[-1] - k].mean(axis=-1)
    idx = rng.integers(0, x.shape[1], size=(B, x.shape[0], x.shape[1]))
    boots = iqm(np.take_along_axis(np.broadcast_to(x, (B, *x.shape)), idx, axis=2).reshape(B, -1))
    return float(iqm(x.ravel())), [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def mean_ci95(values):
    """Percentile bootstrap of a per-task seed mean; fixed resamples shared across report cells."""
    values = np.asarray(values, dtype=float)
    indices = np.random.default_rng(0).integers(0, len(values), size=(B, len(values)))
    boots = values[indices].mean(axis=1)
    return [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def flat(tree, prefix=''):
    out = {}
    for key, value in tree.items():
        if isinstance(value, dict):
            out.update(flat(value, f'{prefix}{key}.'))
        else:
            out[prefix + key] = value
    return out


rng = np.random.default_rng(0)
report = {'recorded_local': datetime.now().astimezone().isoformat(), 'standard': __doc__.split('\n\n')[1],
          'runs': {}, 'final_comparison': {}, 'stage2': {}, 'stage2_fairness': {}}

# final comparison
methods = ('ours', 'sincro', 'reviwo', 'cnn')
grid = {}
for method in methods:
    for t, task in enumerate(FINAL_TASKS):
        for s, seed in enumerate(FINAL_SEEDS):
            rid = final_run(method, task, seed)
            rows, digest = eval_rows(rid)
            sc = scores(rows)
            report['runs'][rid] = {'eval_jsonl_sha256': digest, **sc}
            for metric, vals in sc.items():
                for stat, v in vals.items():
                    grid.setdefault((method, metric, stat), np.zeros((len(FINAL_TASKS), len(FINAL_SEEDS))))[t, s] = v
for (method, metric, stat), x in grid.items():
    entry = report['final_comparison'].setdefault(metric, {}).setdefault(stat, {}).setdefault(method, {})
    entry['per_task_mean'] = {task: float(x[t].mean()) for t, task in enumerate(FINAL_TASKS)}
    entry['per_task_seeds'] = {task: x[t].tolist() for t, task in enumerate(FINAL_TASKS)}
    entry['per_task_mean_ci95'] = {task: mean_ci95(x[t]) for t, task in enumerate(FINAL_TASKS)}
    if metric.endswith('/success'):
        entry['iqm'], entry['iqm_ci95'] = iqm_with_ci(x, rng)

# Stage 2 RL
S2_CELLS = [(arm, task) for arm in S2_ARMS for task in S2_TASKS] + [
    (arm, task) for arm in EXTRA_ARMS for task in S2_TASKS
    if all(closed_out(f's2rl-{arm}-{task}-s{seed}') for seed in S2_SEEDS)]
for arm, task in S2_CELLS:
    if True:
        per_seed = {}
        for seed in S2_SEEDS:
            rid = f's2rl-{arm}-{task}-s{seed}'
            rows, digest = eval_rows(rid)
            per_seed[seed] = scores(rows)
            report['runs'][rid] = {'eval_jsonl_sha256': digest, **per_seed[seed]}
        summary = {}
        for metric in per_seed[S2_SEEDS[0]]:
            for stat in STATS:
                vals = [per_seed[s][metric][stat] for s in S2_SEEDS]
                summary.setdefault(metric, {})[stat] = {'mean': float(np.mean(vals)), 'seeds': vals,
                                                       'mean_ci95': mean_ci95(vals)}
        report['stage2'][f'{arm}/{task}'] = summary

# Stage 2 fairness checks
registry = [json.loads(line) for line in (ROOT / 'experiments/registry.jsonl').read_text().splitlines()]
launches = {}
for event in registry:
    if event['event'] == 'launched' and event['id'].startswith('s2rl-'):
        launches.setdefault(event['id'], []).append(event)
fair = report['stage2_fairness']
config_diffs, commits, episodes, gpus, fps, pauses = {}, Counter(), Counter(), {}, {}, {}
for task in S2_TASKS:
    for seed in S2_SEEDS:
        ref = flat(yaml.safe_load((RUNS / f's2rl-full-{task}-s{seed}/config.yaml').read_text()))
        for arm in [a for a, t in S2_CELLS if t == task]:
            rid = f's2rl-{arm}-{task}-s{seed}'
            cfg = flat(yaml.safe_load((RUNS / rid / 'config.yaml').read_text()))
            diff = sorted(k for k in set(cfg) | set(ref) if cfg.get(k) != ref.get(k) and not k.startswith('run.'))
            config_diffs[rid] = diff
            assert cfg['seed'] == seed and cfg['task'] == task, rid
            prov = json.loads((RUNS / rid / 'provenance.json').read_text())
            commits[(prov['git_commit'], prov['host'])] += 1
            rows = [json.loads(line) for line in eval_path(rid).read_text().splitlines() if line.strip()]
            episodes[tuple(sorted({r['episodes'] for r in rows}))] += 1
            gpus[rid] = [e.get('gpu', '')[-12:] for e in launches.get(rid, [])]
            train = [json.loads(line) for line in (RUNS / rid / 'train.jsonl').read_text().splitlines()
                     if line.strip()]
            fps[rid] = float(np.median([r['fps'] for r in train if 'fps' in r]))
            pauses[rid] = [e for e in registry if e['id'] == rid and e['event'] == 'checkpoint_paused']
fair['rl_config_keys_differing_from_full'] = {k: v for k, v in config_diffs.items() if v}
fair['allowed_difference'] = 'vision.export_path (the encoder under test); run.* paths excluded'
fair['unexpected_config_differences'] = {k: [d for d in v if d != 'vision.export_path']
                                         for k, v in config_diffs.items() if any(d != 'vision.export_path' for d in v)}
fair['provenance_commit_host_counts'] = {f'{c} @ {h}': n for (c, h), n in commits.items()}
fair['evaluation_episodes_per_point'] = {str(k): v for k, v in episodes.items()}
fair['gpu_suffix_per_attempt'] = gpus
fair['median_fps'] = fps
fair['checkpoint_pauses'] = {k: [{'step': e.get('checkpoint_step'), 'reason': e.get('reason')} for e in v]
                             for k, v in pauses.items() if v}
encoders = {}
for arm, task in S2_CELLS:
    if True:
        path = yaml.safe_load((RUNS / f's2rl-{arm}-{task}-s1000/config.yaml').read_text())['vision']['export_path']
        import torch  # noqa: E402
        payload = torch.load(ROOT / path, map_location='cpu', weights_only=True)
        encoders[f'{arm}/{task}'] = {'export': path, 'sha256': sha(ROOT / path), 'step': payload['step'],
                                     'frame_strides': payload['frame_strides'],
                                     'encoder_config': payload['encoder_config']}
ref_enc = {t: encoders[f'full/{t}']['encoder_config'] for t in S2_TASKS}
fair['encoder_exports'] = encoders
fair['encoder_config_differences_from_full'] = {
    k: {kk: [ref_enc[k.split('/')[1]].get(kk), v['encoder_config'].get(kk)]
        for kk in set(v['encoder_config']) | set(ref_enc[k.split('/')[1]])
        if v['encoder_config'].get(kk) != ref_enc[k.split('/')[1]].get(kk)}
    for k, v in encoders.items() if not k.startswith('full/')}

out = ROOT / 'runs/setup' / f"rl-report-{datetime.now():%Y%m%d-%H%M%S}.json"
out.write_text(json.dumps(report, indent=1, default=str) + '\n')

# printed summary
print('== Final comparison: IQM [95% CI] of success over 8 tasks x 3 seeds')
for kind in PATHS:
    for stat in STATS:
        cells = []
        for m in methods:
            e = report['final_comparison'][f'{kind}/success'][stat][m]
            cells.append(f"{m} {e['iqm']:.2f} [{e['iqm_ci95'][0]:.2f},{e['iqm_ci95'][1]:.2f}]")
        print(f'{kind:8s} {stat:6s} ' + '  '.join(cells))
print('\n== Final comparison: per-task mean RETURN, training cameras (last5 / final / auc)')
for task in FINAL_TASKS:
    cells = []
    for m in methods:
        e = report['final_comparison']['train/return']
        cells.append(f"{m} {e['last5'][m]['per_task_mean'][task]:6.0f}/{e['final'][m]['per_task_mean'][task]:6.0f}"
                     f"/{e['auc'][m]['per_task_mean'][task]:6.0f}")
    print(f'{task:16s} ' + '  '.join(cells))
print('\n== Stage 2: seed means (last5 / final / auc)')
for key, summary in report['stage2'].items():
    s = summary['train/success']
    r = summary['train/return']
    h = summary['heldout/success']
    j = summary['traj/success']
    print(f"{key:24s} train succ {s['last5']['mean']:.2f}/{s['final']['mean']:.2f}/{s['auc']['mean']:.2f}  "
          f"return {r['last5']['mean']:6.0f}/{r['final']['mean']:6.0f}/{r['auc']['mean']:6.0f}  "
          f"heldout {h['last5']['mean']:.2f}/{h['final']['mean']:.2f}/{h['auc']['mean']:.2f}  "
          f"traj {j['last5']['mean']:.2f}/{j['final']['mean']:.2f}/{j['auc']['mean']:.2f}")
print('\n== Stage 2 fairness')
print('unexpected RL config differences:', fair['unexpected_config_differences'] or 'none')
print('provenance commit@host:', fair['provenance_commit_host_counts'])
print('evaluation episodes per point:', fair['evaluation_episodes_per_point'])
print('encoder config differences from full:', fair['encoder_config_differences_from_full'])
print('checkpoint pauses:', fair['checkpoint_pauses'])
print('median fps by arm:', {arm: round(float(np.median([v for k, v in fps.items() if f'-{arm}-' in k])), 1)
                             for arm in dict.fromkeys(a for a, _ in S2_CELLS)})
print('\nREPORT', out, sha(out))
