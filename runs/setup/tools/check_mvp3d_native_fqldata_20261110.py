"""Read-only, CPU A1 / MVP-STATUS V1 a-c check of native 3D-MVP fqldata.

Usage: python -I -B check_mvp3d_native_fqldata_20261110.py TASK CONDITION
       [--dataset PATH] [--split PATH] [--out JSON]

Fit exactly the trainer's OfflineTransitions(TRAIN).features standardisation.
Report coordinator statistics on standardised TRAIN and VALIDATION, with seed 0
and at most 20 sampled rows per timeline. Only PASS prints a proposed pause-lift
patch. This tool NEVER executes that patch or writes any admission pause file.
Exit codes: 0 PASS, 1 FAIL, 2 REFUSED (invalid/incomplete input or missing API).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import h5py  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path('/home/ws/ws/splatter4d')
INPUTS = Path('/home/ws/ws/realworld-ops/calib-loma/ops/results/learning_inputs')
TASKS = ('drawer-open', 'mug-hanging')
CONDITIONS = ('mvp3d_single', 'mvp3d_multiview')
PAUSE_NAME = 'mvp3d-native-fql-feature-check-20261110'


class Refused(RuntimeError):
    pass


def require(ok, why):
    if not ok:
        raise Refused(why)


def trainer_api():
    sys.path.insert(0, str(ROOT))
    try:
        from s4d.rwlearn.fql_data import OfflineTransitions, fit_feature_standardization
    except ImportError as exc:
        raise Refused(f'required trainer API missing in {ROOT}; install the native feature-standardisation '
                      'release before checking (no substitute fit is used)') from exc
    return OfflineTransitions, fit_feature_standardization


def check(path, episodes, mean, std):
    """Same accumulation, timeline sampling, Xavier projection and LayerNorm as featcheck_norm.py."""
    rng = np.random.default_rng(0)
    n, s1, s2, sub = 0, None, None, []
    with h5py.File(path, 'r') as f:
        for ep in episodes:
            for _, g in f['episodes'][ep].items():
                x = g['features'][:].astype(np.float64)
                require(x.ndim == 2 and len(x) > 0 and x.shape[1] == len(mean), f'invalid features: {ep}/{g.name}')
                require(np.isfinite(x).all(), f'nonfinite features: {ep}/{g.name}')
                x = (x - mean) / std
                require(np.isfinite(x).all(), 'nonfinite standardised features')
                if s1 is None:
                    s1, s2 = np.zeros(x.shape[1]), np.zeros(x.shape[1])
                n += len(x)
                s1 += x.sum(0)
                s2 += (x ** 2).sum(0)
                sub.append(x[rng.choice(len(x), min(len(x), 20), replace=False)])
    require(n > 0, 'empty feature split')
    mu = s1 / n
    sd = np.sqrt(np.maximum(s2 / n - mu ** 2, 0))
    sub = np.concatenate(sub)
    W = rng.uniform(-1, 1, (256, sub.shape[1])) * np.sqrt(6 / (sub.shape[1] + 256))
    H = sub @ W.T
    hstd = H.std(1, keepdims=True)
    signal = float(np.sqrt((sd ** 2).sum()))
    offset = float(np.linalg.norm(mu) / signal) if signal > 0 else None
    # Undefined/constant projections fail closed, with JSON null instead of NaN.
    ln_std = (float(((H - H.mean(1, keepdims=True)) / hstd).std(0).mean())
              if np.all(hstd > 0) else None)
    return {'rows': int(n), 'dim': int(sub.shape[1]), 'sampled_rows': int(len(sub)),
            'offset_to_signal': offset, 'ln_unit_std': ln_std,
            'per_dim_sd_median': float(np.median(sd)), 'mean_abs_median': float(np.median(np.abs(mu)))}


def split_pass(stats, offset_limit):
    offset, ln_std = stats['offset_to_signal'], stats['ln_unit_std']
    return bool(offset is not None and ln_std is not None and math.isfinite(offset) and math.isfinite(ln_std)
                and offset <= offset_limit and ln_std >= 0.5)


def verdict(train, validation):
    checks = {'a_train': split_pass(train, 0.01), 'b_validation': split_pass(validation, 1.0),
              'c_coordinator_both': split_pass(train, 10.0) and split_pass(validation, 10.0)}
    return checks, all(checks.values())


def lift_proposal(task, condition, train, validation):
    prefix = f'rwcal-20261110-fql-{condition}-{task}-'
    reason = (f'A1 / MVP-STATUS V1 a-c PASS: {condition}/{task} native_realsense; train '
              f'offset_to_signal={train["offset_to_signal"]:.9g} <= 0.01, '
              f'LN unit std={train["ln_unit_std"]:.9g} >= 0.5; validation '
              f'offset_to_signal={validation["offset_to_signal"]:.9g} <= 1.0, '
              f'LN unit std={validation["ln_unit_std"]:.9g} >= 0.5; '
              'train-split trainer standardisation, seed 0, 20 rows/timeline')
    # A complete patch for the registrar's existing lock + atomic_write pattern.
    # The coordinator can paste this into that module's context; it is text only.
    patch = '\n'.join([
        '# Coordinator patch: locks/require/atomic_write/json_bytes/ROOT from register_mvp3d_native_20261110.py',
        'with locks(ROOT):',
        "    path = ROOT / 'runs/setup/admission-pauses.json'",
        '    doc = json.loads(path.read_bytes())',
        f'    matches = [p for p in doc["pauses"] if p["name"] == {PAUSE_NAME!r}]',
        '    require(len(matches) == 1, "expected one native FQL pause")',
        '    pause = matches[0]',
        f'    require(pause.get("active") is True and {prefix!r} in pause["prefixes"], "cell not held")',
        '    require(not pause.get("ids"), "unexpected explicit ids; inspect before lifting")',
        f'    pause["prefixes"].remove({prefix!r})',
        '    pause["active"] = bool(pause["prefixes"])',
        f'    pause["reason"] += " | " + {reason!r}',
        '    atomic_write(path, json_bytes(doc))',
    ])
    return {'name': PAUSE_NAME, 'remove_prefix': prefix, 'append_reason': reason,
            'active_after': 'true while prefixes remain; false when the last prefix is removed',
            'inactive_when_last_prefix_removed': True, 'python_patch': patch}


def evaluate(task, condition, dataset, split_path):
    require(task in TASKS and condition in CONDITIONS, 'unknown native task/condition')
    split = json.loads(split_path.read_bytes())
    require(isinstance(split, dict), 'invalid split document')
    for name in ('train', 'validation'):
        require(isinstance(split.get(name), list) and split[name]
                and all(isinstance(ep, str) and ep for ep in split[name]), f'need a nonempty {name} split')
    episodes = split['train'] + split['validation']
    require(len(set(episodes)) == len(episodes), 'train/validation episodes overlap or repeat')
    with h5py.File(dataset, 'r') as f:
        depth = f.attrs.get('training_depth')
        if isinstance(depth, bytes):
            depth = depth.decode()
        require(depth == 'native_realsense', 'dataset attrs training_depth must equal native_realsense')
        require(f.attrs.get('condition') == condition, 'dataset condition differs from requested cell')
        require(all(ep in f['episodes'] for ep in episodes), 'split episode missing from finished dataset')
        for ep in episodes:
            require(len(f['episodes'][ep]) > 0, f'episode has no timelines: {ep}')
            for g in f['episodes'][ep].values():
                require(g.attrs.get('task') == task, f'dataset task differs from requested cell: {ep}')
    OfflineTransitions, fit_feature_standardization = trainer_api()
    train = OfflineTransitions(dataset, split['train'], 'cpu')
    require(train.size > 0, 'training split has no transitions')
    stats = fit_feature_standardization(train.features)
    del train
    mean, std = stats['mean'].astype(np.float64), stats['std'].astype(np.float64)
    require(mean.ndim == std.ndim == 1 and mean.shape == std.shape and np.isfinite(mean).all()
            and np.isfinite(std).all() and np.all(std > 0), 'invalid trainer standardisation statistics')
    a = check(dataset, split['train'], mean, std)
    b = check(dataset, split['validation'], mean, std)
    checks, passed = verdict(a, b)
    result = {'schema': 'mvp3d-native-fqldata-check/20261110', 'dry_run': True,
              'task': task, 'condition': condition, 'dataset': str(dataset), 'split': str(split_path),
              'training_depth': depth, 'fit': {k: v for k, v in stats.items() if k not in ('mean', 'std')},
              'fit_source': 's4d.rwlearn.fql_data.OfflineTransitions(TRAIN).features -> fit_feature_standardization',
              'std_min': float(std.min()), 'std_max': float(std.max()),
              'sampling': {'seed': 0, 'rows_per_timeline': 20, 'projection_units': 256},
              'norm_train': a, 'norm_validation': b, 'V1_pass': checks, 'pass': passed,
              'status': 'PASS' if passed else 'FAIL'}
    if passed:
        result['pause_lift'] = lift_proposal(task, condition, a, b)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', choices=TASKS)
    parser.add_argument('condition', choices=CONDITIONS)
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--split', type=Path)
    parser.add_argument('--out', type=Path, help='optional JSON report only; never a pause-file write')
    args = parser.parse_args(argv)
    dataset = args.dataset or ROOT / f'runs/rwcal-20261110-fqldata-{args.condition}-{args.task}-native/dataset.h5'
    split = args.split or INPUTS / args.task / 'split_seed0.json'
    if args.out:
        require(args.out.resolve().name not in {'queue.yaml', 'registry.jsonl', 'admission-pauses.json'}
                and args.out.resolve() not in {dataset.resolve(), split.resolve(),
                                               (ROOT / 'experiments/queue.yaml').resolve(),
                                               (ROOT / 'experiments/registry.jsonl').resolve(),
                                               (ROOT / 'runs/setup/admission-pauses.json').resolve()},
                '--out must be a report, not an input, queue, registry or pause file')
    result = evaluate(args.task, args.condition, dataset, split)
    raw = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.out:
        args.out.write_text(raw)
    print(raw, end='')
    return 0 if result['pass'] else 1


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (Refused, OSError, ValueError, RuntimeError, KeyError, TypeError, ImportError) as exc:
        print(f'REFUSED: {exc}', file=sys.stderr)
        raise SystemExit(2) from exc
