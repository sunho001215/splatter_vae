"""Investigation of the loss spike in afk2-seed0-bin-picking-200k (user request 2026-10-21). Read-only on the run.

    CUDA_VISIBLE_DEVICES=<MIG> python -I afk2_binpick_spike.py batches [--from 11900 --to 12100]
        Reconstructs the exact training batches of the given steps (the loader's deterministic order: a
        DistributedSampler permutation with seed train.seed + epoch, batch 16, drop_last) and validates each sample:
        finite tensors, depth range, camera matrices, actions (finite, magnitude, zero beyond 2*k2), the target/history
        frame indices against the episode length, and the motion targets. Writes batches-<from>-<to>.json.
    CUDA_VISIBLE_DEVICES=<MIG> python -I afk2_binpick_spike.py checkpoints --steps 10000 20000 ...
        Decodes a fixed set of training batches (the steps 11950-12000 of the spike and a reference block) and the first
        validation batches with each checkpoint (inference only) and records Gaussian statistics (centres per camera:
        behind the camera, inside near, outside the frustum; the visibility violation split into its terms, and its
        largest single-point contribution), opacity and scale distributions, and the slot embeddings' similarity.
"""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path('/home/ws/ws/splatter4d')
RUN = ROOT / 'runs/pretrain/afk2-seed0-bin-picking-200k'
OUT = ROOT / 'runs/setup/afk2-binpick-spike-20261021'
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from torch.utils.data.distributed import DistributedSampler  # noqa: E402

from train import build_data  # noqa: E402  (scripts/train.py; GPU guard on import)

from s4d.config import load_config  # noqa: E402
from s4d.config import get  # noqa: E402


def config():
    return load_config([str(RUN / 'config.yaml')])


def step_indices(train_ds, cfg, steps):
    """Dataset indices of every training step in ``steps`` (the batch consumed by forward step s, 0-based)."""
    batch = int(get(cfg, 'train.batch_size', 16))
    seed = int(get(cfg, 'train.seed', 0))
    per_epoch = len(train_ds) // batch
    out, cache = {}, {}
    for s in steps:
        epoch, pos = divmod(s, per_epoch)
        if epoch not in cache:
            sampler = DistributedSampler(train_ds, num_replicas=1, rank=0, shuffle=True, seed=seed, drop_last=True)
            sampler.set_epoch(epoch)
            cache[epoch] = list(iter(sampler))
        out[s] = cache[epoch][pos * batch:(pos + 1) * batch]
    return out, per_epoch


def check_sample(ds, index, sample):
    ep, t, (k2,) = ds.samples[index]
    length = int(ds._h5()['episodes'][ep].attrs['length'])
    issues = []
    for key, value in sample.items():
        if torch.is_tensor(value) and value.is_floating_point() and not torch.isfinite(value).all():
            issues.append(f'non-finite {key}')
    depth = sample['depth']
    if (depth < 0).any():
        issues.append('negative depth')
    acts = sample['actions']
    if (acts[2 * k2:].abs().sum() > 0):
        issues.append('actions beyond 2*k2')
    target = [t + j * k2 for j in range(3)]
    if max(target) > length - 1 or t - 4 < 0:
        issues.append(f'frame index outside the episode (length {length})')
    for key in ('w2c', 'c2w'):
        m = sample[key]
        if not torch.allclose(m[:, 3], torch.tensor([0., 0., 0., 1.]).expand_as(m[:, 3])):
            issues.append(f'{key} last row')
    eye = sample['w2c'] @ sample['c2w']
    if not torch.allclose(eye, torch.eye(4).expand_as(eye), atol=1e-4):
        issues.append('w2c is not the inverse of c2w')
    return {
        'index': int(index), 'episode': ep, 't': int(t), 'k2': int(k2), 'length': length, 'issues': issues,
        'depth_valid_fraction': float((depth > 0).float().mean()), 'depth_max': float(depth.max()),
        'action_abs_max': float(acts[:2 * k2].abs().max()), 'action_norm_mean': float(acts[:2 * k2].norm(dim=-1).mean()),
        'motion3d_max_m': float(sample['motion3d'].abs().max()),
        'motion_score_mean': float(sample['motion_score'].float().mean()) if 'motion_score' in sample else None,
    }


def cmd_batches(args):
    cfg = config()
    train_ds, _, _, _ = build_data(cfg)
    steps = list(range(args.start, args.end))
    idx, per_epoch = step_indices(train_ds, cfg, steps)
    report = {'run': str(RUN), 'dataset_len': len(train_ds), 'steps_per_epoch': per_epoch, 'steps': {}}
    for s in steps:
        rows = [check_sample(train_ds, i, train_ds[i]) for i in idx[s]]
        report['steps'][s] = {
            'epoch': s // per_epoch, 'k2': [r['k2'] for r in rows], 'episodes': [r['episode'] for r in rows],
            't': [r['t'] for r in rows], 'issues': sorted({i for r in rows for i in r['issues']}),
            'action_abs_max': max(r['action_abs_max'] for r in rows),
            'motion3d_max_m': max(r['motion3d_max_m'] for r in rows),
            'depth_valid_fraction_min': min(r['depth_valid_fraction'] for r in rows),
            'samples': rows,
        }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f'batches-{args.start}-{args.end}.json'
    path.write_text(json.dumps(report, indent=1) + '\n')
    flagged = {s: v['issues'] for s, v in report['steps'].items() if v['issues']}
    print(json.dumps({'dataset_len': len(train_ds), 'steps_per_epoch': per_epoch,
                      'epoch_of_first_step': steps[0] // per_epoch, 'flagged_steps': flagged}, indent=1))
    for s in steps:
        v = report['steps'][s]
        if args.verbose or s >= args.focus_from:
            print(s, 'k2', v['k2'], 'act', round(v['action_abs_max'], 3), 'motion', round(v['motion3d_max_m'], 3),
                  'depth_valid_min', round(v['depth_valid_fraction_min'], 3), v['issues'])
    print(path)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('batches')
    b.add_argument('--from', dest='start', type=int, default=11900)
    b.add_argument('--to', dest='end', type=int, default=12100)
    b.add_argument('--focus-from', type=int, default=11990)
    b.add_argument('--verbose', action='store_true')
    args = ap.parse_args()
    if args.cmd == 'batches':
        cmd_batches(args)


if __name__ == '__main__':
    main()
