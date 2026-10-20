"""Post-hoc diagnostics of an action_future checkpoint (inference only), for the registered technical gate G3.

On the validation windows of the run's own (future-objective) loaders, each stride separately, the decoder's
displacements are evaluated under four conditions with the SAME encoding and current-time Gaussians:
- correct: the recorded actions[t:t+2k];
- shuffled: the actions of another validation window of the batch (same stride; fixed permutation, seed 0);
- reversed: the valid prefix in reverse order (zeros beyond it unchanged);
- persistence: no displacement (decoder without actions), the trivial static prediction.
Reported per target time t / t+k / t+2k: PSNR, moving-region PSNR (target motion score > 0.5), SSIM, depth AbsRel
(all frames), and the rendered 3D displacement error (relative EPE 0->1, 1->2, 0->2; 1.0 = no motion predicted).

    CUDA_VISIBLE_DEVICES=<MIG uuid> nice -n 15 .venv/bin/python -I runs/setup/tools/actfut_diagnostics.py \
        --run runs/pretrain/af-seed0-hammer-200k --checkpoint .../step_0200000.pt --out <json>
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import os  # noqa: E402
sys.path.insert(0, os.path.join(os.environ.get('S4D_REPO', '/home/ws/ws/splatter4d'), 'scripts'))  # worktree before merge
from _bootstrap import REPO, guard_gpus  # noqa: E402

GPU_MAPPING = guard_gpus()

import torch  # noqa: E402
from train import build_data  # noqa: E402

from s4d.config import load_config  # noqa: E402
from s4d.train.checkpoint import load_checkpoint  # noqa: E402
from s4d.train.loop import build_model, forward_losses, move_batch  # noqa: E402
from s4d.train.workers import fork_safe_iter  # noqa: E402

KEYS = ('psnr_t0', 'psnr_t1', 'psnr_t2', 'psnr_moving_t0', 'psnr_moving_t1', 'psnr_moving_t2', 'ssim', 'depth_absrel',
        'rel_epe_01', 'rel_epe_12', 'rel_epe_02')


def reversed_prefix(actions, count):
    out = actions.clone()
    for b, n in enumerate(count.tolist()):
        out[b, :n] = actions[b, :n].flip(0)
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-batches', type=int, default=0, help='0 = every validation window')
    args = ap.parse_args()
    torch.set_num_threads(4)
    run = REPO / args.run
    cfg = load_config([run / 'config.yaml'])
    cfg.setdefault('eval', {})['heldout_sets'] = False
    cfg['train']['num_workers'] = 4
    device = torch.device('cuda', 0)
    model = build_model(cfg).to(device).eval()
    step = load_checkpoint(REPO / args.checkpoint, model.encoder, model.decoder, restore_rng=False)
    decoder_forward = model.decoder.forward
    _, val_loaders, _, _ = build_data(cfg)
    gen = torch.Generator().manual_seed(0)
    result = {'run': args.run, 'checkpoint': args.checkpoint, 'step': step, 'gpu': GPU_MAPPING, 'strides': {}}
    t0 = time.time()
    for stride, loader in val_loaders.items():
        sums = {c: defaultdict(float) for c in ('correct', 'shuffled', 'reversed', 'persistence')}
        counts = {c: defaultdict(float) for c in sums}
        windows = 0
        for i, raw in enumerate(fork_safe_iter(loader)):
            if args.max_batches and i >= args.max_batches:
                break
            batch = move_batch(raw, device)
            B = batch['images'].shape[0]
            perm = torch.randperm(B, generator=gen)
            if B > 1:
                while bool((perm == torch.arange(B)).any()):
                    perm = torch.randperm(B, generator=gen)
            variants = {
                'correct': batch,
                'shuffled': {**batch, 'actions': batch['actions'][perm.to(device)]},
                'reversed': {**batch, 'actions': reversed_prefix(batch['actions'], batch['action_count'])},
                'persistence': batch,
            }
            for cond, b in variants.items():
                if cond == 'persistence':  # same encoding and current-time Gaussians, zero displacement
                    model.decoder.forward = lambda slots, actions=None, action_count=None: decoder_forward(slots)
                try:
                    out = forward_losses(model, b, cfg, step, source=torch.zeros(B, dtype=torch.long, device=device),
                                         mask_ratio=0.0)
                finally:
                    model.decoder.forward = decoder_forward
                for k in KEYS:
                    v = out['metrics'].get(k)
                    if v is not None and torch.isfinite(v):
                        sums[cond][k] += float(v) * B
                        counts[cond][k] += B
            windows += B
        result['strides'][str(stride)] = {
            'windows': windows,
            **{c: {k: sums[c][k] / counts[c][k] for k in KEYS if counts[c][k]} for c in sums},
        }
        print(stride, json.dumps({c: {k: round(v, 3) for k, v in result['strides'][str(stride)][c].items()
                                      if k in ('psnr_moving_t2', 'rel_epe_02')} for c in sums}), flush=True)
    result['seconds'] = round(time.time() - t0, 1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
