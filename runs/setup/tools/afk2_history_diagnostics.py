"""Does future_k2 use its history? Post-hoc history interventions on a future_k2 checkpoint (inference only; user
instruction 2026-10-23: hold the current image, the future actions and the horizon fixed and replace or permute only
the history images; action-shuffle sensitivity alone does not establish history use).

On the validation windows of the run's own future-objective loaders (future strides k2 in --strides), the encoder
history (t-4, t-2, t) is varied in its two HISTORY frames only; the current frame t, the recorded actions[t:t+2*k2], k2,
the targets and every camera stay the same:
- intact:   (t-4, t-2, t), the trained input;
- static:   (t, t, t): both history frames replaced by the current image (no motion visible);
- shuffled: (t-4', t-2', t): the history frames of another validation window of the batch (fixed derangement, seed 0);
- swapped:  (t-2, t-4, t): the two history frames in reverse order (reversed apparent motion);
- persistence (reference): intact encoding, zero displacement.
Reported per condition (training cameras; source camera 0): PSNR, moving-region PSNR (target motion score > 0.5) at t
(the observed current frame, a reconstruction), t+k2 and t+2*k2 (forecasts), SSIM, depth AbsRel and relative 3D EPE
0->1, 1->2, 0->2 (1.0 = no motion predicted), and the encoder state's cosine similarity to the intact state.
    CUDA_VISIBLE_DEVICES=<MIG uuid> nice -n 15 .venv/bin/python -I runs/setup/tools/afk2_history_diagnostics.py \
        --run runs/pretrain/afk2-seed0-hammer-200k --checkpoint .../step_0200000.pt --out <json>
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.join(os.environ.get('S4D_REPO', '/home/ws/ws/splatter4d'), 'scripts'))
from _bootstrap import REPO, guard_gpus  # noqa: E402

GPU_MAPPING = guard_gpus()
import torch  # noqa: E402
from train import build_data  # noqa: E402

from s4d.config import load_config  # noqa: E402
from s4d.train.checkpoint import load_checkpoint  # noqa: E402
from s4d.train.evaluate import encode_states  # noqa: E402
from s4d.train.loop import build_model, forward_losses, move_batch  # noqa: E402
from s4d.train.workers import fork_safe_iter  # noqa: E402

KEYS = ('psnr_t0', 'psnr_t1', 'psnr_t2', 'psnr_moving_t0', 'psnr_moving_t1', 'psnr_moving_t2', 'ssim', 'depth_absrel',
        'rel_epe_01', 'rel_epe_12', 'rel_epe_02')
CONDITIONS = ('intact', 'static', 'shuffled', 'swapped', 'persistence')


def derangement(B, gen):
    perm = torch.randperm(B, generator=gen)
    while B > 1 and bool((perm == torch.arange(B)).any()):
        perm = torch.randperm(B, generator=gen)
    return perm


def with_history(batch, frames):
    """frames: (B,3,…) index tensors of the new history for images and motion scores (current frame unchanged)."""
    out = dict(batch)
    for key in ('history_images', 'history_motion_score'):
        out[key] = frames(batch[key])
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--strides', default='2,4,6')
    ap.add_argument('--max-batches', type=int, default=0, help='0 = every validation window')
    args = ap.parse_args()
    torch.set_num_threads(4)
    run = REPO / args.run
    cfg = load_config([run / 'config.yaml'])
    assert cfg['data'].get('objective') == 'future', 'a future-objective run is required'
    cfg.setdefault('eval', {})['heldout_sets'] = False
    cfg['data']['val_future_strides'] = [int(s) for s in args.strides.split(',')]
    cfg['train']['num_workers'] = 4
    device = torch.device('cuda', 0)
    model = build_model(cfg).to(device).eval()
    step = load_checkpoint(REPO / args.checkpoint, model.encoder, model.decoder, restore_rng=False)
    decoder_forward = model.decoder.forward
    _, val_loaders, _, _ = build_data(cfg)
    gen = torch.Generator().manual_seed(0)
    result = {'run': args.run, 'checkpoint': args.checkpoint, 'step': step, 'gpu': GPU_MAPPING, 'conditions': CONDITIONS,
              'strides': {}}
    t0 = time.time()
    for stride, loader in val_loaders.items():
        sums = {c: defaultdict(float) for c in CONDITIONS}
        counts = {c: defaultdict(float) for c in CONDITIONS}
        cos = defaultdict(float)
        windows = 0
        for i, raw in enumerate(fork_safe_iter(loader)):
            if args.max_batches and i >= args.max_batches:
                break
            batch = move_batch(raw, device)
            B = batch['images'].shape[0]
            perm = derangement(B, gen).to(device)
            variants = {
                'intact': batch,
                'static': with_history(batch, lambda x: x[:, [2, 2, 2]]),
                'shuffled': with_history(batch, lambda x: torch.cat((x[perm][:, :2], x[:, 2:]), 1)),
                'swapped': with_history(batch, lambda x: x[:, [1, 0, 2]]),
                'persistence': batch,
            }
            ref_state = encode_states(model, batch['history_images']).flatten(2)
            for cond, b in variants.items():
                if cond == 'persistence':
                    model.decoder.forward = lambda slots, actions=None, future_stride=None: decoder_forward(slots)
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
                if cond in ('static', 'shuffled', 'swapped'):
                    s = encode_states(model, b['history_images']).flatten(2)
                    cos[cond] += float(torch.nn.functional.cosine_similarity(s, ref_state, dim=-1).mean()) * B
            windows += B
        result['strides'][str(stride)] = {
            'windows': windows,
            **{c: {k: sums[c][k] / counts[c][k] for k in KEYS if counts[c][k]} for c in CONDITIONS},
            'state_cosine_to_intact': {c: v / windows for c, v in cos.items()},
        }
        print(stride, json.dumps({c: {k: round(v, 3) for k, v in result['strides'][str(stride)][c].items()
                                      if k in ('psnr_moving_t2', 'rel_epe_02')} for c in CONDITIONS}), flush=True)
    result['seconds'] = round(time.time() - t0, 1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
