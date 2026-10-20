"""Gate G1 on real data and a real checkpoint of variant action_future (inference + one backward pass; no update).

For each stride k of the run's validation loaders (future objective), on one validation batch:
1. alignment: target indices (t, t+k, t+2k), history (t-2k, t-k, t), actions == file actions[t:t+2k], zeros beyond;
2. current-time Gaussians identical when the future actions change (random actions, same counts);
3. first displacement identical when only second-interval actions change;
4. outputs identical when values beyond each sample's valid prefix change (decoder called directly);
5. encoder states identical when target-side fields (images, depth, motion, score) and actions change;
6. one training-mode backward pass: finite loss and gradients; non-zero gradient norms of the encoder and the
   action branch (prefix MLP and both displacement heads).

    CUDA_VISIBLE_DEVICES=<MIG uuid> nice -n 15 .venv/bin/python -I runs/setup/tools/actfut_checks.py \
        --run <run dir> --checkpoint <ckpt> --out <json>
"""
import argparse
import json
import sys
from pathlib import Path

import os  # noqa: E402
sys.path.insert(0, os.path.join(os.environ.get('S4D_REPO', '/home/ws/ws/splatter4d'), 'scripts'))  # worktree before merge
from _bootstrap import REPO, guard_gpus  # noqa: E402

GPU_MAPPING = guard_gpus()

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from train import build_data  # noqa: E402

from s4d.config import get, load_config  # noqa: E402
from s4d.train.checkpoint import load_checkpoint  # noqa: E402
from s4d.train.loop import build_model, forward_losses, move_batch  # noqa: E402
from s4d.train.workers import fork_safe_iter  # noqa: E402

STATIC = ('xyz', 'scales', 'quats', 'opacity', 'rgb')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    run = Path(args.run) if Path(args.run).is_absolute() else REPO / args.run
    cfg = load_config([run / 'config.yaml'])
    cfg.setdefault('eval', {})['heldout_sets'] = False
    cfg['train']['num_workers'] = 2
    cfg['data']['val_strides'] = [2, 4, 6]
    device = torch.device('cuda', 0)
    model = build_model(cfg).to(device)
    ckpt = Path(args.checkpoint) if Path(args.checkpoint).is_absolute() else REPO / args.checkpoint
    step = load_checkpoint(ckpt, model.encoder, model.decoder, restore_rng=False)
    _, val_loaders, _, _ = build_data(cfg)
    h5 = h5py.File(Path(get(cfg, 'data.root')) / f"{get(cfg, 'data.task')}.hdf5", 'r')
    report = {'run': str(run), 'checkpoint': str(ckpt), 'step': step, 'gpu': GPU_MAPPING, 'strides': {}}
    gen = torch.Generator(device='cpu').manual_seed(0)
    for k, loader in val_loaders.items():
        batch = move_batch(next(iter(fork_safe_iter(loader))), device)
        B = batch['images'].shape[0]
        res = {}
        # 1. alignment against the file
        ok = True
        for b in range(B):
            ep, tgt, hist = batch['meta']['episode'][b], batch['meta']['t_indices'][b], batch['meta']['history_t_indices'][b]
            t = tgt[0]
            ok &= list(tgt) == [t, t + k, t + 2 * k] and list(hist) == [t - 2 * k, t - k, t]
            n = int(batch['action_count'][b])
            ok &= n == 2 * k
            ok &= bool(np.array_equal(batch['actions'][b, :n].cpu().numpy(), h5['episodes'][ep]['action'][t:t + n]))
            ok &= not bool(batch['actions'][b, n:].any())
            ok &= bool(np.array_equal(batch['history_images'][b, :, 0].permute(0, 2, 3, 1).cpu().numpy(),
                                      h5['episodes'][ep]['rgb'][hist[0]:hist[2] + 1:k, 0]))
        res['alignment'] = bool(ok)
        model.eval()
        with torch.no_grad():
            enc = model.encoder((batch['history_images'][:, :, 0].float() / 255), None, mask_ratio=0.0)['slots']
            count = batch['action_count']
            rand = lambda: torch.rand(batch['actions'].shape, generator=gen).to(device) * 2 - 1  # noqa: E731
            valid = (torch.arange(12, device=device)[None] < count[:, None]).float()[..., None]
            a0 = batch['actions']
            a1 = rand() * valid
            g0, g1 = model.decoder(enc, a0, count), model.decoder(enc, a1, count)
            res['current_time_independent_of_actions'] = all(torch.equal(getattr(g0, f), getattr(g1, f)) for f in STATIC)
            res['displacement_changes_with_actions'] = not torch.equal(g0.delta12, g1.delta12)
            second = (torch.arange(12, device=device)[None] >= (count // 2)[:, None]).float()[..., None] * valid
            a2 = a0 * (1 - second) + rand() * second
            g2 = model.decoder(enc, a2, count)
            res['first_displacement_independent_of_second_interval'] = torch.equal(g0.delta01, g2.delta01)
            a3 = a0 + (1 - valid) * 9.0  # forbidden values beyond the prefix
            g3 = model.decoder(enc, a3, count)
            res['padding_invariant'] = all(torch.equal(getattr(g0, f), getattr(g3, f)) for f in (*STATIC, 'delta01', 'delta12'))
            other = {**batch, 'images': torch.roll(batch['images'], 1, 0), 'depth': batch['depth'].flip(0),
                     'motion3d': batch['motion3d'] * 2, 'motion_score': batch['motion_score'].flip(0), 'actions': a1}
            src = torch.zeros(B, dtype=torch.long, device=device)
            o0 = forward_losses(model, batch, cfg, step, source=src, mask_ratio=0.0)
            o1 = forward_losses(model, other, cfg, step, source=src, mask_ratio=0.0)
            res['encoder_independent_of_targets_and_actions'] = torch.equal(o0['slots'], o1['slots'])
        model.train()
        model.zero_grad(set_to_none=True)
        out = forward_losses(model, batch, cfg, step, source=src)
        out['total'].backward()
        norms = {}
        for name, module in (('encoder', model.encoder), ('action_encoder', model.decoder.action_encoder),
                             ('motion01', model.decoder.groups[1].motion01), ('motion12', model.decoder.groups[1].motion12)):
            grads = [p.grad for p in module.parameters() if p.grad is not None]
            norms[name] = float(torch.sqrt(sum((g.float() ** 2).sum() for g in grads))) if grads else 0.0
        res['loss_finite'] = bool(torch.isfinite(out['total']))
        res['grad_norms'] = norms
        res['grads_finite_nonzero'] = all(np.isfinite(v) and v > 0 for v in norms.values())
        res['passed'] = all(v for kk, v in res.items() if isinstance(v, bool))
        report['strides'][str(k)] = res
        print(k, json.dumps(res), flush=True)
    report['passed'] = all(r['passed'] for r in report['strides'].values())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    print('PASSED' if report['passed'] else 'FAILED')


if __name__ == '__main__':
    main()
