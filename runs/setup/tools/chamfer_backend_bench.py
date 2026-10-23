"""Chamfer nearest-neighbour backends at realistic training shapes (2026-10-23): the fused Triton kernel vs the exact
chunked fallback (used where Triton cannot build, e.g. an image without a C compiler).

1. Agreement and kernel cost at the training shapes: decoded centres (B*T = 48 clouds x 8192 Gaussians) vs fused
   training-camera depth (48 x 4096 points), both directions: identical nearest indices (fraction), max |d2| difference,
   chamfer_loss value and the gradient w.r.t. the centres (max abs and relative difference), time per call and peak memory.
2. End to end: forward + backward of forward_losses on real training batches (batch 16, the run's own config) for each
   config given (chamfer_lean and perframe_temporal by default), with each backend: median step time over ``--steps``
   steps after warm-up, peak GPU memory, and the agreement of the total loss and of the encoder / decoder gradients on
   identical seeded batches.
    CUDA_VISIBLE_DEVICES=<uuid> .venv/bin/python -I runs/setup/tools/chamfer_backend_bench.py --out <json>
"""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.environ.get('S4D_REPO', '/home/ws/ws/splatter4d'), 'scripts'))
from _bootstrap import REPO, guard_gpus  # noqa: E402

GPU_MAPPING = guard_gpus()
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from train import build_data  # noqa: E402

from s4d.config import load_config  # noqa: E402
from s4d.data.contract import collate  # noqa: E402
from s4d.losses import chamfer as cm  # noqa: E402
from s4d.train.loop import build_model, forward_losses, move_batch  # noqa: E402

CONFIGS = {
    'chamfer_lean': ['configs/metaworld/base.yaml', 'configs/metaworld/tasks/hammer.yaml',
                     'configs/metaworld/variants/chamfer_lean.yaml'],
    'perframe_temporal': ['configs/metaworld/base.yaml', 'configs/metaworld/tasks/hammer.yaml',
                          'configs/metaworld/variants/chamfer_lean.yaml', 'configs/metaworld/variants/perframe_temporal.yaml'],
}


def set_backend(fused, device):
    cm._FUSED_OK[str(device)] = fused


def timed(fn, reps=10):
    torch.cuda.synchronize()
    out, times = None, []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        out = fn()
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b))
    return out, statistics.median(times[2:]) / 1000.0


def kernel_part(device):
    g = torch.Generator(device=device).manual_seed(0)
    centres = (torch.rand(16, 3, 8192, 3, device=device, generator=g) - 0.5) * torch.tensor([0.8, 0.8, 0.4], device=device)
    targets = (torch.rand(16, 3, 4096, 3, device=device, generator=g) - 0.5) * torch.tensor([0.8, 0.8, 0.4], device=device)
    valid = torch.ones(16, 3, dtype=torch.bool, device=device)
    res = {}
    nn = {}
    for name, fused in (('fused', True), ('fallback', False)):
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        (d1, i1), t1 = timed(lambda: cm.nearest_neighbours(centres.flatten(0, 1), targets.flatten(0, 1), fused=fused))
        (d2, i2), t2 = timed(lambda: cm.nearest_neighbours(targets.flatten(0, 1), centres.flatten(0, 1), fused=fused))
        set_backend(fused, device)
        x = centres.clone().requires_grad_(True)

        def loss_and_grad():
            x.grad = None
            out = cm.chamfer_loss(x, targets, valid)
            out['loss'].sum().backward()
            return out['loss'].detach().clone(), x.grad.detach().clone()

        (loss, grad), t3 = timed(loss_and_grad, reps=6)
        nn[name] = (d1, i1, d2, i2, loss, grad)
        res[name] = {'nn_pred_to_target_s': t1, 'nn_target_to_pred_s': t2, 'chamfer_loss_fwd_bwd_s': t3,
                     'peak_mem_gb': (torch.cuda.max_memory_allocated() - base) / 2**30}
    a, b = nn['fused'], nn['fallback']
    res['agreement'] = {
        'index_equal_fraction_pred_to_target': float((a[1] == b[1]).float().mean()),
        'index_equal_fraction_target_to_pred': float((a[3] == b[3]).float().mean()),
        'max_abs_d2_diff': float(max((a[0] - b[0]).abs().max(), (a[2] - b[2]).abs().max())),
        'loss_max_rel_diff': float(((a[4] - b[4]).abs() / a[4].abs().clamp_min(1e-12)).max()),
        'grad_max_abs_diff': float((a[5] - b[5]).abs().max()),
        'grad_rel_diff': float((a[5] - b[5]).norm() / a[5].norm().clamp_min(1e-12)),
    }
    return res


def end_to_end(name, files, device, steps):
    cfg = load_config([REPO / f for f in files], ['train.num_workers=2'])
    train_ds = build_data(cfg)[0]
    loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2, collate_fn=collate,
                        generator=torch.Generator().manual_seed(7))
    batches = []
    for i, b in enumerate(loader):
        if i >= steps + 3:
            break
        batches.append(b)
    torch.manual_seed(0)
    model = build_model(cfg).to(device).train()
    state = {k: v.clone() for k, v in model.state_dict().items()}
    out = {}
    grads = {}
    pairs = (('fused', True), ('fallback', True)) if SAME else (('fused', True), ('fallback', False))
    for backend, fused in pairs:  # --same: fused twice (run-to-run nondeterminism baseline)
        set_backend(fused, device)
        model.load_state_dict(state)
        torch.cuda.reset_peak_memory_stats()
        times, totals = [], []
        for i, raw in enumerate(batches):
            batch = move_batch(raw, device)
            torch.manual_seed(100 + i)
            source = torch.zeros(16, dtype=torch.long, device=device)
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            t0 = time.time()
            res = forward_losses(model, batch, cfg, 30000, source=source)
            res['total'].backward()
            torch.cuda.synchronize()
            times.append(time.time() - t0)
            totals.append(float(res['total']))
            if i == 0:
                grads[backend] = {part: torch.cat([p.grad.detach().reshape(-1).float() for p in getattr(model, part).parameters()
                                                   if p.grad is not None]) for part in ('encoder', 'decoder')}
        out[backend] = {'median_step_s': statistics.median(times[3:]), 'peak_mem_gb': torch.cuda.max_memory_allocated() / 2**30,
                        'totals': totals}
    out['agreement'] = {
        'total_max_rel_diff': max(abs(a - b) / max(abs(a), 1e-12) for a, b in zip(out['fused']['totals'], out['fallback']['totals'])),
        **{f'{part}_grad_rel_diff': float((grads['fused'][part] - grads['fallback'][part]).norm()
                                          / grads['fused'][part].norm().clamp_min(1e-12)) for part in ('encoder', 'decoder')},
    }
    out['slowdown'] = out['fallback']['median_step_s'] / out['fused']['median_step_s']
    return out


SAME = '--same' in sys.argv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--steps', type=int, default=12)
    ap.add_argument('--configs', default='chamfer_lean,perframe_temporal')
    ap.add_argument('--same', action='store_true')
    args = ap.parse_args()
    device = torch.device('cuda', 0)
    assert cm.fused_available(device), 'the comparison needs the fused kernel on this device'
    result = {'gpu': GPU_MAPPING, 'kernel_shapes': {'clouds': 48, 'gaussians': 8192, 'target_points': 4096},
              'kernel': None if SAME else kernel_part(device), 'end_to_end': {}, 'same_backend_baseline': SAME}
    print(json.dumps(result['kernel'], indent=1), flush=True)
    for name in args.configs.split(','):
        result['end_to_end'][name] = end_to_end(name, CONFIGS[name], device, args.steps)
        r = result['end_to_end'][name]
        print(name, {k: (round(v['median_step_s'], 3), round(v['peak_mem_gb'], 2)) for k, v in r.items() if k in ('fused', 'fallback')},
              r['agreement'], round(r['slowdown'], 2), flush=True)
    cm._FUSED_OK.pop(str(device), None)
    Path(args.out).write_text(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
