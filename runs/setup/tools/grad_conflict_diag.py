"""Loss-gradient diagnostics on a small fixed set of training batches (2026-10-23; inference + backward, no update).

For each model (a training checkpoint with its own run config) and the same fixed training batches (seeded loader,
``--batches`` x 16 windows; per-batch seeds fix the source camera, masks, drop path, synthetic cameras and sampled
points identically for every model), one training-mode forward at ``--step`` (temporal ramp 1) computes every loss term of
the union of the compared objectives:
  reconstruction: rgb (L1 + SSIM), coverage, depth_l1 (expected depth), synth_render (synthetic views)
  geometry:       chamfer (chamfer_lean), depth_grad, depth_hard, visibility (the reference terms chamfer_lean removed)
  invariance:     infonce, consistency
  motion:         motion (M3D)
each with the weight of the reference (geometry terms) or of chamfer_lean (chamfer, weight 5), exactly as it enters
the total; a model's ACTIVE terms are those of its own configuration, the others are counterfactual (computed on the same
forward, never trained by that model) and account for the removed depth / visibility terms as confounds. For every term
the gradient is taken w.r.t. (a) the encoder parameters, (b) the decoder parameters and (c) the encoder's state
(the slots, i.e. the representation the policy reads); reported per model: gradient norms, pairwise cosine similarities
(mean over batches), the fraction of batches with negative cosine, and cosines between the summed groups
(reconstruction, geometry, invariance, motion) using active terms only.
    CUDA_VISIBLE_DEVICES=<MIG uuid> nice -n 15 .venv/bin/python -I runs/setup/tools/grad_conflict_diag.py --task hammer \
        --model original=runs/pretrain/i2-m3d-seed0-hammer/checkpoints/step_0200000.pt \
        --model chamfer_lean=runs/pretrain/chl-seed0-hammer-200k/checkpoints/step_0200000.pt --out <json>
"""
import argparse
import copy
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
from torch.utils.data import DataLoader  # noqa: E402
from train import build_data  # noqa: E402

from s4d.config import load_config  # noqa: E402
from s4d.data.contract import collate  # noqa: E402
from s4d.train.checkpoint import load_checkpoint  # noqa: E402
from s4d.train.loop import build_model, forward_losses, move_batch  # noqa: E402

GROUPS = {'reconstruction': ('rgb', 'coverage', 'depth_l1', 'synth_render'),
          'geometry': ('chamfer', 'depth_grad', 'depth_hard', 'visibility'),
          'invariance': ('infonce', 'consistency'), 'motion': ('motion',)}
TERMS = [t for g in GROUPS.values() for t in g]
REFERENCE_WEIGHTS = {'depth_grad': 0.5, 'depth_hard': 0.5, 'visibility': 1.0, 'chamfer': 5.0}


def union_config(cfg):
    """The model's config with every term of the union enabled (counterfactual terms at the reference / chamfer_lean
    weights); returns (union cfg, active term set of the model's own config)."""
    loss = cfg['loss']
    active = {'rgb', 'coverage', 'depth_l1', 'infonce', 'consistency', 'motion'}
    if (cfg.get('aug') or {}).get('synth_render'):
        active.add('synth_render')
    for term in ('chamfer', 'depth_grad', 'depth_hard', 'visibility'):
        if float(loss.get(term, {'depth_grad': 0.5, 'depth_hard': 0.5, 'visibility': 1.0}.get(term, 0.0))) > 0:
            active.add(term)
    union = copy.deepcopy(cfg)
    for term, weight in REFERENCE_WEIGHTS.items():
        if term not in active:
            union['loss'][term] = weight
    union['loss'].setdefault('chamfer_points', 4096)
    union['loss'].setdefault('chamfer_voxel_m', 0.01)
    union['loss']['motion_feature_render'] = 'log_steps'
    return union, active


def weighted_terms(out, cfg, ramp=1.0):
    """Undetached weighted terms whose sum is the total (``out['terms']`` captured from the forward)."""
    loss, aug, T = cfg['loss'], cfg.get('aug') or {}, out['terms']
    mean_t = lambda key: torch.stack([T[f'{key}_t{t}'] for t in range(3)]).mean()  # noqa: E731 (ramp 1: mean over t)
    w = {'rgb': float(loss.get('rgb', 1.0)), 'coverage': float(loss.get('coverage', 0.1)),
         'depth_l1': float(loss.get('depth_l1', 1.0)), 'depth_grad': float(loss.get('depth_grad', 0.5)),
         'depth_hard': float(loss.get('depth_hard', 0.5)), 'chamfer': float(loss.get('chamfer', 0.0))}
    terms = {k: w[k] * mean_t(k) for k in w if any(f'{k}_t0' == key for key in T)}
    terms['motion'] = ramp * float(loss.get('motion', 1.0)) * T['motion']
    terms['infonce'] = float(loss.get('invariance', 1.0)) * T['infonce']
    terms['consistency'] = float(loss.get('consistency', 0.5)) * T['consistency']
    if 'visibility' in T:
        terms['visibility'] = float(loss.get('visibility', 1.0)) * T['visibility']
    if 'synth_render' in T:
        terms['synth_render'] = float(aug.get('synth_render_weight', 1.0)) * T['synth_render']
    return terms


def forward_with_terms(model, batch, cfg, step, source):
    """forward_losses, also returning the undetached tensors behind every entry of out['losses'] (matched by storage:
    a detached tensor shares its original's data pointer)."""
    captured = []
    original_detach = torch.Tensor.detach

    def recording(self):
        out = original_detach(self)
        if self.requires_grad:
            captured.append(self)
        return out

    slots_box = {}
    encoder = model.encoder
    original_forward = encoder.forward

    def encoder_forward(*a, **k):
        res = original_forward(*a, **k)
        slots_box.setdefault('slots', res['slots'])  # the first call: the real training views
        return res

    torch.Tensor.detach = recording
    encoder.forward = encoder_forward
    try:
        out = forward_losses(model, batch, cfg, step, source=source)
    finally:
        torch.Tensor.detach = original_detach
        encoder.forward = original_forward
    by_ptr = {}
    for t in captured:
        by_ptr.setdefault((t.data_ptr(), tuple(t.shape)), t)
    out['terms'] = {k: by_ptr[(v.data_ptr(), tuple(v.shape))] for k, v in out['losses'].items()
                    if (v.data_ptr(), tuple(v.shape)) in by_ptr}
    out['state'] = slots_box['slots']
    return out


def flat_grad(loss, params):
    gs = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1).float() for g, p in zip(gs, params)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True)
    ap.add_argument('--model', action='append', required=True, help='name=checkpoint (run config next to it)')
    ap.add_argument('--batches', type=int, default=8)
    ap.add_argument('--step', type=int, default=200000)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    torch.set_num_threads(4)
    device = torch.device('cuda', 0)
    result = {'task': args.task, 'gpu': GPU_MAPPING, 'batches': args.batches, 'step': args.step, 'terms': TERMS,
              'groups': GROUPS, 'models': {}}
    fixed = None
    t0 = time.time()
    for spec in args.model:
        name, ckpt = spec.split('=', 1)
        cfg = load_config([(REPO / ckpt).parents[1] / 'config.yaml'])
        assert cfg['data']['task'] == args.task
        cfg['train']['num_workers'] = 4
        union, active = union_config(cfg)
        if fixed is None:  # the same training windows for every model (data configuration identical across them)
            train_ds = build_data(cfg)[0]
            loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2, collate_fn=collate,
                                generator=torch.Generator().manual_seed(1234))
            fixed = []
            for i, b in enumerate(loader):
                if i >= args.batches:
                    break
                fixed.append(b)
        model = build_model(union).to(device)
        load_checkpoint(REPO / ckpt, model.encoder, model.decoder, restore_rng=False)
        model.train()
        enc = [p for p in model.encoder.parameters() if p.requires_grad]
        dec = [p for p in model.decoder.parameters() if p.requires_grad]
        norms = defaultdict(lambda: defaultdict(list))
        cos = defaultdict(lambda: defaultdict(list))
        for bi, raw in enumerate(fixed):
            batch = move_batch(raw, device)
            torch.manual_seed(10_000 + bi)
            source = torch.randint(batch['images'].shape[2], (batch['images'].shape[0],), generator=torch.Generator().manual_seed(bi)).to(device)
            out = forward_with_terms(model, batch, union, args.step, source)
            terms = weighted_terms(out, union)
            total_check = sum(terms.values())
            if not torch.allclose(total_check.detach(), out['total'].detach(), rtol=1e-3, atol=1e-4):
                raise RuntimeError(f'term sum {float(total_check)} != total {float(out["total"])}')
            state = out['state']
            grads = {}
            for key, value in terms.items():
                grads[key] = {'encoder': flat_grad(value, enc), 'decoder': flat_grad(value, dec),
                              'state': torch.autograd.grad(value, state, retain_graph=True, allow_unused=True)[0]}
                if grads[key]['state'] is None:
                    grads[key]['state'] = torch.zeros_like(state)
                grads[key]['state'] = grads[key]['state'].reshape(-1).float()
            for g, members in GROUPS.items():
                act = [m for m in members if m in active and m in grads]
                if act:
                    grads[f'group:{g}'] = {part: sum(grads[m][part] for m in act) for part in ('encoder', 'decoder', 'state')}
            keys = list(grads)
            for part in ('encoder', 'decoder', 'state'):
                for k in keys:
                    norms[part][k].append(float(grads[k][part].norm()))
                for i, a in enumerate(keys):
                    for b in keys[i + 1:]:
                        na, nb = grads[a][part].norm(), grads[b][part].norm()
                        c = float((grads[a][part] @ grads[b][part]) / (na * nb)) if na > 0 and nb > 0 else float('nan')
                        cos[part][f'{a}|{b}'].append(c)
            del out, terms, grads
            torch.cuda.empty_cache()
        mean = lambda v: float(torch.tensor(v).nanmean())  # noqa: E731
        result['models'][name] = {
            'checkpoint': ckpt, 'active_terms': sorted(active),
            'grad_norm': {p: {k: mean(v) for k, v in d.items()} for p, d in norms.items()},
            'cosine': {p: {k: mean(v) for k, v in d.items()} for p, d in cos.items()},
            'negative_cosine_fraction': {p: {k: float((torch.tensor(v) < 0).float().mean()) for k, v in d.items()}
                                         for p, d in cos.items()},
        }
        m = result['models'][name]
        print(name, 'active', m['active_terms'])
        print('  state grad norms', {k: round(v, 4) for k, v in m['grad_norm']['state'].items()})
        print('  state group cosines', {k: round(v, 3) for k, v in m['cosine']['state'].items() if k.startswith('group:')})
        del model
        torch.cuda.empty_cache()
    result['seconds'] = round(time.time() - t0, 1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
