"""Instrumented replay of afk2-seed0-bin-picking-200k from its step-10000 checkpoint (spike investigation 2026-10-21).

Runs the run's own code (worktree .worktrees/afk2-replay at commit bfb4ccb, whose 286 fingerprinted source files equal
the launch record) and its saved config, resuming from a COPY of checkpoints/step_0010000.pt in a separate directory
(the original run is not touched), with the same seed, data order (index-level resume) and restored RNG states.
Only wrappers are added, no training change:
  - visibility_loss: per step, the violation split into its terms (du, dv off-image; dz depth range), centres behind a
    camera (z < 0), inside the near plane (0 <= z < near) and beyond far, the largest single-point violation and its
    share of the total, the smallest positive depth, and the gradient of the visibility term on the centres;
  - forward_losses: every loss term of every step (the run logged one step in 50);
  - clip_grad_norm_: pre-clip gradient norms (total, encoder, decoder).
Per-step records -> <out>/per_step.jsonl; the 32 most violating centres of the first anomalous steps -> <out>/points/.

    CUDA_VISIBLE_DEVICES=<MIG> python -I afk2_binpick_replay.py --stop 12700 [--out DIR]
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path('/home/ws/ws/splatter4d')
WT = ROOT / '.worktrees/afk2-replay'
RUN = ROOT / 'runs/pretrain/afk2-seed0-bin-picking-200k'

ap = argparse.ArgumentParser()
ap.add_argument('--stop', type=int, default=12700)
ap.add_argument('--from-step', type=int, default=10000, help='the saved checkpoint to start from (10k steps apart)')
ap.add_argument('--out', default=str(ROOT / 'runs/setup/afk2-binpick-spike-20261021/replay-from-10k'))
ap.add_argument('--name', default='replay-from-10k')
ap.add_argument('--fixed', action='store_true', help='substitute the candidate fix: off-image terms only in front of near')
args = ap.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
(out / 'points').mkdir(exist_ok=True)
source = out / f'resume-step_{args.from_step:07d}.pt'
if not source.exists():
    shutil.copy2(RUN / f'checkpoints/step_{args.from_step:07d}.pt', source)

sys.path.insert(0, str(WT / 'scripts'))
sys.path.insert(0, str(WT))
import os  # noqa: E402

os.chdir(WT)
import train as T  # noqa: E402  (GPU guard)
import torch  # noqa: E402

import s4d.train.loop as L  # noqa: E402
from s4d.geometry import project  # noqa: E402
from s4d.losses import regularizers as R  # noqa: E402

REC = {}
LOG = (out / 'per_step.jsonl').open('a')
DUMPED = []


def vis_wrapper(xyz_seq, w2c, K, height, width, near, far):
    res = ORIG_VIS(xyz_seq, w2c, K, height, width, near, far)
    if not (torch.is_grad_enabled() and xyz_seq.requires_grad):
        return res
    with torch.no_grad():
        B, S, N, _ = xyz_seq.shape
        V = w2c.shape[1]
        pts = xyz_seq.float()[:, :, None].expand(B, S, V, N, 3)
        uv, z = project(pts, K.float()[:, None].expand(B, S, V, 3, 3), w2c.float()[:, None].expand(B, S, V, 4, 4))
        u, v = uv.unbind(-1)
        m = R.MARGIN_PIXELS
        du = (torch.relu(-m - u) + torch.relu(u - (width + m))) / width
        dv = (torch.relu(-m - v) + torch.relu(v - (height + m))) / height
        dz = (torch.relu(near - z) + torch.relu(z - far)) / (far - near)
        viol = torch.nan_to_num(du + dv + dz, nan=1.0, posinf=1.0, neginf=1.0)
        total = viol.sum().clamp_min(1e-12)
        pos = torch.where(z > 0, z, torch.full_like(z, float('inf')))
        REC['vis'] = {
            'per_state': [float(x) for x in res.detach()],
            'du_mean': float(du.mean()), 'dv_mean': float(dv.mean()), 'dz_mean': float(dz.mean()),
            'du_max': float(du.max()), 'dv_max': float(dv.max()), 'dz_max': float(dz.max()),
            'viol_max': float(viol.max()), 'viol_max_share': float(viol.max() / total),
            'top32_share': float(viol.flatten().topk(32).values.sum() / total),
            'behind': int((z < 0).sum()), 'inside_near': int(((z >= 0) & (z < near)).sum()),
            'beyond_far': int((z > far).sum()), 'off_image': int(((du + dv) > 0).sum()),
            'z_min_positive': float(pos.min()), 'points': int(viol.numel()),
        }
        if len(DUMPED) < 6 and REC['vis']['viol_max'] > 1.0:
            top = viol.flatten().topk(32)
            idx = torch.stack(torch.unravel_index(top.indices, viol.shape), -1)  # (32, 4): b, s, v, n
            DUMPED.append(REC.get('step'))
            torch.save({'step': REC.get('step'), 'index_bsvn': idx.cpu(), 'violation': top.values.cpu(),
                        'z': z.flatten()[top.indices].cpu(), 'u': u.flatten()[top.indices].cpu(),
                        'v': v.flatten()[top.indices].cpu(),
                        'xyz': xyz_seq.detach().float()[idx[:, 0], idx[:, 1], idx[:, 3]].cpu()},
                       out / 'points' / f'step_{REC.get("step")}.pt')
    grad = torch.autograd.grad(res.mean(), xyz_seq, retain_graph=True)[0].float()
    REC['vis_grad'] = {'norm': float(grad.norm()), 'max_point_norm': float(grad.norm(dim=-1).max())}
    return res


def fwd_wrapper(model, batch, cfg, step, *a, **kw):
    if model.training:
        REC.clear()
        REC['step'] = int(step)
        REC['model'] = model
    out_ = ORIG_FWD(model, batch, cfg, step, *a, **kw)
    if model.training:
        REC['losses'] = {k: float(v) for k, v in out_['losses'].items() if torch.is_tensor(v) and v.numel() == 1}
    return out_


def clip_wrapper(parameters, max_norm, *a, **kw):
    model = REC.get('model')
    norms = {}
    if model is not None:
        for name in ('encoder', 'decoder'):
            g = [p.grad.detach().float().norm() for p in getattr(model, name).parameters() if p.grad is not None]
            norms[name] = float(torch.stack(g).norm()) if g else 0.0
    total = ORIG_CLIP(parameters, max_norm, *a, **kw)
    if 'step' in REC:
        LOG.write(json.dumps({'step': REC['step'], 'losses': REC.get('losses'), 'vis': REC.get('vis'),
                              'vis_grad': REC.get('vis_grad'), 'grad_preclip_total': float(total),
                              'grad_preclip': norms}) + '\n')
        LOG.flush()
    return total


def fixed_visibility_loss(xyz_seq, w2c, K, height, width, near, far):
    """Candidate fix: the off-image terms (du, dv) are applied only to centres in front of the near plane, where
    the projection is defined and bounded; centres at or behind it keep the linear depth-range term dz. Identical to
    the original for every point with z >= near."""
    B, S, N, _ = xyz_seq.shape
    V = w2c.shape[1]
    pts = xyz_seq.float()[:, :, None].expand(B, S, V, N, 3)
    uv, z = project(pts, K.float()[:, None].expand(B, S, V, 3, 3), w2c.float()[:, None].expand(B, S, V, 4, 4))
    u, v = uv.unbind(-1)
    m = R.MARGIN_PIXELS
    front = z >= near
    du = torch.where(front, (torch.relu(-m - u) + torch.relu(u - (width + m))) / width, torch.zeros_like(u))
    dv = torch.where(front, (torch.relu(-m - v) + torch.relu(v - (height + m))) / height, torch.zeros_like(v))
    dz = (torch.relu(near - z) + torch.relu(z - far)) / (far - near)
    violation = torch.nan_to_num(du + dv + dz, nan=1.0, posinf=1.0, neginf=1.0)
    return violation.mean(dim=(0, 2, 3))


ORIG_VIS, ORIG_FWD, ORIG_CLIP = L.visibility_loss, L.forward_losses, torch.nn.utils.clip_grad_norm_
if args.fixed:
    ORIG_VIS = fixed_visibility_loss
L.visibility_loss = vis_wrapper
L.forward_losses = fwd_wrapper
torch.nn.utils.clip_grad_norm_ = clip_wrapper
sys.argv = ['train.py', '--config', str(RUN / 'config.yaml'), '--name', args.name, '--output-root', str(out.parent),
            '--resume', str(source), '--no-wandb', '--set', f'train.stop_step={args.stop}', 'train.save_every=100000']
T.main()
