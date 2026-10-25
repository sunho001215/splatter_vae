"""Representation diagnostics of the encoder pilots (runs/setup/encp-definition-20261025.json "diagnostics"; inference
only: every encoder is in eval mode, unmasked, exactly the frozen DrM policy state ``Encoder.policy_state``; the script
asserts ``not encoder.training`` before every state computation).

For one task and arms ``name=export.pt`` (joint three-frame states): RL-spaced windows (frames t-4, t-2, t; stride 2).
- Probes, both fitted on training-episode windows seen from the training cameras (the in-training probe loader,
  ``probe_batches`` x 16 windows x 6 cameras): ridge (standardised inputs, lambda 1e-2, as variant_repr_probes.py) and a
  2-layer MLP (Linear(D,256)-GELU-Linear(256,out), standardised inputs/targets, AdamW lr 1e-3, weight decay 1e-4, batch
  256, <= 300 epochs, early stopping with patience 20 on 20 % of the training episodes held back, best epoch restored).
  Targets at the current frame t: hand and object position, hand and object velocity ((x_t - x_{t-4}) / (2 dt)): R^2 and
  mean Euclidean error (mm, mm/s); object-minus-hand: mean error and spread (mm), split by phase (approach = gripper
  opening >= 0.9 and hand-object distance > 3 cm; contact = otherwise). Evaluated on validation windows from the training
  cameras and transferred to the held-out cameras (far, near, trajectory sets).
- Retrieval: cross-view top-1 (training cameras; far held-out, near and trajectory queries against the training
  cameras), effective rank of the training-camera validation states (effective_rank_retrieval, state_statistics).
- History dependence: cosine between the intact state and the state with both history frames replaced by frame t.
- Within-episode retrieval: every stride-2 window of every validation episode; query from training camera a, gallery
  from camera b != a holding the same window and every window of the episode with 6 <= |t0' - t0| <= 40; top-1 = the
  same window first (cosine); all ordered camera pairs.
- Hard-negative cosine for every arm: one negative per window drawn by the hn sampler and robot-state rule
  (s4d/data/hardneg.py, the registered thresholds, torch generator seed 0, identical draws for every arm), then the hn
  training metrics (same-camera cosine, pairwise and gallery within-episode top-1).
    CUDA_VISIBLE_DEVICES=<MIG uuid> nice -n 15 .venv/bin/python -I encp_repr_diagnostics.py --task hammer \
        --arm chlrm=runs/chlrm-export-seed0-hammer-200k/encoder.pt --arm ... --out <json>
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.environ.get('S4D_REPO', '/home/ws/ws/splatter4d'), 'scripts'))
from _bootstrap import REPO, guard_gpus  # noqa: E402

GPU_MAPPING = guard_gpus()
import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from train import build_data  # noqa: E402

from s4d.config import load_config  # noqa: E402
from s4d.data.hardneg import DEFAULTS, HardNegativeSampler, robot_state  # noqa: E402
from s4d.data.metaworld.dataset import split_episodes  # noqa: E402
from s4d.diag.probes import cross_view_retrieval, r2_score, retrieval_top1, ridge_fit, ridge_predict  # noqa: E402
from s4d.losses.invariance import state_statistics  # noqa: E402
from s4d.model.encoder import load_encoder  # noqa: E402
from s4d.train.evaluate import frame_interval  # noqa: E402
from s4d.train.loop import hardneg_metrics  # noqa: E402
from s4d.train.workers import fork_safe_iter  # noqa: E402

GROUPS = {'train': 'images', 'heldout': 'eval_images', 'near': 'near_images', 'traj': 'traj_images'}
TARGETS = ('hand_pos', 'obj_pos', 'hand_vel', 'obj_vel', 'rel_pos')
OPEN, NEAR_M = 0.9, 0.03
STRIDE, MIN_GAP, MAX_GAP = 2, 6, 40


def targets(probe_state, dt):
    hand, obj = probe_state[:, :, 0:3].double(), probe_state[:, :, 4:7].double()
    dt = dt.double()[:, None]
    return {'hand_pos': hand[:, 2], 'obj_pos': obj[:, 2], 'rel_pos': obj[:, 2] - hand[:, 2],
            'hand_vel': (hand[:, 2] - hand[:, 0]) / (2 * dt), 'obj_vel': (obj[:, 2] - obj[:, 0]) / (2 * dt)}


def phase(probe_state):
    """approach (True) = gripper opening >= 0.9 and hand-object distance > 3 cm at frame t; contact otherwise."""
    p = probe_state[:, 2].double()
    return (p[:, 3] >= OPEN) & ((p[:, 4:7] - p[:, 0:3]).norm(dim=-1) > NEAR_M)


class Arm:
    def __init__(self, spec, device):
        self.name, path = spec.split('=', 1)
        self.path = path
        self.encoder = load_encoder(REPO / path).to(device).eval()
        assert int(self.encoder.cfg.num_frames) == 3, f'{self.name}: joint three-frame encoders only'

    @torch.no_grad()
    def states(self, images):
        """images (B,3,V,3,H,W) uint8 (any device) -> (B,V,D) float32 CPU policy states."""
        assert not self.encoder.training
        B, T, V = images.shape[:3]
        x = images.float().div(255.0).permute(0, 2, 1, 3, 4, 5).reshape(B * V, T, *images.shape[3:])
        s = self.encoder.policy_state(x.to(next(self.encoder.parameters()).device))
        assert not self.encoder.training
        return s.view(B, V, -1).cpu()


def mlp_probe(x, y, groups, seed=0, device='cuda'):
    """Fit Linear-GELU-Linear on (x, y) with episode-level early stopping; returns a predict function."""
    gen = torch.Generator().manual_seed(seed)
    eps = sorted(set(groups))
    order = torch.randperm(len(eps), generator=gen).tolist()
    held = {eps[i] for i in order[:max(1, round(0.2 * len(eps)))]}
    hold = torch.tensor([g in held for g in groups])
    x, y = x.float(), y.float()
    xm, xs = x[~hold].mean(0), x[~hold].std(0).clamp_min(1e-6)
    ym, ys = y[~hold].mean(0), y[~hold].std(0).clamp_min(1e-6)
    xt, yt = ((x - xm) / xs).to(device), ((y - ym) / ys).to(device)
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(x.shape[1], 256), torch.nn.GELU(), torch.nn.Linear(256, y.shape[1])).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    tr, va = (~hold).nonzero().flatten().to(device), hold.nonzero().flatten().to(device)
    best, best_state, wait = math.inf, None, 0
    for epoch in range(300):
        net.train()
        perm = tr[torch.randperm(len(tr), generator=gen).to(device)]
        for i in range(0, len(perm), 256):
            idx = perm[i:i + 256]
            loss = F.mse_loss(net(xt[idx]), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            val = float(F.mse_loss(net(xt[va]), yt[va]))
        if val < best - 1e-6:
            best, best_state, wait = val, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= 20:
                break
    net.load_state_dict(best_state)
    net.eval()

    def predict(xq):
        with torch.no_grad():
            return (net(((xq.float() - xm) / xs).to(device)).cpu() * ys + ym).double()

    return predict, {'epochs': epoch + 1, 'best_val_mse_std': best, 'held_back_episodes': len(held)}


def probe_report(fit_x, fit_y, fit_groups, sets, device):
    out = {'ridge': {}, 'mlp': {}, 'mlp_fit': {}}
    for name in TARGETS:
        W, m, s = ridge_fit(fit_x, fit_y[name])
        ridge = lambda xq, W=W, m=m, s=s: ridge_predict(W, m, s, xq)  # noqa: E731
        mlp, info = mlp_probe(fit_x, fit_y[name], fit_groups, device=device)
        out['mlp_fit'][name] = info
        for kind, predict in (('ridge', ridge), ('mlp', mlp)):
            for g, (xs, ys, ph) in sets.items():
                pred, true = predict(xs), ys[name].double()
                err = (pred - true).norm(dim=-1) * 1000.0  # mm (positions) or mm/s (velocities)
                rec = {'r2': r2_score(pred, true), 'err_mm': float(err.mean()),
                       'spread_mm': float((true - true.mean(0)).norm(dim=-1).mean() * 1000.0), 'n': len(true)}
                if name == 'rel_pos':
                    for label, sel in (('approach', ph), ('contact', ~ph)):
                        rec[f'err_mm_{label}'] = float(err[sel].mean()) if sel.any() else None
                        rec[f'spread_mm_{label}'] = float((true[sel] - true[sel].mean(0)).norm(dim=-1).mean() * 1000.0) \
                            if sel.any() else None
                        rec[f'n_{label}'] = int(sel.sum())
                out[kind][f'{name}/{g}'] = rec
    return out


def episode_windows(path, episodes):
    """Every stride-2 window of each episode: rgb (L,6,H,W,3) per episode and the (L,4) robot states."""
    with h5py.File(path, 'r') as f:
        train_cams = np.where(f['cameras']['is_train'][:].astype(bool))[0]
        V = len(train_cams)
        for ep in episodes:
            g = f['episodes'][ep]
            yield ep, g['rgb'][:, :V], robot_state(g['obs'][:, :4], 'metaworld')


@torch.no_grad()
def within_episode(arms, path, episodes, device):
    """Within-episode retrieval (gaps 6..40) and hn metrics with identical sampler draws for every arm."""
    sums = {a.name: {'within_top1': [], 'hn': []} for a in arms}
    gen = torch.Generator().manual_seed(0)
    draws_total = 0
    for ep, rgb, robot in episode_windows(path, episodes):
        L = len(rgb)
        starts = np.arange(0, L - 2 * STRIDE)
        sampler = HardNegativeSampler({ep: robot}, **DEFAULTS)
        negs = [sampler.sample(ep, int(t0), STRIDE, gen) for t0 in starts]
        draws_total += len(negs)
        frames = torch.from_numpy(rgb).permute(0, 1, 4, 2, 3)  # (L,V,3,H,W)
        for a in arms:
            states = []
            for i in range(0, len(starts), 64):
                t0 = torch.from_numpy(starts[i:i + 64])
                idx = torch.stack([t0, t0 + STRIDE, t0 + 2 * STRIDE], 1)  # (b,3)
                states.append(a.states(frames[idx]))  # (b,3,V,3,H,W) -> (b,V,D)
            S = F.normalize(torch.cat(states).float(), dim=-1)  # (N,V,D), window index == t0
            N, V = S.shape[:2]
            gap = (torch.arange(N)[:, None] - torch.arange(N)[None]).abs()
            allowed = (gap >= MIN_GAP) & (gap <= MAX_GAP)
            allowed.fill_diagonal_(True)
            correct = []
            for ca in range(V):
                for cb in range(V):
                    if ca == cb:
                        continue
                    sim = (S[:, ca] @ S[:, cb].T).masked_fill(~allowed, -math.inf)
                    correct.append(sim.argmax(1) == torch.arange(N))
            sums[a.name]['within_top1'].append(torch.stack(correct).float())  # (pairs, N)
            valid = torch.tensor([n is not None for n in negs])
            neg_idx = torch.tensor([n if n is not None else 0 for n in negs])
            m = hardneg_metrics(S, S[neg_idx], valid, torch.zeros(N))
            sums[a.name]['hn'].append({k: float(v) for k, v in m.items() if k != 'hardneg_hand_disp_mean'} | {'n': N})
    out = {}
    for a in arms:
        top1 = torch.cat(sums[a.name]['within_top1'], 1)
        hn = sums[a.name]['hn']
        n = sum(h['n'] for h in hn)
        out[a.name] = {'within_episode_top1': float(top1.mean()), 'within_windows': int(top1.shape[1]),
                       **{k: sum(h[k] * h['n'] for h in hn if not math.isnan(h[k])) /
                          max(1, sum(h['n'] for h in hn if not math.isnan(h[k])))
                          for k in ('hardneg_cosine', 'hardneg_within_episode_top1', 'hardneg_gallery_top1',
                                    'hardneg_valid_fraction')}, 'hn_windows': n}
    return out, draws_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True)
    ap.add_argument('--arm', action='append', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--probe-batches', type=int, default=24)
    args = ap.parse_args()
    torch.set_num_threads(4)
    device = torch.device('cuda', 0)
    cfg = load_config([REPO / 'configs/metaworld/base.yaml', REPO / f'configs/metaworld/tasks/{args.task}.yaml'])
    cfg['data'].update(strides=[2], val_strides=[2])
    cfg.setdefault('eval', {})['heldout_sets'] = True
    cfg['train']['num_workers'] = 4
    _, val_loaders, probe_loaders, n_train = build_data(cfg)
    arms = [Arm(s, device) for s in args.arm]
    static = lambda x: x[:, [2, 2, 2]]  # noqa: E731
    fit = {a.name: [] for a in arms}
    fit_y, fit_groups = [], []
    for i, batch in enumerate(fork_safe_iter(probe_loaders[2])):
        if i >= args.probe_batches:
            break
        V = batch['images'].shape[2]
        y = targets(batch['probe_state'], frame_interval(batch))
        fit_y.append({k: v.repeat_interleave(V, 0) for k, v in y.items()})
        fit_groups += [e for e in batch['meta']['episode'] for _ in range(V)]
        for a in arms:
            fit[a.name].append(a.states(batch['images']).flatten(0, 1))
    fit_y = {k: torch.cat([y[k] for y in fit_y]) for k in fit_y[0]}
    val = {a.name: {g: {'intact': [], 'static': [], 'paired_train': []} for g in GROUPS} for a in arms}
    val_y, val_ph = {g: [] for g in GROUPS}, {g: [] for g in GROUPS}
    windows = 0
    for batch in fork_safe_iter(val_loaders[2]):
        y = targets(batch['probe_state'], frame_interval(batch))
        ph = phase(batch['probe_state'])
        windows += batch['images'].shape[0]
        for g, key in GROUPS.items():
            if key not in batch:
                continue
            V = batch[key].shape[2]
            val_y[g].append({k: v.repeat_interleave(V, 0) for k, v in y.items()})
            val_ph[g].append(ph.repeat_interleave(V, 0))
            for a in arms:
                val[a.name][g]['intact'].append(a.states(batch[key]))
                val[a.name][g]['static'].append(a.states(static(batch[key])))
                if g != 'train':  # the same windows' training-camera states (the retrieval gallery)
                    val[a.name][g]['paired_train'].append(val[a.name]['train']['intact'][-1])
    path = Path(cfg['data']['root']) / f'{args.task}.hdf5'
    _, val_eps = split_episodes(path, float(cfg['data']['train_ratio']), 0)
    within, draws = within_episode(arms, path, val_eps, device)
    result = {'task': args.task, 'gpu': GPU_MAPPING, 'val_windows': windows, 'fit_states': len(fit_y['hand_pos']),
              'validation_episodes': val_eps, 'hn_sampler_draws': draws, 'groups': {g: len(v) for g, v in val_y.items()},
              'arms': {}}
    t0 = time.time()
    for a in arms:
        sets = {}
        for g in GROUPS:
            if not val_y[g]:
                continue
            yi = {k: torch.cat([y[k] for y in val_y[g]]) for k in val_y[g][0]}
            sets[g] = (torch.cat([s.flatten(0, 1) for s in val[a.name][g]['intact']]), yi, torch.cat(val_ph[g]))
        train_states = torch.cat(val[a.name]['train']['intact'])  # (M,6,D)
        retrieval = {'train': cross_view_retrieval(train_states, n_train)['retrieval_top1_train']}
        for g in ('heldout', 'near', 'traj'):  # query camera of the group vs every training camera, same windows
            rows = val[a.name][g]['intact']
            if not rows:
                continue
            both = torch.cat((torch.cat(val[a.name][g]['paired_train']), torch.cat(rows)), 1)
            retrieval[g] = cross_view_retrieval(both, n_train)['retrieval_top1_heldout']
            retrieval[f'{g}_windows'] = int(both.shape[0])
        cos = {g: float(F.cosine_similarity(torch.cat([s.flatten(0, 1) for s in val[a.name][g]['intact']]),
                                            torch.cat([s.flatten(0, 1) for s in val[a.name][g]['static']]), dim=-1).mean())
               for g in GROUPS if val[a.name][g]['intact']}
        result['arms'][a.name] = {
            'export': a.path, 'state_dim': int(train_states.shape[-1]),
            'probes': probe_report(torch.cat(fit[a.name]), fit_y, fit_groups, sets, device),
            'retrieval_top1': retrieval,
            'effective_rank_retrieval': float(state_statistics(train_states.reshape(-1, train_states.shape[-1]))['effective_rank']),
            'history_cosine_static_vs_intact': cos,
            **within[a.name],
        }
        r = result['arms'][a.name]
        print(a.name, 'hand_vel r2 ridge/mlp', round(r['probes']['ridge']['hand_vel/train']['r2'], 3),
              round(r['probes']['mlp']['hand_vel/train']['r2'], 3), 'within', round(r['within_episode_top1'], 3),
              'hn_cos', round(r['hardneg_cosine'], 3), flush=True)
    result['seconds'] = round(time.time() - t0, 1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
