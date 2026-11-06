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
from s4d.diag import _encp_repr  # noqa: E402
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
    return _encp_repr.targets(probe_state, dt)


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
    (net, xm, xs, ym, ys), info = _encp_repr.mlp_probe(
        x, y, groups, seed=seed, device=device, torch=torch, F=F, math=math,
    )

    # Keep the returned callable's module identity and globals at this entry point.
    def predict(xq):
        with torch.no_grad():
            return (net(((xq.float() - xm) / xs).to(device)).cpu() * ys + ym).double()

    return predict, info


def probe_report(fit_x, fit_y, fit_groups, sets, device):
    return _encp_repr.probe_report(
        fit_x, fit_y, fit_groups, sets, device,
        TARGETS=TARGETS, ridge_fit=ridge_fit, ridge_predict=ridge_predict, mlp_probe=mlp_probe, r2_score=r2_score,
    )


def episode_windows(path, episodes):
    """Every stride-2 window of each episode: rgb (L,6,H,W,3) per episode and the (L,4) robot states."""
    yield from _encp_repr.episode_windows(path, episodes, h5py=h5py, np=np, robot_state=robot_state)


@torch.no_grad()
def within_episode(arms, path, episodes, device):
    """Within-episode retrieval (gaps 6..40) and hn metrics with identical sampler draws for every arm."""
    return _encp_repr.within_episode(
        arms, path, episodes, device,
        torch=torch, np=np, F=F, math=math, episode_windows=episode_windows,
        HardNegativeSampler=HardNegativeSampler, DEFAULTS=DEFAULTS, STRIDE=STRIDE,
        MIN_GAP=MIN_GAP, MAX_GAP=MAX_GAP, hardneg_metrics=hardneg_metrics,
    )


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
