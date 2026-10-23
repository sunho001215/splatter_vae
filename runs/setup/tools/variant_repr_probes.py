"""Representation probes of the RL encoders, uniform across variants (2026-10-23; inference only).

For one task and a list of arms ``name=export.pt:mode`` (mode: joint | latest | stack, the policy input of
s4d/rl/encoders.py), the policy state of every RL-spaced window (frames t-4, t-2, t: stride 2 = the RL observation
spacing) is computed exactly as the frozen DrM policy computes it from the export (``Encoder.policy_state``: eval mode,
no masking, bf16 autocast) for four camera groups of the validation episodes: training cameras (6), held-out cameras (4),
and the near and moving-trajectory held-out sets (render_heldout_sets.py).

Probes: ridge regression (standardised inputs, lambda 1e-2, s4d.diag.probes) fitted on training-episode windows seen from
the training cameras (the in-training protocol's probe loader: ``probe_batches`` x 16 windows x 6 cameras), R^2 on the
validation windows per camera group. Targets: hand, object and object-minus-hand position at the CURRENT frame t (the
in-training probes use the middle frame), hand and object velocity (x_t - x_{t-4}) / (2 dt); R^2 and the mean
Euclidean error (with the target's own spread).
History dependence: the same windows with both history frames replaced by the current frame (t, t, t); reported are the
state cosine to the intact state, the R^2 of the intact-fitted probes on static-history states (fixed) and of probes
refitted on static-history training states (refit). For a "latest" arm the history is ignored by construction.
Retrieval: cross-view top-1 on the validation windows (training cameras; held-out query vs training gallery).
    CUDA_VISIBLE_DEVICES=<MIG uuid> nice -n 15 .venv/bin/python -I runs/setup/tools/variant_repr_probes.py --task hammer \
        --arm chl200=runs/chl-export-seed0-hammer-200k/encoder.pt:joint --arm ... --out <json>
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.environ.get('S4D_REPO', '/home/ws/ws/splatter4d'), 'scripts'))
from _bootstrap import REPO, guard_gpus  # noqa: E402

GPU_MAPPING = guard_gpus()
import torch  # noqa: E402
from train import build_data  # noqa: E402

from s4d.config import load_config  # noqa: E402
from s4d.diag.probes import cross_view_retrieval, fit_and_score, ridge_fit, ridge_predict  # noqa: E402
from s4d.model.encoder import load_encoder  # noqa: E402
from s4d.train.evaluate import frame_interval  # noqa: E402
from s4d.train.workers import fork_safe_iter  # noqa: E402

GROUPS = {'train': 'images', 'heldout': 'eval_images', 'near': 'near_images', 'traj': 'traj_images'}


def mean_error(train_x, train_y, sets):
    """Mean Euclidean error of the same ridge probes (metres or m/s) and the target's own spread (mean distance to
    its mean), per target and set: R^2 alone is uninformative for targets with little variance (e.g. rel_pos when the
    object is held)."""
    out = {}
    for name, y in train_y.items():
        W, m, s = ridge_fit(train_x, y)
        for set_name, (xs, ys) in sets.items():
            err = (ridge_predict(W, m, s, xs) - ys[name].double()).norm(dim=-1).mean()
            out[f'err_{name}_{set_name}'] = float(err)
            out[f'spread_{name}_{set_name}'] = float((ys[name].double() - ys[name].double().mean(0)).norm(dim=-1).mean())
    return out


def targets(probe_state, dt):
    """probe_state (B,3,11): hand xyz 0:3, gripper 3, object xyz 4:7, quaternion 7:11; dt (B,) seconds."""
    hand, obj = probe_state[:, :, 0:3].double(), probe_state[:, :, 4:7].double()
    dt = dt.double()[:, None]
    return {'hand_pos': hand[:, 2], 'obj_pos': obj[:, 2], 'rel_pos': obj[:, 2] - hand[:, 2],
            'hand_vel': (hand[:, 2] - hand[:, 0]) / (2 * dt), 'obj_vel': (obj[:, 2] - obj[:, 0]) / (2 * dt)}


class Arm:
    def __init__(self, spec, device):
        self.name, rest = spec.split('=', 1)
        path, self.mode = rest.rsplit(':', 1)
        self.path = path
        self.encoder = load_encoder(REPO / path).to(device)
        frames = int(self.encoder.cfg.num_frames)
        if (self.mode == 'joint') != (frames > 1) or self.mode not in ('joint', 'latest', 'stack'):
            raise ValueError(f'{self.name}: mode {self.mode} does not fit num_frames {frames}')

    @torch.no_grad()
    def states(self, images_u8):
        """images (B,3,V,3,H,W) uint8 -> policy states (B,V,D) float32 on the CPU."""
        B, T, V = images_u8.shape[:3]
        x = images_u8.float().div(255.0).permute(0, 2, 1, 3, 4, 5).reshape(B * V, T, *images_u8.shape[3:])
        if self.mode == 'latest':
            s = self.encoder.policy_state(x[:, -1:])
        elif self.mode == 'stack':
            s = self.encoder.policy_state(x.reshape(B * V * T, 1, *x.shape[2:])).view(B * V, -1)
        else:
            s = self.encoder.policy_state(x)
        return s.view(B, V, -1).cpu()


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
    fit = {a.name: {'intact': [], 'static': []} for a in arms}
    fit_y = []
    for i, batch in enumerate(fork_safe_iter(probe_loaders[2])):
        if i >= args.probe_batches:
            break
        V = batch['images'].shape[2]
        y = targets(batch['probe_state'], frame_interval(batch))
        fit_y.append({k: v.repeat_interleave(V, 0) for k, v in y.items()})
        for a in arms:
            images = batch['images'].to(device)
            fit[a.name]['intact'].append(a.states(images).flatten(0, 1))
            fit[a.name]['static'].append(a.states(static(images)).flatten(0, 1))
    fit_y = {k: torch.cat([y[k] for y in fit_y]) for k in fit_y[0]}
    val = {a.name: {g: {'intact': [], 'static': []} for g in GROUPS} for a in arms}
    val_y = {g: [] for g in GROUPS}
    windows = 0
    for batch in fork_safe_iter(val_loaders[2]):
        y = targets(batch['probe_state'], frame_interval(batch))
        windows += batch['images'].shape[0]
        for g, key in GROUPS.items():
            if key not in batch:
                continue
            V = batch[key].shape[2]
            val_y[g].append({k: v.repeat_interleave(V, 0) for k, v in y.items()})
            images = batch[key].to(device)
            for a in arms:
                val[a.name][g]['intact'].append(a.states(images))
                val[a.name][g]['static'].append(a.states(static(images)))
    result = {'task': args.task, 'gpu': GPU_MAPPING, 'val_windows': windows, 'fit_states': len(fit_y['hand_pos']),
              'groups': {g: len(v) for g, v in val_y.items()}, 'arms': {}}
    t0 = time.time()
    for a in arms:
        tr_int, tr_sta = torch.cat(fit[a.name]['intact']), torch.cat(fit[a.name]['static'])
        sets_int, sets_sta, cos = {}, {}, {}
        for g in GROUPS:
            if not val_y[g]:
                continue
            yi = {k: torch.cat([y[k] for y in val_y[g]]) for k in val_y[g][0]}
            si = torch.cat([s.flatten(0, 1) for s in val[a.name][g]['intact']])
            ss = torch.cat([s.flatten(0, 1) for s in val[a.name][g]['static']])
            sets_int[g], sets_sta[g] = (si, yi), (ss, yi)
            cos[g] = float(torch.nn.functional.cosine_similarity(si, ss, dim=-1).mean())
        intact = fit_and_score(tr_int, fit_y, sets_int)
        fixed = fit_and_score(tr_int, fit_y, sets_sta)
        refit = fit_and_score(tr_sta, fit_y, sets_sta)
        train_states = torch.cat(val[a.name]['train']['intact'])  # (M,6,D)
        retrieval = {}
        if val[a.name]['heldout']['intact']:
            both = torch.cat((train_states, torch.cat(val[a.name]['heldout']['intact'])), 1)
            retrieval = cross_view_retrieval(both, n_train)
        result['arms'][a.name] = {
            'export': a.path, 'mode': a.mode, 'state_dim': int(tr_int.shape[1]),
            'r2': intact, 'r2_static_history_fixed_probe': fixed, 'r2_static_history_refit': refit,
            'error': mean_error(tr_int, fit_y, sets_int),
            'state_cosine_static_vs_intact': cos, 'retrieval': retrieval,
        }
        print(a.name, {k: round(v, 3) for k, v in intact.items() if k.endswith('_train') or k.endswith('_traj')}, flush=True)
    result['seconds'] = round(time.time() - t0, 1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
