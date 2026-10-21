"""Training-only numerical and gradient-scale check that fixes the Chamfer weight of variant chamfer_lean (2026-10-21).

Before any candidate training: on training-split batches of hammer and pick-place, at two model states (a fresh
seed-0 initialisation with the task's anchor statistics, and the trained reference C + D0 + M3D at 100k), compute the
gradient norm over all model parameters of
  R = 0.5 * hard-depth L1 + 0.5 * depth-gradient loss + 1.0 * visibility   (the geometry terms the candidate removes,
      with their reference weights and the reference temporal weighting), and
  C = the bidirectional Chamfer loss at weight 1 (s4d/losses/chamfer.py, same temporal weighting),
plus the values and gradient norms of the expected-depth L1 term (kept) for scale.
Rule (fixed before running): lambda = median over (task, state, batch) of |grad R| / |grad C|, rounded to the nearest
value of the 1-2-5 series, so Chamfer replaces the removed geometry supervision at a comparable gradient scale.
No training, no sweep.

    CUDA_VISIBLE_DEVICES=<MIG uuid> .venv/bin/python -I runs/setup/tools/chamfer_weight_check.py --worktree <path>
"""
import argparse
import json
import math
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--worktree', required=True)
ap.add_argument('--batches', type=int, default=4)
ap.add_argument('--tag', default='')
args = ap.parse_args()
WT = Path(args.worktree)
sys.path.insert(0, str(WT / 'scripts'))
sys.path.insert(0, str(WT))
from _bootstrap import guard_gpus  # noqa: E402

GPU_MAPPING = guard_gpus()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from train import build_data  # noqa: E402

from s4d.config import load_config  # noqa: E402
from s4d.data.contract import collate  # noqa: E402
from s4d.losses.chamfer import chamfer_loss, target_clouds  # noqa: E402
from s4d.losses.depth import depth_gradient_loss, depth_l1  # noqa: E402
from s4d.losses.regularizers import visibility_loss  # noqa: E402
from s4d.losses.rgb import pixel_weights  # noqa: E402
from s4d.model.gaussians import GaussianSet  # noqa: E402
from s4d.model.render import render_hard_depth, render_rgbd  # noqa: E402
from s4d.train.checkpoint import load_checkpoint  # noqa: E402
from s4d.train.loop import _per_t, _rows, build_model, temporal_ramp  # noqa: E402

REPO = Path('/home/ws/ws/splatter4d')
REF = {'hammer': 'runs/pretrain/i2-m3d-seed0-hammer', 'pick-place': 'runs/pretrain/i2-m3d-seed0-pick-place'}
OVERRIDES = ['train.steps=200000', 'train.seed=0', 'aug.synth_views=2', 'aug.synth_render=true',
             'loss.depth_valid_far=true', 'loss.motion_space=gaussian']


def grad_norm(term, params):
    gs = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
    return math.sqrt(sum(float((g.float() ** 2).sum()) for g in gs if g is not None))


def main():
    device = torch.device('cuda', 0)
    rows = []
    for task, ref in REF.items():
        cfg = load_config([REPO / 'configs/metaworld/base.yaml', REPO / f'configs/metaworld/tasks/{task}.yaml'],
                          [*OVERRIDES])
        cfg['data']['num_workers'] = 0
        train_set, _, _, _ = build_data(cfg)
        near, far = float(cfg['render']['near']), float(cfg['render']['far'])
        gen = torch.Generator().manual_seed(0)
        idxs = [torch.randint(len(train_set), (16,), generator=gen).tolist() for _ in range(args.batches)]
        for state in ('init', 'reference_100k'):
            torch.manual_seed(0)
            model = build_model(cfg).to(device)
            step = 0
            if state == 'init':  # anchors from the task's workspace statistics, as s4d/train/loop.py train() does
                model.decoder.set_anchor_statistics(json.loads(Path(cfg['model']['anchor_stats']).read_text()))
            if state == 'reference_100k':
                step = load_checkpoint(REPO / ref / 'checkpoints/step_0100000.pt', model.encoder, model.decoder,
                                       restore_rng=False)
            model.train()
            params = [p for p in model.parameters() if p.requires_grad]
            ramp = temporal_ramp(step, int(cfg['loss'].get('ramp_steps', 20000)))
            for bi, idx in enumerate(idxs):
                batch = collate([train_set[i] for i in idx])
                batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                images = batch['images']
                B, T, V, _, H, W = images.shape
                enc_in = (images.float() / 255).permute(0, 2, 1, 3, 4, 5).reshape(B * V, T, 3, H, W)
                score = batch['motion_score'].float()
                score_in = score.permute(0, 2, 1, 3, 4, 5).reshape(B * V, T, 1, H, W)
                source = torch.randint(V, (B,), device=device)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    _, _, gs = model(enc_in, score_in, V, source)
                gs = GaussianSet(gs.xyz.float(), gs.scales.float(), gs.quats.float(), gs.opacity.float(), gs.rgb.float(),
                                 gs.delta01.float(), gs.delta12.float(), gs.group)
                w2c, K = batch['w2c'], batch['K']
                xyz_seq = gs.xyz_sequence()
                r = render_rgbd(gs, xyz_seq, w2c, K, H, W, near, far)
                depth_t = batch['depth'].float()
                valid = (_rows(depth_t) > 0) & (_rows(depth_t) < far)
                weights = pixel_weights(_rows(score), 1.0)

                def tw(per_t):
                    return (per_t[0] + ramp * (per_t[1] + per_t[2])) / (1.0 + 2.0 * ramp)

                dl1 = tw(_per_t(depth_l1(_rows(r['depth']), _rows(depth_t), valid, weights), B, T, V))
                dgrad = tw(_per_t(depth_gradient_loss(_rows(r['depth']), _rows(depth_t), valid, weights=weights), B, T, V))
                hard = render_hard_depth(gs, xyz_seq, w2c, K, H, W, near, far)
                dhard = tw(_per_t(depth_l1(_rows(hard['depth']), _rows(depth_t), valid, weights), B, T, V))
                vis = tw(visibility_loss(xyz_seq, w2c, K, H, W, near, far))
                removed = 0.5 * dhard + 0.5 * dgrad + 1.0 * vis
                clouds, cvalid, stats = target_clouds(batch['depth'][:, :, :, 0], K, batch['c2w'], far, True, 0.01, 4096)
                ch = chamfer_loss(xyz_seq, clouds, cvalid)
                cham = tw(ch['loss'])
                row = {'task': task, 'state': state, 'batch': bi,
                       'value': {'removed': float(removed), 'depth_hard': float(dhard), 'depth_grad': float(dgrad),
                                 'visibility': float(vis), 'depth_l1': float(dl1), 'chamfer': float(cham),
                                 'chamfer_pred_to_target_t0': float(ch['pred_to_target'][0]),
                                 'chamfer_target_to_pred_t0': float(ch['target_to_pred'][0])},
                       'grad': {'removed': grad_norm(removed, params), 'depth_l1': grad_norm(dl1, params),
                                'chamfer': grad_norm(cham, params), 'hard_half': grad_norm(0.5 * dhard, params),
                                'grad_half': grad_norm(0.5 * dgrad, params), 'visibility': grad_norm(vis, params),
                                'depth_removed': grad_norm(0.5 * dhard + 0.5 * dgrad, params)},
                       'cloud': {k: float(v) for k, v in stats.items()}}
                row['ratio_removed_over_chamfer'] = row['grad']['removed'] / max(row['grad']['chamfer'], 1e-12)
                row['ratio_depth_removed_over_chamfer'] = row['grad']['depth_removed'] / max(row['grad']['chamfer'], 1e-12)
                rows.append(row)
                print(json.dumps({k: row[k] for k in ('task', 'state', 'batch', 'ratio_removed_over_chamfer')} |
                                 {'grad': {k: round(v, 4) for k, v in row['grad'].items()},
                                  'value': {k: round(v, 6) for k, v in row['value'].items()}}), flush=True)
            del model
            torch.cuda.empty_cache()
    # Rule v2 (2026-10-21, after rule v1 below proved ill-posed): the removed terms' gradient at initialisation is
    # dominated by the visibility regulariser (12 on hammer, ~1,140 on pick-place: out-of-frustum initial Gaussians, a
    # transient clipped at grad_clip = 1), which Chamfer does not replace; v1's median then falls between two clusters.
    # v2 uses the depth-type terms Chamfer replaces as geometry supervision: 0.5 hard-depth + 0.5 depth-gradient.
    ratios_v1 = [r['ratio_removed_over_chamfer'] for r in rows]
    ratios = [r['ratio_depth_removed_over_chamfer'] for r in rows]
    med = float(np.median(ratios))
    series = sorted({m * 10 ** e for m in (1, 2, 5) for e in range(-3, 6)})
    lam = min(series, key=lambda s: abs(math.log(s) - math.log(med)))
    out = {'rule': 'v2: lambda = median |grad (0.5 hard-depth + 0.5 depth-gradient)| / |grad C| over task x state x batch, '
                   'rounded to the nearest 1-2-5 value (v1 used R including the visibility term; see the comment above)',
           'v1_median_ratio_including_visibility': float(np.median(ratios_v1)),
           'median_ratio': med, 'ratio_range': [float(min(ratios)), float(max(ratios))], 'lambda': lam,
           'rows': rows, 'gpu': GPU_MAPPING, 'worktree': str(WT)}
    (REPO / f'runs/setup/chamfer-weight-check-20261021{args.tag}.json').write_text(json.dumps(out, indent=1))
    print('LAMBDA', lam, 'median ratio', med, 'range', out['ratio_range'])


if __name__ == '__main__':
    main()
