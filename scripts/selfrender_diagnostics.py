"""Checkpoint diagnostics for the self-render replacement variant (user amendment 2026-10-29, item 4): object/robot
reconstruction, self-render coverage, temporal coherence and feature statistics of pretraining checkpoints, intended
for the steps around the warm-up end (20k) and the ramp end (40k) of sr-csw4 runs and the same steps of the matching
enc-csw4 runs. Diagnostics only: nothing here selects or tunes a setting.

Data: windows (stride 2) of the task's validation episodes (``--split``; the run's own split manifest), TRAINING cameras
only; evaluation-camera images/depth are never read. Per checkpoint (eval-mode encoder, no masking, source camera 0):

  recon_*        decoded scene rendered into the 6 training cameras at t0..t2: PSNR on robot/object pixels (bodies
                 that move within some training episode; body segmentation) and on the rest, alpha and depth AbsRel on
                 robot/object pixels;
  sr_*           the same scene rendered (no gradient) from 2 cameras per window drawn from the frozen viewpoint domain
                 (fixed seed): alpha mean, fraction of 16 px temporal tokens with mean alpha >= 0.9, fraction of views
                 with fewer eligible tokens than the 48 kept (coverage exception);
  temporal_*     coherence over t0 -> t1 -> t2: at the training cameras the rendered frame change on robot/object pixels
                 vs the observed change (mean absolute values and their correlation) and the rendered change on static
                 pixels (flicker); at the domain cameras the rendered frame change;
  feat_*         encodings (slots, eval mode) of the real views and of the self-rendered views: cosine real-real across
                 cameras of a window, real-self-render of the same window, and across windows (negatives); per-dimension
                 std and effective rank of the real states.

    CUDA_VISIBLE_DEVICES=<uuid> python -E -s scripts/selfrender_diagnostics.py \
        --checkpoint <run>/checkpoints/step_0020000.pt <run>/checkpoints/step_0040000.pt --out <dir> \
        [--windows 32] [--split validation]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _bootstrap  # noqa: F401  (sys.path)
from _bootstrap import REPO, guard_gpus

guard_gpus()

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from s4d.data.contract import collate  # noqa: E402
from s4d.data.metaworld.dataset import MetaworldWindowDataset  # noqa: E402
from s4d.losses.invariance import state_statistics  # noqa: E402
from s4d.model.gaussians import GaussianSet  # noqa: E402
from s4d.model.render import render_rgbd  # noqa: E402
from s4d.train.augment import PATCH_COVERAGE, domain_cameras, load_viewpoint_domain  # noqa: E402
from s4d.train.loop import build_model  # noqa: E402

DOMAIN = "configs/metaworld/selfrender_domain_v1.yaml"
DYNAMIC_BODY_M = 0.01
NEAR, FAR = 0.05, 3.0


def psnr(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> float:
    se = ((a - b) ** 2).mean(-3, keepdim=True)
    m = mask.float()
    mse = (se * m).sum() / m.sum().clamp_min(1)
    return float(-10 * torch.log10(mse.clamp_min(1e-10)))


def corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-12))


@torch.no_grad()
def diagnose(path: Path, windows: int, split: str, domain: dict, device, panel: Path | None) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=False)
    cfg = state["config"]
    model = build_model(cfg).to(device).eval()
    model.encoder.load_state_dict(state["encoder"])
    model.decoder.load_state_dict(state["decoder"])
    root, task = Path(cfg["data"]["root"]), cfg["data"]["task"]
    data = root / f"{task}.hdf5"
    manifest = json.loads((root / "splits" / f"{task}_seed{int(cfg['data'].get('split_seed', 0))}.json").read_text())
    ds = MetaworldWindowDataset(data, manifest[split], strides=(2,))
    rng = np.random.default_rng(0)
    picks = sorted(rng.choice(len(ds), min(windows, len(ds)), replace=False).tolist())
    with h5py.File(data, "r") as f:
        moving = None
        for ep in manifest["train"]:
            xpos = f["episodes"][ep]["xpos"][:]
            span = np.linalg.norm(xpos.max(0) - xpos.min(0), axis=-1) > DYNAMIC_BODY_M
            moving = span if moving is None else moving | span
        dyn_ids = torch.as_tensor(np.nonzero(moving)[0], device=device)
        bodies = [torch.from_numpy(f["episodes"][ds.samples[i][0]]["body_id"][[ds.samples[i][1] + 2 * k for k in range(3)]]
                                   .astype(np.int64)) for i in picks]
    acc: dict[str, list] = {}
    real_states, sr_states = [], []
    gen = torch.Generator().manual_seed(1234)
    for c0 in range(0, len(picks), 8):
        idx = picks[c0 : c0 + 8]
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in collate([ds[i] for i in idx]).items()}
        body = torch.stack(bodies[c0 : c0 + 8]).to(device)  # (B,T,V,H,W)
        images = batch["images"].float() / 255.0  # (B,T,V,3,H,W)
        B, T, V, _, H, W = images.shape
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            slots = model.encoder(images.permute(0, 2, 1, 3, 4, 5).reshape(B * V, T, 3, H, W), None, mask_ratio=0.0)["slots"]
            slots = slots.view(B, V, *slots.shape[1:])
            gs = model.decoder(slots[:, 0])
        gs = GaussianSet(*(getattr(gs, k).float() for k in ("xyz", "scales", "quats", "opacity", "rgb", "delta01",
                                                             "delta12")), gs.group)
        xyz = gs.xyz_sequence()
        r = render_rgbd(gs, xyz, batch["w2c"], batch["K"], H, W, NEAR, FAR)
        rgb, alpha, depth = r["rgb"].clamp(0, 1), r["alpha"], r["depth"]
        dyn = torch.isin(body, dyn_ids)[:, :, :, None]  # (B,T,V,1,H,W)
        gt_depth = batch["depth"]
        valid = (gt_depth > 0) & (gt_depth < FAR)
        acc.setdefault("recon_psnr_robot_object", []).append(psnr(rgb, images, dyn))
        acc.setdefault("recon_psnr_rest", []).append(psnr(rgb, images, ~dyn))
        acc.setdefault("recon_alpha_robot_object", []).append(float(alpha[dyn].mean()))
        m = dyn & valid
        acc.setdefault("recon_depth_absrel_robot_object", []).append(
            float(((depth - gt_depth).abs() / gt_depth.clamp_min(1e-6))[m].mean()))
        # temporal coherence at the training cameras
        d_gt = (images[:, 1:] - images[:, :-1]).abs().mean(3, keepdim=True)
        d_r = (rgb[:, 1:] - rgb[:, :-1]).abs().mean(3, keepdim=True)
        dm, sm = dyn[:, 1:] | dyn[:, :-1], ~(dyn[:, 1:] | dyn[:, :-1])
        acc.setdefault("temporal_change_robot_object_rendered", []).append(float(d_r[dm].mean()))
        acc.setdefault("temporal_change_robot_object_observed", []).append(float(d_gt[dm].mean()))
        acc.setdefault("temporal_change_robot_object_corr", []).append(corr(d_r[dm], d_gt[dm]))
        acc.setdefault("temporal_flicker_static_rendered", []).append(float(d_r[sm].mean()))
        acc.setdefault("temporal_flicker_static_observed", []).append(float(d_gt[sm].mean()))
        # self-render from the frozen domain (fixed seed)
        state_rng = torch.get_rng_state()
        torch.set_rng_state(gen.get_state())
        cams = domain_cameras(B, 2, batch["K"][:, 0], H, W, domain)
        gen.set_state(torch.get_rng_state())
        torch.set_rng_state(state_rng)
        s = render_rgbd(gs, xyz, cams["w2c"], cams["K"], H, W, NEAR, FAR)
        s_rgb, s_alpha = s["rgb"].clamp(0, 1), s["alpha"]  # (B,T,2,...)
        frac = F.avg_pool2d(s_alpha.permute(0, 2, 1, 3, 4, 5).reshape(-1, 1, H, W), model.encoder.cfg.patch_size)
        eligible = (frac >= PATCH_COVERAGE).view(B * 2, -1)
        keep = int(round(eligible.shape[1] * (1 - model.encoder.cfg.mask_ratio)))
        acc.setdefault("sr_alpha_mean", []).append(float(s_alpha.mean()))
        acc.setdefault("sr_token_eligible_fraction", []).append(float(eligible.float().mean()))
        acc.setdefault("sr_coverage_exception_views", []).append(float((eligible.sum(1) < keep).float().mean()))
        acc.setdefault("temporal_change_domain_rendered", []).append(float((s_rgb[:, 1:] - s_rgb[:, :-1]).abs().mean()))
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            s_in = s_rgb.permute(0, 2, 1, 3, 4, 5).reshape(B * 2, T, 3, H, W)
            s_slots = model.encoder(s_in, None, mask_ratio=0.0)["slots"]
        real_states.append(slots.float().mean(2))  # (B,V,D)
        sr_states.append(s_slots.float().view(B, 2, *s_slots.shape[1:]).mean(2))
        if panel is not None and c0 == 0:
            from PIL import Image

            rows = [images[0, 0], rgb[0, 0]]  # (V,3,H,W) observed / reconstructed at t0
            rows += [torch.cat([s_rgb[0, t], torch.zeros(V - 2, 3, H, W, device=device)]) for t in range(T)]
            grid = torch.cat([torch.cat(list(r_), -1) for r_ in rows], -2)
            Image.fromarray((grid.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).save(panel)
    real = F.normalize(torch.cat(real_states), dim=-1)  # (N,V,D)
    sr = F.normalize(torch.cat(sr_states), dim=-1)
    N, V = real.shape[:2]
    rr = torch.einsum("nvd,nwd->nvw", real, real)
    off = ~torch.eye(V, dtype=torch.bool, device=device)
    flat = real.reshape(N * V, -1)
    same = (torch.arange(N * V, device=device)[:, None] // V) == (torch.arange(N * V, device=device)[None] // V)
    out = {k: float(np.mean(v)) for k, v in acc.items()}
    out.update({
        "feat_cos_real_real_same_window": float(rr[:, off].mean()),
        "feat_cos_real_selfrender_same_window": float(torch.einsum("nvd,nwd->nvw", real, sr).mean()),
        "feat_cos_real_other_windows": float((flat @ flat.t())[~same].mean()),
        **{f"feat_{k}": float(v) for k, v in state_statistics(real.reshape(N * V, -1)).items()},
        "step": int(state["step"]), "task": task, "windows": len(picks), "split": split,
        "variant_aug": {k: v for k, v in (cfg.get("aug") or {}).items()},
    })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--windows", type=int, default=32)
    ap.add_argument("--split", default="validation", choices=("validation", "train"))
    ap.add_argument("--domain", default=DOMAIN)
    args = ap.parse_args()
    out = Path(args.out)
    out = out if out.is_absolute() else REPO / out
    out.mkdir(parents=True, exist_ok=True)
    domain = load_viewpoint_domain(args.domain)
    device = torch.device("cuda")
    results = {}
    for ckpt in args.checkpoint:
        path = Path(ckpt)
        name = f"{path.parent.parent.name}_{path.stem}"
        results[name] = diagnose(path, args.windows, args.split, domain, device, out / f"{name}.jpg")
        print(name, json.dumps(results[name]), flush=True)
    (out / "selfrender_diagnostics.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
