"""Capacity ceiling of the Gaussian budget: fit free Gaussians directly to one frame (no encoder, no decoder).

Initialised from the lifted ground-truth depth of the six training cameras and optimised with the training
renderer and the RGB L1 + 0.2 D-SSIM loss, this bounds the PSNR the decoder could reach with the same number of
Gaussians on the same views. Writes ``<out>/capacity.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import REPO, guard_gpus  # noqa: E402

GPU_MAPPING = guard_gpus()

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from s4d.data.metaworld.dataset import MetaworldWindowDataset  # noqa: E402
from s4d.geometry import lift_depth  # noqa: E402
from s4d.losses.rgb import rgb_loss  # noqa: E402
from s4d.model.gaussians import SCENE_GROUP, GaussianSet  # noqa: E402
from s4d.model.render import render_rgbd, require_prebuilt_renderer  # noqa: E402


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(-10 * torch.log10(F.mse_loss(pred, target)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path", required=True)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--count", type=int, default=8192, help="number of Gaussians (method budget: 768*8 + 256*8)")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out).resolve()
    if REPO.resolve() not in out.parents:
        raise ValueError("outputs must stay inside the repository")
    out.mkdir(parents=True, exist_ok=True)
    require_prebuilt_renderer()
    torch.manual_seed(0)
    dev = torch.device("cuda")
    sample = MetaworldWindowDataset(args.path, [args.episode], strides=(2,), with_eval=True)[0]
    images = sample["images"][0].float().to(dev) / 255  # (V,3,H,W) at t0
    depth = sample["depth"][0, :, 0].to(dev)
    K, w2c, c2w = sample["K"].to(dev), sample["w2c"].to(dev), sample["c2w"].to(dev)
    H, W = images.shape[-2:]
    points = lift_depth(depth, K, c2w)[depth > 0]
    colors = images.permute(0, 2, 3, 1)[depth > 0]
    pick = torch.randperm(len(points), device=dev)[: args.count]
    n = len(pick)
    params = {
        "xyz": points[pick].clone(),
        "log_scale": torch.full((n, 3), -4.6, device=dev),  # 1 cm
        "quat": F.normalize(torch.randn(n, 4, device=dev), dim=-1),
        "opacity_logit": torch.full((n,), 2.0, device=dev),
        "rgb_logit": torch.logit(colors[pick].clamp(0.02, 0.98)),
    }
    for p in params.values():
        p.requires_grad_(True)
    optimiser = torch.optim.Adam(
        [
            {"params": [params["xyz"]], "lr": 1e-3},
            {"params": [params["log_scale"], params["quat"]], "lr": 5e-3},
            {"params": [params["opacity_logit"], params["rgb_logit"]], "lr": 2.5e-2},
        ]
    )
    zeros = torch.zeros(1, n, 3, device=dev)

    def gaussians() -> GaussianSet:
        return GaussianSet(
            params["xyz"][None],
            params["log_scale"].exp().clamp(5e-4, 0.08)[None],
            F.normalize(params["quat"], dim=-1)[None],
            torch.sigmoid(params["opacity_logit"])[None],
            torch.sigmoid(params["rgb_logit"])[None],
            zeros,
            zeros,
            torch.full((n,), SCENE_GROUP, device=dev, dtype=torch.long),
        )

    history = []
    for step in range(1, args.steps + 1):
        gs = gaussians()
        rendered = render_rgbd(gs, gs.xyz[:, None], w2c[None], K[None], H, W, 0.05, 3.0)["rgb"][0, 0]
        loss = rgb_loss(rendered, images, torch.ones_like(images[:, :1]), 0.2).mean()
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        if step % 500 == 0 or step == 1:
            history.append({"step": step, "loss": float(loss), "psnr_train": psnr(rendered.detach().clamp(0, 1), images)})
    with torch.no_grad():
        gs = gaussians()
        train = render_rgbd(gs, gs.xyz[:, None], w2c[None], K[None], H, W, 0.05, 3.0)["rgb"][0, 0].clamp(0, 1)
        eval_images = sample["eval_images"][0].float().to(dev) / 255
        held = render_rgbd(
            gs, gs.xyz[:, None], sample["eval_w2c"].to(dev)[None], sample["eval_K"].to(dev)[None], H, W, 0.05, 3.0
        )["rgb"][0, 0].clamp(0, 1)
    report = {
        "gpu_mapping": GPU_MAPPING,
        "episode": args.episode,
        "count": n,
        "steps": args.steps,
        "psnr_train_cameras": psnr(train, images),
        "psnr_heldout_cameras": psnr(held, eval_images),
        "history": history,
    }
    (out / "capacity.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=1))


if __name__ == "__main__":
    main()
