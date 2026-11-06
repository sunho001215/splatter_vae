"""SNeRL validation on the real-world rig (user decision 2026-11-06 ~11:20: 3 input cameras; validation on SEPARATE
episodes; input-view and novel-view reconstruction reported SEPARATELY).

* Episodes: the validation episodes of the run's split manifest (``split_seed0.json``: seeded per-task 10 % of the
  episodes, ``s4d.rwlearn.store.split_episodes``; fixed, recorded, never trained on). Scenes: the fixed official draws
  of the validation dataset (``RealworldSNeRLScenes.draw(k)``, seed + 1): episode, step and target camera.
* ``val/input_view/*`` - INPUT-VIEW reconstruction, the training condition: all 3 cameras are encoder inputs and the
  image of one of them (the drawn target camera) is rendered. Measures how well the latent reproduces views it saw.
* ``val/novel_view_loco/*`` - NOVEL-VIEW reconstruction by LEAVE-ONE-CAMERA-OUT at evaluation only: the drawn target
  camera is removed from the encoder inputs (the other 2 cameras are encoded; SNeRL's encoder mean-pools per-view
  features, so it accepts 2 views) and the held-out camera is rendered. The rig has exactly 3 cameras, so this is the
  only novel view with ground truth. Caveat (label it in every report): the encoder was trained with 3 inputs, so this
  measures generalisation to an unseen viewpoint under one fewer input view, not Meta-World's 3-of-6 held-out-view
  protocol, and is not comparable to it.

Metrics of both (same as the Meta-World validation, ``scripts/train_snerl.py::validate``): PSNR of the rendered RGB,
semantic accuracy against the SAM 3 classes, MSE to the online DINO teacher target, depth MAE (m) on valid sensor depth.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

PROTOCOLS = ("input_view", "novel_view_loco")
METRICS = ("psnr", "semantic_acc", "dino_mse", "depth_mae_m")


def protocol_views(num_cams: int, target: int) -> dict:
    """Encoder input cameras and the rendered camera of each protocol for one drawn target camera."""
    if num_cams < 2:
        raise ValueError("novel-view evaluation needs at least 2 cameras")
    return {"input_view": (list(range(num_cams)), target),
            "novel_view_loco": ([c for c in range(num_cams) if c != target], target)}


@torch.no_grad()
def realworld_validate(model, extractor, val_set, n: int, chunk: int, out_png: Path | None) -> dict:
    from s4d.baselines.snerl import viz
    from s4d.baselines.snerl.model import dino_target, render_camera, to_device_scene

    model.train(False)
    metrics = {p: {m: [] for m in METRICS} for p in PROTOCOLS}
    rows = []
    for k in range(n):
        ep, t, _, target = val_set.draw(k)
        row = []
        for proto, (inputs, cam) in protocol_views(val_set.num_cams, target).items():
            scene = to_device_scene(val_set.scene(ep, t, inputs, cam, with_depth=True), model.device)
            latent = model.encoder(scene["obs_rgb"].permute(0, 3, 1, 2), scene["obs_c2w"].reshape(-1, 16))
            H, W = scene["target_rgb"].shape[:2]
            K_cv, c2w_cv = val_set.camera(ep, cam)
            out = render_camera(model, latent, K_cv, c2w_cv, H, W, chunk, val_set.lookat, val_set.scale)
            mse = torch.mean((out["rgb"] - scene["target_rgb"]) ** 2)
            m = metrics[proto]
            m["psnr"].append(float(-10.0 * torch.log10(mse)))
            m["semantic_acc"].append(float((out["semantic"].argmax(-1) == scene["target_semantic"]).float().mean()))
            teacher = dino_target(extractor, scene["target_rgb"])
            m["dino_mse"].append(float(torch.mean((out["dino"] - teacher) ** 2)))
            gt_depth = scene["target_depth"]
            valid = gt_depth > 0
            m["depth_mae_m"].append(float((out["depth"] - gt_depth).abs()[valid].mean()))
            if out_png is not None and k < 2:
                to_np = lambda x: (x.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)  # noqa: E731
                row += [viz.tile(to_np(scene["target_rgb"]), f"GT cam {cam}"),
                        viz.tile(to_np(out["rgb"]), f"{proto} in {inputs}"),
                        viz.tile(viz.class_image(out["semantic"].argmax(-1).cpu().numpy()), f"{proto} classes")]
        if row:
            rows.append(row)
    if rows:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        from PIL import Image

        Image.fromarray(viz.grid(rows)).save(out_png)
    model.train(True)
    return {f"val/{p}/{m}": float(np.mean(v)) for p, ms in metrics.items() for m, v in ms.items()}
