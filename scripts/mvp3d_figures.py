"""Sanity panels for the 3D-MVP port (docs/figures/baselines/mvp3d/).

- ``input_cameras.png``: the six training-camera RGB images of the chosen time step;
- ``views_6cam.png`` / ``views_3cam.png`` / ``views_1cam.png``: the five virtual views (rows RGB, depth, xyz) of the
  fused cloud of all six training cameras (the pretraining input), of three cameras (RL condition A) and of one
  camera (RL condition B), at the model's resolution;
- ``views_6cam_220px.png``: the same cloud at RVT's / the paper's 220 px, to show the renderer's conventions;
- ``reconstruction.png`` (with ``--checkpoint``): validation samples (6-, 3- and 1-camera clouds), rows RGB,
  masked RGB input, reconstruction, depth, reconstructed depth.

    CUDA_VISIBLE_DEVICES=<uuid> python -I scripts/mvp3d_figures.py --config configs/baselines/mvp3d/hammer.yaml \
        --checkpoint runs/smoke/mvp3d-hammer-smoke/checkpoints/latest.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import REPO, guard_gpus  # noqa: E402

guard_gpus()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import default_collate  # noqa: E402

from s4d.baselines.data import split_episodes  # noqa: E402
from s4d.baselines.mvp3d.data import MVPTimeSteps, batch_to_device  # noqa: E402
from s4d.baselines.mvp3d.model import MVP3D, MVP3DConfig, random_view_mask  # noqa: E402
from s4d.baselines.mvp3d.renderer import render_rgbd  # noqa: E402
from s4d.baselines.mvp3d.viz import grid, reconstruction_panel, views_panel  # noqa: E402
from s4d.config import load_config  # noqa: E402


def sample_with_cameras(ds: MVPTimeSteps, index: int, cameras: list[int]) -> dict:
    """Sample ``index`` restricted to ``cameras`` (pretraining samples always fuse all six)."""
    s = ds[(index, 0)]
    group = ds._episode(s["episode"])
    s["view_mask"][:] = False
    s["view_mask"][cameras] = True
    s["rgb"][:] = 0
    s["depth"][:] = 0
    s["rgb"][cameras] = torch.from_numpy(group["rgb"][s["t"], cameras])
    s["depth"][cameras] = torch.from_numpy(group["depth"][s["t"], cameras].astype(np.float32) * ds.depth_unit_m)
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--index", type=int, default=40, help="time step index within the split")
    ap.add_argument("--single-camera", type=int, default=1, help="training camera of the single-camera cloud (B)")
    ap.add_argument("--three-cameras", default="1,2,4", help="training cameras of the 3-camera cloud (A)")
    ap.add_argument("--out", default=str(REPO / "docs/figures/baselines/mvp3d"))
    args = ap.parse_args()

    cfg = load_config([args.config])
    model_cfg = MVP3DConfig(**cfg["model"])
    bounds, radius = cfg["render"]["bounds"], float(cfg["render"]["radius"])
    ds = MVPTimeSteps(cfg["dataset"]["hdf5_path"], split_episodes(cfg["data_interface"]["split_manifest"], args.split))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    full = sample_with_cameras(ds, args.index, list(range(6)))
    three = sample_with_cameras(ds, args.index, [int(c) for c in args.three_cameras.split(",")])
    single = sample_with_cameras(ds, args.index, [args.single_camera])
    print(f"sample {full['episode']} t={full['t']}")
    cams = [full["rgb"][v].numpy() for v in range(6)]
    Image.fromarray(grid([cams], scale=1)).save(out / "input_cameras.png")
    batch = batch_to_device(default_collate([full, three, single]), device)

    def views(size: int) -> torch.Tensor:
        return render_rgbd(batch["rgb"], batch["depth"], batch["K"], batch["c2w"], bounds, size, radius,
                           view_mask=batch["view_mask"]).cpu()

    images = views(model_cfg.img_size)
    Image.fromarray(views_panel(images[0])).save(out / "views_6cam.png")
    Image.fromarray(views_panel(images[1])).save(out / "views_3cam.png")
    Image.fromarray(views_panel(images[2])).save(out / "views_1cam.png")
    Image.fromarray(views_panel(views(220)[0], scale=1)).save(out / "views_6cam_220px.png")
    if args.checkpoint:
        model = MVP3D(model_cfg).to(device).eval()
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        g = model_cfg.img_size // model_cfg.img_patch_size
        gen = torch.Generator(device=device).manual_seed(0)
        mask = random_view_mask(3, model_cfg.num_img, g, model_cfg.mask_ratio, gen, device)
        with torch.no_grad():
            res = model(images.to(device), mask)
        panels = [reconstruction_panel(images[i], res["masked"][i].cpu(), res["pred"][i].cpu()) for i in range(3)]
        gap = np.full((8, panels[0].shape[1], 3), 255, np.uint8)
        Image.fromarray(np.concatenate([panels[0], gap, panels[1], gap, panels[2]])).save(out / "reconstruction.png")
        print(f"checkpoint step {ckpt['global_step']}: loss {float(res['loss']):.4f}")


if __name__ == "__main__":
    main()
