"""3D-MVP pretraining on one Meta-World task (Qi et al., CVPR 2025, §3.2; reimplemented, no official code).

Point clouds of random training-camera subsets of training-split time steps are rendered into RVT-1's five virtual
views (s4d/baselines/mvp3d/renderer.py), 75 % of each view's patches have their RGB set to 0, and the RVT-1 MVT
encoder E with a lightweight MAE decoder reconstructs the RGB-D of all views (Eq. 6). AdamW, lr 1e-4, weight decay
0.01, batch 3, 15 epochs (one epoch = one pass over the task's training time steps). Every setting comes from
``configs/baselines/mvp3d/<task>.yaml``; see docs/baselines_mvp3d.md.

Outputs under ``<output-root>/<name>``: config.yaml, provenance.json, metrics.jsonl, eval/*.png,
checkpoints/latest.pt (model, optimizer, position in the epoch, every RNG state), encoder.pt (the RL export) and
encoder_init.pt (the same encoder at its random initialisation, written once at the start of a fresh run).
``--resume auto`` (default) continues losslessly from checkpoints/latest.pt: the data order, camera subsets and masks
are functions of (seed, epoch, position, step), and dropout's CUDA RNG state is restored.

    CUDA_VISIBLE_DEVICES=<uuid> python -I scripts/train_mvp3d.py --config configs/baselines/mvp3d/hammer.yaml \
        --name mvp3d-hammer
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import REPO, guard_gpus, require_passed_tests  # noqa: E402

GPU_MAPPING = guard_gpus()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader, default_collate  # noqa: E402

from s4d.baselines.data import split_episodes  # noqa: E402
from s4d.baselines.mvp3d.data import EpochBatches, MVPTimeSteps, batch_to_device, fixed_items  # noqa: E402
from s4d.baselines.mvp3d.model import MVP3D, MVP3DConfig, random_view_mask, save_export  # noqa: E402
from s4d.baselines.mvp3d.renderer import render_rgbd  # noqa: E402
from s4d.baselines.mvp3d.viz import reconstruction_panel  # noqa: E402
from s4d.config import dump_config, load_config  # noqa: E402
from s4d.diag.wandb_log import init_wandb  # noqa: E402
from ops.resources.gpu_guard import GPUIsolationError, validate_runtime_path  # noqa: E402
from ops.jobs.run_identity import record_run_identity  # noqa: E402

SHORT_RUN_STEPS = 1000


def step_generator(device: torch.device, seed: int, step: int) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed * 1_000_003 + step)


def render(inputs: dict, render_cfg: dict, img_size: int) -> torch.Tensor:
    with torch.no_grad():
        return render_rgbd(
            inputs["rgb"], inputs["depth"], inputs["K"], inputs["c2w"], render_cfg["bounds"], img_size,
            render_cfg["radius"], view_mask=inputs["view_mask"],
        )


@torch.no_grad()
def validate(model: MVP3D, dataset, items, render_cfg: dict, batch_size: int, device, out_png: Path) -> dict:
    """Eq. 6 on fixed validation samples (fixed camera subsets and masks); panel of the first sample."""
    model.eval()
    cfg = model.cfg
    g = cfg.img_size // cfg.img_patch_size
    totals, count = {"loss": 0.0, "loss_rgb": 0.0, "loss_depth": 0.0}, 0
    for first in range(0, len(items), batch_size):
        chunk = items[first : first + batch_size]
        images = render(batch_to_device(default_collate([dataset[i] for i in chunk]), device), render_cfg, cfg.img_size)
        mask = random_view_mask(len(chunk), cfg.num_img, g, cfg.mask_ratio, step_generator(device, 7, first), device)
        out = model(images, mask)
        for key in totals:
            totals[key] += float(out[key]) * len(chunk)
        count += len(chunk)
        if first == 0:
            out_png.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(reconstruction_panel(images[0], out["masked"][0], out["pred"][0])).save(out_png)
    model.train()
    return {f"val/{k}": v / count for k, v in totals.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--output-root", default=str(REPO / "runs/pretrain"))
    ap.add_argument("--set", nargs="*", default=[], help="config overrides key.path=value")
    ap.add_argument("--resume", choices=("auto", "never"), default="auto",
                    help="auto: continue from checkpoints/latest.pt when present; never: refuse an existing run")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    cfg = load_config([args.config], args.set)
    ds_cfg, train_cfg, interface = dict(cfg["dataset"]), dict(cfg["train"]), dict(cfg["data_interface"])
    render_cfg = {"bounds": [float(b) for b in cfg["render"]["bounds"]], "radius": float(cfg["render"]["radius"])}
    model_cfg = MVP3DConfig(**cfg["model"])
    seed, batch_size = int(train_cfg["seed"]), int(train_cfg["batch_size"])

    manifest = interface["split_manifest"]
    num_views = int(ds_cfg["num_views"])
    train_set = MVPTimeSteps(ds_cfg["hdf5_path"], split_episodes(manifest, "train"), num_views,
                             ds_cfg.get("max_episodes"), ds_cfg.get("max_frames_per_demo"))
    val_set = MVPTimeSteps(ds_cfg["hdf5_path"], split_episodes(manifest, "validation"), num_views)
    val_items = fixed_items(len(val_set), int(train_cfg["val_samples"]))
    steps_per_epoch = len(train_set) // batch_size
    max_steps = train_cfg.get("max_steps")
    total_steps = int(max_steps) if max_steps is not None else int(train_cfg["epochs"]) * steps_per_epoch
    if total_steps > SHORT_RUN_STEPS:
        require_passed_tests()

    try:
        run_dir = validate_runtime_path(Path(args.output_root) / args.name, repository=REPO)
    except GPUIsolationError as exc:
        raise ValueError("pretraining outputs must stay inside an authorized runtime root") from exc
    latest = run_dir / "checkpoints" / "latest.pt"
    if args.resume == "never" and latest.is_file():
        raise SystemExit(f"{latest} exists and --resume never was given")
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    cfg["run"] = {"gpus": GPU_MAPPING, "dir": str(run_dir), "steps_per_epoch": steps_per_epoch,
                  "total_steps": total_steps, "train_samples": len(train_set)}
    record_run_identity(cfg, REPO, run_dir, native_diagnostic_steps=total_steps)
    dump_config(cfg, run_dir / "config.yaml")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # cuDNN's default convolution backward is not bitwise reproducible; the deterministic kernels make a resumed run
    # equal an uninterrupted one (the tiny convolutions cost no measurable time)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(train_cfg["device"])
    model = MVP3D(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["lr"]),
                                  weight_decay=float(train_cfg["weight_decay"]))
    model_dict = asdict(model_cfg)
    global_step, epoch, batch_in_epoch = 0, 0, 0
    if latest.is_file():
        ckpt = torch.load(latest, map_location="cpu", weights_only=False)  # own checkpoint (RNG payload)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        global_step, epoch, batch_in_epoch = int(ckpt["global_step"]), int(ckpt["epoch"]), int(ckpt["batch_in_epoch"])
        random.setstate(ckpt["python_rng"])
        np.random.set_state(ckpt["numpy_rng"])
        torch.set_rng_state(ckpt["torch_rng"])
        torch.cuda.set_rng_state(ckpt["cuda_rng"])
    else:  # the pretrained-vs-random check uses exactly this initialisation
        save_export(run_dir / "encoder_init.pt", model.encoder, model_dict, render_cfg, 0, random_init=True)

    wandb_run = init_wandb(cfg, args.name, cfg["wandb"]["project"], not args.no_wandb, run_dir)
    metrics_file = open(run_dir / "metrics.jsonl", "a")

    def log(record: dict) -> None:
        metrics_file.write(json.dumps({"step": global_step, **record}) + "\n")
        metrics_file.flush()
        if wandb_run is not None:
            wandb_run.log(record, step=global_step)

    def save() -> None:
        tmp = latest.with_suffix(".tmp")
        torch.save(
            {
                "global_step": global_step,
                "epoch": epoch,
                "batch_in_epoch": batch_in_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(),
            },
            tmp,
        )
        tmp.replace(latest)
        save_export(run_dir / "encoder.pt", model.encoder, model_dict, render_cfg, global_step)

    g = model_cfg.img_size // model_cfg.img_patch_size
    log_every, eval_every, save_every = (int(train_cfg[k]) for k in ("log_every", "eval_every", "save_every"))
    workers = int(ds_cfg["num_workers"])
    window = {"loss": [], "loss_rgb": [], "loss_depth": [], "points": []}
    tick = time.time()
    model.train()
    while global_step < total_steps:
        sampler = EpochBatches(len(train_set), batch_size, seed, epoch, start=batch_in_epoch)
        loader = DataLoader(train_set, batch_sampler=sampler, num_workers=workers,
                            pin_memory=bool(ds_cfg["pin_memory"]), prefetch_factor=4 if workers else None)
        for batch in loader:
            if global_step >= total_steps:
                break
            inputs = batch_to_device(batch, device)
            images = render(inputs, render_cfg, model_cfg.img_size)
            mask = random_view_mask(batch_size, model_cfg.num_img, g, model_cfg.mask_ratio,
                                    step_generator(device, seed, global_step), device)
            out = model(images, mask)
            optimizer.zero_grad(set_to_none=True)
            out["loss"].backward()
            optimizer.step()
            for key in ("loss", "loss_rgb", "loss_depth"):
                window[key].append(float(out[key].detach()))
            window["points"].append(float((images[:, 0, 6] != -1).float().mean()))
            if global_step % log_every == 0:
                elapsed = time.time() - tick
                tick = time.time()
                log(
                    {
                        "train/loss": window["loss"][-1],
                        **{f"train/{k}_avg": float(np.mean(v)) for k, v in window.items() if k != "points"},
                        "train/top_view_coverage": float(np.mean(window["points"])),
                        "train/views_per_sample": float(inputs["view_mask"].float().sum(1).mean()),
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "train/epoch": epoch,
                        "train/step_time": elapsed / (log_every if global_step else 1),
                        "train/gpu_mem_gb": torch.cuda.max_memory_allocated() / 2**30,
                    }
                )
                window = {k: [] for k in window}
            if eval_every > 0 and global_step > 0 and global_step % eval_every == 0:
                out_png = run_dir / "eval" / f"step_{global_step:08d}.png"
                log(validate(model, val_set, val_items, render_cfg, batch_size, device, out_png))
            global_step += 1
            batch_in_epoch += 1
            if save_every > 0 and global_step % save_every == 0:
                save()
        else:
            epoch += 1
            batch_in_epoch = 0
    log(validate(model, val_set, val_items, render_cfg, batch_size, device,
                 run_dir / "eval" / f"step_{global_step:08d}.png"))
    save()
    (run_dir / "completion.json").write_text(json.dumps({"status": "completed", "steps": global_step}))
    metrics_file.close()
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
