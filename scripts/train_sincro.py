"""SinCro pretraining on one Meta-World task: reference trainer logic, splatter4d data interface and outputs.

The training step, encoding and rendering are the reference ``baselines/SinCro/train.py`` functions
(``s4d/baselines/sincro/training.py``); this entry point replaces only data loading (our HDF5 layout and saved
split manifest), output locations, logging, checkpoint/resume and the encoder export. Every hyperparameter comes
from ``configs/baselines/sincro/<task>.yaml`` (the reference configs). See docs/BASELINES.md.

    CUDA_VISIBLE_DEVICES=<uuid> python -I scripts/train_sincro.py --config configs/baselines/sincro/hammer.yaml \
        --name sincro-hammer
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
from torch.utils.data import DataLoader  # noqa: E402

import s4d.baselines.sincro.nerf as sincro_nerf  # noqa: E402
from s4d.baselines.data import SinCroWindows, split_episodes  # noqa: E402
from s4d.baselines.sincro.encoder import save_export  # noqa: E402
from s4d.baselines.sincro.training import (  # noqa: E402
    DatasetConfig,
    ExperimentConfig,
    SimpleArgs,
    SinCroModelConfig,
    TrainConfig,
    encode_sincro,
    forward_sincro_batch,
    render_full_image,
    update_learning_rate,
)
from s4d.config import dump_config, load_config  # noqa: E402
from s4d.diag.wandb_log import init_wandb  # noqa: E402

SHORT_RUN_STEPS = 1000


@torch.no_grad()
def validate(dataset, latent_embed, render_kwargs_test, args, model_cfg, rng, out_png: Path) -> dict:
    """Reference ``run_validation``: encode one random validation window without masking and render every view."""
    latent_embed.eval()
    sample = dataset[int(rng.integers(len(dataset)))]
    device = next(latent_embed.parameters()).device
    images = sample["images"][None].to(device)  # (1, T, H, V, W, 3)
    K, c2w = sample["K"].to(device), sample["c2w"].to(device)
    _, T, H, V, W, _ = images.shape
    latent, _, _, primary, refs = encode_sincro(images, latent_embed, model_cfg, mask_ratio=0.0)
    test_latent = latent.reshape(1, T, 1, -1)[:, -1, 0]
    rendered, gt, psnr = [], [], []
    for v in range(V):
        pred = render_full_image(H, W, K[v], c2w[v], latent=test_latent, args=args, render_kwargs=render_kwargs_test)
        target = (images[0, -1, :, v].cpu().numpy() * 255).astype(np.uint8)
        mse = np.mean((pred.astype(np.float32) / 255 - target.astype(np.float32) / 255) ** 2)
        rendered.append(pred)
        gt.append(target)
        psnr.append(float(-10 * np.log10(mse + 1e-8)))
    grid = np.concatenate([np.concatenate(gt, axis=1), np.concatenate(rendered, axis=1)], axis=0)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(out_png)
    latent_embed.train()
    return {
        "val/mean_psnr": float(np.mean(psnr)),
        "val/min_psnr": min(psnr),
        "val/max_psnr": max(psnr),
        "val/primary_view": primary,
        "val/ref_views": refs,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--output-root", default=str(REPO / "runs/pretrain"))
    ap.add_argument("--set", nargs="*", default=[], help="config overrides key.path=value")
    ap.add_argument("--no-wandb", action="store_true")
    args_cli = ap.parse_args()

    cfg = load_config([args_cli.config], args_cli.set)
    ds_cfg = DatasetConfig(**cfg["dataset"])
    model_cfg = SinCroModelConfig(**cfg["model"])
    train_cfg = TrainConfig(**cfg["train"])
    interface = dict(cfg["data_interface"])
    model_cfg.num_views = ds_cfg.num_views
    if train_cfg.max_global_steps is not None and train_cfg.max_global_steps > SHORT_RUN_STEPS:
        require_passed_tests()
    run_dir = (Path(args_cli.output_root) / args_cli.name).resolve()
    if REPO.resolve() not in run_dir.parents:
        raise ValueError("pretraining outputs must stay inside the repository")
    (run_dir / "nerf").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    cfg["run"] = {"gpus": GPU_MAPPING, "dir": str(run_dir)}
    dump_config(cfg, run_dir / "config.yaml")

    torch.manual_seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)
    random.seed(train_cfg.seed)
    device = torch.device(train_cfg.device)
    sincro_nerf.device = device  # as in the reference trainer: the upstream module renders on its global device

    frame_spacing = int(interface["frame_spacing"])
    window_args = {
        "sequence_length": ds_cfg.sequence_length,
        "temporal_stride": ds_cfg.temporal_stride,
        "frame_spacing": frame_spacing,
        "num_views": ds_cfg.num_views,
    }
    train_set = SinCroWindows(ds_cfg.hdf5_path, split_episodes(interface["split_manifest"], "train"), **window_args)
    val_set = SinCroWindows(ds_cfg.hdf5_path, split_episodes(interface["split_manifest"], "validation"), **window_args)
    loader = DataLoader(
        train_set,
        batch_size=ds_cfg.batch_size,
        shuffle=True,
        num_workers=ds_cfg.num_workers,
        pin_memory=ds_cfg.pin_memory,
        drop_last=True,
        persistent_workers=ds_cfg.num_workers > 0,
    )

    exp_cfg = ExperimentConfig(basedir=str(run_dir), expname="nerf")
    simple_args = SimpleArgs(model_cfg, train_cfg, ds_cfg, exp_cfg)
    render_kwargs_train, render_kwargs_test, _, _, optimizer, latent_embed = sincro_nerf.create_nerf(
        simple_args, simple_args.basedir, simple_args.expname
    )
    # Upstream MV_run_nerf.train() adds the scene bounds to the render kwargs; the reference trainer never did, so
    # it rendered within the default [0, 1] m. The reference config's bounds are applied here.
    bounds = {"near": float(model_cfg.near), "far": float(model_cfg.far)}
    render_kwargs_train.update(bounds)
    render_kwargs_test.update(bounds)
    latent_embed.to(device)

    latest = run_dir / "checkpoints" / "latest.pt"
    global_step, epoch = 0, 0
    if latest.is_file():
        ckpt = torch.load(latest, map_location="cpu", weights_only=False)  # own checkpoint (RNG payload)
        render_kwargs_train["network_fn"].load_state_dict(ckpt["network_fn_state_dict"])
        render_kwargs_train["network_fine"].load_state_dict(ckpt["network_fine_state_dict"])
        latent_embed.load_state_dict(ckpt["latent_embed_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        global_step, epoch = int(ckpt["global_step"]), int(ckpt["epoch"])
        np.random.set_state(ckpt["numpy_rng"])
        torch.set_rng_state(ckpt["torch_rng"])
        torch.cuda.set_rng_state(ckpt["cuda_rng"])
        random.setstate(ckpt["python_rng"])

    wandb_run = init_wandb(cfg, args_cli.name, cfg["wandb"]["project"], not args_cli.no_wandb, run_dir)
    metrics_file = open(run_dir / "metrics.jsonl", "a")
    val_rng = np.random.default_rng(train_cfg.seed + global_step)

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
                "network_fn_state_dict": render_kwargs_train["network_fn"].state_dict(),
                "network_fine_state_dict": render_kwargs_train["network_fine"].state_dict(),
                "latent_embed_state_dict": latent_embed.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(),
                "python_rng": random.getstate(),
            },
            tmp,
        )
        tmp.replace(latest)
        save_export(run_dir / "encoder.pt", latent_embed, asdict(model_cfg), frame_spacing, global_step)

    max_steps = train_cfg.max_global_steps
    tick = time.time()
    while max_steps is None or global_step < max_steps:
        for batch in loader:
            if max_steps is not None and global_step >= max_steps:
                break
            latent_embed.train()
            batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
            loss, stats = forward_sincro_batch(
                batch, latent_embed, render_kwargs_train, simple_args, model_cfg, global_step=global_step
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr = update_learning_rate(optimizer, train_cfg, global_step)
            if global_step % train_cfg.i_print == 0:
                elapsed = time.time() - tick
                tick = time.time()
                log(
                    {
                        **{f"train/{k}": v for k, v in stats.items()},
                        "train/lr": lr,
                        "train/epoch": epoch,
                        "train/step_time": elapsed / (train_cfg.i_print if global_step else 1),
                        "train/gpu_mem_gb": torch.cuda.max_memory_allocated() / 2**30,
                    }
                )
            if train_cfg.eval_every > 0 and global_step > 0 and global_step % train_cfg.eval_every == 0:
                log(
                    validate(
                        val_set,
                        latent_embed,
                        render_kwargs_test,
                        simple_args,
                        model_cfg,
                        val_rng,
                        run_dir / "eval" / f"step_{global_step:08d}.png",
                    )
                )
            global_step += 1
            if train_cfg.save_every > 0 and global_step % train_cfg.save_every == 0:
                save()
        else:
            epoch += 1
            if max_steps is None and epoch >= train_cfg.num_epochs:
                break
    save()
    (run_dir / "completion.json").write_text(json.dumps({"status": "completed", "steps": global_step}))
    metrics_file.close()
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
