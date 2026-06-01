from __future__ import annotations

import argparse
import os
import random
import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import wandb
import yaml

from agents.diffusion_policy.dataset import (
    HDF5DiffusionPolicyDataset,
    RandomCameraBatchCollator,
    split_demo_keys,
)
from agents.diffusion_policy.model import DDPMScheduler, DiffusionPolicy, EMAModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _as_image_size(value: Any, cfg: Dict[str, Any]) -> tuple[int, int]:
    if value is None:
        return int(cfg["env"]["image_height"]), int(cfg["env"]["image_width"])
    if isinstance(value, int):
        return int(value), int(value)
    if len(value) != 2:
        raise ValueError(f"dataset.image_size must be an int or [H, W], got {value}")
    return int(value[0]), int(value[1])


def _move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = dict(batch)
    out["image"] = batch["image"].to(device, non_blocking=True)
    out["action"] = batch["action"].to(device, non_blocking=True)
    out["proprio"] = batch["proprio"].to(device, non_blocking=True)
    out["camera_index"] = batch["camera_index"].to(device, non_blocking=True)
    return out


def build_dataloaders(cfg: Dict[str, Any]) -> tuple[DataLoader, DataLoader, HDF5DiffusionPolicyDataset]:
    ds_cfg = cfg["dataset"]
    train_cfg = cfg["training"]
    hdf5_path = str(ds_cfg["hdf5_path"])
    train_keys, val_keys = split_demo_keys(
        hdf5_path,
        val_ratio=float(ds_cfg.get("val_ratio", 0.05)),
        seed=int(cfg.get("seed", 0)),
        max_demos=ds_cfg.get("max_demos", None),
    )

    image_size = _as_image_size(ds_cfg.get("image_size", None), cfg)
    camera_names = ds_cfg.get("camera_names", None)
    horizon = int(cfg["policy"]["horizon"])
    n_obs_steps = int(cfg["policy"]["n_obs_steps"])
    proprio_key = ds_cfg.get("proprio_key", "obs_env/obs")

    train_ds = HDF5DiffusionPolicyDataset(
        hdf5_path=hdf5_path,
        demo_keys=train_keys,
        horizon=horizon,
        n_obs_steps=n_obs_steps,
        camera_names=camera_names,
        proprio_key=proprio_key,
        image_size=image_size,
    )
    val_ds = HDF5DiffusionPolicyDataset(
        hdf5_path=hdf5_path,
        demo_keys=val_keys,
        horizon=horizon,
        n_obs_steps=n_obs_steps,
        camera_names=train_ds.camera_names,
        proprio_key=proprio_key,
        image_size=image_size,
    )

    loader_kwargs = dict(
        batch_size=int(train_cfg.get("batch_size", 64)),
        num_workers=int(train_cfg.get("num_workers", 4)),
        pin_memory=bool(train_cfg.get("pin_memory", True)),
        persistent_workers=bool(train_cfg.get("persistent_workers", False)),
        drop_last=bool(train_cfg.get("drop_last", True)),
    )
    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        collate_fn=RandomCameraBatchCollator(mode=str(ds_cfg.get("train_camera_sampling", "random"))),
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        collate_fn=RandomCameraBatchCollator(
            mode=str(ds_cfg.get("val_camera_sampling", "fixed")),
            fixed_camera_index=int(ds_cfg.get("val_camera_index", 0)),
        ),
        batch_size=int(train_cfg.get("val_batch_size", train_cfg.get("batch_size", 64))),
        num_workers=int(train_cfg.get("val_num_workers", train_cfg.get("num_workers", 4))),
        pin_memory=bool(train_cfg.get("pin_memory", True)),
        persistent_workers=bool(train_cfg.get("persistent_workers", False)),
        drop_last=False,
    )
    return train_loader, val_loader, train_ds


def build_lr_scheduler(optimizer: torch.optim.Optimizer, cfg: Dict[str, Any]):
    train_cfg = cfg["training"]
    name = str(train_cfg.get("lr_scheduler", "none")).lower()
    if name in {"none", "constant"}:
        return None
    warmup = int(train_cfg.get("lr_warmup_steps", 0))
    total = int(train_cfg["max_train_steps"])

    def lr_lambda(step: int) -> float:
        if warmup > 0 and step < warmup:
            return float(step + 1) / float(warmup)
        if name == "cosine":
            progress = (step - warmup) / max(1, total - warmup)
            return 0.5 * (1.0 + np.cos(np.pi * min(1.0, progress)))
        raise ValueError(f"Unsupported lr_scheduler={name}")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


@torch.no_grad()
def validate(
    model: DiffusionPolicy,
    loader: DataLoader,
    noise_scheduler: DDPMScheduler,
    device: torch.device,
    max_batches: Optional[int],
) -> Dict[str, float]:
    was_training = model.training
    model.eval()
    losses = []
    for idx, batch in enumerate(loader):
        if max_batches is not None and idx >= int(max_batches):
            break
        batch = _move_batch(batch, device)
        out = model.compute_loss(batch, noise_scheduler)
        losses.append(float(out.loss.item()))
    model.train(was_training)
    return {"val/loss": float(np.mean(losses)) if losses else float("nan")}


def save_checkpoint(
    path: Path,
    *,
    cfg: Dict[str, Any],
    step: int,
    epoch: int,
    model: DiffusionPolicy,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Any,
    ema: Optional[EMAModel],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cfg": cfg,
        "step": int(step),
        "epoch": int(epoch),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None,
        "ema": ema.state_dict() if ema is not None else None,
    }
    torch.save(payload, path)


def maybe_resume(
    cfg: Dict[str, Any],
    model: DiffusionPolicy,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Any,
    ema: Optional[EMAModel],
    device: torch.device,
) -> tuple[int, int]:
    resume_path = cfg["training"].get("resume_checkpoint", None)
    if not resume_path:
        return 0, 0
    payload = torch.load(str(resume_path), map_location=device)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    if lr_scheduler is not None and payload.get("lr_scheduler") is not None:
        lr_scheduler.load_state_dict(payload["lr_scheduler"])
    if ema is not None and payload.get("ema") is not None:
        ema.load_state_dict(payload["ema"])
    return int(payload.get("step", 0)), int(payload.get("epoch", 0))


def init_wandb(cfg: Dict[str, Any]) -> bool:
    wandb_cfg = cfg.get("wandb", {})
    enabled = bool(wandb_cfg.get("enabled", True))
    if not enabled:
        return False
    tags = wandb_cfg.get("tags", [])
    if isinstance(tags, str):
        tags = [tags]
    wandb.init(
        project=str(wandb_cfg.get("project", "diffusion-policy")),
        entity=wandb_cfg.get("entity", None),
        name=wandb_cfg.get("name", None),
        group=wandb_cfg.get("group", None),
        tags=list(tags),
        config=cfg,
    )
    wandb.define_metric("step")
    wandb.define_metric("train/*", step_metric="step")
    wandb.define_metric("val/*", step_metric="step")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Train UNet-based Diffusion Policy from HDF5 demos.")
    parser.add_argument("--config", type=str, required=True, help="Environment-specific YAML config.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    seed = int(cfg.get("seed", 0))
    set_seed(seed)
    image_h, image_w = _as_image_size(cfg["dataset"].get("image_size", None), cfg)
    cfg.setdefault("env", {})
    cfg["env"]["image_height"] = int(image_h)
    cfg["env"]["image_width"] = int(image_w)
    cfg.setdefault("vision", {})
    cfg["vision"].setdefault("img_height", int(image_h))
    cfg["vision"].setdefault("img_width", int(image_w))

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    train_loader, val_loader, train_ds = build_dataloaders(cfg)

    model = DiffusionPolicy(cfg, action_dim=train_ds.action_dim, proprio_dim=train_ds.proprio_dim).to(device)
    model.normalizer.load_stats(train_ds.compute_normalizer_stats())
    noise_scheduler = DDPMScheduler.from_config(dict(cfg.get("noise_scheduler", {})))

    train_cfg = cfg["training"]
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(train_cfg.get("lr", 1e-4)),
        betas=tuple(float(v) for v in train_cfg.get("betas", [0.95, 0.999])),
        eps=float(train_cfg.get("eps", 1e-8)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-6)),
    )
    lr_scheduler = build_lr_scheduler(optimizer, cfg)
    ema = EMAModel(model, decay=float(train_cfg.get("ema_decay", 0.9999))) if bool(train_cfg.get("use_ema", True)) else None
    global_step, start_epoch = maybe_resume(cfg, model, optimizer, lr_scheduler, ema, device)

    use_wandb = init_wandb(cfg)
    max_train_steps = int(train_cfg["max_train_steps"])
    log_every = int(train_cfg.get("log_every_steps", 100))
    val_every = int(train_cfg.get("val_every_steps", 1000))
    ckpt_every = int(train_cfg.get("checkpoint_every_steps", 5000))
    checkpoint_dir = Path(str(train_cfg.get("checkpoint_dir", "checkpoints/diffusion_policy")))
    grad_clip = float(train_cfg.get("gradient_clip_norm", 1.0))
    gradient_accumulate_every = int(train_cfg.get("gradient_accumulate_every", 1))
    max_val_batches = train_cfg.get("max_val_batches", None)
    amp_enabled = bool(train_cfg.get("use_amp", False)) and device.type == "cuda"
    scaler = GradScaler(enabled=amp_enabled)
    recent_losses: deque[float] = deque(maxlen=max(1, log_every))

    model.train()
    epoch = start_epoch
    while global_step < max_train_steps:
        epoch += 1
        for batch_idx, batch in enumerate(train_loader):
            batch = _move_batch(batch, device)
            with autocast(enabled=amp_enabled):
                loss_out = model.compute_loss(batch, noise_scheduler)
                loss = loss_out.loss / gradient_accumulate_every

            scaler.scale(loss).backward()
            if (batch_idx + 1) % gradient_accumulate_every != 0:
                continue

            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if lr_scheduler is not None:
                lr_scheduler.step()
            if ema is not None:
                ema.update(model)

            global_step += 1
            recent_losses.append(float(loss_out.loss.detach().item()))

            if global_step % log_every == 0:
                log_data = {
                    "step": global_step,
                    "train/loss": float(np.mean(recent_losses)),
                    "train/lr": float(optimizer.param_groups[0]["lr"]),
                    "train/camera_index": float(batch["camera_index"][0].item()),
                }
                if use_wandb:
                    wandb.log(log_data, step=global_step)
                else:
                    print(log_data)

            if global_step % val_every == 0:
                eval_model = ema.averaged_model if ema is not None else model
                val_metrics = validate(eval_model, val_loader, noise_scheduler, device, max_val_batches)
                val_metrics["step"] = global_step
                if use_wandb:
                    wandb.log(val_metrics, step=global_step)
                else:
                    print(val_metrics)

            if ckpt_every > 0 and global_step % ckpt_every == 0:
                save_checkpoint(
                    checkpoint_dir / f"step_{global_step:08d}.pt",
                    cfg=cfg,
                    step=global_step,
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    ema=ema,
                )

            if global_step >= max_train_steps:
                break

    save_checkpoint(
        checkpoint_dir / "latest.pt",
        cfg=cfg,
        step=global_step,
        epoch=epoch,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        ema=ema,
    )
    train_loader.dataset.close()
    val_loader.dataset.close()
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()

