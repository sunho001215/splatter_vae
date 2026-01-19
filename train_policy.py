#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from torch.utils.data import DataLoader

from policy_utils.config_loader import load_diffusion_config_json
from policy_utils.dataset_hdf5 import RobosuiteHDF5DiffusionDataset, WindowSpec
from policy_utils.stats10 import (
    estimate_pose10_stats,
    estimate_pose10_action_stats_from_actions,
    build_dataset_stats_for_lerobot
)
from utils.general_utils import set_random_seed

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy


def _make_window_from_cfg(cfg: Any) -> WindowSpec:
    """
    Build dataset temporal indices from the diffusion config.

    Typical diffusion policy defaults:
      n_obs_steps = 2  -> obs_indices = [-1, 0]
      horizon     = 16 -> act_indices = [-1, 0, ..., 14] (len=16)

    This keeps your dataset aligned with what the policy expects.
    """
    n_obs_steps = int(getattr(cfg, "n_obs_steps", 2))
    horizon = int(getattr(cfg, "horizon", 16))

    # [-n_obs_steps+1, ..., 0]
    obs_indices = list(range(-n_obs_steps + 1, 1))
    # [-1, 0, ..., horizon-2]  length=horizon
    act_indices = list(range(-1, horizon - 1))

    return WindowSpec(obs_indices=obs_indices, act_indices=act_indices)


def _build_optimizer_from_cfg(cfg: Any, policy: torch.nn.Module) -> torch.optim.Optimizer:
    """
    Your config.json includes optimizer hyperparams; use them if present.

    Note: LeRobot's own "example 3" constructs the optimizer in code. 
    Some LeRobot training pipelines (e.g. scripts/train.py) may consume these fields,
    but for a standalone script it's normal to instantiate the optimizer yourself.
    """
    lr = float(getattr(cfg, "optimizer_lr", 1e-4))
    betas = getattr(cfg, "optimizer_betas", (0.95, 0.999))
    eps = float(getattr(cfg, "optimizer_eps", 1e-8))
    wd = float(getattr(cfg, "optimizer_weight_decay", 0.0))

    # Ensure betas is a tuple of floats
    betas = (float(betas[0]), float(betas[1]))

    return torch.optim.Adam(policy.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=wd)


def _build_scheduler_from_cfg(
    cfg: Any,
    optimizer: torch.optim.Optimizer,
    total_steps: int,
):
    """
    Minimal scheduler support to match common LeRobot configs:
      - scheduler_name: "cosine"
      - scheduler_warmup_steps: int

    If transformers is available, we use its warmup+cosine scheduler.
    Otherwise, we fall back to a simple cosine schedule without warmup.
    """
    name = str(getattr(cfg, "scheduler_name", "none")).lower()
    warmup = int(getattr(cfg, "scheduler_warmup_steps", 0))

    if name in ("none", "null", ""):
        return None

    if name == "cosine":
        try:
            from transformers import get_cosine_schedule_with_warmup

            return get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=warmup,
                num_training_steps=total_steps,
            )
        except Exception:
            # Fallback: cosine without warmup
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps))

    # Unknown scheduler -> no scheduler
    return None

def _extract_loss(policy_out: Any) -> torch.Tensor:
    if isinstance(policy_out, tuple) and len(policy_out) >= 1:
        return policy_out[0]
    if isinstance(policy_out, dict) and "loss" in policy_out:
        return policy_out["loss"]
    raise RuntimeError(f"Unsupported policy output type: {type(policy_out)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", type=str, required=True)
    ap.add_argument("--config_json", type=str, required=True, help="Path to DiffusionPolicy config.json")
    ap.add_argument("--out_dir", type=str, default="checkpoints/diffusion_policy/run1")
    ap.add_argument("--seed", type=int, default=42)

    # Epoch-based training
    ap.add_argument("--train_epochs", type=int, default=3000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--save_every_epoch", type=int, default=100)

    # Device select (single-GPU selection)
    ap.add_argument("--device", type=str, default="cuda:0", help="e.g. cuda:0, cuda:1, or cpu")

    # Stats cache
    ap.add_argument("--max_stat_frames", type=int, default=200_000)
    ap.add_argument("--stats_cache", type=str, default="")

    args = ap.parse_args()
    set_random_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or "cpu" in args.device else "cpu")
    print(f"[info] device={device}")

    # -------------------- Load DiffusionConfig from JSON --------------------
    # Override config device/use_amp here if you want CLI to win.
    cfg, cfg_raw = load_diffusion_config_json(
        args.config_json
    )

    # Window (obs/action temporal indexing) derived from cfg
    window = _make_window_from_cfg(cfg)

    # -------------------- Dataset + DataLoader (NO CROPS HERE) --------------------
    dataset = RobosuiteHDF5DiffusionDataset(
        hdf5_path=args.hdf5,
        window=window,
        seed=args.seed,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    # -------------------- Dataset stats for normalization --------------------
    stats_cache_path = Path(args.stats_cache) if args.stats_cache else (out_dir / "dataset_stats.pt")
    if stats_cache_path.exists():
        print(f"[stats] loading cached stats from: {stats_cache_path}")
        dataset_stats = torch.load(stats_cache_path, map_location="cpu")
    else:
        print(f"[stats] computing 10D stats (max_frames={args.max_stat_frames}) ...")
        state_stats = estimate_pose10_stats(args.hdf5, max_frames=args.max_stat_frames)
        action_stats = estimate_pose10_action_stats_from_actions(args.hdf5, max_frames=args.max_stat_frames)
        dataset_stats = build_dataset_stats_for_lerobot(state_stats, action_stats)
        torch.save(dataset_stats, stats_cache_path)
        print(f"[stats] saved stats to: {stats_cache_path}")

    # -------------------- Policy --------------------
    policy = DiffusionPolicy(cfg, dataset_stats=dataset_stats).to(device)
    policy.train()

    # -------------------- Optimizer/Scheduler --------------------
    # LeRobot’s standalone example constructs optimizer in code. 
    optimizer = _build_optimizer_from_cfg(cfg, policy)

    total_steps = len(dataloader) * int(args.train_epochs)
    scheduler = _build_scheduler_from_cfg(cfg, optimizer, total_steps)

    # AMP setting usually exists in config.json ("use_amp"), but we guard it.
    use_amp = (device.type == "cuda") and bool(getattr(cfg, "use_amp", False))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # -------------------- Training loop --------------------
    global_step = 0
    for epoch in range(1, args.train_epochs + 1):
        running_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            batch = {
                k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                out = policy.forward(batch)
                loss = _extract_loss(out)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if scheduler is not None:
                scheduler.step()

            running_loss += float(loss.item())
            n_batches += 1

            if global_step % args.log_every == 0:
                print(f"[train] epoch={epoch:04d} step={global_step:08d} loss={loss.item():.4f}")

            global_step += 1

        avg_loss = running_loss / max(1, n_batches)
        print(f"[epoch] epoch={epoch:04d} avg_loss={avg_loss:.4f} batches={n_batches}")

        do_save = (args.save_every_epoch > 0 and epoch % args.save_every_epoch == 0) or (epoch == args.train_epochs)
        if do_save:
            ckpt_dir = out_dir / f"epoch_{epoch:04d}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            policy.save_pretrained(ckpt_dir)
            torch.save(dataset_stats, ckpt_dir / "dataset_stats.pt")

            print(f"[save] saved checkpoint to: {ckpt_dir}")

    print(f"[DONE] training finished. outputs at: {out_dir}")


if __name__ == "__main__":
    main()
