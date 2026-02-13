from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import wandb
import yaml

# Custom imports
from policies.models.policy import PolicyConfig, AlohaUnleashedFlowPolicy
from policies.models.vision import build_vision_encoder
from policies.utils.dataloader import RobosuiteHDF5DiffusionDataset, WindowSpec

# flow_matching imports
from flow_matching.path import CondOTProbPath

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_dataloader(cfg: Dict[str, Any]) -> DataLoader:
    """
    Build dataloader that matches the horizon-based window spec:
      WindowSpec(horizon=cfg.policy.pred_horizon)
    """
    data_cfg = cfg["data"]
    horizon = int(cfg["policy"]["pred_horizon"])

    ds = RobosuiteHDF5DiffusionDataset(
        hdf5_path=data_cfg["hdf5_path"],
        window=WindowSpec(horizon=horizon),
        camera_names=data_cfg.get("camera_names", None),
        seed=int(cfg.get("seed", 0)),
    )

    dl = DataLoader(
        ds,
        batch_size=int(data_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(data_cfg.get("num_workers", 4)),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        drop_last=True,
    )
    return dl


def build_policy(cfg: Dict[str, Any], device: torch.device) -> AlohaUnleashedFlowPolicy:
    """
    Build vision encoder + transformer policy.
    Vision encoder is selected inside build_vision_encoder(cfg).
    """
    vision = build_vision_encoder(cfg).to(device)

    pcfg = PolicyConfig(**cfg["policy"])
    policy = AlohaUnleashedFlowPolicy(cfg=pcfg, vision=vision).to(device)
    return policy


def apply_lr_warmup(optimizer: torch.optim.Optimizer, base_lr: float, step: int, warmup_steps: int) -> float:
    """
    Linear warmup to base_lr for warmup_steps.
    Returns the LR used.
    """
    if warmup_steps <= 0:
        lr = base_lr
    elif step < warmup_steps:
        lr = base_lr * float(step + 1) / float(warmup_steps)
    else:
        lr = base_lr

    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    # Device
    device = torch.device(cfg.get("device", "cpu"))

    # wandb
    wb_cfg = cfg.get("wandb", {})
    use_wandb = bool(wb_cfg.get("enabled", True))
    if use_wandb:
        wandb.init(
            project=str(wb_cfg.get("project", "aloha-unleashed-rectified-flow")),
            entity=wb_cfg.get("entity", None),
            name=wb_cfg.get("run_name", None),
            config=cfg,
        )

    # Data / model
    dl = make_dataloader(cfg)
    policy = build_policy(cfg, device)

    # Rectified Flow probability path (straight-line / CondOT)
    prob_path = CondOTProbPath()

    # Optimizer
    optim_cfg = cfg["optim"]
    base_lr = float(optim_cfg["lr"])
    weight_decay = float(optim_cfg.get("weight_decay", 0.0))
    warmup_steps = int(optim_cfg.get("warmup_steps", 0))
    grad_clip = float(optim_cfg.get("grad_clip_norm", 1.0))
    max_steps = int(optim_cfg.get("max_steps", 200_000))

    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=base_lr,
        weight_decay=weight_decay,
    )

    # Logging / checkpointing
    log_cfg = cfg.get("logging", {})
    log_every = int(log_cfg.get("log_every", 50))
    ckpt_every = int(log_cfg.get("ckpt_every", 5000))
    out_dir = str(log_cfg.get("out_dir", "./runs/rectified_flow"))
    os.makedirs(out_dir, exist_ok=True)

    # Training loop
    policy.train()
    step = 0
    start_time = time.time()

    while step < max_steps:
        for batch in dl:
            if step >= max_steps:
                break

            # -----------------------------
            # Load batch (matches dataloader)
            # -----------------------------
            obs_img = batch["observation.image"].to(device)     # (B,V,3,H,W)
            obs_state = batch["observation.state"].to(device)   # (B,P)
            x1 = batch["action"].to(device)                     # (B,Horizon,A)

            # Encode observation once
            obs_memory = policy.encode_obs(obs_img, obs_state)

            # -----------------------------
            # Rectified Flow training sample
            # -----------------------------
            # Source distribution sample (Gaussian noise)
            x0 = torch.randn_like(x1)

            # Sample times t ~ Uniform(0,1) per batch element
            B = x1.shape[0]
            t = torch.rand((B,), device=device)

            # CondOTProbPath gives x_t and target velocity dx_t/dt
            # PathSample docs: x_t and dx_t are provided. (Rectified Flow target) 
            ps = prob_path.sample(x_0=x0, x_1=x1, t=t)
            x_t = ps.x_t.to(device)
            v_target = ps.dx_t.to(device)

            # Predict velocity field from policy
            v_pred = policy.predict_velocity(x_t=x_t, t=t, obs_memory=obs_memory)

            # MSE loss on velocities
            loss = F.mse_loss(v_pred, v_target)

            # -----------------------------
            # Optimize
            # -----------------------------
            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)

            lr = apply_lr_warmup(optimizer, base_lr, step, warmup_steps)
            optimizer.step()

            # -----------------------------
            # Logging
            # -----------------------------
            if (step % log_every) == 0:
                elapsed = max(time.time() - start_time, 1e-9)
                steps_per_sec = float(step + 1) / elapsed

                logs = {
                    "train/loss_rectified_mse": float(loss.item()),
                    "train/lr": float(lr),
                    "train/steps_per_sec": steps_per_sec,
                    "train/step": step,
                }

                if use_wandb:
                    wandb.log(logs, step=step)
                else:
                    print(logs)

            # -----------------------------
            # Checkpoint
            # -----------------------------
            if (step % ckpt_every) == 0 and step > 0:
                ckpt_path = os.path.join(out_dir, f"policy_step_{step:08d}.pt")
                torch.save(
                    {
                        "policy_state_dict": policy.state_dict(),
                        "step": step,
                        "config": cfg,
                    },
                    ckpt_path,
                )
                if use_wandb:
                    wandb.log({"train/ckpt_saved": 1}, step=step)

            step += 1

    # Final save
    final_path = os.path.join(out_dir, "policy_final.pt")
    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "step": step,
            "config": cfg,
        },
        final_path,
    )

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
