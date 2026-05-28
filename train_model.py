from __future__ import annotations

import argparse
import glob
import os
from typing import Optional

import torch
import wandb
import yaml

from dataset.dataloader import build_train_valid_loaders_robosuite
from models.splatter import (
    SplatterConfig,
    SplatterDataConfig,
    SplatterModelConfig,
    default_splatter_channels,
)
from models.splatter_pretraining import train_splatter_vae
from models.vae import CodebookConfig, SplatterVAE
from models.splatter_train_config import TrainConfig
from utils.general_utils import set_random_seed


torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Hierarchical SplatterVAE")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def build_splatter_config(cfg: dict, img_height: int, img_width: int) -> SplatterConfig:
    spl_cfg = cfg.get("splatter", {})
    spl_data_cfg_dict = dict(spl_cfg.get("data", {}))
    spl_model_cfg_dict = dict(spl_cfg.get("model", {}))

    # Match the renderer config to the actual training batch resolution.
    spl_data_cfg_dict["img_height"] = img_height
    spl_data_cfg_dict["img_width"] = img_width

    return SplatterConfig(
        data=SplatterDataConfig(**spl_data_cfg_dict),
        model=SplatterModelConfig(**spl_model_cfg_dict),
    )


def build_vae(cfg: dict, img_height: int, img_width: int) -> SplatterVAE:
    cb_cfg = cfg.get("codebook", {})
    inv_cb_cfg = CodebookConfig(**cb_cfg.get("invariant", {}))
    dep_cb_cfg = CodebookConfig(**cb_cfg.get("dependent", {}))

    vit_cfg = dict(cfg.get("vit", {}))
    model_cfg = dict(cfg.get("model", {}))
    spl_model_cfg = cfg.get("splatter", {}).get("model", {})

    max_sh_degree = int(spl_model_cfg.get("max_sh_degree", 1))
    num_gaussians_per_pixel = int(spl_model_cfg.get("num_gaussians_per_pixel", 5))
    splatter_channels = int(
        cfg.get("splatter", {}).get(
            "splatter_channels",
            default_splatter_channels(
                max_sh_degree=max_sh_degree,
                num_gaussians_per_pixel=num_gaussians_per_pixel,
            ),
        )
    )

    return SplatterVAE(
        vit_cfg=vit_cfg,
        invariant_cb_config=inv_cb_cfg,
        dependent_cb_config=dep_cb_cfg,
        img_height=img_height,
        img_width=img_width,
        splatter_channels=splatter_channels,
        fusion_style=str(model_cfg.get("fusion_style", "cat")),
        use_dependent_vq=bool(model_cfg.get("use_dependent_vq", True)),
        is_dependent_ae=bool(model_cfg.get("is_dependent_ae", True)),
        use_invariant_vq=bool(model_cfg.get("use_invariant_vq", True)),
        is_invariant_ae=bool(model_cfg.get("is_invariant_ae", True)),
        dep_input_mask_ratio=float(model_cfg.get("dep_input_mask_ratio", 0.0)),
        dep_mask_eval=bool(model_cfg.get("dep_mask_eval", False)),
        dpt_features=int(vit_cfg.get("dpt_features", 256)),
    )


def build_metaworld_loaders(cfg: dict):
    ds_cfg = cfg.get("dataset", {})
    dataset_path = ds_cfg.get("hdf5_path", None)
    if dataset_path is None:
        raise ValueError('Config field "dataset.hdf5_path" is required.')

    seed = int(ds_cfg.get("seed", 42))
    set_random_seed(seed)

    return build_train_valid_loaders_robosuite(
        dataset_path=dataset_path,
        batch_size=int(ds_cfg.get("batch_size", 32)),
        num_workers=int(ds_cfg.get("num_workers", 8)),
        pin_memory=bool(ds_cfg.get("pin_memory", True)),
        train_ratio=float(ds_cfg.get("train_ratio", 0.90)),
        seed=seed,
        num_episodes=ds_cfg.get("num_episodes", None),
        max_frames_per_demo=ds_cfg.get("max_frames_per_demo", None),
        views=ds_cfg.get("views", ds_cfg.get("camera_names", None)),
        camera_num=ds_cfg.get("camera_num", None),
        min_time_gap=int(ds_cfg.get("min_time_gap", 25)),
    )


def init_wandb(cfg: dict) -> None:
    wandb_cfg = cfg.get("wandb", {})
    if not bool(wandb_cfg.get("enabled", True)):
        return

    wandb_tags = wandb_cfg.get("tags", [])
    if isinstance(wandb_tags, str):
        wandb_tags = [wandb_tags]
    wandb_tags = [str(tag) for tag in wandb_tags if str(tag).strip()]

    run = wandb.init(
        project=wandb_cfg.get("project", "splattervae"),
        entity=wandb_cfg.get("entity", None),
        name=wandb_cfg.get("run_name", None),
        config=cfg,
        tags=wandb_tags,
    )

    # Use one explicit training step axis for every logged value. This avoids
    # relying on W&B's internal step counter, which increments on each
    # wandb.log() call and can diverge when train/validation logs happen at the
    # same optimizer step.
    run.define_metric("global_step")
    run.define_metric("train/*", step_metric="global_step")
    run.define_metric("val/*", step_metric="global_step")
    print(f"[wandb] Logging to project: {wandb_cfg.get('project', 'splattervae')}")


def find_resume_checkpoint(cfg_train: TrainConfig) -> Optional[str]:
    if not cfg_train.resume_from_last or not os.path.isdir(cfg_train.ckpt_dir):
        return None

    ckpt_candidates = sorted(glob.glob(os.path.join(cfg_train.ckpt_dir, "step_*.pth")))
    if not ckpt_candidates:
        return None

    resume_ckpt = ckpt_candidates[-1]
    print(f"[Resume] Found latest checkpoint: {resume_ckpt}")
    return resume_ckpt


def main() -> None:
    cli_args = parse_args()
    with open(cli_args.config, "r") as f:
        cfg = yaml.safe_load(f)

    train_loader, valid_loader = build_metaworld_loaders(cfg)

    sample_batch = next(iter(train_loader))
    _, camera_num, _, img_height, img_width = sample_batch["images"].shape
    print(f"[Info] Training image resolution: H={img_height}, W={img_width}, sampled_views={camera_num}")

    train_cfg_dict = cfg.get("train", {})
    train_cfg_dict.pop("use_amp", None)
    cfg_train = TrainConfig(**train_cfg_dict)

    splatter_cfg = build_splatter_config(cfg, img_height=img_height, img_width=img_width)
    vae = build_vae(cfg, img_height=img_height, img_width=img_width)

    init_wandb(cfg)

    train_splatter_vae(
        vae=vae,
        splatter_cfg=splatter_cfg,
        train_dataloader=train_loader,
        cfg_train=cfg_train,
        valid_dataloader=valid_loader,
        resume_ckpt=find_resume_checkpoint(cfg_train),
    )

    print("Training finished.")


if __name__ == "__main__":
    main()
