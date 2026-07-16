from __future__ import annotations

import argparse
import glob
import os
from dataclasses import fields
from typing import Optional

import torch
import yaml
import wandb

from dataset.dataloader import build_train_valid_loaders_metaworld
from models.splatter import (
    SplatterConfig,
    SplatterDataConfig,
    SplatterModelConfig,
    default_splatter_channels,
)
from models.pretraining import train_splatter_vae
from models.vae import SplatterVAE
from models.train_config import TrainConfig
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


def _filter_dataclass_kwargs(values: dict, cls: type, section: str) -> dict:
    allowed = {field.name for field in fields(cls)}
    ignored = sorted(set(values) - allowed)
    if ignored:
        print(f"[Config] Ignoring deprecated/unknown {section} keys: {ignored}")
    return {key: value for key, value in values.items() if key in allowed}


def build_splatter_config(cfg: dict, img_height: int, img_width: int) -> SplatterConfig:
    spl_cfg = cfg.get("splatter", {})
    spl_data_cfg_dict = dict(spl_cfg.get("data", {}))
    spl_model_cfg_dict = dict(spl_cfg.get("model", {}))

    # Match the renderer config to the actual training batch resolution.
    spl_data_cfg_dict["img_height"] = img_height
    spl_data_cfg_dict["img_width"] = img_width

    return SplatterConfig(
        data=SplatterDataConfig(**_filter_dataclass_kwargs(spl_data_cfg_dict, SplatterDataConfig, "splatter.data")),
        model=SplatterModelConfig(**_filter_dataclass_kwargs(spl_model_cfg_dict, SplatterModelConfig, "splatter.model")),
    )


def build_vae(cfg: dict, img_height: int, img_width: int) -> SplatterVAE:
    vit_cfg = dict(cfg.get("vit", {}))
    model_cfg = dict(cfg.get("model", {}))
    spl_model_cfg = cfg.get("splatter", {}).get("model", {})

    gaussians_per_pixel = int(spl_model_cfg.get("gaussians_per_pixel", 1))
    max_sh_degree = int(spl_model_cfg.get("max_sh_degree", 1))
    splatter_channels = int(
        cfg.get("splatter", {}).get(
            "splatter_channels",
            default_splatter_channels(
                gaussians_per_pixel=gaussians_per_pixel,
                max_sh_degree=max_sh_degree,
            ),
        )
    )

    return SplatterVAE(
        vit_cfg=vit_cfg,
        img_height=img_height,
        img_width=img_width,
        splatter_channels=splatter_channels,
        dep_mask_eval=bool(model_cfg.get("dep_mask_eval", False)),
        dpt_features=int(vit_cfg.get("dpt_features", 256)),
        temporal_window=int(model_cfg.get("temporal_window", cfg.get("dataset", {}).get("temporal_window", 3))),
        inv_tube_mask_ratio=float(model_cfg.get("inv_tube_mask_ratio", 0.50)),
        dep_mask_ratio=float(model_cfg.get("dep_mask_ratio", 0.75)),
        tube_mask_per_view=bool(model_cfg.get("tube_mask_per_view", True)),
        state_dim=int(model_cfg.get("state_dim", 256)),
        view_dim=model_cfg.get("view_dim", None),
        use_single_state_vector=bool(model_cfg.get("use_single_state_vector", True)),
        dependent_uses_first_timestep_only=bool(model_cfg.get("dependent_uses_first_timestep_only", True)),
        use_temporal_delta_decoder=bool(model_cfg.get("use_temporal_delta_decoder", True)),
        decoder_condition_mode=str(model_cfg.get("decoder_condition_mode", "concat")),
        gaussians_per_pixel=gaussians_per_pixel,
        delta_xyz_scale=float(model_cfg.get("delta_xyz_scale", 0.05)),
    )


def build_metaworld_loaders(cfg: dict):
    ds_cfg = cfg.get("dataset", {})
    dataset_path = ds_cfg.get("hdf5_paths", ds_cfg.get("hdf5_path", None))
    if dataset_path is None:
        raise ValueError('Config field "dataset.hdf5_path" or "dataset.hdf5_paths" is required.')

    seed = int(ds_cfg.get("seed", 42))
    set_random_seed(seed)

    return build_train_valid_loaders_metaworld(
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
        temporal_window=int(ds_cfg.get("temporal_window", cfg.get("model", {}).get("temporal_window", 3))),
        temporal_stride=int(ds_cfg.get("temporal_stride", ds_cfg.get("timestep_interval", 1))),
        temporal_stride_list=ds_cfg.get("temporal_stride_list", None),
        temporal_min_state_change=float(ds_cfg.get("temporal_min_state_change", 0.0)),
        temporal_gripper_change_weight=float(ds_cfg.get("temporal_gripper_change_weight", 0.05)),
        use_depth=bool(ds_cfg.get("use_depth", False)),
        use_segmentation_mask=bool(ds_cfg.get("use_segmentation_mask", False)),
        selected_seg_ids=ds_cfg.get("selected_seg_ids", ds_cfg.get("mask_object_ids", None)),
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
    explicit_ckpt = getattr(cfg_train, "resume_from_checkpoint", None)
    if explicit_ckpt:
        explicit_ckpt = os.path.expanduser(str(explicit_ckpt))
        if not os.path.isfile(explicit_ckpt):
            raise FileNotFoundError(f"Configured resume_from_checkpoint does not exist: {explicit_ckpt}")
        print(f"[Resume] Using configured checkpoint: {explicit_ckpt}")
        return explicit_ckpt

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
    if sample_batch["images"].dim() == 6:
        _, temporal_window, camera_num, _, img_height, img_width = sample_batch["images"].shape
    else:
        temporal_window = 1
        _, camera_num, _, img_height, img_width = sample_batch["images"].shape
    has_depth = "depths" in sample_batch
    has_masks = "masks" in sample_batch
    print(
        f"[Info] Training image resolution: H={img_height}, W={img_width}, "
        f"temporal_window={temporal_window}, temporal_stride_list={cfg.get('dataset', {}).get('temporal_stride_list', [cfg.get('dataset', {}).get('temporal_stride', cfg.get('dataset', {}).get('timestep_interval', 1))])}, "
        f"temporal_min_state_change={cfg.get('dataset', {}).get('temporal_min_state_change', 0.0)}, "
        f"sampled_views={camera_num}, depth={has_depth}, masks={has_masks}"
    )

    train_cfg_dict = dict(cfg.get("train", {}))
    train_cfg_dict.pop("use_amp", None)
    cfg_train = TrainConfig(**_filter_dataclass_kwargs(train_cfg_dict, TrainConfig, "train"))

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
