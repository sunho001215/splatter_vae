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
from models.gaussian.parameterization import (
    SplatterConfig,
    SplatterDataConfig,
    SplatterModelConfig,
    gaussian_params_per_gaussian,
)
from models.training.loop import train_splatter_vae
from models.splattervae.model import SplatterVAE
from models.training.config import TrainConfig
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


def _gaussian_config_values(cfg: dict) -> dict:
    values = dict(cfg.get("gaussian", {}))
    bounds = dict(values.pop("world_bounds", {}))
    if bounds:
        values["world_bounds_min"] = bounds.get("min")
        values["world_bounds_max"] = bounds.get("max")
    return values


def build_splatter_config(cfg: dict, img_height: int, img_width: int) -> SplatterConfig:
    renderer_values = dict(cfg.get("renderer", {}))
    renderer_values["img_height"] = img_height
    renderer_values["img_width"] = img_width
    gaussian_values = _gaussian_config_values(cfg)
    return SplatterConfig(
        data=SplatterDataConfig(
            **_filter_dataclass_kwargs(renderer_values, SplatterDataConfig, "renderer")
        ),
        model=SplatterModelConfig(
            **_filter_dataclass_kwargs(gaussian_values, SplatterModelConfig, "gaussian")
        ),
    )


def build_vae(cfg: dict, img_height: int, img_width: int) -> SplatterVAE:
    vit_cfg = dict(cfg.get("vit", {}))
    model_cfg = dict(cfg.get("model", {}))
    masking_cfg = dict(model_cfg.get("masking", {}))
    decoder_cfg = dict(model_cfg.get("decoder", {}))
    motion_cfg = dict(model_cfg.get("motion", {}))
    gaussian_cfg = SplatterModelConfig(
        **_filter_dataclass_kwargs(
            _gaussian_config_values(cfg), SplatterModelConfig, "gaussian"
        )
    )
    return SplatterVAE(
        vit_cfg=vit_cfg,
        img_height=img_height,
        img_width=img_width,
        gaussian_params_per_gaussian=gaussian_params_per_gaussian(
            gaussian_cfg.max_sh_degree
        ),
        inv_tube_mask_ratio=float(masking_cfg.get("inv_tube_mask_ratio", 0.50)),
        tube_mask_per_view=bool(masking_cfg.get("tube_mask_per_view", True)),
        state_dim=int(model_cfg.get("state_dim", 256)),
        flow_patch_threshold_pixels=float(
            masking_cfg.get("flow_patch_threshold_pixels", 0.5)
        ),
        motion_translation_max=float(motion_cfg.get("translation_max", 0.5)),
        temporal_modeling=bool(model_cfg.get("temporal_modeling", True)),
        decoder_num_parent_tokens=int(decoder_cfg.get("num_parent_tokens", 256)),
        decoder_gaussians_per_parent=int(
            decoder_cfg.get("gaussians_per_parent", 8)
        ),
        decoder_dim=int(decoder_cfg.get("dim", 128)),
        decoder_depth=int(decoder_cfg.get("depth", 2)),
        decoder_num_heads=int(decoder_cfg.get("num_heads", 4)),
        decoder_mlp_ratio=float(decoder_cfg.get("mlp_ratio", 4.0)),
    )


def build_metaworld_loaders(cfg: dict):
    ds_cfg = cfg.get("dataset", {})
    dataset_path = ds_cfg.get("hdf5_paths", ds_cfg.get("hdf5_path", None))
    if dataset_path is None:
        raise ValueError('Config field "dataset.hdf5_path" or "dataset.hdf5_paths" is required.')

    seed = int(ds_cfg.get("seed", 42))
    set_random_seed(seed)

    if "views" not in ds_cfg or not ds_cfg["views"]:
        raise ValueError('Config field "dataset.views" must be an explicit non-empty camera list.')
    use_segmentation_mask = bool(
        cfg.get("train", {}).get("use_segmentation_mask", True)
    )
    if use_segmentation_mask and "selected_seg_ids" not in ds_cfg:
        raise ValueError(
            'Config field "dataset.selected_seg_ids" is required when '
            '"train.use_segmentation_mask" is true.'
        )
    return build_train_valid_loaders_metaworld(
        dataset_path=dataset_path,
        views=list(ds_cfg["views"]),
        selected_seg_ids=ds_cfg.get("selected_seg_ids"),
        use_segmentation_mask=use_segmentation_mask,
        batch_size=int(ds_cfg.get("batch_size", 16)),
        num_workers=int(ds_cfg.get("num_workers", 8)),
        pin_memory=bool(ds_cfg.get("pin_memory", True)),
        train_ratio=float(ds_cfg.get("train_ratio", 0.90)),
        seed=seed,
        train_temporal_strides=ds_cfg.get("train_temporal_strides", [3, 6, 9]),
        validation_temporal_stride=int(ds_cfg.get("validation_temporal_stride", 9)),
        split_manifest_path=ds_cfg.get("split_manifest_path"),
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
    mandatory = {"images", "depths", "masks", "optical_flows", "K", "c2w", "w2c"}
    missing = mandatory.difference(sample_batch)
    if missing:
        raise RuntimeError(f"Dataset contract violation; missing keys: {sorted(missing)}")
    if sample_batch["images"].dim() != 6 or sample_batch["images"].shape[1] != 3:
        raise RuntimeError(f"Expected image batches as (B,3,A,3,H,W), got {sample_batch['images'].shape}.")
    _, timesteps, num_views, _, img_height, img_width = sample_batch["images"].shape
    expected_dtypes = {
        "images": torch.uint8,
        "depths": torch.float32,
        "masks": torch.bool,
        "optical_flows": torch.float16,
        "K": torch.float32,
        "c2w": torch.float32,
        "w2c": torch.float32,
    }
    for key, expected_dtype in expected_dtypes.items():
        if sample_batch[key].dtype != expected_dtype:
            raise RuntimeError(f"Dataset {key} dtype must be {expected_dtype}, got {sample_batch[key].dtype}.")
    print(
        f"[Info] Training image resolution: H={img_height}, W={img_width}, "
        f"temporal_window={timesteps}, train_temporal_strides={cfg['dataset']['train_temporal_strides']}, "
        f"validation_temporal_stride={cfg['dataset']['validation_temporal_stride']}, "
        f"use_segmentation_mask={cfg.get('train', {}).get('use_segmentation_mask', True)}, "
        f"sampled_views={num_views}, mandatory_modalities=ok"
    )

    train_cfg_dict = dict(cfg.get("train", {}))
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
