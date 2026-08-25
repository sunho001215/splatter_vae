from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from dataset.droid.dataset import DROIDDatasetConfig, DROIDLogicalDataset, droid_collate
from dataset.droid.safety import assert_paths_outside_source, validate_derived_root
from dataset.droid.sampling import (
    EpisodeGroupedDistributedSampler,
    MotionCropConfig,
    TemporalSamplingConfig,
)
from models.gaussian.parameterization import (
    SplatterConfig,
    SplatterDataConfig,
    SplatterModelConfig,
    gaussian_params_per_gaussian,
)
from models.splattervae import SplatterVAE, ViTSmallConfig
from models.training import TrainConfig
from models.training.distributed import (
    cuda_device_diagnostics,
    destroy_distributed,
    initialize_distributed,
    seed_distributed,
    wrap_ddp,
)
from models.training.loop import train_droid
from models.training.visualization import save_droid_validation_visualization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pretrain the DROID temporal ViT-S representation."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--per-gpu-batch", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--workspace-stats", default=None)
    parser.add_argument(
        "--allow-unvalidated-workspace",
        action="store_true",
        help="Development smoke tests only; real training should use computed DROID workspace statistics.",
    )
    return parser.parse_args()


def _only_dataclass_fields(cls, values: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {item.name for item in fields(cls)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {sorted(unknown)}")
    return dict(values)


def _validate_fixed_pipeline_contract(config: Mapping[str, Any]) -> None:
    """Fail fast instead of silently ignoring architecture-sensitive YAML keys."""
    preprocessing = config["preprocessing"]
    crop = preprocessing["crop"]
    expected = {
        "preprocessing.source_width": (int(preprocessing["source_width"]), 320),
        "preprocessing.source_height": (int(preprocessing["source_height"]), 180),
        "preprocessing.padded_size": (int(preprocessing["padded_size"]), 320),
        "preprocessing.pad_top": (int(preprocessing["pad_top"]), 70),
        "preprocessing.pad_bottom": (int(preprocessing["pad_bottom"]), 70),
        "preprocessing.output_size": (int(preprocessing["output_size"]), 224),
        "preprocessing.padding_value": (
            str(preprocessing["padding_value"]),
            "imagenet_mean",
        ),
        "preprocessing.crop.size_sampling": (
            str(crop["size_sampling"]),
            "uniform",
        ),
        "preprocessing.crop.center_mode": (
            str(crop["center_mode"]),
            "optical_flow_argmax",
        ),
        "preprocessing.crop.low_motion_fallback": (
            str(crop["low_motion_fallback"]),
            "image_center",
        ),
        "vit.normalization": (str(config["vit"]["normalization"]).lower(), "rmsnorm"),
        "vit.initialize_from": (
            str(config["vit"]["initialize_from"]).lower(),
            "scratch",
        ),
        "model.temporal_anchor": (str(config["model"]["temporal_anchor"]), "current"),
        "contrastive.distributed_negatives": (
            bool(config["contrastive"]["distributed_negatives"]),
            True,
        ),
        "decoder.conditioning": (
            str(config["decoder"]["conditioning"]),
            "multi_token_cross_attention",
        ),
        "calibration.world_frame": (
            str(config["calibration"]["world_frame"]),
            "robot_base",
        ),
        "calibration.stage": (int(config["calibration"]["stage"]), 0),
        "novel_view.backend": (str(config["novel_view"]["backend"]), "see3d"),
        "depth.teacher": (str(config["depth"]["teacher"]).lower(), "xlens"),
        "depth.inference_resolution": (
            tuple(config["depth"]["inference_resolution"]),
            (320, 180),
        ),
        "depth.cache_format": (
            str(config["depth"]["cache_format"]),
            "hdf5_shards",
        ),
        "flow.teacher": (str(config["flow"]["teacher"]).lower(), "waft"),
        "flow.cache_format": (
            str(config["flow"]["cache_format"]),
            "hdf5_shards",
        ),
        "novel_view.pose_sampling.mode": (
            str(config["novel_view"]["pose_sampling"]["mode"]),
            "interpolate",
        ),
        "novel_view.geometric_warp.use_xlens_depth": (
            bool(config["novel_view"]["geometric_warp"]["use_xlens_depth"]),
            True,
        ),
        "novel_view.geometric_warp.fuse_two_external_views": (
            bool(config["novel_view"]["geometric_warp"]["fuse_two_external_views"]),
            True,
        ),
        "novel_view.validation.required_before_enable": (
            bool(config["novel_view"]["validation"]["required_before_enable"]),
            True,
        ),
        "optimizer.name": (str(config["optimizer"]["name"]).lower(), "adamw"),
        "optimizer.schedule": (str(config["optimizer"]["schedule"]).lower(), "cosine"),
        "distributed.backend": (str(config["distributed"]["backend"]).lower(), "nccl"),
    }
    mismatches = {
        name: {"configured": configured, "required": required}
        for name, (configured, required) in expected.items()
        if configured != required
    }
    if int(crop["min_size"]) != 180 or int(crop["max_size"]) != 320:
        mismatches["preprocessing.crop.size_interval"] = {
            "configured": (crop["min_size"], crop["max_size"]),
            "required": (180, 320),
        }
    if not bool(config["calibration"]["use_exterior_cameras_only"]):
        mismatches["calibration.use_exterior_cameras_only"] = {
            "configured": False,
            "required": True,
        }
    temporal_strides = tuple(
        int(value) for value in config["dataset"]["temporal_strides"]
    )
    required_flow_gaps = tuple(
        sorted({gap for stride in temporal_strides for gap in (stride, 2 * stride)})
    )
    configured_flow_gaps = tuple(
        sorted({int(value) for value in config["flow"]["cached_gaps"]})
    )
    if configured_flow_gaps != required_flow_gaps:
        mismatches["flow.cached_gaps"] = {
            "configured": configured_flow_gaps,
            "required_from_temporal_strides": required_flow_gaps,
        }
    if mismatches:
        raise ValueError(f"Unsupported DROID pipeline configuration: {mismatches}")


def _dataset_config(config: Mapping[str, Any], split: str) -> DROIDDatasetConfig:
    dataset = config["dataset"]
    preprocessing = config["preprocessing"]
    temporal = TemporalSamplingConfig(
        strides=tuple(int(value) for value in dataset["temporal_strides"]),
        probabilities=tuple(
            float(value) for value in dataset["temporal_probabilities"]
        ),
        validation_stride=int(dataset["validation_stride"]),
    )
    crop = preprocessing["crop"]
    motion_crop = MotionCropConfig(
        min_size=int(crop["min_size"]),
        max_size=int(crop["max_size"]),
        size_sampling=str(crop["size_sampling"]),
        padded_size=int(preprocessing["padded_size"]),
        pad_top=int(preprocessing["pad_top"]),
        pad_bottom=int(preprocessing["pad_bottom"]),
        output_size=int(preprocessing["output_size"]),
        center_mode=str(crop["center_mode"]),
        flow_aggregation=str(crop["flow_aggregation"]),
        flow_smoothing_kernel=int(crop["flow_smoothing_kernel"]),
        low_motion_threshold=float(crop["low_motion_threshold"]),
        low_motion_fallback=str(crop["low_motion_fallback"]),
    )
    normalization = preprocessing["rgb_normalization"]
    if str(preprocessing["padding_value"]) != "imagenet_mean":
        raise ValueError("Only neutral ImageNet-mean RGB padding is supported.")
    mean = tuple(float(value) for value in normalization["mean"])
    return DROIDDatasetConfig(
        droid_root=str(dataset["droid_root"]),
        calibration_manifest=str(dataset["calibration_manifest"]),
        split=split,
        seed=int(dataset.get("seed", 42)),
        temporal=temporal,
        motion_crop=motion_crop,
        normalize_mean=mean,
        normalize_std=tuple(float(value) for value in normalization["std"]),
        rgb_padding_value=tuple(value * 255.0 for value in mean),
        require_depth_cache=bool(dataset.get("require_xlens_cache", True)),
        require_flow_cache=bool(dataset.get("require_waft_cache", True)),
    )


def _model_and_renderer(
    config: Mapping[str, Any],
) -> tuple[SplatterVAE, SplatterConfig]:
    vit_values = dict(config["vit"])
    vit_values.pop("normalization", None)
    vit_values.pop("initialize_from", None)
    vit = ViTSmallConfig(**_only_dataclass_fields(ViTSmallConfig, vit_values))
    decoder_values = dict(config["decoder"])
    for key in (
        "conditioning",
        "workspace_parameter_status",
        "require_validated_workspace_parameters",
    ):
        decoder_values.pop(key, None)
    model_cfg = config["model"]
    contrastive = config["contrastive"]
    renderer_cfg = config["renderer"]
    splatter = SplatterConfig(
        data=SplatterDataConfig(
            img_height=int(renderer_cfg["image_height"]),
            img_width=int(renderer_cfg["image_width"]),
            znear=float(renderer_cfg["znear"]),
            zfar=float(renderer_cfg["zfar"]),
            white_background=bool(renderer_cfg["white_background"]),
        ),
        model=SplatterModelConfig(
            max_sh_degree=int(renderer_cfg["max_sh_degree"]),
            scale_min=tuple(float(value) for value in renderer_cfg["scale_min"]),
            scale_max=tuple(float(value) for value in renderer_cfg["scale_max"]),
        ),
    )
    model = SplatterVAE(
        vit_config=vit,
        gaussian_parameters_per_gaussian=gaussian_params_per_gaussian(
            splatter.model.max_sh_degree
        ),
        decoder_config=decoder_values,
        masking_ratio=float(model_cfg["masking_ratio"]),
        motion_visible_fraction=float(model_cfg["motion_visible_fraction"]),
        projector_hidden_dimension=int(contrastive["projector_hidden_dimension"]),
        projector_output_dimension=int(contrastive["projector_output_dimension"]),
        motion_translation_max=float(model_cfg["motion_translation_max_m"]),
    )
    return model, splatter


def _train_config(config: Mapping[str, Any], resume: str | None) -> TrainConfig:
    optimizer = config["optimizer"]
    loss = config["loss"]
    depth = config["depth"]
    flow = config["flow"]
    novel = config["novel_view"]
    novel_training = novel["training"]
    novel_supervision = novel["supervision"]
    logging = config["logging"]
    betas = optimizer["betas"]
    return TrainConfig(
        num_epochs=int(optimizer["num_epochs"]),
        max_global_steps=None
        if optimizer["max_global_steps"] is None
        else int(optimizer["max_global_steps"]),
        reference_lr=float(optimizer["reference_lr"]),
        reference_batch_size=int(optimizer["reference_batch_size"]),
        decoder_lr_multiplier=float(optimizer["decoder_lr_multiplier"]),
        min_lr=float(optimizer["min_lr"]),
        warmup_steps=int(optimizer["warmup_steps"]),
        warmup_fraction=float(optimizer["warmup_fraction"]),
        weight_decay=float(optimizer["weight_decay"]),
        adam_beta1=float(betas[0]),
        adam_beta2=float(betas[1]),
        gradient_clip_norm=float(optimizer["gradient_clip_norm"]),
        gradient_accumulation_steps=int(optimizer["gradient_accumulation_steps"]),
        bf16=bool(optimizer["bf16"]),
        tf32=bool(optimizer["tf32"]),
        rgb_l1_weight=float(loss["rgb_l1"]),
        ssim_weight=float(loss["ssim"]),
        contrastive_weight=float(config["contrastive"]["weight"]),
        metric_depth_weight=float(depth["metric_depth_weight"]),
        scale_invariant_depth_weight=float(depth["scale_invariant_depth_weight"]),
        flow_weight=float(flow["weight"]),
        visibility_weight=float(loss["visibility"]),
        gaussian_regularization_weight=float(loss["gaussian_regularization"]),
        synthetic_view_weight=float(novel_training["loss_weight"]),
        contrastive_temperature=float(config["contrastive"]["temperature"]),
        depth_confidence_threshold=float(depth["confidence_threshold"]),
        scale_invariant_mean_weight=float(depth["scale_invariant_mean_weight"]),
        flow_pair_weights=tuple(float(value) for value in flow["pair_weights"]),
        flow_alpha_threshold=float(flow["alpha_threshold"]),
        flow_smooth_l1_beta=float(flow["smooth_l1_beta"]),
        novel_view_enabled=bool(novel["enabled"]),
        novel_view_probability=float(novel_training["probability"]),
        novel_view_warmup_steps=int(novel_training["warmup_steps"]),
        novel_view_minimum_confidence=float(novel_training["minimum_confidence"]),
        novel_view_require_cached=bool(novel_training["require_cached"]),
        novel_view_supervise_rgb=bool(novel_supervision["rgb"]),
        novel_view_supervise_metric_depth=bool(novel_supervision["metric_depth"]),
        novel_view_supervise_scale_invariant_depth=bool(
            novel_supervision["scale_invariant_depth"]
        ),
        seed=int(config["dataset"].get("seed", 42)),
        checkpoint_dir=str(logging["checkpoint_dir"]),
        resume_checkpoint=resume,
        validation_every_steps=int(logging["validation_every_steps"]),
        checkpoint_every_steps=int(logging["checkpoint_every_steps"]),
        visualization_every_steps=int(logging["visualization_every_steps"]),
        scalar_log_every_steps=int(logging["scalar_every_steps"]),
        validation_batches=int(logging["validation_batches"]),
    )


def _loader(
    dataset: DROIDLogicalDataset,
    config: Mapping[str, Any],
    *,
    rank: int,
    world_size: int,
    training: bool,
) -> DataLoader:
    dataset_cfg = config["dataset"]
    sampler = EpisodeGroupedDistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=training,
        seed=int(dataset_cfg.get("seed", 42)),
        drop_last=training and bool(dataset_cfg.get("drop_last", True)),
    )
    workers = int(dataset_cfg["workers"])
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(dataset_cfg["per_gpu_logical_batch"]),
        "sampler": sampler,
        "num_workers": workers,
        "pin_memory": bool(dataset_cfg["pin_memory"]),
        "drop_last": training and bool(dataset_cfg.get("drop_last", True)),
        "collate_fn": droid_collate,
    }
    if workers > 0:
        kwargs["persistent_workers"] = bool(dataset_cfg["persistent_workers"])
        kwargs["prefetch_factor"] = int(dataset_cfg["prefetch_factor"])
    return DataLoader(**kwargs)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _validate_fixed_pipeline_contract(config)
    if args.max_steps is not None:
        config["optimizer"]["max_global_steps"] = int(args.max_steps)
    if args.per_gpu_batch is not None:
        config["dataset"]["per_gpu_logical_batch"] = int(args.per_gpu_batch)
    if args.workers is not None:
        config["dataset"]["workers"] = int(args.workers)
    if args.workspace_stats is not None:
        statistics = json.loads(Path(args.workspace_stats).read_text(encoding="utf-8"))
        proposal = statistics["proposed_parameters"]
        config["decoder"].update(
            {
                "global_center": proposal["global_center"],
                "anchor_initial_spread": proposal["anchor_initial_spread"],
                "parent_displacement_scale": proposal["parent_displacement_scale"],
                "child_radius": proposal["child_radius"],
                "workspace_parameter_status": "computed_from_droid_xlens_stats_requires_training_pilot",
            }
        )
        config["renderer"].update(
            {"znear": proposal["znear"], "zfar": proposal["zfar"]}
        )
    dataset_cfg = config["dataset"]
    droid_root = Path(dataset_cfg["droid_root"]).expanduser().resolve()
    if not droid_root.is_dir():
        raise FileNotFoundError(f"DROID RLDS source is not mounted at {droid_root}.")
    derived_root = validate_derived_root(dataset_cfg["derived_root"], droid_root)
    assert_paths_outside_source(
        [
            derived_root,
            dataset_cfg["calibration_manifest"],
            dataset_cfg["xlens_cache_index"],
            dataset_cfg["waft_cache_index"],
            config["logging"]["checkpoint_dir"],
        ],
        droid_root,
    )
    decoder = config["decoder"]
    if (
        decoder.get("require_validated_workspace_parameters", True)
        and decoder.get("workspace_parameter_status")
        != "validated_from_droid_xlens_stats_and_pilot"
        and not args.allow_unvalidated_workspace
    ):
        raise RuntimeError(
            "DROID workspace parameters are still provisional. Run compute_droid_workspace_stats.py "
            "and the geometry pilot, then mark the selected values validated; use "
            "--allow-unvalidated-workspace only for synthetic development smoke tests."
        )
    if config["novel_view"]["enabled"]:
        summary = Path(config["novel_view"]["validation"]["summary_path"])
        if not summary.is_file():
            raise RuntimeError(
                "See3D augmentation cannot be enabled before real-target validation."
            )
        validation = json.loads(summary.read_text(encoding="utf-8"))
        if not bool(validation.get("approved_for_training", False)):
            raise RuntimeError(
                "See3D validation summary has not approved augmentation for training."
            )

    context = initialize_distributed()
    try:
        seed_distributed(int(dataset_cfg.get("seed", 42)), context)
        diagnostics = cuda_device_diagnostics(context)
        if context.is_main:
            print(json.dumps({"gpu": diagnostics}, indent=2), flush=True)
        train_cfg = _train_config(config, args.resume)
        torch.backends.cuda.matmul.allow_tf32 = train_cfg.tf32
        torch.backends.cudnn.allow_tf32 = train_cfg.tf32
        torch.set_float32_matmul_precision("high" if train_cfg.tf32 else "highest")
        see3d_index = Path(dataset_cfg["see3d_cache_index"])
        see3d_cache = str(see3d_index) if see3d_index.is_file() else None
        if (
            train_cfg.novel_view_enabled
            and train_cfg.novel_view_require_cached
            and see3d_cache is None
        ):
            raise FileNotFoundError(
                f"Enabled See3D training requires cache index {see3d_index}."
            )
        train_dataset = DROIDLogicalDataset(
            _dataset_config(config, "train"),
            depth_cache=dataset_cfg["xlens_cache_index"],
            flow_cache=dataset_cfg["waft_cache_index"],
            see3d_cache=see3d_cache,
        )
        validation_dataset = DROIDLogicalDataset(
            _dataset_config(config, "validation"),
            depth_cache=dataset_cfg["xlens_cache_index"],
            flow_cache=dataset_cfg["waft_cache_index"],
            see3d_cache=see3d_cache,
        )
        train_loader = _loader(
            train_dataset,
            config,
            rank=context.rank,
            world_size=context.world_size,
            training=True,
        )
        validation_loader = _loader(
            validation_dataset,
            config,
            rank=context.rank,
            world_size=context.world_size,
            training=False,
        )
        model, splatter = _model_and_renderer(config)
        model.to(context.device)
        use_ddp = bool(config["distributed"].get("use_ddp", True))
        if context.world_size > 1 and not use_ddp:
            raise RuntimeError("WORLD_SIZE > 1 requires distributed.use_ddp=true.")
        if use_ddp:
            model = wrap_ddp(
                model,
                context,
                find_unused_parameters=bool(
                    config["distributed"].get("find_unused_parameters", False)
                ),
            )
        logger = None
        if context.is_main and bool(config["logging"].get("wandb_enabled", False)):
            import wandb

            logger = wandb.init(
                project=config["logging"]["wandb_project"],
                entity=config["logging"].get("wandb_entity"),
                name=config["logging"].get("run_name"),
                config=config,
            )
        final_state = train_droid(
            model,
            splatter,
            train_loader,
            validation_loader,
            train_cfg,
            context,
            per_gpu_logical_batch=int(dataset_cfg["per_gpu_logical_batch"]),
            logger=logger,
            visualization_callback=lambda step, payload: (
                save_droid_validation_visualization(
                    Path(dataset_cfg["derived_root"]) / "logs" / "visualizations",
                    step,
                    payload,
                )
            ),
        )
        if context.is_main:
            print(f"Training finished at {final_state}", flush=True)
        if logger is not None:
            logger.finish()
    finally:
        destroy_distributed(context)


if __name__ == "__main__":
    main()
