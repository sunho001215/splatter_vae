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
from models.training.online_preprocessing import (
    OnlinePreprocessingConfig,
    OnlineTeacherPipeline,
)
from models.training.visualization import save_droid_validation_visualization
from preprocessing.lagernvs.official import LagerNVSDROIDTeacher
from preprocessing.lagernvs.pose import LagerTargetPoseConfig
from preprocessing.memfof import MEMFOFDROIDTeacher
from preprocessing.xlens.official import XLensDROIDTeacher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pretrain the DROID temporal ViT-S representation."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--per-gpu-batch", type=int, default=None)
    parser.add_argument("--gradient-accumulation", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--workspace-stats", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--visualization-dir", default=None)
    parser.add_argument("--checkpoint-every-steps", type=int, default=None)
    parser.add_argument("--validation-every-steps", type=int, default=None)
    parser.add_argument("--visualization-every-steps", type=int, default=None)
    parser.add_argument("--scalar-every-steps", type=int, default=None)
    parser.add_argument("--validation-batches", type=int, default=None)
    parser.add_argument("--num-visualization-samples", type=int, default=None)
    parser.add_argument("--wandb-enabled", action="store_true")
    parser.add_argument("--run-name", default=None)
    novel_group = parser.add_mutually_exclusive_group()
    novel_group.add_argument(
        "--novel-view-enabled",
        action="store_true",
        help="Run LagerNVS online for every logical sample.",
    )
    novel_group.add_argument(
        "--novel-view-disabled",
        action="store_true",
        help="Completely bypass LagerNVS (MEMFOF and X-Lens remain online).",
    )
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
        "dataset.droid_root": (
            str(Path(config["dataset"]["droid_root"]).expanduser().resolve()),
            "/home/ws/data/droid",
        ),
        "novel_view.backend": (str(config["novel_view"]["backend"]), "lagernvs"),
        "depth.teacher": (str(config["depth"]["teacher"]).lower(), "xlens"),
        "depth.inference_resolution": (
            tuple(config["depth"]["inference_resolution"]),
            (320, 180),
        ),
        "flow.backend": (str(config["flow"]["backend"]).lower(), "memfof"),
        "flow.native_resolution": (bool(config["flow"]["native_resolution"]), True),
        "flow.iterations": (int(config["flow"]["iterations"]), 2),
        "novel_view.target_pose.mode": (
            str(config["novel_view"]["target_pose"]["mode"]),
            "interpolate_with_bounded_perturbation",
        ),
        "novel_view.canonical.image_size": (
            int(config["novel_view"]["canonical"]["image_size"]),
            256,
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
    forbidden_novel = {
        key
        for key in config["novel_view"]
        if "probability" in key or "warmup" in key
    }
    if forbidden_novel:
        mismatches["novel_view.boolean_only"] = {
            "configured_forbidden_keys": sorted(forbidden_novel),
            "required": "enabled Boolean with no stochastic skipping",
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
    if str(preprocessing["padding_value"]) != "imagenet_mean":
        raise ValueError("Only neutral ImageNet-mean RGB padding is supported.")
    return DROIDDatasetConfig(
        droid_root=str(dataset["droid_root"]),
        calibration_manifest=str(dataset["calibration_manifest"]),
        split=split,
        seed=int(dataset.get("seed", 42)),
        temporal=temporal,
        motion_crop=motion_crop,
    )


def _online_preprocessor(
    config: Mapping[str, Any], device: torch.device
) -> OnlineTeacherPipeline:
    preprocessing = config["preprocessing"]
    normalization = preprocessing["rgb_normalization"]
    motion_crop = _dataset_config(config, "train").motion_crop
    depth = config["depth"]
    flow = config["flow"]
    novel = config["novel_view"]
    memfof = MEMFOFDROIDTeacher(
        model_id=str(flow["checkpoint"]),
        revision=str(flow["checkpoint_revision"]),
        iterations=int(flow["iterations"]),
        device=device,
        cache_dir=flow.get("cache_dir"),
        amp_dtype=None,
    )
    xlens = XLensDROIDTeacher(
        str(depth["official_repo_path"]),
        str(depth["checkpoint_path"]),
        architecture_config=depth.get("architecture_config"),
        device=str(device),
        amp_dtype=str(depth.get("amp_dtype", "bf16")),
    )
    lager = None
    pose_config = None
    if bool(novel["enabled"]):
        dtype_name = str(novel.get("dtype", "bf16")).lower()
        dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float32
        lager = LagerNVSDROIDTeacher(
            novel["official_repo_path"],
            novel.get("checkpoint_path"),
            cache_dir=novel.get("cache_dir"),
            device=device,
            dtype=dtype,
            microbatch_size=int(novel["microbatch_size"]),
            canonical_focal_px=float(novel["canonical"]["focal_px"]),
        )
        pose_values = dict(novel["target_pose"])
        pose_values.pop("mode", None)
        pose_values.pop("scene_center", None)
        pose_config = LagerTargetPoseConfig(
            **_only_dataclass_fields(LagerTargetPoseConfig, pose_values)
        )
    configured_center = novel["target_pose"].get("scene_center")
    scene_center = (
        tuple(float(value) for value in configured_center)
        if configured_center is not None
        else tuple(float(value) for value in config["decoder"]["global_center"])
    )
    return OnlineTeacherPipeline(
        memfof,
        xlens,
        OnlinePreprocessingConfig(
            motion_crop=motion_crop,
            normalize_mean=tuple(float(value) for value in normalization["mean"]),
            normalize_std=tuple(float(value) for value in normalization["std"]),
            rgb_padding_value=tuple(
                float(value) * 255.0 for value in normalization["mean"]
            ),
        ),
        lagernvs=lager,
        novel_pose_config=pose_config,
        scene_center=scene_center,
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
        novel_view_rgb_weight=float(novel["loss"]["weight"]),
        contrastive_temperature=float(config["contrastive"]["temperature"]),
        depth_confidence_threshold=float(depth["confidence_threshold"]),
        scale_invariant_mean_weight=float(depth["scale_invariant_mean_weight"]),
        flow_pair_weights=tuple(float(value) for value in flow["pair_weights"]),
        flow_alpha_threshold=float(flow["alpha_threshold"]),
        flow_smooth_l1_beta=float(flow["smooth_l1_beta"]),
        novel_view_enabled=bool(novel["enabled"]),
        novel_view_supported_weight=float(novel["loss"]["supported_weight"]),
        novel_view_unsupported_weight=float(novel["loss"]["unsupported_weight"]),
        seed=int(config["dataset"].get("seed", 42)),
        checkpoint_dir=str(logging["checkpoint_dir"]),
        resume_checkpoint=resume,
        validation_every_steps=int(logging["validation_every_steps"]),
        checkpoint_every_steps=int(logging["checkpoint_every_steps"]),
        visualization_every_steps=int(logging["visualization_every_steps"]),
        scalar_log_every_steps=int(logging["scalar_every_steps"]),
        validation_batches=int(logging["validation_batches"]),
        num_visualization_samples=int(logging["num_visualization_samples"]),
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
        kwargs["multiprocessing_context"] = "spawn"
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
    if args.gradient_accumulation is not None:
        config["optimizer"]["gradient_accumulation_steps"] = int(
            args.gradient_accumulation
        )
    if args.workers is not None:
        config["dataset"]["workers"] = int(args.workers)
    logging_overrides = {
        "checkpoint_dir": args.checkpoint_dir,
        "visualization_dir": args.visualization_dir,
        "checkpoint_every_steps": args.checkpoint_every_steps,
        "validation_every_steps": args.validation_every_steps,
        "visualization_every_steps": args.visualization_every_steps,
        "scalar_every_steps": args.scalar_every_steps,
        "validation_batches": args.validation_batches,
        "num_visualization_samples": args.num_visualization_samples,
        "run_name": args.run_name,
    }
    for name, value in logging_overrides.items():
        if value is not None:
            config["logging"][name] = value
    if args.wandb_enabled:
        config["logging"]["wandb_enabled"] = True
    if args.novel_view_enabled:
        config["novel_view"]["enabled"] = True
    if args.novel_view_disabled:
        config["novel_view"]["enabled"] = False
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
            config["depth"]["checkpoint_path"],
            config["flow"]["cache_dir"],
            config["novel_view"]["cache_dir"],
            config["logging"]["checkpoint_dir"],
            config["logging"]["visualization_dir"],
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
        train_dataset = DROIDLogicalDataset(
            _dataset_config(config, "train"),
        )
        validation_dataset = DROIDLogicalDataset(
            _dataset_config(config, "validation"),
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
        online_preprocessor = _online_preprocessor(config, context.device)
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
        def visualization_callback(step: int, payload: dict[str, Any]) -> None:
            logging_cfg = config["logging"]
            paths = save_droid_validation_visualization(
                logging_cfg["visualization_dir"],
                step,
                payload,
                num_samples=int(logging_cfg["num_visualization_samples"]),
                depth_range_m=tuple(
                    float(value) for value in logging_cfg["depth_display_range_m"]
                ),
            )
            if logger is not None:
                import wandb

                logger.log(
                    {key: wandb.Image(str(path)) for key, path in paths.items()},
                    step=step,
                )

        final_state = train_droid(
            model,
            splatter,
            train_loader,
            validation_loader,
            train_cfg,
            context,
            per_gpu_logical_batch=int(dataset_cfg["per_gpu_logical_batch"]),
            online_preprocessor=online_preprocessor,
            logger=logger,
            visualization_callback=visualization_callback,
        )
        if context.is_main:
            print(f"Training finished at {final_state}", flush=True)
        if logger is not None:
            logger.finish()
    finally:
        destroy_distributed(context)


if __name__ == "__main__":
    main()
