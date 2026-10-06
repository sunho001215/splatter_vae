from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from dataset.droid.codecs import NumericCodecConfig
from dataset.droid.dataset import (
    DROIDDatasetConfig,
    DROIDPreprocessedDataset,
    droid_collate,
)
from dataset.droid.preprocessed_manifest import (
    RETAINED_RAW_STRIDE,
    TEMPORAL_GAP_RAW,
    TEMPORAL_WINDOW,
    load_stage0_manifest,
)
from dataset.droid.safety import assert_paths_outside_source, validate_derived_root
from dataset.droid.sampling import EpisodeGroupedDistributedSampler, MotionCropConfig
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

FOUNDATION_MODULE_PREFIXES = (
    "depth_anything_3",
    "megaflow",
    "preprocessing.da3",
    "preprocessing.megaflow",
    "preprocessing.lagernvs.official",
)


def loaded_foundation_modules() -> list[str]:
    return sorted(
        name
        for name in sys.modules
        if any(
            name == prefix or name.startswith(prefix + ".")
            for prefix in FOUNDATION_MODULE_PREFIXES
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the DROID temporal ViT-S representation exclusively from the "
            "offline Stage-0 indexed dataset."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--preprocessed-root",
        default=None,
        help="Override dataset.preprocessed_root (for example, for a pilot cache).",
    )
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
    parser.add_argument(
        "--allow-unvalidated-workspace",
        action="store_true",
        help=(
            "Development smoke tests only; production training requires workspace "
            "bounds validated from cached DA3 depth."
        ),
    )
    return parser.parse_args()


def _only_dataclass_fields(cls: type, values: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {item.name for item in fields(cls)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {sorted(unknown)}")
    return dict(values)


def _validate_fixed_pipeline_contract(config: Mapping[str, Any]) -> None:
    """Reject stale/ambiguous settings instead of changing Stage-0 semantics."""

    dataset = config["dataset"]
    preprocessing = config["preprocessing"]
    crop = preprocessing["crop"]
    masking = config["masking"]
    depth = config["depth"]
    flow = config["flow"]
    novel = config["novel_view"]
    expected = {
        "dataset.source_root": (
            str(Path(dataset["source_root"]).expanduser().resolve()),
            "/home/ws/data/droid",
        ),
        "dataset.preprocessed_root": (
            str(Path(dataset["preprocessed_root"]).expanduser().resolve()),
            "/home/ws/data/droid_stage0_preprocessed",
        ),
        "dataset.eligibility": (
            str(dataset["eligibility"]),
            "canonical_stage0_calibration_valid_only",
        ),
        "dataset.retained_raw_stride": (
            int(dataset["retained_raw_stride"]),
            RETAINED_RAW_STRIDE,
        ),
        "dataset.temporal_gap_raw": (
            int(dataset["temporal_gap_raw"]),
            TEMPORAL_GAP_RAW,
        ),
        "dataset.temporal_window": (
            int(dataset["temporal_window"]),
            TEMPORAL_WINDOW,
        ),
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
            "middle_frame_forward_megaflow_argmax",
        ),
        "preprocessing.crop.flow_alignment": (
            str(crop["flow_alignment"]),
            "forward_splat_f01_to_t1_then_max_f12",
        ),
        "preprocessing.crop.flow_aggregation": (
            str(crop["flow_aggregation"]),
            "max",
        ),
        "preprocessing.crop.low_motion_fallback": (
            str(crop["low_motion_fallback"]),
            "image_center",
        ),
        "depth.source": (str(depth["source"]), "cached_da3"),
        "depth.encoding": (str(depth["encoding"]), "uint16_mm"),
        "depth.invalid_value": (int(depth["invalid_value"]), 0),
        "depth.resolution": (tuple(depth["resolution"]), (180, 320)),
        "flow.source": (str(flow["source"]), "cached_megaflow"),
        "flow.gap_raw_timesteps": (
            int(flow["gap_raw_timesteps"]),
            TEMPORAL_GAP_RAW,
        ),
        "flow.resolution": (tuple(flow["resolution"]), (180, 320)),
        "flow.encoding": (str(flow["encoding"]), "int16_fixed_1_over_64"),
        "flow.invalid_sentinel": (int(flow["invalid_sentinel"]), -32768),
        "novel_view.source": (str(novel["source"]), "cached_lagernvs"),
        "novel_view.enabled": (bool(novel["enabled"]), True),
        "novel_view.targets_per_timestep": (
            int(novel["targets_per_timestep"]),
            4,
        ),
        "novel_view.canonical_size": (int(novel["canonical_size"]), 256),
        "masking.ratio": (float(masking["ratio"]), 0.60),
        "masking.validity_aware": (bool(masking["validity_aware"]), True),
        "masking.tube_mask": (bool(masking["tube_mask"]), True),
        "masking.motion_visible_fraction": (
            float(masking["motion_visible_fraction"]),
            0.50,
        ),
        "vit.normalization": (
            str(config["vit"]["normalization"]).lower(),
            "rmsnorm",
        ),
        "vit.initialize_from": (
            str(config["vit"]["initialize_from"]).lower(),
            "scratch",
        ),
        "model.temporal_anchor": (
            str(config["model"]["temporal_anchor"]),
            "current",
        ),
        "contrastive.distributed_negatives": (
            bool(config["contrastive"]["distributed_negatives"]),
            True,
        ),
        "decoder.conditioning": (
            str(config["decoder"]["conditioning"]),
            "multi_token_cross_attention",
        ),
        "optimizer.name": (str(config["optimizer"]["name"]).lower(), "adamw"),
        "optimizer.schedule": (
            str(config["optimizer"]["schedule"]).lower(),
            "cosine",
        ),
        "distributed.backend": (
            str(config["distributed"]["backend"]).lower(),
            "nccl",
        ),
    }
    mismatches = {
        name: {"configured": configured, "required": required}
        for name, (configured, required) in expected.items()
        if configured != required
    }
    if (int(crop["min_size"]), int(crop["max_size"])) != (180, 320):
        mismatches["preprocessing.crop.size_interval"] = {
            "configured": (crop["min_size"], crop["max_size"]),
            "required": (180, 320),
        }
    forbidden_fragments = ("xlens", "memfof", "online_teacher", "temporal_prob")
    serialized = json.dumps(config, sort_keys=True).lower()
    stale = [fragment for fragment in forbidden_fragments if fragment in serialized]
    if stale:
        mismatches["obsolete_configuration"] = {
            "configured": stale,
            "required": "no online/X-Lens/MEMFOF/random-stride settings",
        }
    forbidden_runtime_keys = {
        "flow": sorted(
            set(flow)
            & {
                "backend",
                "iterations",
                "precision",
                "official_repo_path",
                "cache_dir",
            }
        ),
        "depth": sorted(
            set(depth)
            & {
                "teacher",
                "amp_dtype",
                "checkpoint_path",
                "official_repo_path",
                "confidence_threshold",
            }
        ),
        "novel_view": sorted(
            set(novel)
            & {
                "backend",
                "dtype",
                "microbatch_size",
                "probability",
                "warmup_steps",
                "cache_dir",
                "official_repo_path",
            }
        ),
    }
    forbidden_runtime_keys = {
        section: keys for section, keys in forbidden_runtime_keys.items() if keys
    }
    if forbidden_runtime_keys:
        mismatches["offline_only_configuration"] = {
            "configured": forbidden_runtime_keys,
            "required": "cached sources with no training-time inference controls",
        }
    if mismatches:
        raise ValueError(f"Unsupported DROID cached pipeline configuration: {mismatches}")


def _motion_crop_config(config: Mapping[str, Any]) -> MotionCropConfig:
    preprocessing = config["preprocessing"]
    crop = preprocessing["crop"]
    return MotionCropConfig(
        min_size=int(crop["min_size"]),
        max_size=int(crop["max_size"]),
        size_sampling=str(crop["size_sampling"]),
        padded_size=int(preprocessing["padded_size"]),
        pad_top=int(preprocessing["pad_top"]),
        pad_bottom=int(preprocessing["pad_bottom"]),
        output_size=int(preprocessing["output_size"]),
        center_mode="optical_flow_argmax",
        flow_aggregation=str(crop["flow_aggregation"]),
        flow_smoothing_kernel=int(crop["flow_smoothing_kernel"]),
        low_motion_threshold=float(crop["low_motion_threshold"]),
        low_motion_fallback=str(crop["low_motion_fallback"]),
    )


def _dataset_config(config: Mapping[str, Any], split: str) -> DROIDDatasetConfig:
    dataset = config["dataset"]
    numeric = config["storage"]["numeric"]
    return DROIDDatasetConfig(
        preprocessed_root=str(dataset["preprocessed_root"]),
        split=split,
        seed=int(dataset.get("seed", 42)),
        motion_crop=_motion_crop_config(config),
        reader_cache_size=int(dataset.get("reader_cache_size", 8)),
        verify_member_checksums=bool(dataset.get("verify_member_checksums", False)),
        numeric=NumericCodecConfig(
            cname=str(numeric["codec"]),
            compression_level=int(numeric["compression_level"]),
            shuffle=str(numeric["shuffle"]),
        ),
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
    model_cfg = config["model"]
    masking = config["masking"]
    contrastive = config["contrastive"]
    model = SplatterVAE(
        vit_config=vit,
        gaussian_parameters_per_gaussian=gaussian_params_per_gaussian(
            splatter.model.max_sh_degree
        ),
        decoder_config=decoder_values,
        masking_ratio=float(masking["ratio"]),
        motion_visible_fraction=float(masking["motion_visible_fraction"]),
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
        max_global_steps=(
            None
            if optimizer["max_global_steps"] is None
            else int(optimizer["max_global_steps"])
        ),
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
        scale_invariant_mean_weight=float(depth["scale_invariant_mean_weight"]),
        flow_pair_weights=tuple(float(value) for value in flow["pair_weights"]),
        flow_alpha_threshold=float(flow["alpha_threshold"]),
        flow_smooth_l1_beta=float(flow["smooth_l1_beta"]),
        novel_view_enabled=True,
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
    dataset: DROIDPreprocessedDataset,
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


def _apply_workspace_statistics(
    config: dict[str, Any], statistics_path: str
) -> None:
    statistics = json.loads(Path(statistics_path).read_text(encoding="utf-8"))
    if statistics.get("depth_source") != "cached_da3":
        raise ValueError("Workspace statistics must be computed from cached DA3 depth.")
    proposal = statistics["proposed_parameters"]
    config["decoder"].update(
        {
            "global_center": proposal["global_center"],
            "anchor_initial_spread": proposal["anchor_initial_spread"],
            "parent_displacement_scale": proposal["parent_displacement_scale"],
            "child_radius": proposal["child_radius"],
            "workspace_parameter_status": str(
                statistics.get(
                    "workspace_parameter_status", "computed_from_cached_da3"
                )
            ),
        }
    )
    config["renderer"].update(
        {"znear": proposal["znear"], "zfar": proposal["zfar"]}
    )


def _validate_cached_manifest(config: Mapping[str, Any]) -> dict[str, Any]:
    dataset = config["dataset"]
    root = Path(dataset["preprocessed_root"]).expanduser().resolve()
    manifest = load_stage0_manifest(root)
    checks = {
        "source_root": str(Path(dataset["source_root"]).expanduser().resolve()),
        "eligibility": str(dataset["eligibility"]),
        "retained_raw_stride": int(dataset["retained_raw_stride"]),
        "temporal_gap_raw": int(dataset["temporal_gap_raw"]),
        "temporal_window": int(dataset["temporal_window"]),
    }
    mismatch = {
        key: {"configured": value, "manifest": manifest.get(key)}
        for key, value in checks.items()
        if manifest.get(key) != value
    }
    if mismatch:
        raise ValueError(
            f"Cached Stage-0 manifest does not match training config: {mismatch}"
        )
    return manifest


def main() -> None:
    args = parse_args()
    imported_teachers = loaded_foundation_modules()
    if imported_teachers:
        raise RuntimeError(
            "Cached DROID training imported foundation-model modules: "
            f"{imported_teachers}"
        )
    config_path = Path(args.config).expanduser().resolve()
    config: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _validate_fixed_pipeline_contract(config)
    if args.preprocessed_root is not None:
        config["dataset"]["preprocessed_root"] = str(
            Path(args.preprocessed_root).expanduser().resolve()
        )
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
    if args.workspace_stats is not None:
        _apply_workspace_statistics(config, args.workspace_stats)

    dataset_cfg = config["dataset"]
    source_root = Path(dataset_cfg["source_root"]).expanduser().resolve()
    preprocessed_root = validate_derived_root(
        dataset_cfg["preprocessed_root"], source_root
    )
    assert_paths_outside_source(
        (
            preprocessed_root,
            config["logging"]["checkpoint_dir"],
            config["logging"]["visualization_dir"],
        ),
        source_root,
    )
    manifest = _validate_cached_manifest(config)
    decoder = config["decoder"]
    if (
        bool(decoder.get("require_validated_workspace_parameters", True))
        and decoder.get("workspace_parameter_status")
        != "validated_from_cached_da3_pilot"
        and not args.allow_unvalidated_workspace
    ):
        raise RuntimeError(
            "DROID workspace parameters are not yet validated from cached DA3 depth. "
            "Run the cached workspace-statistics and geometry pilot first; use "
            "--allow-unvalidated-workspace only for development smoke tests."
        )

    context = initialize_distributed()
    try:
        seed_distributed(int(dataset_cfg.get("seed", 42)), context)
        if context.is_main:
            print(
                json.dumps(
                    {
                        "gpu": cuda_device_diagnostics(context),
                        "cached_dataset": {
                            "root": str(preprocessed_root),
                            "schema_signature": manifest["schema_signature"],
                            "counts": manifest["counts"],
                        },
                        "foundation_models_loaded": loaded_foundation_modules(),
                    },
                    indent=2,
                ),
                flush=True,
            )
        train_cfg = _train_config(config, args.resume)
        torch.backends.cuda.matmul.allow_tf32 = train_cfg.tf32
        torch.backends.cudnn.allow_tf32 = train_cfg.tf32
        torch.set_float32_matmul_precision("high" if train_cfg.tf32 else "highest")
        train_dataset = DROIDPreprocessedDataset(_dataset_config(config, "train"))
        validation_dataset = DROIDPreprocessedDataset(
            _dataset_config(config, "validation")
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
