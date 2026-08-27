from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from pathlib import Path
from typing import Any

import torch
import yaml

from dataset.droid.dataset import DROIDLogicalDataset, droid_collate
from dataset.droid.safety import validate_derived_root
from models.gaussian.parameterization import WorldSpaceGaussianParameterization
from models.training.distributed import move_to_device
from models.training.losses import cross_view_info_nce
from models.training.reconstruction import compute_droid_reconstruction
from models.training.validation import _detach_to_cpu
from models.training.visualization import save_droid_validation_visualization
from preprocessing.common import sha256_file, verify_cuda_device
from scripts.train_droid import (
    _dataset_config,
    _model_and_renderer,
    _online_preprocessor,
    _train_config,
    _validate_fixed_pipeline_contract,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate DROID checkpoints on one deterministic online-teacher set."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", action="append", default=[])
    parser.add_argument("--checkpoint-dir", action="append", default=[])
    parser.add_argument("--workspace-stats", default=None)
    parser.add_argument(
        "--output-root", default="outputs/checkpoint_analysis"
    )
    parser.add_argument("--validation-samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _checkpoint_paths(args: argparse.Namespace) -> list[Path]:
    values = [Path(value).expanduser().resolve() for value in args.checkpoint]
    for directory in args.checkpoint_dir:
        values.extend(Path(directory).expanduser().resolve().glob("step-*.pt"))
    unique = {path: None for path in values if path.is_file()}
    if not unique:
        raise FileNotFoundError("No checkpoint files were selected.")
    return sorted(unique)


def _masked_values(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask.expand_as(value).bool()
    return value[expanded]


def _percentile(value: torch.Tensor, q: float) -> float:
    return float(torch.quantile(value.float(), q)) if value.numel() else float("nan")


def _metrics(
    prediction: dict[str, torch.Tensor],
    reconstruction: dict[str, Any],
    batch: dict[str, Any],
    contrastive: torch.Tensor,
    contrast_metrics: dict[str, torch.Tensor],
) -> dict[str, float]:
    rendered_rgb = reconstruction["rendered_rgb"].float()
    target_rgb = batch["target_rgb"].float()
    rgb_mask = batch["target_image_validity"].bool()
    rgb_difference = _masked_values(rendered_rgb - target_rgb, rgb_mask)
    rgb_mse = rgb_difference.square().mean()
    rgb_psnr = -10.0 * torch.log10(rgb_mse.clamp_min(1.0e-12))
    rgb_ssim = 1.0 - 2.0 * reconstruction["dssim_loss"].float()

    rendered_depth = reconstruction["rendered_expected_depth"].float()
    target_depth = batch["target_depth"].float()
    depth_mask = (
        batch["target_depth_validity"].bool()
        & batch["target_image_validity"].bool()
        & (reconstruction["rendered_alpha"].float() > 0.01)
        & torch.isfinite(rendered_depth)
        & (rendered_depth > 0.0)
    )
    depth_error = _masked_values(rendered_depth - target_depth, depth_mask)
    depth_target = _masked_values(target_depth, depth_mask)

    flow_difference = reconstruction["rendered_flow"].float() - batch[
        "target_flow"
    ].float()
    flow_epe = torch.linalg.vector_norm(flow_difference, dim=3, keepdim=True)
    flow_mask = (
        batch["target_flow_validity"].bool()
        & reconstruction["rendered_flow_validity"].bool()
    )
    valid_flow_epe = flow_epe[flow_mask]

    cls = prediction["cls_tokens_by_view"].float()
    patches = prediction["current_patch_tokens_by_view"].float()
    cloud = reconstruction["gaussian_pc_anchor"]
    valid_gaussians = cloud["valid_mask"].bool()
    xyz = cloud["xyz"].float()[valid_gaussians]
    scaling = cloud["scaling"].float()[valid_gaussians]
    opacity = cloud["opacity"].float().squeeze(-1)[valid_gaussians]
    child = prediction["child_offsets"].float().norm(dim=-1)

    return {
        "rgb_l1": float(reconstruction["rgb_l1_loss"]),
        "rgb_psnr": float(rgb_psnr),
        "rgb_ssim": float(rgb_ssim),
        "rgb_lpips": float("nan"),
        "depth_l1": float(depth_error.abs().mean()),
        "depth_absrel": float(
            (depth_error.abs() / depth_target.clamp_min(1.0e-6)).mean()
        ),
        "depth_rmse": float(depth_error.square().mean().sqrt()),
        "depth_si": float(reconstruction["scale_invariant_depth_loss"]),
        "flow_epe": float(valid_flow_epe.mean()),
        "flow_loss": float(reconstruction["flow_loss"]),
        "contrastive_loss": float(contrastive),
        "positive_cosine": float(contrast_metrics["positive_cosine_similarity"]),
        "negative_cosine": float(contrast_metrics["negative_cosine_similarity"]),
        "cls_norm": float(cls.norm(dim=-1).mean()),
        "patch_token_norm": float(patches.norm(dim=-1).mean()),
        "patch_token_variance": float(
            patches.var(dim=(0, 1, 2), unbiased=False).mean()
        ),
        "gaussian_visibility": 1.0
        - float(reconstruction["out_of_frustum_fraction"]),
        "gaussian_out_of_frustum": float(
            reconstruction["out_of_frustum_fraction"]
        ),
        "gaussian_visible_pixel_fraction": float(
            reconstruction["rendered_visible_pixel_fraction"]
        ),
        "gaussian_xyz_x_mean": float(xyz[:, 0].mean()),
        "gaussian_xyz_y_mean": float(xyz[:, 1].mean()),
        "gaussian_xyz_z_mean": float(xyz[:, 2].mean()),
        "gaussian_xyz_x_std": float(xyz[:, 0].std(unbiased=False)),
        "gaussian_xyz_y_std": float(xyz[:, 1].std(unbiased=False)),
        "gaussian_xyz_z_std": float(xyz[:, 2].std(unbiased=False)),
        "gaussian_scale_p05": _percentile(scaling, 0.05),
        "gaussian_scale_median": _percentile(scaling, 0.50),
        "gaussian_scale_p95": _percentile(scaling, 0.95),
        "gaussian_opacity_p05": _percentile(opacity, 0.05),
        "gaussian_opacity_mean": float(opacity.mean()),
        "gaussian_opacity_p95": _percentile(opacity, 0.95),
        "child_displacement_mean": float(child.mean()),
        "child_displacement_max": float(child.amax()),
        "novel_rgb_l1": float("nan"),
        "novel_psnr": float("nan"),
        "novel_ssim": float("nan"),
        "novel_support_coverage": float("nan"),
        "target_pose_coverage": float("nan"),
        "target_alpha": float("nan"),
        "translation_jitter_m": float("nan"),
        "rotation_jitter_deg": float("nan"),
    }


def _write_svg_plot(path: Path, rows: list[dict[str, Any]], metrics: list[str]) -> None:
    width, height = 900, 360
    margin = 55
    finite_values = [
        float(row[name])
        for row in rows
        for name in metrics
        if math.isfinite(float(row[name]))
    ]
    if not finite_values:
        return
    low, high = min(finite_values), max(finite_values)
    if abs(high - low) < 1.0e-12:
        high = low + 1.0
    colors = ("#e63946", "#457b9d", "#2a9d8f", "#f4a261")
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="black"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="black"/>',
    ]
    for metric_index, name in enumerate(metrics):
        points = []
        for index, row in enumerate(rows):
            value = float(row[name])
            if not math.isfinite(value):
                continue
            x = margin + (width - 2 * margin) * index / max(1, len(rows) - 1)
            y = height - margin - (height - 2 * margin) * (value - low) / (high - low)
            points.append(f"{x:.2f},{y:.2f}")
        if points:
            color = colors[metric_index % len(colors)]
            elements.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{" ".join(points)}"/>'
            )
            elements.append(
                f'<text x="{margin + 160 * metric_index}" y="25" fill="{color}" font-size="14">{name}</text>'
            )
    for index, row in enumerate(rows):
        x = margin + (width - 2 * margin) * index / max(1, len(rows) - 1)
        elements.append(
            f'<text x="{x:.2f}" y="{height-12}" text-anchor="middle" font-size="11">{row["step"]}</text>'
        )
    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def _best(rows: list[dict[str, Any]], metric: str, *, maximum: bool = False):
    finite = [row for row in rows if math.isfinite(float(row[metric]))]
    if not finite:
        return None
    selected = (max if maximum else min)(finite, key=lambda row: float(row[metric]))
    return selected["checkpoint"]


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    _validate_fixed_pipeline_contract(config)
    config["novel_view"]["enabled"] = False
    if args.workspace_stats:
        statistics = json.loads(Path(args.workspace_stats).read_text(encoding="utf-8"))
        proposal = statistics["proposed_parameters"]
        config["decoder"].update(
            {
                "global_center": proposal["global_center"],
                "anchor_initial_spread": proposal["anchor_initial_spread"],
                "parent_displacement_scale": proposal["parent_displacement_scale"],
                "child_radius": proposal["child_radius"],
            }
        )
        config["renderer"].update(
            {"znear": proposal["znear"], "zfar": proposal["zfar"]}
        )
    droid_root = Path(config["dataset"]["droid_root"]).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve(strict=False)
    validate_derived_root(output_root, droid_root).mkdir(parents=True, exist_ok=True)
    plots_root = output_root / "plots"
    qualitative_root = output_root / "qualitative"
    plots_root.mkdir(parents=True, exist_ok=True)
    qualitative_root.mkdir(parents=True, exist_ok=True)
    checkpoints = _checkpoint_paths(args)

    device = torch.device("cuda:0")
    gpu = verify_cuda_device()
    pipeline = _online_preprocessor(config, device)
    dataset = DROIDLogicalDataset(_dataset_config(config, "validation"))
    sample_count = min(int(args.validation_samples), len(dataset))
    cpu_batch = droid_collate([dataset[index] for index in range(sample_count)])
    raw_batch = move_to_device(cpu_batch, device)
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(int(args.seed))
        batch = pipeline(raw_batch, novel_enabled=False, seed=int(args.seed))

    train_config = _train_config(config, None)
    background = torch.zeros(3, device=device)
    rows: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    for checkpoint_path in checkpoints:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        inventory_row = {
            "path": str(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
            "sha256": sha256_file(checkpoint_path),
            "architecture": checkpoint.get("architecture"),
            "training_state": checkpoint.get("training_state"),
            "decoder_configuration": checkpoint.get("decoder_configuration"),
        }
        decoder_configuration = checkpoint.get("decoder_configuration")
        if not isinstance(decoder_configuration, dict):
            inventory_row["evaluation_status"] = "excluded_missing_decoder_configuration"
            inventory.append(inventory_row)
            continue
        checkpoint_config = copy.deepcopy(config)
        checkpoint_config["decoder"].update(decoder_configuration)
        model, splatter = _model_and_renderer(checkpoint_config)
        if decoder_configuration != model.decoder_configuration():
            inventory_row["evaluation_status"] = "excluded_invalid_decoder_configuration"
            inventory.append(inventory_row)
            continue
        model.load_state_dict(checkpoint["model"], strict=True)
        model.to(device).eval()
        parameterization = WorldSpaceGaussianParameterization(splatter).to(device)
        with torch.no_grad(), torch.random.fork_rng(devices=[0]):
            torch.manual_seed(int(args.seed))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model(
                    batch["representation_histories"],
                    batch["representation_flows"],
                    batch["representation_validity"],
                )
                contrastive, contrast_metrics = cross_view_info_nce(
                    prediction["projected_cls_by_view"],
                    train_config.contrastive_temperature,
                )
            reconstruction = compute_droid_reconstruction(
                parameterization,
                splatter,
                prediction,
                batch,
                train_config,
                motion_translation_max=model.motion_translation_max,
                background_color=background,
                return_renders=True,
                novel_view_enabled=False,
            )
        step = int(checkpoint.get("training_state", {}).get("global_step", -1))
        row: dict[str, Any] = {
            "checkpoint": str(checkpoint_path),
            "checkpoint_name": checkpoint_path.stem,
            "step": step,
            **_metrics(
                prediction, reconstruction, batch, contrastive, contrast_metrics
            ),
        }
        rows.append(row)
        inventory_row["evaluation_status"] = "evaluated_real_droid_common_validation_set"
        inventory.append(inventory_row)
        save_droid_validation_visualization(
            qualitative_root / checkpoint_path.stem,
            step,
            {
                "batch": _detach_to_cpu(batch),
                "prediction": _detach_to_cpu(prediction),
                "reconstruction": _detach_to_cpu(reconstruction),
            },
            num_samples=1,
            depth_range_m=tuple(
                float(value) for value in config["logging"]["depth_display_range_m"]
            ),
        )
        del model, parameterization, checkpoint, prediction, reconstruction
        torch.cuda.empty_cache()

    rows.sort(key=lambda row: (row["step"], row["checkpoint"]))
    if not rows:
        raise ValueError("No selected checkpoints matched the active DROID model.")
    fieldnames = list(rows[0])
    with (output_root / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    _write_svg_plot(
        plots_root / "reconstruction.svg", rows, ["rgb_l1", "depth_absrel", "flow_epe"]
    )
    _write_svg_plot(
        plots_root / "representation.svg",
        rows,
        ["positive_cosine", "negative_cosine", "patch_token_variance"],
    )
    _write_svg_plot(
        plots_root / "gaussians.svg",
        rows,
        ["gaussian_visibility", "gaussian_opacity_mean", "child_displacement_mean"],
    )

    ranks: dict[str, float] = {row["checkpoint"]: 0.0 for row in rows}
    for metric in ("rgb_l1", "depth_absrel", "flow_epe", "contrastive_loss"):
        for rank, row in enumerate(sorted(rows, key=lambda item: float(item[metric]))):
            ranks[row["checkpoint"]] += rank
    best_overall = min(ranks, key=ranks.get)
    collapse = [
        row["checkpoint"]
        for row in rows
        if row["positive_cosine"] - row["negative_cosine"] < 0.01
        or row["patch_token_variance"] < 1.0e-4
    ]
    summary = {
        "schema_version": 1,
        "gpu": gpu,
        "dataset_root": str(droid_root),
        "deterministic_validation_samples": sample_count,
        "seed": int(args.seed),
        "inventory": inventory,
        "evaluated_checkpoint_count": len(rows),
        "best_rgb_checkpoint": _best(rows, "rgb_l1"),
        "best_depth_checkpoint": _best(rows, "depth_absrel"),
        "best_dynamics_checkpoint": _best(rows, "flow_epe"),
        "best_representation_checkpoint": _best(
            rows, "contrastive_loss"
        ),
        "best_novel_view_checkpoint": None,
        "best_overall_checkpoint": best_overall,
        "analysis": {
            "representation_collapse_warning": collapse,
            "opacity_collapse_warning": [
                row["checkpoint"]
                for row in rows
                if row["gaussian_opacity_mean"] < 0.02
                or row["gaussian_opacity_mean"] > 0.98
            ],
            "novel_view_status": (
                "not_evaluated: official LagerNVS checkpoint is gated and no HF token is available"
            ),
            "lpips_status": "not_available in the active environment",
            "camera_leakage_status": (
                "not inferable from the short real-DROID pilot checkpoints"
            ),
        },
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
