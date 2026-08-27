from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

from dataset.droid.dataset import DROIDLogicalDataset, droid_collate
from dataset.droid.safety import source_tree_fingerprint, validate_derived_root
from models.training.distributed import move_to_device
from models.training.losses import masked_rgb_reconstruction_losses
from preprocessing.common import verify_cuda_device
from preprocessing.lagernvs.camera import canonical_intrinsics
from preprocessing.lagernvs.official import LagerNVSDROIDTeacher
from preprocessing.lagernvs.pose import LagerTargetPoseConfig, sample_safe_target_poses
from preprocessing.xlens.official import XLensDROIDTeacher
from scripts.train_droid import (
    _dataset_config,
    _only_dataclass_fields,
    _validate_fixed_pipeline_contract,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate official LagerNVS on real calibrated DROID views."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--workspace-stats", required=True)
    parser.add_argument(
        "--output-root", default="outputs/teacher_validation/lagernvs/inference"
    )
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _selected_indices(dataset: DROIDLogicalDataset, count: int) -> list[int]:
    ranges = dataset.episode_index_ranges
    selected = np.linspace(
        0, len(ranges) - 1, num=min(int(count), len(ranges)), dtype=np.int64
    )
    return [int(ranges[index][0]) for index in selected]


def _cuda_measure(callable_):
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.reset_peak_memory_stats()
    begin.record()
    output = callable_()
    end.record()
    end.synchronize()
    return output, {
        "cuda_ms": float(begin.elapsed_time(end)),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / (1024**3),
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / (1024**3),
    }


def _rgb_panel(image: torch.Tensor, title: str) -> Image.Image:
    value = (
        image.detach()
        .float()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .byte()
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    panel = Image.fromarray(value, mode="RGB")
    canvas = Image.new("RGB", (panel.width, panel.height + 24), "white")
    canvas.paste(panel, (0, 24))
    ImageDraw.Draw(canvas).text((5, 5), title, fill="black")
    return canvas


def _mask_panel(mask: torch.Tensor, title: str) -> Image.Image:
    value = mask.detach().float().clamp(0.0, 1.0).mul(255.0).byte().cpu().numpy()
    if value.ndim == 3:
        value = value[0]
    return _rgb_panel(
        torch.from_numpy(value).float()[None].expand(3, -1, -1) / 255.0,
        title,
    )


def _save_grid(path: Path, panels: list[Image.Image]) -> None:
    width = sum(panel.width for panel in panels)
    height = max(panel.height for panel in panels)
    canvas = Image.new("RGB", (width, height), "white")
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width
    canvas.save(path, quality=94)


def _known_pose_metrics(
    predicted: torch.Tensor,
    target: torch.Tensor,
    validity: torch.Tensor,
) -> dict[str, float | str]:
    l1, dssim, _ = masked_rgb_reconstruction_losses(predicted, target, validity)
    difference = (predicted.float() - target.float()).square().mean(dim=1, keepdim=True)
    mse = (difference * validity.float()).sum() / validity.float().sum().clamp_min(1.0)
    return {
        "rgb_l1": float(l1),
        "psnr": float(-10.0 * torch.log10(mse.clamp_min(1.0e-12))),
        "ssim": float(1.0 - 2.0 * dssim),
        "lpips": float("nan"),
        "lpips_status": "not_available_in_active_environment",
        "interpretation": "source-pose self-reconstruction upper bound; target source is visible to the teacher",
    }


def _blocked_payload(
    *,
    config: dict[str, Any],
    gpu: dict[str, Any],
    droid_root: Path,
    source_fingerprint: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    novel = config["novel_view"]
    return {
        "schema_version": 1,
        "status": "blocked",
        "reason": reason,
        "gpu": gpu,
        "dataset_root": str(droid_root),
        "dataset_read_only": True,
        "source_fingerprint": source_fingerprint,
        "checkpoint_id": novel["checkpoint_id"],
        "checkpoint_revision": novel["checkpoint_revision"],
        "repository_revision": novel["official_repository_revision"],
        "blocked_outputs": [
            "real_droid_lagernvs_inference",
            "known_pose_psnr_ssim_lpips",
            "novel_target_quality_grids",
        ],
    }


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    _validate_fixed_pipeline_contract(config)
    droid_root = Path(config["dataset"]["droid_root"]).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve(strict=False)
    validate_derived_root(output_root, droid_root).mkdir(parents=True, exist_ok=True)
    (output_root / "qualitative").mkdir(exist_ok=True)
    source_before = source_tree_fingerprint(droid_root)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly the authorized GPU for LagerNVS validation.")
    device = torch.device("cuda:0")
    gpu = verify_cuda_device()
    novel = config["novel_view"]

    try:
        load_started = time.perf_counter()
        lager = LagerNVSDROIDTeacher(
            novel["official_repo_path"],
            novel.get("checkpoint_path"),
            cache_dir=novel.get("cache_dir"),
            device=device,
            dtype=torch.bfloat16,
            microbatch_size=int(novel["microbatch_size"]),
            canonical_focal_px=float(novel["canonical"]["focal_px"]),
        )
        torch.cuda.synchronize(device)
        lager_load_seconds = time.perf_counter() - load_started
    except Exception as exc:  # noqa: BLE001 - write the exact external blocker
        payload = _blocked_payload(
            config=config,
            gpu=gpu,
            droid_root=droid_root,
            source_fingerprint=source_before,
            reason=f"{type(exc).__name__}: {exc}",
        )
        (output_root / "summary.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(payload, indent=2), flush=True)
        return

    depth_cfg = config["depth"]
    xlens = XLensDROIDTeacher(
        str(depth_cfg["official_repo_path"]),
        str(depth_cfg["checkpoint_path"]),
        architecture_config=depth_cfg.get("architecture_config"),
        device=str(device),
        amp_dtype=str(depth_cfg.get("amp_dtype", "bf16")),
    )
    dataset = DROIDLogicalDataset(_dataset_config(config, "validation"))
    indices = _selected_indices(dataset, int(args.samples))
    cpu_batch = droid_collate([dataset[index] for index in indices])
    raw = move_to_device(cpu_batch, device)
    depth, xlens_timing = _cuda_measure(
        lambda: xlens(raw["raw_histories"], raw["raw_K"], raw["raw_c2w"])
    )

    workspace = json.loads(Path(args.workspace_stats).read_text(encoding="utf-8"))
    scene_center = torch.tensor(
        workspace["proposed_parameters"]["global_center"],
        device=device,
        dtype=torch.float32,
    )
    pose_values = dict(novel["target_pose"])
    pose_values.pop("mode", None)
    pose_values.pop("scene_center", None)
    pose_config = LagerTargetPoseConfig(
        **_only_dataclass_fields(LagerTargetPoseConfig, pose_values)
    )
    target_K = canonical_intrinsics(
        (len(indices),),
        focal_px=float(novel["canonical"]["focal_px"]),
        device=device,
    )
    scenarios = (
        (
            "near_source_alpha_0.20",
            replace(
                pose_config,
                alpha_min=0.20,
                alpha_max=0.20,
                translation_max_baseline_fraction=0.0,
                rotation_max_degrees=0.0,
            ),
        ),
        (
            "mid_interpolation_alpha_0.50",
            replace(
                pose_config,
                alpha_min=0.50,
                alpha_max=0.50,
                translation_max_baseline_fraction=0.0,
                rotation_max_degrees=0.0,
            ),
        ),
        ("bounded_perturbation", pose_config),
    )
    rows = []
    timings = []
    current_rgb = raw["raw_histories"][:, :, 2]
    for scenario_index, (scenario, scenario_config) in enumerate(scenarios):
        target, pose_timing = _cuda_measure(
            lambda scenario_config=scenario_config, scenario_index=scenario_index: sample_safe_target_poses(
                raw["raw_c2w"],
                raw["raw_K"],
                target_K,
                depth["metric_depth"][:, 2],
                depth["confidence"][:, 2],
                depth["validity"][:, 2],
                scene_center,
                scenario_config,
                seed=int(args.seed) + scenario_index,
            )
        )
        prepared, preprocessing_timing = _cuda_measure(
            lambda target=target: lager.prepare_inputs(
                current_rgb,
                raw["raw_K"],
                raw["raw_c2w"],
                target["target_c2w"],
            )
        )
        generated, inference_timing = _cuda_measure(
            lambda prepared=prepared: lager.infer_prepared(prepared)
        )
        timings.append(
            {
                "scenario": scenario,
                "pose_sampling": pose_timing,
                "preprocessing": preprocessing_timing,
                "inference": inference_timing,
            }
        )
        for sample in range(len(indices)):
            rows.append(
                {
                    "scenario": scenario,
                    "sample": sample,
                    "dataset_index": indices[sample],
                    "episode_id": cpu_batch["episode_id"][sample],
                    "alpha": float(target["alpha"][sample]),
                    "source_coverage": float(target["source_coverage"][sample]),
                    "translation_jitter_m": float(
                        target["translation_perturbation_magnitude"][sample]
                    ),
                    "rotation_jitter_deg": float(
                        target["rotation_perturbation_degrees"][sample]
                    ),
                    "rejected_candidates": int(
                        target["rejected_candidates"][sample]
                    ),
                    "fallback_used": bool(target["fallback_used"][sample]),
                }
            )
            _save_grid(
                output_root
                / "qualitative"
                / f"{scenario}-sample-{sample:02d}.jpg",
                [
                    _rgb_panel(
                        generated["canonical_source_rgb"][sample, 0], "source A"
                    ),
                    _rgb_panel(
                        generated["canonical_source_rgb"][sample, 1], "source B"
                    ),
                    _rgb_panel(generated["generated_rgb"][sample], "LagerNVS target"),
                    _mask_panel(target["support_mask"][sample], "X-Lens support"),
                ],
            )

    known_pose = {}
    for camera in range(2):
        prepared = lager.prepare_inputs(
            current_rgb,
            raw["raw_K"],
            raw["raw_c2w"],
            raw["raw_c2w"][:, camera],
        )
        output, timing = _cuda_measure(
            lambda prepared=prepared: lager.infer_prepared(prepared)
        )
        known_pose[f"camera_{camera}"] = {
            **_known_pose_metrics(
                output["generated_rgb"],
                output["canonical_source_rgb"][:, camera],
                output["canonical_source_validity"][:, camera],
            ),
            "timing": timing,
        }

    with (output_root / "target_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    source_after = source_tree_fingerprint(droid_root)
    if source_after != source_before:
        raise RuntimeError("The read-only DROID source changed during validation.")
    payload = {
        "schema_version": 1,
        "status": "success",
        "gpu": gpu,
        "dataset_root": str(droid_root),
        "dataset_read_only": True,
        "source_fingerprint": source_after,
        "dataset_indices": indices,
        "checkpoint_id": novel["checkpoint_id"],
        "checkpoint_revision": novel["checkpoint_revision"],
        "repository_revision": novel["official_repository_revision"],
        "checkpoint_path": lager.checkpoint_path,
        "cold_load_seconds": lager_load_seconds,
        "xlens_timing": xlens_timing,
        "scenario_timings": timings,
        "known_source_pose_upper_bound": known_pose,
        "novel_target_count": len(rows),
        "coverage_min": min(float(row["source_coverage"]) for row in rows),
        "coverage_mean": sum(float(row["source_coverage"]) for row in rows)
        / len(rows),
        "novel_view_ground_truth_status": (
            "unavailable: DROID exposes only the two synchronized calibrated exterior views; "
            "known-pose metrics are explicitly a source-visible upper bound"
        ),
    }
    (output_root / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
