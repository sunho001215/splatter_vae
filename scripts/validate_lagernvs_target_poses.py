from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image

from dataset.droid.dataset import DROIDLogicalDataset, droid_collate
from dataset.droid.safety import source_tree_fingerprint, validate_derived_root
from models.training.distributed import move_to_device
from preprocessing.common import sha256_file, verify_cuda_device
from preprocessing.lagernvs.camera import canonical_intrinsics
from preprocessing.lagernvs.pose import LagerTargetPoseConfig, sample_safe_target_poses
from preprocessing.xlens.official import XLensDROIDTeacher
from scripts.train_droid import (
    _dataset_config,
    _only_dataclass_fields,
    _validate_fixed_pipeline_contract,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate real-DROID LagerNVS target poses with online X-Lens source "
            "coverage, without requiring the gated LagerNVS checkpoint."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--workspace-stats", required=True)
    parser.add_argument(
        "--output-root", default="outputs/teacher_validation/lagernvs/target_poses"
    )
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--draws-per-sample", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _selected_indices(dataset: DROIDLogicalDataset, count: int) -> list[int]:
    ranges = dataset.episode_index_ranges
    if not ranges:
        raise ValueError("No calibration-valid validation episodes are available.")
    selected = np.linspace(
        0, len(ranges) - 1, num=min(int(count), len(ranges)), dtype=np.int64
    )
    return [int(ranges[index][0]) for index in selected]


def _percentiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(array.min()),
        "p05": float(np.percentile(array, 5)),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "p95": float(np.percentile(array, 95)),
        "maximum": float(array.max()),
    }


def _correlation(rows: list[dict[str, Any]], name: str) -> float | None:
    x = np.asarray([float(row[name]) for row in rows], dtype=np.float64)
    y = np.asarray([float(row["source_coverage"]) for row in rows], dtype=np.float64)
    if len(x) < 2 or float(x.std()) < 1.0e-12 or float(y.std()) < 1.0e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _write_scatter_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    width, height, margin = 900, 420, 55
    coverage = np.asarray([row["source_coverage"] for row in rows], dtype=float)
    alpha = np.asarray([row["alpha"] for row in rows], dtype=float)
    jitter = np.asarray([row["translation_baseline_fraction"] for row in rows])
    low = min(0.55, float(coverage.min()) - 0.02)
    high = min(1.0, max(0.65, float(coverage.max()) + 0.01))
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="black"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="black"/>',
        f'<text x="{width/2}" y="{height-12}" text-anchor="middle">target alpha</text>',
        f'<text x="18" y="{height/2}" transform="rotate(-90 18 {height/2})" text-anchor="middle">source coverage</text>',
        f'<line x1="{margin}" y1="{height-margin-(0.60-low)/(high-low)*(height-2*margin)}" x2="{width-margin}" y2="{height-margin-(0.60-low)/(high-low)*(height-2*margin)}" stroke="#d62728" stroke-dasharray="6 4"/>',
    ]
    for a, c, j in zip(alpha, coverage, jitter, strict=True):
        x = margin + (a - 0.15) / 0.70 * (width - 2 * margin)
        y = height - margin - (c - low) / max(high - low, 1.0e-8) * (
            height - 2 * margin
        )
        radius = 2.5 + 80.0 * float(j)
        elements.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" fill="#1f77b4" fill-opacity="0.55"/>'
        )
    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    _validate_fixed_pipeline_contract(config)
    droid_root = Path(config["dataset"]["droid_root"]).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve(strict=False)
    validate_derived_root(output_root, droid_root).mkdir(parents=True, exist_ok=True)
    (output_root / "plots").mkdir(exist_ok=True)
    (output_root / "qualitative").mkdir(exist_ok=True)
    source_before = source_tree_fingerprint(droid_root)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly the authorized GPU for target-pose validation.")
    device = torch.device("cuda:0")
    gpu = verify_cuda_device()

    depth_cfg = config["depth"]
    teacher = XLensDROIDTeacher(
        str(depth_cfg["official_repo_path"]),
        str(depth_cfg["checkpoint_path"]),
        architecture_config=depth_cfg.get("architecture_config"),
        device=str(device),
        amp_dtype=str(depth_cfg.get("amp_dtype", "bf16")),
    )
    dataset = DROIDLogicalDataset(_dataset_config(config, "validation"))
    indices = _selected_indices(dataset, int(args.samples))
    decode_started = time.perf_counter()
    cpu_batch = droid_collate([dataset[index] for index in indices])
    decode_seconds = time.perf_counter() - decode_started
    raw = move_to_device(cpu_batch, device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    xlens = teacher(raw["raw_histories"], raw["raw_K"], raw["raw_c2w"])
    end.record()
    end.synchronize()
    xlens_ms = float(start.elapsed_time(end))

    novel = config["novel_view"]
    pose_values = dict(novel["target_pose"])
    pose_values.pop("mode", None)
    pose_values.pop("scene_center", None)
    pose_config = LagerTargetPoseConfig(
        **_only_dataclass_fields(LagerTargetPoseConfig, pose_values)
    )
    workspace = json.loads(Path(args.workspace_stats).read_text(encoding="utf-8"))
    scene_center = torch.tensor(
        workspace["proposed_parameters"]["global_center"],
        device=device,
        dtype=torch.float32,
    )
    target_K = canonical_intrinsics(
        (len(indices),),
        focal_px=float(novel["canonical"]["focal_px"]),
        device=device,
    )
    rows: list[dict[str, Any]] = []
    sampling_ms: list[float] = []
    qualitative_written = 0

    def record_result(
        result: dict[str, Any],
        *,
        draw_label: str,
        scenario: str,
        difficulty_group: str,
        write_qualitative: bool,
    ) -> None:
        nonlocal qualitative_written
        for sample in range(len(indices)):
            baseline = float(result["baseline"][sample])
            translation = float(result["translation_perturbation_magnitude"][sample])
            rotation = float(result["rotation_perturbation_degrees"][sample])
            alpha = float(result["alpha"][sample])
            row = {
                "sample": sample,
                "dataset_index": indices[sample],
                "episode_id": cpu_batch["episode_id"][sample],
                "draw": draw_label,
                "scenario": scenario,
                "alpha": alpha,
                "interpolation_mode": "scene_centered_arc"
                if bool(result["scene_centered_arc_used"][sample])
                else "linear_fallback",
                "baseline_m": baseline,
                "translation_jitter_m": translation,
                "translation_baseline_fraction": translation / max(baseline, 1.0e-8),
                "rotation_jitter_deg": rotation,
                "source_coverage": float(result["source_coverage"][sample]),
                "distance_from_camera_a_m": float(
                    result["distance_from_camera_a"][sample]
                ),
                "distance_from_camera_b_m": float(
                    result["distance_from_camera_b"][sample]
                ),
                "rejected_candidates": int(result["rejected_candidates"][sample]),
                "fallback_used": bool(result["fallback_used"][sample]),
                "final_reason": result["final_reason"][sample],
                "difficulty_group": difficulty_group,
            }
            rows.append(row)
            if qualitative_written < 12 and (
                write_qualitative
                or row["rejected_candidates"]
                or row["fallback_used"]
            ):
                support = (
                    result["support_mask"][sample, 0]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                    * 255
                )
                Image.fromarray(support, mode="L").save(
                    output_root
                    / "qualitative"
                    / f"support-s{sample:02d}-{draw_label}.png"
                )
                qualitative_written += 1

    controlled = (
        ("near_source_alpha_0.20", 0.20, "near_source_minimal_perturbation"),
        ("mid_interpolation_alpha_0.50", 0.50, "mid_interpolation_minimal_perturbation"),
        ("near_source_alpha_0.80", 0.80, "near_source_minimal_perturbation"),
    )
    for scenario_index, (scenario, alpha, group) in enumerate(controlled):
        controlled_config = replace(
            pose_config,
            alpha_min=alpha,
            alpha_max=alpha,
            translation_max_baseline_fraction=0.0,
            rotation_max_degrees=0.0,
        )
        start.record()
        result = sample_safe_target_poses(
            raw["raw_c2w"],
            raw["raw_K"],
            target_K,
            xlens["metric_depth"][:, 2],
            xlens["confidence"][:, 2],
            xlens["validity"][:, 2],
            scene_center,
            controlled_config,
            seed=int(args.seed) + scenario_index,
        )
        end.record()
        end.synchronize()
        sampling_ms.append(float(start.elapsed_time(end)))
        record_result(
            result,
            draw_label=f"controlled-{scenario_index}",
            scenario=scenario,
            difficulty_group=group,
            write_qualitative=scenario_index == 0,
        )

    for draw in range(int(args.draws_per_sample)):
        start.record()
        result = sample_safe_target_poses(
            raw["raw_c2w"],
            raw["raw_K"],
            target_K,
            xlens["metric_depth"][:, 2],
            xlens["confidence"][:, 2],
            xlens["validity"][:, 2],
            scene_center,
            pose_config,
            seed=int(args.seed) + 1000 + draw,
        )
        end.record()
        end.synchronize()
        sampling_ms.append(float(start.elapsed_time(end)))
        record_result(
            result,
            draw_label=f"random-{draw:03d}",
            scenario="default_bounded_perturbation",
            difficulty_group="bounded_perturbation",
            write_qualitative=draw == 0,
        )

    with (output_root / "target_pose_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _write_scatter_svg(output_root / "plots" / "coverage_vs_difficulty.svg", rows)
    groups = sorted({row["difficulty_group"] for row in rows})
    summary = {
        "schema_version": 1,
        "gpu": gpu,
        "dataset_root": str(droid_root),
        "dataset_read_only": True,
        "dataset_indices": indices,
        "episode_ids": list(cpu_batch["episode_id"]),
        "scene_center": scene_center.cpu().tolist(),
        "configuration": {
            "alpha_min": pose_config.alpha_min,
            "alpha_max": pose_config.alpha_max,
            "prefer_scene_centered_arc": pose_config.prefer_scene_centered_arc,
            "translation_max_baseline_fraction": pose_config.translation_max_baseline_fraction,
            "rotation_max_degrees": pose_config.rotation_max_degrees,
            "min_source_coverage": pose_config.min_source_coverage,
            "max_resample_attempts": pose_config.max_resample_attempts,
        },
        "candidates": len(rows),
        "alpha": _percentiles([float(row["alpha"]) for row in rows]),
        "coverage": _percentiles([float(row["source_coverage"]) for row in rows]),
        "translation_baseline_fraction": _percentiles(
            [float(row["translation_baseline_fraction"]) for row in rows]
        ),
        "rotation_jitter_deg": _percentiles(
            [float(row["rotation_jitter_deg"]) for row in rows]
        ),
        "distance_from_camera_a_m": _percentiles(
            [float(row["distance_from_camera_a_m"]) for row in rows]
        ),
        "distance_from_camera_b_m": _percentiles(
            [float(row["distance_from_camera_b_m"]) for row in rows]
        ),
        "scene_centered_arc_fraction": statistics.fmean(
            row["interpolation_mode"] == "scene_centered_arc" for row in rows
        ),
        "candidate_rejection_count": sum(
            int(row["rejected_candidates"]) for row in rows
        ),
        "samples_with_rejection_fraction": statistics.fmean(
            int(row["rejected_candidates"]) > 0 for row in rows
        ),
        "fallback_fraction": statistics.fmean(bool(row["fallback_used"]) for row in rows),
        "below_coverage_threshold_fraction": statistics.fmean(
            float(row["source_coverage"]) < pose_config.min_source_coverage
            for row in rows
        ),
        "difficulty_groups": {
            group: {
                "count": sum(row["difficulty_group"] == group for row in rows),
                "coverage": _percentiles(
                    [
                        float(row["source_coverage"])
                        for row in rows
                        if row["difficulty_group"] == group
                    ]
                ),
            }
            for group in groups
        },
        "coverage_correlations": {
            name: _correlation(rows, name)
            for name in (
                "alpha",
                "translation_jitter_m",
                "rotation_jitter_deg",
                "distance_from_camera_a_m",
                "distance_from_camera_b_m",
            )
        },
        "timing": {
            "droid_decode_seconds": decode_seconds,
            "xlens_batch_ms": xlens_ms,
            "xlens_ms_per_logical_sample": xlens_ms / len(indices),
            "pose_sampling_coverage_ms_per_batch_mean": statistics.fmean(sampling_ms),
            "pose_sampling_coverage_ms_per_target_mean": statistics.fmean(sampling_ms)
            / len(indices),
        },
        "xlens_checkpoint": str(Path(depth_cfg["checkpoint_path"]).resolve()),
        "xlens_checkpoint_sha256": sha256_file(depth_cfg["checkpoint_path"]),
        "lagernvs_inference_status": (
            "not_run: official facebook/lagernvs_dl3dv_2-6_v_256 checkpoint is gated"
        ),
        "source_fingerprint": source_tree_fingerprint(droid_root),
    }
    if summary["source_fingerprint"] != source_before:
        raise RuntimeError("The read-only DROID source changed during validation.")
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
