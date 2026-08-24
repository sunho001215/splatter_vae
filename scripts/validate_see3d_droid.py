from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from dataset.droid.cache import HDF5CacheReader, calibration_manifest_version
from dataset.droid.calibration import load_calibration_manifest
from dataset.droid.rlds import TFDSRLDSBackend
from dataset.droid.safety import validate_derived_root
from preprocessing.common import (
    configure_external_model_caches,
    git_revision,
    sha256_file,
    verify_cuda_device,
)
from preprocessing.see3d.geometry import synthetic_confidence, warp_rgbd_to_camera
from preprocessing.see3d.metrics import PerceptualMetrics, depth_metrics, rgb_metrics
from preprocessing.see3d.official import OfficialSee3DCompleter
from preprocessing.see3d.visualization import (
    colorize_scalar,
    save_see3d_validation_grid,
)
from preprocessing.xlens.official import XLensDROIDTeacher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate official See3D on held-out real DROID views."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--maximum-samples", type=int, default=None)
    parser.add_argument(
        "--approve-for-precompute",
        action="store_true",
        help="Approve an already completed/inspected validation for cache generation.",
    )
    parser.add_argument(
        "--approve-for-training",
        action="store_true",
        help="Approve an already completed/inspected validation for training.",
    )
    parser.add_argument(
        "--approve-existing-summary",
        action="store_true",
        help="Update only the existing summary; never bypasses the validation run.",
    )
    return parser.parse_args()


def _warp_as_fused(warp: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    zeros = np.zeros_like(warp["depth"], dtype=np.float32)
    return {
        **warp,
        "relative_depth_disagreement": zeros,
        "depth_disagreement": zeros,
        "rgb_disagreement": zeros,
        "overlap": np.zeros_like(warp["validity"], dtype=bool),
    }


def _region_metrics(
    generated: np.ndarray,
    target: np.ndarray,
    generated_depth: np.ndarray,
    target_depth: np.ndarray,
    depth_validity: np.ndarray,
    mask: np.ndarray,
    perceptual: PerceptualMetrics,
) -> dict[str, float]:
    result = rgb_metrics(generated, target, mask)
    result.update(depth_metrics(generated_depth, target_depth, depth_validity & mask))
    result.update(perceptual(generated, target, mask))
    return result


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    dataset_cfg = config["dataset"]
    novel_cfg = config["novel_view"]
    see3d_cfg = novel_cfg["see3d"]
    validation_cfg = novel_cfg["validation"]
    droid_root = Path(dataset_cfg["droid_root"]).expanduser().resolve()
    configured_summary = Path(validation_cfg["summary_path"]).expanduser().resolve()
    summary_path = (
        validate_derived_root(configured_summary.parent, droid_root)
        / configured_summary.name
    )
    approval_requested = bool(args.approve_for_precompute or args.approve_for_training)
    if args.approve_existing_summary:
        if not approval_requested:
            raise ValueError(
                "--approve-existing-summary requires --approve-for-precompute or "
                "--approve-for-training."
            )
        if not summary_path.is_file():
            raise FileNotFoundError(
                "No completed See3D validation summary exists to approve."
            )
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if not bool(existing.get("validation_completed")):
            raise RuntimeError(
                "The existing See3D summary is not a completed validation."
            )
        existing["approved_for_precompute"] = bool(
            existing.get("approved_for_precompute")
            or args.approve_for_precompute
            or args.approve_for_training
        )
        existing["approved_for_training"] = bool(
            existing.get("approved_for_training") or args.approve_for_training
        )
        temporary = summary_path.with_suffix(summary_path.suffix + ".partial")
        temporary.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
        temporary.replace(summary_path)
        print(json.dumps(existing, indent=2))
        return
    if approval_requested:
        raise ValueError(
            "Approval is deliberately separate from inference. First run validation and inspect "
            "its metrics/grids, then rerun with --approve-existing-summary."
        )
    if not droid_root.is_dir():
        raise FileNotFoundError(f"DROID source is not mounted at {droid_root}.")
    output_root = validate_derived_root(
        Path(dataset_cfg["derived_root"]) / "see3d" / "validation", droid_root
    )
    output_root.mkdir(parents=True, exist_ok=True)
    configure_external_model_caches(dataset_cfg["derived_root"], droid_root)
    print(json.dumps({"gpu": verify_cuda_device()}, indent=2), flush=True)
    if not see3d_cfg.get("checkpoint_path"):
        raise ValueError(
            "novel_view.see3d.checkpoint_path must point to official released weights."
        )
    if not config["depth"].get("checkpoint_path"):
        raise ValueError(
            "depth.checkpoint_path is required to evaluate generated-view geometry."
        )
    completer = OfficialSee3DCompleter(
        see3d_cfg["repo_path"],
        see3d_cfg["checkpoint_path"],
        seed=int(dataset_cfg.get("seed", 42)),
        super_resolution=bool(see3d_cfg.get("super_resolution", False)),
    )
    xlens_repository = Path(config["depth"]["official_repo_path"])
    xlens = XLensDROIDTeacher(
        str(xlens_repository),
        str(config["depth"]["checkpoint_path"]),
        architecture_config=str(xlens_repository / "configs" / "xlens_vits.yaml"),
    )
    perceptual = PerceptualMetrics(
        dinov2_repository=validation_cfg.get("dinov2_repository")
    )
    depth_cache = HDF5CacheReader(dataset_cfg["xlens_cache_index"])
    flow_cache = HDF5CacheReader(dataset_cfg["waft_cache_index"])
    manifest_version = calibration_manifest_version(dataset_cfg["calibration_manifest"])
    depth_cache.require_compatible(
        teacher_name="X-Lens",
        calibration_version=manifest_version,
        checkpoint=(
            f"{Path(config['depth']['checkpoint_path']).name}:"
            f"{sha256_file(config['depth']['checkpoint_path'])}"
        ),
    )
    flow_cache.require_compatible(
        teacher_name="WAFT",
        calibration_version=manifest_version,
    )
    entries = [
        entry
        for entry in load_calibration_manifest(dataset_cfg["calibration_manifest"])
        if entry.get("valid")
        and entry.get("dataset_split") == "validation"
        and int(entry["num_steps"]) > 4
    ]
    rng = random.Random(int(dataset_cfg.get("seed", 42)))
    rng.shuffle(entries)
    maximum = args.maximum_samples or int(validation_cfg.get("maximum_samples", 100))
    entries = entries[:maximum]
    backend = TFDSRLDSBackend(droid_root)
    rows: list[dict[str, Any]] = []
    motion_threshold = float(validation_cfg.get("motion_threshold_pixels", 3.0))
    for sample_index, entry in enumerate(entries):
        timestep = rng.randint(0, int(entry["num_steps"]) - 4)
        episode = backend.get_episode(entry["rlds_split"], int(entry["rlds_ordinal"]))
        images = np.asarray(episode["images"], dtype=np.uint8)
        for source_index, target_index in ((0, 1), (1, 0)):
            source_camera = entry["exterior_cameras"][source_index]
            target_camera = entry["exterior_cameras"][target_index]
            source_depth = depth_cache.read_depth(
                entry["episode_id"], source_camera["logical_id"], timestep
            )
            warp = warp_rgbd_to_camera(
                images[timestep, source_index],
                source_depth["metric_depth"],
                source_depth["confidence"],
                source_depth["validity"],
                source_camera["intrinsics_rlds"],
                source_camera["c2w"],
                target_camera["intrinsics_rlds"],
                target_camera["c2w"],
            )
            raw_generated = completer.complete(
                [images[timestep, source_index]],
                warp["rgb"].astype(np.uint8),
                warp["validity"],
                preserve_observed_pixels=False,
            )
            confidence_map, view_confidence = synthetic_confidence(
                _warp_as_fused(warp), raw_generated
            )
            generated = raw_generated.copy()
            if bool(novel_cfg["geometric_warp"]["preserve_observed_pixels"]):
                observed = np.asarray(warp["validity"], dtype=bool)
                generated[observed] = np.asarray(warp["rgb"], dtype=np.uint8)[observed]
            generated_depth_prediction = xlens.predict(
                [images[timestep, source_index], generated],
                [source_camera["intrinsics_rlds"], target_camera["intrinsics_rlds"]],
                [source_camera["c2w"], target_camera["c2w"]],
            )
            generated_depth = generated_depth_prediction["metric_depth"][1]
            generated_depth_validity = generated_depth_prediction["validity"][1]
            real_depth = depth_cache.read_depth(
                entry["episode_id"], target_camera["logical_id"], timestep
            )
            teacher_flow = flow_cache.read_flow(
                entry["episode_id"], target_camera["logical_id"], timestep, 3
            )
            motion = np.linalg.norm(
                np.asarray(teacher_flow["forward_flow"], np.float32), axis=-1
            )
            flow_validity = np.asarray(teacher_flow["validity"], dtype=bool)
            low_motion = flow_validity & (motion <= motion_threshold)
            high_motion = flow_validity & (motion > motion_threshold)
            depth_validity = (
                generated_depth_validity
                & np.asarray(real_depth["validity"], dtype=bool)
                & (np.asarray(real_depth["confidence"]) > 0)
            )
            metrics_by_region = {
                "all": _region_metrics(
                    generated,
                    images[timestep, target_index],
                    generated_depth,
                    real_depth["metric_depth"],
                    depth_validity,
                    np.ones((180, 320), dtype=bool),
                    perceptual,
                ),
                "low_motion": _region_metrics(
                    generated,
                    images[timestep, target_index],
                    generated_depth,
                    real_depth["metric_depth"],
                    depth_validity,
                    low_motion,
                    perceptual,
                ),
                "high_motion": _region_metrics(
                    generated,
                    images[timestep, target_index],
                    generated_depth,
                    real_depth["metric_depth"],
                    depth_validity,
                    high_motion,
                    perceptual,
                ),
            }
            row: dict[str, Any] = {
                "episode_id": entry["episode_id"],
                "timestep": timestep,
                "direction": f"{source_camera['logical_id']}->{target_camera['logical_id']}",
                "source_serial": source_camera["serial"],
                "target_serial": target_camera["serial"],
                "warp_coverage": float(np.asarray(warp["validity"]).mean()),
                "view_confidence": view_confidence,
            }
            for region, metrics in metrics_by_region.items():
                for name, value in metrics.items():
                    row[f"{region}/{name}"] = value
            rows.append(row)
            error = np.abs(
                generated.astype(np.float32) - images[timestep, target_index]
            ).mean(axis=-1)
            depth_error = np.abs(
                generated_depth - np.asarray(real_depth["metric_depth"], np.float32)
            )
            direction = f"{source_index}_to_{target_index}"
            save_see3d_validation_grid(
                output_root / "grids" / f"{sample_index:05d}_{direction}.jpg",
                [
                    ("real source RGB", images[timestep, source_index]),
                    ("geometric target warp", warp["rgb"]),
                    (
                        "warp validity / holes",
                        np.asarray(warp["validity"], np.uint8) * 255,
                    ),
                    ("See3D completion", generated),
                    ("real target RGB", images[timestep, target_index]),
                    ("RGB error", colorize_scalar(error, minimum=0, maximum=64)),
                    (
                        "real-target X-Lens depth",
                        colorize_scalar(real_depth["metric_depth"]),
                    ),
                    ("generated-view X-Lens depth", colorize_scalar(generated_depth)),
                    (
                        "depth error",
                        colorize_scalar(depth_error, minimum=0, maximum=0.5),
                    ),
                    (
                        "WAFT motion magnitude",
                        colorize_scalar(motion, minimum=0, maximum=16),
                    ),
                    (
                        "synthetic confidence",
                        colorize_scalar(confidence_map, minimum=0, maximum=1),
                    ),
                ],
            )
        print(
            f"[See3D validation] {sample_index + 1}/{len(entries)} {entry['episode_id']}",
            flush=True,
        )
    if not rows:
        raise RuntimeError(
            "No calibration-valid held-out DROID samples were available for See3D validation."
        )
    numeric_keys = [
        key
        for key, value in rows[0].items()
        if isinstance(value, (int, float)) and key != "timestep"
    ]
    aggregate = {
        key: float(np.nanmean([float(row[key]) for row in rows]))
        for key in numeric_keys
    }
    summary = {
        "validation_completed": True,
        "approved_for_precompute": False,
        "approved_for_training": False,
        "sample_directions": len(rows),
        "metrics": aggregate,
        "see3d_revision": git_revision(see3d_cfg["repo_path"]),
        "stage0_calibration_manifest_only": True,
        "qualitative_review_targets": [
            "robot arm",
            "gripper",
            "manipulated object",
            "interaction/contact regions",
            "thin structures",
            "occlusion boundaries",
        ],
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (output_root / "per_sample_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_root / "per_sample_metrics.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
