from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml

from dataset.droid.cache import HDF5CacheReader, calibration_manifest_version
from dataset.droid.calibration import load_calibration_manifest
from dataset.droid.safety import validate_derived_root
from preprocessing.workspace import (
    backproject_z_depth_to_world,
    percentile_dict,
    propose_gaussian_workspace_parameters,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute robust DROID base-frame workspace statistics."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--maximum-episodes", type=int, default=None)
    return parser.parse_args()


def _bounded_merge(
    existing: np.ndarray, new: np.ndarray, maximum: int, rng: np.random.Generator
) -> np.ndarray:
    if len(new) == 0:
        return existing
    combined = np.concatenate((existing, new), axis=0)
    if len(combined) <= maximum:
        return combined
    selected = rng.choice(len(combined), size=maximum, replace=False)
    return combined[selected]


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    dataset_cfg = config["dataset"]
    stats_cfg = config["workspace_statistics"]
    droid_root = Path(dataset_cfg["droid_root"]).expanduser().resolve()
    if not droid_root.is_dir():
        raise FileNotFoundError(f"DROID source is not mounted at {droid_root}.")
    output = Path(stats_cfg["output_path"]).expanduser().resolve(strict=False)
    validate_derived_root(output.parent, droid_root).mkdir(parents=True, exist_ok=True)
    entries = [
        entry
        for entry in load_calibration_manifest(dataset_cfg["calibration_manifest"])
        if entry.get("valid")
    ]
    random.Random(int(dataset_cfg.get("seed", 42))).shuffle(entries)
    maximum_episodes = args.maximum_episodes or int(stats_cfg["maximum_episodes"])
    entries = entries[:maximum_episodes]
    depth_cache = HDF5CacheReader(dataset_cfg["xlens_cache_index"])
    depth_cache.require_compatible(
        teacher_name="X-Lens",
        calibration_version=calibration_manifest_version(
            dataset_cfg["calibration_manifest"]
        ),
    )
    rng = np.random.default_rng(int(dataset_cfg.get("seed", 42)))
    point_samples = np.empty((0, 3), dtype=np.float32)
    depth_samples = np.empty((0,), dtype=np.float32)
    camera_positions = []
    maximum_points = int(stats_cfg.get("maximum_geometry_samples", 2_000_000))
    frames_per_episode = int(stats_cfg["frames_per_episode"])
    confidence_threshold = float(stats_cfg["depth_confidence_threshold"])
    for episode_index, entry in enumerate(entries):
        frames = np.linspace(
            0,
            int(entry["num_steps"]) - 1,
            num=min(frames_per_episode, int(entry["num_steps"])),
            dtype=np.int64,
        )
        for camera in entry["exterior_cameras"]:
            K = np.asarray(camera["intrinsics_rlds"], dtype=np.float64)
            c2w = np.asarray(camera["c2w"], dtype=np.float64)
            camera_positions.append(c2w[:3, 3])
            for frame in frames:
                item = depth_cache.read_depth(
                    entry["episode_id"], camera["logical_id"], int(frame)
                )
                depth = np.asarray(item["metric_depth"], dtype=np.float32)
                confidence = np.asarray(item["confidence"], dtype=np.float32)
                valid = (
                    np.asarray(item["validity"], dtype=bool)
                    & np.isfinite(depth)
                    & np.isfinite(confidence)
                    & (depth > 0.0)
                    & (confidence >= confidence_threshold)
                )
                points = backproject_z_depth_to_world(depth, K, c2w)[valid]
                valid_depth = depth[valid]
                if len(points) > 4096:
                    choice = rng.choice(len(points), size=4096, replace=False)
                    points = points[choice]
                    valid_depth = valid_depth[choice]
                point_samples = _bounded_merge(
                    point_samples, points.astype(np.float32), maximum_points, rng
                )
                depth_samples = _bounded_merge(
                    depth_samples, valid_depth.astype(np.float32), maximum_points, rng
                )
        print(
            f"[workspace] {episode_index + 1}/{len(entries)} {entry['episode_id']}",
            flush=True,
        )
    proposal = propose_gaussian_workspace_parameters(point_samples, depth_samples)
    payload = {
        "schema_version": 1,
        "world_frame": "robot_base",
        "depth_teacher": "X-Lens",
        "episodes": len(entries),
        "geometry_sample_count": len(point_samples),
        "coordinate_median": np.median(point_samples, axis=0).tolist(),
        "coordinate_percentiles": percentile_dict(
            point_samples, stats_cfg["coordinate_percentiles"]
        ),
        "depth_percentiles": percentile_dict(
            depth_samples, stats_cfg["depth_percentiles"]
        ),
        "camera_position_percentiles": percentile_dict(
            np.asarray(camera_positions), stats_cfg["coordinate_percentiles"]
        ),
        "proposed_parameters": asdict(proposal),
        "parameter_status": "computed_from_droid_xlens_stats_requires_training_pilot",
    }
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
