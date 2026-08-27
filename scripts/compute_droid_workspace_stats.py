from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
import yaml

from dataset.droid.calibration import load_calibration_manifest
from dataset.droid.rlds import TFDSRLDSBackend
from dataset.droid.safety import source_tree_fingerprint, validate_derived_root
from preprocessing.common import sha256_file, verify_cuda_device
from preprocessing.workspace import (
    backproject_z_depth_to_world,
    estimate_scene_center_from_camera_axes,
    percentile_dict,
    propose_gaussian_workspace_parameters,
)
from preprocessing.xlens.official import XLensDROIDTeacher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute robust DROID workspace statistics with online X-Lens."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--maximum-episodes", type=int, default=None)
    parser.add_argument("--frames-per-episode", type=int, default=None)
    parser.add_argument("--teacher-frame-batch", type=int, default=3)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output", default=None)
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


def _camera_tensors(entry: dict, images: np.ndarray, frames: np.ndarray):
    history = torch.from_numpy(images[frames]).permute(1, 0, 4, 2, 3)[None]
    K = torch.from_numpy(
        np.stack(
            [camera["intrinsics_rlds"] for camera in entry["exterior_cameras"]]
        ).astype(np.float32)
    )[None]
    c2w = torch.from_numpy(
        np.stack([camera["c2w"] for camera in entry["exterior_cameras"]]).astype(
            np.float32
        )
    )[None]
    return history, K, c2w


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    dataset_cfg = config["dataset"]
    depth_cfg = config["depth"]
    stats_cfg = config["workspace_statistics"]
    droid_root = Path(dataset_cfg["droid_root"]).expanduser().resolve()
    if not droid_root.is_dir():
        raise FileNotFoundError(f"DROID source is not mounted at {droid_root}.")
    output = Path(args.output or stats_cfg["output_path"]).expanduser().resolve(
        strict=False
    )
    validate_derived_root(output.parent, droid_root).mkdir(parents=True, exist_ok=True)
    source_before = source_tree_fingerprint(droid_root)
    manifest_path = Path(
        args.manifest or dataset_cfg["calibration_manifest"]
    ).expanduser().resolve()
    entries = [
        entry
        for entry in load_calibration_manifest(manifest_path)
        if entry.get("valid")
    ]
    random.Random(int(dataset_cfg.get("seed", 42))).shuffle(entries)
    maximum_episodes = args.maximum_episodes or int(stats_cfg["maximum_episodes"])
    entries = entries[:maximum_episodes]
    if not entries:
        raise ValueError("No calibration-valid DROID episodes are available.")
    camera_pose_pairs = np.stack(
        [
            np.stack(
                [np.asarray(camera["c2w"]) for camera in entry["exterior_cameras"]]
            )
            for entry in entries
        ]
    )
    scene_center, per_episode_scene_centers, axis_diagnostics = (
        estimate_scene_center_from_camera_axes(
            camera_pose_pairs,
            maximum_axis_residual_m=float(
                stats_cfg.get("maximum_axis_intersection_residual_m", 0.35)
            ),
        )
    )
    workspace_radius_m = float(stats_cfg.get("workspace_geometry_radius_m", 1.0))

    diagnostics = verify_cuda_device()
    device = torch.device("cuda:0")
    teacher = XLensDROIDTeacher(
        str(depth_cfg["official_repo_path"]),
        str(depth_cfg["checkpoint_path"]),
        architecture_config=depth_cfg.get("architecture_config"),
        device=str(device),
        amp_dtype=str(depth_cfg.get("amp_dtype", "bf16")),
    )
    backend = TFDSRLDSBackend(droid_root)
    rng = np.random.default_rng(int(dataset_cfg.get("seed", 42)))
    point_samples = np.empty((0, 3), dtype=np.float32)
    depth_samples = np.empty((0,), dtype=np.float32)
    all_valid_depth_samples = np.empty((0,), dtype=np.float32)
    confidence_samples = np.empty((0,), dtype=np.float32)
    camera_positions = camera_pose_pairs[..., :3, 3].reshape(-1, 3)
    unfiltered_geometry_count = 0
    maximum_points = int(stats_cfg.get("maximum_geometry_samples", 2_000_000))
    frames_per_episode = args.frames_per_episode or int(
        stats_cfg["frames_per_episode"]
    )
    confidence_threshold = float(stats_cfg["depth_confidence_threshold"])
    frame_batch = max(1, int(args.teacher_frame_batch))
    inference_ms: list[float] = []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)

    for episode_index, entry in enumerate(entries):
        episode = backend.get_episode(
            str(entry["rlds_split"]), int(entry["rlds_ordinal"])
        )
        images = np.asarray(episode["images"], dtype=np.uint8)
        frames = np.linspace(
            0,
            int(entry["num_steps"]) - 1,
            num=min(frames_per_episode, int(entry["num_steps"])),
            dtype=np.int64,
        )
        for start in range(0, len(frames), frame_batch):
            selected = frames[start : start + frame_batch]
            histories, K, c2w = _camera_tensors(entry, images, selected)
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            prediction = teacher(histories, K, c2w)
            end.record()
            end.synchronize()
            inference_ms.append(float(begin.elapsed_time(end)))
            depth = prediction["metric_depth"][0, :, :, 0].cpu().numpy()
            confidence = prediction["confidence"][0, :, :, 0].cpu().numpy()
            validity = prediction["validity"][0, :, :, 0].cpu().numpy()
            for time_index in range(len(selected)):
                for camera_index, camera in enumerate(entry["exterior_cameras"]):
                    depth_image = depth[time_index, camera_index]
                    confidence_image = confidence[time_index, camera_index]
                    valid = (
                        validity[time_index, camera_index]
                        & np.isfinite(depth_image)
                        & np.isfinite(confidence_image)
                        & (depth_image > 0.0)
                        & (confidence_image >= confidence_threshold)
                    )
                    points = backproject_z_depth_to_world(
                        depth_image,
                        np.asarray(camera["intrinsics_rlds"], dtype=np.float64),
                        np.asarray(camera["c2w"], dtype=np.float64),
                    )[valid]
                    valid_depth = depth_image[valid]
                    valid_confidence = confidence_image[valid]
                    unfiltered_geometry_count += len(points)
                    all_valid_depth_samples = _bounded_merge(
                        all_valid_depth_samples,
                        valid_depth.astype(np.float32),
                        maximum_points,
                        rng,
                    )
                    workspace_geometry = (
                        np.linalg.norm(points - scene_center[None], axis=-1)
                        <= workspace_radius_m
                    )
                    points = points[workspace_geometry]
                    valid_depth = valid_depth[workspace_geometry]
                    valid_confidence = valid_confidence[workspace_geometry]
                    if len(points) > 4096:
                        choice = rng.choice(len(points), size=4096, replace=False)
                        points = points[choice]
                        valid_depth = valid_depth[choice]
                        valid_confidence = valid_confidence[choice]
                    point_samples = _bounded_merge(
                        point_samples, points.astype(np.float32), maximum_points, rng
                    )
                    depth_samples = _bounded_merge(
                        depth_samples,
                        valid_depth.astype(np.float32),
                        maximum_points,
                        rng,
                    )
                    confidence_samples = _bounded_merge(
                        confidence_samples,
                        valid_confidence.astype(np.float32),
                        maximum_points,
                        rng,
                    )
        print(
            f"[workspace online X-Lens] {episode_index + 1}/{len(entries)} "
            f"{entry['episode_id']}",
            flush=True,
        )

    depth_low, depth_high = np.percentile(all_valid_depth_samples, (1.0, 99.0))
    workspace_depth_low = float(np.percentile(depth_samples, 1.0))
    proposal = replace(
        propose_gaussian_workspace_parameters(point_samples, depth_samples),
        global_center=tuple(float(value) for value in scene_center),
        znear=max(0.01, min(float(depth_low), workspace_depth_low) * 0.5),
        zfar=max(float(depth_low) + 0.1, float(depth_high) * 1.2),
    )
    elapsed = time.perf_counter() - started
    payload = {
        "schema_version": 2,
        "world_frame": "robot_base",
        "dataset_root": str(droid_root),
        "dataset_read_only": True,
        "calibration_manifest": str(manifest_path),
        "depth_teacher": "X-Lens online",
        "xlens_checkpoint": str(Path(depth_cfg["checkpoint_path"]).resolve()),
        "xlens_checkpoint_sha256": sha256_file(depth_cfg["checkpoint_path"]),
        "gpu": diagnostics,
        "episodes": len(entries),
        "frames_per_episode": frames_per_episode,
        "scene_center_method": "median_forward_camera_optical_axis_intersection",
        "scene_center": scene_center.tolist(),
        "scene_center_pair_count": len(per_episode_scene_centers),
        "scene_center_pair_percentiles": percentile_dict(
            per_episode_scene_centers, stats_cfg["coordinate_percentiles"]
        ),
        "camera_axis_distance_and_residual_percentiles": percentile_dict(
            axis_diagnostics, stats_cfg["coordinate_percentiles"]
        ),
        "workspace_geometry_radius_m": workspace_radius_m,
        "unfiltered_geometry_sample_count": unfiltered_geometry_count,
        "geometry_sample_count": len(point_samples),
        "coordinate_median": np.median(point_samples, axis=0).tolist(),
        "coordinate_percentiles": percentile_dict(
            point_samples, stats_cfg["coordinate_percentiles"]
        ),
        "depth_percentiles": percentile_dict(
            all_valid_depth_samples, stats_cfg["depth_percentiles"]
        ),
        "workspace_filtered_depth_percentiles": percentile_dict(
            depth_samples, stats_cfg["depth_percentiles"]
        ),
        "confidence_percentiles": percentile_dict(
            confidence_samples, stats_cfg["depth_percentiles"]
        ),
        "camera_position_percentiles": percentile_dict(
            np.asarray(camera_positions), stats_cfg["coordinate_percentiles"]
        ),
        "proposed_parameters": asdict(proposal),
        "runtime": {
            "total_seconds": elapsed,
            "xlens_calls": len(inference_ms),
            "xlens_call_mean_ms": float(np.mean(inference_ms)),
            "xlens_call_median_ms": float(np.median(inference_ms)),
            "xlens_call_p95_ms": float(np.percentile(inference_ms, 95.0)),
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / (1024**3),
        },
        "proposed_parameter_status": (
            "computed_from_online_droid_xlens_stats_requires_training_pilot"
        ),
    }
    source_after = source_tree_fingerprint(droid_root)
    if source_before != source_after:
        raise RuntimeError("The read-only DROID source fingerprint changed during stats.")
    payload["source_fingerprint"] = source_after
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
