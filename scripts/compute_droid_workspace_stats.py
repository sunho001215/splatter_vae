from __future__ import annotations

import argparse
import json
import random
import time
from collections import OrderedDict
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from dataset.droid.codecs import (
    NumericCodecConfig,
    decode_numeric_array,
    depth_u16_to_meters,
)
from dataset.droid.integrity import BoundedPrioritySample
from dataset.droid.preprocessed_manifest import (
    load_stage0_manifest,
    sample_key,
    shard_path,
)
from dataset.droid.safety import validate_derived_root
from dataset.droid.shards import IndexedTarReader, write_json_atomic
from preprocessing.workspace import (
    backproject_z_depth_to_world,
    estimate_scene_center_from_camera_axes,
    percentile_dict,
    propose_gaussian_workspace_parameters,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute DROID workspace statistics from cached DA3 metric depth."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--preprocessed-root", default=None)
    parser.add_argument("--maximum-episodes", type=int, default=None)
    parser.add_argument("--frames-per-episode", type=int, default=None)
    parser.add_argument("--maximum-points", type=int, default=2_000_000)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def _reader(
    cache: OrderedDict[int, IndexedTarReader],
    root: Path,
    shard_id: int,
    *,
    maximum_open: int = 8,
) -> IndexedTarReader:
    identifier = int(shard_id)
    if identifier in cache:
        value = cache.pop(identifier)
        cache[identifier] = value
        return value
    value = IndexedTarReader(shard_path(root, "final", identifier))
    cache[identifier] = value
    while len(cache) > maximum_open:
        _removed_id, removed = cache.popitem(last=False)
        removed.close()
    return value


def _parameter_difference(
    proposal: dict[str, Any],
    decoder: dict[str, Any],
    renderer: dict[str, Any],
) -> dict[str, Any]:
    previous = {
        "global_center": decoder["global_center"],
        "anchor_initial_spread": decoder["anchor_initial_spread"],
        "parent_displacement_scale": decoder["parent_displacement_scale"],
        "child_radius": decoder["child_radius"],
        "znear": renderer["znear"],
        "zfar": renderer["zfar"],
    }
    difference: dict[str, Any] = {}
    for name, new_value in proposal.items():
        old_value = previous[name]
        if isinstance(new_value, (list, tuple)):
            difference[name] = (
                np.asarray(new_value, dtype=np.float64)
                - np.asarray(old_value, dtype=np.float64)
            ).tolist()
        else:
            difference[name] = float(new_value) - float(old_value)
    return {"previous_config_values": previous, "absolute_difference": difference}


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    dataset_cfg = config["dataset"]
    stats_cfg = config["workspace_statistics"]
    root = Path(
        args.preprocessed_root or dataset_cfg["preprocessed_root"]
    ).expanduser().resolve()
    manifest = load_stage0_manifest(root)
    output = Path(args.output or stats_cfg["output_path"]).expanduser().resolve(
        strict=False
    )
    validate_derived_root(output.parent, manifest["source_root"])
    if root == output or root in output.parents and output.suffix == "":
        raise ValueError("Workspace-statistics output must be a file, not the dataset root.")

    entries = list(manifest["episodes"])
    random.Random(int(dataset_cfg.get("seed", 42))).shuffle(entries)
    maximum_episodes = args.maximum_episodes or int(stats_cfg["maximum_episodes"])
    entries = entries[: int(maximum_episodes)]
    if not entries:
        raise ValueError("The cached Stage-0 manifest has no eligible episodes.")
    frames_per_episode = args.frames_per_episode or int(
        stats_cfg["frames_per_episode"]
    )
    maximum_points = int(args.maximum_points)
    if frames_per_episode <= 0 or maximum_points < 100:
        raise ValueError("frames-per-episode must be positive and maximum-points >= 100.")

    camera_pose_pairs = np.stack(
        [
            np.stack(
                [
                    np.asarray(camera["c2w"], dtype=np.float64)
                    for camera in entry["exterior_cameras"]
                ]
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
    sampling_seed = int(dataset_cfg.get("seed", 42))
    sample_seeds = np.random.SeedSequence(sampling_seed).generate_state(3)
    point_sampler = BoundedPrioritySample(
        capacity=maximum_points,
        seed=int(sample_seeds[0]),
        item_shape=(3,),
    )
    workspace_depth_sampler = BoundedPrioritySample(
        capacity=maximum_points,
        seed=int(sample_seeds[1]),
    )
    all_depth_sampler = BoundedPrioritySample(
        capacity=maximum_points,
        seed=int(sample_seeds[2]),
    )
    camera_positions = camera_pose_pairs[..., :3, 3].reshape(-1, 3)
    unfiltered_geometry_count = 0
    decoded_maps = 0
    decoded_payload_bytes = 0
    readers: OrderedDict[int, IndexedTarReader] = OrderedDict()
    numeric_cfg = config["storage"]["numeric"]
    numeric = NumericCodecConfig(
        cname=str(numeric_cfg["codec"]),
        compression_level=int(numeric_cfg["compression_level"]),
        shuffle=str(numeric_cfg["shuffle"]),
    )
    started = time.perf_counter()
    try:
        for episode_index, entry in enumerate(entries):
            retained_count = int(entry["retained_count"])
            selected = np.unique(
                np.linspace(
                    0,
                    retained_count - 1,
                    num=min(int(frames_per_episode), retained_count),
                    dtype=np.int64,
                )
            )
            reader = _reader(readers, root, int(entry["shard_id"]))
            K = np.stack(
                [
                    np.asarray(camera["intrinsics_rlds"], dtype=np.float64)
                    for camera in entry["exterior_cameras"]
                ]
            )
            c2w = np.stack(
                [
                    np.asarray(camera["c2w"], dtype=np.float64)
                    for camera in entry["exterior_cameras"]
                ]
            )
            for retained_index in selected:
                key = sample_key(
                    int(entry["global_retained_start"]) + int(retained_index)
                )
                payload = reader.read(key, "depth", verify=True)
                decoded_payload_bytes += len(payload)
                encoded = decode_numeric_array(payload, numeric)
                depth, valid = depth_u16_to_meters(encoded)
                if depth.shape != (2, 180, 320):
                    raise ValueError(f"Cached DA3 depth {key} has shape {depth.shape}.")
                decoded_maps += 2
                for camera in range(2):
                    depth_image = depth[camera]
                    usable = (
                        valid[camera]
                        & np.isfinite(depth_image)
                        & (depth_image > 0.0)
                    )
                    points = backproject_z_depth_to_world(
                        depth_image, K[camera], c2w[camera]
                    )[usable]
                    depth_values = depth_image[usable]
                    unfiltered_geometry_count += len(points)
                    all_depth_sampler.add(depth_values, maximum_per_item=1024)
                    in_workspace = (
                        np.linalg.norm(points - scene_center[None], axis=-1)
                        <= workspace_radius_m
                    )
                    points = points[in_workspace]
                    depth_values = depth_values[in_workspace]
                    point_sampler.add(points, maximum_per_item=1024)
                    workspace_depth_sampler.add(depth_values, maximum_per_item=1024)
            print(
                f"[workspace cached DA3] {episode_index + 1}/{len(entries)} "
                f"{entry['episode_id']}",
                flush=True,
            )
    finally:
        for reader in readers.values():
            reader.close()

    point_samples = point_sampler.result()
    workspace_depth_samples = workspace_depth_sampler.result()
    all_depth_samples = all_depth_sampler.result()
    if len(all_depth_samples) < 100 or len(point_samples) < 100:
        raise ValueError("Cached DA3 produced too few valid workspace geometry samples.")
    depth_low, depth_high = np.percentile(all_depth_samples, (1.0, 99.0))
    workspace_depth_low = float(np.percentile(workspace_depth_samples, 1.0))
    proposal_value = replace(
        propose_gaussian_workspace_parameters(
            point_samples, workspace_depth_samples
        ),
        global_center=tuple(float(value) for value in scene_center),
        znear=max(0.01, min(float(depth_low), workspace_depth_low) * 0.5),
        zfar=max(float(depth_low) + 0.1, float(depth_high) * 1.2),
    )
    proposal = asdict(proposal_value)
    elapsed = time.perf_counter() - started
    payload = {
        "schema_version": 3,
        "world_frame": "robot_base",
        "depth_source": "cached_da3",
        "preprocessed_root": str(root),
        "dataset_schema_signature": manifest["schema_signature"],
        "source_calibration_sha256": manifest["source_calibration_sha256"],
        "da3": {
            "repository": config["depth"]["repository"],
            "repository_revision": config["depth"]["repository_revision"],
            "model_id": config["depth"]["model_id"],
            "checkpoint_revision": config["depth"]["checkpoint_revision"],
            "encoding": config["depth"]["encoding"],
        },
        "episodes": len(entries),
        "frames_per_episode": int(frames_per_episode),
        "decoded_depth_maps": decoded_maps,
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
        "bounded_sampling": {
            "method": "deterministic_per_frame_linspace_then_priority_reservoir",
            "base_seed": sampling_seed,
            "independent_reservoir_seeds": [int(value) for value in sample_seeds],
            "maximum_samples_per_distribution": maximum_points,
            "maximum_samples_per_camera_frame": 1024,
        },
        "unfiltered_geometry_sample_count": unfiltered_geometry_count,
        "geometry_sample_count": len(point_samples),
        "coordinate_median": np.median(point_samples, axis=0).tolist(),
        "coordinate_percentiles": percentile_dict(
            point_samples, stats_cfg["coordinate_percentiles"]
        ),
        "depth_percentiles": percentile_dict(
            all_depth_samples, stats_cfg["depth_percentiles"]
        ),
        "workspace_filtered_depth_percentiles": percentile_dict(
            workspace_depth_samples, stats_cfg["depth_percentiles"]
        ),
        "camera_position_percentiles": percentile_dict(
            np.asarray(camera_positions), stats_cfg["coordinate_percentiles"]
        ),
        "proposed_parameters": proposal,
        "comparison_to_config": _parameter_difference(
            proposal, config["decoder"], config["renderer"]
        ),
        "runtime": {
            "total_seconds": elapsed,
            "decoded_payload_bytes": decoded_payload_bytes,
            "decoded_maps_per_second": decoded_maps / max(elapsed, 1.0e-9),
        },
        "workspace_parameter_status": "computed_from_cached_da3_requires_geometry_pilot",
    }
    write_json_atomic(output, payload)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
