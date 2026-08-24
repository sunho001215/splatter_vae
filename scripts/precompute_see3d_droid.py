from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from dataset.droid.cache import (
    CacheProvenance,
    HDF5CacheReader,
    HDF5ShardWriter,
    calibration_manifest_version,
    sequence_cache_key,
)
from dataset.droid.calibration import load_calibration_manifest
from dataset.droid.rlds import TFDSRLDSBackend
from dataset.droid.safety import validate_derived_root
from preprocessing.common import (
    configure_external_model_caches,
    git_revision,
    path_identity,
    sha256_file,
    verify_cuda_device,
)
from preprocessing.see3d.geometry import (
    fuse_two_warps,
    interpolate_camera_pose,
    interpolate_intrinsics,
    synthetic_confidence,
    warp_rgbd_to_camera,
)
from preprocessing.see3d.official import (
    SEE3D_PREPROCESSING_VERSION,
    OfficialSee3DCompleter,
)
from preprocessing.xlens.official import XLensDROIDTeacher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute confidence-filtered DROID See3D views."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--maximum-episodes", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    dataset_cfg = config["dataset"]
    novel_cfg = config["novel_view"]
    see3d_cfg = novel_cfg["see3d"]
    training_cfg = novel_cfg["training"]
    warp_cfg = novel_cfg["geometric_warp"]
    pose_cfg = novel_cfg["pose_sampling"]
    droid_root = Path(dataset_cfg["droid_root"]).expanduser().resolve()
    if not droid_root.is_dir():
        raise FileNotFoundError(f"DROID source is not mounted at {droid_root}.")
    validation_summary_path = Path(novel_cfg["validation"]["summary_path"])
    if not validation_summary_path.is_file():
        raise RuntimeError("Run real-target See3D validation before precomputation.")
    validation_summary = json.loads(validation_summary_path.read_text(encoding="utf-8"))
    if not validation_summary.get("approved_for_precompute", False):
        raise RuntimeError(
            "See3D validation has not been explicitly approved for precomputation."
        )
    output_root = validate_derived_root(see3d_cfg["cache_root"], droid_root)
    configure_external_model_caches(dataset_cfg["derived_root"], droid_root)
    print(json.dumps({"gpu": verify_cuda_device()}, indent=2), flush=True)
    if not see3d_cfg.get("checkpoint_path") or not config["depth"].get(
        "checkpoint_path"
    ):
        raise ValueError("Official See3D and X-Lens checkpoint paths are required.")
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
    provenance = CacheProvenance(
        teacher_name="See3D+X-Lens",
        checkpoint=(
            f"see3d={path_identity(see3d_cfg['checkpoint_path'])}|"
            f"xlens={Path(config['depth']['checkpoint_path']).name}:"
            f"{sha256_file(config['depth']['checkpoint_path'])}"
        ),
        teacher_version=(
            f"see3d={git_revision(see3d_cfg['repo_path'])}|"
            f"xlens={git_revision(xlens_repository)}"
        ),
        calibration_version=calibration_manifest_version(
            dataset_cfg["calibration_manifest"]
        ),
        preprocessing_version=SEE3D_PREPROCESSING_VERSION,
        resolution=(320, 180),
    )
    depth_cache = HDF5CacheReader(dataset_cfg["xlens_cache_index"])
    depth_cache.require_compatible(
        teacher_name="X-Lens",
        calibration_version=provenance.calibration_version,
        checkpoint=(
            f"{Path(config['depth']['checkpoint_path']).name}:"
            f"{sha256_file(config['depth']['checkpoint_path'])}"
        ),
    )
    entries = [
        entry
        for entry in load_calibration_manifest(dataset_cfg["calibration_manifest"])
        if entry.get("valid")
    ]
    if args.maximum_episodes is not None:
        entries = entries[: max(0, args.maximum_episodes)]
    backend = TFDSRLDSBackend(droid_root)
    rng = random.Random(int(dataset_cfg.get("seed", 42)))
    samples_per_episode = int(training_cfg.get("samples_per_episode", 8))
    minimum_confidence = float(training_cfg["minimum_confidence"])
    rejection_reasons: Counter[str] = Counter()
    accepted_total = 0
    with HDF5ShardWriter(
        output_root,
        provenance,
        shard_prefix="see3d",
        items_per_shard=int(training_cfg.get("items_per_shard", 8)),
        droid_root=droid_root,
    ) as writer:
        for episode_index, entry in enumerate(entries):
            episode = backend.get_episode(
                entry["rlds_split"], int(entry["rlds_ordinal"])
            )
            images = np.asarray(episode["images"], dtype=np.uint8)
            cameras = entry["exterior_cameras"]
            candidates = list(range(int(entry["num_steps"])))
            rng.shuffle(candidates)
            candidates = candidates[: min(samples_per_episode, len(candidates))]
            accepted: dict[str, list[np.ndarray | float | int]] = {
                name: []
                for name in (
                    "timestep",
                    "alpha",
                    "virtual_K",
                    "virtual_c2w",
                    "virtual_w2c",
                    "generated_rgb",
                    "confidence",
                    "warp_validity",
                    "overlap",
                    "source_camera_ids",
                    "warp_depth",
                    "warp_depth_disagreement",
                    "warp_relative_depth_disagreement",
                    "warp_rgb_disagreement",
                    "metric_depth",
                    "depth_confidence",
                    "depth_validity",
                    "geometry_supported_depth",
                )
            }
            for timestep in candidates:
                alpha = rng.uniform(
                    float(pose_cfg["min_alpha"]), float(pose_cfg["max_alpha"])
                )
                virtual_c2w = interpolate_camera_pose(
                    cameras[0]["c2w"], cameras[1]["c2w"], alpha
                )
                virtual_w2c = np.linalg.inv(virtual_c2w)
                virtual_K = interpolate_intrinsics(
                    cameras[0]["intrinsics_rlds"], cameras[1]["intrinsics_rlds"], alpha
                )
                warps = []
                for camera_index, camera in enumerate(cameras):
                    depth = depth_cache.read_depth(
                        entry["episode_id"], camera["logical_id"], timestep
                    )
                    warps.append(
                        warp_rgbd_to_camera(
                            images[timestep, camera_index],
                            depth["metric_depth"],
                            depth["confidence"],
                            depth["validity"],
                            camera["intrinsics_rlds"],
                            camera["c2w"],
                            virtual_K,
                            virtual_c2w,
                        )
                    )
                fused = fuse_two_warps(
                    warps[0],
                    warps[1],
                    relative_depth_tolerance=float(
                        warp_cfg.get("relative_depth_tolerance", 0.05)
                    ),
                    absolute_depth_tolerance_m=float(
                        warp_cfg.get("absolute_depth_tolerance_m", 0.03)
                    ),
                )
                if not fused["validity"].any():
                    rejection_reasons["empty_geometric_warp"] += 1
                    continue
                raw_generated = completer.complete(
                    [images[timestep, 0], images[timestep, 1]],
                    fused["rgb"].astype(np.uint8),
                    fused["validity"],
                    preserve_observed_pixels=False,
                )
                confidence, view_confidence = synthetic_confidence(fused, raw_generated)
                if view_confidence < minimum_confidence:
                    rejection_reasons["low_view_confidence"] += 1
                    continue
                generated = raw_generated.copy()
                if bool(warp_cfg["preserve_observed_pixels"]):
                    observed = np.asarray(fused["validity"], dtype=bool)
                    generated[observed] = np.asarray(fused["rgb"], dtype=np.uint8)[
                        observed
                    ]
                synthetic_depth = xlens.predict(
                    [images[timestep, 0], images[timestep, 1], generated],
                    [
                        cameras[0]["intrinsics_rlds"],
                        cameras[1]["intrinsics_rlds"],
                        virtual_K,
                    ],
                    [cameras[0]["c2w"], cameras[1]["c2w"], virtual_c2w],
                )
                inferred_depth = synthetic_depth["metric_depth"][2]
                inferred_confidence = synthetic_depth["confidence"][2]
                inferred_validity = synthetic_depth["validity"][2]
                geometry_supported = np.asarray(fused["validity"], dtype=bool)
                metric_depth = np.where(
                    geometry_supported, fused["depth"], inferred_depth
                )
                depth_confidence = np.where(
                    geometry_supported,
                    fused["confidence"],
                    float(training_cfg.get("hallucinated_depth_confidence_scale", 0.25))
                    * inferred_confidence
                    * confidence,
                )
                depth_validity = geometry_supported | inferred_validity
                values = {
                    "timestep": int(timestep),
                    "alpha": float(alpha),
                    "virtual_K": virtual_K.astype(np.float32),
                    "virtual_c2w": virtual_c2w.astype(np.float32),
                    "virtual_w2c": virtual_w2c.astype(np.float32),
                    "generated_rgb": generated.astype(np.uint8),
                    "confidence": confidence.astype(np.float16),
                    "warp_validity": geometry_supported.astype(np.uint8),
                    "overlap": np.asarray(fused["overlap"], dtype=np.uint8),
                    "source_camera_ids": fused["source_camera_ids"].astype(np.int8),
                    "warp_depth": fused["depth"].astype(np.float16),
                    "warp_depth_disagreement": fused["depth_disagreement"].astype(
                        np.float16
                    ),
                    "warp_relative_depth_disagreement": fused[
                        "relative_depth_disagreement"
                    ].astype(np.float16),
                    "warp_rgb_disagreement": fused["rgb_disagreement"].astype(
                        np.float16
                    ),
                    "metric_depth": metric_depth.astype(np.float16),
                    "depth_confidence": depth_confidence.astype(np.float16),
                    "depth_validity": depth_validity.astype(np.uint8),
                    "geometry_supported_depth": geometry_supported.astype(np.uint8),
                }
                for name, value in values.items():
                    accepted[name].append(value)
            if accepted["timestep"]:
                writer.add(
                    sequence_cache_key(entry["episode_id"], "see3d"),
                    {name: np.asarray(values) for name, values in accepted.items()},
                    {
                        "episode_id": entry["episode_id"],
                        "source_logical_camera_ids": [
                            camera["logical_id"] for camera in cameras
                        ],
                        "source_physical_serials": [
                            camera["serial"] for camera in cameras
                        ],
                        "accepted_samples": len(accepted["timestep"]),
                        "pose_mode": "interpolate_translation_slerp_rotation",
                        "alpha_range": [pose_cfg["min_alpha"], pose_cfg["max_alpha"]],
                    },
                )
                accepted_total += len(accepted["timestep"])
            else:
                rejection_reasons["episode_no_accepted_views"] += 1
            print(
                f"[See3D cache] {episode_index + 1}/{len(entries)} {entry['episode_id']}",
                flush=True,
            )
    statistics = {
        "episodes_inspected": len(entries),
        "accepted_synthetic_views": accepted_total,
        "rejection_reason_counts": dict(rejection_reasons),
        "minimum_confidence": minimum_confidence,
        "validation_summary": str(validation_summary_path),
    }
    (output_root / "precompute_statistics.json").write_text(
        json.dumps(statistics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(statistics, indent=2))


if __name__ == "__main__":
    main()
