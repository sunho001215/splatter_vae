from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from dataset.droid.cache import (
    CacheProvenance,
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
    sha256_file,
    verify_cuda_device,
)
from preprocessing.xlens.official import XLENS_PREPROCESSING_VERSION, XLensDROIDTeacher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute official X-Lens metric depth for DROID."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--maximum-episodes", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frame_stride != 1:
        raise ValueError(
            "Training caches must cover every DROID frame; frame_stride must be one."
        )
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    dataset_cfg = config["dataset"]
    depth_cfg = config["depth"]
    droid_root = Path(dataset_cfg["droid_root"]).expanduser().resolve()
    if not droid_root.is_dir():
        raise FileNotFoundError(f"DROID source is not mounted at {droid_root}.")
    output_root = validate_derived_root(
        Path(dataset_cfg["derived_root"]) / "xlens", droid_root
    )
    configure_external_model_caches(dataset_cfg["derived_root"], droid_root)
    checkpoint = depth_cfg.get("checkpoint_path")
    if not checkpoint:
        raise ValueError(
            "depth.checkpoint_path must point to an official released X-Lens checkpoint."
        )
    diagnostics = verify_cuda_device()
    print(json.dumps({"gpu": diagnostics}, indent=2), flush=True)
    repository = Path(depth_cfg["official_repo_path"]).expanduser().resolve()
    architecture_config = repository / "configs" / "xlens_vits.yaml"
    teacher = XLensDROIDTeacher(
        str(repository),
        str(checkpoint),
        architecture_config=str(architecture_config)
        if architecture_config.is_file()
        else None,
    )
    provenance = CacheProvenance(
        teacher_name="X-Lens",
        checkpoint=f"{Path(checkpoint).name}:{sha256_file(checkpoint)}",
        teacher_version=git_revision(repository),
        calibration_version=calibration_manifest_version(
            dataset_cfg["calibration_manifest"]
        ),
        preprocessing_version=XLENS_PREPROCESSING_VERSION,
        resolution=(320, 180),
    )
    entries = [
        entry
        for entry in load_calibration_manifest(dataset_cfg["calibration_manifest"])
        if entry.get("valid")
    ]
    if args.maximum_episodes is not None:
        entries = entries[: max(0, args.maximum_episodes)]
    backend = TFDSRLDSBackend(droid_root)
    with HDF5ShardWriter(
        output_root,
        provenance,
        shard_prefix="depth",
        items_per_shard=int(depth_cfg.get("items_per_shard", 64)),
        droid_root=droid_root,
    ) as writer:
        for episode_index, entry in enumerate(entries):
            episode = backend.get_episode(
                entry["rlds_split"], int(entry["rlds_ordinal"])
            )
            images = np.asarray(episode["images"], dtype=np.uint8)
            cameras = entry["exterior_cameras"]
            depths = [[], []]
            confidence = [[], []]
            validity = [[], []]
            for frame in range(int(entry["num_steps"])):
                prediction = teacher.predict(
                    [images[frame, 0], images[frame, 1]],
                    [
                        np.asarray(camera["intrinsics_rlds"], np.float32)
                        for camera in cameras
                    ],
                    [np.asarray(camera["c2w"], np.float32) for camera in cameras],
                )
                for camera_index in range(2):
                    depths[camera_index].append(
                        prediction["metric_depth"][camera_index]
                    )
                    confidence[camera_index].append(
                        prediction["confidence"][camera_index]
                    )
                    validity[camera_index].append(prediction["validity"][camera_index])
            for camera_index, camera in enumerate(cameras):
                writer.add(
                    sequence_cache_key(entry["episode_id"], camera["logical_id"]),
                    {
                        "metric_depth": np.asarray(
                            depths[camera_index], dtype=np.float16
                        ),
                        "confidence": np.asarray(
                            confidence[camera_index], dtype=np.float16
                        ),
                        "validity": np.asarray(validity[camera_index], dtype=np.uint8),
                    },
                    {
                        "episode_id": entry["episode_id"],
                        "logical_camera_id": camera["logical_id"],
                        "physical_serial": camera["serial"],
                        "calibration_source": camera["calibration_source"],
                        "pose_flag": camera["pose_flag"],
                        "frame_count": int(entry["num_steps"]),
                        "resolution": [320, 180],
                        "patch_padding_lrtb": [1, 1, 1, 1],
                    },
                )
            print(
                f"[X-Lens] {episode_index + 1}/{len(entries)} {entry['episode_id']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
