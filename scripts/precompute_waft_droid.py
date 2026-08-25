from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
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
from preprocessing.waft import (
    WAFT_PREPROCESSING_VERSION,
    load_official_waft,
    make_waft_predictor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute official WAFT flow for DROID RLDS."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--maximum-episodes", type=int, default=None)
    parser.add_argument("--maximum-frames", type=int, default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--depth-checkpoint", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    dataset_cfg = config["dataset"]
    flow_cfg = config["flow"]
    droid_root = Path(dataset_cfg["droid_root"]).expanduser().resolve()
    if not droid_root.is_dir():
        raise FileNotFoundError(f"DROID source is not mounted at {droid_root}.")
    output_root = validate_derived_root(
        args.output_root or Path(dataset_cfg["derived_root"]) / "waft", droid_root
    )
    configure_external_model_caches(dataset_cfg["derived_root"], droid_root)
    required_paths = {
        "checkpoint_path": args.checkpoint or flow_cfg.get("checkpoint_path"),
        "depth_checkpoint_path": args.depth_checkpoint
        or flow_cfg.get("depth_checkpoint_path"),
        "config_path": flow_cfg.get("config_path"),
    }
    missing = [name for name, value in required_paths.items() if not value]
    if missing:
        raise ValueError(f"WAFT configuration is missing {missing}.")
    diagnostics = verify_cuda_device()
    print(json.dumps({"gpu": diagnostics}, indent=2), flush=True)
    device = torch.device("cuda:0")
    repository = Path(flow_cfg["official_repo_path"]).expanduser().resolve()
    model = load_official_waft(
        str(repository),
        str(required_paths["config_path"]),
        str(required_paths["checkpoint_path"]),
        str(required_paths["depth_checkpoint_path"]),
        device,
    )
    predictor = make_waft_predictor(
        model, device, iterations=flow_cfg.get("iterations")
    )
    torch.cuda.reset_peak_memory_stats(device)
    checkpoint_identity = "|".join(
        f"{Path(path).name}:{sha256_file(path)}"
        for path in (
            required_paths["checkpoint_path"],
            required_paths["depth_checkpoint_path"],
            required_paths["config_path"],
        )
    )
    manifest_path = args.manifest or dataset_cfg["calibration_manifest"]
    provenance = CacheProvenance(
        teacher_name="WAFT",
        checkpoint=checkpoint_identity,
        teacher_version=git_revision(repository),
        calibration_version=calibration_manifest_version(manifest_path),
        preprocessing_version=WAFT_PREPROCESSING_VERSION,
        resolution=(320, 180),
    )
    strides = tuple(int(value) for value in dataset_cfg["temporal_strides"])
    required_gaps = tuple(
        sorted({gap for stride in strides for gap in (stride, 2 * stride)})
    )
    gaps = tuple(sorted({int(value) for value in flow_cfg["cached_gaps"]}))
    if gaps != required_gaps:
        raise ValueError(
            "WAFT cached gaps must equal {stride,2*stride} for every configured "
            f"temporal stride: expected {required_gaps}, got {gaps}."
        )
    entries = [
        entry
        for entry in load_calibration_manifest(manifest_path)
        if entry.get("valid")
    ]
    if args.maximum_episodes is not None:
        entries = entries[: max(0, args.maximum_episodes)]
    backend = TFDSRLDSBackend(droid_root)
    batch_size = int(flow_cfg.get("inference_batch_size", 16))
    predicted_pairs = 0
    with HDF5ShardWriter(
        output_root,
        provenance,
        shard_prefix="flow",
        items_per_shard=int(flow_cfg.get("items_per_shard", 8)),
        droid_root=droid_root,
    ) as writer:
        for episode_index, entry in enumerate(entries):
            episode = backend.get_episode(
                entry["rlds_split"], int(entry["rlds_ordinal"])
            )
            images = np.asarray(episode["images"], dtype=np.uint8)
            if args.maximum_frames is not None:
                images = images[: max(0, int(args.maximum_frames))]
            if len(images) <= max(gaps):
                raise ValueError(
                    f"Episode {entry['episode_id']} has only {len(images)} selected frames; "
                    f"WAFT gaps require more than {max(gaps)}."
                )
            for camera_index, camera in enumerate(entry["exterior_cameras"]):
                frames = images[:, camera_index]
                arrays: dict[str, np.ndarray] = {}
                for gap in gaps:
                    count = max(0, len(frames) - gap)
                    flow = np.empty((count, 180, 320, 2), dtype=np.float16)
                    validity = np.empty((count, 180, 320), dtype=np.uint8)
                    confidence = np.empty((count, 180, 320), dtype=np.float16)
                    uncertainty = np.empty((count, 180, 320), dtype=np.float16)
                    for start in range(0, count, batch_size):
                        stop = min(count, start + batch_size)
                        prediction = predictor(
                            frames[start:stop], frames[start + gap : stop + gap]
                        )
                        predicted_pairs += stop - start
                        predicted_flow = prediction["forward_flow"]
                        if predicted_flow.shape != (stop - start, 180, 320, 2):
                            raise RuntimeError(
                                f"WAFT returned unexpected shape {predicted_flow.shape}."
                            )
                        predicted_confidence = prediction["confidence"]
                        predicted_uncertainty = prediction["uncertainty"]
                        finite = (
                            np.isfinite(predicted_flow).all(axis=-1)
                            & np.isfinite(predicted_confidence)
                            & np.isfinite(predicted_uncertainty)
                        )
                        flow[start:stop] = np.nan_to_num(
                            predicted_flow, nan=0.0, posinf=0.0, neginf=0.0
                        ).astype(np.float16)
                        validity[start:stop] = finite.astype(np.uint8)
                        confidence[start:stop] = np.where(
                            finite, np.clip(predicted_confidence, 0.0, 1.0), 0.0
                        ).astype(np.float16)
                        uncertainty[start:stop] = np.where(
                            finite, predicted_uncertainty, 0.0
                        ).astype(np.float16)
                    arrays[f"forward_flow_gap_{gap}"] = flow
                    arrays[f"validity_gap_{gap}"] = validity
                    arrays[f"confidence_gap_{gap}"] = confidence
                    arrays[f"uncertainty_gap_{gap}"] = uncertainty
                writer.add(
                    sequence_cache_key(entry["episode_id"], camera["logical_id"]),
                    arrays,
                    {
                        "episode_id": entry["episode_id"],
                        "logical_camera_id": camera["logical_id"],
                        "physical_serial": camera["serial"],
                        "frame_count": int(entry["num_steps"]),
                        "cached_frame_count": int(len(images)),
                        "gaps": list(gaps),
                        "direction": "forward",
                        "flow_units": "RLDS_pixels",
                        "confidence": "official_WAFT_predictive_distribution",
                        "forward_backward_consistency": False,
                        "resolution": [320, 180],
                    },
                )
            print(
                f"[WAFT] {episode_index + 1}/{len(entries)} {entry['episode_id']}",
                flush=True,
            )
    print(
        json.dumps(
            {
                "waft_sanity": {
                    "episodes": len(entries),
                    "predicted_camera_pairs": predicted_pairs,
                    "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device)
                    / (1024**3),
                    "output_root": str(output_root),
                }
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
