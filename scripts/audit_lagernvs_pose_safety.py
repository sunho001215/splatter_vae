from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from dataset.droid.codecs import decode_numeric_array, depth_u16_to_meters
from dataset.droid.preprocessed_manifest import (
    load_stage0_manifest,
    sample_key,
    shard_path,
)
from dataset.droid.records import lager_record_contract
from dataset.droid.safety import validate_derived_root
from dataset.droid.shards import IndexedTarReader, shard_is_complete, write_json_atomic
from preprocessing.lagernvs.camera import canonical_intrinsics
from preprocessing.lagernvs.pose import (
    LagerTargetPoseConfig,
    pose_sampler_contract,
    sample_safe_target_poses,
)
from preprocessing.stage0.workflow import deterministic_item_seed

POSE_AUDIT_SCHEMA_VERSION = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resumably audit LagerNVS target-pose safety from cached DA3 depth."
    )
    parser.add_argument("--root", default="/home/ws/data/droid_stage0_preprocessed")
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--worker-count", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-completed-lager", action="store_true")
    parser.add_argument("--max-shards", type=int)
    return parser.parse_args()


def _empty_statistics() -> dict[str, int | float]:
    return {
        "retained_timestamps": 0,
        "targets": 0,
        "fallback_targets": 0,
        "exceptional_translation_targets": 0,
        "exceptional_safety_threshold_targets": 0,
        "minimum_coverage": 1.0,
        "minimum_clearance_m": float("inf"),
        "maximum_translation_baseline_fraction": 0.0,
        "maximum_rotation_degrees": 0.0,
    }


def _merge_statistics(
    aggregate: dict[str, int | float], value: dict[str, int | float]
) -> None:
    for key in (
        "retained_timestamps",
        "targets",
        "fallback_targets",
        "exceptional_translation_targets",
        "exceptional_safety_threshold_targets",
    ):
        aggregate[key] = int(aggregate[key]) + int(value[key])
    aggregate["minimum_coverage"] = min(
        float(aggregate["minimum_coverage"]), float(value["minimum_coverage"])
    )
    aggregate["minimum_clearance_m"] = min(
        float(aggregate["minimum_clearance_m"]),
        float(value["minimum_clearance_m"]),
    )
    aggregate["maximum_translation_baseline_fraction"] = max(
        float(aggregate["maximum_translation_baseline_fraction"]),
        float(value["maximum_translation_baseline_fraction"]),
    )
    aggregate["maximum_rotation_degrees"] = max(
        float(aggregate["maximum_rotation_degrees"]),
        float(value["maximum_rotation_degrees"]),
    )


def _report_is_complete(
    path: Path,
    schema_signature: str,
    pose_contract_signature: str,
    *,
    shard_id: int,
    retained_timestamps: int,
) -> bool:
    if not path.is_file():
        return False
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        return (
            report.get("status") == "pass"
            and report.get("schema_signature") == schema_signature
            and report.get("pose_contract_signature") == pose_contract_signature
            and int(report.get("pose_audit_schema_version", -1))
            == POSE_AUDIT_SCHEMA_VERSION
            and int(report.get("shard_id", -1)) == int(shard_id)
            and int(report.get("statistics", {}).get("retained_timestamps", -1))
            == int(retained_timestamps)
            and int(report.get("statistics", {}).get("targets", -1))
            == 4 * int(retained_timestamps)
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _audit_shard(
    root: Path,
    manifest: dict[str, Any],
    shard: dict[str, Any],
    *,
    device: torch.device,
    pose_config: LagerTargetPoseConfig,
    pose_contract: dict[str, Any],
) -> dict[str, Any]:
    shard_id = int(shard["shard_id"])
    reader = IndexedTarReader(shard_path(root, "da3", shard_id))
    target_K = canonical_intrinsics(
        (1,),
        focal_px=float(
            manifest["teacher_processing"]["lagernvs"]["canonical_focal_px"]
        ),
        device=device,
    )
    scene_center = torch.tensor(
        manifest["teacher_processing"]["lagernvs"]["configured_scene_center"],
        device=device,
    )
    statistics = _empty_statistics()
    started = time.time()
    try:
        episodes = manifest["episodes"][
            int(shard["episode_start"]) : int(shard["episode_stop"])
        ]
        for entry in episodes:
            K = torch.tensor(
                [[camera["intrinsics_rlds"] for camera in entry["exterior_cameras"]]],
                dtype=torch.float32,
                device=device,
            )
            c2w = torch.tensor(
                [[camera["c2w"] for camera in entry["exterior_cameras"]]],
                dtype=torch.float32,
                device=device,
            )
            for retained_index in range(int(entry["retained_count"])):
                global_index = int(entry["global_retained_start"]) + retained_index
                raw_timestep = retained_index * 3
                key = sample_key(global_index)
                try:
                    encoded = decode_numeric_array(reader.read(key, "depth"))
                    depth_m, valid = depth_u16_to_meters(encoded)
                    poses = sample_safe_target_poses(
                        c2w,
                        K,
                        target_K,
                        torch.from_numpy(depth_m).to(device)[None, :, None],
                        torch.from_numpy(valid).to(device)[None, :, None],
                        scene_center,
                        pose_config,
                        seed=deterministic_item_seed(
                            str(manifest["schema_signature"]),
                            str(entry["episode_id"]),
                            raw_timestep,
                        ),
                    )
                except Exception as error:
                    raise RuntimeError(
                        "Pose audit failed for "
                        f"shard={shard_id}, episode={entry['episode_id']}, "
                        f"key={key}, retained_index={retained_index}, "
                        f"raw_timestep={raw_timestep}: {error}"
                    ) from error

                coverage = poses["source_coverage"][0]
                clearance = poses["geometry_clearance_distance"][0]
                baseline = poses["baseline"][0].clamp_min(1.0e-8)
                translation_fraction = (
                    poses["translation_perturbation_magnitude"][0] / baseline
                )
                rotation = poses["rotation_perturbation_degrees"][0]
                exceptional = poses["safety_translation_limit_escalated"][0]
                relaxed = poses["safety_thresholds_relaxed"][0]
                coverage_threshold = poses["minimum_source_coverage_threshold"][0]
                clearance_threshold = poses["minimum_geometry_distance_threshold"][0]
                source_clearance_reference = poses["source_clearance_reference"][0]
                alpha = poses["alpha"][0]
                expected_relaxed_clearance = (
                    source_clearance_reference
                    * pose_config.source_clearance_reference_fraction
                ).clamp(
                    min=pose_config.minimum_source_calibrated_clearance_m,
                    max=pose_config.exceptional_minimum_geometry_distance_m,
                )
                if not (
                    torch.isfinite(poses["target_c2w"]).all()
                    and torch.all(coverage >= coverage_threshold - 1e-6)
                    and torch.all(clearance >= clearance_threshold - 1e-6)
                    and torch.all(
                        coverage_threshold
                        >= pose_config.exceptional_min_source_coverage - 1e-6
                    )
                    and torch.all(
                        clearance_threshold
                        >= pose_config.minimum_source_calibrated_clearance_m - 1e-6
                    )
                    and torch.allclose(
                        clearance_threshold[relaxed],
                        expected_relaxed_clearance[relaxed],
                        atol=1e-6,
                        rtol=0.0,
                    )
                    and torch.all(
                        coverage_threshold[~relaxed]
                        >= pose_config.min_source_coverage - 1e-6
                    )
                    and torch.all(
                        clearance_threshold[~relaxed]
                        >= pose_config.minimum_geometry_distance_m - 1e-6
                    )
                    and torch.all(
                        translation_fraction
                        <= pose_config.exceptional_translation_max_baseline_fraction
                        + 1e-6
                    )
                    and torch.all(
                        translation_fraction[~exceptional]
                        <= pose_config.translation_max_baseline_fraction + 1e-6
                    )
                    and torch.all(rotation <= pose_config.rotation_max_degrees + 1e-6)
                    and torch.all((alpha[:2] >= 0.15) & (alpha[:2] <= 0.35))
                    and torch.all((alpha[2:] >= 0.65) & (alpha[2:] <= 0.85))
                    and torch.allclose(alpha[2], 1.0 - alpha[1])
                    and torch.allclose(alpha[3], 1.0 - alpha[0])
                ):
                    raise RuntimeError(
                        f"Pose bounds or safety checks failed for key={key}."
                    )
                statistics["retained_timestamps"] = (
                    int(statistics["retained_timestamps"]) + 1
                )
                statistics["targets"] = int(statistics["targets"]) + 4
                statistics["fallback_targets"] = int(
                    statistics["fallback_targets"]
                ) + int(poses["fallback_used"][0].sum())
                statistics["exceptional_translation_targets"] = int(
                    statistics["exceptional_translation_targets"]
                ) + int(exceptional.sum())
                statistics["exceptional_safety_threshold_targets"] = int(
                    statistics["exceptional_safety_threshold_targets"]
                ) + int(relaxed.sum())
                statistics["minimum_coverage"] = min(
                    float(statistics["minimum_coverage"]), float(coverage.min())
                )
                statistics["minimum_clearance_m"] = min(
                    float(statistics["minimum_clearance_m"]), float(clearance.min())
                )
                statistics["maximum_translation_baseline_fraction"] = max(
                    float(statistics["maximum_translation_baseline_fraction"]),
                    float(translation_fraction.max()),
                )
                statistics["maximum_rotation_degrees"] = max(
                    float(statistics["maximum_rotation_degrees"]),
                    float(rotation.max()),
                )
    finally:
        reader.close()
    return {
        "pose_audit_schema_version": POSE_AUDIT_SCHEMA_VERSION,
        "schema_signature": manifest["schema_signature"],
        "pose_contract_signature": pose_contract["signature"],
        "pose_sampler_contract": pose_contract,
        "status": "pass",
        "shard_id": shard_id,
        "statistics": statistics,
        "elapsed_seconds": time.time() - started,
    }


def main() -> None:
    args = parse_args()
    if not 0 <= int(args.worker_id) < int(args.worker_count):
        raise ValueError("worker-id must lie in [0, worker-count).")
    root = Path(args.root).expanduser().resolve()
    manifest = load_stage0_manifest(root)
    validate_derived_root(root, manifest["source_root"])
    schema_signature = str(manifest["schema_signature"])
    device = torch.device(args.device)
    pose_config = LagerTargetPoseConfig()
    pose_contract = pose_sampler_contract(pose_config)
    report_root = root / "reports" / "lagernvs_pose_safety"
    report_root.mkdir(parents=True, exist_ok=True)
    if int(args.worker_id) == 0:
        write_json_atomic(
            root / "metadata" / "lagernvs-pose-safety-contract.json",
            {
                **pose_contract,
                "dataset_schema_signature": schema_signature,
                "lager_record_contract": lager_record_contract(),
                "maximum_manifest_theoretical_metadata_adjustment_bytes": (
                    int(manifest["counts"]["retained_timesteps"])
                    * int(
                        lager_record_contract()[
                            "additional_uncompressed_bytes_per_timestamp_vs_v2"
                        ]
                    )
                ),
                "manifest_contract_note": (
                    "The signed manifest retains the pilot-approved ordinary limits; "
                    "this sidecar records bounded exceptional tiers used only after "
                    "ordinary pose candidates are exhausted."
                ),
            },
        )
    shards = [
        shard
        for shard in manifest["shards"]
        if int(shard["shard_id"]) % int(args.worker_count) == int(args.worker_id)
    ]
    if args.max_shards is not None:
        shards = shards[: int(args.max_shards)]
    aggregate = _empty_statistics()
    completed = skipped = 0
    started = time.time()
    for shard in shards:
        shard_id = int(shard["shard_id"])
        if args.skip_completed_lager and shard_is_complete(
            shard_path(root, "lagernvs", shard_id),
            schema_signature=schema_signature,
            verify_checksums=True,
        ):
            skipped += 1
            continue
        report_path = report_root / f"shard-{shard_id:05d}.json"
        if _report_is_complete(
            report_path,
            schema_signature,
            str(pose_contract["signature"]),
            shard_id=shard_id,
            retained_timestamps=int(shard["retained_count"]),
        ):
            report = json.loads(report_path.read_text(encoding="utf-8"))
        else:
            try:
                report = _audit_shard(
                    root,
                    manifest,
                    shard,
                    device=device,
                    pose_config=pose_config,
                    pose_contract=pose_contract,
                )
            except Exception as error:
                failure_path = report_root / f"shard-{shard_id:05d}.failure.json"
                write_json_atomic(
                    failure_path,
                    {
                        "pose_audit_schema_version": POSE_AUDIT_SCHEMA_VERSION,
                        "schema_signature": schema_signature,
                        "pose_contract_signature": pose_contract["signature"],
                        "pose_sampler_contract": pose_contract,
                        "status": "fail",
                        "shard_id": shard_id,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "time_unix": time.time(),
                    },
                )
                raise
            write_json_atomic(report_path, report)
        _merge_statistics(aggregate, report["statistics"])
        completed += 1
        print(
            json.dumps(
                {
                    "worker_id": int(args.worker_id),
                    "completed_shards": completed,
                    "assigned_shards": len(shards),
                    "shard_id": shard_id,
                    "retained_timestamps": aggregate["retained_timestamps"],
                    "exceptional_translation_targets": aggregate[
                        "exceptional_translation_targets"
                    ],
                    "exceptional_safety_threshold_targets": aggregate[
                        "exceptional_safety_threshold_targets"
                    ],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    elapsed = time.time() - started
    summary = {
        "pose_audit_schema_version": POSE_AUDIT_SCHEMA_VERSION,
        "schema_signature": schema_signature,
        "pose_contract_signature": pose_contract["signature"],
        "pose_sampler_contract": pose_contract,
        "status": "pass",
        "worker_id": int(args.worker_id),
        "worker_count": int(args.worker_count),
        "completed_shards": completed,
        "skipped_completed_lager_shards": skipped,
        "statistics": aggregate,
        "elapsed_seconds": elapsed,
        "retained_timestamps_per_second": (
            float(aggregate["retained_timestamps"]) / elapsed if elapsed > 0.0 else 0.0
        ),
    }
    write_json_atomic(report_root / f"worker-{int(args.worker_id):02d}.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
