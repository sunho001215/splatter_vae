from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

import yaml

from dataset.droid.calibration import (
    OFFICIAL_CALIBRATION_REPOSITORY,
    OFFICIAL_CALIBRATION_REVISION,
    CalibrationThresholds,
    download_official_calibration,
    prepare_calibration_manifest,
)
from dataset.droid.rlds import (
    load_rlds_metadata_index,
    scan_rlds_episode_metadata,
    write_rlds_metadata_index,
)
from dataset.droid.safety import prepare_derived_layout, validate_derived_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download official DROID calibration and build a Stage-0 manifest."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--overwrite-downloads", action="store_true")
    parser.add_argument("--metadata-index", default=None)
    parser.add_argument("--manifest-output", default=None)
    parser.add_argument("--split-output", default=None)
    parser.add_argument(
        "--maximum-episodes",
        type=int,
        default=None,
        help="Bound a representative sanity manifest; omit for the full release.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    dataset_cfg = config["dataset"]
    droid_root = Path(dataset_cfg["droid_root"]).expanduser().resolve()
    derived_root = validate_derived_root(dataset_cfg["derived_root"], droid_root)
    layout = prepare_derived_layout(derived_root, droid_root)
    calibration_dir = layout["calibration"] / "official_posthoc"
    download_official_calibration(
        calibration_dir,
        droid_root=droid_root,
        overwrite=args.overwrite_downloads,
    )
    if args.download_only:
        print(json.dumps({"calibration_dir": str(calibration_dir)}, indent=2))
        return
    if not droid_root.is_dir():
        raise FileNotFoundError(
            f"DROID source mount is unavailable at {droid_root}; downloads completed, "
            "but the RLDS manifest cannot be built."
        )
    metadata_path = (
        Path(args.metadata_index)
        if args.metadata_index
        else layout["manifests"]
        / (
            "rlds_episodes.jsonl"
            if args.maximum_episodes is None
            else f"rlds_episodes.max-{int(args.maximum_episodes)}.jsonl"
        )
    )
    if metadata_path.is_file():
        metadata = load_rlds_metadata_index(metadata_path)
    else:
        metadata = list(
            scan_rlds_episode_metadata(
                droid_root, maximum_episodes=args.maximum_episodes
            )
        )
        write_rlds_metadata_index(metadata, metadata_path, droid_root=droid_root)
    calibration_cfg = config.get("calibration", {})
    configured_release = (
        calibration_cfg.get("official_repository", OFFICIAL_CALIBRATION_REPOSITORY),
        calibration_cfg.get("official_revision", OFFICIAL_CALIBRATION_REVISION),
    )
    required_release = (
        OFFICIAL_CALIBRATION_REPOSITORY,
        OFFICIAL_CALIBRATION_REVISION,
    )
    if configured_release != required_release:
        raise ValueError(
            f"This pipeline requires the official post-hoc release {required_release}, "
            f"got {configured_release}."
        )
    thresholds_cfg = calibration_cfg.get("thresholds", {})
    allowed = {field.name for field in fields(CalibrationThresholds)}
    unknown = set(thresholds_cfg) - allowed
    if unknown:
        raise ValueError(f"Unknown calibration thresholds: {sorted(unknown)}")
    thresholds = CalibrationThresholds(**thresholds_cfg)
    manifest_path = Path(args.manifest_output or dataset_cfg["calibration_manifest"])
    split_output_path = args.split_output or dataset_cfg["split_manifest"]
    validate_derived_root(manifest_path.parent, droid_root)
    validate_derived_root(Path(split_output_path).parent, droid_root)
    result = prepare_calibration_manifest(
        metadata,
        calibration_dir,
        manifest_path,
        thresholds=thresholds,
        validation_fraction=float(dataset_cfg.get("validation_fraction", 0.02)),
        split_seed=int(dataset_cfg.get("split_seed", 42)),
        split_output_path=split_output_path,
        droid_root=droid_root,
    )
    result["rlds_metadata_limit"] = args.maximum_episodes
    result["rlds_metadata_index"] = str(metadata_path.resolve())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
