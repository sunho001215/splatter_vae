from __future__ import annotations

import argparse
import json
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw

from dataset.droid.cache import HDF5CacheReader
from dataset.droid.dataset import DROIDLogicalDataset
from dataset.droid.safety import validate_derived_root
from dataset.droid.sampling import sample_uniform_crop_size
from preprocessing.see3d.visualization import (
    colorize_scalar,
    save_see3d_validation_grid,
)
from scripts.train_droid import _dataset_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize real DROID WAFT-centered variable-FOV crops."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--flow-cache", default=None)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--maximum-candidates", type=int, default=128)
    return parser.parse_args()


def _rgb(value) -> np.ndarray:
    array = value.detach().cpu().movedim(-3, -1).numpy()
    if array.dtype != np.uint8:
        array = (np.clip(array, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    return array


def _with_rectangle(image: np.ndarray, x0: int, y0: int, size: int) -> np.ndarray:
    canvas = Image.fromarray(image)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((x0, y0, x0 + size - 1, y0 + size - 1), outline=(255, 32, 32), width=4)
    return np.asarray(canvas)


def _category(size: int) -> str:
    if size <= 226:
        return "small"
    if size <= 273:
        return "medium"
    return "large"


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    values = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    droid_root = Path(values["dataset"]["droid_root"]).expanduser().resolve()
    output_root = validate_derived_root(args.output_root, droid_root)
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_config = _dataset_config(values, args.split)
    dataset_config = replace(
        dataset_config,
        calibration_manifest=str(
            Path(args.manifest or dataset_config.calibration_manifest)
            .expanduser()
            .resolve()
        ),
        require_depth_cache=False,
        require_flow_cache=True,
        include_crop_debug=True,
    )
    flow_index = Path(
        args.flow_cache or values["dataset"]["waft_cache_index"]
    ).expanduser().resolve()
    dataset = DROIDLogicalDataset(
        dataset_config,
        flow_cache=HDF5CacheReader(flow_index),
    )
    selected: dict[str, dict] = {}
    failures: list[str] = []
    crop_records: list[dict[str, float | int | bool]] = []
    for index in range(min(len(dataset), max(0, int(args.maximum_candidates)))):
        try:
            item = dataset[index]
        except (KeyError, IndexError) as exc:
            failures.append(f"index={index}: {type(exc).__name__}: {exc}")
            continue
        size = int(item["crop_metadata"]["crop_size"][0])
        for camera_index in range(2):
            crop_records.append(
                {
                    key: value[camera_index].item()
                    for key, value in item["crop_metadata"].items()
                }
            )
        category = _category(size)
        if category in selected:
            continue
        debug = item["crop_debug"]
        metadata = {
            key: value[0].item() for key, value in item["crop_metadata"].items()
        }
        x0 = int(metadata["crop_x0"])
        y0 = int(metadata["crop_y0"])
        padded_history = debug["padded_rgb"][0]
        current_padded = _rgb(padded_history[-1])
        cropped = current_padded[y0 : y0 + size, x0 : x0 + size]
        motion = debug["aggregate_motion"][0].numpy()
        panels: list[tuple[str, np.ndarray]] = [
            ("original current 320x180", _rgb(debug["original_rgb"][0, -1])),
            ("aggregate WAFT magnitude", colorize_scalar(motion, minimum=0.0)),
        ]
        for time_index in range(3):
            panels.append(
                (
                    f"padded history {time_index}; same crop",
                    _with_rectangle(
                        _rgb(padded_history[time_index]), x0, y0, size
                    ),
                )
            )
        panels.extend(
            [
                (f"selected {size}x{size} before resize", cropped),
                ("final 224x224", _rgb(item["target_rgb"][-1, 0])),
            ]
        )
        image_path = output_root / f"{category}-crop.jpg"
        save_see3d_validation_grid(image_path, panels, columns=4)
        selected[category] = {
            "dataset_index": index,
            "episode_id": item["episode_id"],
            "history_indices": item["history_indices"].tolist(),
            "camera_serial": item["camera_serials"][0],
            "metadata": metadata,
            "visualization": str(image_path),
        }
    sizes = np.asarray([record["crop_size"] for record in crop_records[::2]])
    all_centers_x = np.asarray([record["crop_center_x"] for record in crop_records])
    all_centers_y = np.asarray([record["crop_center_y"] for record in crop_records])
    audit_rng = random.Random(int(values["dataset"].get("seed", 42)))
    audit_draws = np.asarray(
        [
            sample_uniform_crop_size(dataset_config.motion_crop, audit_rng)
            for _ in range(28_200)
        ],
        dtype=np.int64,
    )
    audit_counts = np.bincount(
        audit_draws - dataset_config.motion_crop.min_size,
        minlength=(
            dataset_config.motion_crop.max_size
            - dataset_config.motion_crop.min_size
            + 1
        ),
    )
    expected = len(audit_draws) / len(audit_counts)
    aggregate_statistics = {
        "logical_real_samples": int(len(sizes)),
        "crop_size_mean": float(sizes.mean()) if len(sizes) else None,
        "crop_size_min": int(sizes.min()) if len(sizes) else None,
        "crop_size_max": int(sizes.max()) if len(sizes) else None,
        "crop_size_histogram": {
            str(size): int((sizes == size).sum())
            for size in sorted(set(sizes.tolist()))
        },
        "crop_center_x_mean": float(all_centers_x.mean()) if len(all_centers_x) else None,
        "crop_center_y_mean": float(all_centers_y.mean()) if len(all_centers_y) else None,
        "real_pixel_fraction_mean": (
            float(np.mean([record["real_pixel_fraction"] for record in crop_records]))
            if crop_records
            else None
        ),
        "flow_peak_mean": (
            float(np.mean([record["flow_peak_value"] for record in crop_records]))
            if crop_records
            else None
        ),
        "selected_crop_flow_mean": (
            float(
                np.mean(
                    [record["selected_crop_flow_mean"] for record in crop_records]
                )
            )
            if crop_records
            else None
        ),
        "low_motion_fallback_fraction": (
            float(
                np.mean(
                    [record["low_motion_fallback_used"] for record in crop_records]
                )
            )
            if crop_records
            else None
        ),
        "configured_uniform_sampler_audit": {
            "draws": int(len(audit_draws)),
            "interval": [
                dataset_config.motion_crop.min_size,
                dataset_config.motion_crop.max_size,
            ],
            "chi_square": float((((audit_counts - expected) ** 2) / expected).sum()),
            "maximum_to_minimum_bin_count_ratio": float(
                audit_counts.max() / audit_counts.min()
            ),
            "minimum_bin_count": int(audit_counts.min()),
            "maximum_bin_count": int(audit_counts.max()),
        },
    }
    summary = {
        "source_droid_root": str(droid_root),
        "flow_cache": str(flow_index),
        "selected": selected,
        "statistics": aggregate_statistics,
        "cache_miss_examples": failures[:10],
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
