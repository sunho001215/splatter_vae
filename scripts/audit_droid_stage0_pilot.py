#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

from dataset.droid.dataset import DROIDDatasetConfig, DROIDPreprocessedDataset
from dataset.droid.integrity import IntegrityConfig, verify_stage0_dataset
from dataset.droid.preprocessed_manifest import load_stage0_manifest
from dataset.droid.shards import write_json_atomic
from preprocessing.workspace import backproject_z_depth_to_world


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantitative and visual quality audit for the real-DROID pilot."
    )
    parser.add_argument(
        "--root", default="/home/ws/data/droid_stage0_preprocessed/pilot"
    )
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--output", default=None)
    parser.add_argument("--visualization-root", default=None)
    parser.add_argument(
        "--previous-workspace-stats",
        default="outputs/teacher_validation/xlens/workspace_stats_76x3.json",
    )
    return parser.parse_args()


def _rgb(value: torch.Tensor | np.ndarray) -> np.ndarray:
    tensor = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    if tensor.shape[0] == 3:
        tensor = np.moveaxis(tensor, 0, -1)
    if np.issubdtype(tensor.dtype, np.floating):
        tensor = np.rint(np.clip(tensor, 0.0, 1.0) * 255.0)
    return tensor.astype(np.uint8)


def _scalar_color(value: np.ndarray, minimum: float, maximum: float) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    normalized = np.clip(
        (array - float(minimum)) / max(float(maximum - minimum), 1.0e-8), 0, 1
    )
    red = np.clip(1.5 - np.abs(4.0 * normalized - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * normalized - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * normalized - 1.0), 0.0, 1.0)
    output = np.stack((red, green, blue), axis=-1)
    output[~np.isfinite(array)] = 0
    return np.rint(output * 255).astype(np.uint8)


def _flow_rgb(flow: np.ndarray, maximum: float = 64.0) -> np.ndarray:
    x, y = np.asarray(flow, dtype=np.float32)
    hue = (np.arctan2(y, x) + np.pi) / (2.0 * np.pi)
    magnitude = np.clip(np.sqrt(x * x + y * y) / maximum, 0.0, 1.0)
    sector = hue * 6.0
    index = np.floor(sector).astype(np.int32) % 6
    fraction = sector - np.floor(sector)
    zero = np.zeros_like(magnitude)
    q = magnitude * (1.0 - fraction)
    t = magnitude * fraction
    candidates = (
        (magnitude, t, zero),
        (q, magnitude, zero),
        (zero, magnitude, t),
        (zero, q, magnitude),
        (t, zero, magnitude),
        (magnitude, zero, q),
    )
    rgb = np.zeros((*magnitude.shape, 3), dtype=np.float32)
    for sector_index, channels in enumerate(candidates):
        mask = index == sector_index
        for channel, values in enumerate(channels):
            rgb[..., channel][mask] = values[mask]
    return np.rint(rgb * 255).astype(np.uint8)


def _grid(path: Path, panels: list[tuple[str, np.ndarray]], columns: int) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    images = [(title, Image.fromarray(np.asarray(image, dtype=np.uint8))) for title, image in panels]
    width = max(image.width for _title, image in images)
    height = max(image.height for _title, image in images) + 22
    rows = (len(images) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * width, rows * height), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    for index, (title, image) in enumerate(images):
        row, column = divmod(index, columns)
        canvas.paste(image, (column * width, row * height + 22))
        draw.text((column * width + 3, row * height + 4), title, fill="white")
    canvas.save(path, quality=95, subsampling=0)
    return str(path)


def _forward_splat_rgb(
    source: np.ndarray,
    flow: np.ndarray,
    validity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    image = np.asarray(source, dtype=np.float32)
    vectors = np.asarray(flow, dtype=np.float32)
    valid = np.squeeze(np.asarray(validity, dtype=bool))
    height, width = image.shape[:2]
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    destination_x = np.rint(x + vectors[0]).astype(np.int64)
    destination_y = np.rint(y + vectors[1]).astype(np.int64)
    usable = (
        valid
        & np.isfinite(vectors).all(axis=0)
        & (destination_x >= 0)
        & (destination_x < width)
        & (destination_y >= 0)
        & (destination_y < height)
    )
    linear = destination_y[usable] * width + destination_x[usable]
    accumulated = np.zeros((height * width, 3), dtype=np.float64)
    weights = np.zeros(height * width, dtype=np.float64)
    np.add.at(accumulated, linear, image[usable])
    np.add.at(weights, linear, 1.0)
    occupied = weights > 0
    output = np.zeros_like(accumulated, dtype=np.float32)
    output[occupied] = accumulated[occupied] / weights[occupied, None]
    return output.reshape(height, width, 3), occupied.reshape(height, width)


def _point_cloud_consistency(
    depth_a: np.ndarray,
    depth_b: np.ndarray,
    valid_a: np.ndarray,
    valid_b: np.ndarray,
    K: np.ndarray,
    c2w: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, float]:
    points_a = backproject_z_depth_to_world(depth_a, K[0], c2w[0])[
        np.asarray(valid_a, dtype=bool)
    ]
    points_b = backproject_z_depth_to_world(depth_b, K[1], c2w[1])[
        np.asarray(valid_b, dtype=bool)
    ]
    if min(len(points_a), len(points_b)) < 100:
        raise ValueError("DA3 pilot has too few valid cross-view points.")
    if len(points_a) > 5000:
        points_a = points_a[rng.choice(len(points_a), 5000, replace=False)]
    if len(points_b) > 5000:
        points_b = points_b[rng.choice(len(points_b), 5000, replace=False)]
    distance_ab = cKDTree(points_b).query(points_a, workers=-1)[0]
    distance_ba = cKDTree(points_a).query(points_b, workers=-1)[0]
    distances = np.concatenate((distance_ab, distance_ba))
    return {
        "median_m": float(np.median(distances)),
        "p90_m": float(np.percentile(distances, 90)),
        "p95_m": float(np.percentile(distances, 95)),
    }


def _selected_items(
    root: Path, count: int
) -> list[tuple[str, int, dict[str, Any]]]:
    output = []
    remaining = int(count)
    for split in ("train", "validation"):
        try:
            dataset = DROIDPreprocessedDataset(
                DROIDDatasetConfig(preprocessed_root=str(root), split=split)
            )
        except ValueError:
            continue
        split_count = (
            remaining
            if split == "validation"
            else max(1, round(int(count) * 0.75))
        )
        selected = np.linspace(
            0, len(dataset) - 1, min(split_count, len(dataset)), dtype=np.int64
        )
        for index in selected:
            output.append((split, int(index), dataset[int(index)]))
        remaining = max(0, int(count) - len(output))
        dataset.close()
    return output[: int(count)]


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    manifest = load_stage0_manifest(root)
    integrity = verify_stage0_dataset(
        IntegrityConfig(
            root=str(root),
            random_samples=max(100, int(args.samples) * 4),
            full_payload_scan=True,
            decode_all_jpegs=True,
            loader_windows=min(16, int(args.samples)),
        )
    )
    items = _selected_items(root, int(args.samples))
    if not items:
        raise ValueError("Pilot dataset has no loadable training windows.")
    rng = np.random.default_rng(20260828)
    depth_valid = []
    depth_positive = []
    cross_view = []
    baselines = []
    flow_rows = []
    alpha_values = []
    support_values = []
    visualization_root = Path(
        args.visualization_root or root / "reports" / "quality_visualizations"
    ).expanduser().resolve()
    visualizations = []

    for sample_number, (split, index, item) in enumerate(items):
        native_depth = item["native_da3_depth"].numpy()[:, :, 0]
        native_depth_validity = item["native_da3_validity"].numpy()[:, :, 0]
        K = item["native_K"].numpy()
        c2w = item["native_c2w"].numpy()
        baselines.append(float(np.linalg.norm(c2w[0, :3, 3] - c2w[1, :3, 3])))
        for time_index in range(3):
            depth_valid.append(float(native_depth_validity[time_index].mean()))
            values = native_depth[time_index][native_depth_validity[time_index]]
            depth_positive.append(values)
            cross_view.append(
                _point_cloud_consistency(
                    native_depth[time_index, 0],
                    native_depth[time_index, 1],
                    native_depth_validity[time_index, 0],
                    native_depth_validity[time_index, 1],
                    K,
                    c2w,
                    rng,
                )
            )

        raw = item["raw_histories"].numpy()
        native_flow = item["native_megaflow_flow"].numpy()
        native_flow_valid = item["native_megaflow_validity"].numpy()
        for pair in range(2):
            for camera in range(2):
                source = np.moveaxis(raw[camera, pair * 1], 0, -1)
                target = np.moveaxis(raw[camera, pair + 1], 0, -1)
                flow = native_flow[pair, camera]
                valid = native_flow_valid[pair, camera, 0]
                warped, occupied = _forward_splat_rgb(source, flow, valid)
                usable = occupied
                warp_l1 = float(
                    np.abs(warped[usable] - target[usable].astype(np.float32)).mean()
                    / 255.0
                )
                zero_l1 = float(
                    np.abs(
                        source[usable].astype(np.float32)
                        - target[usable].astype(np.float32)
                    ).mean()
                    / 255.0
                )
                photo_static = (
                    np.abs(source.astype(np.float32) - target.astype(np.float32)).mean(
                        axis=-1
                    )
                    <= 5.0
                ) & valid
                magnitude = np.linalg.norm(flow, axis=0)
                flow_rows.append(
                    {
                        "warp_l1": warp_l1,
                        "zero_l1": zero_l1,
                        "warp_to_zero_ratio": warp_l1 / max(zero_l1, 1.0e-8),
                        "valid_fraction": float(valid.mean()),
                        "static_flow_median_px": (
                            float(np.median(magnitude[photo_static]))
                            if photo_static.any()
                            else None
                        ),
                        "static_flow_p95_px": (
                            float(np.percentile(magnitude[photo_static], 95))
                            if photo_static.any()
                            else None
                        ),
                        "large_displacement_fraction_gt32px": float(
                            (magnitude[valid] > 32.0).mean()
                        ),
                        "maximum_displacement_px": float(magnitude[valid].max()),
                        "nonfinite_fraction": float((~np.isfinite(flow)).mean()),
                    }
                )

        alpha = item["novel_pose_metadata"]["alpha"].numpy()
        support = item["novel_support_mask"].float().numpy()
        alpha_values.append(alpha)
        support_values.append(support.mean(axis=(2, 3, 4)))

        if sample_number < 8:
            depth_panels = []
            for time_index in range(3):
                for camera in range(2):
                    depth_panels.extend(
                        (
                            (
                                f"t{time_index} cam{camera} RGB",
                                np.moveaxis(raw[camera, time_index], 0, -1),
                            ),
                            (
                                f"t{time_index} cam{camera} cached DA3",
                                _scalar_color(
                                    np.where(
                                        native_depth_validity[time_index, camera],
                                        native_depth[time_index, camera],
                                        np.nan,
                                    ),
                                    0.25,
                                    5.0,
                                ),
                            ),
                        )
                    )
            visualizations.append(
                _grid(
                    visualization_root
                    / f"{sample_number:02d}-{split}-{index}-da3.jpg",
                    depth_panels,
                    columns=4,
                )
            )
            flow_panels = []
            for pair, name in enumerate(("t0->t1", "t1->t2")):
                for camera in range(2):
                    source = np.moveaxis(raw[camera, pair], 0, -1)
                    target = np.moveaxis(raw[camera, pair + 1], 0, -1)
                    flow = native_flow[pair, camera]
                    warped, occupied = _forward_splat_rgb(
                        source, flow, native_flow_valid[pair, camera, 0]
                    )
                    error = np.zeros_like(target)
                    error[occupied] = np.clip(
                        np.abs(warped[occupied] - target[occupied]) * 4, 0, 255
                    )
                    flow_panels.extend(
                        (
                            (f"cam{camera} {name} source", source),
                            (f"cam{camera} {name} target", target),
                            (f"cam{camera} {name} MegaFlow", _flow_rgb(flow)),
                            (
                                f"cam{camera} {name} warped source",
                                np.clip(warped, 0, 255).astype(np.uint8),
                            ),
                            (f"cam{camera} {name} warp error x4", error),
                        )
                    )
            visualizations.append(
                _grid(
                    visualization_root
                    / f"{sample_number:02d}-{split}-{index}-megaflow.jpg",
                    flow_panels,
                    columns=5,
                )
            )
            for time_index in range(3):
                lager_panels = [
                    (
                        f"t{time_index} target {view} alpha={alpha[time_index, view]:.3f}",
                        _rgb(item["novel_rgb"][time_index, view]),
                    )
                    for view in range(4)
                ]
                visualizations.append(
                    _grid(
                        visualization_root
                        / f"{sample_number:02d}-{split}-{index}-lager-t{time_index}.jpg",
                        lager_panels,
                        columns=4,
                    )
                )

    depth_values = np.concatenate(depth_positive)
    cross_median = np.asarray([row["median_m"] for row in cross_view])
    flow_with_static = [
        row for row in flow_rows if row["static_flow_median_px"] is not None
    ]
    alpha = np.concatenate(alpha_values, axis=0)
    support = np.concatenate(support_values, axis=0)
    previous = None
    previous_path = Path(args.previous_workspace_stats).expanduser().resolve()
    if previous_path.is_file():
        previous_payload = json.loads(previous_path.read_text(encoding="utf-8"))
        previous = {
            "path": str(previous_path),
            "depth_percentiles": previous_payload.get("depth_percentiles"),
            "coordinate_percentiles": previous_payload.get("coordinate_percentiles"),
            "proposed_parameters": previous_payload.get("proposed_parameters"),
        }
    da3_gate = (
        float(np.mean(depth_valid)) >= 0.50
        and 0.10 <= float(np.median(depth_values)) <= 10.0
        and float(np.median(cross_median)) <= 0.30
        and min(baselines) >= 0.05
    )
    megaflow_gate = (
        max(row["nonfinite_fraction"] for row in flow_rows) == 0.0
        # Out-of-bounds destinations are expected for genuine large camera or
        # object motion; they are distinct from non-finite/sentinel failures.
        and min(row["valid_fraction"] for row in flow_rows) >= 0.50
        and float(np.median([row["warp_to_zero_ratio"] for row in flow_rows])) < 1.05
        and (
            not flow_with_static
            or float(
                np.median(
                    [row["static_flow_median_px"] for row in flow_with_static]
                )
            )
            <= 2.0
        )
    )
    lager_gate = (
        alpha.shape[1] == 4
        and np.all(alpha[:, :2] <= 0.35 + 1.0e-6)
        and np.all(alpha[:, 2:] >= 0.65 - 1.0e-6)
        and np.all(np.abs(alpha - 0.5) >= 0.15 - 1.0e-6)
        and float(support.mean()) > 0.0
    )
    report = {
        "schema_version": 1,
        "dataset_schema_signature": manifest["schema_signature"],
        "pipeline_signature": manifest["pipeline_signature"],
        "counts": manifest["counts"],
        "samples": len(items),
        "integrity": integrity,
        "da3": {
            "valid_fraction_mean": float(np.mean(depth_valid)),
            "valid_fraction_min": float(np.min(depth_valid)),
            "positive_depth_percentiles_m": {
                str(value): float(np.percentile(depth_values, value))
                for value in (1, 5, 50, 95, 99)
            },
            "cross_view_nn_median_m": float(np.median(cross_median)),
            "cross_view_nn_p95_m": float(np.percentile(cross_median, 95)),
            "baseline_range_m": [min(baselines), max(baselines)],
            "previous_teacher_statistics": previous,
        },
        "megaflow": {
            "fields_audited": len(flow_rows),
            "warp_l1_mean": float(np.mean([row["warp_l1"] for row in flow_rows])),
            "zero_l1_mean": float(np.mean([row["zero_l1"] for row in flow_rows])),
            "warp_to_zero_ratio_median": float(
                np.median([row["warp_to_zero_ratio"] for row in flow_rows])
            ),
            "valid_fraction_min": min(row["valid_fraction"] for row in flow_rows),
            "static_flow_median_px": (
                float(
                    np.median(
                        [row["static_flow_median_px"] for row in flow_with_static]
                    )
                )
                if flow_with_static
                else None
            ),
            "maximum_displacement_px": max(
                row["maximum_displacement_px"] for row in flow_rows
            ),
            "rows": flow_rows,
        },
        "lagernvs": {
            "targets_per_timestep": int(alpha.shape[1]),
            "alpha_min": float(alpha.min()),
            "alpha_max": float(alpha.max()),
            "minimum_distance_from_midpoint": float(np.abs(alpha - 0.5).min()),
            "support_fraction_mean": float(support.mean()),
            "support_fraction_min": float(support.min()),
        },
        "visualizations": visualizations,
        "gates": {
            "data_selection": (
                manifest["eligibility"]
                == "canonical_stage0_calibration_valid_only"
                and int(manifest["retained_raw_stride"]) == 3
                and int(manifest["temporal_gap_raw"]) == 6
                and int(manifest["temporal_window"]) == 3
            ),
            "da3_metric_geometry": bool(da3_gate),
            "megaflow_quality": bool(megaflow_gate),
            "lagernvs_four_targets": bool(lager_gate),
        },
    }
    output = Path(
        args.output or root / "reports" / "quality_audit.json"
    ).expanduser().resolve()
    write_json_atomic(output, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
