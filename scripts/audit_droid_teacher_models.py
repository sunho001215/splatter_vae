#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from dataset.droid.preprocessed_manifest import (
    DEFAULT_LAGER_SCENE_CENTER,
    load_stage0_manifest,
)
from dataset.droid.rlds import TFDSRLDSBackend
from dataset.droid.shards import write_json_atomic

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
AUTHORIZED_GPUS = (
    "GPU-76871c16-ff1d-f7b1-cdf5-fabbaf9df8ce",
    "GPU-d09f0338-71b9-d915-3c7f-e99754a3b639",
    "GPU-6b33eef2-80c5-547e-6a61-7ab81c14d63b",
)


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--pilot-root", default="/home/ws/data/droid_stage0_preprocessed/pilot"
    )
    parser.add_argument("--droid-root", default="/home/ws/data/droid")
    parser.add_argument(
        "--model-cache",
        default="/home/ws/data/droid_stage0_preprocessed/metadata/model_cache",
    )
    parser.add_argument(
        "--scene-center",
        nargs=3,
        type=float,
        default=DEFAULT_LAGER_SCENE_CENTER,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit pinned DA3, MegaFlow, and LagerNVS on real DROID data using "
            "their isolated environments."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    _common(run)
    run.add_argument("--gpu", choices=AUTHORIZED_GPUS, default=AUTHORIZED_GPUS[0])
    run.add_argument(
        "--python-training", default=str(REPOSITORY_ROOT / ".venv/bin/python")
    )
    run.add_argument(
        "--python-da3",
        default=str(REPOSITORY_ROOT / ".preprocessing-envs/da3/bin/python"),
    )
    run.add_argument(
        "--python-megaflow",
        default=str(REPOSITORY_ROOT / ".preprocessing-envs/megaflow/bin/python"),
    )
    run.add_argument(
        "--python-lagernvs",
        default=str(REPOSITORY_ROOT / ".preprocessing-envs/lagernvs/bin/python"),
    )

    worker = subparsers.add_parser("worker")
    _common(worker)
    worker.add_argument("--teacher", choices=("da3", "megaflow", "lagernvs"), required=True)

    combine = subparsers.add_parser("combine")
    _common(combine)
    combine.add_argument("--output", default=None)
    return parser.parse_args()


def _geometry(entry: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    K = np.asarray(
        [camera["intrinsics_rlds"] for camera in entry["exterior_cameras"]],
        dtype=np.float32,
    )
    c2w = np.asarray(
        [camera["c2w"] for camera in entry["exterior_cameras"]], dtype=np.float32
    )
    w2c = np.asarray(
        [camera["w2c"] for camera in entry["exterior_cameras"]], dtype=np.float32
    )
    if K.shape != (2, 3, 3) or c2w.shape != (2, 4, 4) or w2c.shape != (2, 4, 4):
        raise ValueError("Pilot entry does not contain two calibrated cameras.")
    if not np.isfinite(K).all() or not np.isfinite(c2w).all():
        raise ValueError("Pilot camera geometry contains non-finite values.")
    if not np.allclose(c2w @ w2c, np.eye(4), atol=2.0e-3):
        raise ValueError("Pilot c2w/w2c matrices are inconsistent.")
    return K, c2w, w2c


def _language(episode: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for name in (
        "language_instruction",
        "language_instruction_2",
        "language_instruction_3",
    ):
        if name not in episode:
            continue
        for raw in np.asarray(episode[name]).reshape(-1):
            if isinstance(raw, bytes):
                text = raw.decode("utf-8", errors="replace").strip()
            else:
                text = str(raw).strip()
            if text and text not in values:
                values.append(text)
    return values


def _retained_episode(
    backend: TFDSRLDSBackend, entry: dict[str, Any]
) -> tuple[np.ndarray, list[str]]:
    episode = dict(
        backend.get_episode(str(entry["rlds_split"]), int(entry["rlds_ordinal"]))
    )
    images = np.asarray(episode["images"], dtype=np.uint8)
    expected = (int(entry["num_steps"]), 2, 180, 320, 3)
    if images.shape != expected:
        raise ValueError(f"RLDS image shape {images.shape} != {expected}.")
    return np.ascontiguousarray(images[::3]), _language(episode)


def _selected_depth_locations(manifest: dict[str, Any]) -> list[tuple[int, int]]:
    ordered = sorted(
        enumerate(manifest["episodes"]),
        key=lambda value: (
            float(value[1].get("exterior_baseline_m", 0.0)),
            str(value[1]["episode_id"]),
        ),
    )
    positions = sorted({0, len(ordered) // 2, len(ordered) - 1})
    output = []
    for position in positions:
        entry_index, entry = ordered[position]
        retained = int(entry["retained_count"])
        output.append((entry_index, max(0, retained // 2)))
    return output


def _point_cloud(depth: np.ndarray, K: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    y, x = np.mgrid[0 : depth.shape[0] : 6, 0 : depth.shape[1] : 6]
    z = depth[::6, ::6]
    valid = np.isfinite(z) & (z > 0.05) & (z < 20.0)
    x_camera = (x + 0.5 - float(K[0, 2])) / float(K[0, 0]) * z
    y_camera = (y + 0.5 - float(K[1, 2])) / float(K[1, 1]) * z
    camera = np.stack((x_camera, y_camera, z), axis=-1)[valid]
    return camera @ c2w[:3, :3].T + c2w[:3, 3]


def _cross_view_distance(
    depth: np.ndarray, K: np.ndarray, c2w: np.ndarray
) -> dict[str, float]:
    from scipy.spatial import cKDTree

    first = _point_cloud(depth[0], K[0], c2w[0])
    second = _point_cloud(depth[1], K[1], c2w[1])
    if min(len(first), len(second)) < 32:
        raise ValueError("DA3 produced too few points for a cross-view audit.")
    first_to_second = cKDTree(second).query(first, workers=1)[0]
    second_to_first = cKDTree(first).query(second, workers=1)[0]
    distances = np.concatenate((first_to_second, second_to_first))
    return {
        "median_m": float(np.median(distances)),
        "p95_m": float(np.percentile(distances, 95.0)),
        "points_cam_a": len(first),
        "points_cam_b": len(second),
    }


def _depth_rgb(depth: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0.0)
    if not valid.any():
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    low, high = np.percentile(depth[valid], (2.0, 98.0))
    value = np.clip((depth - low) / max(float(high - low), 1.0e-6), 0.0, 1.0)
    rgb = np.stack((value, 1.0 - np.abs(value * 2.0 - 1.0), 1.0 - value), -1)
    rgb[~valid] = 0.0
    return (rgb * 255.0).round().astype(np.uint8)


def _flow_rgb(flow: np.ndarray, maximum: float = 64.0) -> np.ndarray:
    x, y = np.asarray(flow, dtype=np.float32)
    angle = (np.arctan2(y, x) + math.pi) / (2.0 * math.pi)
    magnitude = np.clip(np.hypot(x, y) / float(maximum), 0.0, 1.0)
    hsv = np.stack((angle, np.ones_like(angle), magnitude), axis=-1)
    # Compact HSV-to-RGB implementation for dependency-free audit panels.
    sector = np.floor(hsv[..., 0] * 6.0).astype(np.int32) % 6
    fraction = hsv[..., 0] * 6.0 - np.floor(hsv[..., 0] * 6.0)
    value = hsv[..., 2]
    q = value * (1.0 - fraction)
    t = value * fraction
    zero = np.zeros_like(value)
    choices = (
        (value, t, zero),
        (q, value, zero),
        (zero, value, t),
        (zero, q, value),
        (t, zero, value),
        (value, zero, q),
    )
    rgb = np.zeros((*value.shape, 3), dtype=np.float32)
    for index, channels in enumerate(choices):
        mask = sector == index
        for channel, values in enumerate(channels):
            rgb[..., channel][mask] = values[mask]
    return (rgb * 255.0).round().astype(np.uint8)


def _save_panels(path: Path, panels: list[tuple[str, np.ndarray]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    cell_width, cell_height, title_height = 320, 256, 24
    columns = min(4, len(panels))
    rows = math.ceil(len(panels) / columns)
    canvas = Image.new(
        "RGB", (columns * cell_width, rows * (cell_height + title_height)), (20, 20, 20)
    )
    draw = ImageDraw.Draw(canvas)
    for index, (title, array) in enumerate(panels):
        row, column = divmod(index, columns)
        image = Image.fromarray(np.asarray(array, dtype=np.uint8), mode="RGB")
        image.thumbnail((cell_width, cell_height), Image.Resampling.BILINEAR)
        x = column * cell_width + (cell_width - image.width) // 2
        y = row * (cell_height + title_height) + title_height
        canvas.paste(image, (x, y))
        draw.text(
            (column * cell_width + 4, row * (cell_height + title_height) + 5),
            title,
            fill="white",
        )
    temporary = path.with_suffix(path.suffix + ".partial")
    canvas.save(temporary, format="JPEG", quality=95, subsampling=0)
    os.replace(temporary, path)
    return str(path)


def _part_root(root: Path) -> Path:
    return root / "reports" / "teacher_audit_parts"


def _write_npz_atomic(path: Path, **values: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **values)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _audit_da3(args: argparse.Namespace, manifest: dict[str, Any]) -> dict[str, Any]:
    from preprocessing.da3 import DA3DROIDTeacher

    teacher = DA3DROIDTeacher(
        REPOSITORY_ROOT / "third_party" / "Depth-Anything-3",
        cache_dir=args.model_cache,
        device="cuda:0",
    )
    backend = TFDSRLDSBackend(args.droid_root, cache_size=1)
    rows = []
    panels: list[tuple[str, np.ndarray]] = []
    artifact: dict[str, Any] | None = None
    for entry_index, retained_index in _selected_depth_locations(manifest):
        entry = manifest["episodes"][entry_index]
        retained, language = _retained_episode(backend, entry)
        synchronized = retained[retained_index]
        K, c2w, w2c = _geometry(entry)
        torch.cuda.synchronize()
        started = time.perf_counter()
        output = teacher(synchronized, K, w2c)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        depth = np.asarray(output["metric_depth"], dtype=np.float32)
        valid = np.isfinite(depth) & (depth > 0.0)
        consistency = _cross_view_distance(depth, K, c2w)
        baseline_error = abs(
            float(output["output_baseline_m"]) - float(output["input_baseline_m"])
        ) / max(float(output["input_baseline_m"]), 1.0e-8)
        rows.append(
            {
                "episode_id": entry["episode_id"],
                "raw_timestep": retained_index * 3,
                "language_instructions": language,
                "seconds": elapsed,
                "shape": list(depth.shape),
                "processed_resolution": list(output["processed_resolution"]),
                "output_resolution": list(output["output_resolution"]),
                "is_metric": bool(output["is_metric"]),
                "positive_finite_fraction": float(valid.mean()),
                "positive_depth_percentiles_m": {
                    str(value): float(np.percentile(depth[valid], value))
                    for value in (1, 5, 50, 95, 99)
                },
                "input_baseline_m": float(output["input_baseline_m"]),
                "output_baseline_m": float(output["output_baseline_m"]),
                "baseline_relative_error": baseline_error,
                "extrinsic_max_abs_error": float(output["extrinsic_max_abs_error"]),
                "scale_factor": output["scale_factor"],
                "pose_alignment_scale": float(output["pose_alignment_scale"]),
                "cross_view": consistency,
                "intrinsics": K.tolist(),
                "c2w": c2w.tolist(),
                "w2c": w2c.tolist(),
            }
        )
        panels.extend(
            (
                (f"{entry_index} cam A RGB", synchronized[0]),
                (f"{entry_index} cam A DA3", _depth_rgb(depth[0])),
                (f"{entry_index} cam B RGB", synchronized[1]),
                (f"{entry_index} cam B DA3", _depth_rgb(depth[1])),
            )
        )
        if artifact is None:
            artifact = {
                "episode_index": np.asarray(entry_index, dtype=np.int64),
                "retained_index": np.asarray(retained_index, dtype=np.int64),
                "depth": depth,
            }
    assert artifact is not None
    artifact_path = _part_root(Path(args.pilot_root)) / "da3_lager_sample.npz"
    _write_npz_atomic(artifact_path, **artifact)
    visualization = _save_panels(
        Path(args.pilot_root) / "reports" / "teacher_visualizations" / "da3.jpg",
        panels,
    )
    passed = all(
        row["shape"] == [2, 180, 320]
        and row["is_metric"]
        and row["positive_finite_fraction"] >= 0.50
        and 0.10 <= row["positive_depth_percentiles_m"]["50"] <= 10.0
        and row["baseline_relative_error"] <= 1.0e-3
        and row["extrinsic_max_abs_error"] <= 5.0e-4
        and row["cross_view"]["median_m"] <= 0.30
        for row in rows
    )
    return {
        "teacher": "da3",
        "metadata": teacher.metadata(),
        "samples": rows,
        "mean_images_per_second": len(rows) * 2 / sum(row["seconds"] for row in rows),
        "artifact": str(artifact_path),
        "visualization": visualization,
        "passed": bool(passed),
    }


def _flow_quality(
    flow: np.ndarray, source: np.ndarray, target: np.ndarray
) -> list[dict[str, float]]:
    output = []
    for camera in range(2):
        value = torch.from_numpy(flow[camera]).float()
        src = torch.from_numpy(source[camera]).permute(2, 0, 1).float() / 255.0
        dst = torch.from_numpy(target[camera]).permute(2, 0, 1).float() / 255.0
        y, x = torch.meshgrid(torch.arange(180), torch.arange(320), indexing="ij")
        destination_x = x.float() + value[0]
        destination_y = y.float() + value[1]
        valid = (
            torch.isfinite(value).all(0)
            & (destination_x >= 0)
            & (destination_x < 320)
            & (destination_y >= 0)
            & (destination_y < 180)
        )
        grid = torch.stack(
            (
                2.0 * (destination_x + 0.5) / 320.0 - 1.0,
                2.0 * (destination_y + 0.5) / 180.0 - 1.0,
            ),
            dim=-1,
        )[None]
        sampled = F.grid_sample(
            dst[None], grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )[0]
        warp_error = (sampled - src).abs().mean(0)
        zero_error = (dst - src).abs().mean(0)
        magnitude = torch.linalg.vector_norm(value, dim=0)
        static = (zero_error < 5.0 / 255.0) & valid
        output.append(
            {
                "valid_fraction": float(valid.float().mean()),
                "warp_l1": float(warp_error[valid].mean()),
                "zero_l1": float(zero_error[valid].mean()),
                "warp_to_zero_ratio": float(
                    warp_error[valid].mean() / zero_error[valid].mean().clamp_min(1.0e-8)
                ),
                "magnitude_median_px": float(magnitude[valid].median()),
                "magnitude_p95_px": float(torch.quantile(magnitude[valid], 0.95)),
                "magnitude_p99_px": float(torch.quantile(magnitude[valid], 0.99)),
                "static_flow_median_px": (
                    float(magnitude[static].median()) if static.any() else None
                ),
                "nonfinite_fraction": float((~torch.isfinite(value)).float().mean()),
            }
        )
    return output


def _audit_megaflow(
    args: argparse.Namespace, manifest: dict[str, Any]
) -> dict[str, Any]:
    from preprocessing.megaflow import MegaFlowDROIDTeacher

    teacher = MegaFlowDROIDTeacher(
        REPOSITORY_ROOT / "third_party" / "MegaFlow",
        cache_dir=args.model_cache,
        device="cuda:0",
    )
    backend = TFDSRLDSBackend(args.droid_root, cache_size=1)
    candidates: list[dict[str, float | int]] = []
    episode_cache: dict[int, tuple[np.ndarray, list[str]]] = {}
    for entry_index, entry in enumerate(manifest["episodes"]):
        retained, language = _retained_episode(backend, entry)
        episode_cache[entry_index] = (retained, language)
        for source_index in range(max(0, len(retained) - 2)):
            difference_map = np.abs(
                retained[source_index].astype(np.float32)
                - retained[source_index + 2].astype(np.float32)
            ).mean(axis=-1)
            background = float(np.median(difference_map))
            p95 = float(np.percentile(difference_map, 95.0))
            candidates.append(
                {
                    "mean": float(difference_map.mean()),
                    "background_median": background,
                    "p95": p95,
                    "localized_score": p95 - background,
                    "entry_index": entry_index,
                    "source_index": source_index,
                }
            )
    if not candidates:
        raise ValueError("Pilot contains no raw-gap-6 MegaFlow pairs.")
    by_mean = sorted(candidates, key=lambda value: float(value["mean"]))
    localized_candidates = [
        value for value in candidates if float(value["background_median"]) <= 8.0
    ]
    if not localized_candidates:
        raise ValueError("Pilot has no localized-motion MegaFlow audit candidate.")
    selected = [
        ("near_static", by_mean[len(by_mean) // 10]),
        (
            "localized_motion",
            max(localized_candidates, key=lambda value: float(value["localized_score"])),
        ),
        ("large_displacement", by_mean[-1]),
    ]
    first = selected[0][1]
    first_images = episode_cache[int(first["entry_index"])][0][
        int(first["source_index"])
    ]
    same = torch.from_numpy(first_images).permute(0, 3, 1, 2)
    same = torch.stack((same, same), dim=1)
    torch.cuda.synchronize()
    identical_started = time.perf_counter()
    identical = teacher.infer_sequence(same).numpy()[:, 0]
    torch.cuda.synchronize()
    identical_seconds = time.perf_counter() - identical_started
    identical_magnitude = np.linalg.norm(identical, axis=1)
    identical_metrics = {
        "shape": list(identical.shape),
        "nonfinite_fraction": float((~np.isfinite(identical)).mean()),
        "magnitude_median_px": float(np.nanmedian(identical_magnitude)),
        "magnitude_p95_px": float(np.nanpercentile(identical_magnitude, 95.0)),
        "seconds": identical_seconds,
    }
    rows = []
    panels: list[tuple[str, np.ndarray]] = []
    for label, candidate in selected:
        entry_index = int(candidate["entry_index"])
        source_index = int(candidate["source_index"])
        retained, language = episode_cache[entry_index]
        source = retained[source_index]
        target = retained[source_index + 2]
        sequence = torch.from_numpy(np.stack((source, target), axis=1)).permute(
            0, 1, 4, 2, 3
        )
        torch.cuda.synchronize()
        started = time.perf_counter()
        flow = teacher.infer_sequence(sequence).numpy()[:, 0]
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        quality = _flow_quality(flow, source, target)
        rows.append(
            {
                "label": label,
                "episode_id": manifest["episodes"][entry_index]["episode_id"],
                "raw_timestep": source_index * 3,
                "target_raw_timestep": source_index * 3 + 6,
                "language_instructions": language,
                "rgb_difference_u8": float(candidate["mean"]),
                "background_difference_median_u8": float(
                    candidate["background_median"]
                ),
                "difference_p95_u8": float(candidate["p95"]),
                "shape": list(flow.shape),
                "seconds": elapsed,
                "cameras": quality,
            }
        )
        for camera in range(2):
            panels.extend(
                (
                    (f"{label} cam {camera} t", source[camera]),
                    (f"{label} cam {camera} t+6", target[camera]),
                    (f"{label} cam {camera} flow", _flow_rgb(flow[camera])),
                )
            )
    identical_gate = (
        identical_metrics["shape"] == [2, 2, 180, 320]
        and identical_metrics["nonfinite_fraction"] == 0.0
        and identical_metrics["magnitude_median_px"] <= 0.25
        and identical_metrics["magnitude_p95_px"] <= 1.0
    )
    flattened = [camera for row in rows for camera in row["cameras"]]
    real_gate = (
        all(row["shape"] == [2, 2, 180, 320] for row in rows)
        and max(camera["nonfinite_fraction"] for camera in flattened) == 0.0
        and min(camera["valid_fraction"] for camera in flattened) >= 0.50
        and min(
            camera["valid_fraction"]
            for row in rows[:2]
            for camera in row["cameras"]
        )
        >= 0.75
        and all(
            float(np.median([camera["warp_to_zero_ratio"] for camera in row["cameras"]]))
            < 1.05
            for row in rows
        )
        and all(
            camera["static_flow_median_px"] is None
            or camera["static_flow_median_px"] <= 2.0
            for camera in rows[0]["cameras"]
        )
    )
    visualization = _save_panels(
        Path(args.pilot_root) / "reports" / "teacher_visualizations" / "megaflow.jpg",
        panels,
    )
    return {
        "teacher": "megaflow",
        "metadata": teacher.metadata(),
        "identical_frame": identical_metrics,
        "real_gap6": rows,
        "identical_frame_gate": bool(identical_gate),
        "real_gap6_gate": bool(real_gate),
        "mean_fields_per_second": (
            (2 + len(rows) * 2)
            / (identical_seconds + sum(row["seconds"] for row in rows))
        ),
        "visualization": visualization,
        "passed": bool(identical_gate and real_gate),
    }


def _audit_lagernvs(
    args: argparse.Namespace, manifest: dict[str, Any]
) -> dict[str, Any]:
    from preprocessing.lagernvs.camera import canonical_intrinsics
    from preprocessing.lagernvs.official import LagerNVSDROIDTeacher
    from preprocessing.lagernvs.pose import (
        LagerTargetPoseConfig,
        sample_safe_target_poses,
    )
    from preprocessing.stage0.workflow import deterministic_item_seed

    artifact_path = _part_root(Path(args.pilot_root)) / "da3_lager_sample.npz"
    if not artifact_path.is_file():
        raise FileNotFoundError("Run the DA3 teacher audit before LagerNVS.")
    with np.load(artifact_path, allow_pickle=False) as artifact:
        entry_index = int(artifact["episode_index"])
        retained_index = int(artifact["retained_index"])
        depth_np = np.asarray(artifact["depth"], dtype=np.float32)
    entry = manifest["episodes"][entry_index]
    backend = TFDSRLDSBackend(args.droid_root, cache_size=1)
    retained, language = _retained_episode(backend, entry)
    synchronized = retained[retained_index]
    K_np, c2w_np, _w2c = _geometry(entry)
    teacher = LagerNVSDROIDTeacher(
        REPOSITORY_ROOT / "third_party" / "LagerNVS",
        cache_dir=args.model_cache,
        device="cuda:0",
        dtype=torch.bfloat16,
        microbatch_size=1,
    )
    K = torch.from_numpy(K_np)[None].to(teacher.device)
    c2w = torch.from_numpy(c2w_np)[None].to(teacher.device)
    target_K = canonical_intrinsics(
        (1,), focal_px=teacher.canonical_focal_px, device=teacher.device
    )
    depth = torch.from_numpy(depth_np)[:, None][None].to(teacher.device)
    validity = torch.isfinite(depth) & (depth > 0.0)
    poses = sample_safe_target_poses(
        c2w,
        K,
        target_K,
        depth,
        validity,
        torch.tensor(args.scene_center, device=teacher.device),
        LagerTargetPoseConfig(),
        seed=deterministic_item_seed(
            str(manifest["schema_signature"]),
            str(entry["episode_id"]),
            retained_index * 3,
        ),
    )
    source = torch.from_numpy(synchronized).permute(0, 3, 1, 2)[None]
    prepared = teacher.prepare_inputs(source, K, c2w, poses["target_c2w"])
    reconstructor_calls = 0

    def count_reconstructor(_module, _inputs, _output) -> None:
        nonlocal reconstructor_calls
        reconstructor_calls += 1

    hook = teacher.model.reconstructor.register_forward_hook(count_reconstructor)
    torch.cuda.synchronize()
    started = time.perf_counter()
    amortized = teacher.infer_prepared(prepared)["generated_rgb"]
    torch.cuda.synchronize()
    amortized_seconds = time.perf_counter() - started
    hook.remove()

    reference = []
    torch.cuda.synchronize()
    reference_started = time.perf_counter()
    for target_index in range(4):
        images = torch.cat(
            (
                prepared["canonical_source_rgb"],
                torch.zeros_like(prepared["canonical_source_rgb"][:, :1]),
            ),
            dim=1,
        )
        rays = torch.cat(
            (prepared["rays"][:, :2], prepared["target_rays"][:, target_index : target_index + 1]),
            dim=1,
        )
        tokens = torch.cat(
            (
                prepared["camera_tokens"][:, :2],
                prepared["camera_tokens"][:, 2 + target_index : 3 + target_index],
            ),
            dim=1,
        )
        autocast = (
            torch.autocast("cuda", dtype=teacher.teacher_dtype)
            if teacher.device.type == "cuda"
            else nullcontext()
        )
        with torch.inference_mode(), autocast:
            reference.append(teacher.model(images, rays, tokens, num_cond_views=2)[:, 2:])
    reference = torch.cat(reference, dim=1).float()
    torch.cuda.synchronize()
    reference_seconds = time.perf_counter() - reference_started
    difference = (amortized - reference).abs()
    alpha = poses["alpha"][0].detach().cpu().numpy()
    target_c2w = poses["target_c2w"][0]
    target_w2c = poses["target_w2c"][0]
    generated = (
        amortized[0].mul(255.0).round().clamp(0, 255).byte().cpu().permute(0, 2, 3, 1).numpy()
    )
    panels = [("source cam A", synchronized[0]), ("source cam B", synchronized[1])]
    panels.extend(
        (f"target {index} alpha={alpha[index]:.3f}", generated[index])
        for index in range(4)
    )
    visualization = _save_panels(
        Path(args.pilot_root) / "reports" / "teacher_visualizations" / "lagernvs.jpg",
        panels,
    )
    canonical_K = prepared["canonical_K"][0, 0].detach().cpu().numpy()
    expected_K = np.asarray(
        ((186.5, 0.0, 128.0), (0.0, 186.5, 128.0), (0.0, 0.0, 1.0)),
        dtype=np.float32,
    )
    pose_inverse_error = float(
        (target_c2w @ target_w2c - torch.eye(4, device=target_c2w.device)).abs().max()
    )
    alpha_gate = (
        0.15 <= alpha[0] <= 0.25
        and 0.25 <= alpha[1] <= 0.35
        and 0.65 <= alpha[2] <= 0.75
        and 0.75 <= alpha[3] <= 0.85
        and np.allclose(alpha[2:], (1.0 - alpha[1], 1.0 - alpha[0]), atol=2.0e-6)
        and np.all(np.abs(alpha - 0.5) >= 0.15 - 1.0e-6)
    )
    passed = (
        list(amortized.shape) == [1, 4, 3, 256, 256]
        and torch.isfinite(amortized).all().item()
        and float(amortized.min()) >= 0.0
        and float(amortized.max()) <= 1.0
        and reconstructor_calls == 1
        and float(difference.max()) <= 5.0e-3
        and alpha_gate
        and np.allclose(canonical_K, expected_K, atol=1.0e-5)
        and pose_inverse_error <= 3.0e-3
        and float(poses["support_mask"].float().mean()) > 0.0
    )
    return {
        "teacher": "lagernvs",
        "metadata": teacher.metadata(),
        "episode_id": entry["episode_id"],
        "raw_timestep": retained_index * 3,
        "language_instructions": language,
        "shape": list(amortized.shape),
        "alpha": alpha.tolist(),
        "canonical_K": canonical_K.tolist(),
        "target_c2w": target_c2w.detach().cpu().tolist(),
        "target_w2c": target_w2c.detach().cpu().tolist(),
        "pose_inverse_max_abs_error": pose_inverse_error,
        "support_fraction": float(poses["support_mask"].float().mean()),
        "reconstructor_forward_calls_for_four_targets": reconstructor_calls,
        "amortized_vs_four_official_calls_max_abs": float(difference.max()),
        "amortized_vs_four_official_calls_mean_abs": float(difference.mean()),
        "amortized_seconds": amortized_seconds,
        "four_independent_official_calls_seconds": reference_seconds,
        "amortization_speedup": reference_seconds / amortized_seconds,
        "targets_per_second": 4.0 / amortized_seconds,
        "visualization": visualization,
        "passed": bool(passed),
    }


def _worker(args: argparse.Namespace) -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible not in AUTHORIZED_GPUS:
        raise RuntimeError("Teacher audit must expose exactly one authorized GPU UUID.")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Teacher audit expected one visible CUDA device.")
    root = Path(args.pilot_root).expanduser().resolve()
    manifest = load_stage0_manifest(root)
    expected_scene_center = tuple(
        float(value)
        for value in manifest["teacher_processing"]["lagernvs"][
            "configured_scene_center"
        ]
    )
    if tuple(float(value) for value in args.scene_center) != expected_scene_center:
        raise ValueError("Teacher audit scene center differs from the signed manifest.")
    started = time.time()
    if args.teacher == "da3":
        payload = _audit_da3(args, manifest)
    elif args.teacher == "megaflow":
        payload = _audit_megaflow(args, manifest)
    else:
        payload = _audit_lagernvs(args, manifest)
    payload.update(
        {
            "schema_version": 1,
            "pilot_schema_signature": manifest["schema_signature"],
            "pipeline_signature": manifest["pipeline_signature"],
            "started_unix": started,
            "finished_unix": time.time(),
            "gpu": visible,
        }
    )
    output = _part_root(root) / f"{args.teacher}.json"
    write_json_atomic(output, payload)
    print(json.dumps(payload, indent=2), flush=True)
    if not payload["passed"]:
        raise SystemExit(f"{args.teacher} teacher quality gate failed.")


def _combine(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.pilot_root).expanduser().resolve()
    manifest = load_stage0_manifest(root)
    parts = {}
    for name in ("da3", "megaflow", "lagernvs"):
        path = _part_root(root) / f"{name}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        parts[name] = json.loads(path.read_text(encoding="utf-8"))
        if parts[name]["pilot_schema_signature"] != manifest["schema_signature"]:
            raise ValueError(f"{name} audit belongs to another pilot manifest.")
        if parts[name]["pipeline_signature"] != manifest["pipeline_signature"]:
            raise ValueError(f"{name} audit belongs to another pipeline contract.")
    gates = {
        "da3_two_view_metric_path": bool(parts["da3"]["passed"]),
        "megaflow_identical_frame": bool(parts["megaflow"]["identical_frame_gate"]),
        "megaflow_real_gap6": bool(parts["megaflow"]["real_gap6_gate"]),
        "lagernvs_amortized_four_target": bool(parts["lagernvs"]["passed"]),
    }
    report = {
        "schema_version": 1,
        "pilot_schema_signature": manifest["schema_signature"],
        "pipeline_signature": manifest["pipeline_signature"],
        "gates": gates,
        "passed": all(gates.values()),
        "teachers": parts,
    }
    output = Path(
        getattr(args, "output", None)
        or root / "reports" / "teacher_model_audit.json"
    ).expanduser().resolve()
    write_json_atomic(output, report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def _run(args: argparse.Namespace) -> None:
    root = Path(args.pilot_root).expanduser().resolve()
    manifest = load_stage0_manifest(root)
    python = {
        "da3": args.python_da3,
        "megaflow": args.python_megaflow,
        "lagernvs": args.python_lagernvs,
    }
    state_path = root / "progress" / "teacher-model-audit.json"
    state: dict[str, Any] = {
        "started_unix": time.time(),
        "gpu": args.gpu,
        "workers": [],
    }
    for teacher in ("da3", "megaflow", "lagernvs"):
        part_path = _part_root(root) / f"{teacher}.json"
        if part_path.is_file():
            existing = json.loads(part_path.read_text(encoding="utf-8"))
            if (
                existing.get("pilot_schema_signature")
                == manifest["schema_signature"]
                and existing.get("pipeline_signature")
                == manifest["pipeline_signature"]
                and existing.get("passed") is True
            ):
                state["workers"].append(
                    {
                        "teacher": teacher,
                        "status": "skipped_verified_pass",
                        "report": str(part_path),
                    }
                )
                write_json_atomic(state_path, state)
                continue
        # Preserve the virtualenv entry-point symlink; resolving it would invoke
        # the shared base interpreter without this environment's site-packages.
        executable = Path(python[teacher]).expanduser().absolute()
        if not executable.is_file():
            raise FileNotFoundError(executable)
        command = [
            str(executable),
            str(Path(__file__).resolve()),
            "worker",
            "--teacher",
            teacher,
            "--pilot-root",
            str(root),
            "--droid-root",
            str(Path(args.droid_root).expanduser().resolve()),
            "--model-cache",
            str(Path(args.model_cache).expanduser().resolve()),
            "--scene-center",
            *(str(value) for value in args.scene_center),
        ]
        log = root / "logs" / "teacher-audit" / f"{teacher}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(REPOSITORY_ROOT)
        environment["CUDA_VISIBLE_DEVICES"] = args.gpu
        environment["TOKENIZERS_PARALLELISM"] = "false"
        worker_state = {
            "teacher": teacher,
            "command": command,
            "log": str(log),
            "start_time_unix": time.time(),
            "exit_code": None,
        }
        state["workers"].append(worker_state)
        write_json_atomic(state_path, state)
        with log.open("ab", buffering=0) as stream:
            result = subprocess.run(
                command,
                cwd=REPOSITORY_ROOT,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        worker_state["exit_code"] = int(result.returncode)
        worker_state["end_time_unix"] = time.time()
        write_json_atomic(state_path, state)
        if result.returncode:
            raise RuntimeError(f"{teacher} audit failed; inspect {log}.")
    report = _combine(args)
    if not report["passed"]:
        raise SystemExit("Teacher model audit did not pass every gate.")


def main() -> None:
    args = parse_args()
    if args.command == "run":
        _run(args)
    elif args.command == "worker":
        _worker(args)
    elif args.command == "combine":
        _combine(args)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
