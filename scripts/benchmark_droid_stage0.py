#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from dataset.droid.codecs import (
    JPEGCodecConfig,
    NumericCodecConfig,
    decode_jpeg,
    decode_numeric_array,
    encode_jpeg,
)
from dataset.droid.dataset import (
    DROIDDatasetConfig,
    DROIDPreprocessedDataset,
    droid_collate,
)
from dataset.droid.preprocessed_manifest import (
    load_stage0_manifest,
    sample_key,
    shard_path,
)
from dataset.droid.records import unpack_lager_record, unpack_real_rgb_record
from dataset.droid.shards import IndexedTarReader, shard_sidecars, write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Stage-0 codecs, storage, and cached DataLoader."
    )
    parser.add_argument(
        "--pilot-root",
        default="/home/ws/data/droid_stage0_preprocessed/pilot",
    )
    parser.add_argument(
        "--full-root",
        default="/home/ws/data/droid_stage0_preprocessed",
        help="Root containing the exact full manifest (payload shards need not exist).",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--visualization-root", default=None)
    parser.add_argument("--decode-seconds", type=float, default=2.0)
    parser.add_argument("--numeric-seconds", type=float, default=2.0)
    parser.add_argument("--loader-batches", type=int, default=100)
    parser.add_argument("--loader-warmup-batches", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--safe-margin-fraction", type=float, default=0.20)
    parser.add_argument("--recovery-shards", type=int, default=6)
    return parser.parse_args()


def _psnr(reference: np.ndarray, decoded: np.ndarray, mask: np.ndarray | None = None) -> float:
    difference = (
        reference.astype(np.float64) - decoded.astype(np.float64)
    ) ** 2
    if mask is not None:
        values = difference[np.asarray(mask, dtype=bool)]
    else:
        values = difference.reshape(-1)
    mse = float(values.mean()) if values.size else 0.0
    return float("inf") if mse == 0.0 else 10.0 * math.log10(255.0**2 / mse)


def _ssim(reference: np.ndarray, decoded: np.ndarray) -> float:
    first = (
        torch.from_numpy(reference.copy())
        .permute(2, 0, 1)
        .float()
        .unsqueeze(0)
        / 255.0
    )
    second = (
        torch.from_numpy(decoded.copy())
        .permute(2, 0, 1)
        .float()
        .unsqueeze(0)
        / 255.0
    )
    kernel = 11
    padding = kernel // 2
    mu_x = F.avg_pool2d(first, kernel, 1, padding)
    mu_y = F.avg_pool2d(second, kernel, 1, padding)
    sigma_x = F.avg_pool2d(first.square(), kernel, 1, padding) - mu_x.square()
    sigma_y = F.avg_pool2d(second.square(), kernel, 1, padding) - mu_y.square()
    sigma_xy = F.avg_pool2d(first * second, kernel, 1, padding) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    value = (
        (2.0 * mu_x * mu_y + c1)
        * (2.0 * sigma_xy + c2)
        / (
            (mu_x.square() + mu_y.square() + c1)
            * (sigma_x + sigma_y + c2)
        ).clamp_min(1.0e-12)
    )
    return float(value.mean())


def _edge_mask(image: np.ndarray) -> np.ndarray:
    value = (
        torch.from_numpy(image.copy())
        .permute(2, 0, 1)
        .float()
        .mean(dim=0, keepdim=True)
        .unsqueeze(0)
    )
    sobel_x = value.new_tensor(
        ((-1.0, 0.0, 1.0), (-2.0, 0.0, 2.0), (-1.0, 0.0, 1.0))
    ).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(-1, -2)
    magnitude = torch.sqrt(
        F.conv2d(value, sobel_x, padding=1).square()
        + F.conv2d(value, sobel_y, padding=1).square()
    )[0, 0]
    threshold = torch.quantile(magnitude, 0.75)
    spatial = (magnitude >= threshold).numpy()
    return np.repeat(spatial[..., None], 3, axis=-1)


def _load_codec_references(root: Path) -> list[tuple[str, str, np.ndarray]]:
    output = []
    for path in sorted((root / "metadata" / "codec_references").glob("*.npz")):
        kind = path.name.split("-", 1)[0]
        with np.load(path, allow_pickle=False) as values:
            identifiers = values["identifiers"]
            images = values["images"]
            for identifier, image in zip(identifiers, images, strict=True):
                output.append((kind, str(identifier), np.asarray(image, dtype=np.uint8)))
    if not output:
        raise FileNotFoundError(
            f"No pre-JPEG codec references exist under {root}; rerun pilot workers "
            "with --codec-audit-images > 0."
        )
    return output


def _codec_quality(
    references: list[tuple[str, str, np.ndarray]],
    subsampling: str,
) -> tuple[dict[str, Any], dict[tuple[str, str, int], bytes]]:
    rows = []
    payloads: dict[tuple[str, str, int], bytes] = {}
    for kind, identifier, image in references:
        edge = _edge_mask(image)
        for quality in (95, 97):
            payload = encode_jpeg(
                image,
                JPEGCodecConfig(quality=quality, subsampling=subsampling),
            )
            decoded = decode_jpeg(payload)
            payloads[(kind, identifier, quality)] = payload
            rows.append(
                {
                    "kind": kind,
                    "identifier": identifier,
                    "quality": quality,
                    "bytes": len(payload),
                    "psnr_db": _psnr(image, decoded),
                    "ssim": _ssim(image, decoded),
                    "edge_psnr_db": _psnr(image, decoded, edge),
                    "edge_mae_u8": float(
                        np.abs(
                            image.astype(np.float32) - decoded.astype(np.float32)
                        )[edge].mean()
                    ),
                }
            )
    summary: dict[str, Any] = {"sample_images": len(references), "by_kind": {}}
    for kind in sorted({row["kind"] for row in rows}):
        summary["by_kind"][kind] = {}
        for quality in (95, 97):
            selected = [
                row for row in rows
                if row["kind"] == kind and row["quality"] == quality
            ]
            summary["by_kind"][kind][f"q{quality}"] = {
                metric: float(
                    np.mean([row[metric] for row in selected], dtype=np.float64)
                )
                for metric in (
                    "bytes",
                    "psnr_db",
                    "ssim",
                    "edge_psnr_db",
                    "edge_mae_u8",
                )
            }
    summary["rows"] = rows
    return summary, payloads


def _comparison_visualizations(
    references: list[tuple[str, str, np.ndarray]],
    payloads: dict[tuple[str, str, int], bytes],
    output_root: Path,
    *,
    maximum_per_kind: int = 12,
) -> list[str]:
    output_root.mkdir(parents=True, exist_ok=True)
    paths = []
    selected: list[tuple[str, str, np.ndarray]] = []
    selected_counts: Counter[str] = Counter()
    for value in references:
        kind = value[0]
        if selected_counts[kind] >= int(maximum_per_kind):
            continue
        selected.append(value)
        selected_counts[kind] += 1
    for index, (kind, identifier, reference) in enumerate(selected):
        q95 = decode_jpeg(payloads[(kind, identifier, 95)])
        q97 = decode_jpeg(payloads[(kind, identifier, 97)])
        panels = (
            ("pre-JPEG reference", reference),
            ("JPEG Q95", q95),
            ("JPEG Q97", q97),
            (
                "Q95 abs error x8",
                np.clip(
                    np.abs(reference.astype(np.int16) - q95.astype(np.int16)) * 8,
                    0,
                    255,
                ).astype(np.uint8),
            ),
            (
                "Q97 abs error x8",
                np.clip(
                    np.abs(reference.astype(np.int16) - q97.astype(np.int16)) * 8,
                    0,
                    255,
                ).astype(np.uint8),
            ),
        )
        width, height = reference.shape[1], reference.shape[0]
        canvas = Image.new("RGB", (width * len(panels), height + 24), (20, 20, 20))
        draw = ImageDraw.Draw(canvas)
        for panel, (title, image) in enumerate(panels):
            canvas.paste(Image.fromarray(image), (panel * width, 24))
            draw.text((panel * width + 4, 5), title, fill="white")
        path = output_root / f"{index:02d}-{kind}.jpg"
        canvas.save(path, quality=95, subsampling=0)
        paths.append(str(path))
    return paths


def _timed_rate(
    payloads: list[bytes],
    callback: Callable[[bytes], Any],
    seconds: float,
    *,
    synchronize: Callable[[], None] | None = None,
) -> dict[str, float]:
    if not payloads:
        return {"items_per_second": 0.0, "megabytes_per_second": 0.0}
    iterations = 0
    total_bytes = 0
    started = time.perf_counter()
    while time.perf_counter() - started < float(seconds):
        payload = payloads[iterations % len(payloads)]
        callback(payload)
        iterations += 1
        total_bytes += len(payload)
    if synchronize is not None:
        synchronize()
    elapsed = time.perf_counter() - started
    return {
        "items_per_second": iterations / elapsed,
        "megabytes_per_second": total_bytes / elapsed / 1.0e6,
        "elapsed_seconds": elapsed,
    }


def _jpeg_decode_benchmarks(
    payloads: list[bytes], seconds: float
) -> dict[str, Any]:
    output = {
        "pillow": _timed_rate(payloads, decode_jpeg, seconds),
    }
    try:
        from torchvision.io import decode_jpeg as torchvision_decode_jpeg

        encoded = [torch.frombuffer(bytearray(value), dtype=torch.uint8) for value in payloads]
        output["torchvision_cpu"] = _timed_rate(
            [bytes(value.numpy()) for value in encoded],
            lambda payload: torchvision_decode_jpeg(
                torch.frombuffer(bytearray(payload), dtype=torch.uint8),
                device="cpu",
            ),
            seconds,
        )
        if torch.cuda.is_available():
            output["torchvision_cuda_nvjpeg"] = _timed_rate(
                payloads,
                lambda payload: torchvision_decode_jpeg(
                    torch.frombuffer(bytearray(payload), dtype=torch.uint8),
                    device="cuda",
                ),
                seconds,
                synchronize=torch.cuda.synchronize,
            )
    except Exception as error:  # noqa: BLE001 - optional backend failures are data.
        output["torchvision_error"] = f"{type(error).__name__}: {error}"
    return output


def _pilot_storage(root: Path) -> tuple[dict[str, Any], list[bytes], list[bytes]]:
    manifest = load_stage0_manifest(root)
    bytes_by_component: Counter[str] = Counter()
    counts: Counter[str] = Counter()
    jpeg_payloads: list[bytes] = []
    numeric_payloads: list[bytes] = []
    container_bytes = 0
    for shard in manifest["shards"]:
        path = shard_path(root, "final", int(shard["shard_id"]))
        reader = IndexedTarReader(path)
        index_path, completion_path = shard_sidecars(path)
        container_bytes += (
            path.stat().st_size
            + index_path.stat().st_size
            + completion_path.stat().st_size
        )
        episodes = manifest["episodes"][
            int(shard["episode_start"]) : int(shard["episode_stop"])
        ]
        for entry in episodes:
            for retained_index in range(int(entry["retained_count"])):
                key = sample_key(
                    int(entry["global_retained_start"]) + retained_index
                )
                rgb_record = reader.read(key, "rgb")
                real = unpack_real_rgb_record(rgb_record)
                bytes_by_component["real_rgb_jpeg"] += sum(map(len, real))
                bytes_by_component["record_framing"] += len(rgb_record) - sum(map(len, real))
                counts["real_rgb_images"] += 2
                jpeg_payloads.extend(real)

                depth = reader.read(key, "depth")
                bytes_by_component["da3_depth"] += len(depth)
                counts["depth_records"] += 1
                numeric_payloads.append(depth)

                if reader.contains(key, "flow"):
                    flow = reader.read(key, "flow")
                    bytes_by_component["megaflow"] += len(flow)
                    counts["flow_records"] += 1
                    numeric_payloads.append(flow)

                lager_record = reader.read(key, "lager")
                lager, _metadata, _support = unpack_lager_record(lager_record)
                lager_jpeg_bytes = sum(map(len, lager))
                bytes_by_component["lager_jpeg"] += lager_jpeg_bytes
                bytes_by_component["lager_metadata_support_framing"] += (
                    len(lager_record) - lager_jpeg_bytes
                )
                counts["lager_images"] += 4
                jpeg_payloads.extend(lager)

                meta = reader.read(key, "meta")
                bytes_by_component["timestep_metadata"] += len(meta)
                counts["retained_timesteps"] += 1
        reader.close()
    payload_total = sum(bytes_by_component.values())
    return (
        {
            "counts": dict(counts),
            "bytes_by_component": dict(bytes_by_component),
            "payload_total": payload_total,
            "container_index_sidecar_total": container_bytes,
            "container_overhead": container_bytes - payload_total,
        },
        jpeg_payloads,
        numeric_payloads,
    )


def _numeric_benchmark(
    payloads: list[bytes],
    numeric: NumericCodecConfig,
    seconds: float,
) -> dict[str, Any]:
    unique_shapes: dict[tuple[int, ...], list[bytes]] = {}
    for payload in payloads:
        decoded = decode_numeric_array(payload, numeric)
        unique_shapes.setdefault(tuple(decoded.shape), []).append(payload)
    output = {}
    for shape, values in unique_shapes.items():
        sample = decode_numeric_array(values[0], numeric)
        raw_bytes = sample.nbytes
        compressed = [len(value) for value in values]
        timing = _timed_rate(
            values,
            lambda payload: decode_numeric_array(payload, numeric),
            seconds,
        )
        timing["uncompressed_gigabytes_per_second"] = (
            timing["items_per_second"] * raw_bytes / 1.0e9
        )
        output[str(shape)] = {
            "records": len(values),
            "raw_bytes_per_record": raw_bytes,
            "compressed_bytes_mean": float(np.mean(compressed)),
            "compression_ratio": raw_bytes / float(np.mean(compressed)),
            "decompression": timing,
        }
    return output


def _project_storage(
    pilot: dict[str, Any],
    full_manifest: dict[str, Any],
    *,
    safe_margin_fraction: float,
    recovery_shards: int,
) -> dict[str, Any]:
    pilot_counts = pilot["counts"]
    full_counts = full_manifest["counts"]
    pilot_bytes = pilot["bytes_by_component"]
    retained_ratio = (
        int(full_counts["retained_timesteps"])
        / int(pilot_counts["retained_timesteps"])
    )
    flow_ratio = (
        int(full_counts["flow_timestamps"])
        / int(pilot_counts["flow_records"])
    )
    image_ratios = {
        "real_rgb_jpeg": int(full_counts["real_rgb_images"])
        / int(pilot_counts["real_rgb_images"]),
        "lager_jpeg": int(full_counts["lager_jpeg_images"])
        / int(pilot_counts["lager_images"]),
    }
    projected = {
        "real_rgb_jpeg": round(
            pilot_bytes["real_rgb_jpeg"] * image_ratios["real_rgb_jpeg"]
        ),
        "da3_depth": round(pilot_bytes["da3_depth"] * retained_ratio),
        "megaflow": round(pilot_bytes["megaflow"] * flow_ratio),
        "lager_jpeg": round(
            pilot_bytes["lager_jpeg"] * image_ratios["lager_jpeg"]
        ),
        "lager_metadata_support_framing": round(
            pilot_bytes["lager_metadata_support_framing"] * retained_ratio
        ),
        "record_framing": round(pilot_bytes["record_framing"] * retained_ratio),
        "timestep_metadata": round(
            pilot_bytes["timestep_metadata"] * retained_ratio
        ),
        "tar_index_sidecar_overhead": round(
            pilot["container_overhead"] * retained_ratio
        ),
    }
    final_bytes = sum(projected.values())
    # The current restart-safe workflow retains stage shards until finalization.
    # Their payload is approximately one additional final dataset, excluding
    # final TAR/index overhead. Include it explicitly in peak capacity.
    staging_bytes = sum(
        projected[name]
        for name in (
            "real_rgb_jpeg",
            "da3_depth",
            "megaflow",
            "lager_jpeg",
            "lager_metadata_support_framing",
            "record_framing",
        )
    )
    average_shard = final_bytes / max(1, len(full_manifest["shards"]))
    recovery_bytes = round(max(0, int(recovery_shards)) * average_shard)
    peak_before_margin = final_bytes + staging_bytes + recovery_bytes
    safe_required = math.ceil(
        peak_before_margin * (1.0 + float(safe_margin_fraction))
    )
    return {
        "projected_final_bytes_by_component": projected,
        "projected_final_bytes": final_bytes,
        "projected_staging_bytes_at_peak": staging_bytes,
        "recovery_overhead_bytes": recovery_bytes,
        "safe_margin_fraction": float(safe_margin_fraction),
        "safe_required_bytes": safe_required,
    }


def _loader_benchmark(
    root: Path,
    *,
    batches: int,
    warmup: int,
    batch_size: int,
    workers: int,
    prefetch_factor: int,
) -> dict[str, Any]:
    dataset = DROIDPreprocessedDataset(
        DROIDDatasetConfig(preprocessed_root=str(root), split="train")
    )
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(batch_size),
        "shuffle": True,
        "num_workers": int(workers),
        "pin_memory": True,
        "drop_last": True,
        "collate_fn": droid_collate,
    }
    if workers:
        kwargs.update(
            {
                "persistent_workers": True,
                "prefetch_factor": int(prefetch_factor),
                "multiprocessing_context": "spawn",
            }
        )
    loader = DataLoader(**kwargs)
    iterator = iter(loader)
    for _ in range(min(int(warmup), len(loader))):
        next(iterator)
    measured = min(int(batches), max(0, len(loader) - int(warmup)))
    started = time.perf_counter()
    images = 0
    samples = 0
    for _ in range(measured):
        batch = next(iterator)
        logical = int(batch["target_rgb"].shape[0])
        samples += logical
        images += logical * (6 + 12)
    elapsed = time.perf_counter() - started
    dataset.close()
    if measured == 0:
        raise ValueError("Pilot train split is too small for a loader benchmark.")
    return {
        "batches": measured,
        "seconds": elapsed,
        "batches_per_second": measured / elapsed,
        "samples_per_second": samples / elapsed,
        "decoded_rgb_images_per_second": images / elapsed,
        "workers": int(workers),
        "batch_size": int(batch_size),
    }


def main() -> None:
    args = parse_args()
    pilot_root = Path(args.pilot_root).expanduser().resolve()
    full_root = Path(args.full_root).expanduser().resolve()
    pilot_manifest = load_stage0_manifest(pilot_root)
    full_manifest = load_stage0_manifest(full_root)
    references = _load_codec_references(pilot_root)
    subsampling = str(
        pilot_manifest["encoding"]["real_rgb"]["chroma_subsampling"]
    )
    quality, quality_payloads = _codec_quality(references, subsampling)
    visualization_root = Path(
        args.visualization_root
        or pilot_root / "reports" / "codec_visualizations"
    ).expanduser().resolve()
    visualizations = _comparison_visualizations(
        references, quality_payloads, visualization_root
    )
    pilot_storage, jpeg_payloads, numeric_payloads = _pilot_storage(pilot_root)
    numeric_cfg = pilot_manifest["encoding"]["numeric_compression"]
    numeric = NumericCodecConfig(
        cname=numeric_cfg["codec"],
        compression_level=int(numeric_cfg["compression_level"]),
        shuffle=numeric_cfg["shuffle"],
    )
    storage = _project_storage(
        pilot_storage,
        full_manifest,
        safe_margin_fraction=float(args.safe_margin_fraction),
        recovery_shards=int(args.recovery_shards),
    )
    stat = os.statvfs(full_root)
    free_bytes = int(stat.f_bavail * stat.f_frsize)
    storage.update(
        {
            "filesystem_free_bytes": free_bytes,
            "capacity_pass": free_bytes >= int(storage["safe_required_bytes"]),
            "free_margin_after_safe_requirement_bytes": (
                free_bytes - int(storage["safe_required_bytes"])
            ),
            "theoretical_uncompressed_bytes": full_manifest[
                "theoretical_uncompressed_bytes"
            ],
        }
    )
    report = {
        "schema_version": 1,
        "pilot_schema_signature": pilot_manifest["schema_signature"],
        "full_schema_signature": full_manifest["schema_signature"],
        "pilot_pipeline_signature": pilot_manifest["pipeline_signature"],
        "full_pipeline_signature": full_manifest["pipeline_signature"],
        "pilot_counts": pilot_manifest["counts"],
        "full_counts": full_manifest["counts"],
        "jpeg": {
            "subsampling": subsampling,
            "quality_comparison": quality,
            "decode": _jpeg_decode_benchmarks(
                jpeg_payloads, float(args.decode_seconds)
            ),
            "visualizations": visualizations,
        },
        "numeric": _numeric_benchmark(
            numeric_payloads, numeric, float(args.numeric_seconds)
        ),
        "pilot_storage": pilot_storage,
        "storage_projection": storage,
        "loader": _loader_benchmark(
            pilot_root,
            batches=int(args.loader_batches),
            warmup=int(args.loader_warmup_batches),
            batch_size=int(args.batch_size),
            workers=int(args.workers),
            prefetch_factor=int(args.prefetch_factor),
        ),
    }
    output = Path(
        args.output or pilot_root / "reports" / "benchmark.json"
    ).expanduser().resolve()
    write_json_atomic(output, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
