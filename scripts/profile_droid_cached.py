#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from dataset.droid.dataset import DROIDPreprocessedDataset, droid_collate
from dataset.droid.preprocessed_manifest import load_stage0_manifest
from models.gaussian.parameterization import WorldSpaceGaussianParameterization
from models.training.distributed import move_to_device
from models.training.losses import cross_view_info_nce
from models.training.reconstruction import compute_droid_reconstruction
from scripts.train_droid import (
    _dataset_config,
    _model_and_renderer,
    _train_config,
    _validate_cached_manifest,
    _validate_fixed_pipeline_contract,
    loaded_foundation_modules,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile the cached DROID training step without foundation models."
    )
    parser.add_argument(
        "--config", default="config/splattervae/droid/pretrain.yaml"
    )
    parser.add_argument(
        "--preprocessed-root",
        default="/home/ws/data/droid_stage0_preprocessed/pilot",
    )
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"mean": 0.0, "median": 0.0, "p95": 0.0}
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95.0)),
    }


class _CudaStages:
    def __init__(self) -> None:
        self.pending: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = (
            defaultdict(list)
        )
        self.values: dict[str, list[float]] = defaultdict(list)

    def wrap(self, name: str, callback: Callable[[], Any]) -> Any:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = callback()
        end.record()
        self.pending[name].append((start, end))
        return result

    def flush(self) -> None:
        torch.cuda.synchronize()
        for name, pairs in self.pending.items():
            self.values[name].extend(float(start.elapsed_time(end)) for start, end in pairs)
        self.pending.clear()


class _SystemSampler:
    def __init__(self, gpu_identifier: str) -> None:
        self.gpu_identifier = gpu_identifier
        self.gpu_values: list[float] = []
        self.cpu_values: list[float] = []
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(0.5):
            self.cpu_values.append(float(psutil.cpu_percent(interval=None)))
            try:
                output = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "-i",
                        self.gpu_identifier,
                        "--query-gpu=utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    stderr=subprocess.DEVNULL,
                    timeout=2,
                )
                self.gpu_values.append(float(output.strip().splitlines()[0]))
            except (OSError, ValueError, subprocess.SubprocessError):
                pass

    def __enter__(self):
        psutil.cpu_percent(interval=None)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._stop.set()
        self.thread.join(timeout=3)


def _forbidden_modules() -> list[str]:
    return loaded_foundation_modules()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Cached training profiling requires one authorized CUDA GPU.")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible.count(",") or not visible.startswith("GPU-"):
        raise RuntimeError(
            "Expose exactly one authorized GPU UUID through CUDA_VISIBLE_DEVICES."
        )
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    _validate_fixed_pipeline_contract(config)
    config["dataset"]["preprocessed_root"] = str(
        Path(args.preprocessed_root).expanduser().resolve()
    )
    config["dataset"]["per_gpu_logical_batch"] = int(args.batch_size)
    config["dataset"]["workers"] = int(args.workers)
    config["dataset"]["prefetch_factor"] = int(args.prefetch_factor)
    _validate_cached_manifest(config)
    if _forbidden_modules():
        raise RuntimeError("A foundation-model module was imported before profiling.")

    dataset_config = replace(
        _dataset_config(config, "train"), profile_timings=True
    )
    dataset = DROIDPreprocessedDataset(dataset_config)
    dataset_manifest = load_stage0_manifest(args.preprocessed_root)
    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(args.batch_size),
        "shuffle": True,
        "num_workers": int(args.workers),
        "pin_memory": True,
        "drop_last": True,
        "collate_fn": droid_collate,
    }
    if args.workers:
        loader_kwargs.update(
            {
                "persistent_workers": True,
                "prefetch_factor": int(args.prefetch_factor),
                "multiprocessing_context": "spawn",
            }
        )
    loader = DataLoader(**loader_kwargs)
    if len(loader) < int(args.warmup) + int(args.iterations):
        raise ValueError("Pilot dataset is too small for requested profile iterations.")

    device = torch.device("cuda:0")
    model, splatter = _model_and_renderer(config)
    model.train().to(device)
    train_config = _train_config(config, None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
    parameterization = WorldSpaceGaussianParameterization(splatter).to(device)
    background = torch.zeros(3, device=device)
    stage_events = _CudaStages()
    hook_starts: dict[str, torch.cuda.Event] = {}
    hook_handles = []

    def register(name: str, module: torch.nn.Module) -> None:
        def before(_module, _inputs):
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            hook_starts[name] = event

        def after(_module, _inputs, _output):
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            stage_events.pending[name].append((hook_starts.pop(name), end))

        hook_handles.append(module.register_forward_pre_hook(before))
        hook_handles.append(module.register_forward_hook(after))

    register("vit", model.encoder)
    register("contrastive_projector", model.contrastive_projector)
    register("gaussian_decoder", model.gaussian_decoder)

    iterator = iter(loader)
    for _ in range(int(args.warmup)):
        cpu_batch = next(iterator)
        batch = move_to_device(cpu_batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = model(
                batch["representation_histories"],
                batch["representation_middle_motion"],
                batch["representation_validity"],
            )
        reconstruction = compute_droid_reconstruction(
            parameterization,
            splatter,
            prediction,
            batch,
            train_config,
            motion_translation_max=model.motion_translation_max,
            background_color=background,
            novel_view_enabled=True,
        )
        reconstruction["loss"].backward()
        optimizer.zero_grad(set_to_none=True)
        stage_events.pending.clear()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)

    wall_values: dict[str, list[float]] = defaultdict(list)
    dataset_values: dict[str, list[float]] = defaultdict(list)
    valid_counts = []
    keep_counts = []
    masking_ratios = []
    partial_patch_counts = []
    process = psutil.Process()
    io_before = process.io_counters()
    disk_before = psutil.disk_io_counters()
    profile_started = time.perf_counter()
    samples = 0
    with _SystemSampler(visible) as system:
        for _ in range(int(args.iterations)):
            iteration_started = time.perf_counter()
            loader_started = time.perf_counter()
            cpu_batch = next(iterator)
            wall_values["dataloader"].append(
                (time.perf_counter() - loader_started) * 1000.0
            )
            for name, values in cpu_batch["profile_timings_ms"].items():
                dataset_values[name].extend(
                    float(value) for value in values.reshape(-1)
                )

            transfer_started = time.perf_counter()
            batch = move_to_device(cpu_batch, device)
            torch.cuda.synchronize()
            wall_values["cpu_to_gpu"].append(
                (time.perf_counter() - transfer_started) * 1000.0
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = stage_events.wrap(
                    "model_forward_total",
                    lambda batch=batch: model(
                        batch["representation_histories"],
                        batch["representation_middle_motion"],
                        batch["representation_validity"],
                    ),
                )
                contrastive = stage_events.wrap(
                    "contrastive_loss",
                    lambda prediction=prediction: cross_view_info_nce(
                        prediction["projected_cls_by_view"],
                        train_config.contrastive_temperature,
                    )[0],
                )
            reconstruction = compute_droid_reconstruction(
                parameterization,
                splatter,
                prediction,
                batch,
                train_config,
                motion_translation_max=model.motion_translation_max,
                background_color=background,
                novel_view_enabled=True,
                stage_runner=stage_events.wrap,
            )
            total = reconstruction["loss"] + contrastive.float()
            if not torch.isfinite(total):
                raise FloatingPointError("Cached profile produced a non-finite loss.")
            stage_events.wrap("backward", total.backward)
            finite_gradients = all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            )
            if not finite_gradients:
                raise FloatingPointError("Cached profile produced non-finite gradients.")
            stage_events.wrap("optimizer", optimizer.step)
            stage_events.flush()
            wall_values["iteration_total"].append(
                (time.perf_counter() - iteration_started) * 1000.0
            )
            samples += int(batch["target_rgb"].shape[0])

            patch_validity = prediction["patch_validity_by_view"].bool()
            n_valid = patch_validity.sum(dim=-1).cpu().numpy()
            n_keep = prediction["num_visible_patches_by_view"].cpu().numpy()
            valid_counts.extend(n_valid.reshape(-1).tolist())
            keep_counts.extend(n_keep.reshape(-1).tolist())
            masking_ratios.extend(((n_valid - n_keep) / n_valid).reshape(-1).tolist())
            pixel_validity = batch["representation_validity"][:, :, -1].float()
            patch_fraction = F.avg_pool2d(
                pixel_validity.flatten(0, 1), 16, 16
            ).flatten(1)
            partial_patch_counts.extend(
                ((patch_fraction > 0.0) & (patch_fraction < 1.0))
                .sum(dim=-1)
                .cpu()
                .tolist()
            )
    elapsed = time.perf_counter() - profile_started
    io_after = process.io_counters()
    disk_after = psutil.disk_io_counters()
    for handle in hook_handles:
        handle.remove()
    dataset.close()
    forbidden = _forbidden_modules()
    if forbidden:
        raise RuntimeError(
            f"Normal cached training imported foundation modules: {forbidden}"
        )

    all_components = {
        name: _summary(values) for name, values in stage_events.values.items()
    }
    all_components.update(
        {f"wall/{name}": _summary(values) for name, values in wall_values.items()}
    )
    all_components.update(
        {
            f"dataset/{name}": _summary(values)
            for name, values in dataset_values.items()
        }
    )
    process_read = max(0, int(io_after.read_bytes - io_before.read_bytes))
    system_read = (
        max(0, int(disk_after.read_bytes - disk_before.read_bytes))
        if disk_before is not None and disk_after is not None
        else None
    )
    report = {
        "schema_version": 1,
        "dataset_schema_signature": dataset_manifest["schema_signature"],
        "pipeline_signature": dataset_manifest["pipeline_signature"],
        "cached_dataset_root": str(Path(args.preprocessed_root).resolve()),
        "iterations": int(args.iterations),
        "samples": samples,
        "elapsed_seconds": elapsed,
        "samples_per_second": samples / elapsed,
        "decoded_rgb_images_per_second": samples * 18 / elapsed,
        "components_ms": all_components,
        "gpu_utilization_percent": _summary(system.gpu_values),
        "cpu_utilization_percent": _summary(system.cpu_values),
        "process_disk_read_bytes": process_read,
        "process_disk_read_megabytes_per_second": process_read / elapsed / 1.0e6,
        "system_disk_read_bytes": system_read,
        "system_disk_read_megabytes_per_second": (
            None if system_read is None else system_read / elapsed / 1.0e6
        ),
        "peak_allocated_vram_gib": torch.cuda.max_memory_allocated(device)
        / (1024**3),
        "peak_reserved_vram_gib": torch.cuda.max_memory_reserved(device) / (1024**3),
        "masking": {
            "n_valid": {
                "min": min(valid_counts),
                "mean": float(np.mean(valid_counts)),
                "max": max(valid_counts),
            },
            "n_keep": {
                "min": min(keep_counts),
                "mean": float(np.mean(keep_counts)),
                "max": max(keep_counts),
            },
            "actual_content_masking_ratio_mean": float(np.mean(masking_ratios)),
            "actual_content_masking_ratio_min": float(np.min(masking_ratios)),
            "actual_content_masking_ratio_max": float(np.max(masking_ratios)),
            "partial_valid_patches_per_view_mean": float(
                np.mean(partial_patch_counts)
            ),
        },
        "all_losses_finite": True,
        "all_gradients_finite": True,
        "foundation_model_modules_loaded": forbidden,
    }
    output = Path(
        args.output
        or Path(args.preprocessed_root) / "reports" / "cached_training_profile.json"
    ).expanduser().resolve()
    write = output.with_suffix(output.suffix + ".partial")
    write.parent.mkdir(parents=True, exist_ok=True)
    write.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(write, output)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
