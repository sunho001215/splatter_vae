# All loop-local profiling closures are invoked synchronously before their
# captured iteration state changes.
# ruff: noqa: B023

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import yaml
from huggingface_hub import get_token

from dataset.droid.dataset import DROIDLogicalDataset, droid_collate
from dataset.droid.safety import source_tree_fingerprint, validate_derived_root
from models.gaussian.parameterization import WorldSpaceGaussianParameterization
from models.splattervae import SplatterVAE
from models.training.distributed import move_to_device
from models.training.losses import cross_view_info_nce
from models.training.online_preprocessing import OnlineTeacherPipeline
from models.training.reconstruction import compute_droid_reconstruction
from models.training.validation import _detach_to_cpu
from preprocessing.common import verify_cuda_device
from preprocessing.lagernvs.official import LagerNVSDROIDTeacher
from preprocessing.lagernvs.pose import LagerTargetPoseConfig
from scripts.train_droid import (
    _dataset_config,
    _model_and_renderer,
    _online_preprocessor,
    _only_dataclass_fields,
    _train_config,
    _validate_fixed_pipeline_contract,
)

GIB = 1024**3
TRAINABLE_ONLY = "trainable_only"
ONLINE_OFF = "memfof_xlens_all_resident"
ONLINE_OFFLOAD = "memfof_xlens_sequential_offload"
ONLINE_ON = "memfof_xlens_lagernvs_all_resident"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile steady-state all-online DROID training with CUDA timing."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--workspace-stats", default=None)
    parser.add_argument("--output-root", default="outputs/profiling")
    parser.add_argument("--batch-sizes", default="1,2,4,8")
    parser.add_argument("--warmup-iterations", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--mode", choices=("off", "on", "both"), default="both")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--offload-batch-size",
        type=int,
        default=8,
        help="Logical batch for the no-reload sequential teacher-offload experiment.",
    )
    parser.add_argument(
        "--skip-offload-experiment",
        action="store_true",
        help="Skip the MEMFOF/X-Lens CPU/GPU sequential residency comparison.",
    )
    parser.add_argument(
        "--lagernvs-microbatch-sizes",
        default="1,2,4",
        help="Teacher-only microbatches to benchmark if the official checkpoint loads.",
    )
    parser.add_argument(
        "--memory-safety-fraction",
        type=float,
        default=0.85,
        help="Maximum device-memory fraction for the safe-batch recommendation.",
    )
    return parser.parse_args()


def _parse_positive_integers(value: str, label: str) -> list[int]:
    output = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not output or any(item <= 0 for item in output):
        raise ValueError(f"{label} must contain positive comma-separated integers.")
    return output


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("Cannot summarize an empty measurement list.")
    ordered = sorted(values)

    def percentile(q: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        position = q * (len(ordered) - 1)
        low = math.floor(position)
        high = math.ceil(position)
        weight = position - low
        return ordered[low] * (1.0 - weight) + ordered[high] * weight

    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
    }


def _cleanup_allocator(device: torch.device) -> None:
    gc.collect()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)


def _measure_cuda(
    device: torch.device, callable_: Callable[[], Any]
) -> tuple[Any, dict[str, float]]:
    """Synchronously measure one component and its incremental CUDA memory."""

    torch.cuda.synchronize(device)
    allocated_before = torch.cuda.memory_allocated(device)
    reserved_before = torch.cuda.memory_reserved(device)
    torch.cuda.reset_peak_memory_stats(device)
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    cpu_started = time.perf_counter()
    begin.record()
    value = callable_()
    end.record()
    end.synchronize()
    cpu_ms = (time.perf_counter() - cpu_started) * 1000.0
    peak_allocated = torch.cuda.max_memory_allocated(device)
    return value, {
        "cuda_ms": float(begin.elapsed_time(end)),
        "cpu_ms": cpu_ms,
        "allocated_before_gib": allocated_before / GIB,
        "allocated_after_gib": torch.cuda.memory_allocated(device) / GIB,
        "reserved_before_gib": reserved_before / GIB,
        "reserved_after_gib": torch.cuda.memory_reserved(device) / GIB,
        "peak_allocated_gib": peak_allocated / GIB,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / GIB,
        "incremental_peak_allocated_gib": max(
            0.0, (peak_allocated - allocated_before) / GIB
        ),
    }


def _accumulate_measurement(
    measurements: dict[str, dict[str, float]],
    name: str,
    value: dict[str, float],
) -> None:
    """Sum repeated per-sample stages into one per-iteration record."""

    if name not in measurements:
        measurements[name] = dict(value)
        return
    current = measurements[name]
    current["cuda_ms"] += value["cuda_ms"]
    current["cpu_ms"] += value["cpu_ms"]
    current["allocated_after_gib"] = value["allocated_after_gib"]
    current["reserved_after_gib"] = value["reserved_after_gib"]
    current["peak_allocated_gib"] = max(
        current["peak_allocated_gib"], value["peak_allocated_gib"]
    )
    current["peak_reserved_gib"] = max(
        current["peak_reserved_gib"], value["peak_reserved_gib"]
    )
    current["incremental_peak_allocated_gib"] = max(
        current["incremental_peak_allocated_gib"],
        value["incremental_peak_allocated_gib"],
    )


def _residual_measurement(
    total_cuda_ms: float,
    total_cpu_ms: float,
    component_cuda_ms: float,
    component_cpu_ms: float,
    device: torch.device,
) -> dict[str, float]:
    allocated = torch.cuda.memory_allocated(device) / GIB
    reserved = torch.cuda.memory_reserved(device) / GIB
    return {
        "cuda_ms": max(0.0, total_cuda_ms - component_cuda_ms),
        "cpu_ms": max(0.0, total_cpu_ms - component_cpu_ms),
        "allocated_before_gib": allocated,
        "allocated_after_gib": allocated,
        "reserved_before_gib": reserved,
        "reserved_after_gib": reserved,
        "peak_allocated_gib": allocated,
        "peak_reserved_gib": reserved,
        "incremental_peak_allocated_gib": 0.0,
    }


def _repeat_batched(value: Any, count: int) -> Any:
    """Repeat a detached one-sample prepared batch without rerunning teachers."""

    if torch.is_tensor(value):
        if value.dim() > 0 and value.shape[0] == 1:
            repeats = (int(count),) + (1,) * (value.dim() - 1)
            return value.repeat(repeats)
        return value.clone()
    if isinstance(value, dict):
        return {key: _repeat_batched(item, count) for key, item in value.items()}
    if isinstance(value, list):
        return value * int(count) if len(value) == 1 else list(value)
    if isinstance(value, tuple):
        return tuple(value)
    return value


def _prediction_from_stages(
    encoded: dict[str, torch.Tensor],
    projected: torch.Tensor,
    gaussian: dict[str, torch.Tensor],
    logical_batch: int,
) -> dict[str, torch.Tensor]:
    output = {
        "cls_tokens_by_view": encoded["cls_token"].view(logical_batch, 2, -1),
        "projected_cls_by_view": projected.view(logical_batch, 2, -1),
        "current_patch_tokens_by_view": encoded["current_patch_tokens"].view(
            logical_batch,
            2,
            encoded["current_patch_tokens"].shape[1],
            -1,
        ),
        "patch_masks_by_view": encoded["patch_mask"].view(logical_batch, 2, -1),
        "visible_patch_ids_by_view": encoded["visible_patch_ids"].view(
            logical_batch, 2, -1
        ),
    }
    output.update(gaussian)
    return output


def _component_row(
    mode: str,
    batch_size: int,
    component: str,
    measurements: list[dict[str, float]],
    total_wall_mean_ms: float,
) -> dict[str, Any]:
    cuda = _summary([value["cuda_ms"] for value in measurements])
    cpu = _summary([value["cpu_ms"] for value in measurements])
    return {
        "mode": mode,
        "logical_batch": batch_size,
        "component": component,
        "cuda_ms_mean": cuda["mean"],
        "cuda_ms_median": cuda["median"],
        "cuda_ms_p90": cuda["p90"],
        "cuda_ms_p95": cuda["p95"],
        "cpu_ms_mean": cpu["mean"],
        "percent_total": 100.0 * cuda["mean"] / max(total_wall_mean_ms, 1.0e-9),
        "allocated_before_gib": min(
            value["allocated_before_gib"] for value in measurements
        ),
        "allocated_after_gib": max(
            value["allocated_after_gib"] for value in measurements
        ),
        "peak_allocated_gib": max(
            value["peak_allocated_gib"] for value in measurements
        ),
        "peak_reserved_gib": max(
            value["peak_reserved_gib"] for value in measurements
        ),
        "incremental_peak_allocated_gib": max(
            value["incremental_peak_allocated_gib"] for value in measurements
        ),
    }


def _teacher_to(
    pipeline: OnlineTeacherPipeline,
    teacher_name: str,
    device: torch.device | str,
) -> None:
    teacher = getattr(pipeline, teacher_name)
    if teacher is None:
        raise RuntimeError(f"Teacher {teacher_name} is unavailable.")
    teacher.to(device)
    if str(device) == "cpu":
        torch.cuda.empty_cache()


def _profile_batch(
    *,
    mode: str,
    batch_size: int,
    cpu_batch: dict[str, Any] | None,
    fixed_prepared_cpu: dict[str, Any] | None,
    pipeline: OnlineTeacherPipeline,
    model: SplatterVAE,
    parameterization: WorldSpaceGaussianParameterization,
    splatter,
    train_config,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    warmup_iterations: int,
    iterations: int,
) -> dict[str, Any]:
    component_values: dict[str, list[dict[str, float]]] = {}
    total_values: list[float] = []
    resident_allocated = torch.cuda.memory_allocated(device) / GIB
    resident_reserved = torch.cuda.memory_reserved(device) / GIB

    if mode == TRAINABLE_ONLY and fixed_prepared_cpu is None:
        raise ValueError("Trainable-only profiling requires a prepared CPU batch.")
    if mode != TRAINABLE_ONLY and cpu_batch is None:
        raise ValueError("Online profiling requires a raw CPU batch.")

    def append_iteration(values: dict[str, dict[str, float]]) -> None:
        for name, measurement in values.items():
            component_values.setdefault(name, []).append(measurement)

    for iteration in range(warmup_iterations + iterations):
        measured = iteration >= warmup_iterations
        iteration_started = time.perf_counter()
        iteration_components: dict[str, dict[str, float]] = {}

        def run(name: str, callable_: Callable[[], Any]) -> Any:
            value, timing = _measure_cuda(device, callable_)
            _accumulate_measurement(iteration_components, name, timing)
            return value

        if mode == TRAINABLE_ONLY:
            prepared = run(
                "host_to_device", lambda: move_to_device(fixed_prepared_cpu, device)
            )
        else:
            raw_batch = run(
                "host_to_device", lambda: move_to_device(cpu_batch, device)
            )
            if mode == ONLINE_OFFLOAD:
                run(
                    "memfof_model_transfer_to_gpu",
                    lambda: _teacher_to(pipeline, "memfof", device),
                )
            memfof = run(
                "memfof_inference", lambda: pipeline.infer_memfof(raw_batch)
            )
            if mode == ONLINE_OFFLOAD:
                run(
                    "memfof_model_offload_to_cpu",
                    lambda: _teacher_to(pipeline, "memfof", "cpu"),
                )
            motion_maps = run(
                "memfof_motion_aggregation",
                lambda: pipeline.compute_motion_maps(raw_batch, memfof),
            )
            selections = run(
                "motion_crop_selection",
                lambda: pipeline.select_motion_crops(
                    raw_batch, memfof, motion_maps=motion_maps
                ),
            )
            if mode == ONLINE_OFFLOAD:
                run(
                    "xlens_model_transfer_to_gpu",
                    lambda: _teacher_to(pipeline, "xlens", device),
                )
            xlens_prepared = run(
                "xlens_preprocessing",
                lambda: pipeline.xlens.prepare_inputs(
                    raw_batch["raw_histories"],
                    raw_batch["raw_K"],
                    raw_batch["raw_c2w"],
                ),
            )
            xlens = run(
                "xlens_inference",
                lambda: pipeline.xlens.infer_prepared(xlens_prepared),
            )
            if mode == ONLINE_OFFLOAD:
                run(
                    "xlens_model_offload_to_cpu",
                    lambda: _teacher_to(pipeline, "xlens", "cpu"),
                )
            prepared = run(
                "image_depth_flow_transforms",
                lambda: pipeline.prepare_real_batch(
                    raw_batch,
                    memfof_output=memfof,
                    xlens_output=xlens,
                    selections=selections,
                ),
            )

            if mode == ONLINE_ON:
                pose_names = {
                    "target_pose_interpolation",
                    "target_pose_bounded_perturbation",
                    "target_pose_coverage_validation",
                    "target_pose_safety_checks",
                }
                target, pose_total = _measure_cuda(
                    device,
                    lambda: pipeline.sample_novel_target(
                        prepared,
                        seed=17_003 + iteration,
                        stage_runner=run,
                    ),
                )
                pose_component_cuda = sum(
                    iteration_components.get(name, {}).get("cuda_ms", 0.0)
                    for name in pose_names
                )
                pose_component_cpu = sum(
                    iteration_components.get(name, {}).get("cpu_ms", 0.0)
                    for name in pose_names
                )
                _accumulate_measurement(
                    iteration_components,
                    "target_pose_resampling_control",
                    _residual_measurement(
                        pose_total["cuda_ms"],
                        pose_total["cpu_ms"],
                        pose_component_cuda,
                        pose_component_cpu,
                        device,
                    ),
                )
                lager_prepared = run(
                    "lagernvs_preprocessing",
                    lambda: pipeline.prepare_lagernvs_inputs(prepared, target),
                )
                teacher = run(
                    "lagernvs_inference",
                    lambda: pipeline.lagernvs.infer_prepared(lager_prepared),
                )
                prepared = pipeline.merge_novel_view(prepared, target, teacher)

        histories = prepared["representation_histories"].flatten(0, 1)
        flows = prepared["representation_flows"].flatten(0, 1)
        validity = prepared["representation_validity"].flatten(0, 1)

        def encode() -> dict[str, torch.Tensor]:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return model.encoder(
                    histories,
                    optical_flows=flows,
                    image_validity=validity,
                    apply_mask=True,
                )

        encoded = run("vit_s_encoder", encode)

        def project() -> torch.Tensor:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return model.contrastive_projector(encoded["cls_token"])

        projected = run("contrastive_head", project)

        def decode() -> dict[str, torch.Tensor]:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return model.gaussian_decoder(encoded["decoder_tokens"][0::2])

        gaussian = run("gaussian_decoder_and_heads", decode)
        prediction = _prediction_from_stages(encoded, projected, gaussian, batch_size)
        contrastive = run(
            "contrastive_loss",
            lambda: cross_view_info_nce(
                projected.view(batch_size, 2, -1),
                train_config.contrastive_temperature,
            )[0],
        )
        reconstruction = compute_droid_reconstruction(
            parameterization,
            splatter,
            prediction,
            prepared,
            train_config,
            motion_translation_max=model.motion_translation_max,
            background_color=torch.zeros(3, device=device),
            return_renders=False,
            novel_view_enabled=mode == ONLINE_ON,
            stage_runner=run,
        )
        total = reconstruction["loss"] + train_config.contrastive_weight * contrastive
        run("backward", total.backward)
        run("optimizer", optimizer.step)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        if measured:
            total_values.append((time.perf_counter() - iteration_started) * 1000.0)
            append_iteration(iteration_components)
        prepared = histories = flows = validity = None
        encoded = projected = gaussian = prediction = None
        contrastive = reconstruction = total = None
        if mode != TRAINABLE_ONLY:
            raw_batch = memfof = motion_maps = selections = None
            xlens_prepared = xlens = None
        if mode == ONLINE_ON:
            target = lager_prepared = teacher = None

    total_summary = _summary(total_values)
    rows = [
        _component_row(
            mode, batch_size, name, measurements, total_summary["mean"]
        )
        for name, measurements in component_values.items()
    ]
    peak_allocated = max(row["peak_allocated_gib"] for row in rows)
    peak_reserved = max(row["peak_reserved_gib"] for row in rows)
    component_cuda_sum = sum(row["cuda_ms_mean"] for row in rows)
    return {
        "mode": mode,
        "logical_batch": batch_size,
        "status": "success",
        "total_iteration_wall_ms": total_summary,
        "component_cuda_ms_sum": component_cuda_sum,
        "unattributed_wall_ms_mean": max(
            0.0, total_summary["mean"] - component_cuda_sum
        ),
        "iterations_per_second": 1000.0 / total_summary["mean"],
        "logical_samples_per_second": 1000.0 * batch_size / total_summary["mean"],
        "effective_frames_per_second": 1000.0 * batch_size * 6 / total_summary["mean"],
        "resident_allocated_gib": resident_allocated,
        "resident_reserved_gib": resident_reserved,
        "integrated_peak_allocated_gib": peak_allocated,
        "integrated_peak_reserved_gib": peak_reserved,
        "incremental_activation_peak_gib": max(
            0.0, peak_allocated - resident_allocated
        ),
        "components": rows,
    }


def _profile_target_pose_stages(
    *,
    pipeline: OnlineTeacherPipeline,
    prepared_one_cpu: dict[str, Any],
    batch_size: int,
    device: torch.device,
    warmup_iterations: int,
    iterations: int,
) -> dict[str, Any]:
    prepared = move_to_device(_repeat_batched(prepared_one_cpu, batch_size), device)
    measurements: dict[str, list[dict[str, float]]] = {}
    total_timings: list[dict[str, float]] = []
    last_target = None
    for iteration in range(warmup_iterations + iterations):
        measured = iteration >= warmup_iterations
        per_iteration: dict[str, dict[str, float]] = {}

        def run(name: str, callable_: Callable[[], Any]) -> Any:
            value, timing = _measure_cuda(device, callable_)
            _accumulate_measurement(per_iteration, name, timing)
            return value

        target, total = _measure_cuda(
            device,
            lambda: pipeline.sample_novel_target(
                prepared, seed=91_007 + iteration, stage_runner=run
            ),
        )
        component_cuda = sum(item["cuda_ms"] for item in per_iteration.values())
        component_cpu = sum(item["cpu_ms"] for item in per_iteration.values())
        _accumulate_measurement(
            per_iteration,
            "target_pose_resampling_control",
            _residual_measurement(
                total["cuda_ms"],
                total["cpu_ms"],
                component_cuda,
                component_cpu,
                device,
            ),
        )
        if measured:
            total_timings.append(total)
            for name, timing in per_iteration.items():
                measurements.setdefault(name, []).append(timing)
        last_target = target
    assert last_target is not None
    total_summary = _summary([item["cuda_ms"] for item in total_timings])
    rows = [
        _component_row(
            "target_pose_only", batch_size, name, values, total_summary["mean"]
        )
        for name, values in measurements.items()
    ]
    return {
        "status": "success",
        "logical_batch": batch_size,
        "total_cuda_ms": total_summary,
        "ms_per_target_mean": total_summary["mean"] / batch_size,
        "coverage_mean": float(last_target["source_coverage"].float().mean()),
        "coverage_min": float(last_target["source_coverage"].float().amin()),
        "rejected_candidates": int(last_target["rejected_candidates"].sum()),
        "fallback_count": int(last_target["fallback_used"].sum()),
        "components": rows,
    }


def _profile_lagernvs_microbatches(
    *,
    pipeline: OnlineTeacherPipeline,
    prepared_one_cpu: dict[str, Any],
    logical_batch: int,
    microbatch_sizes: list[int],
    device: torch.device,
    warmup_iterations: int,
    iterations: int,
) -> list[dict[str, Any]]:
    batch = move_to_device(_repeat_batched(prepared_one_cpu, logical_batch), device)
    target = pipeline.sample_novel_target(batch, seed=701)
    prepared, preprocessing = _measure_cuda(
        device, lambda: pipeline.prepare_lagernvs_inputs(batch, target)
    )
    results = []
    for microbatch in microbatch_sizes:
        pipeline.lagernvs.microbatch_size = int(microbatch)
        _cleanup_allocator(device)
        values = []
        try:
            for iteration in range(warmup_iterations + iterations):
                output, timing = _measure_cuda(
                    device, lambda: pipeline.lagernvs.infer_prepared(prepared)
                )
                if iteration >= warmup_iterations:
                    values.append(timing)
                del output
            latency = _summary([item["cuda_ms"] for item in values])
            results.append(
                {
                    "status": "success",
                    "logical_batch": logical_batch,
                    "microbatch": microbatch,
                    "preprocessing_cuda_ms": preprocessing["cuda_ms"],
                    "inference_cuda_ms": latency,
                    "ms_per_target_mean": latency["mean"] / logical_batch,
                    "targets_per_second": 1000.0 * logical_batch / latency["mean"],
                    "peak_allocated_gib": max(
                        item["peak_allocated_gib"] for item in values
                    ),
                    "peak_reserved_gib": max(
                        item["peak_reserved_gib"] for item in values
                    ),
                }
            )
        except torch.OutOfMemoryError as exc:
            results.append(
                {
                    "status": "out_of_memory",
                    "logical_batch": logical_batch,
                    "microbatch": microbatch,
                    "reason": str(exc),
                }
            )
            _cleanup_allocator(device)
    return results


def _profile_or_oom(callable_: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        return callable_()
    except torch.OutOfMemoryError as exc:
        return {"status": "out_of_memory", "reason": str(exc)}


def _memory_recommendations(
    profiles: list[dict[str, Any]],
    batch_sizes: list[int],
    total_memory_gib: float,
    safety_fraction: float,
) -> dict[str, Any]:
    recommendations = {}
    modes = sorted(
        {profile.get("mode") for profile in profiles if profile.get("mode")}
    )
    for mode in modes:
        candidates = [profile for profile in profiles if profile.get("mode") == mode]
        successful = [profile for profile in candidates if profile["status"] == "success"]
        failed = [profile for profile in candidates if profile["status"] != "success"]
        if not successful:
            recommendations[mode] = {
                "maximum_tested_batch": max(batch_sizes),
                "maximum_successful_batch": None,
                "recommended_safe_batch": None,
            }
            continue
        safe = [
            profile
            for profile in successful
            if max(
                profile["integrated_peak_allocated_gib"],
                profile["integrated_peak_reserved_gib"],
            )
            <= safety_fraction * total_memory_gib
        ]
        throughput_reference = max(
            profile["logical_samples_per_second"] for profile in safe or successful
        )
        practical = [
            profile
            for profile in safe
            if profile["logical_samples_per_second"] >= 0.95 * throughput_reference
        ]
        recommended = max(
            practical or safe or successful,
            key=lambda profile: profile["logical_batch"],
        )
        maximum_success = max(profile["logical_batch"] for profile in successful)
        larger_failures = sorted(
            profile["logical_batch"]
            for profile in failed
            if profile.get("logical_batch", 0) > maximum_success
        )
        accumulation = math.ceil(256 / recommended["logical_batch"])
        recommendations[mode] = {
            "maximum_tested_batch": max(
                profile.get("logical_batch", 0) for profile in candidates
            ),
            "maximum_successful_batch": maximum_success,
            "technical_fit_boundary": (
                [maximum_success, larger_failures[0]] if larger_failures else None
            ),
            "recommended_safe_batch": recommended["logical_batch"],
            "recommended_gradient_accumulation_for_effective_256": accumulation,
            "resulting_effective_batch_world_size_1": (
                recommended["logical_batch"] * accumulation
            ),
            "memory_safety_fraction": safety_fraction,
            "recommendation_policy": (
                "largest batch within memory limit and 95% of best safe throughput"
            ),
        }
    return recommendations


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    _validate_fixed_pipeline_contract(config)
    requested_lagernvs = args.mode in ("on", "both")
    config["novel_view"]["enabled"] = False
    if args.workspace_stats:
        statistics_payload = json.loads(
            Path(args.workspace_stats).read_text(encoding="utf-8")
        )
        proposal = statistics_payload["proposed_parameters"]
        config["decoder"].update(
            {
                "global_center": proposal["global_center"],
                "anchor_initial_spread": proposal["anchor_initial_spread"],
                "parent_displacement_scale": proposal["parent_displacement_scale"],
                "child_radius": proposal["child_radius"],
            }
        )
        config["renderer"].update(
            {"znear": proposal["znear"], "zfar": proposal["zfar"]}
        )

    droid_root = Path(config["dataset"]["droid_root"]).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve(strict=False)
    validate_derived_root(output_root, droid_root).mkdir(parents=True, exist_ok=True)
    source_before = source_tree_fingerprint(droid_root)
    batch_sizes = _parse_positive_integers(args.batch_sizes, "Batch sizes")
    microbatch_sizes = _parse_positive_integers(
        args.lagernvs_microbatch_sizes, "LagerNVS microbatch sizes"
    )
    if not 0.0 < float(args.memory_safety_fraction) < 1.0:
        raise ValueError("Memory safety fraction must lie between zero and one.")
    device = torch.device("cuda:0")
    gpu = verify_cuda_device()
    total_memory_gib = torch.cuda.get_device_properties(device).total_memory / GIB

    cold_started = time.perf_counter()
    allocated_initial = torch.cuda.memory_allocated(device)
    model, splatter = _model_and_renderer(config)
    model.to(device).train()
    allocated_after_model = torch.cuda.memory_allocated(device)
    pipeline = _online_preprocessor(config, device)
    torch.cuda.synchronize(device)
    allocated_after_base_teachers = torch.cuda.memory_allocated(device)
    cold_seconds = time.perf_counter() - cold_started
    parameterization = WorldSpaceGaussianParameterization(splatter).to(device)
    train_config = _train_config(config, None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-5, fused=True)

    pipeline.memfof.to("cpu")
    pipeline.xlens.to("cpu")
    _cleanup_allocator(device)
    resident_model_only = torch.cuda.memory_allocated(device)
    pipeline.memfof.to(device)
    torch.cuda.synchronize(device)
    memfof_resident_delta = torch.cuda.memory_allocated(device) - resident_model_only
    pipeline.memfof.to("cpu")
    _cleanup_allocator(device)
    pipeline.xlens.to(device)
    torch.cuda.synchronize(device)
    xlens_resident_delta = torch.cuda.memory_allocated(device) - resident_model_only
    pipeline.memfof.to(device)
    torch.cuda.synchronize(device)
    all_base_resident = torch.cuda.memory_allocated(device)

    dataset_started = time.perf_counter()
    dataset = DROIDLogicalDataset(_dataset_config(config, "train"))
    dataset_initialization_ms = (time.perf_counter() - dataset_started) * 1000.0
    decode_started = time.perf_counter()
    item = dataset[0]
    decode_cold_ms = (time.perf_counter() - decode_started) * 1000.0
    decode_started = time.perf_counter()
    second_item = dataset[1]
    decode_cached_ms = (time.perf_counter() - decode_started) * 1000.0
    collate_started = time.perf_counter()
    _ = droid_collate([item, second_item])
    collate_two_samples_ms = (time.perf_counter() - collate_started) * 1000.0

    raw_one = move_to_device(droid_collate([item]), device)
    with torch.inference_mode():
        memfof_one = pipeline.infer_memfof(raw_one)
        xlens_one = pipeline.infer_xlens(raw_one)
        prepared_one = pipeline.prepare_real_batch(
            raw_one, memfof_output=memfof_one, xlens_output=xlens_one
        )
    prepared_one_cpu = _detach_to_cpu(prepared_one)
    del raw_one, memfof_one, xlens_one, prepared_one
    pipeline.memfof.to("cpu")
    pipeline.xlens.to("cpu")
    _cleanup_allocator(device)

    profiles: list[dict[str, Any]] = []
    for batch_size in batch_sizes:
        fixed_cpu = _repeat_batched(prepared_one_cpu, batch_size)
        _cleanup_allocator(device)
        profile = _profile_or_oom(
            lambda batch_size=batch_size, fixed_cpu=fixed_cpu: _profile_batch(
                mode=TRAINABLE_ONLY,
                batch_size=batch_size,
                cpu_batch=None,
                fixed_prepared_cpu=fixed_cpu,
                pipeline=pipeline,
                model=model,
                parameterization=parameterization,
                splatter=splatter,
                train_config=train_config,
                optimizer=optimizer,
                device=device,
                warmup_iterations=int(args.warmup_iterations),
                iterations=int(args.iterations),
            )
        )
        profile.update({"mode": TRAINABLE_ONLY, "logical_batch": batch_size})
        profiles.append(profile)
        del fixed_cpu
        optimizer.zero_grad(set_to_none=True)
        _cleanup_allocator(device)
        if not args.quiet:
            print(json.dumps(profile, indent=2), flush=True)

    pipeline.memfof.to(device)
    pipeline.xlens.to(device)
    _cleanup_allocator(device)
    if args.mode in ("off", "both"):
        for batch_size in batch_sizes:
            cpu_batch = droid_collate([item] * batch_size)
            _cleanup_allocator(device)
            profile = _profile_or_oom(
                lambda batch_size=batch_size, cpu_batch=cpu_batch: _profile_batch(
                    mode=ONLINE_OFF,
                    batch_size=batch_size,
                    cpu_batch=cpu_batch,
                    fixed_prepared_cpu=None,
                    pipeline=pipeline,
                    model=model,
                    parameterization=parameterization,
                    splatter=splatter,
                    train_config=train_config,
                    optimizer=optimizer,
                    device=device,
                    warmup_iterations=int(args.warmup_iterations),
                    iterations=int(args.iterations),
                )
            )
            profile.update({"mode": ONLINE_OFF, "logical_batch": batch_size})
            profiles.append(profile)
            optimizer.zero_grad(set_to_none=True)
            _cleanup_allocator(device)
            if not args.quiet:
                print(json.dumps(profile, indent=2), flush=True)

    residency_experiment: dict[str, Any] = {
        "requested": not args.skip_offload_experiment
    }
    if not args.skip_offload_experiment and args.mode in ("off", "both"):
        offload_batch = int(args.offload_batch_size)
        pipeline.memfof.to("cpu")
        pipeline.xlens.to("cpu")
        _cleanup_allocator(device)
        cpu_batch = droid_collate([item] * offload_batch)
        offload_profile = _profile_or_oom(
            lambda: _profile_batch(
                mode=ONLINE_OFFLOAD,
                batch_size=offload_batch,
                cpu_batch=cpu_batch,
                fixed_prepared_cpu=None,
                pipeline=pipeline,
                model=model,
                parameterization=parameterization,
                splatter=splatter,
                train_config=train_config,
                optimizer=optimizer,
                device=device,
                warmup_iterations=int(args.warmup_iterations),
                iterations=int(args.iterations),
            )
        )
        offload_profile.update(
            {"mode": ONLINE_OFFLOAD, "logical_batch": offload_batch}
        )
        profiles.append(offload_profile)
        matching = next(
            (
                profile
                for profile in profiles
                if profile.get("mode") == ONLINE_OFF
                and profile.get("logical_batch") == offload_batch
                and profile.get("status") == "success"
            ),
            None,
        )
        residency_experiment.update(
            {
                "logical_batch": offload_batch,
                "all_resident": matching,
                "sequential_offload": offload_profile,
            }
        )
        if matching is not None and offload_profile.get("status") == "success":
            residency_experiment.update(
                {
                    "peak_allocated_vram_saved_gib": matching[
                        "integrated_peak_allocated_gib"
                    ]
                    - offload_profile["integrated_peak_allocated_gib"],
                    "latency_penalty_percent": 100.0
                    * (
                        offload_profile["total_iteration_wall_ms"]["mean"]
                        / matching["total_iteration_wall_ms"]["mean"]
                        - 1.0
                    ),
                }
            )
        pipeline.memfof.to(device)
        pipeline.xlens.to(device)
        _cleanup_allocator(device)

    novel = config["novel_view"]
    pose_values = dict(novel["target_pose"])
    pose_values.pop("mode", None)
    pose_values.pop("scene_center", None)
    pipeline.novel_pose_config = LagerTargetPoseConfig(
        **_only_dataclass_fields(LagerTargetPoseConfig, pose_values)
    )
    configured_center = novel["target_pose"].get("scene_center")
    pipeline.scene_center = tuple(
        float(value)
        for value in (
            configured_center
            if configured_center is not None
            else config["decoder"]["global_center"]
        )
    )
    target_pose_profile = None
    if requested_lagernvs:
        _cleanup_allocator(device)
        target_pose_profile = _profile_target_pose_stages(
            pipeline=pipeline,
            prepared_one_cpu=prepared_one_cpu,
            batch_size=min(max(batch_sizes), 8),
            device=device,
            warmup_iterations=int(args.warmup_iterations),
            iterations=int(args.iterations),
        )
        _cleanup_allocator(device)

    lagernvs_status: dict[str, Any] = {
        "requested": requested_lagernvs,
        "checkpoint_id": novel["checkpoint_id"],
        "checkpoint_revision": novel["checkpoint_revision"],
        "repository_revision": novel["official_repository_revision"],
        # ``hf auth login`` stores the token in the Hugging Face cache; it need
        # not be duplicated in HF_TOKEN for hf_hub_download to authenticate.
        "hf_auth_available": bool(get_token()),
    }
    lagernvs_microbatch_profiles: list[dict[str, Any]] = []
    if requested_lagernvs:
        try:
            started = time.perf_counter()
            lager = LagerNVSDROIDTeacher(
                novel["official_repo_path"],
                novel.get("checkpoint_path"),
                cache_dir=novel.get("cache_dir"),
                device=device,
                dtype=torch.bfloat16,
                microbatch_size=int(novel["microbatch_size"]),
                canonical_focal_px=float(novel["canonical"]["focal_px"]),
            )
            torch.cuda.synchronize(device)
            pipeline.lagernvs = lager
            lager_resident_delta = (
                torch.cuda.memory_allocated(device) - all_base_resident
            ) / GIB
            lagernvs_status.update(
                {
                    "status": "loaded",
                    "cold_load_seconds": time.perf_counter() - started,
                    "resident_model_memory_gib": lager_resident_delta,
                    "checkpoint_path": lager.checkpoint_path,
                }
            )
            for batch_size in batch_sizes:
                cpu_batch = droid_collate([item] * batch_size)
                _cleanup_allocator(device)
                profile = _profile_or_oom(
                    lambda batch_size=batch_size, cpu_batch=cpu_batch: _profile_batch(
                        mode=ONLINE_ON,
                        batch_size=batch_size,
                        cpu_batch=cpu_batch,
                        fixed_prepared_cpu=None,
                        pipeline=pipeline,
                        model=model,
                        parameterization=parameterization,
                        splatter=splatter,
                        train_config=train_config,
                        optimizer=optimizer,
                        device=device,
                        warmup_iterations=int(args.warmup_iterations),
                        iterations=int(args.iterations),
                    )
                )
                profile.update({"mode": ONLINE_ON, "logical_batch": batch_size})
                profiles.append(profile)
                optimizer.zero_grad(set_to_none=True)
                _cleanup_allocator(device)
                if not args.quiet:
                    print(json.dumps(profile, indent=2), flush=True)
            lagernvs_microbatch_profiles = _profile_lagernvs_microbatches(
                pipeline=pipeline,
                prepared_one_cpu=prepared_one_cpu,
                logical_batch=min(max(batch_sizes), 8),
                microbatch_sizes=microbatch_sizes,
                device=device,
                warmup_iterations=int(args.warmup_iterations),
                iterations=int(args.iterations),
            )
        except Exception as exc:  # noqa: BLE001 - preserve exact external blocker
            lagernvs_status.update(
                {
                    "status": "blocked",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "blocked_components": [
                        "lagernvs_inference",
                        "lagernvs_on_every_iteration_profile",
                        "lagernvs_vram_and_microbatch_scaling",
                        "lagernvs_offload_comparison",
                    ],
                }
            )
            _cleanup_allocator(device)

    recommendations = _memory_recommendations(
        profiles,
        batch_sizes,
        total_memory_gib,
        float(args.memory_safety_fraction),
    )
    source_after = source_tree_fingerprint(droid_root)
    if source_before != source_after:
        raise RuntimeError("The read-only DROID source changed during profiling.")
    successful = [profile for profile in profiles if profile["status"] == "success"]
    payload = {
        "schema_version": 2,
        "gpu": gpu,
        "gpu_total_memory_gib": total_memory_gib,
        "dataset_root": str(droid_root),
        "dataset_read_only": True,
        "source_fingerprint": source_after,
        "cold_model_and_base_teacher_load_seconds": cold_seconds,
        "data_and_calibration": {
            "dataset_and_calibration_index_initialization_ms": dataset_initialization_ms,
            "droid_decode_cold_ms": decode_cold_ms,
            "droid_decode_same_episode_cached_ms": decode_cached_ms,
            "collate_two_samples_ms": collate_two_samples_ms,
            "note": "DataLoader workers prefetch decode; steady-state GPU tables exclude asynchronous decode.",
        },
        "resident_model_memory": {
            "initial_allocated_gib": allocated_initial / GIB,
            "trainable_model_delta_gib": (
                allocated_after_model - allocated_initial
            )
            / GIB,
            "memfof_model_delta_gib": memfof_resident_delta / GIB,
            "xlens_model_delta_gib": xlens_resident_delta / GIB,
            "memfof_plus_xlens_observed_delta_gib": (
                allocated_after_base_teachers - allocated_after_model
            )
            / GIB,
            "all_base_models_resident_allocated_gib": all_base_resident / GIB,
        },
        "warmup_iterations": int(args.warmup_iterations),
        "measured_iterations": int(args.iterations),
        "profiles": profiles,
        "target_pose_profile": target_pose_profile,
        "lagernvs": lagernvs_status,
        "lagernvs_microbatch_profiles": lagernvs_microbatch_profiles,
        "residency_experiment": residency_experiment,
        "batch_recommendations": recommendations,
        "effective_batch_contract": {
            "logical_batch": "per-rank number of two-camera, three-frame samples",
            "frames_per_logical_sample": 6,
            "world_size_profiled": 1,
            "target_effective_optimization_batch": 256,
        },
    }
    (output_root / "profile.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    component_rows = [
        component
        for profile in successful
        for component in profile.get("components", [])
    ]
    if target_pose_profile is not None:
        component_rows.extend(target_pose_profile.get("components", []))
    if component_rows:
        fields = sorted({key for row in component_rows for key in row})
        with (output_root / "components.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(component_rows)
    if args.quiet:
        print(
            json.dumps(
                {
                    "profile": str(output_root / "profile.json"),
                    "component_table": str(output_root / "components.csv"),
                    "profile_count": len(profiles),
                    "lagernvs_status": lagernvs_status.get("status", "not_requested"),
                },
                indent=2,
            ),
            flush=True,
        )
    else:
        print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
