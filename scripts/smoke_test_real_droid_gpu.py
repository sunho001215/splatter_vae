from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from dataset.droid.dataset import DROIDLogicalDataset, droid_collate
from dataset.droid.safety import source_tree_fingerprint, validate_derived_root
from dataset.droid.sampling import sample_uniform_crop_size
from models.gaussian.parameterization import WorldSpaceGaussianParameterization
from models.training.distributed import move_to_device
from models.training.losses import cross_view_info_nce
from models.training.reconstruction import compute_droid_reconstruction
from models.training.validation import _detach_to_cpu
from models.training.visualization import save_droid_validation_visualization
from preprocessing.common import verify_cuda_device
from preprocessing.lagernvs.camera import canonical_intrinsics
from preprocessing.lagernvs.pose import LagerTargetPoseConfig, sample_safe_target_poses
from scripts.train_droid import (
    _dataset_config,
    _model_and_renderer,
    _online_preprocessor,
    _only_dataclass_fields,
    _train_config,
    _validate_fixed_pipeline_contract,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run real-DROID online MEMFOF/X-Lens preprocessing, Gaussian rendering, "
            "and forward/backward validation."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workspace-stats", default=None)
    parser.add_argument("--no-backward", action="store_true")
    return parser.parse_args()


def _cuda_time(callable_):
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    value = callable_()
    end.record()
    end.synchronize()
    return value, float(begin.elapsed_time(end))


def _selected_indices(dataset: DROIDLogicalDataset, count: int) -> list[int]:
    ranges = dataset.episode_index_ranges
    if not ranges:
        raise ValueError("DROID dataset exposes no calibrated episode ranges.")
    episode_indices = np.linspace(
        0, len(ranges) - 1, num=min(int(count), len(ranges)), dtype=np.int64
    )
    return [int(ranges[index][0]) for index in episode_indices]


def _crop_argmax_checks(batch: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    metadata = batch["crop_metadata"]
    for sample in range(batch["raw_histories"].shape[0]):
        for camera in range(2):
            size = int(metadata["crop_size"][sample, camera])
            half_left = size // 2
            half_right = size - half_left
            maximum = 320 - half_right
            motion = batch["smoothed_motion"][sample, camera]
            feasible = motion[
                half_left : maximum + 1, half_left : maximum + 1
            ]
            index = int(feasible.reshape(-1).argmax())
            expected_y = half_left + index // feasible.shape[1]
            expected_x = half_left + index % feasible.shape[1]
            fallback = bool(
                metadata["low_motion_fallback_used"][sample, camera]
            )
            actual_x = int(metadata["crop_center_x"][sample, camera])
            actual_y = int(metadata["crop_center_y"][sample, camera])
            matches = (
                (actual_x, actual_y) == (160, 160)
                if fallback
                else (actual_x, actual_y) == (expected_x, expected_y)
            )
            if not matches:
                raise AssertionError("A real crop center does not match motion argmax.")
            rows.append(
                {
                    "sample": sample,
                    "camera": camera,
                    "crop_size": size,
                    "center": [actual_x, actual_y],
                    "expected_motion_argmax": [expected_x, expected_y],
                    "flow_peak": float(metadata["flow_peak_value"][sample, camera]),
                    "fallback": fallback,
                    "matches": matches,
                }
            )
    return rows


def _uniform_crop_audit(config, seed: int, draws: int = 100_000):
    rng = random.Random(int(seed))
    sizes = np.asarray(
        [sample_uniform_crop_size(config, rng) for _ in range(draws)],
        dtype=np.int64,
    )
    counts = np.bincount(sizes - config.min_size, minlength=141)
    expected = draws / len(counts)
    chi_square = float(np.sum((counts - expected) ** 2 / expected))
    return {
        "draws": draws,
        "minimum": int(sizes.min()),
        "maximum": int(sizes.max()),
        "mean": float(sizes.mean()),
        "expected_mean": 250.0,
        "counts_180_to_320": counts.tolist(),
        "maximum_relative_bin_deviation": float(
            np.max(np.abs(counts - expected) / expected)
        ),
        "chi_square_140_dof": chi_square,
    }


def _flow_direction_audit(
    histories: torch.Tensor, flows: torch.Tensor
) -> list[dict[str, float | int | str]]:
    """Check the official middle->neighbor convention by photometric warping."""

    images = histories.float() / 255.0
    batch, views, _times, _channels, height, width = images.shape
    middle = images[:, :, 1].reshape(batch * views, 3, height, width)
    u = torch.arange(width, device=images.device, dtype=torch.float32)
    v = torch.arange(height, device=images.device, dtype=torch.float32)
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    base = torch.stack((uu, vv), dim=-1)[None]
    rows = []
    for direction, (neighbor_index, name) in enumerate(
        ((0, "middle_to_previous"), (2, "middle_to_next"))
    ):
        neighbor = images[:, :, neighbor_index].reshape(
            batch * views, 3, height, width
        )
        flow = flows[:, :, direction].reshape(batch * views, 2, height, width)
        for sign, label in ((1.0, "official_plus_flow"), (-1.0, "opposite_sign")):
            coords = base + sign * flow.permute(0, 2, 3, 1)
            inside = (
                (coords[..., 0] >= 0.0)
                & (coords[..., 0] <= width - 1.0)
                & (coords[..., 1] >= 0.0)
                & (coords[..., 1] <= height - 1.0)
            )
            grid = coords.clone()
            grid[..., 0] = 2.0 * grid[..., 0] / (width - 1.0) - 1.0
            grid[..., 1] = 2.0 * grid[..., 1] / (height - 1.0) - 1.0
            warped = F.grid_sample(
                neighbor,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            residual = (warped - middle).abs().mean(dim=1)
            rows.append(
                {
                    "direction_index": direction,
                    "direction": name,
                    "warp_convention": label,
                    "photometric_l1": float(residual[inside].mean()),
                    "in_bounds_fraction": float(inside.float().mean()),
                }
            )
        zero_residual = (neighbor - middle).abs().mean(dim=1)
        rows.append(
            {
                "direction_index": direction,
                "direction": name,
                "warp_convention": "zero_flow_baseline",
                "photometric_l1": float(zero_residual.mean()),
                "in_bounds_fraction": 1.0,
            }
        )
    return rows


def _memfof_quality_assessment(
    photometric_rows: list[dict[str, float | int | str]],
    identical_flow: torch.Tensor,
) -> dict[str, Any]:
    """Turn real and identical-frame diagnostics into an explicit pass/fail."""

    direction_checks = []
    for direction in ("middle_to_previous", "middle_to_next"):
        rows = [row for row in photometric_rows if row["direction"] == direction]
        official = next(
            float(row["photometric_l1"])
            for row in rows
            if row["warp_convention"] == "official_plus_flow"
        )
        zero = next(
            float(row["photometric_l1"])
            for row in rows
            if row["warp_convention"] == "zero_flow_baseline"
        )
        ratio = official / max(zero, 1.0e-8)
        direction_checks.append(
            {
                "direction": direction,
                "official_warp_l1": official,
                "zero_flow_l1": zero,
                "official_to_zero_error_ratio": ratio,
                "pass": ratio <= 1.25,
                "criterion": "official warp error <= 1.25 * zero-flow error",
            }
        )
    identical_magnitude = torch.linalg.vector_norm(
        identical_flow.float(), dim=3
    )
    identical = {
        "magnitude_mean_px": float(identical_magnitude.mean()),
        "magnitude_p95_px": float(torch.quantile(identical_magnitude, 0.95)),
    }
    identical["pass"] = (
        identical["magnitude_mean_px"] <= 1.0
        and identical["magnitude_p95_px"] <= 2.0
    )
    identical["criterion"] = "identical-frame mean <= 1 px and p95 <= 2 px"
    passed = bool(all(row["pass"] for row in direction_checks) and identical["pass"])
    return {
        "status": "passed" if passed else "failed_real_photometric_sanity",
        "pass": passed,
        "direction_contract": "confirmed_from_pinned_official_source",
        "direction_checks": direction_checks,
        "identical_frame_check": identical,
        "interpretation": (
            "The mandated native-resolution, exactly-two-iteration teacher is "
            "not reliable enough for supervision on this audit cohort."
            if not passed
            else "The teacher passed the configured real and identical-frame checks."
        ),
    }


def main() -> None:
    args = parse_args()
    values = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    _validate_fixed_pipeline_contract(values)
    values["novel_view"]["enabled"] = False
    droid_root = Path(values["dataset"]["droid_root"]).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve(strict=False)
    validate_derived_root(output_root, droid_root).mkdir(parents=True, exist_ok=True)
    source_before = source_tree_fingerprint(droid_root)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly the authorized GPU before this smoke test.")
    device = torch.device("cuda:0")
    diagnostics = verify_cuda_device()

    # Import/load PyTorch teacher kernels before TensorFlow initializes the
    # in-process RLDS decoder.  Importing Triton after TensorFlow is a known
    # native-runtime conflict in this environment; the training entrypoint uses
    # this same safe initialization order.
    model, splatter = _model_and_renderer(values)
    model.to(device).train(not args.no_backward)
    parameterization = WorldSpaceGaussianParameterization(splatter).to(device)
    train_config = _train_config(values, None)
    background = torch.zeros(3, device=device)
    pipeline = _online_preprocessor(values, device)

    dataset_config = _dataset_config(values, "train")
    dataset = DROIDLogicalDataset(dataset_config)
    indices = _selected_indices(dataset, args.samples)
    decode_started = time.perf_counter()
    items = [dataset[index] for index in indices]
    decode_ms = (time.perf_counter() - decode_started) * 1000.0
    cpu_batch = droid_collate(items)
    raw_batch = move_to_device(cpu_batch, device)

    torch.cuda.reset_peak_memory_stats(device)
    memfof, memfof_ms = _cuda_time(lambda: pipeline.infer_memfof(raw_batch))
    identical_histories = raw_batch["raw_histories"][:1, :, 1:2].expand(
        -1, -1, 3, -1, -1, -1
    )
    identical_memfof, identical_memfof_ms = _cuda_time(
        lambda: pipeline.memfof(identical_histories)
    )
    xlens, xlens_ms = _cuda_time(lambda: pipeline.infer_xlens(raw_batch))
    batch, transforms_ms = _cuda_time(
        lambda: pipeline.prepare_real_batch(
            raw_batch, memfof_output=memfof, xlens_output=xlens
        )
    )
    crop_checks = _crop_argmax_checks(batch)

    pose_values = dict(values["novel_view"]["target_pose"])
    pose_values.pop("mode", None)
    pose_values.pop("scene_center", None)
    pose_config = LagerTargetPoseConfig(
        **_only_dataclass_fields(LagerTargetPoseConfig, pose_values)
    )
    scene_center = values["decoder"]["global_center"]
    if args.workspace_stats:
        statistics = json.loads(Path(args.workspace_stats).read_text(encoding="utf-8"))
        scene_center = statistics["proposed_parameters"]["global_center"]
    canonical_K = canonical_intrinsics(
        (len(items),),
        focal_px=float(values["novel_view"]["canonical"]["focal_px"]),
        device=device,
    )
    target_pose, pose_sampling_ms = _cuda_time(
        lambda: sample_safe_target_poses(
            batch["raw_c2w"],
            batch["raw_K"],
            canonical_K,
            batch["native_xlens_depth"][:, 2],
            batch["native_xlens_confidence"][:, 2],
            batch["native_xlens_validity"][:, 2],
            torch.tensor(scene_center, dtype=torch.float32, device=device),
            pose_config,
            seed=int(args.seed),
        )
    )

    def model_forward():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = model(
                batch["representation_histories"],
                batch["representation_flows"],
                batch["representation_validity"],
            )
            contrastive, contrast_metrics = cross_view_info_nce(
                prediction["projected_cls_by_view"],
                train_config.contrastive_temperature,
            )
        return prediction, contrastive, contrast_metrics

    (prediction, contrastive, contrast_metrics), model_ms = _cuda_time(model_forward)
    reconstruction, rendering_loss_ms = _cuda_time(
        lambda: compute_droid_reconstruction(
            parameterization,
            splatter,
            prediction,
            batch,
            train_config,
            motion_translation_max=model.motion_translation_max,
            background_color=background,
            return_renders=True,
            novel_view_enabled=False,
        )
    )
    total = reconstruction["loss"] + train_config.contrastive_weight * contrastive
    if not torch.isfinite(total):
        raise FloatingPointError(f"Real-DROID loss is non-finite: {total}")
    backward_ms = 0.0
    gradients = {}
    if not args.no_backward:
        _, backward_ms = _cuda_time(total.backward)
        gradients = {
            "encoder": any(
                parameter.grad is not None for parameter in model.encoder.parameters()
            ),
            "gaussian_decoder": any(
                parameter.grad is not None
                for parameter in model.gaussian_decoder.parameters()
            ),
            "contrastive_projector": any(
                parameter.grad is not None
                for parameter in model.contrastive_projector.parameters()
            ),
        }
        if not all(gradients.values()):
            raise RuntimeError(f"Real-DROID smoke test missed gradients: {gradients}")

    detached_payload = {
        "batch": _detach_to_cpu(batch),
        "reconstruction": _detach_to_cpu(reconstruction),
        "prediction": _detach_to_cpu(prediction),
    }
    visualization = save_droid_validation_visualization(
        output_root / "visualization",
        0,
        detached_payload,
        num_samples=len(items),
        depth_range_m=tuple(
            float(value) for value in values["logging"]["depth_display_range_m"]
        ),
    )
    source_after = source_tree_fingerprint(droid_root)
    if source_before != source_after:
        raise RuntimeError("The read-only DROID source changed during smoke testing.")

    pose_metadata = {
        key: value.detach().cpu().tolist() if torch.is_tensor(value) else value
        for key, value in target_pose.items()
        if key not in {"target_c2w", "target_w2c", "base_c2w", "support_mask"}
    }
    valid_depth = batch["native_xlens_depth"][batch["native_xlens_validity"]]
    valid_confidence = batch["native_xlens_confidence"][
        batch["native_xlens_validity"]
    ]
    flow_magnitude = torch.linalg.vector_norm(
        batch["native_memfof_flow"].float(), dim=3
    )
    photometric_audit = _flow_direction_audit(
        batch["raw_histories"], batch["native_memfof_flow"]
    )
    result = {
        "gpu": diagnostics,
        "dataset_root": str(droid_root),
        "dataset_read_only": True,
        "source_fingerprint": source_after,
        "dataset_indices": indices,
        "episode_ids": list(cpu_batch["episode_id"]),
        "history_indices": cpu_batch["history_indices"].tolist(),
        "raw_rgb_shape": list(batch["raw_histories"].shape),
        "model_input_shape": list(batch["representation_histories"].shape),
        "native_memfof_flow_shape": list(batch["native_memfof_flow"].shape),
        "native_xlens_depth_shape": list(batch["native_xlens_depth"].shape),
        "transformed_K": batch["target_K"][:, 2].detach().cpu().tolist(),
        "crop_argmax_checks": crop_checks,
        "crop_uniformity": _uniform_crop_audit(dataset_config.motion_crop, args.seed),
        "memfof": {
            "directions": ["middle_to_previous", "middle_to_next"],
            "iterations": 2,
            "finite_fraction": float(
                torch.isfinite(batch["native_memfof_flow"]).float().mean()
            ),
            "valid_fraction": float(batch["native_memfof_validity"].float().mean()),
            "magnitude_mean": float(flow_magnitude.mean()),
            "magnitude_p95": float(torch.quantile(flow_magnitude, 0.95)),
            "photometric_direction_audit": photometric_audit,
            "quality_assessment": _memfof_quality_assessment(
                photometric_audit, identical_memfof["flow"]
            ),
        },
        "xlens": {
            "valid_fraction": float(batch["native_xlens_validity"].float().mean()),
            "depth_min_m": float(valid_depth.amin()),
            "depth_median_m": float(valid_depth.median()),
            "depth_p95_m": float(torch.quantile(valid_depth, 0.95)),
            "depth_max_m": float(valid_depth.amax()),
            "confidence_median": float(valid_confidence.median()),
        },
        "target_pose_sampling_without_lagernvs_inference": pose_metadata,
        "loss": float(total.detach()),
        "contrastive_loss": float(contrastive.detach()),
        "positive_cosine_similarity": float(
            contrast_metrics["positive_cosine_similarity"].detach()
        ),
        "rgb_l1_loss": float(reconstruction["rgb_l1_loss"].detach()),
        "flow_loss": float(reconstruction["flow_loss"].detach()),
        "metric_depth_loss": float(reconstruction["metric_depth_loss"].detach()),
        "scale_invariant_depth_loss": float(
            reconstruction["scale_invariant_depth_loss"].detach()
        ),
        "rendered_visible_pixel_fraction": float(
            reconstruction["rendered_visible_pixel_fraction"].detach()
        ),
        "out_of_frustum_fraction": float(
            reconstruction["out_of_frustum_fraction"].detach()
        ),
        "gradients": gradients,
        "parameter_counts": model.parameter_counts(),
        "timing_ms": {
            "droid_decode": decode_ms,
            "memfof": memfof_ms,
            "memfof_identical_frame_diagnostic": identical_memfof_ms,
            "xlens": xlens_ms,
            "transforms_and_crop": transforms_ms,
            "target_pose_sampling_and_coverage": pose_sampling_ms,
            "model_forward": model_ms,
            "rendering_and_losses": rendering_loss_ms,
            "backward": backward_ms,
        },
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
        "peak_reserved_gpu_memory_gib": torch.cuda.max_memory_reserved(device)
        / (1024**3),
        "visualization": {key: str(path) for key, path in visualization.items()},
    }
    result_path = output_root / "result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
