from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DistributedSampler

from .transforms import (
    MODEL_SIZE,
    PAD_BOTTOM,
    PAD_TOP,
    PADDED_SIZE,
    RLDS_HEIGHT,
    RLDS_WIDTH,
    SpatialTransform,
    motion_crop_transform,
)


@dataclass(frozen=True)
class TemporalSamplingConfig:
    strides: tuple[int, ...] = (1, 3, 6)
    probabilities: tuple[float, ...] = (0.40, 0.35, 0.25)
    validation_stride: int = 3

    def __post_init__(self) -> None:
        if not self.strides or len(self.strides) != len(self.probabilities):
            raise ValueError(
                "Temporal strides and probabilities must be non-empty and aligned."
            )
        if any(int(value) <= 0 for value in self.strides):
            raise ValueError("Temporal strides must be positive.")
        if any(float(value) < 0 for value in self.probabilities):
            raise ValueError("Temporal probabilities must be non-negative.")
        if abs(sum(self.probabilities) - 1.0) > 1.0e-6:
            raise ValueError("Temporal probabilities must sum to one.")
        if self.validation_stride not in self.strides:
            raise ValueError("Validation stride must be one of the configured strides.")


@dataclass(frozen=True)
class MotionCropConfig:
    """Unified variable-FOV crop used by every DROID representation view."""

    min_size: int = 180
    max_size: int = 320
    size_sampling: str = "uniform"
    padded_size: int = PADDED_SIZE
    pad_top: int = PAD_TOP
    pad_bottom: int = PAD_BOTTOM
    output_size: int = MODEL_SIZE
    center_mode: str = "optical_flow_argmax"
    flow_aggregation: str = "max"
    flow_smoothing_kernel: int = 15
    low_motion_threshold: float = 1.0e-4
    low_motion_fallback: str = "image_center"

    def __post_init__(self) -> None:
        if self.size_sampling != "uniform":
            raise ValueError("DROID crop-size sampling must be uniform.")
        if self.center_mode != "optical_flow_argmax":
            raise ValueError("DROID crop centers must use the optical-flow argmax.")
        if self.flow_aggregation not in ("max", "mean", "sum"):
            raise ValueError("Flow aggregation must be max, mean, or sum.")
        if self.low_motion_fallback != "image_center":
            raise ValueError("The only degenerate low-motion fallback is image_center.")
        if self.min_size <= 0 or self.max_size < self.min_size:
            raise ValueError("Crop sizes must define a non-empty positive interval.")
        if self.max_size > self.padded_size:
            raise ValueError("Maximum crop size exceeds the padded canvas.")
        if self.pad_top + RLDS_HEIGHT + self.pad_bottom != self.padded_size:
            raise ValueError("Configured vertical padding does not produce the canvas.")
        if RLDS_WIDTH != self.padded_size:
            raise ValueError("DROID padding must not alter the RLDS image width.")
        if self.flow_smoothing_kernel <= 0 or self.flow_smoothing_kernel % 2 == 0:
            raise ValueError("Flow smoothing kernel must be a positive odd integer.")
        if self.low_motion_threshold < 0:
            raise ValueError("Low-motion threshold must be non-negative.")


@dataclass(frozen=True)
class MotionCropMetadata:
    crop_size: int
    crop_x0: int
    crop_y0: int
    crop_center_x: int
    crop_center_y: int
    resize_scale: float
    real_pixel_fraction: float
    flow_peak_value: float
    selected_crop_flow_mean: float
    low_motion_fallback_used: bool


@dataclass(frozen=True)
class MotionCropSelection:
    transform: SpatialTransform
    metadata: MotionCropMetadata
    aggregate_motion_map: torch.Tensor
    smoothed_motion_map: torch.Tensor


def history_indices(current_timestep: int, stride: int) -> tuple[int, int, int]:
    current = int(current_timestep)
    step = int(stride)
    if step <= 0:
        raise ValueError("Temporal stride must be positive.")
    indices = (current - 2 * step, current - step, current)
    if indices[0] < 0:
        raise ValueError(
            f"Current timestep {current} has insufficient history for stride {step}."
        )
    return indices


def sample_temporal_stride(
    valid_strides: Sequence[int],
    config: TemporalSamplingConfig,
    rng: random.Random,
) -> int:
    valid = {int(value) for value in valid_strides}
    choices = [stride for stride in config.strides if stride in valid]
    if not choices:
        raise ValueError("No configured temporal stride is valid for this timestep.")
    weights = [config.probabilities[config.strides.index(stride)] for stride in choices]
    total = sum(weights)
    return int(
        rng.choices(choices, weights=[value / total for value in weights], k=1)[0]
    )


def sample_uniform_crop_size(config: MotionCropConfig, rng: random.Random) -> int:
    """Sample every integer in [min_size,max_size] with equal probability."""
    return int(rng.randint(int(config.min_size), int(config.max_size)))


def aggregate_flow_magnitude(
    flows: torch.Tensor,
    validity: torch.Tensor | None = None,
    *,
    aggregation: str = "max",
) -> torch.Tensor:
    """Aggregate MEMFOF backward/forward flow on the native 180x320 grid."""
    if flows.dim() != 4 or flows.shape[1] != 2:
        raise ValueError(f"Expected flow (pairs,2,H,W), got {tuple(flows.shape)}.")
    if tuple(flows.shape[-2:]) != (RLDS_HEIGHT, RLDS_WIDTH):
        raise ValueError("Flow crop selection requires the original RLDS grid.")
    magnitude = torch.linalg.vector_norm(flows.float(), dim=1)
    magnitude = torch.nan_to_num(magnitude, nan=0.0, posinf=0.0, neginf=0.0)
    if validity is not None:
        mask = validity.bool()
        if mask.dim() == 4 and mask.shape[1] == 1:
            mask = mask[:, 0]
        if mask.shape != magnitude.shape:
            raise ValueError(
                f"Flow validity {tuple(mask.shape)} does not match {tuple(magnitude.shape)}."
            )
        magnitude = magnitude.masked_fill(~mask, 0.0)
    if aggregation == "max":
        return magnitude.amax(dim=0)
    if aggregation == "mean":
        return magnitude.mean(dim=0)
    if aggregation == "sum":
        return magnitude.sum(dim=0)
    raise ValueError(f"Unknown flow aggregation {aggregation!r}.")


def _pad_motion_map(score: torch.Tensor, config: MotionCropConfig) -> torch.Tensor:
    if score.shape != (RLDS_HEIGHT, RLDS_WIDTH):
        raise ValueError(f"Expected motion map (180,320), got {tuple(score.shape)}.")
    return F.pad(score[None, None], (0, 0, config.pad_top, config.pad_bottom))[0, 0]


def smooth_motion_map(score: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if score.dim() != 2:
        raise ValueError("Motion smoothing expects a two-dimensional map.")
    kernel = int(kernel_size)
    if kernel <= 0 or kernel % 2 == 0:
        raise ValueError("Motion smoothing kernel must be a positive odd integer.")
    return F.avg_pool2d(
        score[None, None],
        kernel_size=kernel,
        stride=1,
        padding=kernel // 2,
    )[0, 0]


def _real_pixel_fraction(transform: SpatialTransform) -> float:
    source_top = transform.pad_top
    source_bottom = transform.pad_top + transform.source_height
    crop_top = transform.crop_y
    crop_bottom = transform.crop_y + transform.crop_size
    vertical_overlap = max(0, min(source_bottom, crop_bottom) - max(source_top, crop_top))
    source_left = transform.pad_left
    source_right = transform.pad_left + transform.source_width
    crop_left = transform.crop_x
    crop_right = transform.crop_x + transform.crop_size
    horizontal_overlap = max(
        0, min(source_right, crop_right) - max(source_left, crop_left)
    )
    return float(vertical_overlap * horizontal_overlap) / float(transform.crop_size**2)


def build_motion_maps(
    flows: torch.Tensor,
    validity: torch.Tensor | None,
    config: MotionCropConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate, pad, and smooth MEMFOF motion before crop selection.

    Keeping this stage separate makes the online profiler able to distinguish
    flow post-processing from the small feasible-argmax crop-selection step.
    Both returned maps use the padded ``320 x 320`` image grid.
    """

    aggregate = aggregate_flow_magnitude(
        flows, validity, aggregation=config.flow_aggregation
    )
    padded = _pad_motion_map(aggregate, config)
    smoothed = smooth_motion_map(padded, config.flow_smoothing_kernel)
    return padded, smoothed


def select_motion_crop_from_maps(
    aggregate_motion_map: torch.Tensor,
    smoothed_motion_map: torch.Tensor,
    crop_size: int,
    config: MotionCropConfig,
) -> MotionCropSelection:
    """Select the highest-motion feasible crop from precomputed motion maps."""

    expected = (config.padded_size, config.padded_size)
    if aggregate_motion_map.shape != expected or smoothed_motion_map.shape != expected:
        raise ValueError(
            "Motion maps must use the padded square grid; got "
            f"{tuple(aggregate_motion_map.shape)} and "
            f"{tuple(smoothed_motion_map.shape)}."
        )
    size = int(crop_size)
    if size < config.min_size or size > config.max_size:
        raise ValueError(f"Crop size {size} is outside the configured interval.")

    half_left = size // 2
    half_right = size - half_left
    minimum = half_left
    maximum = config.padded_size - half_right
    feasible = smoothed_motion_map[
        minimum : maximum + 1, minimum : maximum + 1
    ]
    if feasible.numel() == 0:
        raise RuntimeError("No feasible center exists for the sampled crop size.")
    flat_index = int(feasible.reshape(-1).argmax().item())
    feasible_width = int(feasible.shape[1])
    center_y = minimum + flat_index // feasible_width
    center_x = minimum + flat_index % feasible_width
    peak = float(feasible.reshape(-1)[flat_index].item())
    fallback = peak <= float(config.low_motion_threshold)
    if fallback:
        center_x = center_y = config.padded_size // 2

    transform = motion_crop_transform(
        size,
        center_x,
        center_y,
        padded_size=config.padded_size,
        pad_top=config.pad_top,
        pad_bottom=config.pad_bottom,
        output_size=config.output_size,
    )
    selected = aggregate_motion_map[
        transform.crop_y : transform.crop_y + size,
        transform.crop_x : transform.crop_x + size,
    ]
    metadata = MotionCropMetadata(
        crop_size=size,
        crop_x0=transform.crop_x,
        crop_y0=transform.crop_y,
        crop_center_x=center_x,
        crop_center_y=center_y,
        resize_scale=transform.resize_scale,
        real_pixel_fraction=_real_pixel_fraction(transform),
        flow_peak_value=peak,
        selected_crop_flow_mean=float(selected.mean().item()),
        low_motion_fallback_used=fallback,
    )
    return MotionCropSelection(
        transform,
        metadata,
        aggregate_motion_map,
        smoothed_motion_map,
    )


def select_motion_crop(
    flows: torch.Tensor,
    validity: torch.Tensor | None,
    crop_size: int,
    config: MotionCropConfig,
) -> MotionCropSelection:
    """Select the deterministic highest-flow feasible center for one camera."""
    aggregate, smoothed = build_motion_maps(flows, validity, config)
    return select_motion_crop_from_maps(aggregate, smoothed, crop_size, config)


def randomize_camera_order(rng: random.Random) -> tuple[int, int]:
    return (0, 1) if rng.random() < 0.5 else (1, 0)


class EpisodeGroupedDistributedSampler(DistributedSampler):
    """DistributedSampler that keeps shuffled frame indices grouped by episode.

    DROID TFDS episodes are expensive to materialize. The ordinary shuffled
    sampler scatters adjacent requests across the entire dataset and defeats
    each worker's episode cache. This variant still gives every rank the same
    sample count, supports ``set_epoch``, and pads/drops exactly like
    ``DistributedSampler``, while making episode decoding sequential at scale.
    """

    def __iter__(self):
        ranges = getattr(self.dataset, "episode_index_ranges", None)
        if ranges is None:
            return super().__iter__()
        generator = torch.Generator()
        generator.manual_seed(int(self.seed) + int(self.epoch))
        episode_order = list(range(len(ranges)))
        if self.shuffle:
            episode_order = torch.randperm(len(ranges), generator=generator).tolist()
        indices: list[int] = []
        for episode_index in episode_order:
            start, stop = ranges[episode_index]
            block = torch.arange(int(start), int(stop), dtype=torch.long)
            if self.shuffle and len(block) > 1:
                block = block[torch.randperm(len(block), generator=generator)]
            indices.extend(block.tolist())
        if len(indices) != len(self.dataset):
            raise RuntimeError(
                "Episode index ranges do not cover the DROID dataset exactly."
            )
        if self.drop_last:
            indices = indices[: self.total_size]
        elif len(indices) < self.total_size:
            padding = self.total_size - len(indices)
            repeats = (padding + len(indices) - 1) // len(indices)
            indices += (indices * repeats)[:padding]
        if len(indices) != self.total_size:
            raise RuntimeError(
                "Distributed DROID sampler produced an invalid global sample count."
            )
        start = self.rank * self.num_samples
        rank_indices = indices[start : start + self.num_samples]
        if len(rank_indices) != self.num_samples:
            raise RuntimeError(
                "Distributed DROID sampler produced an invalid rank sample count."
            )
        return iter(rank_indices)
