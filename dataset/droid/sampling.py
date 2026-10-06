from __future__ import annotations

import random
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
        if self.flow_aggregation != "max":
            raise ValueError("Aligned F01/F12 motion maps must use max aggregation.")
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


def sample_uniform_crop_size(config: MotionCropConfig, rng: random.Random) -> int:
    """Sample every integer in [min_size,max_size] with equal probability."""
    return int(rng.randint(int(config.min_size), int(config.max_size)))


def forward_splat_flow_magnitude(
    flow: torch.Tensor,
    validity: torch.Tensor | None = None,
    *,
    fill_holes: bool = True,
) -> torch.Tensor:
    """Move a forward-flow magnitude from its source grid to its target grid.

    Nearest-pixel max splatting preserves thin high-motion structures without
    averaging them into the background. A single 3x3 max fill closes obvious
    rasterization holes before the later configured motion smoothing.
    """

    if flow.shape != (2, RLDS_HEIGHT, RLDS_WIDTH):
        raise ValueError(
            f"Expected one native flow (2,180,320), got {tuple(flow.shape)}."
        )
    finite = torch.isfinite(flow).all(dim=0)
    if validity is not None:
        mask = validity.bool()
        if mask.shape == (1, RLDS_HEIGHT, RLDS_WIDTH):
            mask = mask[0]
        if mask.shape != (RLDS_HEIGHT, RLDS_WIDTH):
            raise ValueError("Forward-splat validity must be (1,180,320) or (180,320).")
        finite &= mask
    y, x = torch.meshgrid(
        torch.arange(RLDS_HEIGHT, device=flow.device),
        torch.arange(RLDS_WIDTH, device=flow.device),
        indexing="ij",
    )
    destination_x = torch.round(x.float() + flow[0].float()).long()
    destination_y = torch.round(y.float() + flow[1].float()).long()
    usable = (
        finite
        & (destination_x >= 0)
        & (destination_x < RLDS_WIDTH)
        & (destination_y >= 0)
        & (destination_y < RLDS_HEIGHT)
    )
    output = torch.zeros(
        RLDS_HEIGHT * RLDS_WIDTH, device=flow.device, dtype=torch.float32
    )
    if usable.any():
        linear = destination_y[usable] * RLDS_WIDTH + destination_x[usable]
        magnitude = torch.linalg.vector_norm(flow.float(), dim=0)[usable]
        output.scatter_reduce_(0, linear, magnitude, reduce="amax", include_self=True)
    output = output.reshape(RLDS_HEIGHT, RLDS_WIDTH)
    if fill_holes and usable.any():
        occupancy = torch.zeros_like(output, dtype=torch.bool)
        occupancy.reshape(-1).scatter_(0, linear, True)
        neighborhood = F.max_pool2d(output[None, None], 3, 1, 1)[0, 0]
        neighbor_occupancy = F.max_pool2d(occupancy[None, None].float(), 3, 1, 1)[
            0, 0
        ].bool()
        output = torch.where(~occupancy & neighbor_occupancy, neighborhood, output)
    return output


def middle_frame_motion_map(
    flow_01: torch.Tensor,
    validity_01: torch.Tensor | None,
    flow_12: torch.Tensor,
    validity_12: torch.Tensor | None,
) -> torch.Tensor:
    """Combine F01 and F12 only after moving F01 onto the t1 source grid."""

    warped_01 = forward_splat_flow_magnitude(flow_01, validity_01)
    if flow_12.shape != (2, RLDS_HEIGHT, RLDS_WIDTH):
        raise ValueError("F12 must use the native (2,180,320) middle-frame grid.")
    direct_12 = torch.nan_to_num(
        torch.linalg.vector_norm(flow_12.float(), dim=0),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if validity_12 is not None:
        valid_12 = validity_12.bool()
        if valid_12.shape == (1, RLDS_HEIGHT, RLDS_WIDTH):
            valid_12 = valid_12[0]
        if valid_12.shape != (RLDS_HEIGHT, RLDS_WIDTH):
            raise ValueError("F12 validity must be (1,180,320) or (180,320).")
        direct_12 = direct_12.masked_fill(~valid_12, 0.0)
    return torch.maximum(warped_01, direct_12)


def build_middle_frame_motion_maps(
    flow_01: torch.Tensor,
    validity_01: torch.Tensor | None,
    flow_12: torch.Tensor,
    validity_12: torch.Tensor | None,
    config: MotionCropConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    native = middle_frame_motion_map(flow_01, validity_01, flow_12, validity_12)
    padded = _pad_motion_map(native, config)
    return padded, smooth_motion_map(padded, config.flow_smoothing_kernel)


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
    vertical_overlap = max(
        0, min(source_bottom, crop_bottom) - max(source_top, crop_top)
    )
    source_left = transform.pad_left
    source_right = transform.pad_left + transform.source_width
    crop_left = transform.crop_x
    crop_right = transform.crop_x + transform.crop_size
    horizontal_overlap = max(
        0, min(source_right, crop_right) - max(source_left, crop_left)
    )
    return float(vertical_overlap * horizontal_overlap) / float(transform.crop_size**2)


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
    feasible = smoothed_motion_map[minimum : maximum + 1, minimum : maximum + 1]
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


class EpisodeGroupedDistributedSampler(DistributedSampler):
    """Keep windows grouped by episode so indexed-shard reads retain locality."""

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
