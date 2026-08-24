from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch.utils.data import DistributedSampler

from .transforms import LOCAL_CROP_SIZE, RLDS_WIDTH


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
class LocalCropConfig:
    motion_probability: float = 0.50
    random_probability: float = 0.25
    center_probability: float = 0.25
    crop_size: int = LOCAL_CROP_SIZE

    def __post_init__(self) -> None:
        probabilities = (
            self.motion_probability,
            self.random_probability,
            self.center_probability,
        )
        if any(value < 0 for value in probabilities):
            raise ValueError("Crop probabilities must be non-negative.")
        if abs(sum(probabilities) - 1.0) > 1.0e-6:
            raise ValueError("Crop probabilities must sum to one.")
        if self.crop_size != LOCAL_CROP_SIZE:
            raise ValueError(
                f"The initial DROID local view is fixed at {LOCAL_CROP_SIZE}x{LOCAL_CROP_SIZE}."
            )


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


def _motion_crop_x(flow_magnitude: torch.Tensor, crop_size: int) -> int:
    if flow_magnitude.dim() == 3:
        score = flow_magnitude.amax(dim=0)
    elif flow_magnitude.dim() == 2:
        score = flow_magnitude
    else:
        raise ValueError(
            "Motion crop selection expects (pairs,H,W) or (H,W) flow magnitude."
        )
    width = int(score.shape[-1])
    if crop_size >= width:
        return 0
    column_score = torch.nan_to_num(score.float(), nan=0.0).sum(dim=-2)
    window = torch.ones(1, 1, crop_size, device=score.device)
    totals = torch.nn.functional.conv1d(column_score.view(1, 1, width), window)[0, 0]
    return int(totals.argmax().item())


def choose_local_crop_x(
    flow_magnitude: torch.Tensor,
    config: LocalCropConfig,
    rng: random.Random,
) -> tuple[int, str]:
    maximum_x = RLDS_WIDTH - int(config.crop_size)
    draw = rng.random()
    if draw < config.motion_probability:
        return _motion_crop_x(flow_magnitude, int(config.crop_size)), "motion"
    if draw < config.motion_probability + config.random_probability:
        return rng.randint(0, maximum_x), "random"
    return maximum_x // 2, "center"


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
