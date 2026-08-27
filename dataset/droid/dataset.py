from __future__ import annotations

import bisect
import multiprocessing as mp
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .calibration import load_calibration_manifest
from .rlds import EpisodeBackend, TFDSRLDSBackend
from .sampling import (
    MotionCropConfig,
    TemporalSamplingConfig,
    history_indices,
    randomize_camera_order,
    sample_temporal_stride,
    sample_uniform_crop_size,
)
from .transforms import RLDS_HEIGHT, RLDS_WIDTH

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class DROIDDatasetConfig:
    droid_root: str = "/home/ws/data/droid"
    calibration_manifest: str = (
        "/ws/data/ws/droid_splattervae/manifests/calibration.jsonl.gz"
    )
    split: str = "train"
    seed: int = 42
    temporal: TemporalSamplingConfig = field(default_factory=TemporalSamplingConfig)
    motion_crop: MotionCropConfig = field(default_factory=MotionCropConfig)

    def __post_init__(self) -> None:
        if self.split not in ("train", "validation"):
            raise ValueError("DROID split must be 'train' or 'validation'.")


def _chw_rgb(image: np.ndarray) -> torch.Tensor:
    value = torch.as_tensor(np.asarray(image))
    if value.shape == (RLDS_HEIGHT, RLDS_WIDTH, 3):
        value = value.permute(2, 0, 1)
    if value.shape != (3, RLDS_HEIGHT, RLDS_WIDTH):
        raise ValueError(f"Expected one RLDS RGB image, got {tuple(value.shape)}.")
    if value.dtype != torch.uint8:
        if not value.dtype.is_floating_point:
            raise TypeError(f"Unsupported RGB dtype {value.dtype}.")
        value = (value.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
    return value.contiguous()


def normalize_encoder_rgb(
    rgb: torch.Tensor,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> torch.Tensor:
    if rgb.shape[-3] != 3:
        raise ValueError("Encoder RGB must have three channels.")
    value = rgb.float() / 255.0 if rgb.dtype == torch.uint8 else rgb.float()
    mean_tensor = value.new_tensor(tuple(mean)).view(
        *((1,) * (value.dim() - 3)), 3, 1, 1
    )
    std_tensor = value.new_tensor(tuple(std)).view(
        *((1,) * (value.dim() - 3)), 3, 1, 1
    )
    return (value - mean_tensor) / std_tensor


class DROIDLogicalDataset(Dataset[dict[str, Any]]):
    """One physical history with two synchronized calibrated exterior views.

    The loader performs only read-only RLDS decoding, temporal sampling, and
    calibration lookup. Frozen X-Lens and MEMFOF teachers, motion-centered
    cropping, and all modality transforms run online on the training device.
    No semantic segmentation or persistent neural-teacher cache is part of the
    batch contract.
    """

    def __init__(
        self,
        config: DROIDDatasetConfig,
        *,
        backend: EpisodeBackend | None = None,
        manifest_entries: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        self.config = config
        loaded = (
            load_calibration_manifest(config.calibration_manifest)
            if manifest_entries is None
            else [dict(entry) for entry in manifest_entries]
        )
        self.entries = [
            dict(entry)
            for entry in loaded
            if bool(entry.get("valid")) and entry.get("dataset_split") == config.split
        ]
        if not self.entries:
            raise ValueError(
                f"No calibration-valid DROID episodes exist in split {config.split!r}."
            )
        self.backend = backend or TFDSRLDSBackend(config.droid_root)
        # The training DataLoader uses the spawn context so CUDA state is never
        # inherited by RLDS decoder workers.  Its shared epoch counter must be
        # created by that same context; mixing a fork-created SemLock with
        # spawned/forkserver workers caused a native worker crash on Linux.
        self._epoch = mp.get_context("spawn").Value("q", 0, lock=True)
        self._first_current: list[int] = []
        self._cumulative: list[int] = []
        count = 0
        for entry in self.entries:
            stride = (
                config.temporal.validation_stride
                if config.split == "validation"
                else min(config.temporal.strides)
            )
            first = 2 * int(stride)
            available = max(0, int(entry["num_steps"]) - first)
            self._first_current.append(first)
            count += available
            self._cumulative.append(count)
        if count == 0:
            raise ValueError(
                "Calibration-valid episodes contain no complete temporal histories."
            )

    def set_epoch(self, epoch: int) -> None:
        with self._epoch.get_lock():
            self._epoch.value = int(epoch)

    @property
    def epoch(self) -> int:
        return int(self._epoch.value)

    def __len__(self) -> int:
        return self._cumulative[-1]

    @property
    def episode_index_ranges(self) -> tuple[tuple[int, int], ...]:
        starts = (0, *self._cumulative[:-1])
        return tuple(
            (int(start), int(stop))
            for start, stop in zip(starts, self._cumulative, strict=True)
        )

    def sample_episode_index(self, index: int) -> int:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return bisect.bisect_right(self._cumulative, index)

    def _resolve_index(self, index: int) -> tuple[int, int]:
        episode_index = self.sample_episode_index(index)
        start = 0 if episode_index == 0 else self._cumulative[episode_index - 1]
        current = self._first_current[episode_index] + (index - start)
        return episode_index, int(current)

    def _rng(self, index: int) -> random.Random:
        split_offset = 0 if self.config.split == "train" else 9_223_372_036_854_775
        return random.Random(
            int(self.config.seed)
            + split_offset
            + self.epoch * 1_000_000_007
            + int(index) * 65_537
        )

    @staticmethod
    def _camera_history(
        images: np.ndarray, indices: Sequence[int], camera_index: int
    ) -> torch.Tensor:
        return torch.stack([_chw_rgb(images[frame, camera_index]) for frame in indices])

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode_index, current = self._resolve_index(index)
        entry = self.entries[episode_index]
        rng = self._rng(index)
        if self.config.split == "validation":
            stride = int(self.config.temporal.validation_stride)
        else:
            valid_strides = [
                value for value in self.config.temporal.strides if current >= 2 * value
            ]
            stride = sample_temporal_stride(valid_strides, self.config.temporal, rng)
        indices = history_indices(current, stride)
        episode = self.backend.get_episode(
            str(entry["rlds_split"]), int(entry["rlds_ordinal"])
        )
        images = np.asarray(episode["images"])
        expected = (int(entry["num_steps"]), 2, RLDS_HEIGHT, RLDS_WIDTH, 3)
        if images.shape != expected:
            raise ValueError(
                f"RLDS image array disagrees with manifest: {images.shape} != {expected}."
            )
        camera_a, camera_b = randomize_camera_order(rng)
        camera_indices = (camera_a, camera_b)
        cameras = [entry["exterior_cameras"][value] for value in camera_indices]
        histories = torch.stack(
            [self._camera_history(images, indices, value) for value in camera_indices]
        )
        output: dict[str, Any] = {
            "raw_histories": histories,
            "raw_K": torch.stack(
                [
                    torch.as_tensor(camera["intrinsics_rlds"], dtype=torch.float32)
                    for camera in cameras
                ]
            ),
            "raw_c2w": torch.stack(
                [
                    torch.as_tensor(camera["c2w"], dtype=torch.float32)
                    for camera in cameras
                ]
            ),
            "raw_w2c": torch.stack(
                [
                    torch.as_tensor(camera["w2c"], dtype=torch.float32)
                    for camera in cameras
                ]
            ),
            "sampled_crop_size": torch.tensor(
                sample_uniform_crop_size(self.config.motion_crop, rng),
                dtype=torch.long,
            ),
            "calibration_validity": torch.tensor(True),
            "current_timestep": torch.tensor(current, dtype=torch.long),
            "history_indices": torch.tensor(indices, dtype=torch.long),
            "temporal_stride": torch.tensor(stride, dtype=torch.long),
            "camera_order": torch.tensor(camera_indices, dtype=torch.long),
            "camera_serials": tuple(str(camera["serial"]) for camera in cameras),
            "episode_id": str(entry["episode_id"]),
        }
        for key in ("action", "action_dict"):
            if key not in episode:
                continue
            value = episode[key]
            if isinstance(value, Mapping):
                output[key] = {
                    name: torch.as_tensor(np.asarray(component)[current])
                    for name, component in value.items()
                }
            else:
                output[key] = torch.as_tensor(np.asarray(value)[current])
        observation_state = {}
        for key in ("cartesian_position", "gripper_position", "joint_position"):
            if key in episode:
                observation_state[key] = torch.as_tensor(
                    np.asarray(episode[key])[current]
                )
        if observation_state:
            output["robot_state"] = observation_state
        return output


def droid_collate(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Default-style recursive collate that preserves string metadata cleanly."""

    from torch.utils.data._utils.collate import default_collate

    tensor_values = {
        key: [item[key] for item in batch]
        for key in batch[0]
        if not isinstance(batch[0][key], (str, tuple))
    }
    output = {key: default_collate(values) for key, values in tensor_values.items()}
    for key in batch[0]:
        if isinstance(batch[0][key], (str, tuple)):
            output[key] = [item[key] for item in batch]
    return output
