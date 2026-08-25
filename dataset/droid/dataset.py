from __future__ import annotations

import bisect
import multiprocessing as mp
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .cache import HDF5CacheReader, calibration_manifest_version
from .calibration import load_calibration_manifest
from .rlds import EpisodeBackend, TFDSRLDSBackend
from .sampling import (
    MotionCropConfig,
    TemporalSamplingConfig,
    history_indices,
    randomize_camera_order,
    sample_temporal_stride,
    sample_uniform_crop_size,
    select_motion_crop,
)
from .transforms import (
    IMAGENET_NEUTRAL_RGB,
    MODEL_SIZE,
    RLDS_HEIGHT,
    RLDS_WIDTH,
    SpatialTransform,
    image_validity_mask,
    pad_to_square,
    transform_confidence,
    transform_depth,
    transform_flow,
    transform_intrinsics,
    transform_rgb,
    transform_validity,
)

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
    normalize_mean: tuple[float, float, float] = IMAGENET_MEAN
    normalize_std: tuple[float, float, float] = IMAGENET_STD
    rgb_padding_value: tuple[float, float, float] = IMAGENET_NEUTRAL_RGB
    require_depth_cache: bool = True
    require_flow_cache: bool = True
    include_crop_debug: bool = False

    def __post_init__(self) -> None:
        if self.split not in ("train", "validation"):
            raise ValueError("DROID split must be 'train' or 'validation'.")
        if len(self.normalize_mean) != 3 or len(self.normalize_std) != 3:
            raise ValueError("RGB normalization requires three-channel mean and std.")
        if any(float(value) <= 0.0 for value in self.normalize_std):
            raise ValueError("RGB normalization standard deviations must be positive.")
        if len(self.rgb_padding_value) != 3:
            raise ValueError("RGB padding requires three channel values.")


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


def _one_channel(value: np.ndarray, *, name: str, dtype: torch.dtype) -> torch.Tensor:
    tensor = torch.as_tensor(np.asarray(value), dtype=dtype)
    if tensor.shape == (RLDS_HEIGHT, RLDS_WIDTH):
        tensor = tensor[None]
    elif tensor.shape == (RLDS_HEIGHT, RLDS_WIDTH, 1):
        tensor = tensor.permute(2, 0, 1)
    if tensor.shape != (1, RLDS_HEIGHT, RLDS_WIDTH):
        raise ValueError(
            f"Expected {name} on the 180x320 grid, got {tuple(tensor.shape)}."
        )
    return tensor.contiguous()


def _flow_chw(value: np.ndarray) -> torch.Tensor:
    tensor = torch.as_tensor(np.asarray(value), dtype=torch.float32)
    if tensor.shape == (RLDS_HEIGHT, RLDS_WIDTH, 2):
        tensor = tensor.permute(2, 0, 1)
    if tensor.shape != (2, RLDS_HEIGHT, RLDS_WIDTH):
        raise ValueError(
            f"Expected WAFT flow on the 180x320 grid, got {tuple(tensor.shape)}."
        )
    return tensor.contiguous()


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
    std_tensor = value.new_tensor(tuple(std)).view(*((1,) * (value.dim() - 3)), 3, 1, 1)
    return (value - mean_tensor) / std_tensor


class DROIDLogicalDataset(Dataset[dict[str, Any]]):
    """One item is one physical DROID history with two calibrated exterior views.

    One uniformly sampled variable-FOV size is shared across the positive pair.
    Each camera independently selects its highest-WAFT feasible crop center, and
    that camera transform is shared by all three history frames and geometric
    modalities. No semantic segmentation field exists in this contract.
    """

    def __init__(
        self,
        config: DROIDDatasetConfig,
        *,
        backend: EpisodeBackend | None = None,
        depth_cache: HDF5CacheReader | str | Path | None = None,
        flow_cache: HDF5CacheReader | str | Path | None = None,
        see3d_cache: HDF5CacheReader | str | Path | None = None,
        manifest_entries: Sequence[Mapping[str, Any]] | None = None,
    ):
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
        self.depth_cache = self._cache_reader(depth_cache)
        self.flow_cache = self._cache_reader(flow_cache)
        self.see3d_cache = self._cache_reader(see3d_cache)
        if config.require_depth_cache and self.depth_cache is None:
            raise ValueError("DROID training requires a precomputed X-Lens cache.")
        if config.require_flow_cache and self.flow_cache is None:
            raise ValueError("DROID training requires a precomputed WAFT cache.")
        if manifest_entries is None:
            manifest_version = calibration_manifest_version(config.calibration_manifest)
            if self.depth_cache is not None:
                self.depth_cache.require_compatible(
                    teacher_name="X-Lens",
                    calibration_version=manifest_version,
                )
            if self.flow_cache is not None:
                self.flow_cache.require_compatible(
                    teacher_name="WAFT",
                    calibration_version=manifest_version,
                )
            if self.see3d_cache is not None:
                self.see3d_cache.require_compatible(
                    teacher_name="See3D+X-Lens",
                    calibration_version=manifest_version,
                )
        self._epoch = mp.Value("q", 0, lock=True)
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

    @staticmethod
    def _cache_reader(
        value: HDF5CacheReader | str | Path | None,
    ) -> HDF5CacheReader | None:
        if value is None or isinstance(value, HDF5CacheReader):
            return value
        return HDF5CacheReader(value)

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

    def _read_depth(
        self, episode_id: str, camera_id: str, frame: int
    ) -> tuple[torch.Tensor, ...]:
        if self.depth_cache is None:
            zeros = torch.zeros(1, RLDS_HEIGHT, RLDS_WIDTH)
            return zeros, zeros.clone(), zeros.bool()
        item = self.depth_cache.read_depth(episode_id, camera_id, frame)
        return (
            _one_channel(
                item["metric_depth"], name="metric depth", dtype=torch.float32
            ),
            _one_channel(
                item["confidence"], name="depth confidence", dtype=torch.float32
            ),
            _one_channel(item["validity"], name="depth validity", dtype=torch.bool),
        )

    def _read_flow(
        self, episode_id: str, camera_id: str, frame: int, gap: int
    ) -> tuple[torch.Tensor, ...]:
        if self.flow_cache is None:
            zeros = torch.zeros(2, RLDS_HEIGHT, RLDS_WIDTH)
            return (
                zeros,
                torch.zeros(1, RLDS_HEIGHT, RLDS_WIDTH, dtype=torch.bool),
                torch.zeros(1, RLDS_HEIGHT, RLDS_WIDTH),
            )
        item = self.flow_cache.read_flow(episode_id, camera_id, frame, gap)
        confidence = item.get("confidence")
        return (
            _flow_chw(item["forward_flow"]),
            _one_channel(item["validity"], name="flow validity", dtype=torch.bool),
            (
                _one_channel(confidence, name="flow confidence", dtype=torch.float32)
                if confidence is not None
                else _one_channel(
                    item["validity"], name="flow confidence", dtype=torch.float32
                )
            ),
        )

    @staticmethod
    def _camera_history(
        images: np.ndarray, indices: Sequence[int], camera_index: int
    ) -> torch.Tensor:
        return torch.stack([_chw_rgb(images[frame, camera_index]) for frame in indices])

    @staticmethod
    def _apply_rgb(
        history: torch.Tensor,
        transform: SpatialTransform,
        padding_value: Sequence[float],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        transformed = transform_rgb(
            history, transform, padding_value=tuple(padding_value)
        )
        validity = image_validity_mask(transform, leading_shape=(history.shape[0],))
        return transformed, validity

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
        expected_prefix = (int(entry["num_steps"]), 2, RLDS_HEIGHT, RLDS_WIDTH, 3)
        if images.shape != expected_prefix:
            raise ValueError(
                f"RLDS image array disagrees with manifest: {images.shape} != {expected_prefix}."
            )
        camera_a, camera_b = randomize_camera_order(rng)
        cameras = [
            entry["exterior_cameras"][camera_a],
            entry["exterior_cameras"][camera_b],
        ]
        episode_id = str(entry["episode_id"])
        histories_raw = [
            self._camera_history(images, indices, camera)
            for camera in (camera_a, camera_b)
        ]

        flow_pairs = (
            (indices[0], stride),
            (indices[1], stride),
            (indices[0], 2 * stride),
        )
        raw_flows: list[torch.Tensor] = []
        raw_flow_validity: list[torch.Tensor] = []
        raw_flow_confidence: list[torch.Tensor] = []
        raw_depths: list[torch.Tensor] = []
        raw_depth_confidence: list[torch.Tensor] = []
        raw_depth_validity: list[torch.Tensor] = []
        for camera in cameras:
            logical_id = str(camera["logical_id"])
            flows, flow_validity, flow_confidence = zip(
                *(
                    self._read_flow(episode_id, logical_id, start, gap)
                    for start, gap in flow_pairs
                ),
                strict=True,
            )
            raw_flows.append(torch.stack(flows))
            raw_flow_validity.append(torch.stack(flow_validity))
            raw_flow_confidence.append(torch.stack(flow_confidence))
            depths, confidence, validity = zip(
                *(self._read_depth(episode_id, logical_id, frame) for frame in indices),
                strict=True,
            )
            raw_depths.append(torch.stack(depths))
            raw_depth_confidence.append(torch.stack(confidence))
            raw_depth_validity.append(torch.stack(validity))

        # A logical positive pair samples one FOV, while each physical camera
        # independently finds its strongest feasible WAFT region. Each selected
        # transform is then fixed across all three history frames and modalities.
        crop_size = sample_uniform_crop_size(self.config.motion_crop, rng)
        crop_selections = [
            select_motion_crop(
                raw_flows[camera_index],
                raw_flow_validity[camera_index],
                crop_size,
                self.config.motion_crop,
            )
            for camera_index in range(2)
        ]
        transforms = [selection.transform for selection in crop_selections]
        transformed_histories, transformed_image_validity = zip(
            *(
                self._apply_rgb(
                    history, transform, self.config.rgb_padding_value
                )
                for history, transform in zip(
                    histories_raw, transforms, strict=True
                )
            ),
            strict=True,
        )
        representation_raw = torch.stack(transformed_histories)
        representation_validity = torch.stack(transformed_image_validity)
        transformed_flows = [
            transform_flow(value, transform)
            for value, transform in zip(raw_flows, transforms, strict=True)
        ]

        target_rgb = torch.stack(transformed_histories, dim=1).float() / 255.0
        target_image_validity = torch.stack(transformed_image_validity, dim=1)
        target_depth = torch.stack(
            [
                transform_depth(value, transform)
                for value, transform in zip(raw_depths, transforms, strict=True)
            ],
            dim=1,
        )
        target_depth_confidence = torch.stack(
            [
                transform_confidence(value, transform)
                for value, transform in zip(
                    raw_depth_confidence, transforms, strict=True
                )
            ],
            dim=1,
        )
        target_depth_validity = (
            torch.stack(
                [
                    transform_validity(value, transform)
                    for value, transform in zip(
                        raw_depth_validity, transforms, strict=True
                    )
                ],
                dim=1,
            )
            & target_image_validity
        )
        target_flow = torch.stack(transformed_flows, dim=1)
        target_flow_validity = (
            torch.stack(
                [
                    transform_validity(value, transform)
                    for value, transform in zip(
                        raw_flow_validity, transforms, strict=True
                    )
                ],
                dim=1,
            )
            & target_image_validity
        )
        target_flow_confidence = torch.stack(
            [
                transform_confidence(value, transform)
                for value, transform in zip(
                    raw_flow_confidence, transforms, strict=True
                )
            ],
            dim=1,
        )

        K_rlds = torch.stack(
            [
                torch.as_tensor(camera["intrinsics_rlds"], dtype=torch.float32)
                for camera in cameras
            ]
        )
        representation_K = torch.stack(
            [
                transform_intrinsics(K, transform)
                for K, transform in zip(K_rlds, transforms, strict=True)
            ]
        )
        target_K = representation_K[None].expand(3, -1, -1, -1).contiguous()
        target_c2w = (
            torch.stack(
                [
                    torch.as_tensor(camera["c2w"], dtype=torch.float32)
                    for camera in cameras
                ]
            )[None]
            .expand(3, -1, -1, -1)
            .contiguous()
        )
        target_w2c = (
            torch.stack(
                [
                    torch.as_tensor(camera["w2c"], dtype=torch.float32)
                    for camera in cameras
                ]
            )[None]
            .expand(3, -1, -1, -1)
            .contiguous()
        )

        output: dict[str, Any] = {
            "representation_histories": normalize_encoder_rgb(
                representation_raw,
                self.config.normalize_mean,
                self.config.normalize_std,
            ),
            "representation_flows": torch.stack(
                [value[:2] for value in transformed_flows]
            ),
            "representation_validity": representation_validity,
            "representation_K": representation_K[:, None]
            .expand(-1, 3, -1, -1)
            .contiguous(),
            "target_rgb": target_rgb,
            "target_image_validity": target_image_validity,
            "target_depth": target_depth,
            "target_depth_confidence": target_depth_confidence,
            "target_depth_validity": target_depth_validity,
            "target_flow": target_flow,
            "target_flow_validity": target_flow_validity,
            "target_flow_confidence": target_flow_confidence,
            "target_K": target_K,
            "target_c2w": target_c2w,
            "target_w2c": target_w2c,
            "calibration_validity": torch.tensor(True),
            "current_timestep": torch.tensor(current, dtype=torch.long),
            "history_indices": torch.tensor(indices, dtype=torch.long),
            "temporal_stride": torch.tensor(stride, dtype=torch.long),
            "camera_order": torch.tensor((camera_a, camera_b), dtype=torch.long),
            "camera_serials": tuple(str(camera["serial"]) for camera in cameras),
            "episode_id": episode_id,
            "crop_metadata": {
                key: torch.tensor(
                    [asdict(selection.metadata)[key] for selection in crop_selections],
                    dtype=(
                        torch.bool
                        if key == "low_motion_fallback_used"
                        else (
                            torch.long
                            if key
                            in {
                                "crop_size",
                                "crop_x0",
                                "crop_y0",
                                "crop_center_x",
                                "crop_center_y",
                            }
                            else torch.float32
                        )
                    ),
                )
                for key in asdict(crop_selections[0].metadata)
            },
        }
        if self.config.include_crop_debug:
            output["crop_debug"] = {
                "original_rgb": torch.stack(histories_raw),
                "padded_rgb": torch.stack(
                    [
                        pad_to_square(
                            history,
                            transform,
                            padding_value=self.config.rgb_padding_value,
                        )
                        for history, transform in zip(
                            histories_raw, transforms, strict=True
                        )
                    ]
                ),
                "aggregate_motion": torch.stack(
                    [selection.aggregate_motion_map for selection in crop_selections]
                ),
                "smoothed_motion": torch.stack(
                    [selection.smoothed_motion_map for selection in crop_selections]
                ),
            }
        synthetic = (
            None
            if self.see3d_cache is None
            else self.see3d_cache.read_synthetic_view(
                episode_id, current, occurrence=rng.randrange(2**31)
            )
        )
        # See3D is optional and precomputed on the RLDS grid. Reuse camera A's
        # sampled spatial transform so its virtual K and all synthetic targets
        # remain mutually consistent; See3D is disabled by default.
        synthetic_tx = transforms[0]
        if synthetic is None:
            output.update(
                {
                    "synthetic_available": torch.tensor(False),
                    "synthetic_rgb": torch.zeros(3, MODEL_SIZE, MODEL_SIZE),
                    "synthetic_image_validity": image_validity_mask(synthetic_tx),
                    "synthetic_confidence": torch.zeros(1, MODEL_SIZE, MODEL_SIZE),
                    "synthetic_depth": torch.zeros(1, MODEL_SIZE, MODEL_SIZE),
                    "synthetic_depth_confidence": torch.zeros(
                        1, MODEL_SIZE, MODEL_SIZE
                    ),
                    "synthetic_depth_validity": torch.zeros(
                        1, MODEL_SIZE, MODEL_SIZE, dtype=torch.bool
                    ),
                    "synthetic_geometry_supported": torch.zeros(
                        1, MODEL_SIZE, MODEL_SIZE, dtype=torch.bool
                    ),
                    "synthetic_K": torch.eye(3),
                    "synthetic_c2w": torch.eye(4),
                    "synthetic_w2c": torch.eye(4),
                    "synthetic_view_confidence": torch.tensor(0.0),
                }
            )
        else:
            synthetic_rgb = (
                transform_rgb(
                    _chw_rgb(synthetic["generated_rgb"]),
                    synthetic_tx,
                    padding_value=self.config.rgb_padding_value,
                ).float()
                / 255.0
            )
            synthetic_confidence = transform_confidence(
                _one_channel(
                    synthetic["confidence"],
                    name="synthetic confidence",
                    dtype=torch.float32,
                ),
                synthetic_tx,
            )
            synthetic_depth_confidence = (
                transform_confidence(
                    _one_channel(
                        synthetic["depth_confidence"],
                        name="synthetic depth confidence",
                        dtype=torch.float32,
                    ),
                    synthetic_tx,
                )
                * synthetic_confidence
            )
            synthetic_depth_validity = transform_validity(
                _one_channel(
                    synthetic["depth_validity"],
                    name="synthetic depth validity",
                    dtype=torch.bool,
                ),
                synthetic_tx,
            )
            synthetic_geometry = transform_validity(
                _one_channel(
                    synthetic["geometry_supported_depth"],
                    name="synthetic geometry support",
                    dtype=torch.bool,
                ),
                synthetic_tx,
            )
            output.update(
                {
                    "synthetic_available": torch.tensor(True),
                    "synthetic_rgb": synthetic_rgb,
                    "synthetic_image_validity": image_validity_mask(synthetic_tx),
                    "synthetic_confidence": synthetic_confidence,
                    "synthetic_depth": transform_depth(
                        _one_channel(
                            synthetic["metric_depth"],
                            name="synthetic depth",
                            dtype=torch.float32,
                        ),
                        synthetic_tx,
                    ),
                    "synthetic_depth_confidence": synthetic_depth_confidence,
                    "synthetic_depth_validity": synthetic_depth_validity,
                    "synthetic_geometry_supported": synthetic_geometry,
                    "synthetic_K": transform_intrinsics(
                        torch.as_tensor(synthetic["virtual_K"], dtype=torch.float32),
                        synthetic_tx,
                    ),
                    "synthetic_c2w": torch.as_tensor(
                        synthetic["virtual_c2w"], dtype=torch.float32
                    ),
                    "synthetic_w2c": torch.as_tensor(
                        synthetic["virtual_w2c"], dtype=torch.float32
                    ),
                    "synthetic_view_confidence": synthetic_confidence[
                        image_validity_mask(synthetic_tx)
                    ].mean(),
                }
            )
        for key in ("action", "action_dict"):
            if key in episode:
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
