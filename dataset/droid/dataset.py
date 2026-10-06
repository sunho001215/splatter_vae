from __future__ import annotations

import bisect
import multiprocessing as mp
import random
import time
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .codecs import (
    NumericCodecConfig,
    decode_jpeg,
    decode_numeric_array,
    depth_u16_to_meters,
    flow_i16_to_pixels,
)
from .preprocessed_manifest import (
    RETAINED_RAW_STRIDE,
    TEMPORAL_GAP_RAW,
    load_stage0_manifest,
    resolve_window,
    sample_key,
    shard_path,
)
from .records import (
    unpack_lager_record,
    unpack_real_rgb_record,
    unpack_timestep_metadata,
)
from .sampling import (
    MotionCropConfig,
    build_middle_frame_motion_maps,
    sample_uniform_crop_size,
    select_motion_crop_from_maps,
)
from .shards import IndexedTarReader
from .transforms import (
    IMAGENET_NEUTRAL_RGB,
    MODEL_SIZE,
    RLDS_HEIGHT,
    RLDS_WIDTH,
    apply_spatial_transform,
    image_validity_mask,
    transform_depth,
    transform_flow,
    transform_intrinsics,
    transform_rgb,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
LAGERNVS_CANONICAL_FOCAL_PX = 186.5


def _lager_intrinsics(leading_shape: tuple[int, ...]) -> torch.Tensor:
    K = torch.zeros(*leading_shape, 3, 3, dtype=torch.float32)
    K[..., 0, 0] = LAGERNVS_CANONICAL_FOCAL_PX
    K[..., 1, 1] = LAGERNVS_CANONICAL_FOCAL_PX
    K[..., 0, 2] = 128.0
    K[..., 1, 2] = 128.0
    K[..., 2, 2] = 1.0
    return K


@dataclass(frozen=True)
class DROIDDatasetConfig:
    preprocessed_root: str = "/home/ws/data/droid_stage0_preprocessed"
    split: str = "train"
    seed: int = 42
    motion_crop: MotionCropConfig = field(default_factory=MotionCropConfig)
    reader_cache_size: int = 8
    verify_member_checksums: bool = False
    profile_timings: bool = False
    numeric: NumericCodecConfig = field(default_factory=NumericCodecConfig)

    def __post_init__(self) -> None:
        if self.split not in ("train", "validation"):
            raise ValueError("DROID split must be 'train' or 'validation'.")
        if self.reader_cache_size <= 0:
            raise ValueError("Indexed-TAR reader cache size must be positive.")


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


def _decode_rgb_record(payload: bytes) -> torch.Tensor:
    images = [
        torch.from_numpy(decode_jpeg(value).copy()).permute(2, 0, 1)
        for value in unpack_real_rgb_record(payload)
    ]
    output = torch.stack(images)
    if output.shape != (2, 3, RLDS_HEIGHT, RLDS_WIDTH):
        raise ValueError(
            f"Cached real RGB record has invalid shape {tuple(output.shape)}."
        )
    return output


def _native_flow_validity(
    flow: torch.Tensor, encoded_validity: torch.Tensor
) -> torch.Tensor:
    if flow.shape != (2, RLDS_HEIGHT, RLDS_WIDTH):
        raise ValueError("Native flow must be (2,180,320).")
    validity = encoded_validity.bool()
    if validity.shape == (1, RLDS_HEIGHT, RLDS_WIDTH):
        validity = validity[0]
    y, x = torch.meshgrid(
        torch.arange(RLDS_HEIGHT),
        torch.arange(RLDS_WIDTH),
        indexing="ij",
    )
    destination_x = x.float() + flow[0].float()
    destination_y = y.float() + flow[1].float()
    usable = (
        validity
        & torch.isfinite(flow).all(dim=0)
        & (destination_x >= 0.0)
        & (destination_x < RLDS_WIDTH)
        & (destination_y >= 0.0)
        & (destination_y < RLDS_HEIGHT)
    )
    return usable[None]


def _transform_flow_validity(
    flow: torch.Tensor,
    native_validity: torch.Tensor,
    transform: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    transformed = transform_flow(flow, transform).float()
    source_validity = apply_spatial_transform(
        native_validity.bool(),
        transform,
        mode="nearest",
        padding_value=0.0,
    ).bool()
    geometric = image_validity_mask(transform)
    source_validity &= geometric
    y, x = torch.meshgrid(
        torch.arange(MODEL_SIZE, dtype=torch.float32),
        torch.arange(MODEL_SIZE, dtype=torch.float32),
        indexing="ij",
    )
    destination_x = x + transformed[0]
    destination_y = y + transformed[1]
    in_bounds = (
        (destination_x >= 0.0)
        & (destination_x < MODEL_SIZE)
        & (destination_y >= 0.0)
        & (destination_y < MODEL_SIZE)
    )
    grid = torch.stack(
        (
            2.0 * (destination_x + 0.5) / MODEL_SIZE - 1.0,
            2.0 * (destination_y + 0.5) / MODEL_SIZE - 1.0,
        ),
        dim=-1,
    )[None]
    destination_validity = F.grid_sample(
        geometric[None].float(),
        grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=False,
    )[0].bool()
    validity = source_validity & destination_validity & in_bounds[None]
    return transformed.masked_fill(~validity, 0.0), validity


class DROIDPreprocessedDataset(Dataset[dict[str, Any]]):
    """Cached-only Stage-0 windows: no RLDS or foundation-model runtime."""

    def __init__(self, config: DROIDDatasetConfig) -> None:
        self.config = config
        self.manifest = load_stage0_manifest(config.preprocessed_root)
        self.entries = [
            dict(entry)
            for entry in self.manifest["episodes"]
            if entry["dataset_split"] == config.split
        ]
        if not self.entries:
            raise ValueError(
                f"No cached Stage-0 episodes exist in split {config.split!r}."
            )
        self._cumulative: list[int] = []
        count = 0
        for entry in self.entries:
            count += int(entry["training_window_count"])
            self._cumulative.append(count)
        if count == 0:
            raise ValueError(
                "Cached Stage-0 split contains no complete gap-6 histories."
            )
        self._epoch = mp.get_context("spawn").Value("q", 0, lock=True)
        self._readers: OrderedDict[int, IndexedTarReader] = OrderedDict()

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_readers"] = OrderedDict()
        return state

    def close(self) -> None:
        readers = getattr(self, "_readers", None)
        if readers is None:
            return
        for reader in readers.values():
            reader.close()
        readers.clear()

    def __del__(self) -> None:
        self.close()

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

    def _resolve_index(self, index: int) -> tuple[dict[str, Any], int]:
        episode_index = self.sample_episode_index(index)
        start = 0 if episode_index == 0 else self._cumulative[episode_index - 1]
        return self.entries[episode_index], int(index - start)

    def _reader(self, shard_id: int) -> IndexedTarReader:
        identifier = int(shard_id)
        if identifier in self._readers:
            reader = self._readers.pop(identifier)
            self._readers[identifier] = reader
            return reader
        reader = IndexedTarReader(
            shard_path(self.config.preprocessed_root, "final", identifier)
        )
        self._readers[identifier] = reader
        while len(self._readers) > int(self.config.reader_cache_size):
            _key, removed = self._readers.popitem(last=False)
            removed.close()
        return reader

    def _rng(self, index: int) -> random.Random:
        split_offset = 0 if self.config.split == "train" else 9_223_372_036_854_775
        epoch = self.epoch if self.config.split == "train" else 0
        return random.Random(
            int(self.config.seed)
            + split_offset
            + epoch * 1_000_000_007
            + int(index) * 65_537
        )

    def _decode_window(
        self, entry: Mapping[str, Any], retained_start: int
    ) -> dict[str, Any]:
        timings = {
            "shard_read": 0.0,
            "jpeg_decode": 0.0,
            "depth_decompress": 0.0,
            "flow_decompress": 0.0,
            "metadata_decode": 0.0,
        }
        reference = resolve_window(entry, retained_start)
        reader = self._reader(int(entry["shard_id"]))
        rgb_values = []
        depth_values = []
        lager_values = []
        lager_metadata = []
        support_values = []
        for retained_index, raw_timestep in zip(
            reference.retained_indices, reference.raw_timesteps, strict=True
        ):
            key = sample_key(int(entry["global_retained_start"]) + retained_index)
            started = time.perf_counter()
            metadata_payload = reader.read(
                key, "meta", verify=self.config.verify_member_checksums
            )
            timings["shard_read"] += time.perf_counter() - started
            started = time.perf_counter()
            metadata = unpack_timestep_metadata(metadata_payload)
            timings["metadata_decode"] += time.perf_counter() - started
            expected = {
                "episode_index": int(entry["manifest_episode_index"]),
                "episode_id": str(entry["episode_id"]),
                "retained_index": int(retained_index),
                "raw_timestep": int(raw_timestep),
            }
            if any(metadata[name] != value for name, value in expected.items()):
                raise ValueError(
                    f"Cached timestep metadata mismatch for {key}: {metadata}."
                )
            started = time.perf_counter()
            rgb_payload = reader.read(
                key, "rgb", verify=self.config.verify_member_checksums
            )
            timings["shard_read"] += time.perf_counter() - started
            started = time.perf_counter()
            rgb_values.append(_decode_rgb_record(rgb_payload))
            timings["jpeg_decode"] += time.perf_counter() - started
            started = time.perf_counter()
            depth_payload = reader.read(
                key, "depth", verify=self.config.verify_member_checksums
            )
            timings["shard_read"] += time.perf_counter() - started
            started = time.perf_counter()
            encoded_depth = decode_numeric_array(
                depth_payload,
                self.config.numeric,
            )
            if encoded_depth.shape != (2, RLDS_HEIGHT, RLDS_WIDTH):
                raise ValueError("Cached DA3 depth must be (2,180,320).")
            depth, depth_validity = depth_u16_to_meters(encoded_depth)
            timings["depth_decompress"] += time.perf_counter() - started
            depth_values.append(
                (
                    torch.from_numpy(depth)[:, None],
                    torch.from_numpy(depth_validity)[:, None],
                )
            )
            started = time.perf_counter()
            lager_payload = reader.read(
                key, "lager", verify=self.config.verify_member_checksums
            )
            timings["shard_read"] += time.perf_counter() - started
            started = time.perf_counter()
            jpeg_values, pose_metadata, support = unpack_lager_record(lager_payload)
            timings["metadata_decode"] += time.perf_counter() - started
            started = time.perf_counter()
            lager_rgb = torch.stack(
                [
                    torch.from_numpy(decode_jpeg(value).copy()).permute(2, 0, 1)
                    for value in jpeg_values
                ]
            )
            timings["jpeg_decode"] += time.perf_counter() - started
            if lager_rgb.shape != (4, 3, 256, 256):
                raise ValueError(
                    "Each cached timestamp must contain four 256x256 Lager JPEGs."
                )
            lager_values.append(lager_rgb)
            lager_metadata.append(pose_metadata)
            support_values.append(torch.from_numpy(support))

        flow_values = []
        flow_validity_values = []
        for retained_index in reference.retained_indices[:2]:
            key = sample_key(int(entry["global_retained_start"]) + retained_index)
            started = time.perf_counter()
            flow_payload = reader.read(
                key, "flow", verify=self.config.verify_member_checksums
            )
            timings["shard_read"] += time.perf_counter() - started
            started = time.perf_counter()
            encoded = decode_numeric_array(
                flow_payload,
                self.config.numeric,
            )
            if encoded.shape != (2, 2, RLDS_HEIGHT, RLDS_WIDTH):
                raise ValueError(
                    "Cached MegaFlow must be (2 cameras,2 components,180,320)."
                )
            flow, encoded_validity = flow_i16_to_pixels(encoded)
            timings["flow_decompress"] += time.perf_counter() - started
            flow_tensor = torch.from_numpy(flow)
            encoded_validity_tensor = torch.from_numpy(encoded_validity)
            flow_values.append(flow_tensor)
            flow_validity_values.append(
                torch.stack(
                    [
                        _native_flow_validity(
                            flow_tensor[camera], encoded_validity_tensor[camera]
                        )
                        for camera in range(2)
                    ]
                )
            )
        return {
            "raw_rgb": torch.stack(rgb_values),
            "native_depth": torch.stack([value[0] for value in depth_values]),
            "native_depth_validity": torch.stack([value[1] for value in depth_values]),
            "native_flow": torch.stack(flow_values),
            "native_flow_validity": torch.stack(flow_validity_values),
            "novel_rgb_u8": torch.stack(lager_values),
            "novel_support_mask": torch.stack(support_values),
            "novel_pose_metadata": {
                name: torch.from_numpy(
                    np.stack([value[name] for value in lager_metadata])
                )
                for name in lager_metadata[0]
            },
            "reference": reference,
            "_profile_timings_ms": {
                name: value * 1000.0 for name, value in timings.items()
            },
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        item_started = time.perf_counter()
        index_started = item_started
        entry, retained_start = self._resolve_index(index)
        index_ms = (time.perf_counter() - index_started) * 1000.0
        decoded = self._decode_window(entry, retained_start)
        transform_started = time.perf_counter()
        rng = self._rng(index)
        crop_size = sample_uniform_crop_size(self.config.motion_crop, rng)
        raw_rgb = decoded["raw_rgb"]
        native_depth = decoded["native_depth"]
        native_depth_validity = decoded["native_depth_validity"]
        native_flow = decoded["native_flow"]
        native_flow_validity = decoded["native_flow_validity"]

        cropped_rgb = []
        cropped_depth = []
        cropped_depth_validity = []
        cropped_flow = []
        cropped_flow_validity = []
        cropped_image_validity = []
        transformed_K = []
        representation_motion = []
        selections = []
        K = torch.tensor(
            [camera["intrinsics_rlds"] for camera in entry["exterior_cameras"]],
            dtype=torch.float32,
        )
        for camera in range(2):
            aggregate, smoothed = build_middle_frame_motion_maps(
                native_flow[0, camera],
                native_flow_validity[0, camera],
                native_flow[1, camera],
                native_flow_validity[1, camera],
                self.config.motion_crop,
            )
            selection = select_motion_crop_from_maps(
                aggregate, smoothed, crop_size, self.config.motion_crop
            )
            selections.append(selection)
            transform = selection.transform
            selected_motion = selection.aggregate_motion_map[
                transform.crop_y : transform.crop_y + transform.crop_size,
                transform.crop_x : transform.crop_x + transform.crop_size,
            ]
            representation_motion.append(
                F.interpolate(
                    selected_motion[None, None],
                    size=(MODEL_SIZE, MODEL_SIZE),
                    mode="bilinear",
                    align_corners=False,
                )[0]
            )
            rgb = transform_rgb(
                raw_rgb[:, camera],
                transform,
                padding_value=IMAGENET_NEUTRAL_RGB,
            )
            depth = transform_depth(native_depth[:, camera], transform).float()
            image_validity = image_validity_mask(transform, leading_shape=(3,))
            depth_validity = (
                apply_spatial_transform(
                    native_depth_validity[:, camera],
                    transform,
                    mode="nearest",
                    padding_value=0.0,
                ).bool()
                & image_validity
                & (depth > 0.0)
            )
            per_flow = [
                _transform_flow_validity(
                    native_flow[pair, camera],
                    native_flow_validity[pair, camera],
                    transform,
                )
                for pair in range(2)
            ]
            cropped_rgb.append(rgb)
            cropped_depth.append(depth)
            cropped_depth_validity.append(depth_validity)
            cropped_flow.append(torch.stack([value[0] for value in per_flow]))
            cropped_flow_validity.append(torch.stack([value[1] for value in per_flow]))
            cropped_image_validity.append(image_validity)
            transformed_K.append(transform_intrinsics(K[camera], transform))

        rgb_by_camera = torch.stack(cropped_rgb)
        depth_by_camera = torch.stack(cropped_depth)
        depth_validity_by_camera = torch.stack(cropped_depth_validity)
        flow_by_camera = torch.stack(cropped_flow)
        flow_validity_by_camera = torch.stack(cropped_flow_validity)
        image_validity_by_camera = torch.stack(cropped_image_validity)
        camera_K = torch.stack(transformed_K)
        time_major_K = camera_K[None].expand(3, -1, -1, -1).contiguous()
        c2w = torch.tensor(
            [camera["c2w"] for camera in entry["exterior_cameras"]], dtype=torch.float32
        )
        w2c = torch.tensor(
            [camera["w2c"] for camera in entry["exterior_cameras"]], dtype=torch.float32
        )
        target_K_lager = _lager_intrinsics((3, 4))
        metadata = decoded["novel_pose_metadata"]
        crop_metadata = {
            name: torch.tensor([getattr(value.metadata, name) for value in selections])
            for name in (
                "crop_size",
                "crop_x0",
                "crop_y0",
                "crop_center_x",
                "crop_center_y",
                "resize_scale",
                "real_pixel_fraction",
                "flow_peak_value",
                "selected_crop_flow_mean",
                "low_motion_fallback_used",
            )
        }
        reference = decoded["reference"]
        output = {
            # Cached source-resolution RGB is retained only for diagnostics and
            # visualization; it is never fetched from RLDS during training.
            "raw_histories": raw_rgb.permute(1, 0, 2, 3, 4).contiguous(),
            "representation_histories": normalize_encoder_rgb(rgb_by_camera),
            "representation_flows": flow_by_camera,
            "representation_middle_motion": torch.stack(representation_motion),
            "representation_flow_validity": flow_validity_by_camera,
            "representation_validity": image_validity_by_camera,
            "representation_K": camera_K[:, None].expand(-1, 3, -1, -1).contiguous(),
            "target_rgb": rgb_by_camera.permute(1, 0, 2, 3, 4).float() / 255.0,
            "target_image_validity": image_validity_by_camera.permute(1, 0, 2, 3, 4),
            "target_depth": depth_by_camera.permute(1, 0, 2, 3, 4),
            "target_depth_validity": depth_validity_by_camera.permute(1, 0, 2, 3, 4),
            "target_flow": flow_by_camera.permute(1, 0, 2, 3, 4),
            "target_flow_validity": flow_validity_by_camera.permute(1, 0, 2, 3, 4),
            "target_K": time_major_K,
            "target_c2w": c2w[None].expand(3, -1, -1, -1).contiguous(),
            "target_w2c": w2c[None].expand(3, -1, -1, -1).contiguous(),
            "native_K": K,
            "native_c2w": c2w,
            "native_w2c": w2c,
            "novel_rgb": decoded["novel_rgb_u8"].float() / 255.0,
            "novel_K": target_K_lager,
            "novel_c2w": metadata["target_c2w"],
            "novel_w2c": metadata["target_w2c"],
            "novel_base_c2w": metadata["base_c2w"],
            "novel_support_mask": decoded["novel_support_mask"].bool(),
            "novel_pose_metadata": {
                name: value
                for name, value in metadata.items()
                if name
                not in {
                    "target_c2w",
                    "target_w2c",
                    "base_c2w",
                    "source_clearance_reference",
                }
            },
            "crop_metadata": crop_metadata,
            "native_megaflow_flow": native_flow,
            "native_megaflow_validity": native_flow_validity,
            "native_da3_depth": native_depth,
            "native_da3_validity": native_depth_validity,
            "aggregate_motion": torch.stack(
                [selection.aggregate_motion_map for selection in selections]
            ),
            "smoothed_motion": torch.stack(
                [selection.smoothed_motion_map for selection in selections]
            ),
            "calibration_validity": torch.tensor(True),
            "sampled_crop_size": torch.tensor(crop_size, dtype=torch.long),
            "history_raw_timesteps": torch.tensor(
                reference.raw_timesteps, dtype=torch.long
            ),
            "history_retained_indices": torch.tensor(
                reference.retained_indices, dtype=torch.long
            ),
            "retained_raw_stride": torch.tensor(RETAINED_RAW_STRIDE, dtype=torch.long),
            "temporal_gap_raw": torch.tensor(TEMPORAL_GAP_RAW, dtype=torch.long),
            "camera_order": torch.tensor((0, 1), dtype=torch.long),
            "camera_serials": tuple(
                str(camera["serial"]) for camera in entry["exterior_cameras"]
            ),
            "episode_id": str(entry["episode_id"]),
            "shard_id": torch.tensor(int(entry["shard_id"]), dtype=torch.long),
        }
        if self.config.profile_timings:
            timings = decoded["_profile_timings_ms"]
            timings.update(
                {
                    "index_lookup": index_ms,
                    "motion_crop_and_transforms": (
                        time.perf_counter() - transform_started
                    )
                    * 1000.0,
                    "dataset_total": (time.perf_counter() - item_started) * 1000.0,
                }
            )
            output["profile_timings_ms"] = timings
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
