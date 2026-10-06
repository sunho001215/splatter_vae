from __future__ import annotations

import bisect
import json
import os
import random
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .codecs import (
    NumericCodecConfig,
    decode_jpeg,
    decode_numeric_array,
    depth_u16_to_meters,
    flow_i16_to_pixels,
)
from .preprocessed_manifest import (
    RETAINED_RAW_STRIDE,
    load_stage0_manifest,
    sample_key,
    shard_path,
)
from .records import (
    LAGER_RECORD_MAGIC,
    LAGER_V3_RECORD_MAGIC,
    LEGACY_LAGER_RECORD_MAGIC,
    unpack_lager_record,
    unpack_real_rgb_record,
    unpack_timestep_metadata,
)
from .safety import validate_derived_root
from .shards import (
    IndexedTarReader,
    shard_is_complete,
    shard_sidecars,
    validate_indexed_shard,
    write_json_atomic,
)


@dataclass
class _RunningScalar:
    count: int = 0
    total: float = 0.0
    minimum: float = float("inf")

    def add(self, value: float) -> None:
        numeric = float(value)
        self.count += 1
        self.total += numeric
        self.minimum = min(self.minimum, numeric)

    @property
    def mean(self) -> float | None:
        return self.total / self.count if self.count else None

    @property
    def minimum_or_none(self) -> float | None:
        return self.minimum if self.count else None


@dataclass
class BoundedPrioritySample:
    """Keep a deterministic bounded sample without retaining one array per item."""

    capacity: int
    seed: int
    item_shape: tuple[int, ...] = ()
    flush_size: int = 500_000
    _values: np.ndarray = field(init=False)
    _priorities: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32), init=False
    )
    _value_chunks: list[np.ndarray] = field(default_factory=list, init=False)
    _priority_chunks: list[np.ndarray] = field(default_factory=list, init=False)
    _buffered: int = field(default=0, init=False)
    _rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        if self.capacity <= 0 or self.flush_size <= 0:
            raise ValueError("Bounded sample sizes must be positive.")
        self.item_shape = tuple(int(size) for size in self.item_shape)
        if any(size <= 0 for size in self.item_shape):
            raise ValueError("Bounded sample item dimensions must be positive.")
        self._values = np.empty((0, *self.item_shape), dtype=np.float32)
        self._rng = np.random.default_rng(int(self.seed))

    def add(self, values: np.ndarray, *, maximum_per_item: int = 2048) -> None:
        limit = int(maximum_per_item)
        if limit <= 0:
            raise ValueError("maximum_per_item must be positive.")
        value = np.asarray(values, dtype=np.float32)
        if self.item_shape:
            dimensions = len(self.item_shape)
            if (
                value.ndim < dimensions
                or tuple(value.shape[-dimensions:]) != self.item_shape
            ):
                raise ValueError(
                    f"Expected sample items with trailing shape {self.item_shape}, "
                    f"got {value.shape}."
                )
            value = value.reshape((-1, *self.item_shape))
        else:
            value = value.reshape(-1)
        count = int(value.shape[0])
        if count > limit:
            # One deterministic random phase per call gives spatially stratified
            # coverage without allocating or sorting a full-size random array.
            step = count / limit
            phase = float(self._rng.random())
            selected = np.floor(
                (np.arange(limit, dtype=np.float64) + phase) * step
            ).astype(np.int64)
            value = value[selected]
            count = limit
        if not count:
            return
        self._value_chunks.append(value.copy())
        self._priority_chunks.append(self._rng.random(count, dtype=np.float32))
        self._buffered += count
        if self._buffered >= self.flush_size:
            self._flush()

    def _flush(self) -> None:
        if not self._buffered:
            return
        values = np.concatenate((self._values, *self._value_chunks))
        priorities = np.concatenate((self._priorities, *self._priority_chunks))
        if len(values) > self.capacity:
            selected = np.argpartition(priorities, -self.capacity)[-self.capacity :]
            values = values[selected]
            priorities = priorities[selected]
        self._values = values
        self._priorities = priorities
        self._value_chunks.clear()
        self._priority_chunks.clear()
        self._buffered = 0

    def result(self) -> np.ndarray:
        self._flush()
        return self._values


@dataclass(frozen=True)
class IntegrityConfig:
    root: str
    random_samples: int = 512
    seed: int = 20260828
    verify_shard_checksums: bool = True
    full_payload_scan: bool = False
    decode_all_jpegs: bool = False
    loader_windows: int = 32

    def __post_init__(self) -> None:
        if self.random_samples <= 0:
            raise ValueError("Integrity checking requires at least one random sample.")
        if self.loader_windows < 0:
            raise ValueError("loader_windows must be nonnegative.")
        if self.decode_all_jpegs and not self.full_payload_scan:
            raise ValueError("decode_all_jpegs requires full_payload_scan.")


def classify_partial_artifacts(
    root: str | os.PathLike[str],
) -> tuple[list[str], list[str]]:
    """Separate active partial outputs from explicitly archived child trees."""

    dataset_root = Path(root).expanduser().resolve()
    active: list[str] = []
    archived: list[str] = []
    for path in dataset_root.rglob("*.partial"):
        relative = path.relative_to(dataset_root)
        destination = archived if relative.parts[0] in {"pilot", "recovery"} else active
        destination.append(str(relative))
    return sorted(active), sorted(archived)


def _suffix(name: str) -> str:
    if "." not in name:
        raise ValueError(f"Malformed indexed member name {name!r}.")
    return name.rsplit(".", 1)[1]


def _expected_names(manifest: dict[str, Any], shard: dict[str, Any]) -> set[str]:
    expected: set[str] = set()
    episodes = manifest["episodes"][
        int(shard["episode_start"]) : int(shard["episode_stop"])
    ]
    for entry in episodes:
        count = int(entry["retained_count"])
        for retained_index in range(count):
            key = sample_key(int(entry["global_retained_start"]) + retained_index)
            expected.update(
                f"{key}.{suffix}" for suffix in ("meta", "rgb", "depth", "lager")
            )
            if retained_index + 2 < count:
                expected.add(f"{key}.flow")
    return expected


def _validate_pose_metadata(
    metadata: dict[str, np.ndarray], *, record_magic: bytes
) -> None:
    c2w = np.asarray(metadata["target_c2w"], dtype=np.float64)
    w2c = np.asarray(metadata["target_w2c"], dtype=np.float64)
    base = np.asarray(metadata["base_c2w"], dtype=np.float64)
    if c2w.shape != (4, 4, 4) or w2c.shape != (4, 4, 4) or base.shape != (4, 4, 4):
        raise ValueError("Lager pose matrices do not have four 4x4 targets.")
    for name, poses in (("target_c2w", c2w), ("target_w2c", w2c), ("base_c2w", base)):
        if not np.isfinite(poses).all():
            raise ValueError(f"{name} contains non-finite values.")
        if not np.allclose(poses[:, 3], (0.0, 0.0, 0.0, 1.0), atol=2.0e-4):
            raise ValueError(f"{name} has invalid homogeneous rows.")
        rotations = poses[:, :3, :3]
        if not np.allclose(
            rotations @ np.swapaxes(rotations, -1, -2),
            np.eye(3),
            atol=3.0e-3,
        ):
            raise ValueError(f"{name} contains non-orthonormal rotations.")
        if not np.allclose(np.linalg.det(rotations), 1.0, atol=3.0e-3):
            raise ValueError(f"{name} contains improper rotations.")
    if not np.allclose(c2w @ w2c, np.eye(4), atol=3.0e-3):
        raise ValueError("Lager target c2w/w2c matrices are inconsistent.")

    alpha = np.asarray(metadata["alpha"], dtype=np.float64)
    bands = ((0.15, 0.25), (0.25, 0.35), (0.65, 0.75), (0.75, 0.85))
    if alpha.shape != (4,) or any(
        not (low - 1.0e-6 <= value <= high + 1.0e-6)
        for value, (low, high) in zip(alpha, bands, strict=True)
    ):
        raise ValueError(f"Lager alpha values violate stratified bands: {alpha}.")
    if not np.allclose(alpha[2:], (1.0 - alpha[1], 1.0 - alpha[0]), atol=2.0e-6):
        raise ValueError(f"Lager alpha values are not symmetric: {alpha}.")
    if np.any(np.abs(alpha - 0.5) < 0.15 - 1.0e-6):
        raise ValueError(f"Lager alpha entered the excluded center region: {alpha}.")
    for name in (
        "baseline",
        "translation_perturbation",
        "translation_perturbation_magnitude",
        "rotation_perturbation_degrees",
        "source_coverage",
        "minimum_source_coverage_threshold",
        "minimum_geometry_distance",
        "geometry_clearance_distance",
        "minimum_geometry_distance_threshold",
        "target_scene_center",
        "distance_from_camera_a",
        "distance_from_camera_b",
    ):
        if not np.isfinite(metadata[name]).all():
            raise ValueError(f"Lager metadata {name} contains non-finite values.")
    baseline = np.asarray(metadata["baseline"], dtype=np.float64)
    translation = np.asarray(
        metadata["translation_perturbation_magnitude"], dtype=np.float64
    )
    rotation = np.asarray(metadata["rotation_perturbation_degrees"], dtype=np.float64)
    translation_escalated = np.asarray(
        metadata["safety_translation_limit_escalated"], dtype=np.bool_
    )
    thresholds_relaxed = np.asarray(
        metadata["safety_thresholds_relaxed"], dtype=np.bool_
    )
    coverage_threshold = np.asarray(
        metadata["minimum_source_coverage_threshold"], dtype=np.float64
    )
    clearance_threshold = np.asarray(
        metadata["minimum_geometry_distance_threshold"], dtype=np.float64
    )
    if np.any(baseline <= 0.0):
        raise ValueError("Lager source-camera baselines must be positive.")
    translation_limit = np.where(translation_escalated, 0.05, 0.03) * baseline
    if np.any(translation > translation_limit + 1.0e-5):
        raise ValueError("Lager translation perturbation exceeds its recorded tier.")
    if np.any(rotation > 3.0 + 1.0e-5):
        raise ValueError("Lager rotation perturbation exceeds three degrees.")
    actual_translation = c2w[:, :3, 3] - base[:, :3, 3]
    if not np.allclose(
        actual_translation, metadata["translation_perturbation"], atol=1.0e-5
    ) or not np.allclose(
        np.linalg.norm(actual_translation, axis=-1), translation, atol=1.0e-5
    ):
        raise ValueError("Lager translation metadata disagrees with pose matrices.")
    relative_rotation = np.swapaxes(base[:, :3, :3], -1, -2) @ c2w[:, :3, :3]
    skew = np.stack(
        (
            relative_rotation[:, 2, 1] - relative_rotation[:, 1, 2],
            relative_rotation[:, 0, 2] - relative_rotation[:, 2, 0],
            relative_rotation[:, 1, 0] - relative_rotation[:, 0, 1],
        ),
        axis=-1,
    )
    actual_rotation = np.rad2deg(
        np.arctan2(
            np.linalg.norm(skew, axis=-1),
            np.trace(relative_rotation, axis1=-2, axis2=-1) - 1.0,
        )
    )
    if (
        np.any(translation < 0.0)
        or np.any(rotation < 0.0)
        or not np.allclose(actual_rotation, rotation, atol=0.03)
    ):
        raise ValueError("Lager rotation metadata disagrees with pose matrices.")
    expected_coverage_threshold = np.where(thresholds_relaxed, 0.40, 0.60)
    if not np.allclose(coverage_threshold, expected_coverage_threshold, atol=1.0e-6):
        raise ValueError("Lager source-coverage threshold metadata is inconsistent.")
    if record_magic == LAGER_RECORD_MAGIC:
        source_reference = np.asarray(
            metadata["source_clearance_reference"], dtype=np.float64
        )
        if source_reference.shape != (4,) or not np.isfinite(source_reference).all():
            raise ValueError("Lager source-clearance reference is invalid.")
        expected_clearance_threshold = np.where(
            thresholds_relaxed,
            np.clip(0.80 * source_reference, 0.02, 0.05),
            0.08,
        )
        if not np.allclose(
            clearance_threshold, expected_clearance_threshold, atol=1.0e-6
        ):
            raise ValueError("Lager clearance-threshold metadata is inconsistent.")
    elif record_magic == LAGER_V3_RECORD_MAGIC:
        if np.any(
            np.abs(clearance_threshold[~thresholds_relaxed] - 0.08) > 1.0e-6
        ) or np.any(
            (clearance_threshold[thresholds_relaxed] < 0.05 - 1.0e-6)
            | (clearance_threshold[thresholds_relaxed] > 0.06 + 1.0e-6)
        ):
            raise ValueError("Legacy-v3 Lager clearance thresholds are inconsistent.")
    elif not np.allclose(clearance_threshold, 0.08, atol=1.0e-6):
        raise ValueError("Legacy-v2 Lager clearance thresholds are inconsistent.")
    if np.any(np.asarray(metadata["source_coverage"]) < coverage_threshold - 1.0e-5):
        raise ValueError(
            "Lager target source coverage is below its recorded threshold."
        )
    if np.any(
        np.asarray(metadata["geometry_clearance_distance"])
        < clearance_threshold - 1.0e-5
    ):
        raise ValueError("Lager target clearance is below its recorded threshold.")
    if np.any(np.asarray(metadata["alpha_resample_count"]) < 0):
        raise ValueError("Lager alpha resample counts must be nonnegative.")


def _validate_pose_against_depth(
    metadata: dict[str, np.ndarray],
    depth_m: np.ndarray,
    entry: dict[str, Any],
) -> None:
    """Independently corroborate sampled pose metadata from depth using NumPy.

    No pose-sampler helper or teacher is imported. Sampling every second pixel
    at half-integer pixel centers matches the signed storage geometry contract.
    """

    cameras = entry["exterior_cameras"]
    source_c2w = np.asarray([camera["c2w"] for camera in cameras], dtype=np.float64)
    intrinsics = np.asarray(
        [camera["intrinsics_rlds"] for camera in cameras], dtype=np.float64
    )
    height, width = depth_m.shape[-2:]
    vv, uu = np.meshgrid(
        np.arange(0, height, 2, dtype=np.float64) + 0.5,
        np.arange(0, width, 2, dtype=np.float64) + 0.5,
        indexing="ij",
    )
    points = []
    for view in range(2):
        z = np.asarray(depth_m[view, ::2, ::2], dtype=np.float64)
        rays = (
            np.stack((uu, vv, np.ones_like(uu)), axis=-1)
            @ np.linalg.inv(intrinsics[view]).T
        )
        camera_points = rays * z[..., None]
        world = camera_points @ source_c2w[view, :3, :3].T + source_c2w[view, :3, 3]
        points.append(world[np.isfinite(z) & (z > 0.0)])
    world_points = np.concatenate(points)
    if not len(world_points) or not np.isfinite(world_points).all():
        raise ValueError(
            "Cannot audit Lager poses without finite positive cached depth."
        )
    source_centers = source_c2w[:, :3, 3]
    baseline = np.linalg.norm(source_centers[1] - source_centers[0])
    if not np.allclose(metadata["baseline"], baseline, atol=1.0e-5):
        raise ValueError("Lager baseline disagrees with calibrated source cameras.")
    target_centers = np.asarray(metadata["target_c2w"], dtype=np.float64)[:, :3, 3]
    source_clearances = np.array(
        [
            np.quantile(np.linalg.norm(world_points - center, axis=-1), 0.01)
            for center in source_centers
        ]
    )
    alpha = np.asarray(metadata["alpha"], dtype=np.float64)
    if "source_clearance_reference" in metadata:
        reference = (1.0 - alpha) * source_clearances[0] + alpha * source_clearances[1]
        if not np.allclose(
            metadata["source_clearance_reference"], reference, atol=5e-5
        ):
            raise ValueError(
                "Lager source-clearance reference disagrees with cached depth."
            )
    for view, center in enumerate(target_centers):
        distances = np.linalg.norm(world_points - center, axis=-1)
        measured = np.array((distances.min(), np.quantile(distances, 0.01)))
        reported = np.array(
            (
                metadata["minimum_geometry_distance"][view],
                metadata["geometry_clearance_distance"][view],
            )
        )
        if not np.allclose(measured, reported, atol=5e-5):
            raise ValueError("Lager target clearance disagrees with cached depth.")
        if (
            measured[1]
            < float(metadata["minimum_geometry_distance_threshold"][view]) - 1e-5
        ):
            raise ValueError("Lager target fails its clearance gate on cached depth.")
    for view, name in enumerate(("distance_from_camera_a", "distance_from_camera_b")):
        if not np.allclose(
            metadata[name],
            np.linalg.norm(target_centers - source_centers[view], axis=-1),
            atol=1e-5,
        ):
            raise ValueError(
                "Lager source-distance metadata disagrees with calibrated poses."
            )


def _validate_payloads(
    reader: IndexedTarReader,
    entry: dict[str, Any],
    retained_index: int,
    numeric: NumericCodecConfig,
    *,
    decode_jpegs: bool,
    verify_depth_geometry: bool = False,
) -> dict[str, Any]:
    global_index = int(entry["global_retained_start"]) + int(retained_index)
    key = sample_key(global_index)
    metadata = unpack_timestep_metadata(reader.read(key, "meta", verify=True))
    expected_metadata = {
        "global_retained_index": global_index,
        "episode_index": int(entry["manifest_episode_index"]),
        "episode_id": str(entry["episode_id"]),
        "retained_index": int(retained_index),
        "raw_timestep": int(retained_index) * RETAINED_RAW_STRIDE,
    }
    if metadata != expected_metadata:
        raise ValueError(
            f"Timestep metadata mismatch for {key}: {metadata} != {expected_metadata}."
        )

    real_jpegs = unpack_real_rgb_record(reader.read(key, "rgb", verify=True))
    if len(real_jpegs) != 2:
        raise ValueError(f"{key} does not contain two real JPEGs.")
    depth_encoded = decode_numeric_array(
        reader.read(key, "depth", verify=True), numeric
    )
    if depth_encoded.dtype != np.dtype("<u2") or depth_encoded.shape != (2, 180, 320):
        raise ValueError(f"{key} has invalid DA3 depth {depth_encoded.shape}.")
    depth_m, depth_valid = depth_u16_to_meters(depth_encoded)
    if not np.isfinite(depth_m).all():
        raise ValueError(f"{key} decodes to non-finite DA3 depth.")
    depth_valid_fraction = float(depth_valid.mean())
    if depth_valid_fraction < 0.50:
        raise ValueError(
            f"{key} has an implausibly low DA3 valid fraction "
            f"({depth_valid_fraction:.4f})."
        )

    has_flow = int(retained_index) + 2 < int(entry["retained_count"])
    flow_valid_fraction = None
    if has_flow:
        flow_encoded = decode_numeric_array(
            reader.read(key, "flow", verify=True), numeric
        )
        if flow_encoded.dtype != np.dtype("<i2") or flow_encoded.shape != (
            2,
            2,
            180,
            320,
        ):
            raise ValueError(f"{key} has invalid MegaFlow {flow_encoded.shape}.")
        flow, flow_valid = flow_i16_to_pixels(flow_encoded)
        if not np.isfinite(flow).all():
            raise ValueError(f"{key} decodes to non-finite MegaFlow.")
        flow_valid_fraction = float(flow_valid.mean())
    elif reader.contains(key, "flow"):
        raise ValueError(f"{key} unexpectedly stores flow without t+6.")

    lager_payload = reader.read(key, "lager", verify=True)
    lager_record_magic = bytes(memoryview(lager_payload)[:8])
    lager_jpegs, lager_metadata, support = unpack_lager_record(lager_payload)
    if len(lager_jpegs) != 4 or support.shape != (4, 1, 256, 256):
        raise ValueError(f"{key} does not contain exactly four Lager targets.")
    _validate_pose_metadata(lager_metadata, record_magic=lager_record_magic)
    if verify_depth_geometry:
        _validate_pose_against_depth(lager_metadata, depth_m, entry)
    if decode_jpegs:
        for payload in real_jpegs:
            image = decode_jpeg(payload)
            if image.shape != (180, 320, 3):
                raise ValueError(f"{key} real JPEG has shape {image.shape}.")
        for payload in lager_jpegs:
            image = decode_jpeg(payload)
            if image.shape != (256, 256, 3):
                raise ValueError(f"{key} Lager JPEG has shape {image.shape}.")
    return {
        "depth_valid_fraction": depth_valid_fraction,
        "valid_depth_values": depth_m[depth_valid],
        "flow_valid_fraction": flow_valid_fraction,
        "lager_support_fraction": float(support.mean()),
        "lager_record_schema": (
            "v4"
            if lager_record_magic == LAGER_RECORD_MAGIC
            else "v3_legacy"
            if lager_record_magic == LAGER_V3_RECORD_MAGIC
            else "v2_legacy"
            if lager_record_magic == LEGACY_LAGER_RECORD_MAGIC
            else "unknown"
        ),
        "lager_exceptional_translation_targets": int(
            np.asarray(
                lager_metadata["safety_translation_limit_escalated"],
                dtype=np.bool_,
            ).sum()
        ),
        "lager_relaxed_safety_targets": int(
            np.asarray(
                lager_metadata["safety_thresholds_relaxed"], dtype=np.bool_
            ).sum()
        ),
    }


def _sample_locations(
    manifest: dict[str, Any], count: int, seed: int
) -> list[tuple[dict[str, Any], int]]:
    total = int(manifest["counts"]["retained_timesteps"])
    requested = min(int(count), total)
    evenly = np.linspace(0, total - 1, num=requested, dtype=np.int64).tolist()
    rng = random.Random(int(seed))
    random_count = min(requested, total)
    random_values = rng.sample(range(total), random_count)
    global_indices = sorted(set(evenly + random_values))
    stops = [
        int(entry["global_retained_start"]) + int(entry["retained_count"])
        for entry in manifest["episodes"]
    ]
    output = []
    for global_index in global_indices:
        episode_index = bisect.bisect_right(stops, global_index)
        entry = manifest["episodes"][episode_index]
        output.append(
            (
                entry,
                int(global_index) - int(entry["global_retained_start"]),
            )
        )
    return output


def _loader_check(root: Path, count: int, seed: int) -> dict[str, Any]:
    if count == 0:
        return {"windows_checked": 0}
    from .dataset import DROIDDatasetConfig, DROIDPreprocessedDataset

    shapes = None
    checked = 0
    per_split: dict[str, int] = {}
    for split in ("train", "validation"):
        try:
            dataset = DROIDPreprocessedDataset(
                DROIDDatasetConfig(
                    preprocessed_root=str(root),
                    split=split,
                    seed=int(seed),
                    verify_member_checksums=True,
                )
            )
        except ValueError:
            continue
        sample_count = min(int(count), len(dataset))
        indices = np.linspace(0, len(dataset) - 1, sample_count, dtype=np.int64)
        for index in indices:
            item = dataset[int(index)]
            if item["history_raw_timesteps"].tolist()[1:] != [
                int(item["history_raw_timesteps"][0]) + 6,
                int(item["history_raw_timesteps"][0]) + 12,
            ]:
                raise ValueError("Cached DataLoader returned a non-gap-6 history.")
            current_shapes = {
                "real_rgb": list(item["target_rgb"].shape),
                "depth": list(item["target_depth"].shape),
                "flow": list(item["target_flow"].shape),
                "lager_rgb": list(item["novel_rgb"].shape),
            }
            expected = {
                "real_rgb": [3, 2, 3, 224, 224],
                "depth": [3, 2, 1, 224, 224],
                "flow": [2, 2, 2, 224, 224],
                "lager_rgb": [3, 4, 3, 256, 256],
            }
            if current_shapes != expected:
                raise ValueError(
                    f"Cached DataLoader modality shapes are invalid: {current_shapes}."
                )
            shapes = current_shapes
            checked += 1
        per_split[split] = sample_count
        dataset.close()
    if checked == 0:
        raise ValueError("Integrity checker could not load any training windows.")
    return {
        "windows_checked": checked,
        "per_split": per_split,
        "shapes": shapes,
    }


def verify_stage0_dataset(config: IntegrityConfig) -> dict[str, Any]:
    root = Path(config.root).expanduser().resolve()
    manifest = load_stage0_manifest(root)
    numeric_config = manifest["encoding"]["numeric_compression"]
    numeric = NumericCodecConfig(
        cname=str(numeric_config["codec"]),
        compression_level=int(numeric_config["compression_level"]),
        shuffle=str(numeric_config["shuffle"]),
    )
    started = time.perf_counter()
    partials, archived_partials = classify_partial_artifacts(root)
    if partials:
        raise ValueError(f"Unexpected partial artifacts remain: {partials[:20]}")

    expected_shards = len(manifest["shards"])
    final_tar_paths = sorted((root / "shards").glob("stage0-*.tar"))
    if len(final_tar_paths) != expected_shards:
        raise ValueError(
            f"Final shard count {len(final_tar_paths)} != expected {expected_shards}."
        )
    payload_bytes: Counter[str] = Counter()
    container_bytes = 0
    actual_member_count = 0
    for shard in manifest["shards"]:
        shard_id = int(shard["shard_id"])
        path = shard_path(root, "final", shard_id)
        if not shard_is_complete(
            path,
            schema_signature=str(manifest["schema_signature"]),
            verify_checksums=False,
        ):
            raise ValueError(f"Final shard is missing or invalid: {path}.")
        if config.verify_shard_checksums:
            validate_indexed_shard(path, deep=False)
        reader = IndexedTarReader(path)
        try:
            expected = _expected_names(manifest, shard)
            actual = set(reader.entries)
            missing = expected - actual
            unexpected = actual - expected
            if missing or unexpected:
                raise ValueError(
                    f"Shard {shard_id} member mismatch: missing={len(missing)}, "
                    f"unexpected={len(unexpected)}, examples="
                    f"{sorted(missing)[:3]}/{sorted(unexpected)[:3]}."
                )
            actual_member_count += len(actual)
            for name, index_entry in reader.entries.items():
                payload_bytes[_suffix(name)] += int(index_entry.size)
            index_path, completion_path = shard_sidecars(path)
            container_bytes += (
                path.stat().st_size
                + index_path.stat().st_size
                + completion_path.stat().st_size
            )
        finally:
            # A full dataset has millions of indexed members. Retaining every
            # parsed index until the payload scan would needlessly consume many
            # GiB, so accounting is deliberately one shard at a time.
            reader.close()

    scan_locations = _sample_locations(manifest, config.random_samples, config.seed)
    geometry_sample_indices = {
        int(entry["global_retained_start"]) + retained_index
        for entry, retained_index in scan_locations
    }
    if config.full_payload_scan:
        scan_iterator = (
            (entry, retained_index)
            for entry in manifest["episodes"]
            for retained_index in range(int(entry["retained_count"]))
        )
        scan_count = int(manifest["counts"]["retained_timesteps"])
    else:
        scan_iterator = iter(scan_locations)
        scan_count = len(scan_locations)
    depth_sample = BoundedPrioritySample(capacity=2_000_000, seed=config.seed)
    depth_valid_fractions = _RunningScalar()
    flow_valid_fractions = _RunningScalar()
    support_fractions = _RunningScalar()
    lager_record_schemas: Counter[str] = Counter()
    exceptional_translation_targets = 0
    relaxed_safety_targets = 0
    readers: OrderedDict[int, IndexedTarReader] = OrderedDict()

    def cached_reader(shard_id: int) -> IndexedTarReader:
        identifier = int(shard_id)
        if identifier in readers:
            value = readers.pop(identifier)
            readers[identifier] = value
            return value
        value = IndexedTarReader(shard_path(root, "final", identifier))
        readers[identifier] = value
        while len(readers) > 4:
            _evicted_id, evicted = readers.popitem(last=False)
            evicted.close()
        return value

    try:
        for entry, retained_index in scan_iterator:
            result = _validate_payloads(
                cached_reader(int(entry["shard_id"])),
                entry,
                retained_index,
                numeric,
                decode_jpegs=(config.decode_all_jpegs or not config.full_payload_scan),
                verify_depth_geometry=(
                    int(entry["global_retained_start"]) + retained_index
                    in geometry_sample_indices
                ),
            )
            depth_valid_fractions.add(result["depth_valid_fraction"])
            depth_sample.add(result["valid_depth_values"], maximum_per_item=64)
            if result["flow_valid_fraction"] is not None:
                flow_valid_fractions.add(result["flow_valid_fraction"])
            support_fractions.add(result["lager_support_fraction"])
            lager_record_schemas[str(result["lager_record_schema"])] += 1
            exceptional_translation_targets += int(
                result["lager_exceptional_translation_targets"]
            )
            relaxed_safety_targets += int(result["lager_relaxed_safety_targets"])
    finally:
        for reader in readers.values():
            reader.close()

    sampled_depth = depth_sample.result()
    if not len(sampled_depth):
        raise ValueError("No positive cached DA3 depths were decoded.")
    loader = _loader_check(root, config.loader_windows, config.seed)
    report = {
        "schema_version": 1,
        "status": "passed",
        "root": str(root),
        "dataset_schema_signature": manifest["schema_signature"],
        "counts": manifest["counts"],
        "expected_shards": expected_shards,
        "actual_shards": len(final_tar_paths),
        "actual_indexed_members": actual_member_count,
        "unexpected_partial_files": partials,
        "archived_partial_files": archived_partials,
        "payload_bytes_by_suffix": dict(sorted(payload_bytes.items())),
        "payload_bytes_total": int(sum(payload_bytes.values())),
        "container_index_and_sidecar_bytes": container_bytes,
        "container_overhead_bytes": container_bytes - int(sum(payload_bytes.values())),
        "samples_payload_validated": scan_count,
        "full_payload_scan": bool(config.full_payload_scan),
        "all_jpegs_decoded": bool(config.decode_all_jpegs),
        "depth": {
            "valid_fraction_mean": depth_valid_fractions.mean,
            "valid_fraction_min": depth_valid_fractions.minimum_or_none,
            "bounded_distribution_sample_size": int(sampled_depth.size),
            "maximum_distribution_candidates_per_timestamp": 64,
            "positive_depth_percentiles_m": {
                str(value): float(np.percentile(sampled_depth, value))
                for value in (0.1, 1.0, 5.0, 50.0, 95.0, 99.0, 99.9)
            },
        },
        "flow": {
            "valid_fraction_mean": flow_valid_fractions.mean,
            "valid_fraction_min": flow_valid_fractions.minimum_or_none,
        },
        "lagernvs": {
            "targets_per_timestep": 4,
            "support_fraction_mean": support_fractions.mean,
            "support_fraction_min": support_fractions.minimum_or_none,
            "record_timestamps_by_schema": dict(sorted(lager_record_schemas.items())),
            "exceptional_translation_targets": exceptional_translation_targets,
            "relaxed_safety_threshold_targets": relaxed_safety_targets,
            "independent_depth_geometry_timestamps_checked": len(
                geometry_sample_indices
            ),
        },
        "loader": loader,
        "elapsed_seconds": time.perf_counter() - started,
    }
    return report


def write_final_manifest(
    root: str | os.PathLike[str],
    integrity_report: dict[str, Any],
) -> Path:
    dataset_root = Path(root).expanduser().resolve()
    manifest = load_stage0_manifest(dataset_root)
    validate_derived_root(dataset_root, manifest["source_root"])
    if integrity_report.get("status") != "passed":
        raise ValueError("A passing integrity report is required for finalization.")
    if integrity_report.get("dataset_schema_signature") != manifest["schema_signature"]:
        raise ValueError("Integrity report belongs to a different dataset schema.")
    stage_metadata = {}
    for path in sorted((dataset_root / "metadata" / "stages").glob("*.json")):
        stage_metadata[path.name] = json.loads(path.read_text(encoding="utf-8"))
    environment_metadata = {}
    for path in sorted((dataset_root / "metadata" / "environments").glob("*.json")):
        environment_metadata[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    required_environments = {"training", "da3", "megaflow", "lagernvs"}
    if (
        manifest["mode"] == "full"
        and set(environment_metadata) != required_environments
    ):
        raise FileNotFoundError(
            "Full dataset finalization requires exact training/DA3/MegaFlow/"
            "LagerNVS environment inventories."
        )
    source_verification = None
    source_before_path = dataset_root / "metadata" / "source-tree-before-full-run.json"
    source_after_path = dataset_root / "metadata" / "source-tree-after-full-run.json"
    if source_before_path.is_file() and source_after_path.is_file():
        source_before = json.loads(source_before_path.read_text(encoding="utf-8"))
        source_after = json.loads(source_after_path.read_text(encoding="utf-8"))
        if source_before != source_after:
            raise ValueError(
                "DROID source fingerprints differ before and after preprocessing."
            )
        source_verification = {
            "unchanged": True,
            "before": source_before,
            "after": source_after,
        }
    elif manifest["mode"] == "full":
        raise FileNotFoundError(
            "Full dataset finalization requires before/after DROID source fingerprints."
        )
    pose_contract_path = (
        dataset_root / "metadata" / "lagernvs-pose-safety-contract.json"
    )
    if not pose_contract_path.is_file():
        raise FileNotFoundError(
            f"LagerNVS pose-safety contract is missing: {pose_contract_path}"
        )
    pose_contract = json.loads(pose_contract_path.read_text(encoding="utf-8"))
    if pose_contract.get("dataset_schema_signature") != manifest["schema_signature"]:
        raise ValueError("LagerNVS pose-safety contract belongs to another dataset.")
    payload = {
        "schema_version": 1,
        "dataset_schema_signature": manifest["schema_signature"],
        "counts": manifest["counts"],
        "encoding": manifest["encoding"],
        "foundation_models": manifest["foundation_models"],
        "shard_count": len(manifest["shards"]),
        "integrity": integrity_report,
        "lagernvs_pose_safety_contract": pose_contract,
        "stage_provenance": stage_metadata,
        "environment_provenance": environment_metadata,
        "source_read_only_verification": source_verification,
        "finalized_unix": time.time(),
    }
    path = dataset_root / "manifests" / "final.json"
    write_json_atomic(path, payload)
    return path
