from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .safety import DEFAULT_DROID_ROOT, validate_derived_root

CACHE_FORMAT_VERSION = 1


def calibration_manifest_version(path: str | os.PathLike[str]) -> str:
    """Content identity used to bind every derived cache to one manifest."""
    manifest = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with manifest.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return f"{manifest.name}:sha256:{digest.hexdigest()}"


def cache_item_key(
    episode_id: str, camera_id: str, frame_index: int, *, gap: int | None = None
) -> str:
    parts = [str(episode_id), str(camera_id), f"frame={int(frame_index)}"]
    if gap is not None:
        parts.append(f"gap={int(gap)}")
    return "|".join(parts)


def sequence_cache_key(episode_id: str, camera_id: str) -> str:
    return f"{episode_id}|{camera_id}|sequence"


def _group_name(key: str) -> str:
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CacheProvenance:
    teacher_name: str
    checkpoint: str
    teacher_version: str
    calibration_version: str
    preprocessing_version: str
    resolution: tuple[int, int]
    cache_format_version: int = CACHE_FORMAT_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {
            "teacher_name": self.teacher_name,
            "checkpoint": self.checkpoint,
            "teacher_version": self.teacher_version,
            "calibration_version": self.calibration_version,
            "preprocessing_version": self.preprocessing_version,
            "resolution": list(self.resolution),
            "cache_format_version": int(self.cache_format_version),
        }


class HDF5ShardWriter:
    """Write many cache items into bounded HDF5 shards plus one compact index."""

    def __init__(
        self,
        cache_root: str | os.PathLike[str],
        provenance: CacheProvenance,
        *,
        shard_prefix: str,
        items_per_shard: int = 2048,
        droid_root: str | os.PathLike[str] = DEFAULT_DROID_ROOT,
    ):
        if int(items_per_shard) <= 0:
            raise ValueError("items_per_shard must be positive.")
        self.root = validate_derived_root(cache_root, droid_root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.provenance = provenance
        self.shard_prefix = str(shard_prefix)
        self.items_per_shard = int(items_per_shard)
        self.index: dict[str, dict[str, Any]] = {}
        self._shard_index = -1
        self._items_in_shard = 0
        self._handle: h5py.File | None = None
        self._shard_path: Path | None = None

    def _open_next_shard(self) -> None:
        if self._handle is not None:
            self._handle.flush()
            self._handle.close()
        self._shard_index += 1
        self._items_in_shard = 0
        self._shard_path = self.root / f"{self.shard_prefix}-{self._shard_index:06d}.h5"
        if self._shard_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing cache shard {self._shard_path}."
            )
        self._handle = h5py.File(self._shard_path, "w")
        for key, value in self.provenance.as_dict().items():
            self._handle.attrs[key] = (
                json.dumps(value) if isinstance(value, list) else value
            )

    def add(
        self,
        key: str,
        arrays: Mapping[str, np.ndarray],
        metadata: Mapping[str, Any],
    ) -> None:
        if key in self.index:
            raise ValueError(f"Duplicate cache key: {key}")
        if not arrays:
            raise ValueError("A cache item must contain at least one array.")
        if self._handle is None or self._items_in_shard >= self.items_per_shard:
            self._open_next_shard()
        assert self._handle is not None and self._shard_path is not None
        group_name = _group_name(key)
        group = self._handle.create_group(f"items/{group_name}")
        group.attrs["key"] = key
        group.attrs["metadata"] = json.dumps(dict(metadata), separators=(",", ":"))
        for name, value in arrays.items():
            array = np.asarray(value)
            if array.dtype == np.dtype("O"):
                raise TypeError(f"Object arrays are not supported in caches ({name}).")
            compression = "lzf" if array.ndim > 0 and array.size > 256 else None
            group.create_dataset(name, data=array, compression=compression)
        self.index[key] = {
            "shard": self._shard_path.name,
            "group": f"items/{group_name}",
            "metadata": dict(metadata),
        }
        self._items_in_shard += 1

    def close(self) -> Path:
        if self._handle is not None:
            self._handle.flush()
            self._handle.close()
            self._handle = None
        payload = {
            "provenance": self.provenance.as_dict(),
            "items": self.index,
        }
        destination = self.root / f"{self.shard_prefix}-index.json"
        temporary = destination.with_suffix(".json.partial")
        temporary.write_text(
            json.dumps(payload, separators=(",", ":")), encoding="utf-8"
        )
        os.replace(temporary, destination)
        return destination

    def __enter__(self) -> HDF5ShardWriter:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is None:
            self.close()
        elif self._handle is not None:
            self._handle.close()
            self._handle = None


class HDF5CacheReader:
    def __init__(
        self,
        index_path: str | os.PathLike[str],
        *,
        expected: CacheProvenance | None = None,
    ):
        self.index_path = Path(index_path).expanduser().resolve()
        payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        self.provenance = dict(payload["provenance"])
        self.items: dict[str, dict[str, Any]] = dict(payload["items"])
        if expected is not None and self.provenance != expected.as_dict():
            raise ValueError(
                "Cache provenance mismatch; refusing to mix teacher/checkpoint/preprocessing versions. "
                f"expected={expected.as_dict()}, found={self.provenance}"
            )
        if int(self.provenance.get("cache_format_version", -1)) != CACHE_FORMAT_VERSION:
            raise ValueError("Unsupported cache format version.")
        self._handles: dict[str, h5py.File] = {}

    def require_compatible(
        self,
        *,
        teacher_name: str,
        calibration_version: str,
        resolution: tuple[int, int] = (320, 180),
        checkpoint: str | None = None,
    ) -> None:
        """Reject caches from a different teacher, calibration, grid, or checkpoint."""
        expected = {
            "teacher_name": str(teacher_name),
            "calibration_version": str(calibration_version),
            "resolution": list(map(int, resolution)),
        }
        if checkpoint is not None:
            expected["checkpoint"] = str(checkpoint)
        mismatches = {
            key: {"expected": value, "found": self.provenance.get(key)}
            for key, value in expected.items()
            if self.provenance.get(key) != value
        }
        if mismatches:
            raise ValueError(
                "Cache is incompatible with this DROID run; refusing to mix provenance: "
                f"{mismatches}"
            )

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_handles"] = {}
        return state

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __contains__(self, key: str) -> bool:
        return key in self.items

    def metadata(self, key: str) -> dict[str, Any]:
        return dict(self.items[key].get("metadata", {}))

    def read(self, key: str) -> dict[str, np.ndarray]:
        group = self._group(key)
        return {name: np.asarray(dataset) for name, dataset in group.items()}

    def _group(self, key: str) -> h5py.Group:
        location = self.items.get(key)
        if location is None:
            raise KeyError(f"Cache item does not exist: {key}")
        shard_name = str(location["shard"])
        if shard_name not in self._handles:
            self._handles[shard_name] = h5py.File(
                self.index_path.parent / shard_name, "r"
            )
        group = self._handles[shard_name][str(location["group"])]
        if str(group.attrs["key"]) != key:
            raise RuntimeError("Cache index/group key mismatch.")
        return group

    def read_depth(
        self, episode_id: str, camera_id: str, frame_index: int
    ) -> dict[str, np.ndarray]:
        frame_key = cache_item_key(episode_id, camera_id, frame_index)
        if frame_key in self:
            item = self.read(frame_key)
        else:
            group = self._group(sequence_cache_key(episode_id, camera_id))
            frame = int(frame_index)
            item = {
                name: np.asarray(group[name][frame])
                for name in ("metric_depth", "confidence", "validity")
                if name in group
            }
        required = {"metric_depth", "confidence", "validity"}
        missing = required.difference(item)
        if missing:
            raise KeyError(f"Depth cache item is missing {sorted(missing)}.")
        return item

    def read_flow(
        self,
        episode_id: str,
        camera_id: str,
        frame_index: int,
        gap: int,
    ) -> dict[str, np.ndarray]:
        frame_key = cache_item_key(episode_id, camera_id, frame_index, gap=gap)
        if frame_key in self:
            item = self.read(frame_key)
        else:
            group = self._group(sequence_cache_key(episode_id, camera_id))
            suffix = f"gap_{int(gap)}"
            frame = int(frame_index)
            names = {
                "forward_flow": f"forward_flow_{suffix}",
                "validity": f"validity_{suffix}",
                "confidence": f"confidence_{suffix}",
                "uncertainty": f"uncertainty_{suffix}",
                "forward_backward_error": f"forward_backward_error_{suffix}",
            }
            item = {
                output_name: np.asarray(group[stored_name][frame])
                for output_name, stored_name in names.items()
                if stored_name in group
            }
        required = {"forward_flow", "validity"}
        missing = required.difference(item)
        if missing:
            raise KeyError(f"Flow cache item is missing {sorted(missing)}.")
        return item

    def read_synthetic_view(
        self,
        episode_id: str,
        frame_index: int,
        *,
        occurrence: int = 0,
    ) -> dict[str, np.ndarray] | None:
        key = sequence_cache_key(episode_id, "see3d")
        if key not in self:
            return None
        group = self._group(key)
        timesteps = np.asarray(group["timestep"], dtype=np.int64)
        matches = np.flatnonzero(timesteps == int(frame_index))
        if len(matches) == 0:
            return None
        selected = int(matches[int(occurrence) % len(matches)])
        return {name: np.asarray(dataset[selected]) for name, dataset in group.items()}
