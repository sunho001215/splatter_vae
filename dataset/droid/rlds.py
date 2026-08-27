from __future__ import annotations

import json
import os
from collections import OrderedDict
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .calibration import RLDSEpisodeMetadata
from .safety import DEFAULT_DROID_ROOT, validate_derived_root

EXTERIOR_IMAGE_KEYS = (
    "exterior_image_1_left",
    "exterior_image_2_left",
)


def _load_tensorflow_stack():
    try:
        import tensorflow as tf
        import tensorflow_datasets as tfds
    except ImportError as exc:
        raise RuntimeError(
            "DROID RLDS access requires tensorflow and tensorflow-datasets. "
            "Install the project preprocessing dependencies."
        ) from exc
    # DROID decoding is CPU I/O. Never let TensorFlow reserve the PyTorch GPU.
    tf.config.set_visible_devices([], "GPU")
    return tf, tfds


def _canonical_read_config(tf: Any, tfds: Any) -> Any:
    """Return the one canonical TFDS ordering used by DROID manifests.

    DROID is sharded across 2,048 TFRecords.  TFDS otherwise interleaves blocks
    from many shards, so an ordinal obtained by enumerating the complete split
    does not identify the same episode as an absolute split slice.  Reading one
    shard at a time makes the complete-split order identical to TFDS absolute
    slicing, while explicit deterministic options keep that address stable.
    """

    options = tf.data.Options()
    options.deterministic = True
    return tfds.ReadConfig(
        options=options,
        try_autocache=False,
        interleave_cycle_length=1,
        interleave_block_length=1,
        num_parallel_calls_for_interleave_files=1,
        num_parallel_calls_for_decode=1,
        skip_prefetch=True,
    )


def find_tfds_builder_directory(droid_root: str | os.PathLike[str]) -> Path:
    root = Path(droid_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"DROID RLDS root does not exist: {root}")
    if (root / "dataset_info.json").is_file():
        return root
    candidates = sorted(root.rglob("dataset_info.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No TFDS dataset_info.json was found under DROID root {root}."
        )
    droid_candidates = [
        path.parent for path in candidates if "droid" in path.parent.as_posix().lower()
    ]
    directories = droid_candidates or [path.parent for path in candidates]
    if len(directories) != 1:
        raise ValueError(
            f"Ambiguous TFDS builder directories under {root}: {[str(path) for path in directories]}"
        )
    return directories[0]


def _decode_text(value: Any) -> str:
    if hasattr(value, "numpy"):
        value = value.numpy()
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _episode_length(steps: Any) -> int:
    cardinality = steps.cardinality()
    if hasattr(cardinality, "numpy"):
        value = int(cardinality.numpy())
        if value >= 0:
            return value
    return sum(1 for _ in steps)


def scan_rlds_episode_metadata(
    droid_root: str | os.PathLike[str],
    *,
    splits: Iterable[str] | None = None,
    maximum_episodes: int | None = None,
) -> Iterator[RLDSEpisodeMetadata]:
    """Read only episode metadata and ordinals from a local DROID TFDS release."""
    if maximum_episodes is not None and int(maximum_episodes) <= 0:
        return
    _tf, tfds = _load_tensorflow_stack()
    builder = tfds.builder_from_directory(str(find_tfds_builder_directory(droid_root)))
    selected_splits = (
        list(splits) if splits is not None else sorted(builder.info.splits)
    )
    inspected = 0
    for split in selected_splits:
        dataset = builder.as_dataset(
            split=split,
            shuffle_files=False,
            read_config=_canonical_read_config(_tf, tfds),
        )
        for ordinal, episode in enumerate(dataset):
            metadata = episode["episode_metadata"]
            yield RLDSEpisodeMetadata(
                rlds_split=str(split),
                rlds_ordinal=int(ordinal),
                file_path=_decode_text(metadata["file_path"]),
                recording_folderpath=_decode_text(metadata["recording_folderpath"]),
                num_steps=_episode_length(episode["steps"]),
            )
            inspected += 1
            if maximum_episodes is not None and inspected >= int(maximum_episodes):
                return


def write_rlds_metadata_index(
    entries: Iterable[RLDSEpisodeMetadata],
    output_path: str | os.PathLike[str],
    *,
    droid_root: str | os.PathLike[str] = DEFAULT_DROID_ROOT,
) -> int:
    destination = Path(output_path).expanduser().resolve(strict=False)
    validate_derived_root(destination.parent, droid_root).mkdir(
        parents=True, exist_ok=True
    )
    temporary = destination.with_suffix(destination.suffix + ".partial")
    count = 0
    with temporary.open("w", encoding="utf-8") as stream:
        for entry in entries:
            stream.write(json.dumps(entry.__dict__, separators=(",", ":")) + "\n")
            count += 1
    os.replace(temporary, destination)
    return count


def load_rlds_metadata_index(path: str | os.PathLike[str]) -> list[RLDSEpisodeMetadata]:
    entries = []
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                entries.append(RLDSEpisodeMetadata(**json.loads(line)))
    return entries


class EpisodeBackend(Protocol):
    def get_episode(self, split: str, ordinal: int) -> Mapping[str, Any]: ...


class TFDSRLDSBackend:
    """Process-local random episode access with a small episode LRU.

    DROID manifest ordinals are defined by the canonical, sequential-shard
    ``as_dataset`` scan above.  TFDS's default parallel shard interleave and
    ``as_data_source`` do not share the absolute-slice ordering for this
    release and therefore must not be used to resolve those ordinals.  The
    episode LRU and grouped sampler amortize materialization.
    """

    def __init__(self, droid_root: str | os.PathLike[str], cache_size: int = 2):
        self.droid_root = str(Path(droid_root).expanduser().resolve())
        self.builder_dir = str(find_tfds_builder_directory(self.droid_root))
        self.cache_size = max(1, int(cache_size))
        self._builder = None
        self._cache: OrderedDict[tuple[str, int], Mapping[str, Any]] = OrderedDict()

    def __getstate__(self):
        state = dict(self.__dict__)
        state.update({"_builder": None, "_cache": OrderedDict()})
        return state

    def _ensure_builder(self):
        if self._builder is None:
            _tf, tfds = _load_tensorflow_stack()
            self._builder = tfds.builder_from_directory(self.builder_dir)
        return self._builder

    def _load_raw_episode(self, split: str, ordinal: int) -> Any:
        builder = self._ensure_builder()
        _tf, tfds = _load_tensorflow_stack()
        # A TFDS absolute slice resolves the target shard and only skips within
        # that shard. Calling ``.skip(ordinal)`` on the whole 95k-episode split
        # would reread all preceding ~1.7 TiB records for late ordinals.
        split_slice = f"{split}[{int(ordinal)}:{int(ordinal) + 1}]"
        dataset = builder.as_dataset(
            split=split_slice,
            shuffle_files=False,
            read_config=_canonical_read_config(_tf, tfds),
        )
        try:
            return next(iter(tfds.as_numpy(dataset)))
        except StopIteration as exc:
            raise IndexError(
                f"RLDS episode ordinal {ordinal} is outside split {split}."
            ) from exc

    @staticmethod
    def _materialize(raw: Any) -> Mapping[str, Any]:
        _tf, tfds = _load_tensorflow_stack()
        if isinstance(raw, Mapping):
            episode = raw
        else:
            episode = tfds.as_numpy(raw)
        steps_value = episode["steps"]
        if isinstance(steps_value, Mapping):
            steps = steps_value
        else:
            step_list = list(steps_value)
            if not step_list:
                raise ValueError("RLDS episode has no steps.")
            steps = {
                "observation": {
                    key: np.stack([step["observation"][key] for step in step_list])
                    for key in EXTERIOR_IMAGE_KEYS
                }
            }
            for optional in ("action", "action_dict"):
                if optional in step_list[0]:
                    if isinstance(step_list[0][optional], Mapping):
                        steps[optional] = {
                            key: np.stack([step[optional][key] for step in step_list])
                            for key in step_list[0][optional]
                        }
                    else:
                        steps[optional] = np.stack(
                            [step[optional] for step in step_list]
                        )
            for optional in (
                "cartesian_position",
                "gripper_position",
                "joint_position",
            ):
                if optional in step_list[0]["observation"]:
                    steps["observation"][optional] = np.stack(
                        [step["observation"][optional] for step in step_list]
                    )
        observation = steps["observation"]
        output: dict[str, Any] = {
            "images": np.stack(
                [
                    np.asarray(observation[key], dtype=np.uint8)
                    for key in EXTERIOR_IMAGE_KEYS
                ],
                axis=1,
            )
        }
        for optional in ("action", "action_dict"):
            if optional in steps:
                output[optional] = steps[optional]
        for optional in ("cartesian_position", "gripper_position", "joint_position"):
            if optional in observation:
                output[optional] = np.asarray(observation[optional])
        return output

    def get_episode(self, split: str, ordinal: int) -> Mapping[str, Any]:
        key = (str(split), int(ordinal))
        if key in self._cache:
            value = self._cache.pop(key)
            self._cache[key] = value
            return value
        value = self._materialize(self._load_raw_episode(*key))
        self._cache[key] = value
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return value


class MemoryEpisodeBackend:
    """Small deterministic backend used by unit and smoke tests."""

    def __init__(self, episodes: Mapping[tuple[str, int], Mapping[str, Any]]):
        self.episodes = dict(episodes)

    def get_episode(self, split: str, ordinal: int) -> Mapping[str, Any]:
        return self.episodes[(str(split), int(ordinal))]
