"""Adapter interface. A new recording format needs a new subclass and a config file, nothing else."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from s4d.realworld.contract import EpisodeMeta

_REGISTRY: dict = {}


def register_adapter(cls):
    _REGISTRY[cls.name] = cls
    return cls


def get_adapter(name: str, config: dict) -> EpisodeAdapter:
    if name not in _REGISTRY:
        import s4d.realworld.adapters.synthetic_adapter  # noqa: F401  (registers)
        import s4d.realworld.adapters.template  # noqa: F401
    try:
        return _REGISTRY[name](config)
    except KeyError:
        raise KeyError(f"unknown adapter {name!r}; registered: {sorted(_REGISTRY)}") from None


class EpisodeAdapter(ABC):
    """Reads one recording format and yields contract objects. Adapters only READ the source recording.

    Required behaviour:
    * ``list_episodes`` returns stable, path-safe episode ids.
    * ``read_meta`` returns a complete ``EpisodeMeta``: every stream with its true modality (an infrared stream is
      ``modality="infrared"``, 1 channel, even if a viewer shows it as RGB), the intrinsics exactly as recorded
      plus their pixel convention, the stereo geometry (status ``missing`` if the recording has none - never a
      guessed baseline), robot state with units, the robot description and provenance with source file hashes.
      Anything the adapter had to assume goes into ``EpisodeMeta.assumptions``.
    * ``iter_frames`` yields ``(frame_index, timestamp_ns, array)`` exactly as recorded (dtype and channels
      untouched; GRAY (H,W), RGB/BGR (H,W,3)), in timestamp order, so bounded chunks can be processed.
    """

    name = "base"
    version = "0"

    def __init__(self, config: dict):
        self.config = dict(config)

    @abstractmethod
    def list_episodes(self) -> list[str]: ...

    @abstractmethod
    def read_meta(self, episode_id: str) -> EpisodeMeta: ...

    @abstractmethod
    def iter_frames(self, episode_id: str, stream_id: str, start: int = 0,
                    stop: int | None = None) -> Iterator[tuple[int, int, np.ndarray]]: ...

    def source_files(self, episode_id: str) -> list[Path]:
        """Files hashed into the provenance (default: none)."""
        return []
