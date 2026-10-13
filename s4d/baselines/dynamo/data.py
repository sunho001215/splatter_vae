"""DynaMo data adapter: our Meta-World HDF5 -> the windows of the official ``TrajectorySlicerDataset``.

Every training camera's video of every split episode is one single-camera trajectory (V = 1). Windows follow the
official slicer (``datasets/core.py:128-141`` and ``:208-211``, ``frame_step`` path, no future conditioning, no
extra actions): for a trajectory of ``T`` frames, ``span = (window - 1) * frame_step + 1`` and the windows are
``range(start, start + span, frame_step)`` for ``start in range(T - span)``. ``range`` excludes ``T - span``, so
the last complete window of each trajectory is never used; that upstream off-by-one is kept. Trajectories shorter
than ``window`` (upstream compares with ``window``, not ``span``) contribute nothing.

A sample is the window's frames as ``(window, 1, 3, H, W)`` uint8. ``to_official_obs`` turns a batch into the official
float input in [0, 1] on the device: ``obs / 255.0`` computed on the CPU, as upstream datasets do in their loaders
(``datasets/your_dataset.py:18``), bit for bit (a 256-entry table; CUDA's division by a scalar multiplies by the
reciprocal, which can differ in the last bit). Only the training cameras are used; the HDF5 file is opened lazily in
each worker.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch

from s4d.baselines.data import _EpisodeFile


def official_slices(lengths: list[int], window: int, frame_step: int) -> list[tuple[int, int, int]]:
    """``TrajectorySlicerDataset.slices`` for trajectories of these lengths: ``(trajectory, start, end)``."""
    span = (window - 1) * frame_step + 1
    slices = []
    for i, length in enumerate(lengths):
        if length - window < 0:
            continue
        slices += [(i, start, start + span) for start in range(length - span)]
    return slices


class DynaMoWindows(_EpisodeFile):
    """Single-camera windows of ``window`` frames, ``frame_step`` simulator steps apart, over (episode, camera)."""

    def __init__(
        self,
        path: str | Path,
        episodes: list[str],
        window: int = 5,
        frame_step: int = 2,
        max_episodes: int | None = None,
    ):
        super().__init__(path, episodes, max_episodes, None)
        self.window, self.frame_step = int(window), int(frame_step)
        self.cameras = list(range(self.num_train_cams))  # training cameras are stored first (checked below)
        with h5py.File(self.path, "r") as f:
            is_train = f["cameras"]["is_train"][:].astype(bool)
        if not np.array_equal(np.where(is_train)[0], np.arange(self.num_train_cams)):
            raise ValueError("training cameras must be stored first in the camera table")
        self.trajectories = [(ep, cam) for ep in self.episodes for cam in self.cameras]
        self.slices = official_slices([self.lengths[ep] for ep, _ in self.trajectories], self.window, self.frame_step)
        if not self.slices:
            raise ValueError("no DynaMo windows: every trajectory is too short")

    def __len__(self) -> int:
        return len(self.slices)

    def window_frames(self, index: int) -> tuple[str, int, list[int]]:
        """(episode, camera, simulator frame indices) of a window."""
        i, start, end = self.slices[index]
        ep, cam = self.trajectories[i]
        return ep, cam, list(range(start, end, self.frame_step))

    def __getitem__(self, index: int) -> torch.Tensor:
        i, start, end = self.slices[index]
        ep, cam = self.trajectories[i]
        rgb = self._rgb(ep)[start:end:self.frame_step, cam]  # (window, H, W, 3) uint8
        return torch.from_numpy(np.ascontiguousarray(rgb)).permute(0, 3, 1, 2).unsqueeze(1).contiguous()


UNIT = torch.arange(256, dtype=torch.float32) / 255.0  # CPU true division: the official per-value result


def to_official_obs(batch: torch.Tensor, device: torch.device | str) -> torch.Tensor:
    """(N, T, V, 3, H, W) uint8 -> the official float observation ``obs / 255.0`` in [0, 1] on ``device``."""
    if batch.dtype != torch.uint8:
        raise TypeError(f"expected uint8 frames, got {batch.dtype}")
    return UNIT.to(device)[batch.to(device, non_blocking=True).long()]
