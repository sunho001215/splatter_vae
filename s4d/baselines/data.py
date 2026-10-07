"""Data adapters: the splatter4d Meta-World HDF5 layout -> the sample dicts of the reference baseline dataloaders.

Only the data interface changes; every sample has the shapes, value ranges and camera conventions that the
reference ``baselines/{SinCro,ReViWo}/dataloader.py`` (sunho001215/splatter_vae @ c0abf56) produced:

- SinCro: ``images`` (T, H, V, W, 3) float in [0, 1]; ``K`` (V, 3, 3) with the principal point at ``(W-1)/2``
  (integer pixel centres, as the reference collector and NeRF's ``get_rays`` assume); ``c2w`` (V, 4, 4) the
  camera-to-world transform in the MuJoCo/OpenGL camera frame (-z forward, +y up). Our files store K in
  continuous pixel coordinates (principal point ``W/2``) and OpenCV ``c2w``; both are converted here.
- ReViWo: ``images`` (V, 3, H, W) float in [-1, 1], one state per sample.

Only the six training cameras are used, episodes come from the saved split manifest, and SinCro frames are
``frame_spacing`` simulator steps apart (2: the RL observation spacing, decided by the user; the reference used
consecutive frames).
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

GL_FROM_CV = np.diag([1.0, -1.0, -1.0, 1.0])


def split_episodes(manifest: str | Path, split: str) -> list[str]:
    """Episodes of ``split`` ("train" or "validation") from a saved ``splits/<task>_seed<s>.json`` manifest."""
    return list(json.loads(Path(manifest).read_text())[split])


def reference_cameras(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """(K, c2w) of the training cameras in the reference conventions (integer pixel centres, OpenGL frame)."""
    with h5py.File(path, "r") as f:
        cams = f["cameras"]
        train = np.where(cams["is_train"][:].astype(bool))[0]
        if not np.array_equal(train, np.arange(len(train))):
            raise ValueError("training cameras must be stored first in the camera table")
        K = cams["K"][:][train].astype(np.float64)
        c2w_cv = cams["c2w"][:][train].astype(np.float64)
    K[:, 0, 2] -= 0.5
    K[:, 1, 2] -= 0.5
    return K.astype(np.float32), (c2w_cv @ GL_FROM_CV).astype(np.float32)


class _EpisodeFile(Dataset):
    """Reference caps: the first ``max_episodes`` episodes, the first ``max_frames_per_demo`` frames of each."""

    def __init__(self, path: str | Path, episodes: list[str], max_episodes: int | None, max_frames_per_demo: int | None):
        self.path, self.episodes = str(path), list(episodes)[:max_episodes]
        self._file: h5py.File | None = None
        with h5py.File(self.path, "r") as f:
            self.num_train_cams = int(f["cameras"]["is_train"][:].astype(bool).sum())
            self.lengths = {ep: int(f["episodes"][ep].attrs["length"]) for ep in self.episodes}
        if max_frames_per_demo is not None:
            self.lengths = {ep: min(length, int(max_frames_per_demo)) for ep, length in self.lengths.items()}

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_file"] = None
        return state

    def _rgb(self, ep: str):
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        return self._file["episodes"][ep]["rgb"]


class SinCroWindows(_EpisodeFile):
    """Reference ``MetaWorldSinCroSequenceDataset``: one sample per window start, every ``temporal_stride`` frames."""

    def __init__(
        self,
        path: str | Path,
        episodes: list[str],
        sequence_length: int = 3,
        temporal_stride: int = 3,
        frame_spacing: int = 2,
        num_views: int = 6,
        max_episodes: int | None = None,
        max_frames_per_demo: int | None = None,
    ):
        super().__init__(path, episodes, max_episodes, max_frames_per_demo)
        if num_views > self.num_train_cams:
            raise ValueError(f"num_views={num_views} exceeds the {self.num_train_cams} training cameras")
        self.sequence_length, self.frame_spacing, self.num_views = sequence_length, frame_spacing, num_views
        self.K, self.c2w = (torch.from_numpy(a[:num_views]) for a in reference_cameras(path))
        span = (sequence_length - 1) * frame_spacing
        self.indices = [(ep, start) for ep in self.episodes for start in range(0, self.lengths[ep] - span, temporal_stride)]
        if not self.indices:
            raise ValueError("no SinCro windows: sequences are longer than the episodes")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict:
        ep, start = self.indices[index]
        frames = [start + k * self.frame_spacing for k in range(self.sequence_length)]
        rgb = np.stack([self._rgb(ep)[t, : self.num_views] for t in frames])  # (T, V, H, W, 3) uint8
        images = torch.from_numpy(rgb).float().div(255.0).permute(0, 2, 1, 3, 4)  # (T, H, V, W, 3)
        return {
            "images": images.contiguous(),
            "K": self.K.clone(),
            "c2w": self.c2w.clone(),
            "demo_key": ep,
            "hdf5_path": self.path,
        }


class ReViWoStates(_EpisodeFile):
    """Reference ``MetaWorldMultiViewAllCamerasHDF5Dataset``: every timestep, all training cameras, in [-1, 1]."""

    def __init__(
        self,
        path: str | Path,
        episodes: list[str],
        num_views: int = 6,
        max_episodes: int | None = None,
        max_frames_per_demo: int | None = None,
    ):
        super().__init__(path, episodes, max_episodes, max_frames_per_demo)
        if num_views > self.num_train_cams:
            raise ValueError(f"num_views={num_views} exceeds the {self.num_train_cams} training cameras")
        self.num_views = num_views
        self.indices = [(ep, t) for ep in self.episodes for t in range(self.lengths[ep])]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict:
        ep, t = self.indices[index]
        rgb = torch.from_numpy(self._rgb(ep)[t, : self.num_views])  # (V, H, W, 3) uint8
        images = rgb.float().div(255.0).mul(2.0).sub(1.0).permute(0, 3, 1, 2)  # (V, 3, H, W) in [-1, 1]
        return {"images": images.contiguous(), "demo_key": ep, "t": t, "hdf5_path": self.path}
