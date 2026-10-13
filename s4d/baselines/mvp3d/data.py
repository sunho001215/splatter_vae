"""3D-MVP pretraining data from our Meta-World HDF5 files (the paper's in-domain "pretrain on RLBench point clouds"
ablation, §4.4, with our data in place of RLBench).

One sample is one time step of a training-split episode and a random non-empty subset of the six training cameras:
its RGB-D images, K (continuous pixel coordinates, principal point W/2) and OpenCV c2w as stored. The subset size is
uniform in 1..6 and the subset uniform given its size, so single-camera clouds (the RL input) and the full 6-camera
cloud (closest to RVT's fused multi-camera input) each make up 1/6 of the samples. The point cloud itself is built on
the GPU (``renderer.render_rgbd``). Per-sample randomness is keyed by a global sample id from the sampler, so the
data order and content depend only on (seed, epoch, position) and a resumed run continues the same sequence.
"""

from __future__ import annotations

import h5py
import numpy as np
import torch
from torch.utils.data import Sampler

from s4d.baselines.data import _EpisodeFile


def camera_subset(rng: np.random.Generator, num_views: int) -> np.ndarray:
    """Sorted indices of a random non-empty camera subset: size ~ U{1..num_views}, then a uniform subset."""
    k = int(rng.integers(1, num_views + 1))
    return np.sort(rng.choice(num_views, size=k, replace=False))


class MVPTimeSteps(_EpisodeFile):
    """Every time step of ``episodes`` (reference caps ``max_episodes`` / ``max_frames_per_demo``)."""

    def __init__(
        self,
        path,
        episodes: list[str],
        num_views: int = 6,
        seed: int = 0,
        max_episodes: int | None = None,
        max_frames_per_demo: int | None = None,
    ):
        super().__init__(path, episodes, max_episodes, max_frames_per_demo)
        if num_views > self.num_train_cams:
            raise ValueError(f"num_views={num_views} exceeds the {self.num_train_cams} training cameras")
        with h5py.File(self.path, "r") as f:
            cams = f["cameras"]
            train = np.where(cams["is_train"][:].astype(bool))[0]
            if not np.array_equal(train, np.arange(len(train))):
                raise ValueError("training cameras must be stored first in the camera table")
            self.K = torch.from_numpy(cams["K"][:][:num_views].astype(np.float32))
            self.c2w = torch.from_numpy(cams["c2w"][:][:num_views].astype(np.float32))
            self.depth_unit_m = float(f.attrs["depth_unit_m"])
        self.num_views, self.seed = int(num_views), int(seed)
        self.indices = [(ep, t) for ep in self.episodes for t in range(self.lengths[ep])]

    def __len__(self) -> int:
        return len(self.indices)

    def _episode(self, ep: str):
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        return self._file["episodes"][ep]

    def __getitem__(self, item) -> dict:
        """``item``: (index, sample key) from ``EpochBatches``, or a plain index (key = index)."""
        index, key = item if isinstance(item, tuple) else (item, item)
        ep, t = self.indices[int(index)]
        cams = camera_subset(np.random.default_rng([self.seed, int(key)]), self.num_views)
        group = self._episode(ep)
        H, W = group["rgb"].shape[2:4]
        rgb = np.zeros((self.num_views, H, W, 3), np.uint8)
        depth = np.zeros((self.num_views, H, W), np.float32)
        rgb[cams] = group["rgb"][t, cams.tolist()]
        depth[cams] = group["depth"][t, cams.tolist()].astype(np.float32) * np.float32(self.depth_unit_m)
        view_mask = np.zeros(self.num_views, bool)
        view_mask[cams] = True
        return {
            "rgb": torch.from_numpy(rgb),  # (V, H, W, 3) uint8, zero for cameras outside the subset
            "depth": torch.from_numpy(depth),  # (V, H, W) metres, 0 = invalid
            "view_mask": torch.from_numpy(view_mask),
            "K": self.K.clone(),
            "c2w": self.c2w.clone(),
            "episode": ep,
            "t": t,
        }


def batch_to_device(batch: dict, device) -> dict:
    """Collated samples -> the ``render_rgbd`` inputs on ``device`` (rgb (B, V, 3, H, W) float in [0, 1])."""
    rgb = batch["rgb"].to(device, non_blocking=True).permute(0, 1, 4, 2, 3).float().div_(255.0)
    out = {"rgb": rgb}
    for key in ("depth", "view_mask", "K", "c2w"):
        out[key] = batch[key].to(device, non_blocking=True)
    return out


class EpochBatches(Sampler):
    """Batches of one epoch in a permutation fixed by (seed, epoch), starting at batch ``start`` (lossless resume).
    Yields lists of (dataset index, global sample key); incomplete last batches are dropped."""

    def __init__(self, size: int, batch_size: int, seed: int, epoch: int, start: int = 0):
        self.size, self.batch_size, self.seed, self.epoch, self.start = size, batch_size, seed, epoch, start
        self.num_batches = size // batch_size

    def __len__(self) -> int:
        return max(0, self.num_batches - self.start)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed * 1_000_003 + self.epoch)
        perm = torch.randperm(self.size, generator=generator).tolist()
        for b in range(self.start, self.num_batches):
            first = b * self.batch_size
            yield [(perm[first + i], self.epoch * self.size + first + i) for i in range(self.batch_size)]


FIXED_KEY_OFFSET = 2**40  # above every training key (epoch * size + position)


def fixed_items(size: int, count: int) -> list[tuple[int, int]]:
    """A fixed, evenly spread set of (index, key) items, e.g. for validation (keys disjoint from training keys)."""
    count = min(count, size)
    indices = np.linspace(0, size - 1, count).round().astype(int)
    return [(int(i), FIXED_KEY_OFFSET + int(i)) for i in indices]
