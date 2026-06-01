from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def _demo_sort_key(name: str) -> tuple[int, str]:
    """Sort demo1, demo2, ... numerically while keeping arbitrary names stable."""
    if name.startswith("demo"):
        try:
            return int(name.replace("demo", "")), name
        except ValueError:
            pass
    return 10**9, name


def list_demo_keys(hdf5_path: str, max_demos: Optional[int] = None) -> List[str]:
    """Return sorted episode keys under /data."""
    with h5py.File(hdf5_path, "r") as f:
        if "data" not in f:
            raise ValueError(f'{hdf5_path} must contain a top-level "data" group.')
        keys = sorted(list(f["data"].keys()), key=_demo_sort_key)
    return keys if max_demos is None else keys[: int(max_demos)]


def split_demo_keys(
    hdf5_path: str,
    val_ratio: float,
    seed: int,
    max_demos: Optional[int] = None,
) -> tuple[List[str], List[str]]:
    """Split at episode granularity to avoid train/validation leakage."""
    keys = list_demo_keys(hdf5_path, max_demos=max_demos)
    rng = random.Random(int(seed))
    rng.shuffle(keys)
    if len(keys) <= 1:
        return keys, keys[:]
    n_val = max(1, int(round(len(keys) * float(val_ratio))))
    n_val = min(n_val, len(keys) - 1)
    return keys[n_val:], keys[:n_val]


def _read_camera_names(demo_grp: h5py.Group) -> List[str]:
    """Infer camera names from demo attrs or *_rgb datasets."""
    if "camera_names" in demo_grp.attrs:
        raw = demo_grp.attrs["camera_names"]
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        names = json.loads(raw) if isinstance(raw, str) else list(raw)
        return [str(name) for name in names]

    if "obs" not in demo_grp:
        raise ValueError("Demo has no obs group and no camera_names attr.")
    names = []
    for key in demo_grp["obs"].keys():
        if key.endswith("_rgb"):
            names.append(key[:-4])
    if not names:
        raise ValueError("Could not infer cameras; expected obs/{camera}_rgb datasets.")
    return sorted(names)


def _resolve_dataset(root: h5py.Group, key: Optional[str]) -> Optional[h5py.Dataset]:
    """Resolve slash-separated dataset keys relative to a demo group."""
    if key is None or str(key).lower() in {"", "none", "null"}:
        return None
    obj: Any = root
    for part in str(key).strip("/").split("/"):
        if part not in obj:
            return None
        obj = obj[part]
    return obj if isinstance(obj, h5py.Dataset) else None


def _stack_rows(ds: h5py.Dataset, indices: np.ndarray) -> np.ndarray:
    """Read possibly padded indices from h5py without relying on fancy indexing."""
    return np.stack([ds[int(i)] for i in indices], axis=0)


def _resize_frames(frames: np.ndarray, image_size: Optional[Tuple[int, int]]) -> np.ndarray:
    """Resize NHWC uint8 frames if the YAML requests a training resolution."""
    if image_size is None:
        return frames
    target_h, target_w = image_size
    if frames.shape[1] == target_h and frames.shape[2] == target_w:
        return frames
    resized = [
        cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        for frame in frames
    ]
    return np.stack(resized, axis=0)


@dataclass(frozen=True)
class SequenceIndex:
    demo_key: str
    anchor_t: int


class HDF5DiffusionPolicyDataset(Dataset):
    """
    Sequence dataset for action-diffusion behavior cloning.

    Each item is anchored at a current action time t. The observation window is
    [t - n_obs_steps + 1, ..., t], and the action trajectory starts at the same
    padded sequence origin so Diffusion Policy can later execute the slice
    action_pred[:, n_obs_steps - 1 : n_obs_steps - 1 + n_action_steps].
    """

    def __init__(
        self,
        hdf5_path: str,
        demo_keys: Sequence[str],
        horizon: int,
        n_obs_steps: int,
        camera_names: Optional[Sequence[str]] = None,
        proprio_key: Optional[str] = "obs_env/obs",
        image_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        self.hdf5_path = str(hdf5_path)
        self.demo_keys = list(demo_keys)
        self.horizon = int(horizon)
        self.n_obs_steps = int(n_obs_steps)
        self.camera_names = list(camera_names) if camera_names is not None else None
        self.proprio_key = proprio_key
        self.image_size = image_size
        self._file: Optional[h5py.File] = None

        if self.horizon < self.n_obs_steps:
            raise ValueError("horizon must be >= n_obs_steps for DP action slicing.")
        self.samples: List[SequenceIndex] = []
        self.demo_lengths: Dict[str, int] = {}
        self.action_dim = 0
        self.proprio_dim = 0
        self._index_metadata()

    def _h5(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.hdf5_path, "r")
        return self._file

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __del__(self) -> None:
        self.close()

    def _index_metadata(self) -> None:
        with h5py.File(self.hdf5_path, "r") as f:
            data = f["data"]
            if not self.demo_keys:
                raise ValueError("No demos were provided to HDF5DiffusionPolicyDataset.")
            if self.camera_names is None:
                self.camera_names = _read_camera_names(data[self.demo_keys[0]])

            for demo_key in self.demo_keys:
                demo = data[demo_key]
                if "actions" not in demo:
                    raise ValueError(f'Missing "/data/{demo_key}/actions".')
                action_ds = demo["actions"]
                length = int(action_ds.shape[0])
                if length <= 0:
                    continue
                if self.action_dim == 0:
                    self.action_dim = int(np.prod(action_ds.shape[1:]))

                proprio_ds = _resolve_dataset(demo, self.proprio_key)
                if proprio_ds is not None:
                    length = min(length, int(proprio_ds.shape[0]))
                    if self.proprio_dim == 0:
                        self.proprio_dim = int(np.prod(proprio_ds.shape[1:]))

                obs_grp = demo.get("obs", None)
                if obs_grp is None:
                    raise ValueError(f'Missing "/data/{demo_key}/obs".')
                for cam in self.camera_names:
                    rgb_key = f"{cam}_rgb"
                    if rgb_key not in obs_grp:
                        raise ValueError(f'Missing "/data/{demo_key}/obs/{rgb_key}".')
                    length = min(length, int(obs_grp[rgb_key].shape[0]))

                self.demo_lengths[demo_key] = length
                for anchor_t in range(length):
                    self.samples.append(SequenceIndex(demo_key=demo_key, anchor_t=anchor_t))

        if not self.samples:
            raise ValueError("Dataset contains no valid sequence samples.")

    def __len__(self) -> int:
        return len(self.samples)

    def _sequence_indices(self, anchor_t: int, length: int, count: int) -> np.ndarray:
        seq_start = int(anchor_t) - (self.n_obs_steps - 1)
        raw = seq_start + np.arange(count, dtype=np.int64)
        return np.clip(raw, 0, max(0, int(length) - 1))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[int(idx)]
        f = self._h5()
        demo = f["data"][sample.demo_key]
        length = self.demo_lengths[sample.demo_key]

        obs_indices = self._sequence_indices(sample.anchor_t, length, self.n_obs_steps)
        action_indices = self._sequence_indices(sample.anchor_t, length, self.horizon)

        actions = _stack_rows(demo["actions"], action_indices).reshape(self.horizon, -1)
        proprio_ds = _resolve_dataset(demo, self.proprio_key)
        if proprio_ds is None:
            proprio = np.zeros((self.n_obs_steps, 0), dtype=np.float32)
        else:
            proprio = _stack_rows(proprio_ds, obs_indices).reshape(self.n_obs_steps, -1)

        images_by_camera = []
        obs_grp = demo["obs"]
        for cam in self.camera_names:
            frames = _stack_rows(obs_grp[f"{cam}_rgb"], obs_indices).astype(np.uint8)
            frames = _resize_frames(frames, self.image_size)
            # Store as V, To, C, H, W so the collator can pick one camera per batch.
            images_by_camera.append(np.transpose(frames, (0, 3, 1, 2)))

        return {
            "image": np.stack(images_by_camera, axis=0),
            "action": actions.astype(np.float32),
            "proprio": proprio.astype(np.float32),
            "camera_names": list(self.camera_names),
            "demo_key": sample.demo_key,
            "anchor_t": int(sample.anchor_t),
        }

    def compute_normalizer_stats(self) -> Dict[str, torch.Tensor]:
        """Compute mean/std for actions and proprioception from complete train episodes."""
        action_chunks = []
        proprio_chunks = []
        with h5py.File(self.hdf5_path, "r") as f:
            for demo_key in self.demo_keys:
                demo = f["data"][demo_key]
                length = self.demo_lengths[demo_key]
                action_chunks.append(np.asarray(demo["actions"][:length], dtype=np.float32).reshape(length, -1))
                proprio_ds = _resolve_dataset(demo, self.proprio_key)
                if proprio_ds is not None:
                    proprio_chunks.append(
                        np.asarray(proprio_ds[:length], dtype=np.float32).reshape(length, -1)
                    )

        actions = torch.as_tensor(np.concatenate(action_chunks, axis=0), dtype=torch.float32)
        stats = {
            "action_mean": actions.mean(dim=0),
            "action_std": actions.std(dim=0).clamp_min(1e-6),
        }
        if proprio_chunks:
            proprio = torch.as_tensor(np.concatenate(proprio_chunks, axis=0), dtype=torch.float32)
            stats["proprio_mean"] = proprio.mean(dim=0)
            stats["proprio_std"] = proprio.std(dim=0).clamp_min(1e-6)
        else:
            stats["proprio_mean"] = torch.zeros(0, dtype=torch.float32)
            stats["proprio_std"] = torch.ones(0, dtype=torch.float32)
        return stats


class RandomCameraBatchCollator:
    """Select one camera for the whole mini-batch, matching the multi-view baseline request."""

    def __init__(self, mode: str = "random", fixed_camera_index: int = 0) -> None:
        self.mode = str(mode)
        self.fixed_camera_index = int(fixed_camera_index)

    def __call__(self, batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        images = torch.as_tensor(np.stack([item["image"] for item in batch], axis=0))
        actions = torch.as_tensor(np.stack([item["action"] for item in batch], axis=0), dtype=torch.float32)
        proprio = torch.as_tensor(np.stack([item["proprio"] for item in batch], axis=0), dtype=torch.float32)

        num_views = int(images.shape[1])
        if self.mode == "random":
            camera_index = random.randrange(num_views)
        elif self.mode == "fixed":
            camera_index = self.fixed_camera_index % num_views
        else:
            raise ValueError(f"Unknown camera selection mode: {self.mode}")

        camera_names = batch[0]["camera_names"]
        return {
            "image": images[:, camera_index].contiguous(),
            "action": actions.contiguous(),
            "proprio": proprio.contiguous(),
            "camera_index": torch.full((len(batch),), int(camera_index), dtype=torch.long),
            "camera_name": str(camera_names[camera_index]),
            "demo_key": [item["demo_key"] for item in batch],
            "anchor_t": torch.as_tensor([item["anchor_t"] for item in batch], dtype=torch.long),
        }

