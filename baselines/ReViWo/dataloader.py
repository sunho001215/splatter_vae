import json
import random
from typing import List, Tuple, Dict, Any, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, get_worker_info

from utils.general_utils import image_to_tensor


class RoboSuiteMultiViewAllCamerasHDF5Dataset(Dataset):
    """
    Simple multi-view RoboSuite dataset for ReViWo.

    For each indexed (demo_key, t), this dataset returns:
        - images : (N_cam, 3, H, W) float32 in [-1, 1]
                   where N_cam is the number of camera names (e.g. 6)
        - demo_key : string ID of the episode
        - t        : timestep index (int)
    """

    def __init__(
        self,
        dataset_path: str,
        demo_keys: List[str],
        views: Optional[List[str]] = None,
        max_frames_per_demo: Optional[int] = None,
        seed: int = 0,
        min_time_gap: int = 10,  # kept for API compatibility, but UNUSED
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.demo_keys = list(demo_keys)
        self.max_frames_per_demo = max_frames_per_demo

        # We keep an RNG and a worker_init_fn for potential future randomness,
        # but in this dataset we do NOT use it (no random view or time sampling).
        self.rng = random.Random(seed)

        # Logical views (camera names like "cam0", "cam1", ..., "cam5").
        # If None, we infer from the file's camera_names for the first demo.
        self.views: Optional[List[str]] = None if views is None else list(views)

        # HDF5 handle (opened lazily per worker process)
        self._h5: Optional[h5py.File] = None

        # Per-demo episode lengths (number of timesteps)
        self.demo_lengths: Dict[str, int] = {}

        # Global sample index: each entry is (demo_key, t)
        self.samples: List[Tuple[str, int]] = []

        # Build the sample list and fill demo_lengths / views
        self._index_file_and_build_samples()

        if len(self.views) < 1:
            raise ValueError("You need at least 1 view (camera name).")

    # -------------------------------------------------------------------------
    # Pickle safety for multi-worker DataLoader
    # -------------------------------------------------------------------------
    def __getstate__(self):
        """
        When Dataset is pickled to spawn worker processes, we must not pickle
        an open HDF5 file handle. This ensures each worker will reopen the file.
        """
        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    def __del__(self):
        """
        Best-effort cleanup (not guaranteed to be called in all situations).
        """
        try:
            if self._h5 is not None:
                self._h5.close()
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------
    def _get_h5(self) -> h5py.File:
        """
        Open the HDF5 file lazily. Each DataLoader worker should have its own handle.
        """
        if self._h5 is None:
            self._h5 = h5py.File(self.dataset_path, "r")
        return self._h5

    def _index_file_and_build_samples(self):
        """
        Read per-demo episode lengths once, infer camera names if needed,
        and build the global sample index as a list of (demo_key, t).
        """
        with h5py.File(self.dataset_path, "r") as f:
            if "data" not in f:
                raise ValueError('Invalid dataset: missing top-level group "data".')

            data_grp = f["data"]

            for demo_key in self.demo_keys:
                if demo_key not in data_grp:
                    raise ValueError(f'Demo key "{demo_key}" not found under /data.')

                demo_grp = data_grp[demo_key]

                # -------------------------------------------------------------
                # 1) Camera names for this demo (e.g. ["cam0", "cam1", ...])
                # -------------------------------------------------------------
                if "camera_names" not in demo_grp.attrs:
                    raise ValueError(f'"/data/{demo_key}" has no "camera_names" attribute.')
                camera_names = json.loads(demo_grp.attrs["camera_names"])

                # If views is not specified, infer them from the first demo's camera_names
                if self.views is None:
                    self.views = list(camera_names)
                else:
                    # Ensure all requested views exist in this demo
                    for v in self.views:
                        if v not in camera_names:
                            raise ValueError(
                                f'View "{v}" not found in demo "{demo_key}" camera_names={camera_names}.'
                            )

                # -------------------------------------------------------------
                # 2) Determine episode length T from one view's RGB dataset
                # -------------------------------------------------------------
                if "obs" not in demo_grp:
                    raise ValueError(f'"/data/{demo_key}" has no "obs" group.')
                obs_grp = demo_grp["obs"]

                ref_view = self.views[0]
                ref_name = f"{ref_view}_rgb"
                if ref_name not in obs_grp:
                    raise ValueError(
                        f'Missing dataset "/data/{demo_key}/obs/{ref_name}". '
                        f"Available keys: {list(obs_grp.keys())[:20]} ..."
                    )

                T = int(obs_grp[ref_name].shape[0])
                if self.max_frames_per_demo is not None:
                    T = min(T, int(self.max_frames_per_demo))
                self.demo_lengths[demo_key] = T

                # -------------------------------------------------------------
                # 3) Build sample index: single timestep t for each demo
                # -------------------------------------------------------------
                for t_idx in range(T):
                    self.samples.append((demo_key, t_idx))

        print(
            f"[RoboSuiteMultiViewAllCamerasHDF5Dataset] "
            f"Indexed {len(self.demo_keys)} demos, "
            f"{len(self.samples)} samples (demo, t), "
            f"views={self.views}"
        )

    # -------------------------------------------------------------------------
    # PyTorch Dataset interface
    # -------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns a dict:
          images   : (N_cam, 3, H, W) float32 in [-1, 1]
          demo_key : string
          t        : int (timestep index)
        """
        demo_key, t = self.samples[idx]

        # Open HDF5 file and obs group for this demo
        f = self._get_h5()
        obs_grp = f["data"][demo_key]["obs"]

        # Load all views at this timestep
        images_list = []
        for v in self.views:
            # Each RGB is stored as uint8 (H,W,3) under "<cam>_rgb"
            img_np = np.array(obs_grp[f"{v}_rgb"][t], dtype=np.uint8)  # (H, W, 3)
            # Convert to torch tensor in [-1,1], shape (3,H,W)
            img_tensor = image_to_tensor(img_np)
            images_list.append(img_tensor)

        # Stack into a single tensor (N_cam, 3, H, W)
        images = torch.stack(images_list, dim=0)

        return {
            "images": images,
            "demo_key": demo_key,
            "t": int(t),
        }


# -------------------------------------------------------------------------
# Helpers for building DataLoaders (train / valid)
# -------------------------------------------------------------------------

def _list_demo_keys_robosuite(dataset_path: str) -> List[str]:
    """
    List demo keys under /data in sorted order: demo1, demo2, ...
    """
    with h5py.File(dataset_path, "r") as f:
        if "data" not in f:
            raise ValueError('Invalid dataset: missing top-level group "data".')
        demos = list(f["data"].keys())

    # Sort by numeric suffix after "demo"
    def _demo_index(k: str) -> int:
        try:
            return int(k.replace("demo", ""))
        except Exception:
            return 10**18

    return sorted(demos, key=_demo_index)


def _worker_init_fn(worker_id: int):
    """
    Ensure each DataLoader worker has an independent RNG stream.

    PyTorch sets a deterministic base seed per worker (info.seed),
    so we can use it to seed Python's random.Random for the dataset instance.
    In this dataset we don't actually use RNG, but it's cheap and harmless.
    """
    info = get_worker_info()
    if info is None:
        return
    dataset = info.dataset
    dataset.rng = random.Random(info.seed)


def build_train_valid_loaders_robosuite(
    dataset_path: str,
    batch_size: int = 128,
    num_workers: int = 4,
    pin_memory: bool = True,
    train_ratio: float = 0.9,
    seed: int = 0,
    num_episodes: Optional[int] = None,
    max_frames_per_demo: Optional[int] = None,
    views: Optional[List[str]] = None,
    min_time_gap: int = 25,       # kept for backward-compat API, NOT used
    drop_last_train: bool = True,
    shuffle_train: bool = True,
    shuffle_valid: bool = True,
):
    """
    Build PyTorch DataLoaders from a RoboSuite HDF5 demo file.

    Splitting is done at the episode (demo) level:
      - Shuffle demo keys with `seed`
      - Take first N_train as train, remaining as valid

    Each batch from the resulting DataLoader has:
        batch["images"]: (B, N_cam, 3, H, W)  in [-1, 1]
        batch["demo_key"], batch["t"] for debugging.
    """
    demo_keys = _list_demo_keys_robosuite(dataset_path)
    if num_episodes is not None:
        demo_keys = demo_keys[: int(num_episodes)]

    rng = random.Random(seed)
    rng.shuffle(demo_keys)

    n_total = len(demo_keys)
    n_train = max(1, int(n_total * train_ratio))
    n_train = min(n_train, n_total - 1) if n_total > 1 else n_train

    train_keys = demo_keys[:n_train]
    valid_keys = demo_keys[n_train:] if n_total > 1 else demo_keys[:]

    # If views is not specified, infer from the first train demo's camera_names
    if views is None:
        with h5py.File(dataset_path, "r") as f:
            data_grp = f["data"]
            first_demo = train_keys[0]
            camera_names = json.loads(data_grp[first_demo].attrs["camera_names"])
            views = list(camera_names)

    train_dataset = RoboSuiteMultiViewAllCamerasHDF5Dataset(
        dataset_path=dataset_path,
        demo_keys=train_keys,
        views=views,
        max_frames_per_demo=max_frames_per_demo,
        seed=seed,
        min_time_gap=min_time_gap,   # unused, but kept for signature
    )
    valid_dataset = RoboSuiteMultiViewAllCamerasHDF5Dataset(
        dataset_path=dataset_path,
        demo_keys=valid_keys,
        views=views,
        max_frames_per_demo=max_frames_per_demo,
        seed=seed + 999,
        min_time_gap=min_time_gap,   # unused
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last_train,
        worker_init_fn=_worker_init_fn,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=shuffle_valid,
        num_workers=max(1, num_workers // 2),
        pin_memory=pin_memory,
        drop_last=True,
        worker_init_fn=_worker_init_fn,
    )

    return train_loader, valid_loader
