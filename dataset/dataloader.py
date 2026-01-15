import json
import random
from typing import List, Tuple, Dict, Any, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, get_worker_info

from utils.general_utils import image_to_tensor, invert_4x4


class RoboSuiteMultiViewTemporalHDF5Dataset(Dataset):
    """
    Dataset for the robosuite demos collected with collect_demos.py.

    For each indexed (demo_key, t), it returns:
      - image_i_t   : view_i at timestep t
      - image_j_t   : view_j at timestep t (same demo, same t)  <-- positive for view-invariant contrastive
      - image_i_t1  : view_i at timestep t1 (same demo, but far from t) <-- temporal negative
      - T_ij        : (4,4) transform from camera-i coords to camera-j coords
      - K_i, K_j    : intrinsics
      - some metadata (demo_key, t, t1, view_i, view_j)

    The HDF5 layout (per demo) is:

      /data/demoX
        attrs:
          camera_names : JSON-encoded list of camera names, e.g. ["cam0", ..., "cam5"]
          num_samples  : number of timesteps T

        camera_params/
          intrinsics             : (N_cam, 3, 3)
          extrinsics_world_T_cam : (N_cam, 4, 4)  # cam -> world

        states  : (T, state_dim)
        actions : (T, action_dim)

        obs/
          <cam>_rgb : (T, H, W, 3) uint8
          <cam>_seg : (T, H, W)    int32

        obs_env/
          <obs_key> : (T, ...)

    This dataset focuses on RGB + camera geometry.
    """

    def __init__(
        self,
        dataset_path: str,
        demo_keys: List[str],
        views: Optional[List[str]] = None,
        max_frames_per_demo: Optional[int] = None,
        seed: int = 0,
        min_time_gap: int = 10,
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.demo_keys = list(demo_keys)
        self.max_frames_per_demo = max_frames_per_demo

        # Minimum temporal gap for choosing t1 (|t1 - t| >= min_time_gap)
        self.min_time_gap = int(min_time_gap)

        # RNG (overwritten per worker via _worker_init_fn)
        self.rng = random.Random(seed)

        # Logical views (camera names like cam0, cam1, ...)
        # If None, we will infer from the file's camera_names for the first demo.
        self.views = None if views is None else list(views)

        # HDF5 handle (opened lazily per worker process)
        self._h5: Optional[h5py.File] = None

        # Per-demo metadata
        self.demo_lengths: Dict[str, int] = {}
        self.cam_cache: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}

        # Global sample index: each entry is (demo_key, t)
        self.samples: List[Tuple[str, int]] = []

        self._index_file_and_build_samples()

        if len(self.views) < 2:
            raise ValueError("You need at least 2 views (camera names) to sample (view_i, view_j).")

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
        Read per-demo lengths and camera parameters once (small metadata),
        and build the global sample index as (demo_key, t).
        """
        with h5py.File(self.dataset_path, "r") as f:
            if "data" not in f:
                raise ValueError('Invalid dataset: missing top-level group "data".')

            data_grp = f["data"]

            for demo_key in self.demo_keys:
                if demo_key not in data_grp:
                    raise ValueError(f'Demo key "{demo_key}" not found under /data.')

                demo_grp = data_grp[demo_key]

                # -----------------------------------------------------------------
                # Camera names for this demo (e.g. ["cam0", "cam1", ..., "cam5"])
                # -----------------------------------------------------------------
                if "camera_names" not in demo_grp.attrs:
                    raise ValueError(f'"/data/{demo_key}" has no "camera_names" attribute.')
                camera_names = json.loads(demo_grp.attrs["camera_names"])

                # If views is not specified, infer it from the first demo's camera_names
                if self.views is None:
                    self.views = list(camera_names)
                else:
                    # Ensure all requested views exist in this demo
                    for v in self.views:
                        if v not in camera_names:
                            raise ValueError(
                                f'View "{v}" not found in demo "{demo_key}" camera_names={camera_names}.'
                            )

                # -----------------------------------------------------------------
                # Validate obs group and required RGB datasets
                # -----------------------------------------------------------------
                if "obs" not in demo_grp:
                    raise ValueError(f'"/data/{demo_key}" has no "obs" group.')
                obs_grp = demo_grp["obs"]

                # Determine episode length T from one view's RGB dataset
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

                # -----------------------------------------------------------------
                # Load camera intrinsics/extrinsics for each view from /camera_params
                # -----------------------------------------------------------------
                if "camera_params" not in demo_grp:
                    raise ValueError(f'"/data/{demo_key}" has no "camera_params" group.')
                cam_params_grp = demo_grp["camera_params"]

                if "intrinsics" not in cam_params_grp or "extrinsics_world_T_cam" not in cam_params_grp:
                    raise ValueError(
                        f'"/data/{demo_key}/camera_params" must contain "intrinsics" and "extrinsics_world_T_cam".'
                    )

                intrinsics_ds = cam_params_grp["intrinsics"]              # (N_cam, 3, 3)
                world_T_cam_ds = cam_params_grp["extrinsics_world_T_cam"] # (N_cam, 4, 4)

                camera_names_list = list(camera_names)
                self.cam_cache[demo_key] = {}

                for v in self.views:
                    if v not in camera_names_list:
                        raise ValueError(
                            f'Camera "{v}" not found in camera_names for demo "{demo_key}".'
                        )
                    cam_idx = camera_names_list.index(v)

                    K = np.array(intrinsics_ds[cam_idx], dtype=np.float32)          # (3,3)
                    world_T_cam = np.array(world_T_cam_ds[cam_idx], dtype=np.float32)  # (4,4)

                    # world_T_cam maps camera coords -> world coords.
                    # We need world->camera (w2c), so invert:
                    w2c_gl = invert_4x4(world_T_cam)  # OpenGL-style camera frame

                    # Convert from OpenGL-style camera coordinates to OpenCV-style
                    # (X right, Y down, Z forward) using a fixed similarity transform.
                    S = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
                    w2c = S @ w2c_gl   # world -> cam (OpenCV convention)
                    c2w = invert_4x4(w2c).astype(np.float32)

                    self.cam_cache[demo_key][v] = {
                        "K": K,
                        "w2c": w2c,
                        "c2w": c2w,
                    }

                # -----------------------------------------------------------------
                # Build sample index: (demo_key, t) only
                # -----------------------------------------------------------------
                for t_idx in range(T):
                    self.samples.append((demo_key, t_idx))

        print(
            f"[RoboSuiteMultiViewTemporalHDF5Dataset] "
            f"Indexed {len(self.demo_keys)} demos, "
            f"{len(self.samples)} samples (demo,t), views={self.views}"
        )

    def _sample_view_pair(self) -> Tuple[str, str]:
        """
        Sample two distinct views (view_i, view_j).
        """
        view_i, view_j = self.rng.sample(self.views, 2)
        return view_i, view_j

    def _sample_t1_with_gap(self, T: int, t: int) -> int:
        """
        Sample a timestep t1 such that |t1 - t| >= min_time_gap whenever possible.

        If not possible (episode too short), fall back to:
          - any timestep != t (if possible),
          - else t itself.
        """
        k = self.min_time_gap

        if T <= 1:
            return t

        # Candidate indices with a hard minimum gap
        left_end = max(0, t - k)           # [0, left_end)
        right_start = min(T, t + k)        # [right_start, T)

        candidates: List[int] = []
        if left_end > 0:
            candidates.extend(range(0, left_end))
        if right_start < T:
            candidates.extend(range(right_start, T))

        if len(candidates) > 0:
            return self.rng.choice(candidates)

        # Fallback: episode too short to satisfy the gap.
        # Choose any index != t if possible.
        if T > 1:
            t1 = self.rng.randrange(T - 1)
            if t1 >= t:
                t1 += 1
            return t1

        return t

    # -------------------------------------------------------------------------
    # PyTorch Dataset interface
    # -------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns a dict:
          image_i_t   : (3,H,W) float32 in [-1, 1]
          image_j_t   : (3,H,W) float32 in [-1, 1]
          image_i_t1  : (3,H,W) float32 in [-1, 1]
          T_ij        : (4,4) float32 (cam_i -> cam_j)
          K_i         : (3,3) float32
          K_j         : (3,3) float32
          demo_key, t, t1, view_i, view_j : metadata for debugging/logging
        """
        demo_key, t = self.samples[idx]
        T = self.demo_lengths[demo_key]

        # Randomly sample view pair
        view_i, view_j = self._sample_view_pair()

        # Sample t1 with minimum gap
        t1 = self._sample_t1_with_gap(T=T, t=t)

        # Load images from HDF5
        f = self._get_h5()
        obs_grp = f["data"][demo_key]["obs"]

        # Each RGB is stored as uint8 (H,W,3) under "<cam>_rgb"
        img_i_t_np = np.array(obs_grp[f"{view_i}_rgb"][t], dtype=np.uint8)
        img_j_t_np = np.array(obs_grp[f"{view_j}_rgb"][t], dtype=np.uint8)
        img_i_t1_np = np.array(obs_grp[f"{view_i}_rgb"][t1], dtype=np.uint8)

        # Convert to torch tensors in [-1,1], shape (3,H,W)
        image_i_t = image_to_tensor(img_i_t_np)
        image_j_t = image_to_tensor(img_j_t_np)
        image_i_t1 = image_to_tensor(img_i_t1_np)

        # Camera matrices
        cam_i = self.cam_cache[demo_key][view_i]
        cam_j = self.cam_cache[demo_key][view_j]

        K_i = torch.from_numpy(cam_i["K"])  # (3,3)
        K_j = torch.from_numpy(cam_j["K"])  # (3,3)

        # Compute T_ij: cam_i -> cam_j in OpenCV camera coordinates
        #   X_world = c2w_i * X_cam_i
        #   X_cam_j = w2c_j * X_world
        # => X_cam_j = (w2c_j * c2w_i) * X_cam_i
        T_ij_np = cam_j["w2c"] @ cam_i["c2w"]
        T_ij = torch.from_numpy(T_ij_np.astype(np.float32))  # (4,4)

        return {
            "image_i_t": image_i_t,
            "image_j_t": image_j_t,
            "image_i_t1": image_i_t1,
            "T_ij": T_ij,
            "K_i": K_i,
            "K_j": K_j,
            "demo_key": demo_key,
            "t": int(t),
            "t1": int(t1),
            "view_i": view_i,
            "view_j": view_j,
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
    """
    info = get_worker_info()
    if info is None:
        return
    dataset = info.dataset
    # Re-seed the dataset's RNG so workers don't produce correlated samples.
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
    min_time_gap: int = 25,
    drop_last_train: bool = True,
    shuffle_train: bool = True,
    shuffle_valid: bool = True,
):
    """
    Build PyTorch DataLoaders from a robosuite HDF5 demo file
    collected with collect_demos.py.

    Splitting is done at the episode (demo) level:
      - Shuffle demo keys with `seed`
      - Take first N_train as train, remaining as valid

    Args:
        dataset_path: path to the robosuite HDF5 file
        batch_size: batch size
        num_workers: DataLoader workers
        train_ratio: ratio of episodes used for training
        seed: seed for episode split
        num_episodes: if not None, only use the first num_episodes demos (after sorting)
        max_frames_per_demo: optional cap on frames per episode
        views: optional list of camera names to use (e.g. ["cam0", "cam1", ...]).
               If None, all cameras in the file are used.
        min_time_gap: minimum |t1 - t| gap for temporal sample
        drop_last_train: usually True for contrastive stability (fixed batch size)
        shuffle_train: usually True
        shuffle_valid: usually False for deterministic evaluation
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

    # If views is not specified, infer from the first demo's camera_names
    if views is None:
        with h5py.File(dataset_path, "r") as f:
            data_grp = f["data"]
            first_demo = train_keys[0]
            camera_names = json.loads(data_grp[first_demo].attrs["camera_names"])
            views = list(camera_names)

    train_dataset = RoboSuiteMultiViewTemporalHDF5Dataset(
        dataset_path=dataset_path,
        demo_keys=train_keys,
        views=views,
        max_frames_per_demo=max_frames_per_demo,
        seed=seed,
        min_time_gap=min_time_gap,
    )
    valid_dataset = RoboSuiteMultiViewTemporalHDF5Dataset(
        dataset_path=dataset_path,
        demo_keys=valid_keys,
        views=views,
        max_frames_per_demo=max_frames_per_demo,
        seed=seed + 999,
        min_time_gap=min_time_gap,
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
