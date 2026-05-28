import json
import random
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, get_worker_info

from utils.general_utils import image_to_tensor, invert_4x4


class RoboSuiteMultiViewTemporalHDF5Dataset(Dataset):
    """RoboSuite HDF5 dataset that returns every camera at one timestep.

    SplatterVAE now follows the same all-camera sampling contract used by
    ReViWo: one dataset item is one environment state, and the item contains all
    selected camera viewpoints for that state.  The old temporal ``tk`` images
    are intentionally not loaded because view/state shuffling is now performed
    directly inside the batch with the ``(B, camera_num, ...)`` tensors.

    Returned tensors:
        images: (N_cam, 3, H, W) float32 in [-1, 1]
        K:      (N_cam, 3, 3) camera intrinsics
        c2w:    (N_cam, 4, 4) OpenCV camera-to-world transforms
        w2c:    (N_cam, 4, 4) OpenCV world-to-camera transforms
    """

    def __init__(
        self,
        dataset_path: str,
        demo_keys: List[str],
        views: Optional[List[str]] = None,
        camera_num: Optional[int] = None,
        max_frames_per_demo: Optional[int] = None,
        seed: int = 0,
        min_time_gap: int = 10,
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.demo_keys = list(demo_keys)
        self.max_frames_per_demo = max_frames_per_demo
        self.camera_num = None if camera_num is None else int(camera_num)
        if self.camera_num is not None and self.camera_num <= 0:
            raise ValueError(f"camera_num must be positive when set, got {camera_num}.")

        # ``min_time_gap`` remains in the constructor for backward-compatible
        # config loading, but it is no longer used because we do not sample tk.
        self.min_time_gap = int(min_time_gap)

        # Kept for worker seeding and future stochastic sampling.  This dataset
        # path is deterministic for a given index.
        self.rng = random.Random(seed)

        self.views: Optional[List[str]] = None if views is None else list(views)
        if self.views is not None and self.camera_num is not None:
            self.views = self.views[: self.camera_num]

        # HDF5 handles cannot be pickled safely, so each worker opens lazily.
        self._h5: Optional[h5py.File] = None

        self.demo_lengths: Dict[str, int] = {}
        self.cam_cache: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
        self.samples: List[Tuple[str, int]] = []

        self._index_file_and_build_samples()

        if self.views is None or len(self.views) < 1:
            raise ValueError("You need at least one camera view.")

    def __getstate__(self):
        """Drop the live HDF5 handle before DataLoader worker pickling."""
        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    def __del__(self):
        """Best-effort HDF5 cleanup."""
        try:
            if self._h5 is not None:
                self._h5.close()
        except Exception:
            pass

    def _get_h5(self) -> h5py.File:
        """Open one HDF5 handle per process/worker."""
        if self._h5 is None:
            self._h5 = h5py.File(self.dataset_path, "r")
        return self._h5

    def _index_file_and_build_samples(self) -> None:
        """Cache metadata and build the global ``(demo_key, t)`` index."""
        with h5py.File(self.dataset_path, "r") as f:
            if "data" not in f:
                raise ValueError('Invalid dataset: missing top-level group "data".')

            data_grp = f["data"]
            for demo_key in self.demo_keys:
                if demo_key not in data_grp:
                    raise ValueError(f'Demo key "{demo_key}" not found under /data.')

                demo_grp = data_grp[demo_key]
                if "camera_names" not in demo_grp.attrs:
                    raise ValueError(f'"/data/{demo_key}" has no "camera_names" attribute.')
                camera_names = json.loads(demo_grp.attrs["camera_names"])

                # Infer all cameras from the first demo unless the config passed
                # an explicit camera list.  ``camera_num`` truncates the list so
                # configs can say "use the first six cameras" without spelling
                # every camera name in each YAML file.
                if self.views is None:
                    inferred_views = list(camera_names)
                    if self.camera_num is not None:
                        inferred_views = inferred_views[: self.camera_num]
                    self.views = inferred_views
                else:
                    for view in self.views:
                        if view not in camera_names:
                            raise ValueError(
                                f'View "{view}" not found in demo "{demo_key}" camera_names={camera_names}.'
                            )

                if "obs" not in demo_grp:
                    raise ValueError(f'"/data/{demo_key}" has no "obs" group.')
                obs_grp = demo_grp["obs"]

                # Every selected camera must have an RGB stream.  The reference
                # stream provides the episode length used for this demo.
                ref_view = self.views[0]
                ref_name = f"{ref_view}_rgb"
                if ref_name not in obs_grp:
                    raise ValueError(
                        f'Missing dataset "/data/{demo_key}/obs/{ref_name}". '
                        f"Available keys: {list(obs_grp.keys())[:20]} ..."
                    )
                for view in self.views:
                    rgb_name = f"{view}_rgb"
                    if rgb_name not in obs_grp:
                        raise ValueError(f'Missing dataset "/data/{demo_key}/obs/{rgb_name}".')

                timesteps = int(obs_grp[ref_name].shape[0])
                if self.max_frames_per_demo is not None:
                    timesteps = min(timesteps, int(self.max_frames_per_demo))
                self.demo_lengths[demo_key] = timesteps

                if "camera_params" not in demo_grp:
                    raise ValueError(f'"/data/{demo_key}" has no "camera_params" group.')
                cam_params_grp = demo_grp["camera_params"]
                if "intrinsics" not in cam_params_grp or "extrinsics_world_T_cam" not in cam_params_grp:
                    raise ValueError(
                        f'"/data/{demo_key}/camera_params" must contain "intrinsics" '
                        'and "extrinsics_world_T_cam".'
                    )

                intrinsics_ds = cam_params_grp["intrinsics"]
                world_T_cam_ds = cam_params_grp["extrinsics_world_T_cam"]
                camera_names_list = list(camera_names)
                self.cam_cache[demo_key] = {}

                for view in self.views:
                    cam_idx = camera_names_list.index(view)
                    K = np.array(intrinsics_ds[cam_idx], dtype=np.float32)
                    world_T_cam = np.array(world_T_cam_ds[cam_idx], dtype=np.float32)

                    # The demos store camera-to-world in an OpenGL camera frame.
                    # The renderer/training code uses the OpenCV convention
                    # (x right, y down, z forward), so convert once here and
                    # cache both directions.
                    w2c_gl = invert_4x4(world_T_cam)
                    gl_to_cv = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
                    w2c = gl_to_cv @ w2c_gl
                    c2w = invert_4x4(w2c).astype(np.float32)

                    self.cam_cache[demo_key][view] = {"K": K, "w2c": w2c, "c2w": c2w}

                for t_idx in range(timesteps):
                    self.samples.append((demo_key, t_idx))

        print(
            f"[RoboSuiteMultiViewTemporalHDF5Dataset] Indexed {len(self.demo_keys)} demos, "
            f"{len(self.samples)} samples (demo,t), views={self.views}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Load all selected views for one state.

        No random view-pair sampling happens here.  Keeping camera order stable
        is important because the ReViWo-style view contrastive loss and shuffle
        loss both rely on camera index ``a`` having the same meaning for every
        batch row.
        """
        demo_key, t = self.samples[idx]
        obs_grp = self._get_h5()["data"][demo_key]["obs"]

        images = []
        intrinsics = []
        c2w_mats = []
        w2c_mats = []
        for view in self.views:
            img_np = np.array(obs_grp[f"{view}_rgb"][t], dtype=np.uint8)
            images.append(image_to_tensor(img_np))

            cam = self.cam_cache[demo_key][view]
            intrinsics.append(torch.from_numpy(cam["K"]))
            c2w_mats.append(torch.from_numpy(cam["c2w"]))
            w2c_mats.append(torch.from_numpy(cam["w2c"]))

        return {
            "images": torch.stack(images, dim=0),
            "K": torch.stack(intrinsics, dim=0),
            "c2w": torch.stack(c2w_mats, dim=0),
            "w2c": torch.stack(w2c_mats, dim=0),
            "demo_key": demo_key,
            "t": int(t),
        }


# -------------------------------------------------------------------------
# Helpers for building DataLoaders (train / valid)
# -------------------------------------------------------------------------


def _list_demo_keys_robosuite(dataset_path: str) -> List[str]:
    """List demo keys under /data in sorted order: demo1, demo2, ..."""
    with h5py.File(dataset_path, "r") as f:
        if "data" not in f:
            raise ValueError('Invalid dataset: missing top-level group "data".')
        demos = list(f["data"].keys())

    def _demo_index(key: str) -> int:
        try:
            return int(key.replace("demo", ""))
        except Exception:
            return 10**18

    return sorted(demos, key=_demo_index)


def _worker_init_fn(worker_id: int) -> None:
    """Give each worker an independent Python RNG stream."""
    info = get_worker_info()
    if info is None:
        return
    info.dataset.rng = random.Random(info.seed)


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
    camera_num: Optional[int] = None,
    min_time_gap: int = 25,
    drop_last_train: bool = True,
    shuffle_train: bool = True,
    shuffle_valid: bool = True,
):
    """Build train/validation loaders with all selected cameras per sample."""
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

    if views is None:
        with h5py.File(dataset_path, "r") as f:
            first_demo = train_keys[0]
            views = list(json.loads(f["data"][first_demo].attrs["camera_names"]))
    else:
        views = list(views)

    if camera_num is not None:
        views = views[: int(camera_num)]

    train_dataset = RoboSuiteMultiViewTemporalHDF5Dataset(
        dataset_path=dataset_path,
        demo_keys=train_keys,
        views=views,
        camera_num=camera_num,
        max_frames_per_demo=max_frames_per_demo,
        seed=seed,
        min_time_gap=min_time_gap,
    )
    valid_dataset = RoboSuiteMultiViewTemporalHDF5Dataset(
        dataset_path=dataset_path,
        demo_keys=valid_keys,
        views=views,
        camera_num=camera_num,
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
