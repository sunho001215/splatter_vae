import json
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, get_worker_info

from utils.general_utils import image_to_tensor, invert_4x4

DatasetPathInput = Union[str, Sequence[str]]
DemoRef = Tuple[int, str]
SampleRef = Tuple[int, str, int]


def _normalize_dataset_paths(dataset_path: DatasetPathInput) -> List[str]:
    if isinstance(dataset_path, (str, bytes)):
        paths = [str(dataset_path)]
    else:
        paths = [str(path) for path in dataset_path]
    if not paths:
        raise ValueError("At least one HDF5 dataset path is required.")
    return paths


def _demo_label(file_idx: int, demo_key: str, num_files: int) -> str:
    return demo_key if num_files == 1 else f"file{file_idx}:{demo_key}"


class RoboSuiteMultiViewTemporalHDF5Dataset(Dataset):
    """RoboSuite/Meta-World HDF5 dataset returning every selected camera.

    ``dataset_path`` may be either one HDF5 file or a list of HDF5 files.  When
    multiple files are supplied, the dataset builds one global sample list over
    all ``(file, demo, timestep)`` entries, so a shuffled DataLoader naturally
    samples across environments during training.

    Returned tensors:
        images: (N_cam, 3, H, W) float32 in [-1, 1]
        K:      (N_cam, 3, 3) camera intrinsics
        c2w:    (N_cam, 4, 4) OpenCV camera-to-world transforms
        w2c:    (N_cam, 4, 4) OpenCV world-to-camera transforms
        depths: optional (N_cam, 1, H, W) float32 metric camera-z depth
    """

    def __init__(
        self,
        dataset_path: DatasetPathInput,
        demo_keys: List[Union[str, DemoRef]],
        views: Optional[List[str]] = None,
        camera_num: Optional[int] = None,
        max_frames_per_demo: Optional[int] = None,
        seed: int = 0,
        min_time_gap: int = 10,
        use_depth: bool = False,
    ):
        super().__init__()
        self.dataset_paths = _normalize_dataset_paths(dataset_path)
        self.dataset_path = self.dataset_paths[0]
        self.demo_refs: List[DemoRef] = [
            (int(item[0]), str(item[1])) if isinstance(item, tuple) else (0, str(item))
            for item in demo_keys
        ]
        self.max_frames_per_demo = max_frames_per_demo
        self.camera_num = None if camera_num is None else int(camera_num)
        if self.camera_num is not None and self.camera_num <= 0:
            raise ValueError(f"camera_num must be positive when set, got {camera_num}.")

        # ``min_time_gap`` remains for backward-compatible config loading, but
        # it is no longer used because this dataset returns one state at a time.
        self.min_time_gap = int(min_time_gap)
        self.use_depth = bool(use_depth)
        self.rng = random.Random(seed)

        self.views: Optional[List[str]] = None if views is None else list(views)
        if self.views is not None and self.camera_num is not None:
            self.views = self.views[: self.camera_num]

        self._h5_handles: Dict[int, h5py.File] = {}
        self.demo_lengths: Dict[DemoRef, int] = {}
        self.cam_cache: Dict[DemoRef, Dict[str, Dict[str, np.ndarray]]] = {}
        self.samples: List[SampleRef] = []

        self._index_files_and_build_samples()

        if self.views is None or len(self.views) < 1:
            raise ValueError("You need at least one camera view.")

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5_handles"] = {}
        return state

    def __del__(self):
        try:
            for handle in self._h5_handles.values():
                handle.close()
        except Exception:
            pass

    def _get_h5(self, file_idx: int) -> h5py.File:
        if file_idx not in self._h5_handles:
            self._h5_handles[file_idx] = h5py.File(self.dataset_paths[file_idx], "r")
        return self._h5_handles[file_idx]

    def _index_files_and_build_samples(self) -> None:
        refs_by_file: Dict[int, List[str]] = {}
        for file_idx, demo_key in self.demo_refs:
            refs_by_file.setdefault(file_idx, []).append(demo_key)

        for file_idx, demo_keys in refs_by_file.items():
            dataset_path = self.dataset_paths[file_idx]
            with h5py.File(dataset_path, "r") as f:
                if "data" not in f:
                    raise ValueError(f'Invalid dataset "{dataset_path}": missing top-level group "data".')

                data_grp = f["data"]
                for demo_key in demo_keys:
                    if demo_key not in data_grp:
                        raise ValueError(f'Demo key "{demo_key}" not found under /data in "{dataset_path}".')

                    demo_grp = data_grp[demo_key]
                    if "camera_names" not in demo_grp.attrs:
                        raise ValueError(f'"/data/{demo_key}" in "{dataset_path}" has no "camera_names" attribute.')
                    camera_names = json.loads(demo_grp.attrs["camera_names"])

                    if self.views is None:
                        inferred_views = list(camera_names)
                        if self.camera_num is not None:
                            inferred_views = inferred_views[: self.camera_num]
                        self.views = inferred_views
                    else:
                        for view in self.views:
                            if view not in camera_names:
                                raise ValueError(
                                    f'View "{view}" not found in "{dataset_path}" demo "{demo_key}" '
                                    f"camera_names={camera_names}."
                                )

                    if "obs" not in demo_grp:
                        raise ValueError(f'"/data/{demo_key}" in "{dataset_path}" has no "obs" group.')
                    obs_grp = demo_grp["obs"]

                    ref_view = self.views[0]
                    ref_name = f"{ref_view}_rgb"
                    if ref_name not in obs_grp:
                        raise ValueError(
                            f'Missing dataset "{dataset_path}:/data/{demo_key}/obs/{ref_name}". '
                            f"Available keys: {list(obs_grp.keys())[:20]} ..."
                        )
                    for view in self.views:
                        rgb_name = f"{view}_rgb"
                        if rgb_name not in obs_grp:
                            raise ValueError(f'Missing dataset "{dataset_path}:/data/{demo_key}/obs/{rgb_name}".')
                        depth_name = f"{view}_depth"
                        if self.use_depth and depth_name not in obs_grp:
                            raise ValueError(
                                f'Missing depth dataset "{dataset_path}:/data/{demo_key}/obs/{depth_name}". '
                                "Collect depth images or set dataset.use_depth=false."
                            )

                    timesteps = int(obs_grp[ref_name].shape[0])
                    if self.max_frames_per_demo is not None:
                        timesteps = min(timesteps, int(self.max_frames_per_demo))
                    demo_ref = (file_idx, demo_key)
                    self.demo_lengths[demo_ref] = timesteps

                    if "camera_params" not in demo_grp:
                        raise ValueError(f'"/data/{demo_key}" in "{dataset_path}" has no "camera_params" group.')
                    cam_params_grp = demo_grp["camera_params"]
                    if "intrinsics" not in cam_params_grp or "extrinsics_world_T_cam" not in cam_params_grp:
                        raise ValueError(
                            f'"{dataset_path}:/data/{demo_key}/camera_params" must contain "intrinsics" '
                            'and "extrinsics_world_T_cam".'
                        )

                    intrinsics_ds = cam_params_grp["intrinsics"]
                    world_T_cam_ds = cam_params_grp["extrinsics_world_T_cam"]
                    camera_names_list = list(camera_names)
                    self.cam_cache[demo_ref] = {}

                    for view in self.views:
                        cam_idx = camera_names_list.index(view)
                        K = np.array(intrinsics_ds[cam_idx], dtype=np.float32)
                        world_T_cam = np.array(world_T_cam_ds[cam_idx], dtype=np.float32)

                        w2c_gl = invert_4x4(world_T_cam)
                        gl_to_cv = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
                        w2c = gl_to_cv @ w2c_gl
                        c2w = invert_4x4(w2c).astype(np.float32)

                        self.cam_cache[demo_ref][view] = {"K": K, "w2c": w2c, "c2w": c2w}

                    for t_idx in range(timesteps):
                        self.samples.append((file_idx, demo_key, t_idx))

        print(
            f"[RoboSuiteMultiViewTemporalHDF5Dataset] Indexed {len(self.demo_refs)} demos "
            f"from {len(self.dataset_paths)} file(s), {len(self.samples)} samples, views={self.views}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        file_idx, demo_key, t = self.samples[idx]
        demo_ref = (file_idx, demo_key)
        obs_grp = self._get_h5(file_idx)["data"][demo_key]["obs"]

        images = []
        depths = []
        intrinsics = []
        c2w_mats = []
        w2c_mats = []
        for view in self.views:
            img_np = np.array(obs_grp[f"{view}_rgb"][t], dtype=np.uint8)
            images.append(image_to_tensor(img_np))
            if self.use_depth:
                depth_np = np.array(obs_grp[f"{view}_depth"][t], dtype=np.float32)
                if depth_np.ndim == 3 and depth_np.shape[-1] == 1:
                    depth_np = depth_np[..., 0]
                depths.append(torch.from_numpy(depth_np).unsqueeze(0))

            cam = self.cam_cache[demo_ref][view]
            intrinsics.append(torch.from_numpy(cam["K"]))
            c2w_mats.append(torch.from_numpy(cam["c2w"]))
            w2c_mats.append(torch.from_numpy(cam["w2c"]))

        sample = {
            "images": torch.stack(images, dim=0),
            "K": torch.stack(intrinsics, dim=0),
            "c2w": torch.stack(c2w_mats, dim=0),
            "w2c": torch.stack(w2c_mats, dim=0),
            "demo_key": _demo_label(file_idx, demo_key, len(self.dataset_paths)),
            "hdf5_path": self.dataset_paths[file_idx],
            "file_idx": int(file_idx),
            "t": int(t),
        }
        if self.use_depth:
            sample["depths"] = torch.stack(depths, dim=0)
        return sample


# -------------------------------------------------------------------------
# Helpers for building DataLoaders (train / valid)
# -------------------------------------------------------------------------


def _list_demo_keys_robosuite(dataset_path: str) -> List[str]:
    with h5py.File(dataset_path, "r") as f:
        if "data" not in f:
            raise ValueError(f'Invalid dataset "{dataset_path}": missing top-level group "data".')
        demos = list(f["data"].keys())

    def _demo_index(key: str) -> int:
        try:
            return int(key.replace("demo", ""))
        except Exception:
            return 10**18

    return sorted(demos, key=_demo_index)


def _list_demo_refs_robosuite(dataset_paths: Sequence[str]) -> List[DemoRef]:
    refs: List[DemoRef] = []
    for file_idx, path in enumerate(dataset_paths):
        refs.extend((file_idx, demo_key) for demo_key in _list_demo_keys_robosuite(path))
    return refs


def _worker_init_fn(worker_id: int) -> None:
    info = get_worker_info()
    if info is None:
        return
    info.dataset.rng = random.Random(info.seed)


def build_train_valid_loaders_robosuite(
    dataset_path: DatasetPathInput,
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
    use_depth: bool = False,
):
    """Build train/validation loaders with all selected cameras per sample."""
    dataset_paths = _normalize_dataset_paths(dataset_path)
    demo_refs = _list_demo_refs_robosuite(dataset_paths)
    if num_episodes is not None:
        demo_refs = demo_refs[: int(num_episodes)]

    rng = random.Random(seed)
    rng.shuffle(demo_refs)

    n_total = len(demo_refs)
    n_train = max(1, int(n_total * train_ratio))
    n_train = min(n_train, n_total - 1) if n_total > 1 else n_train
    train_refs = demo_refs[:n_train]
    valid_refs = demo_refs[n_train:] if n_total > 1 else demo_refs[:]

    if views is None:
        first_file_idx, first_demo = train_refs[0]
        with h5py.File(dataset_paths[first_file_idx], "r") as f:
            views = list(json.loads(f["data"][first_demo].attrs["camera_names"]))
    else:
        views = list(views)

    if camera_num is not None:
        views = views[: int(camera_num)]

    train_dataset = RoboSuiteMultiViewTemporalHDF5Dataset(
        dataset_path=dataset_paths,
        demo_keys=train_refs,
        views=views,
        camera_num=camera_num,
        max_frames_per_demo=max_frames_per_demo,
        seed=seed,
        min_time_gap=min_time_gap,
        use_depth=use_depth,
    )
    valid_dataset = RoboSuiteMultiViewTemporalHDF5Dataset(
        dataset_path=dataset_paths,
        demo_keys=valid_refs,
        views=views,
        camera_num=camera_num,
        max_frames_per_demo=max_frames_per_demo,
        seed=seed + 999,
        min_time_gap=min_time_gap,
        use_depth=use_depth,
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
