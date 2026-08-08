import json
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, get_worker_info

from utils.general_utils import image_to_tensor

DemoRef = Tuple[int, str]
SampleRef = Tuple[int, str, int]


def _validate_dataset_path(dataset_path: str) -> str:
    if not isinstance(dataset_path, (str, bytes)) or not str(dataset_path):
        raise ValueError("Exactly one HDF5 dataset path is required per training run.")
    return str(dataset_path)


def _demo_label(file_idx: int, demo_key: str, num_files: int) -> str:
    return demo_key if num_files == 1 else f"file{file_idx}:{demo_key}"


class MetaWorldMultiViewAllCamerasHDF5Dataset(Dataset):
    """Multi-view Meta-World dataset for ReViWo.

    One item is one environment state with all selected cameras. Each training
    run is intentionally scoped to one environment HDF5 file.
    """

    def __init__(
        self,
        dataset_path: str,
        demo_keys: List[Union[str, DemoRef]],
        views: Optional[List[str]] = None,
        max_frames_per_demo: Optional[int] = None,
        seed: int = 0,
        min_time_gap: int = 10,
    ):
        super().__init__()
        self.dataset_path = _validate_dataset_path(dataset_path)
        self.dataset_paths = [self.dataset_path]
        self.demo_refs: List[DemoRef] = [
            (int(item[0]), str(item[1])) if isinstance(item, tuple) else (0, str(item))
            for item in demo_keys
        ]
        self.max_frames_per_demo = max_frames_per_demo
        self.min_time_gap = int(min_time_gap)
        self.rng = random.Random(seed)
        self.views: Optional[List[str]] = None if views is None else list(views)

        self._h5_handles: Dict[int, h5py.File] = {}
        self.demo_lengths: Dict[DemoRef, int] = {}
        self.samples: List[SampleRef] = []

        self._index_files_and_build_samples()

        if self.views is None or len(self.views) < 1:
            raise ValueError("You need at least 1 view (camera name).")

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
                        self.views = list(camera_names)
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

                    timesteps = int(obs_grp[ref_name].shape[0])
                    if self.max_frames_per_demo is not None:
                        timesteps = min(timesteps, int(self.max_frames_per_demo))
                    demo_ref = (file_idx, demo_key)
                    self.demo_lengths[demo_ref] = timesteps

                    for t_idx in range(timesteps):
                        self.samples.append((file_idx, demo_key, t_idx))

        print(
            f"[MetaWorldMultiViewAllCamerasHDF5Dataset] Indexed {len(self.demo_refs)} demos "
            f"from {len(self.dataset_paths)} file(s), {len(self.samples)} samples, views={self.views}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        file_idx, demo_key, t = self.samples[idx]
        obs_grp = self._get_h5(file_idx)["data"][demo_key]["obs"]

        images_list = []
        for view in self.views:
            img_np = np.array(obs_grp[f"{view}_rgb"][t], dtype=np.uint8)
            images_list.append(image_to_tensor(img_np))

        return {
            "images": torch.stack(images_list, dim=0),
            "demo_key": _demo_label(file_idx, demo_key, len(self.dataset_paths)),
            "hdf5_path": self.dataset_paths[file_idx],
            "file_idx": int(file_idx),
            "t": int(t),
        }


# -------------------------------------------------------------------------
# Helpers for building DataLoaders (train / valid)
# -------------------------------------------------------------------------

def _list_demo_keys_metaworld(dataset_path: str) -> List[str]:
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


def _worker_init_fn(worker_id: int):
    info = get_worker_info()
    if info is None:
        return
    info.dataset.rng = random.Random(info.seed)


def build_train_valid_loaders_metaworld(
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
    """Build PyTorch DataLoaders for one environment HDF5 demo file."""
    dataset_path = _validate_dataset_path(dataset_path)
    dataset_paths = [dataset_path]
    demo_refs = [(0, demo_key) for demo_key in _list_demo_keys_metaworld(dataset_path)]
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

    train_dataset = MetaWorldMultiViewAllCamerasHDF5Dataset(
        dataset_path=dataset_path,
        demo_keys=train_refs,
        views=views,
        max_frames_per_demo=max_frames_per_demo,
        seed=seed,
        min_time_gap=min_time_gap,
    )
    valid_dataset = MetaWorldMultiViewAllCamerasHDF5Dataset(
        dataset_path=dataset_path,
        demo_keys=valid_refs,
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
        persistent_workers=(num_workers > 0),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=shuffle_valid,
        num_workers=max(1, num_workers // 2),
        pin_memory=pin_memory,
        drop_last=True,
        worker_init_fn=_worker_init_fn,
        persistent_workers=(num_workers > 0),
    )

    return train_loader, valid_loader
