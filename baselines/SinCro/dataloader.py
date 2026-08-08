import json
from dataclasses import dataclass
from typing import List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

IndexRef = Tuple[int, int, int]


@dataclass
class MetaWorldSinCroDatasetConfig:
    hdf5_path: str = ""
    # number of cameras / views used for training (must be <= number in HDF5)
    num_views: int = 6
    # number of timesteps per training sample (SinCro often uses 3 * time_interval)
    sequence_length: int = 9
    # optional: only take first N episodes from this environment
    max_episodes: Optional[int] = None
    # optional: truncate each demo to first N frames
    max_frames_per_demo: Optional[int] = None
    # stride when making temporal windows inside one demo
    temporal_stride: int = 1
    # if not None, explicit camera names (subset of those saved in HDF5)
    camera_names: Optional[List[str]] = None


class MetaWorldSinCroSequenceDataset(Dataset):
    """MetaWorld HDF5 demos to SinCro-style multi-view temporal windows.

    Each dataset instance is intentionally scoped to one environment HDF5 file.
    """

    def __init__(self, cfg: MetaWorldSinCroDatasetConfig):
        super().__init__()
        self.cfg = cfg
        if not cfg.hdf5_path:
            raise ValueError("Exactly one HDF5 path is required per training run.")
        self.hdf5_path = str(cfg.hdf5_path)
        self.hdf5_paths = [self.hdf5_path]
        self.sequence_length = cfg.sequence_length
        self.num_views = cfg.num_views
        self.temporal_stride = cfg.temporal_stride
        self.max_frames_per_demo = cfg.max_frames_per_demo

        # Metadata only; images are read lazily in __getitem__.
        self.demo_meta = []  # list of dicts with file_idx, name, length, H, W, cam_names
        self.indices: List[IndexRef] = []  # (file_idx, demo_idx, start_t)

        self._build_index()

    def _build_index(self):
        episode_count = 0
        for file_idx, hdf5_path in enumerate(self.hdf5_paths):
            with h5py.File(hdf5_path, "r") as f:
                data_grp = f["data"]
                demo_names = sorted([k for k in data_grp.keys() if k.lower().startswith("demo")])

                remaining = None
                if self.cfg.max_episodes is not None:
                    remaining = int(self.cfg.max_episodes) - episode_count
                    if remaining <= 0:
                        break
                    demo_names = demo_names[:remaining]

                for demo_name in demo_names:
                    demo_grp = data_grp[demo_name]
                    obs_grp = demo_grp["obs"]

                    cam_names_all = json.loads(demo_grp.attrs["camera_names"])
                    if self.cfg.camera_names is not None:
                        cam_names = list(self.cfg.camera_names)
                    else:
                        cam_names = cam_names_all[: self.num_views]

                    if len(cam_names) < self.num_views:
                        raise ValueError(
                            f"{hdf5_path}:{demo_name} only has {len(cam_names_all)} cameras, "
                            f"but num_views={self.num_views} was requested."
                        )
                    missing = [name for name in cam_names if name not in cam_names_all]
                    if missing:
                        raise ValueError(f"{hdf5_path}:{demo_name} missing cameras {missing}.")

                    first_cam = cam_names[0] + "_rgb"
                    T_full, H, W, C = obs_grp[first_cam].shape
                    assert C == 3, "Expected RGB images with 3 channels."

                    T = min(T_full, self.max_frames_per_demo) if self.max_frames_per_demo is not None else T_full

                    demo_idx = len(self.demo_meta)
                    self.demo_meta.append(
                        dict(
                            file_idx=file_idx,
                            hdf5_path=hdf5_path,
                            name=demo_name,
                            length=T,
                            H=H,
                            W=W,
                            cam_names=cam_names,
                        )
                    )
                    episode_count += 1

                    max_start = T - self.sequence_length
                    if max_start < 0:
                        continue

                    for start_t in range(0, max_start + 1, self.temporal_stride):
                        self.indices.append((file_idx, demo_idx, start_t))

        if len(self.demo_meta) == 0:
            raise RuntimeError("No demo groups found under /data in the configured HDF5 files.")
        if len(self.indices) == 0:
            raise RuntimeError(
                "No valid (demo, time) pairs found. Maybe sequence_length is too long for your demos."
            )

        print(
            f"[Dataset] Built MetaWorldSinCroSequenceDataset from {len(self.demo_meta)} demos "
            f"across {len(self.hdf5_paths)} file(s) with {len(self.indices)} temporal windows."
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        _file_idx, demo_idx, start_t = self.indices[idx]
        meta = self.demo_meta[demo_idx]
        demo_name = meta["name"]
        T = self.sequence_length
        H, W = meta["H"], meta["W"]
        cam_names = meta["cam_names"]
        V = len(cam_names)
        hdf5_path = meta["hdf5_path"]

        with h5py.File(hdf5_path, "r") as f:
            demo_grp = f["data"][demo_name]
            obs_grp = demo_grp["obs"]
            cam_param_grp = demo_grp["camera_params"]

            intrinsics_all = np.asarray(cam_param_grp["intrinsics"])
            extrinsics_wTc_all = np.asarray(cam_param_grp["extrinsics_world_T_cam"])

            saved_cam_names = json.loads(demo_grp.attrs["camera_names"])
            name_to_idx = {name: i for i, name in enumerate(saved_cam_names)}

            cam_indices = [name_to_idx[name] for name in cam_names]
            Ks = intrinsics_all[cam_indices]
            c2w = extrinsics_wTc_all[cam_indices]

            images_seq = np.zeros((T, V, H, W, 3), dtype=np.float32)
            for v_idx, cam_name in enumerate(cam_names):
                dset = obs_grp[f"{cam_name}_rgb"]
                frames = dset[start_t : start_t + T].astype(np.float32) / 255.0
                images_seq[:, v_idx] = frames

        images_seq = np.transpose(images_seq, (0, 2, 1, 3, 4))  # [T,H,V,W,3]

        return {
            "images": torch.from_numpy(images_seq),
            "K": torch.from_numpy(Ks).float(),
            "c2w": torch.from_numpy(c2w).float(),
            "demo_key": demo_name if len(self.hdf5_paths) == 1 else f"file{meta['file_idx']}:{demo_name}",
            "hdf5_path": hdf5_path,
        }
