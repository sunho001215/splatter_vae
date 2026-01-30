import json
from dataclasses import dataclass
from typing import List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class RobosuiteSinCroDatasetConfig:
    hdf5_path: str
    # number of cameras / views used for training (must be <= number in HDF5)
    num_views: int = 6
    # number of timesteps per training sample (SinCro often uses 3 * time_interval)
    sequence_length: int = 9
    # optional: only take first N episodes
    max_episodes: Optional[int] = None
    # optional: truncate each demo to first N frames
    max_frames_per_demo: Optional[int] = None
    # stride when making temporal windows inside one demo
    temporal_stride: int = 1
    # if not None, explicit camera names (subset of those saved in HDF5)
    camera_names: Optional[List[str]] = None


class RobosuiteSinCroSequenceDataset(Dataset):
    """
    Dataset that reads your robosuite HDF5 demos and produces SinCro-style
    multi-view temporal windows.

    Each item:
        {
            "images": [T, H, V, W, 3]  float32 in [0,1]
            "K":      [V, 3, 3]        float32
            "c2w":    [V, 4, 4]        float32 (camera-to-world)
        }

    T = sequence_length
    V = num_views
    """

    def __init__(self, cfg: RobosuiteSinCroDatasetConfig):
        super().__init__()
        self.cfg = cfg
        self.hdf5_path = cfg.hdf5_path
        self.sequence_length = cfg.sequence_length
        self.num_views = cfg.num_views
        self.temporal_stride = cfg.temporal_stride
        self.max_frames_per_demo = cfg.max_frames_per_demo

        # We only store *metadata* and indices here; images are read lazily in __getitem__
        self.demo_meta = []  # list of dicts with length, H, W, cam_names
        self.indices: List[Tuple[int, int]] = []  # (demo_idx, start_t)

        self._build_index()

    # ------------------------------------------------------------------ #
    # Index building: which (demo, start_t) pairs are valid training windows?
    # ------------------------------------------------------------------ #
    def _build_index(self):
        with h5py.File(self.hdf5_path, "r") as f:
            data_grp = f["data"]
            # Expect demos named "demo1", "demo2", ...
            demo_names = sorted(
                [k for k in data_grp.keys() if k.lower().startswith("demo")]
            )

            if self.cfg.max_episodes is not None:
                demo_names = demo_names[: self.cfg.max_episodes]

            if len(demo_names) == 0:
                raise RuntimeError("No demo groups found under /data in HDF5.")

            for demo_idx, demo_name in enumerate(demo_names):
                demo_grp = data_grp[demo_name]
                obs_grp = demo_grp["obs"]

                # Camera list saved as JSON string (from your AsyncHDF5Writer)
                cam_names_all = json.loads(demo_grp.attrs["camera_names"])
                if self.cfg.camera_names is not None:
                    cam_names = self.cfg.camera_names
                else:
                    cam_names = cam_names_all[: self.num_views]

                if len(cam_names) < self.num_views:
                    raise ValueError(
                        f"Demo {demo_name} only has {len(cam_names_all)} cameras, "
                        f"but num_views={self.num_views} was requested."
                    )

                # Get sequence length and resolution from the first camera
                first_cam = cam_names[0] + "_rgb"
                T_full, H, W, C = obs_grp[first_cam].shape
                assert C == 3, "Expected RGB images with 3 channels."

                if self.max_frames_per_demo is not None:
                    T = min(T_full, self.max_frames_per_demo)
                else:
                    T = T_full

                self.demo_meta.append(
                    dict(
                        name=demo_name,
                        length=T,
                        H=H,
                        W=W,
                        cam_names=cam_names,
                    )
                )

                # All valid start indices for a window of length `sequence_length`
                max_start = T - self.sequence_length
                if max_start < 0:
                    # Demo too short; skip it
                    continue

                for start_t in range(0, max_start + 1, self.temporal_stride):
                    self.indices.append((demo_idx, start_t))

        if len(self.indices) == 0:
            raise RuntimeError(
                "No valid (demo, time) pairs found. "
                "Maybe sequence_length is too long for your demos."
            )

        print(
            f"[Dataset] Built RobosuiteSinCroSequenceDataset from {len(self.demo_meta)} demos "
            f"with {len(self.indices)} temporal windows."
        )

    # ------------------------------------------------------------------ #
    # Standard PyTorch Dataset API
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        demo_idx, start_t = self.indices[idx]
        meta = self.demo_meta[demo_idx]
        demo_name = meta["name"]
        T = self.sequence_length
        H, W = meta["H"], meta["W"]
        cam_names = meta["cam_names"]
        V = len(cam_names)

        # We *re-open* the file inside __getitem__ to be safe with multiple workers.
        with h5py.File(self.hdf5_path, "r") as f:
            demo_grp = f["data"][demo_name]
            obs_grp = demo_grp["obs"]
            cam_param_grp = demo_grp["camera_params"]

            intrinsics_all = np.asarray(cam_param_grp["intrinsics"])  # [num_cams,3,3]
            extrinsics_wTc_all = np.asarray(
                cam_param_grp["extrinsics_world_T_cam"]
            )  # [num_cams,4,4]

            # Map cam name -> index into intrinsics/extrinsics arrays
            saved_cam_names = json.loads(demo_grp.attrs["camera_names"])
            name_to_idx = {n: i for i, n in enumerate(saved_cam_names)}

            cam_indices = [name_to_idx[n] for n in cam_names]
            Ks = intrinsics_all[cam_indices]  # [V,3,3]
            c2w = extrinsics_wTc_all[cam_indices]  # [V,4,4]

            # ------------------------------------------------------------------
            # Load RGB frames for requested time window and cameras
            # ------------------------------------------------------------------
            # images_seq: [T, V, H, W, 3]
            images_seq = np.zeros((T, V, H, W, 3), dtype=np.float32)

            for v_idx, cam_name in enumerate(cam_names):
                dset = obs_grp[f"{cam_name}_rgb"]
                # slice temporal window [start_t : start_t+T]
                frames = dset[start_t : start_t + T].astype(np.float32) / 255.0
                # frames: [T,H,W,3]
                images_seq[:, v_idx] = frames

        # For SinCro we like [T, H, V, W, 3] with channel-last
        images_seq = np.transpose(images_seq, (0, 2, 1, 3, 4))  # [T,H,V,W,3]

        sample = {
            "images": torch.from_numpy(images_seq),         # float32 [T,H,V,W,3]
            "K": torch.from_numpy(Ks).float(),              # [V,3,3]
            "c2w": torch.from_numpy(c2w).float(),           # [V,4,4]
        }
        return sample