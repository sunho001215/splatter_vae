from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Optional

import h5py
import numpy as np


@dataclass
class DemoMeta:
    env_id: str
    env_name: str
    seed: int
    policy_type: str
    policy_name: str
    camera_names: list[str]
    model_file: str = "unknown"


class HDF5DemoWriter:
    """
    Mirrors your robosuite dataset layout:
      /data/demoX/{states,actions}
      /data/demoX/obs/{cam}_rgb, {cam}_seg
      /data/demoX/obs_env/{...}
      /data/demoX/camera_params/{intrinsics,extrinsics_world_T_cam}
    :contentReference[oaicite:11]{index=11} :contentReference[oaicite:12]{index=12}
    """

    def __init__(self, path: str, mode: str, compression: Optional[str]) -> None:
        self.path = path
        self.mode = mode
        self.compression = compression
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        if mode == "overwrite":
            self.f = h5py.File(path, "w")
        elif mode == "resume":
            self.f = h5py.File(path, "a")
        else:
            raise ValueError("output.mode must be overwrite|resume")

        if "data" not in self.f:
            self.data_grp = self.f.create_group("data")
            self.data_grp.attrs["total"] = 0
        else:
            self.data_grp = self.f["data"]
            if "total" not in self.data_grp.attrs:
                self.data_grp.attrs["total"] = 0

    def next_demo_index(self) -> int:
        existing = [k for k in self.data_grp.keys() if k.startswith("demo")]
        if not existing:
            return 1
        nums = []
        for k in existing:
            try:
                nums.append(int(k.replace("demo", "")))
            except Exception:
                pass
        return (max(nums) + 1) if nums else 1

    def begin_demo(
        self,
        demo_name: str,
        meta: DemoMeta,
        *,
        H: int,
        W: int,
        camera_intrinsics: Dict[str, np.ndarray],
        camera_extrinsics: Dict[str, np.ndarray],
        extra_attrs: Optional[dict] = None,
    ) -> None:
        if demo_name in self.data_grp:
            del self.data_grp[demo_name]
            self.f.flush()

        self.demo_grp = self.data_grp.create_group(demo_name)
        self.demo_grp.attrs["env_id"] = meta.env_id
        self.demo_grp.attrs["env_name"] = meta.env_name
        self.demo_grp.attrs["seed"] = int(meta.seed)
        self.demo_grp.attrs["policy_type"] = meta.policy_type
        self.demo_grp.attrs["policy_name"] = meta.policy_name
        self.demo_grp.attrs["model_file"] = meta.model_file
        self.demo_grp.attrs["num_samples"] = 0
        self.demo_grp.attrs["camera_names"] = json.dumps(meta.camera_names)

        if extra_attrs:
            for k, v in extra_attrs.items():
                self.demo_grp.attrs[str(k)] = v

        cam_grp = self.demo_grp.create_group("camera_params")
        K_stack = np.stack([camera_intrinsics[c] for c in meta.camera_names], axis=0)
        E_stack = np.stack([camera_extrinsics[c] for c in meta.camera_names], axis=0)
        cam_grp.create_dataset("intrinsics", data=K_stack)
        cam_grp.create_dataset("extrinsics_world_T_cam", data=E_stack)

        self.obs_grp = self.demo_grp.create_group("obs")
        self.env_obs_grp = self.demo_grp.create_group("obs_env")

        self.ds_rgb = {}
        self.ds_seg = {}
        for cam in meta.camera_names:
            self.ds_rgb[cam] = self._make_extendable(self.obs_grp, f"{cam}_rgb", (H, W, 3), np.uint8)
            self.ds_seg[cam] = self._make_extendable(self.obs_grp, f"{cam}_seg", (H, W), np.int32)

        self.ds_seg_type = {}  # optional, created if you call append_step with seg_type
        self.ds_states = None
        self.ds_actions = None
        self.ds_rewards = None
        self.ds_dones = None
        self.ds_success = None
        self.ds_obs = None

    def _make_extendable(self, grp, name: str, shape_tail: tuple[int, ...], dtype):
        return grp.create_dataset(
            name,
            shape=(0, *shape_tail),
            maxshape=(None, *shape_tail),
            dtype=dtype,
            chunks=True,
            compression=self.compression,
        )

    def _append_row(self, ds, row: np.ndarray):
        n = ds.shape[0]
        ds.resize((n + 1, *ds.shape[1:]))
        ds[n] = row

    def append_step(
        self,
        *,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        success: bool,
        obs_vec: np.ndarray,
        rgb_by_cam: Dict[str, np.ndarray],
        seg_id_by_cam: Dict[str, np.ndarray],
        seg_type_by_cam: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        # lazy create vector datasets
        if self.ds_states is None:
            self.ds_states = self.demo_grp.create_dataset(
                "states", shape=(0, state.size), maxshape=(None, state.size),
                dtype=np.float64, chunks=True, compression=self.compression
            )
        if self.ds_actions is None:
            self.ds_actions = self.demo_grp.create_dataset(
                "actions", shape=(0, action.size), maxshape=(None, action.size),
                dtype=np.float32, chunks=True, compression=self.compression
            )
        if self.ds_rewards is None:
            self.ds_rewards = self.demo_grp.create_dataset(
                "rewards", shape=(0,), maxshape=(None,),
                dtype=np.float32, chunks=True, compression=self.compression
            )
        if self.ds_dones is None:
            self.ds_dones = self.demo_grp.create_dataset(
                "dones", shape=(0,), maxshape=(None,),
                dtype=np.uint8, chunks=True, compression=self.compression
            )
        if self.ds_success is None:
            self.ds_success = self.demo_grp.create_dataset(
                "success", shape=(0,), maxshape=(None,),
                dtype=np.uint8, chunks=True, compression=self.compression
            )
        if self.ds_obs is None:
            self.ds_obs = self.env_obs_grp.create_dataset(
                "obs", shape=(0, obs_vec.size), maxshape=(None, obs_vec.size),
                dtype=np.float32, chunks=True, compression=self.compression
            )

        self._append_row(self.ds_states, state.astype(np.float64))
        self._append_row(self.ds_actions, action.astype(np.float32))
        self._append_row(self.ds_rewards, np.array(reward, dtype=np.float32))
        self._append_row(self.ds_dones, np.array(int(done), dtype=np.uint8))
        self._append_row(self.ds_success, np.array(int(success), dtype=np.uint8))
        self._append_row(self.ds_obs, obs_vec.astype(np.float32))

        for cam in self.ds_rgb.keys():
            self._append_row(self.ds_rgb[cam], rgb_by_cam[cam].astype(np.uint8))
            self._append_row(self.ds_seg[cam], seg_id_by_cam[cam].astype(np.int32))

        if seg_type_by_cam is not None:
            for cam in seg_type_by_cam.keys():
                if cam not in self.ds_seg_type:
                    self.ds_seg_type[cam] = self._make_extendable(self.obs_grp, f"{cam}_seg_type", self.ds_seg[cam].shape[1:], np.int32)
                self._append_row(self.ds_seg_type[cam], seg_type_by_cam[cam].astype(np.int32))

        # update counts
        self.demo_grp.attrs["num_samples"] = int(self.ds_actions.shape[0])
        self.data_grp.attrs["total"] = int(self.data_grp.attrs["total"]) + 1

    def end_demo(self) -> None:
        self.f.flush()

    def close(self) -> None:
        self.f.flush()
        self.f.close()
