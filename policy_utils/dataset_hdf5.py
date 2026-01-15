"""
A PyTorch Dataset that reads your robosuite HDF5 file and yields batches
compatible with LeRobot's DiffusionPolicy.

Your HDF5 format (from your collector):
  /data/demoX/obs/{cam}_rgb     : (T, H, W, 3) uint8
  /data/demoX/obs_env/<key>     : (T, ...) arrays from robosuite obs dict

What we output per sample:
  - observation.image : (2, 3, 224, 224) float32 in [0, 1]
      Two consecutive frames from a randomly chosen camera among cam0..cam5
  - observation.state : (2, 10) float32
      Two 10D pose vectors matching the two frames (t-1, t).
  - action            : (16, 10) float32
      16 future targets in 10D.

This matches the *structure* used in LeRobot diffusion training examples:
  - observation_delta_indices typically [-1, 0]
  - action_delta_indices typically [-1, 0, ..., 14]  (16 steps)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .pose10 import pose10_from_obs, delta_pose10


@dataclass(frozen=True)
class WindowSpec:
    """
    Defines which timesteps are used for:
      - observation frames / states (e.g., [-1, 0] for two consecutive frames)
      - action targets (e.g., [-1..14] for 16 steps)
    """
    obs_indices: List[int]
    act_indices: List[int]


class RobosuiteHDF5DiffusionDataset(Dataset):
    """
    Dataset that:
      - builds a global index of valid (demo_name, center_t) pairs
      - in __getitem__, samples a random camera

    Notes on HDF5 + DataLoader:
      - Each worker process must open its own HDF5 file handle.
      - We open lazily in each worker the first time __getitem__ runs.
    """

    def __init__(
        self,
        hdf5_path: str,
        window: WindowSpec,
        camera_names: Optional[List[str]] = None,
        action_mode: str = "abs_pose",  # "delta_pose" or "abs_pose"
        seed: int = 0,
    ):
        super().__init__()
        self.hdf5_path = str(hdf5_path)
        self.window = window
        self.action_mode = str(action_mode)
        self.seed = int(seed)

        if self.action_mode not in ("delta_pose", "abs_pose"):
            raise ValueError("action_mode must be 'delta_pose' or 'abs_pose'")

        # Build a global (demo_name, center_t) index so sampling is O(1).
        self._index: List[Tuple[str, int]] = []

        # Cache for per-demo obs_env key selection (speeds up __getitem__).
        self._demo_env_keys: Dict[str, Tuple[str, str, str]] = {}

        # Discover demo list and camera list by reading the file once in the main process.
        with h5py.File(self.hdf5_path, "r") as f:
            if "data" not in f:
                raise RuntimeError("Invalid file: missing '/data' group.")
            data_grp = f["data"]
            demo_names = sorted([k for k in data_grp.keys() if k.startswith("demo")])
            if not demo_names:
                raise RuntimeError("No demos found under /data (expected /data/demo1, /data/demo2, ...)")

            # Determine camera names:
            # - If caller passes camera_names, trust it.
            # - Otherwise read demo attrs["camera_names"] (JSON string) created by your writer.
            if camera_names is None:
                cam_json = data_grp[demo_names[0]].attrs.get("camera_names", None)
                if cam_json is None:
                    raise RuntimeError("Missing demo attrs['camera_names'] (expected JSON list).")
                self.camera_names = list(json.loads(cam_json))
            else:
                self.camera_names = list(camera_names)

            # Support both "cam0" and "cam_0" naming, by remembering both forms.
            # We'll try exact match first, then fallback to underscore variant.
            if len(self.camera_names) != 6:
                raise RuntimeError(f"Expected 6 cameras, got {len(self.camera_names)}: {self.camera_names}")

            # Determine which center_t indices are valid.
            min_obs, max_obs = min(self.window.obs_indices), max(self.window.obs_indices)
            min_act, max_act = min(self.window.act_indices), max(self.window.act_indices)

            # A center_t is valid if all requested indices are within [0, T-1].
            for dn in demo_names:
                demo = data_grp[dn]
                if "obs" not in demo:
                    continue
                obs_grp = demo["obs"]

                # Use the first camera stream to read T,H,W.
                cam0 = self.camera_names[0]
                key0 = self._resolve_cam_rgb_key(obs_grp, cam0)
                if key0 is None:
                    continue
                T = int(obs_grp[key0].shape[0])

                t_min = max(-min_obs, -min_act)
                t_max = min(T - 1 - max_obs, T - 1 - max_act)
                if t_max < t_min:
                    continue

                for t in range(t_min, t_max + 1):
                    self._index.append((dn, t))

        if not self._index:
            raise RuntimeError("No valid windows found. Check demo lengths and window indices.")

        # Worker-local state (opened lazily)
        self._h5: Optional[h5py.File] = None
        self._rng: Optional[np.random.Generator] = None

    def __len__(self) -> int:
        return len(self._index)

    def __del__(self):
        # Best-effort cleanup in case the dataset is destroyed.
        try:
            if self._h5 is not None:
                self._h5.close()
        except Exception:
            pass

    # ------------------------- internal helpers -------------------------

    @staticmethod
    def _resolve_cam_rgb_key(obs_grp: h5py.Group, cam_name: str) -> Optional[str]:
        """
        Resolve the dataset name for an RGB stream.
        Typical keys are:
          - f"{cam}_rgb" (e.g., "cam0_rgb")
        But some people name cameras "cam_0" instead of "cam0".

        We try:
          1) exact cam_name
          2) swap between "cam0" <-> "cam_0" forms
        """
        k = f"{cam_name}_rgb"
        if k in obs_grp:
            return k

        # Try underscore variant mapping
        if cam_name.startswith("cam") and cam_name[3:].isdigit():
            # cam0 -> cam_0
            k2 = f"cam_{cam_name[3:]}_rgb"
            if k2 in obs_grp:
                return k2
        if cam_name.startswith("cam_") and cam_name[4:].isdigit():
            # cam_0 -> cam0
            k2 = f"cam{cam_name[4:]}_rgb"
            if k2 in obs_grp:
                return k2

        return None

    def _ensure_open_and_rng(self):
        """
        Called at the start of __getitem__.

        - Opens an HDF5 handle per worker (important for multiprocessing DataLoader).
        - Creates a per-worker RNG so camera sampling is independent per worker
          but still reproducible overall (seed + worker_id offset).
        """
        if self._h5 is None:
            self._h5 = h5py.File(self.hdf5_path, "r")

        if self._rng is None:
            wi = torch.utils.data.get_worker_info()
            worker_id = 0 if wi is None else int(wi.id)
            self._rng = np.random.default_rng(self.seed + 1000 * worker_id)

    def _pick_obs_env_keys(self, env_obs_grp: h5py.Group, demo_name: str) -> Tuple[str, str, str]:
        """
        Identify keys in /obs_env for:
          - eef position (3,)
          - eef quaternion (4,) in xyzw
          - gripper signal (scalar or small vector)

        We cache per demo_name so this is done once per demo in each worker.
        """
        if demo_name in self._demo_env_keys:
            return self._demo_env_keys[demo_name]

        keys = list(env_obs_grp.keys())

        # Common robosuite naming
        pos_candidates = ["robot0_eef_pos", "eef_pos"]
        quat_candidates = ["robot0_eef_quat", "eef_quat"]
        grip_candidates = [
            "robot0_gripper_qpos",
            "robot0_gripper_pos",
            "robot0_gripper_state",
            "gripper_qpos",
            "gripper_pos",
            "gripper_state",
        ]

        def pick(cands: List[str], shape_pred) -> Optional[str]:
            # 1) direct match on common names
            for c in cands:
                if c in keys and shape_pred(env_obs_grp[c].shape):
                    return c
            # 2) fuzzy fallback: contains substring
            for k in keys:
                if any(tok in k for tok in cands) and shape_pred(env_obs_grp[k].shape):
                    return k
            return None

        pos_k = pick(pos_candidates, lambda s: len(s) >= 2 and s[-1] == 3)
        quat_k = pick(quat_candidates, lambda s: len(s) >= 2 and s[-1] == 4)
        grip_k = pick(grip_candidates, lambda s: True)

        if pos_k is None or quat_k is None or grip_k is None:
            raise RuntimeError(
                "Could not locate required obs_env keys.\n"
                f"Found keys: {keys}\n"
                f"Picked: pos={pos_k}, quat={quat_k}, grip={grip_k}"
            )

        self._demo_env_keys[demo_name] = (pos_k, quat_k, grip_k)
        return pos_k, quat_k, grip_k

    @staticmethod
    def _reduce_gripper(grip_raw: np.ndarray) -> np.ndarray:
        """
        Convert gripper observations into a scalar per timestep.

        - If already (T,) -> use it
        - If (T,2) or similar -> mean over last dim
        """
        grip_raw = np.asarray(grip_raw)
        if grip_raw.ndim == 1:
            return grip_raw.astype(np.float64)
        return np.mean(grip_raw.astype(np.float64), axis=-1)

    def _load_pose10_seq_abs(
        self,
        env_obs_grp: h5py.Group,
        pos_k: str,
        quat_k: str,
        grip_k: str,
        t_indices: List[int],
    ) -> np.ndarray:
        """
        Load absolute 10D pose for multiple timesteps.
        Returns: (N,10) float64
        """
        pos = np.asarray(env_obs_grp[pos_k][t_indices], dtype=np.float64)     # (N,3)
        quat = np.asarray(env_obs_grp[quat_k][t_indices], dtype=np.float64)   # (N,4)
        grip = self._reduce_gripper(env_obs_grp[grip_k][t_indices])           # (N,)

        out = np.zeros((len(t_indices), 10), dtype=np.float64)
        for i in range(len(t_indices)):
            out[i] = pose10_from_obs(pos[i], quat[i], float(grip[i]))
        return out

    def _load_pose10_seq_delta(
        self,
        env_obs_grp: h5py.Group,
        pos_k: str,
        quat_k: str,
        grip_k: str,
        ref_t: int,
        t_indices: List[int],
    ) -> np.ndarray:
        """
        Load delta-pose 10D targets relative to a reference timestep ref_t.
        Returns: (N,10) float64
        """
        pos_all = np.asarray(env_obs_grp[pos_k][[ref_t] + t_indices], dtype=np.float64)   # (1+N,3)
        quat_all = np.asarray(env_obs_grp[quat_k][[ref_t] + t_indices], dtype=np.float64) # (1+N,4)
        grip_all = self._reduce_gripper(env_obs_grp[grip_k][[ref_t] + t_indices])          # (1+N,)

        ref_pos, ref_quat = pos_all[0], quat_all[0]

        out = np.zeros((len(t_indices), 10), dtype=np.float64)
        for i, t in enumerate(t_indices):
            tgt_pos = pos_all[i + 1]
            tgt_quat = quat_all[i + 1]
            tgt_grip = float(grip_all[i + 1])
            out[i] = delta_pose10(ref_pos, ref_quat, tgt_pos, tgt_quat, tgt_grip)
        return out

    # ------------------------- main access -------------------------

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a dict that LeRobot DiffusionPolicy can consume directly.

        Required keys for DiffusionModel.compute_loss include:
        - "observation.state"
        - "action"
        - "action_is_pad"   (padding mask; False where action is real)
        LeRobot asserts these keys exist. Since we only sample valid full windows,
        we set action_is_pad = all False.
        """
        self._ensure_open_and_rng()
        assert self._h5 is not None and self._rng is not None

        demo_name, center_t = self._index[idx]
        demo = self._h5["data"][demo_name]
        obs_grp = demo["obs"]
        env_obs_grp = demo["obs_env"]

        # ---- Randomly pick one camera among the 6 ----
        cam = self.camera_names[int(self._rng.integers(0, len(self.camera_names)))]
        rgb_key = self._resolve_cam_rgb_key(obs_grp, cam)
        if rgb_key is None:
            raise KeyError(f"Could not find RGB dataset for camera '{cam}' in demo '{demo_name}'")

        rgb_ds = obs_grp[rgb_key]  # (T,H,W,3) uint8
        H, W = int(rgb_ds.shape[1]), int(rgb_ds.shape[2])

        # Observations: e.g. [-1,0]
        obs_t = [center_t + di for di in self.window.obs_indices]
        obs_imgs = np.asarray(rgb_ds[obs_t], dtype=np.uint8)  # (n_obs,H,W,3)

        # (2,3,ch,cw) float in [0,1]
        obs_img_t = torch.from_numpy(obs_imgs).permute(0, 3, 1, 2).contiguous().float() / 255.0

        # ---- obs_env keys ----
        pos_k, quat_k, grip_k = self._pick_obs_env_keys(env_obs_grp, demo_name)

        # observation.state: (2,10)
        obs_state = self._load_pose10_seq_abs(env_obs_grp, pos_k, quat_k, grip_k, obs_t)
        obs_state_t = torch.from_numpy(obs_state).float()

        # ---- Action targets timesteps (length == horizon) ----
        act_t = [center_t + di for di in self.window.act_indices]

        if self.action_mode == "abs_pose":
            act_vec = self._load_pose10_seq_abs(env_obs_grp, pos_k, quat_k, grip_k, act_t)
        else:
            act_vec = self._load_pose10_seq_delta(
                env_obs_grp, pos_k, quat_k, grip_k, ref_t=center_t, t_indices=act_t
            )

        act_torch = torch.from_numpy(act_vec).float()

        # ---- REQUIRED by LeRobot diffusion loss: padding mask ----
        # Shape must align with action sequence length (horizon).
        # False = real action, True = padded action.
        action_is_pad = torch.zeros((len(self.window.act_indices),), dtype=torch.bool)

        return {
            "observation.image": obs_img_t,
            "observation.state": obs_state_t,
            "action": act_torch,
            "action_is_pad": action_is_pad,
        }

