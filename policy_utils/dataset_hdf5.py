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
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .pose10 import pose10_from_obs


def pick_obs_keys(obs: Dict[str, Any]) -> Tuple[str, str, str]:
    """Best-effort selection of (eef_pos_key, eef_quat_key, gripper_key) from robosuite obs dict."""
    keys = list(obs.keys())

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

    def pick(cands, shape_pred) -> Optional[str]:
        for c in cands:
            if c in keys and shape_pred(np.asarray(obs[c]).shape):
                return c
        for k in keys:
            if any(tok in k for tok in cands) and shape_pred(np.asarray(obs[k]).shape):
                return k
        return None

    pos_k = pick(pos_candidates, lambda s: (len(s) >= 1 and s[-1] == 3))
    quat_k = pick(quat_candidates, lambda s: (len(s) >= 1 and s[-1] == 4))
    grip_k = pick(grip_candidates, lambda s: True)

    if pos_k is None or quat_k is None or grip_k is None:
        raise RuntimeError(
            "Could not locate required keys in robosuite obs dict.\n"
            f"Found keys: {keys}\n"
            f"Picked: pos={pos_k}, quat={quat_k}, grip={grip_k}"
        )
    return pos_k, quat_k, grip_k

def reduce_gripper(grip_raw: np.ndarray) -> np.ndarray:
    """
    Convert gripper observations into a scalar per timestep.
    - If (...,2) -> select the first finger joint only.
    """
    return grip_raw[..., 0]

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
        seed: int = 0,
    ):
        super().__init__()
        self.hdf5_path = str(hdf5_path)
        self.window = window
        self.seed = int(seed)

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

                t_min = -min_obs
                t_max = T - 1 - max_obs
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

    @staticmethod
    def _h5_take(ds: h5py.Dataset, rows: List[int], col: Optional[int] = None) -> np.ndarray:
        """
        Safe 'take' for h5py datasets.

        h5py fancy indexing requires indices to be strictly increasing (no duplicates).
        When we replicate the last timestep (T-1), indices like [..., T-1, T-1, T-1]
        appear and would crash. This helper:
        1) reads using sorted unique indices
        2) gathers back to the original order (including duplicates)
        """
        rows_np = np.asarray(rows, dtype=np.int64)
        uniq = np.unique(rows_np)  # sorted unique

        if col is None:
            block = ds[uniq]
        else:
            block = ds[uniq, col]

        inv = np.searchsorted(uniq, rows_np)  # map each requested row -> index in uniq
        return block[inv]

    def _clamp_indices_to_last(self, t_list: List[int], T: int) -> List[int]:
        """
        Replicate the last timestep for any out-of-range index.
        This implements: "if timestep > T-1, use T-1".
        (Also clamps negative indices defensively.)
        """
        last = T - 1
        return [last if t > last else (0 if t < 0 else t) for t in t_list]
    
    def _load_pose10_seq_abs_with_grip_state(
        self,
        env_obs_grp: h5py.Group,
        pos_k: str,
        quat_k: str,
        grip_k: str,
        t_indices: List[int],
    ) -> np.ndarray:
        """
        Load absolute 10D pose for multiple timesteps using gripper *state* from obs_env.
        Returns: (N,10) float64
        """
        pos = self._h5_take(env_obs_grp[pos_k], t_indices).astype(np.float64)    # (N,3)
        quat = self._h5_take(env_obs_grp[quat_k], t_indices).astype(np.float64)  # (N,4)

        # Gripper state may be scalar or vector per timestep -> reduce to scalar
        grip_raw = self._h5_take(env_obs_grp[grip_k], t_indices)
        grip = reduce_gripper(np.asarray(grip_raw)).astype(np.float64).reshape(-1)  # (N,)

        out = np.zeros((len(t_indices), 10), dtype=np.float64)
        for i in range(len(t_indices)):
            out[i] = pose10_from_obs(pos[i], quat[i], float(grip[i]))  # <- gripper *state*
        return out

    def _load_pose10_seq_abs_with_grip_cmd(
        self,
        env_obs_grp: h5py.Group,
        pos_k: str,
        quat_k: str,
        grip_cmd_seq: np.ndarray,   # (N,) gripper command at each requested timestep (e.g., +/-1)
        t_indices: List[int],
    ) -> np.ndarray:
        """
        Load absolute 10D pose for multiple timesteps, but use gripper *command*
        (from /actions[:, -1]) instead of gripper *state* (from obs_env).
        Returns: (N,10) float64
        """
        pos = self._h5_take(env_obs_grp[pos_k], t_indices).astype(np.float64)    # (N,3)
        quat = self._h5_take(env_obs_grp[quat_k], t_indices).astype(np.float64)  # (N,4)

        grip_cmd_seq = np.asarray(grip_cmd_seq, dtype=np.float64).reshape(-1)  # (N,)

        out = np.zeros((len(t_indices), 10), dtype=np.float64)
        for i in range(len(t_indices)):
            out[i] = pose10_from_obs(pos[i], quat[i], float(grip_cmd_seq[i]))  # <- command, not state
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
        actions_ds = demo["actions"]

        # ---- Randomly pick one camera among the 6 ----
        cam = self.camera_names[int(self._rng.integers(0, len(self.camera_names)))]
        rgb_key = self._resolve_cam_rgb_key(obs_grp, cam)
        if rgb_key is None:
            raise KeyError(f"Could not find RGB dataset for camera '{cam}' in demo '{demo_name}'")

        rgb_ds = obs_grp[rgb_key]  # (T,H,W,3) uint8
        T, H, W = int(rgb_ds.shape[0]), int(rgb_ds.shape[1]), int(rgb_ds.shape[2])

        # Observations: e.g. [-1,0]
        obs_t = [center_t + di for di in self.window.obs_indices]
        obs_imgs = np.asarray(rgb_ds[obs_t], dtype=np.uint8)  # (n_obs,H,W,3)

        # (2,3,ch,cw) float in [0,1]
        obs_img_t = torch.from_numpy(obs_imgs).permute(0, 3, 1, 2).contiguous().float() / 255.0

        # ---- obs_env keys ----
        pos_k, quat_k, grip_k = pick_obs_keys(env_obs_grp)

        # observation.state: (n_obs, 10)
        # Use gripper *state* from obs_env (not action command)
        obs_state = self._load_pose10_seq_abs_with_grip_state(
            env_obs_grp, pos_k, quat_k, grip_k, obs_t
        )
        obs_state_t = torch.from_numpy(obs_state).float()

        # Action targets timesteps (length == horizon)
        act_t_raw = [center_t + di for di in self.window.act_indices]

        # Replicate last timestep (T-1) for out-of-range action indices
        act_t = self._clamp_indices_to_last(act_t_raw, T)

        # Gripper command for these timesteps (safe read even if act_t has duplicates)
        act_grip_cmd = self._h5_take(actions_ds, act_t, col=-1).astype(np.float64)

        act_vec = self._load_pose10_seq_abs_with_grip_cmd(
            env_obs_grp, pos_k, quat_k, act_grip_cmd, act_t
        )
        act_torch = torch.from_numpy(act_vec).float()

        # Always set action_is_pad = False (no padding is ignored)
        action_is_pad = torch.zeros((len(self.window.act_indices),), dtype=torch.bool)

        return {
            "observation.image": obs_img_t,
            "observation.state": obs_state_t,
            "action": act_torch,
            "action_is_pad": action_is_pad,
        }

