"""
Compute normalization stats for the 10D vector features:
  - observation.state
  - action

LeRobot diffusion examples pass dataset_stats into DiffusionPolicy at init time.

We provide:
  mean/std/min/max for 10D vectors
and a simple (0.5, 0.5) mean/std for images in [0,1] (common safe default).
"""

from __future__ import annotations

import json
from typing import Dict, List

import h5py
import numpy as np
import torch

from .pose10 import pose10_from_obs


def _pick_env_keys(env_obs_grp) -> tuple[str, str, str] | None:
    """
    Best-effort selection of (eef_pos_key, eef_quat_key, gripper_key).
    Matches the dataset heuristics.
    """
    keys = list(env_obs_grp.keys())

    def pick_pos():
        for k in ["robot0_eef_pos", "eef_pos"]:
            if k in keys and env_obs_grp[k].shape[-1] == 3:
                return k
        for k in keys:
            if "eef_pos" in k and env_obs_grp[k].shape[-1] == 3:
                return k
        return None

    def pick_quat():
        for k in ["robot0_eef_quat", "eef_quat"]:
            if k in keys and env_obs_grp[k].shape[-1] == 4:
                return k
        for k in keys:
            if "eef_quat" in k and env_obs_grp[k].shape[-1] == 4:
                return k
        return None

    def pick_grip():
        for k in ["robot0_gripper_qpos", "robot0_gripper_pos", "robot0_gripper_state",
                  "gripper_qpos", "gripper_pos", "gripper_state"]:
            if k in keys:
                return k
        for k in keys:
            if "gripper" in k:
                return k
        return None

    pos_k, quat_k, grip_k = pick_pos(), pick_quat(), pick_grip()
    if pos_k is None or quat_k is None or grip_k is None:
        return None
    return pos_k, quat_k, grip_k


def _reduce_gripper(grip_raw: np.ndarray) -> np.ndarray:
    grip_raw = np.asarray(grip_raw)
    if grip_raw.ndim == 1:
        return grip_raw.astype(np.float64)
    return np.mean(grip_raw.astype(np.float64), axis=-1)


def estimate_pose10_stats(
    hdf5_path: str,
    max_frames: int = 200_000,
) -> Dict[str, torch.Tensor]:
    """
    Single-pass Welford stats for absolute pose10(t) across the dataset.

    We use this same 10D stats object for both:
      - observation.state
      - action
    (LeRobot expects stats per feature name.)

    Returns:
      {"mean","std","min","max"} as float32 torch tensors of shape (10,)
    """
    count = 0
    mean = np.zeros((10,), dtype=np.float64)
    m2 = np.zeros((10,), dtype=np.float64)
    vmin = np.full((10,), np.inf, dtype=np.float64)
    vmax = np.full((10,), -np.inf, dtype=np.float64)

    with h5py.File(hdf5_path, "r") as f:
        data = f["data"]
        demo_names = sorted([k for k in data.keys() if k.startswith("demo")])

        for dn in demo_names:
            demo = data[dn]
            if "obs_env" not in demo:
                continue
            env_obs = demo["obs_env"]

            keys = _pick_env_keys(env_obs)
            if keys is None:
                continue
            pos_k, quat_k, grip_k = keys

            pos = np.asarray(env_obs[pos_k], dtype=np.float64)    # (T,3)
            quat = np.asarray(env_obs[quat_k], dtype=np.float64)  # (T,4)
            grip = _reduce_gripper(np.asarray(env_obs[grip_k]))   # (T,)

            T = int(pos.shape[0])
            for t in range(T):
                x = pose10_from_obs(pos[t], quat[t], float(grip[t]))  # (10,)

                count += 1
                delta = x - mean
                mean += delta / count
                m2 += delta * (x - mean)

                vmin = np.minimum(vmin, x)
                vmax = np.maximum(vmax, x)

                if count >= max_frames:
                    break
            if count >= max_frames:
                break

    if count < 2:
        raise RuntimeError("Not enough frames to estimate stats.")

    var = m2 / (count - 1)
    std = np.sqrt(np.maximum(var, 1e-12))

    return {
        "mean": torch.tensor(mean, dtype=torch.float32),
        "std": torch.tensor(std, dtype=torch.float32),
        "min": torch.tensor(vmin, dtype=torch.float32),
        "max": torch.tensor(vmax, dtype=torch.float32),
    }


def build_dataset_stats_for_lerobot(vec_stats_10d: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Build a LeRobot-compatible dataset_stats dict.

    LeRobot's configs define features with types (STATE/VISUAL/ACTION).
    LeRobot diffusion example passes dataset_metadata.stats into DiffusionPolicy.

    For images:
      - Our dataset yields float images in [0,1].
      - A simple and common normalization is mean=0.5, std=0.5 (per-channel),
        mapping roughly to [-1,1] after normalization.
    """
    img_mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).view(3, 1, 1)
    img_std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).view(3, 1, 1)
    img_min = torch.zeros((3, 1, 1), dtype=torch.float32)
    img_max = torch.ones((3, 1, 1), dtype=torch.float32)

    return {
        "observation.image": {"mean": img_mean, "std": img_std, "min": img_min, "max": img_max},
        "observation.state": dict(vec_stats_10d),
        "action": dict(vec_stats_10d),
    }


def load_camera_names(hdf5_path: str) -> List[str]:
    """
    Read camera_names JSON from the first demo's attributes.
    """
    with h5py.File(hdf5_path, "r") as f:
        data = f["data"]
        demo_names = sorted([k for k in data.keys() if k.startswith("demo")])
        if not demo_names:
            raise RuntimeError("No demos found.")
        cam_json = data[demo_names[0]].attrs.get("camera_names", None)
        if cam_json is None:
            raise RuntimeError("Missing demo attrs['camera_names']")
        return list(json.loads(cam_json))
