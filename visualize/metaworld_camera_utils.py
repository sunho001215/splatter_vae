from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

os.environ.setdefault("MUJOCO_GL", "egl")

import h5py
import numpy as np
import yaml

from dataset.metaworld.collector.camera import (
    CameraPose,
    extrinsics_world_T_cam,
    intrinsics_from_fovy,
    look_at_quat_wxyz,
    spherical_camera_pose,
)

GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def sort_demo_keys(keys: Iterable[str]) -> List[str]:
    def key_fn(value: str) -> int:
        try:
            return int(str(value).replace("demo", ""))
        except Exception:
            return 10**12

    return sorted([str(k) for k in keys], key=key_fn)


def choose_demo_key(dataset_path: str | Path, requested: str | None = None) -> str:
    with h5py.File(dataset_path, "r") as f:
        keys = sort_demo_keys(f["data"].keys())
    if not keys:
        raise ValueError(f"No demos found in {dataset_path}.")
    if requested is None:
        return keys[0]
    if requested not in keys:
        raise ValueError(f"Demo {requested!r} not found. Available examples: {keys[:5]}")
    return requested


def env_config(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    if "env" not in cfg:
        raise KeyError("Expected a DrQ-v2 config with an `env` section.")
    return dict(cfg["env"])


def camera_configs_from_drq(cfg: Mapping[str, Any], camera_names: Sequence[str] | None = None) -> List[Dict[str, Any]]:
    env_cfg = env_config(cfg)
    cameras = [dict(cam) for cam in env_cfg.get("cameras", [])]
    if not cameras and "camera" in env_cfg:
        cameras = [dict(env_cfg["camera"])]
    if not cameras:
        raise ValueError("DrQ-v2 env config does not define `cameras` or `camera`.")
    if camera_names is not None:
        wanted = set(camera_names)
        cameras = [cam for cam in cameras if str(cam.get("name", "")) in wanted]
        missing = [name for name in camera_names if name not in {str(cam.get("name", "")) for cam in cameras}]
        if missing:
            raise ValueError(f"Requested cameras {missing} not found in DrQ-v2 config.")
    return cameras


def lookat_up_from_drq(cfg: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    env_cfg = env_config(cfg)
    lookat = np.asarray(env_cfg.get("lookat", [0.0, 0.6, 0.0]), dtype=np.float64)
    up = np.asarray(env_cfg.get("up", [0.0, 0.0, 1.0]), dtype=np.float64)
    return lookat, up


def camera_poses_from_drq(cfg: Mapping[str, Any], camera_names: Sequence[str] | None = None) -> List[CameraPose]:
    lookat, up = lookat_up_from_drq(cfg)
    poses = []
    for cam in camera_configs_from_drq(cfg, camera_names):
        poses.append(
            spherical_camera_pose(
                name=str(cam.get("name", f"cam{len(poses)}")),
                r=float(cam["r"]),
                theta_deg=float(cam["theta"]),
                phi_deg=float(cam["phi"]),
                lookat=lookat,
                up=up,
                fovy_deg=float(cam.get("fovy", 45.0)),
            )
        )
    return poses


def invert_4x4(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    r = m[:3, :3]
    t = m[:3, 3:4]
    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = r.T
    out[:3, 3:4] = -(r.T @ t)
    return out


def pose_mats(pose: CameraPose, height: int, width: int) -> Dict[str, np.ndarray]:
    k = intrinsics_from_fovy(pose.fovy_deg, int(height), int(width)).astype(np.float32)
    c2w_gl = extrinsics_world_T_cam(pose.pos, pose.quat_wxyz).astype(np.float32)
    w2c_gl = invert_4x4(c2w_gl)
    w2c_cv = GL_TO_CV @ w2c_gl
    c2w_cv = invert_4x4(w2c_cv)
    return {"K": k, "c2w_gl": c2w_gl, "w2c_gl": w2c_gl, "c2w": c2w_cv, "w2c": w2c_cv}


def dataset_camera_mats(demo: h5py.Group, cam_names: Sequence[str], *, opencv: bool = True) -> Dict[str, Dict[str, np.ndarray]]:
    intr = np.asarray(demo["camera_params"]["intrinsics"], dtype=np.float32)
    world_t_cam = np.asarray(demo["camera_params"]["extrinsics_world_T_cam"], dtype=np.float32)
    all_names = json.loads(demo.attrs["camera_names"])
    mats: Dict[str, Dict[str, np.ndarray]] = {}
    for cam in cam_names:
        if cam not in all_names:
            raise ValueError(f"Camera {cam!r} not found in demo cameras {all_names}.")
        idx = all_names.index(cam)
        k = intr[idx]
        c2w_gl = world_t_cam[idx]
        w2c_gl = invert_4x4(c2w_gl)
        if opencv:
            w2c = GL_TO_CV @ w2c_gl
            c2w = invert_4x4(w2c)
        else:
            w2c = w2c_gl
            c2w = c2w_gl
        mats[cam] = {"K": k.astype(np.float32), "c2w": c2w.astype(np.float32), "w2c": w2c.astype(np.float32)}
    return mats


def read_rgb(demo: h5py.Group, cam: str, timestep: int) -> np.ndarray:
    return np.asarray(demo["obs"][f"{cam}_rgb"][int(timestep)], dtype=np.uint8)


def read_state(demo: h5py.Group, timestep: int) -> np.ndarray | None:
    if "states" not in demo:
        return None
    return np.asarray(demo["states"][int(timestep)], dtype=np.float64)


def image_to_tensor(img_rgb: np.ndarray):
    import torch

    img = np.asarray(img_rgb, dtype=np.float32) / 255.0
    return torch.from_numpy(img * 2.0 - 1.0).permute(2, 0, 1)


def tensor_to_uint8_image(x) -> np.ndarray:
    import torch

    if torch.is_tensor(x):
        y = x.detach().cpu().float()
        if y.ndim == 4:
            y = y[0]
        if y.shape[0] in (1, 3):
            y = y.permute(1, 2, 0)
        if y.min() < 0.0:
            y = (y + 1.0) * 0.5
        arr = y.clamp(0.0, 1.0).numpy()
    else:
        arr = np.asarray(x, dtype=np.float32)
        if arr.max() > 1.5:
            arr = arr / 255.0
    return np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)


def normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    return v * 0.0 if n < eps else v / n


def reciprocating_lateral_offsets(num_frames: int, amplitude: float) -> np.ndarray:
    """Return a center -> right -> center -> left -> center lateral path."""
    n = max(1, int(num_frames))
    amp = float(amplitude)
    if n == 1 or abs(amp) < 1e-12:
        return np.zeros(n, dtype=np.float64)

    phase = np.linspace(0.0, 4.0, n, endpoint=True, dtype=np.float64)
    offsets = np.empty(n, dtype=np.float64)
    first = phase < 1.0
    second = (phase >= 1.0) & (phase < 3.0)
    third = phase >= 3.0
    offsets[first] = phase[first]
    offsets[second] = 2.0 - phase[second]
    offsets[third] = phase[third] - 4.0
    return amp * offsets


def camera_from_pose(pose: CameraPose, lookat: np.ndarray):
    import mujoco

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = np.asarray(lookat, dtype=np.float64)
    rel = np.asarray(lookat, dtype=np.float64) - np.asarray(pose.pos, dtype=np.float64)
    cam.distance = float(np.linalg.norm(rel))
    if pose.azimuth_deg is not None and pose.elevation_deg is not None:
        cam.azimuth = float(pose.azimuth_deg)
        cam.elevation = float(pose.elevation_deg)
    else:
        cam.azimuth = float(np.degrees(np.arctan2(rel[1], rel[0])))
        cam.elevation = float(np.degrees(np.arcsin(np.clip(rel[2] / max(cam.distance, 1.0e-12), -1.0, 1.0))))
    return cam


def render_pose_with_renderer(renderer, data, pose: CameraPose, lookat: np.ndarray) -> np.ndarray:
    renderer.update_scene(data, camera=camera_from_pose(pose, lookat))
    return np.asarray(renderer.render(), dtype=np.uint8)


def make_metaworld_env(cfg: Mapping[str, Any], seed: int):
    import gymnasium as gym
    import metaworld  # noqa: F401

    env_cfg = env_config(cfg)
    env_name = str(env_cfg["env_name"])
    gym_env_name = env_name if env_name.endswith("-v3") else f"{env_name}-v3"
    return gym.make(str(env_cfg.get("benchmark_id", "Meta-World/MT1")), env_name=gym_env_name, seed=int(seed))


def unwrap_mujoco(env):
    e = env.unwrapped
    if hasattr(e, "model") and hasattr(e, "data"):
        return e.model, e.data
    if hasattr(e, "sim"):
        return e.sim.model, e.sim.data
    raise RuntimeError("Could not find MuJoCo model/data on env.unwrapped.")


def set_env_state_from_flat(env, state: np.ndarray) -> None:
    import mujoco

    model, data = unwrap_mujoco(env)
    state = np.asarray(state, dtype=np.float64).reshape(-1)
    nq = int(model.nq)
    nv = int(model.nv)
    if state.size < nq + nv:
        raise ValueError(f"Stored state has length {state.size}, but model needs at least nq+nv={nq + nv}.")
    qpos = state[:nq].copy()
    qvel = state[nq : nq + nv].copy()
    base = env.unwrapped
    if hasattr(base, "set_state"):
        base.set_state(qpos, qvel)
    else:
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        mujoco.mj_forward(model, data)


def trajectory_poses(
    cfg: Mapping[str, Any],
    *,
    base_camera: str,
    trajectory: str,
    num_frames: int,
    lateral_amplitude: float = 0.12,
    circular_azimuth_deg: float = 10.0,
    circular_elevation_deg: float = 6.0,
    orbit_center: Sequence[float] | None = None,
    orbit_degrees: float = 360.0,
) -> List[CameraPose]:
    env_cfg = env_config(cfg)
    lookat, up = lookat_up_from_drq(cfg)
    orbit_lookat = np.asarray(orbit_center, dtype=np.float64) if orbit_center is not None else lookat
    cameras = camera_configs_from_drq(cfg)
    by_name = {str(cam.get("name", f"cam{i}")): cam for i, cam in enumerate(cameras)}
    if base_camera not in by_name:
        raise ValueError(f"Base camera {base_camera!r} not in DrQ-v2 cameras {sorted(by_name)}.")
    base = by_name[base_camera]
    base_pose = spherical_camera_pose(
        name=base_camera,
        r=float(base["r"]),
        theta_deg=float(base["theta"]),
        phi_deg=float(base["phi"]),
        lookat=lookat,
        up=up,
        fovy_deg=float(base.get("fovy", 45.0)),
    )

    n = max(1, int(num_frames))
    traj = str(trajectory).lower()
    poses: List[CameraPose] = []
    if traj == "lateral":
        forward = normalize(lookat - base_pose.pos)
        right = normalize(np.cross(forward, up))
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        offsets = reciprocating_lateral_offsets(n, float(lateral_amplitude))
        for i, offset in enumerate(offsets):
            pos = base_pose.pos + offset * right
            poses.append(
                CameraPose(
                    name=f"{base_camera}_lateral_{i:03d}",
                    pos=pos.astype(np.float64),
                    quat_wxyz=look_at_quat_wxyz(pos, lookat, up),
                    fovy_deg=float(base.get("fovy", 45.0)),
                )
            )
        return poses

    if traj == "circular":
        for i in range(n):
            a = 2.0 * math.pi * i / n
            poses.append(
                spherical_camera_pose(
                    name=f"{base_camera}_circular_{i:03d}",
                    r=float(base["r"]),
                    theta_deg=float(base["theta"]) + float(circular_elevation_deg) * math.sin(a),
                    phi_deg=float(base["phi"]) + float(circular_azimuth_deg) * math.cos(a),
                    lookat=lookat,
                    up=up,
                    fovy_deg=float(base.get("fovy", 45.0)),
                )
            )
        return poses

    if traj == "orbit":
        denom = max(1, n - 1)
        for i in range(n):
            frac = i / denom
            poses.append(
                spherical_camera_pose(
                    name=f"{base_camera}_orbit_{i:03d}",
                    r=float(base["r"]),
                    theta_deg=float(base["theta"]),
                    phi_deg=float(base["phi"]) + float(orbit_degrees) * frac,
                    lookat=orbit_lookat,
                    up=up,
                    fovy_deg=float(base.get("fovy", 45.0)),
                )
            )
        return poses

    raise ValueError(f"Unknown trajectory {trajectory!r}; expected lateral, circular, or orbit.")


def labeled_panel(img_rgb: np.ndarray, title: str, subtitle: str | None = None, *, width: int | None = None) -> np.ndarray:
    import cv2

    img = np.asarray(img_rgb, dtype=np.uint8)
    if width is not None and img.shape[1] != width:
        scale = width / img.shape[1]
        img = cv2.resize(img, (width, max(1, int(round(img.shape[0] * scale)))), interpolation=cv2.INTER_AREA)
    top = 36 if subtitle is None else 58
    canvas = np.full((img.shape[0] + top, img.shape[1], 3), 250, dtype=np.uint8)
    canvas[top:] = img
    cv2.putText(canvas, title, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 24, 32), 2, cv2.LINE_AA)
    if subtitle:
        cv2.putText(canvas, subtitle, (10, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (64, 70, 82), 1, cv2.LINE_AA)
    return canvas


def hstack_panels(panels: Sequence[np.ndarray], pad: int = 8, pad_color: int = 244) -> np.ndarray:
    if not panels:
        raise ValueError("Need at least one panel.")
    max_h = max(panel.shape[0] for panel in panels)
    padded = []
    for panel in panels:
        if panel.shape[0] < max_h:
            extra = np.full((max_h - panel.shape[0], panel.shape[1], 3), pad_color, dtype=np.uint8)
            panel = np.concatenate([panel, extra], axis=0)
        padded.append(panel)
    gap = np.full((max_h, int(pad), 3), pad_color, dtype=np.uint8)
    row = padded[0]
    for panel in padded[1:]:
        row = np.concatenate([row, gap, panel], axis=1)
    return row


def write_mp4(path: str | Path, frames: Sequence[np.ndarray], fps: float = 20.0) -> None:
    import cv2

    if not frames:
        raise ValueError("No frames were provided for video writing.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    first = np.asarray(frames[0], dtype=np.uint8)
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}.")
    try:
        for frame in frames:
            frame = np.asarray(frame, dtype=np.uint8)
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
