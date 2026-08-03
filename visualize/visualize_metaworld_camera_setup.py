from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MUJOCO_GL", "egl")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import mujoco_mig_setup  # noqa: F401
except Exception:
    pass

import cv2
import mujoco
import numpy as np

from dataset.metaworld.collector.camera import spherical_camera_pose
from visualize.metaworld_camera_utils import camera_from_pose, camera_poses_from_drq, load_yaml, lookat_up_from_drq, make_metaworld_env, render_pose_with_renderer, unwrap_mujoco


@dataclass(frozen=True)
class ResolvedCameraGeom:
    name: str
    pos: np.ndarray
    forward: np.ndarray
    up: np.ndarray
    fovy_deg: float


CAMERA_COLORS = np.array(
    [
        [0.90, 0.18, 0.22, 1.0],
        [0.10, 0.47, 0.82, 1.0],
        [0.18, 0.63, 0.30, 1.0],
        [0.92, 0.55, 0.13, 1.0],
        [0.55, 0.30, 0.78, 1.0],
        [0.08, 0.62, 0.68, 1.0],
    ],
    dtype=np.float32,
)
CENTER_COLOR = np.array([0.02, 0.02, 0.02, 1.0], dtype=np.float32)
MAT_IDENTITY = np.eye(3, dtype=np.float64).reshape(-1)


def normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros_like(v)
    return v / n


def add_geom(scene, geom_type, *, size, pos, rgba, mat: np.ndarray | None = None) -> None:
    if scene.ngeom >= len(scene.geoms):
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        geom_type,
        np.asarray(size, dtype=np.float64),
        np.asarray(pos, dtype=np.float64),
        MAT_IDENTITY if mat is None else np.asarray(mat, dtype=np.float64).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def add_connector(scene, p0: np.ndarray, p1: np.ndarray, rgba: np.ndarray, *, radius: float, alpha: float | None = None) -> None:
    if scene.ngeom >= len(scene.geoms):
        return
    color = np.asarray(rgba, dtype=np.float32).copy()
    if alpha is not None:
        color[3] = float(alpha)
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        MAT_IDENTITY,
        color,
    )
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        float(radius),
        np.asarray(p0, dtype=np.float64),
        np.asarray(p1, dtype=np.float64),
    )
    geom.rgba[:] = color
    scene.ngeom += 1


def resolved_camera_geoms(renderer: mujoco.Renderer, data: mujoco.MjData, poses: Sequence, lookat: np.ndarray) -> list[ResolvedCameraGeom]:
    out: list[ResolvedCameraGeom] = []
    for pose in poses:
        renderer.update_scene(data, camera=camera_from_pose(pose, lookat))
        scene_cams = renderer.scene.camera
        n = min(2, len(scene_cams))
        pos = np.mean([np.asarray(scene_cams[i].pos, dtype=np.float64) for i in range(n)], axis=0)
        forward = normalize(np.mean([np.asarray(scene_cams[i].forward, dtype=np.float64) for i in range(n)], axis=0))
        up = normalize(np.mean([np.asarray(scene_cams[i].up, dtype=np.float64) for i in range(n)], axis=0))
        out.append(ResolvedCameraGeom(str(pose.name), pos, forward, up, float(pose.fovy_deg)))
    return out


def frustum_corners(cam: ResolvedCameraGeom, scale: float) -> np.ndarray:
    right = normalize(np.cross(cam.forward, cam.up))
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    up = normalize(cam.up)
    center = cam.pos + normalize(cam.forward) * float(scale)
    half_h = np.tan(0.5 * np.deg2rad(float(cam.fovy_deg))) * float(scale)
    half_w = half_h
    return np.stack(
        [
            center - right * half_w - up * half_h,
            center + right * half_w - up * half_h,
            center + right * half_w + up * half_h,
            center - right * half_w + up * half_h,
        ],
        axis=0,
    )


def projection_plane_geometry(cam: ResolvedCameraGeom, scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    forward = normalize(cam.forward)
    up = normalize(cam.up)
    right = normalize(np.cross(forward, up))
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    up = normalize(np.cross(right, forward))
    center = cam.pos + forward * float(scale)
    half_h = np.tan(0.5 * np.deg2rad(float(cam.fovy_deg))) * float(scale)
    half_w = half_h
    corners = np.stack(
        [
            center - right * half_w - up * half_h,
            center + right * half_w - up * half_h,
            center + right * half_w + up * half_h,
            center - right * half_w + up * half_h,
        ],
        axis=0,
    )
    mat = np.stack([right, up, forward], axis=1)
    return center, right, up, corners, mat


def add_projection_plane(scene, cam: ResolvedCameraGeom, color: np.ndarray, *, plane_scale: float, line_radius: float) -> None:
    center, _right, _up, corners, mat = projection_plane_geometry(cam, plane_scale)
    fill = np.asarray(color, dtype=np.float32).copy()
    fill[3] = 0.28
    half_h = np.tan(0.5 * np.deg2rad(float(cam.fovy_deg))) * float(plane_scale)
    half_w = half_h
    add_geom(
        scene,
        mujoco.mjtGeom.mjGEOM_BOX,
        size=[half_w, half_h, max(line_radius * 0.35, 0.0009)],
        pos=center,
        rgba=fill,
        mat=mat,
    )
    for i in range(4):
        add_connector(scene, corners[i], corners[(i + 1) % 4], color, radius=line_radius, alpha=0.98)


def add_camera_marker(scene, cam: ResolvedCameraGeom, color: np.ndarray, *, marker_radius: float, frustum_scale: float, line_radius: float, show_frustum: bool) -> None:
    add_geom(scene, mujoco.mjtGeom.mjGEOM_SPHERE, size=[marker_radius, marker_radius, marker_radius], pos=cam.pos, rgba=color)
    add_projection_plane(scene, cam, color, plane_scale=frustum_scale, line_radius=line_radius)
    corners = frustum_corners(cam, scale=frustum_scale)
    for corner in corners:
        add_connector(scene, cam.pos, corner, color, radius=line_radius, alpha=0.92)
    for i in range(4):
        add_connector(scene, corners[i], corners[(i + 1) % 4], color, radius=line_radius, alpha=0.98)


def add_camera_markers_to_scene(scene, cameras: Sequence[ResolvedCameraGeom], lookat: np.ndarray, *, marker_radius: float, frustum_scale: float, line_radius: float, show_center: bool, show_frustum: bool) -> None:
    if show_center:
        add_geom(
            scene,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[marker_radius * 1.12, marker_radius * 1.12, marker_radius * 1.12],
            pos=np.asarray(lookat, dtype=np.float64),
            rgba=CENTER_COLOR,
        )
    for idx, cam in enumerate(cameras):
        add_camera_marker(
            scene,
            cam,
            CAMERA_COLORS[idx % len(CAMERA_COLORS)],
            marker_radius=marker_radius,
            frustum_scale=frustum_scale,
            line_radius=line_radius,
            show_frustum=show_frustum,
        )


def save_rgb(path: str | Path, img_rgb: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), cv2.cvtColor(np.asarray(img_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError(f"Could not save image to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render MuJoCo-native Meta-World camera setup from a DrQ-v2 camera config.")
    parser.add_argument("--config", required=True, help="DrQ-v2 YAML config. Uses env.cameras, env.lookat, and env.up.")
    parser.add_argument("--out", default="outputs/metaworld_camera_setup.png", help="Output PNG path.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overview_r", type=float, default=1.75)
    parser.add_argument("--overview_theta", type=float, default=-35.0)
    parser.add_argument("--overview_phi", type=float, default=65.0)
    parser.add_argument("--overview_fovy", type=float, default=58.0)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--marker_radius", type=float, default=0.024)
    parser.add_argument("--frustum_scale", type=float, default=0.15)
    parser.add_argument("--line_radius", type=float, default=0.0048)
    parser.add_argument("--show_frustums", action="store_true", help="Accepted for compatibility; camera markers are always square pyramids.")
    parser.add_argument("--hide_center", action="store_true")
    parser.add_argument("--no_legend", action="store_true", help="Accepted for backward compatibility; this renderer never adds text labels.")
    parser.add_argument("--dpi", type=int, default=240, help="Accepted for backward compatibility; MuJoCo rendering uses --height/--width.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    env = make_metaworld_env(cfg, seed=args.seed)
    renderer = None
    try:
        env.reset(seed=args.seed)
        model, data = unwrap_mujoco(env)
        lookat, up = lookat_up_from_drq(cfg)
        poses = camera_poses_from_drq(cfg)
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), int(args.width))
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), int(args.height))
        renderer = mujoco.Renderer(model, height=int(args.height), width=int(args.width))
        cameras = resolved_camera_geoms(renderer, data, poses, lookat)
        overview_pose = spherical_camera_pose(
            name="overview",
            r=float(args.overview_r),
            theta_deg=float(args.overview_theta),
            phi_deg=float(args.overview_phi),
            lookat=lookat,
            up=up,
            fovy_deg=float(args.overview_fovy),
        )
        render_pose_with_renderer(renderer, data, overview_pose, lookat)
        add_camera_markers_to_scene(
            renderer.scene,
            cameras,
            lookat,
            marker_radius=float(args.marker_radius),
            frustum_scale=float(args.frustum_scale),
            line_radius=float(args.line_radius),
            show_center=not bool(args.hide_center),
            show_frustum=bool(args.show_frustums),
        )
        image = np.asarray(renderer.render(), dtype=np.uint8)
    finally:
        if renderer is not None:
            renderer.close()
        env.close()

    save_rgb(args.out, image)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
