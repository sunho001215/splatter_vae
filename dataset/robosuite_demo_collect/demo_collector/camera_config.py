from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


@dataclass(frozen=True)
class CameraDef:
    """
    Minimal MJCF camera definition.
    - name: camera name used everywhere in robosuite (camera_names, render_camera, etc.)
    - pos: world position [x,y,z]
    - quat: MuJoCo quaternion [w,x,y,z] (camera orientation in world frame)
    - fovy: vertical field-of-view in degrees
    """
    name: str
    pos: List[float]
    quat: List[float]
    fovy: float


def _fmt_vec(v: List[float]) -> str:
    return " ".join(f"{x:.6f}" for x in v)


def inject_cameras_into_mjcf_xml(xml_string: str, cameras: List[CameraDef]) -> str:
    """
    Adds <camera .../> elements under <worldbody>.
    If a camera with the same name already exists, we remove it and replace it.
    """
    root = ET.fromstring(xml_string)

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("MJCF has no <worldbody> element; cannot inject cameras.")

    # Remove existing cameras with same names (avoid duplicates)
    existing = list(worldbody.findall("camera"))
    names = {c.name for c in cameras}
    for cam_elem in existing:
        if cam_elem.get("name") in names:
            worldbody.remove(cam_elem)

    # Add new cameras
    for c in cameras:
        cam_elem = ET.Element("camera")
        cam_elem.set("name", c.name)
        cam_elem.set("pos", _fmt_vec(c.pos))
        cam_elem.set("quat", _fmt_vec(c.quat))   # MuJoCo expects w x y z
        cam_elem.set("fovy", f"{c.fovy:.6f}")
        worldbody.append(cam_elem)

    return ET.tostring(root, encoding="unicode")


def apply_custom_cameras_to_env(env, cameras: List[CameraDef]) -> None:
    """
    Grabs the current MJCF XML from the env, injects cameras, and reloads the env.

    Uses robosuite's supported API:
      MujocoEnv.reset_from_xml_string(xml_string)
    """
    if hasattr(env.model, "get_xml"):
        xml_string = env.model.get_xml()
    else:
        xml_string = str(env.model)

    new_xml = inject_cameras_into_mjcf_xml(xml_string, cameras)
    env.reset_from_xml_string(new_xml)

    # Keep env.camera_names consistent (some robosuite code reads this)
    try:
        env.camera_names = [c.name for c in cameras]
    except Exception:
        pass


# ------------------------- spherical camera generation -------------------------

def _normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        raise ValueError(f"Cannot normalize near-zero vector: {v}")
    return v / n


def _mat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """
    Convert 3x3 rotation matrix to quaternion in (w, x, y, z).

    Assumes R is a proper rotation matrix.
    """
    m = R
    tr = float(m[0, 0] + m[1, 1] + m[2, 2])

    if tr > 0.0:
        S = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (m[2, 1] - m[1, 2]) / S
        y = (m[0, 2] - m[2, 0]) / S
        z = (m[1, 0] - m[0, 1]) / S
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        S = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / S
        x = 0.25 * S
        y = (m[0, 1] + m[1, 0]) / S
        z = (m[0, 2] + m[2, 0]) / S
    elif m[1, 1] > m[2, 2]:
        S = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / S
        x = (m[0, 1] + m[1, 0]) / S
        y = 0.25 * S
        z = (m[1, 2] + m[2, 1]) / S
    else:
        S = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / S
        x = (m[0, 2] + m[2, 0]) / S
        y = (m[1, 2] + m[2, 1]) / S
        z = 0.25 * S

    q = np.array([w, x, y, z], dtype=np.float64)
    q = q / np.linalg.norm(q)
    return q


def _look_at_quat_wxyz(
    cam_pos: Sequence[float],
    lookat: Sequence[float],
    up: Sequence[float] = (0.0, 0.0, 1.0),
) -> np.ndarray:
    """
    Build a camera orientation so that the camera looks at `lookat`.

    MuJoCo camera convention:
      - camera looks towards the negative Z axis of the camera frame
      - +X is right in image, +Y is up in image

    We construct world_R_cam where columns are the camera frame axes expressed in world:
      z_cam_world = normalize(cam_pos - lookat)      (camera +Z points "backward")
      x_cam_world = normalize(cross(up_world, z_cam_world))  (camera +X = image right)
      y_cam_world = cross(z_cam_world, x_cam_world)          (camera +Y = image up)
    """
    cam_pos = np.asarray(cam_pos, dtype=np.float64)
    lookat = np.asarray(lookat, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    z_cam = _normalize(cam_pos - lookat)

    x_cam = np.cross(up, z_cam)
    if np.linalg.norm(x_cam) < 1e-6:
        # Degenerate when up is parallel to viewing direction; pick another up
        up2 = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        x_cam = np.cross(up2, z_cam)
    x_cam = _normalize(x_cam)

    y_cam = np.cross(z_cam, x_cam)
    y_cam = _normalize(y_cam)

    R = np.stack([x_cam, y_cam, z_cam], axis=1)  # world_R_cam
    return _mat_to_quat_wxyz(R)


def make_spherical_cameras(
    r: float,
    theta_list: Sequence[float],  # size=2
    phi_list: Sequence[float],    # size=3
    *,
    lookat: Sequence[float],
    up: Sequence[float] = (0.0, 0.0, 1.0),
    degrees: bool = True,
    fovy: float = 45.0,
    name_prefix: str = "cam",
) -> List[CameraDef]:
    """
    Create 6 cameras from spherical coordinates around a `lookat` center.

    Convention used here (common "azimuth + elevation"):
      - phi   = azimuth around +Z (0 along +X, +90 along +Y)
      - theta = elevation from XY plane (0 on plane, +90 straight up)

    Position (relative to lookat):
      x = r * cos(theta) * cos(phi)
      y = r * cos(theta) * sin(phi)
      z = r * sin(theta)

    Orientation:
      camera is rotated to look at `lookat`, respecting MuJoCo camera convention
      (looks along -Z of camera frame).
    """
    theta_list = list(theta_list)
    phi_list = list(phi_list)

    if len(theta_list) != 2:
        raise ValueError(f"Expected theta_list size=2, got {len(theta_list)}")
    if len(phi_list) != 3:
        raise ValueError(f"Expected phi_list size=3, got {len(phi_list)}")

    lookat_np = np.asarray(lookat, dtype=np.float64)

    cams: List[CameraDef] = []
    idx = 0
    for theta in theta_list:
        for phi in phi_list:
            th = float(theta)
            ph = float(phi)
            if degrees:
                th = math.radians(th)
                ph = math.radians(ph)

            x = float(r) * math.cos(th) * math.cos(ph)
            y = float(r) * math.cos(th) * math.sin(ph)
            z = float(r) * math.sin(th)

            pos = lookat_np + np.array([x, y, z], dtype=np.float64)
            quat = _look_at_quat_wxyz(pos, lookat_np, up=up)

            cams.append(
                CameraDef(
                    name=f"{name_prefix}{idx}",
                    pos=pos.tolist(),
                    quat=quat.tolist(),   # [w, x, y, z]
                    fovy=float(fovy),
                )
            )
            idx += 1

    if len(cams) != 6:
        raise RuntimeError(f"Expected 6 cameras, got {len(cams)}")

    return cams
