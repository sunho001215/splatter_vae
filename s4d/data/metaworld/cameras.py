"""The fixed Meta-World camera rig: 6 training cameras and 4 held-out evaluation cameras.

Cameras orbit the look-at point ``LOOKAT`` at distance ``RADIUS``; ``theta`` is the
downward tilt (MuJoCo elevation = -theta) and ``phi`` the azimuth. Extrinsics are
returned in the OpenCV convention (x right, y down, z forward).
"""

from __future__ import annotations

import math

import numpy as np

LOOKAT = (0.0, 0.6, 0.0)
RADIUS = 1.0
FOVY_DEG = 45.0
TRAIN_CAMERAS = ((45.0, 0.0), (45.0, 180.0), (60.0, -30.0), (60.0, 210.0), (30.0, 30.0), (30.0, 150.0))
EVAL_CAMERAS = ((40.0, 60.0), (40.0, 120.0), (55.0, -60.0), (55.0, 240.0))

_GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])


def orbit_c2w_opengl(lookat, distance: float, azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """Camera-to-world transform of MuJoCo's free orbit camera (OpenGL frame: -z forward, +y up)."""
    lookat = np.asarray(lookat, dtype=np.float64)
    az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
    forward = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    up = np.array([-math.sin(el) * math.cos(az), -math.sin(el) * math.sin(az), math.cos(el)])
    z_cam = -forward
    x_cam = np.cross(up, z_cam)
    x_cam /= np.linalg.norm(x_cam)
    y_cam = np.cross(z_cam, x_cam)
    T = np.eye(4)
    T[:3, :3] = np.stack([x_cam, y_cam, z_cam], axis=1)
    T[:3, 3] = lookat - distance * forward
    return T


def opengl_to_opencv_c2w(c2w_gl: np.ndarray) -> np.ndarray:
    return c2w_gl @ _GL_TO_CV


def intrinsics_from_fovy(fovy_deg: float, height: int, width: int) -> np.ndarray:
    """Pinhole K in continuous pixel coordinates (principal point at the image centre)."""
    fy = 0.5 * height / math.tan(0.5 * math.radians(fovy_deg))
    return np.array([[fy, 0.0, 0.5 * width], [0.0, fy, 0.5 * height], [0.0, 0.0, 1.0]])


def camera_rig(height: int, width: int) -> dict:
    """All 10 cameras: names, K, OpenCV c2w/w2c, train flags, azimuth/elevation."""
    specs = [(f"train{i}", th, ph, True) for i, (th, ph) in enumerate(TRAIN_CAMERAS)]
    specs += [(f"eval{i}", th, ph, False) for i, (th, ph) in enumerate(EVAL_CAMERAS)]
    names, K, c2w, is_train, az, el = [], [], [], [], [], []
    for name, theta, phi, train in specs:
        T = opengl_to_opencv_c2w(orbit_c2w_opengl(LOOKAT, RADIUS, phi, -theta))
        names.append(name)
        K.append(intrinsics_from_fovy(FOVY_DEG, height, width))
        c2w.append(T)
        is_train.append(train)
        az.append(phi)
        el.append(-theta)
    c2w = np.stack(c2w)
    return {
        "names": names,
        "K": np.stack(K).astype(np.float32),
        "c2w": c2w.astype(np.float32),
        "w2c": np.linalg.inv(c2w).astype(np.float32),
        "is_train": np.asarray(is_train),
        "azimuth": np.asarray(az, dtype=np.float32),
        "elevation": np.asarray(el, dtype=np.float32),
    }


# Near-view held-out sets (review item 1a). Every camera perturbs one training camera by azimuth, elevation and radius
# offsets and then shifts its centre sideways with the camera re-aimed at LOOKAT, as the RL lateral trajectory does.
# "traj" uses the RL trajectory ranges, "near" half of them. Fixed seeds: the sets are part of the protocol.
HELDOUT_RANGES = {"azimuth_deg": 10.0, "elevation_deg": 6.0, "radius_frac": 0.05, "lateral_m": 0.12}
HELDOUT_SETS = {"near": (0.5, 101), "traj": (1.0, 102)}  # name -> (range scale, seed)
HELDOUT_PER_TRAIN_CAMERA = 2


def perturbed_orbit(
    theta: float, phi: float, d_azimuth: float, d_elevation: float, d_radius: float, lateral: float
) -> tuple[float, float, float]:
    """(azimuth, elevation, distance) of the free camera that perturbs rig camera (theta, phi) and looks at LOOKAT."""
    az, el = math.radians(phi + d_azimuth), math.radians(-theta + d_elevation)
    forward = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    position = np.asarray(LOOKAT) - RADIUS * (1.0 + d_radius) * forward
    right = np.cross(forward, [0.0, 0.0, 1.0])
    position = position + lateral * right / np.linalg.norm(right)
    relative = np.asarray(LOOKAT) - position
    distance = float(np.linalg.norm(relative))
    return (
        math.degrees(math.atan2(relative[1], relative[0])),
        math.degrees(math.asin(np.clip(relative[2] / distance, -1.0, 1.0))),
        distance,
    )


def sample_offsets(rng: np.random.Generator, scale: float = 1.0) -> dict[str, float]:
    return {k: float(rng.uniform(-scale * r, scale * r)) for k, r in HELDOUT_RANGES.items()}


def offsets_to_orbit(base: int, offsets: dict[str, float]) -> tuple[float, float, float]:
    theta, phi = TRAIN_CAMERAS[base]
    return perturbed_orbit(
        theta, phi, offsets["azimuth_deg"], offsets["elevation_deg"], offsets["radius_frac"], offsets["lateral_m"]
    )


def heldout_camera_set(name: str) -> list[dict]:
    """The fixed cameras of held-out set ``name``: base training camera, offsets and the resulting free-camera pose."""
    scale, seed = HELDOUT_SETS[name]
    rng = np.random.default_rng(seed)
    cameras = []
    for base in range(len(TRAIN_CAMERAS)):
        for j in range(HELDOUT_PER_TRAIN_CAMERA):
            offsets = sample_offsets(rng, scale)
            azimuth, elevation, distance = offsets_to_orbit(base, offsets)
            cameras.append(
                {"name": f"{name}{base}_{j}", "set": name, "base": base, **offsets,
                 "azimuth": azimuth, "elevation": elevation, "distance": distance}
            )
    return cameras


def orbit_rig(poses, height: int, width: int) -> dict:
    """K and OpenCV c2w/w2c for free cameras given as (azimuth, elevation, distance) around LOOKAT."""
    c2w = np.stack([opengl_to_opencv_c2w(orbit_c2w_opengl(LOOKAT, d, a, e)) for a, e, d in poses])
    K = np.stack([intrinsics_from_fovy(FOVY_DEG, height, width)] * len(c2w))
    return {"K": K.astype(np.float32), "c2w": c2w.astype(np.float32), "w2c": np.linalg.inv(c2w).astype(np.float32)}


def mujoco_free_camera(azimuth_deg: float, elevation_deg: float, lookat=LOOKAT, distance: float = RADIUS):
    import mujoco  # noqa: PLC0415

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = np.asarray(lookat, dtype=np.float64)
    cam.distance = float(distance)
    cam.azimuth = float(azimuth_deg)
    cam.elevation = float(elevation_deg)
    return cam
