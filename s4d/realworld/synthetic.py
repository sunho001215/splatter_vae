"""Synthetic fixtures (scenes, stereo pairs, robots) for tests. Nothing here is real data or model output."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates

from s4d.realworld.geometry import pixel_centers, unproject


def texture_lookup(X: np.ndarray, Y: np.ndarray, seed: int = 0, cells: int = 64, scale: float = 40.0) -> np.ndarray:
    """Deterministic random texture on world (X, Y), bilinearly interpolated, values in [0, 255]."""
    rng = np.random.default_rng(seed)
    grid = rng.uniform(0, 255, size=(cells, cells))
    return map_coordinates(grid, [(Y * scale) % (cells - 1), (X * scale) % (cells - 1)], order=1, mode="wrap")


def layered_scene_render(K: np.ndarray, hw: tuple[int, int], center_x: float, z_back: float = 2.0,
                         z_front: float = 1.0, front_box=(-0.15, 0.2, -0.15, 0.15)) -> tuple[np.ndarray, np.ndarray]:
    """Render a camera at (center_x, 0, 0), identity rotation (rectified rig), looking at two fronto-parallel planes.

    Returns (image float (H,W), depth camera-z (H,W)). The front plane occupies X in [x0, x1], Y in [y0, y1].
    """
    rays = unproject(pixel_centers(*hw), np.ones(hw), K)
    x0, x1, y0, y1 = front_box
    Xf = center_x + rays[..., 0] * z_front
    Yf = rays[..., 1] * z_front
    front = (Xf >= x0) & (Xf <= x1) & (Yf >= y0) & (Yf <= y1)
    depth = np.where(front, z_front, z_back)
    X = center_x + rays[..., 0] * depth
    Y = rays[..., 1] * depth
    image = np.where(front, texture_lookup(X, Y, seed=1), texture_lookup(X, Y, seed=2))
    return image.astype(np.float32), depth.astype(np.float32)


def rectified_pair(hw=(48, 80), f=60.0, baseline=0.1, cx=None, cy=None):
    H, W = hw
    K = np.array([[f, 0, W / 2 if cx is None else cx], [0, f, H / 2 if cy is None else cy], [0, 0, 1.0]])
    left, depth_l = layered_scene_render(K, hw, 0.0)
    right, depth_r = layered_scene_render(K, hw, baseline)
    return left, right, depth_l, depth_r, K


# ---------------------------------------------------------------- synthetic robot (geometric stand-in, NOT real assets)

_PANDA_JOINTS = [  # franka_description panda_arm kinematics (origins), with primitive stand-in link geometry
    ("panda_joint1", "panda_link0", "panda_link1", "0 0 0.333", "0 0 0"),
    ("panda_joint2", "panda_link1", "panda_link2", "0 0 0", "-1.5707963267948966 0 0"),
    ("panda_joint3", "panda_link2", "panda_link3", "0 -0.316 0", "1.5707963267948966 0 0"),
    ("panda_joint4", "panda_link3", "panda_link4", "0.0825 0 0", "1.5707963267948966 0 0"),
    ("panda_joint5", "panda_link4", "panda_link5", "-0.0825 0.384 0", "-1.5707963267948966 0 0"),
    ("panda_joint6", "panda_link5", "panda_link6", "0 0 0", "1.5707963267948966 0 0"),
    ("panda_joint7", "panda_link6", "panda_link7", "0.088 0 0", "1.5707963267948966 0 0"),
]
_LINK_GEOM = {
    "panda_link0": '<cylinder radius="0.07" length="0.14"/>', "panda_link1": '<cylinder radius="0.06" length="0.2"/>',
    "panda_link2": '<cylinder radius="0.06" length="0.12"/>', "panda_link3": '<cylinder radius="0.055" length="0.2"/>',
    "panda_link4": '<cylinder radius="0.055" length="0.12"/>', "panda_link5": '<cylinder radius="0.05" length="0.22"/>',
    "panda_link6": '<cylinder radius="0.045" length="0.1"/>', "panda_link7": '<cylinder radius="0.04" length="0.08"/>',
}
_LINK_GEOM_OFFSET = {"panda_link0": "0 0 0.07", "panda_link1": "0 0 -0.1", "panda_link3": "0 0 -0.1",
                     "panda_link5": "0 0 -0.11"}


def synthetic_panda_urdf() -> str:
    """Panda kinematic chain (official joint origins) with primitive cylinders as link geometry. A test stand-in:
    the geometry is NOT the real Panda meshes."""
    links = "".join(
        f'<link name="{name}"><visual><origin xyz="{_LINK_GEOM_OFFSET.get(name, "0 0 0")}"/>'
        f"<geometry>{geom}</geometry></visual></link>"
        for name, geom in _LINK_GEOM.items()
    ) + '<link name="panda_link8"/>'
    joints = "".join(
        f'<joint name="{j}" type="revolute"><parent link="{p}"/><child link="{c}"/>'
        f'<origin xyz="{xyz}" rpy="{rpy}"/><axis xyz="0 0 1"/><limit lower="-2.9" upper="2.9"/></joint>'
        for j, p, c, xyz, rpy in _PANDA_JOINTS
    )
    joints += ('<joint name="panda_joint8" type="fixed"><parent link="panda_link7"/><child link="panda_link8"/>'
               '<origin xyz="0 0 0.107" rpy="0 0 0"/></joint>')
    return f'<robot name="synthetic_panda">{links}{joints}</robot>'


def synthetic_gripper_urdf() -> str:
    """Two-finger stand-in with a mimic joint (Robotiq-like topology). NOT the Robotiq 2F-85/2F-140 model."""
    return (
        '<robot name="synthetic_gripper">'
        '<link name="gripper_base"><visual><origin xyz="0 0 0.04"/><geometry><box size="0.07 0.13 0.08"/></geometry>'
        "</visual></link>"
        '<link name="left_finger"><visual><origin xyz="0 0 0.03"/><geometry><box size="0.02 0.012 0.06"/></geometry>'
        "</visual></link>"
        '<link name="right_finger"><visual><origin xyz="0 0 0.03"/><geometry><box size="0.02 0.012 0.06"/></geometry>'
        "</visual></link>"
        '<joint name="finger_joint" type="prismatic"><parent link="gripper_base"/><child link="left_finger"/>'
        '<origin xyz="0 0.0425 0.08"/><axis xyz="0 -1 0"/><limit lower="0" upper="0.0425"/></joint>'
        '<joint name="right_finger_joint" type="prismatic"><parent link="gripper_base"/><child link="right_finger"/>'
        '<origin xyz="0 -0.0425 0.08"/><axis xyz="0 1 0"/><mimic joint="finger_joint" multiplier="1" offset="0"/>'
        '<limit lower="0" upper="0.0425"/></joint>'
        "</robot>"
    )


def synthetic_robot_urdf(mount_yaw: float = np.pi / 2) -> str:
    from s4d.realworld.geometry import make_T
    from s4d.realworld.robot import compose_urdf
    from scipy.spatial.transform import Rotation

    T = make_T(Rotation.from_euler("z", mount_yaw).as_matrix(), [0, 0, 0.011])
    return compose_urdf(synthetic_panda_urdf(), synthetic_gripper_urdf(), "panda_link8", "gripper_base", T)


def look_at_T_cam_base(eye, target, up=(0, 0, 1)) -> np.ndarray:
    """OpenCV camera at ``eye`` looking at ``target`` (base frame) -> T_cam_base."""
    eye, target, up = (np.asarray(v, float) for v in (eye, target, up))
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R_base_cam = np.stack([x, y, z], 1)
    T = np.eye(4)
    T[:3, :3] = R_base_cam.T
    T[:3, 3] = -R_base_cam.T @ eye
    return T


def splat_depth(points_cam: np.ndarray, K: np.ndarray, hw, background: float | None = None,
                center_radius_px: float = 0.25) -> np.ndarray:
    """Nearest-z splat of dense surface samples into a depth image; ``background`` fills empty pixels (0 = invalid).

    Only samples within ``center_radius_px`` of a pixel centre are splatted, so each pixel's depth is the surface
    depth at (almost exactly) its centre - an unbiased fixture for depth-alignment tests (needs dense samples)."""
    H, W = hw
    depth = np.full(hw, np.inf)
    z = points_cam[:, 2]
    ok = z > 1e-6
    uv = points_cam[ok] @ np.asarray(K).T
    uv = uv[:, :2] / uv[:, 2:3]
    zz = z[ok]
    j = np.floor(uv[:, 0]).astype(int)
    i = np.floor(uv[:, 1]).astype(int)
    near = np.hypot(uv[:, 0] - (j + 0.5), uv[:, 1] - (i + 0.5)) <= center_radius_px
    m = near & (i >= 0) & (i < H) & (j >= 0) & (j < W)
    np.minimum.at(depth, (i[m], j[m]), zz[m])
    if background is not None:
        depth = np.minimum(depth, background)
    depth[~np.isfinite(depth)] = 0
    return depth.astype(np.float32)


# ---------------------------------------------------------------- ray-cast box scene (static boxes + one moving box)

STATIC_BOXES = (  # (min corner, max corner) in the base frame
    ((-0.6, -1.5, -0.12), (1.6, 1.5, -0.06)),  # floor
    ((0.15, -0.6, -0.06), (1.0, 0.6, 0.0)),  # table top
    ((-0.5, -1.2, -0.06), (-0.43, 1.2, 1.2)),  # back wall (behind the robot base, facing the cameras)
)
OBJECT_HALF = 0.06


def object_center(t: float) -> np.ndarray:
    return np.array([0.55 + 0.08 * np.sin(t), -0.1 + 0.15 * t / 10, OBJECT_HALF])


def _ray_box(o, d, lo, hi):
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (np.asarray(lo) - o) / d
        t2 = (np.asarray(hi) - o) / d
    tmin = np.nanmax(np.minimum(t1, t2), -1)
    tmax = np.nanmin(np.maximum(t1, t2), -1)
    hit = (tmax >= tmin) & (tmax > 1e-6)
    return np.where(hit, np.where(tmin > 1e-6, tmin, tmax), np.inf)


def render_box_scene(T_cam_base, K, hw, t: float):
    """(image float [0,255] (H,W), depth camera-z (H,W), hit points in base (H,W,3), on_object (H,W) bool)."""
    from s4d.realworld.geometry import invert_T

    T_base_cam = invert_T(T_cam_base)
    rays_cam = unproject(pixel_centers(*hw), np.ones(hw), K)
    d = rays_cam @ T_base_cam[:3, :3].T
    o = np.broadcast_to(T_base_cam[:3, 3], d.shape)
    c = object_center(t)
    boxes = list(STATIC_BOXES) + [(c - OBJECT_HALF, c + OBJECT_HALF)]
    ts = np.stack([_ray_box(o, d, lo, hi) for lo, hi in boxes], -1)
    which = np.argmin(ts, -1)
    tt = np.min(ts, -1)
    P = o + d * np.where(np.isfinite(tt), tt, 0)[..., None]
    on_obj = which == len(boxes) - 1
    local = np.where(on_obj[..., None], P - c, P)  # texture moves with the object
    tex = (texture_lookup(local[..., 0], local[..., 1], seed=3) + texture_lookup(local[..., 1], local[..., 2], seed=4)
           + texture_lookup(local[..., 2], local[..., 0], seed=5)) / 3
    tex = np.where(on_obj, 255 - tex, tex)
    depth = np.where(np.isfinite(tt), tt * rays_cam[..., 2], 0.0)  # rays have z = 1 in the camera frame
    return tex.astype(np.float32), depth.astype(np.float32), P, on_obj & np.isfinite(tt)
