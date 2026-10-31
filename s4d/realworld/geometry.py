"""Camera geometry in the repository conventions (OpenCV axes, continuous pixel coordinates, T_a_b naming).

Everything here is numpy/scipy only and validated on synthetic fixtures (tests/test_realworld_geometry.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation

# ---------------------------------------------------------------- rigid transforms


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, float).reshape(3)
    return T


def invert_T(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, float)
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply T_a_b to (..., 3) points expressed in b."""
    T = np.asarray(T, float)
    return np.asarray(points, float) @ T[:3, :3].T + T[:3, 3]


def pose_from_vec(vec: np.ndarray) -> np.ndarray:
    """(rotvec[3], t[3]) -> 4x4."""
    vec = np.asarray(vec, float)
    return make_T(Rotation.from_rotvec(vec[:3]).as_matrix(), vec[3:6])


def vec_from_pose(T: np.ndarray) -> np.ndarray:
    return np.concatenate([Rotation.from_matrix(np.asarray(T)[:3, :3]).as_rotvec(), np.asarray(T)[:3, 3]])


def rotation_angle_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    cos = (np.trace(np.asarray(R_a).T @ np.asarray(R_b)) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


# ---------------------------------------------------------------- pinhole projection (continuous pixel coords)


def project(points_cam: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Camera-frame (N,3) -> pixels (N,2) and camera-z (N,). Points with z <= 0 get NaN pixels."""
    p = np.asarray(points_cam, float)
    z = p[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = (p @ np.asarray(K, float).T)[..., :2] / z[..., None]
    uv[z <= 0] = np.nan
    return uv, z


def unproject(uv: np.ndarray, depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Pixels (N,2) + camera-z (N,) -> camera-frame points (N,3)."""
    K = np.asarray(K, float)
    uv = np.asarray(uv, float)
    z = np.asarray(depth, float)
    x = (uv[..., 0] - K[0, 2] - K[0, 1] * (uv[..., 1] - K[1, 2]) / K[1, 1]) / K[0, 0]
    y = (uv[..., 1] - K[1, 2]) / K[1, 1]
    return np.stack([x * z, y * z, z], axis=-1)


def pixel_centers(height: int, width: int) -> np.ndarray:
    """(H,W,2) continuous coordinates of pixel centres."""
    jj, ii = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    return np.stack([jj, ii], axis=-1)


def depth_to_points(depth: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Camera-frame points (H,W,3) for every pixel centre and the validity mask (depth > 0, finite)."""
    valid = np.isfinite(depth) & (depth > 0)
    pts = unproject(pixel_centers(*depth.shape), np.where(valid, depth, 0.0), K)
    return pts, valid


# ---------------------------------------------------------------- image-space affine transforms that track K


@dataclass(frozen=True)
class PixelAffine:
    """x' = sx * x + tx, y' = sy * y + ty in continuous coordinates (resize, crop, pad, horizontal flip).

    ``apply_K`` gives the intrinsics of the transformed image, ``apply_uv`` maps pixels; composing keeps the
    image/depth/intrinsics alignment exact through any chain of resize/crop/flip operations.
    """

    sx: float = 1.0
    sy: float = 1.0
    tx: float = 0.0
    ty: float = 0.0

    def then(self, other: PixelAffine) -> PixelAffine:
        return PixelAffine(other.sx * self.sx, other.sy * self.sy, other.sx * self.tx + other.tx,
                           other.sy * self.ty + other.ty)

    def matrix(self) -> np.ndarray:
        return np.array([[self.sx, 0, self.tx], [0, self.sy, self.ty], [0, 0, 1.0]])

    def apply_K(self, K: np.ndarray) -> np.ndarray:
        return self.matrix() @ np.asarray(K, float)

    def apply_uv(self, uv: np.ndarray) -> np.ndarray:
        uv = np.asarray(uv, float)
        return np.stack([self.sx * uv[..., 0] + self.tx, self.sy * uv[..., 1] + self.ty], axis=-1)

    def inverse(self) -> PixelAffine:
        return PixelAffine(1 / self.sx, 1 / self.sy, -self.tx / self.sx, -self.ty / self.sy)

    def to_json(self) -> dict:
        return {"sx": self.sx, "sy": self.sy, "tx": self.tx, "ty": self.ty}


def resize_affine(src_hw: tuple[int, int], dst_hw: tuple[int, int]) -> PixelAffine:
    """Full-image resize: corners map to corners (continuous convention)."""
    return PixelAffine(dst_hw[1] / src_hw[1], dst_hw[0] / src_hw[0], 0.0, 0.0)


def crop_affine(top: int, left: int) -> PixelAffine:
    return PixelAffine(1.0, 1.0, -float(left), -float(top))


def hflip_affine(width: int) -> PixelAffine:
    """Horizontal flip about the image centre: x' = W - x. NOT a valid pinhole change of K alone (it mirrors the
    camera); used only to flip-and-swap stereo pairs and flip results back (see s4d.realworld.stereo)."""
    return PixelAffine(-1.0, 1.0, float(width), 0.0)


def warp_affine(image: np.ndarray, affine: PixelAffine, out_hw: tuple[int, int], *, order: int) -> np.ndarray:
    """Resample ``image`` (H,W[,C]) into the transformed frame. order=0 nearest (depth/masks), 1 bilinear (images).

    Each output pixel centre is mapped back through the inverse affine and sampled at the source location, so the
    output exactly realises ``affine`` (outside samples are 0).
    """
    inv = affine.inverse()
    centers = pixel_centers(*out_hw)
    src = inv.apply_uv(centers) - 0.5  # continuous -> array index coordinates
    coords = [src[..., 1], src[..., 0]]
    image = np.asarray(image)
    if image.ndim == 2:
        return map_coordinates(image, coords, order=order, mode="constant", cval=0).astype(image.dtype)
    return np.stack(
        [map_coordinates(image[..., c], coords, order=order, mode="constant", cval=0) for c in range(image.shape[-1])],
        axis=-1,
    ).astype(image.dtype)


def resize_depth_nearest(depth: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour depth resize (never interpolates across depth discontinuities); 0 stays invalid."""
    return warp_affine(depth, resize_affine(depth.shape, out_hw), out_hw, order=0)


# ---------------------------------------------------------------- distortion (OpenCV plumb_bob / rational)


def distort_normalized(xy: np.ndarray, model: str, coeffs) -> np.ndarray:
    xy = np.asarray(xy, float)
    if model == "none":
        return xy.copy()
    c = list(coeffs) + [0.0] * (8 - len(coeffs))
    if model in ("opencv_plumb_bob", "realsense_brown_conrady", "realsense_modified_brown_conrady"):
        k1, k2, p1, p2, k3 = c[:5]
        k4 = k5 = k6 = 0.0
    elif model == "opencv_rational":
        k1, k2, p1, p2, k3, k4, k5, k6 = c[:8]
    else:
        raise NotImplementedError(f"distortion model {model!r} is not implemented; undistort upstream or add it")
    x, y = xy[..., 0], xy[..., 1]
    r2 = x * x + y * y
    radial = (1 + k1 * r2 + k2 * r2**2 + k3 * r2**3) / (1 + k4 * r2 + k5 * r2**2 + k6 * r2**3)
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return np.stack([xd, yd], axis=-1)


def undistort_normalized(xy_d: np.ndarray, model: str, coeffs, iterations: int = 20) -> np.ndarray:
    """Fixed-point inverse of ``distort_normalized`` (as OpenCV's undistortPoints)."""
    if model == "none":
        return np.asarray(xy_d, float).copy()
    xy = np.asarray(xy_d, float).copy()
    for _ in range(iterations):
        xy = xy - (distort_normalized(xy, model, coeffs) - xy_d)
    return xy


# ---------------------------------------------------------------- stereo rectification (Bouguet, as OpenCV)


@dataclass
class Rectification:
    """Rectifying rotations and the shared rectified camera of one stereo rig.

    R1/R2 rotate left/right raw camera frames into the rectified frames (x_rect = R @ x_raw). In the rectified
    frames the right camera sits at (-baseline, 0, 0) relative to the left, so for a point at depth z:
    x_left - x_right = fx * baseline / z + (cx_left - cx_right) (here cx_left == cx_right, doffs = 0).
    """

    R1: np.ndarray
    R2: np.ndarray
    K_rect: np.ndarray  # shared continuous-convention intrinsics of both rectified images
    baseline_m: float
    size_hw: tuple[int, int]

    def to_json(self) -> dict:
        return {"R1": self.R1.tolist(), "R2": self.R2.tolist(), "K_rect": self.K_rect.tolist(),
                "baseline_m": self.baseline_m, "size_hw": list(self.size_hw)}


def stereo_rectify(K1: np.ndarray, K2: np.ndarray, T_right_left: np.ndarray, size_hw: tuple[int, int],
                   K_rect: np.ndarray | None = None) -> Rectification:
    """Bouguet rectification for a horizontal rig (right camera roughly at +x of the left one).

    The relative rotation is split in half between the two cameras, then both are rotated so the baseline is the
    rectified +x axis. ``K_rect`` defaults to the mean focal of the two inputs with the left principal point.
    """
    T = np.asarray(T_right_left, float)
    R, t = T[:3, :3], T[:3, 3]
    om = Rotation.from_matrix(R).as_rotvec()
    r_half = Rotation.from_rotvec(-om / 2).as_matrix()  # applied to the right camera
    l_half = r_half.T  # applied to the left camera
    tt = r_half @ t  # baseline expressed in the half-rotated right frame
    if tt[0] >= 0:
        raise ValueError("right camera is not at +x of the left camera; check T_right_left (maps left -> right)")
    e1 = tt / np.linalg.norm(tt)
    # Rotate so that R2 @ t = (-baseline, 0, 0): the right camera centre sits at +x of the left one.
    target = np.array([-1.0, 0, 0])
    axis = np.cross(e1, target)
    s = np.linalg.norm(axis)
    if s < 1e-12:
        R_align = np.eye(3)
    else:
        R_align = Rotation.from_rotvec(axis / s * np.arctan2(s, e1 @ target)).as_matrix()
    R1 = R_align @ l_half
    R2 = R_align @ r_half
    if K_rect is None:
        f = (K1[0, 0] + K1[1, 1] + K2[0, 0] + K2[1, 1]) / 4
        K_rect = np.array([[f, 0, K1[0, 2]], [0, f, K1[1, 2]], [0, 0, 1.0]])
    t_rect = R2 @ t
    if t_rect[0] > 0:
        raise ValueError("right camera is not at +x of the left camera after rectification; check T_right_left")
    return Rectification(R1=R1, R2=R2, K_rect=np.asarray(K_rect, float), baseline_m=float(np.linalg.norm(t)),
                         size_hw=tuple(size_hw))


def rectify_map(K_raw: np.ndarray, dist_model: str, dist: tuple, R_rect: np.ndarray, K_rect: np.ndarray,
                size_hw: tuple[int, int]) -> np.ndarray:
    """(H,W,2) raw continuous pixel coordinates sampled by each rectified pixel centre (cv2.initUndistortRectifyMap)."""
    centers = pixel_centers(*size_hw)
    rays = unproject(centers, np.ones(size_hw), K_rect)  # rectified camera frame, z = 1
    raw = rays @ np.asarray(R_rect, float)  # R^T applied row-wise: rectified -> raw
    xy = raw[..., :2] / raw[..., 2:3]
    xy_d = distort_normalized(xy, dist_model, dist)
    K = np.asarray(K_raw, float)
    return np.stack([K[0, 0] * xy_d[..., 0] + K[0, 1] * xy_d[..., 1] + K[0, 2], K[1, 1] * xy_d[..., 1] + K[1, 2]], -1)


def remap(image: np.ndarray, coords_xy: np.ndarray, *, order: int) -> np.ndarray:
    src = coords_xy - 0.5
    coords = [src[..., 1], src[..., 0]]
    image = np.asarray(image)
    if image.ndim == 2:
        return map_coordinates(image.astype(float), coords, order=order, mode="constant", cval=0).astype(image.dtype)
    return np.stack([map_coordinates(image[..., c].astype(float), coords, order=order, mode="constant", cval=0)
                     for c in range(image.shape[-1])], -1).astype(image.dtype)


def rectify_points(uv_raw: np.ndarray, K_raw: np.ndarray, dist_model: str, dist: tuple, R_rect: np.ndarray,
                   K_rect: np.ndarray) -> np.ndarray:
    K = np.asarray(K_raw, float)
    xy_d = np.stack([(uv_raw[..., 0] - K[0, 2]) / K[0, 0], (uv_raw[..., 1] - K[1, 2]) / K[1, 1]], -1)
    xy = undistort_normalized(xy_d, dist_model, dist)
    rays = np.concatenate([xy, np.ones_like(xy[..., :1])], -1) @ np.asarray(R_rect, float).T
    uv, _ = project(rays, K_rect)
    return uv


# ---------------------------------------------------------------- depth transfer between cameras


def reproject_depth(depth_src: np.ndarray, K_src: np.ndarray, T_dst_src: np.ndarray, K_dst: np.ndarray,
                    dst_hw: tuple[int, int], splat: int = 1) -> np.ndarray:
    """Forward-warp metric depth from a source camera (e.g. RealSense IR-left) into another camera (e.g. the colour
    sensor) with a z-buffer: each valid source pixel is unprojected, transformed, projected and kept if nearest.

    ``splat`` > 1 writes a (splat x splat) footprint to close holes from magnification. Pixels that receive no sample
    stay 0 (invalid); occluded/disoccluded regions are never filled by interpolation.
    """
    pts, valid = depth_to_points(depth_src, K_src)
    P = transform_points(T_dst_src, pts[valid])
    uv, z = project(P, K_dst)
    H, W = dst_hw
    out = np.full(dst_hw, np.inf)
    ok = np.isfinite(uv).all(1) & (z > 0)
    for dy in range(splat):
        for dx in range(splat):
            off = (splat - 1) / 2
            j = np.floor(uv[ok, 0] + dx - off).astype(int)
            i = np.floor(uv[ok, 1] + dy - off).astype(int)
            m = (i >= 0) & (i < H) & (j >= 0) & (j < W)
            np.minimum.at(out, (i[m], j[m]), z[ok][m])
    out[~np.isfinite(out)] = 0
    return out.astype(np.float32)


def point_mask(points_cam: np.ndarray, K: np.ndarray, hw: tuple[int, int], depth: np.ndarray | None = None,
               tol_m: float = 0.02) -> np.ndarray:
    """Pixels hit by projected points (e.g. calibrated URDF surface samples -> robot mask). With ``depth``, only points
    within ``tol_m`` of the measured surface count (occluded robot parts are not marked)."""
    uv, z = project(points_cam, K)
    H, W = hw
    ok = np.isfinite(uv).all(1) & (z > 0)
    j = np.floor(np.where(ok, uv[:, 0], -1)).astype(int)
    i = np.floor(np.where(ok, uv[:, 1], -1)).astype(int)
    ok &= (i >= 0) & (i < H) & (j >= 0) & (j < W)
    if depth is not None:
        meas = np.zeros(len(z))
        meas[ok] = depth[i[ok], j[ok]]
        ok &= (meas > 0) & (np.abs(meas - z) <= tol_m)
    mask = np.zeros(hw, bool)
    mask[i[ok], j[ok]] = True
    return mask


def undistort_pixels(uv: np.ndarray, K: np.ndarray, model: str, coeffs) -> np.ndarray:
    """Distorted pixels -> pixels of the ideal pinhole camera with the same K (continuous convention)."""
    K = np.asarray(K, float)
    xy = np.stack([(uv[..., 0] - K[0, 2]) / K[0, 0], (uv[..., 1] - K[1, 2]) / K[1, 1]], -1)
    xy = undistort_normalized(xy, model, coeffs)
    return np.stack([xy[..., 0] * K[0, 0] + K[0, 2], xy[..., 1] * K[1, 1] + K[1, 2]], -1)


def distort_pixels(uv: np.ndarray, K: np.ndarray, model: str, coeffs) -> np.ndarray:
    K = np.asarray(K, float)
    xy = np.stack([(uv[..., 0] - K[0, 2]) / K[0, 0], (uv[..., 1] - K[1, 2]) / K[1, 1]], -1)
    xy = distort_normalized(xy, model, coeffs)
    return np.stack([xy[..., 0] * K[0, 0] + K[0, 2], xy[..., 1] * K[1, 1] + K[1, 2]], -1)
