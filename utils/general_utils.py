import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import math
import random


def load_image_rgb(path: str) -> np.ndarray:
    """
    Load an image as uint8 RGB (H, W, 3) from PNG file.
    """
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Failed to read image at {path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


def image_to_tensor(img_rgb: np.ndarray) -> torch.Tensor:
    """
    Convert uint8 RGB [0, 255] -> float32 tensor in [-1, 1], shape (3, H, W).
    This matches the normalization used in your training code:
        x ∈ [-1,1]
        gt_01 = (x + 1) * 0.5
    """
    img = img_rgb.astype(np.float32) / 255.0          # [0, 1]
    img = img * 2.0 - 1.0                             # [-1, 1]
    img_tensor = torch.from_numpy(img).permute(2, 0, 1)   # (3,H,W)
    return img_tensor


def invert_4x4(m: np.ndarray) -> np.ndarray:
    """
    Invert a 4x4 camera matrix (OpenGL style, row-major).
    Assumes bottom row is [0,0,0,1].
    """
    R = m[:3, :3]
    t = m[:3, 3:4]
    R_inv = R.T
    t_inv = -R_inv @ t
    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = R_inv
    out[:3, 3] = t_inv[:, 0]
    return out


def quaternion_raw_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    From Pytorch3d
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)


def transform_rotations(
    rotations: torch.Tensor,
    source_cv2wT_quat: torch.Tensor,
) -> torch.Tensor:
    """
    Applies a transform that rotates predicted quaternions from camera
    space to world space, same as GaussianSplatPredictor.transform_rotations.

    Args:
        rotations:          (B, N, 4) quaternion in camera space
        source_cv2wT_quat:  (B, 4)   camera-to-world transform quats
                                      (rotation part, transposed rotation).

    Returns:
        rotations_world: (B, N, 4) quaternion in world space.
    """
    Mq = source_cv2wT_quat.unsqueeze(1).expand_as(rotations)
    rotations_world = quaternion_raw_multiply(Mq, rotations)
    return rotations_world


def flatten_vector(x: torch.Tensor) -> torch.Tensor:
    """
    x: (B, C, H, W) --> (B, N, C), N = H * W

    Matches GaussianSplatPredictor.flatten_vector.
    """
    return x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)


def rot6d_to_matrix(rot_6d: torch.Tensor) -> torch.Tensor:
    """
    Convert 6D rotation representation to a valid 3x3 rotation matrix.

    rot_6d: (..., 6) where first 3 numbers are the first column of the
            rotation matrix, and the next 3 numbers are the second column
            BEFORE orthogonalization.

    Returns:
        R: (..., 3, 3) rotation matrices with orthonormal columns.
    """
    a1 = rot_6d[..., 0:3]   # (..., 3)
    a2 = rot_6d[..., 3:6]   # (..., 3)

    # First basis vector
    b1 = F.normalize(a1, dim=-1)

    # Make second vector orthogonal to first
    proj = (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(a2 - proj, dim=-1)

    # Third basis vector via cross-product
    b3 = torch.cross(b1, b2, dim=-1)

    # Stack as columns to form rotation matrix
    R = torch.stack((b1, b2, b3), dim=-1)  # (..., 3, 3)
    return R


def compute_rotation_translation_errors(
    T_pred: torch.Tensor,
    T_gt: torch.Tensor,
):
    """
    Compute:
      - mean rotation angle error (degrees)
      - mean translation (position) error (L2 norm)

    between predicted and ground-truth T_ij.

    T_pred, T_gt: (B, 4, 4)
    """
    device = T_pred.device
    B = T_pred.shape[0]

    R_pred = T_pred[:, :3, :3]  # (B,3,3)
    R_gt   = T_gt[:, :3, :3]    # (B,3,3)

    # Relative rotation: should be close to identity
    R_err = torch.matmul(R_pred, R_gt.transpose(1, 2))  # (B,3,3)
    trace = R_err[:, 0, 0] + R_err[:, 1, 1] + R_err[:, 2, 2]
    cos_theta = torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0)
    angles = torch.acos(cos_theta)  # radians, in [0, pi]
    angles_deg = angles * (180.0 / math.pi)

    angle_error_deg = angles_deg.mean()

    # Translation error
    t_pred = T_pred[:, :3, 3]
    t_gt   = T_gt[:, :3, 3]
    pos_error = torch.norm(t_pred - t_gt, dim=1)  # (B,)
    pos_error_mean = pos_error.mean()

    return angle_error_deg, pos_error_mean


def compute_intrinsics_errors(
    K_pred: torch.Tensor,
    K_gt: torch.Tensor,
):
    """
    Compute mean absolute error for (fx, fy, cx, cy) between predicted and
    ground-truth intrinsics.

    K_pred, K_gt: (B, 3, 3)
    """
    fx_pred = K_pred[:, 0, 0]
    fy_pred = K_pred[:, 1, 1]
    cx_pred = K_pred[:, 0, 2]
    cy_pred = K_pred[:, 1, 2]

    fx_gt = K_gt[:, 0, 0]
    fy_gt = K_gt[:, 1, 1]
    cx_gt = K_gt[:, 0, 2]
    cy_gt = K_gt[:, 1, 2]

    fx_err = (fx_pred - fx_gt).abs().mean()
    fy_err = (fy_pred - fy_gt).abs().mean()
    cx_err = (cx_pred - cx_gt).abs().mean()
    cy_err = (cy_pred - cy_gt).abs().mean()

    return fx_err, fy_err, cx_err, cy_err


def set_random_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    """
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rot6d_to_rotmat_np(rot6d: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Inverse of rotmat_to_rot6d: build a rotation matrix from 6D representation.

    rot6d: (6,) = [a(3), b(3)] corresponding to first 2 columns, then orthonormalize.
    Returns: (3,3)
    """
    r = np.asarray(rot6d, dtype=np.float64).reshape(6)
    a = r[0:3]
    b = r[3:6]

    # Gram-Schmidt
    a = a / (np.linalg.norm(a) + eps)
    b = b - a * float(np.dot(a, b))
    b = b / (np.linalg.norm(b) + eps)
    c = np.cross(a, b)
    c = c / (np.linalg.norm(c) + eps)

    R = np.stack([a, b, c], axis=1)  # columns
    return R


def mat_to_axisangle_np(R: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to axis-angle vector (3,) = axis * angle.

    We try robosuite's transform_utils first (if available), else fallback.
    """
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)

    try:
        from robosuite.utils import transform_utils as T
        aa = T.mat2axisangle(R)
        return np.asarray(aa, dtype=np.float64).reshape(3)
    except Exception:
        # Minimal fallback (numerically ok for small angles; good enough for evaluation)
        tr = float(np.trace(R))
        cos_theta = (tr - 1.0) * 0.5
        cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
        theta = float(np.arccos(cos_theta))

        if theta < 1e-8:
            return np.zeros(3, dtype=np.float64)

        wx = R[2, 1] - R[1, 2]
        wy = R[0, 2] - R[2, 0]
        wz = R[1, 0] - R[0, 1]
        axis = np.array([wx, wy, wz], dtype=np.float64) / (2.0 * np.sin(theta) + 1e-12)
        return axis * theta


def quat_xyzw_to_rotmat_np(q_xyzw: np.ndarray) -> np.ndarray:
    """Use robosuite transform_utils if possible; fallback to your own math if not."""
    q = np.asarray(q_xyzw, dtype=np.float64).reshape(4)
    try:
        from robosuite.utils import transform_utils as T
        # robosuite quat is commonly xyzw
        return np.asarray(T.quat2mat(q), dtype=np.float64).reshape(3, 3)
    except Exception:
        # Fallback: same convention (x,y,z,w)
        x, y, z, w = q
        n = float(np.linalg.norm(q))
        if n < 1e-12:
            return np.eye(3, dtype=np.float64)
        x, y, z, w = (q / n).tolist()

        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z

        return np.array(
            [
                [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy)],
                [2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
                [2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy)],
            ],
            dtype=np.float64,
        )

