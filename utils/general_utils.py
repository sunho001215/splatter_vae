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
