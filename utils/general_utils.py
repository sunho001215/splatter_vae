import torch
import numpy as np
import cv2


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