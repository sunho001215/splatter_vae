import torch
import sys
from datetime import datetime
import numpy as np
import random
from einops import rearrange


def init_sh_transform_matrices(device, max_sh_degree: int):
    """
    Same logic as GaussianSplatPredictor.init_sh_transform_matrices for SH1.

    For L = 1, we need a linear map between SH coefficients and a 3D
    vector basis that rotates like a vector under SO(3). The EDM / 3DGS
    code uses fixed matrices v_to_sh_transform, sh_to_v_transform.

    This is equivalent in spirit to the Π-based formulation in the Splatter
    Image paper, where:
        Y_1(ν) = sqrt(3/(4π)) * Π ν, with Π a 3×3 matrix. :contentReference[oaicite:4]{index=4}
    """
    if max_sh_degree <= 0:
        return None, None

    # From the EDM Gaussian predictor code
    v_to_sh_transform = torch.tensor(
        [[0, 0, -1],
         [-1, 0,  0],
         [0, 1,  0]],
        dtype=torch.float32,
        device=device,
    )
    sh_to_v_transform = v_to_sh_transform.transpose(0, 1)
    return sh_to_v_transform.unsqueeze(0), v_to_sh_transform.unsqueeze(0)


def transform_SHs(
    shs: torch.Tensor,
    sh_to_v_transform: torch.Tensor,
    v_to_sh_transform: torch.Tensor,
    source_cameras_to_world: torch.Tensor,
) -> torch.Tensor:
    """
    Transform SH coefficients from the source *camera* frame to the
    global world frame, exactly as in GaussianSplatPredictor.

    Args:
        shs: (B, N, SH_num, 3)    - per-Gaussian SH coefficients in camera frame
              For L = 1, SH_num should be 3 (the "rest" coefficients).
        source_cameras_to_world: (B, 4, 4) camera->world transforms.

    Returns:
        shs_world: (B, N, SH_num, 3)
    """
    # For L = 1, we only support SH_num == 3 for the higher-order part.
    assert shs.shape[2] == 3, "transform_SHs assumes SH_num == 3 (L = 1)."

    b, n, sh_num, _ = shs.shape

    # Merge Gaussians and RGB channels into a single dimension:
    # (B, N, SH_num, 3) -> (B, N*3, SH_num)
    shs_flat = rearrange(shs, "b n sh_num rgb -> b (n rgb) sh_num")

    # Rotation in vector space for SH1:
    #   v_cam -> v_world via R, and we conjugate with sh_to_v / v_to_sh.
    transforms = torch.bmm(
        sh_to_v_transform.expand(b, 3, 3),   # (B,3,3)
        source_cameras_to_world[:, :3, :3],  # (B,3,3)
    )
    transforms = torch.bmm(
        transforms,
        v_to_sh_transform.expand(b, 3, 3),
    )  # (B,3,3)

    # Apply transform to each (N*RGB) vector:
    shs_transformed = torch.bmm(shs_flat, transforms)   # (B, N*3, SH_num)
    shs_transformed = rearrange(
        shs_transformed, "b (n rgb) sh_num -> b n sh_num rgb", rgb=3
    )
    return shs_transformed