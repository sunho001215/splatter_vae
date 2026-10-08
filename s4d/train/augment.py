"""Training-time view augmentations for viewpoint robustness within the six fixed cameras (review item 2).

* ``random_resized_crop`` (2a): encoder input only; render targets stay untouched.
* ``jitter_cameras`` + ``synthesize_views`` (2b): the training cameras' GT depth and colours are fused and splatted into
  cameras jittered within the RL trajectory ranges; holes become masked tokens through ``coverage_visible``.
* the same jittered cameras serve the self-rendered view consistency term (2c, in ``loop.forward_losses``).

Nothing here runs at evaluation or RL time.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from s4d.data.metaworld.cameras import TRAIN_CAMERAS, offsets_to_orbit, orbit_rig, sample_offsets
from s4d.diag.heldout import splat_depth
from s4d.geometry import lift_depth

PATCH_COVERAGE = 0.9  # a synthetic-view patch is a token only if this fraction of its pixels is covered in every frame


def random_resized_crop(
    images: torch.Tensor, extra: torch.Tensor | None, prob: float, scale=(0.8, 1.0), ratio=(0.95, 1.05)
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """images (R,T,C,H,W), extra (R,T,1,H,W) or None: with probability ``1 - prob`` a row is unchanged, otherwise one
    crop (area fraction in ``scale``, aspect ratio in ``ratio``) is resized back to H x W for all T frames of the row."""
    R, T, C, H, W = images.shape
    device = images.device
    crop = torch.rand(R, device=device) < prob
    if not bool(crop.any()):
        return images, extra
    area = torch.empty(R, device=device).uniform_(*scale)
    log_r = torch.empty(R, device=device).uniform_(math.log(ratio[0]), math.log(ratio[1]))
    aspect = torch.exp(log_r)
    w = torch.sqrt(area * aspect).clamp(max=1.0)  # fractions of the full width/height
    h = torch.sqrt(area / aspect).clamp(max=1.0)
    cx = (torch.rand(R, device=device) * 2 - 1) * (1 - w)  # normalised centre so the crop stays inside the image
    cy = (torch.rand(R, device=device) * 2 - 1) * (1 - h)
    theta = torch.zeros(R, 2, 3, device=device)
    theta[:, 0, 0], theta[:, 0, 2], theta[:, 1, 1], theta[:, 1, 2] = w, cx, h, cy
    identity = torch.tensor([[1.0, 0, 0], [0, 1.0, 0]], device=device)
    theta = torch.where(crop[:, None, None], theta, identity)
    grid = F.affine_grid(theta, (R, C, H, W), align_corners=False)
    grid_t = grid[:, None].expand(R, T, H, W, 2).reshape(R * T, H, W, 2)

    def apply(x):
        out = F.grid_sample(x.reshape(R * T, *x.shape[2:]).float(), grid_t, mode="bilinear", align_corners=False)
        return out.view(x.shape).to(x.dtype)

    return apply(images), (apply(extra) if extra is not None else None)


def jitter_cameras(
    batch_size: int, views: int, K: torch.Tensor, height: int, width: int, scale: float = 1.0, seed: int | None = None
) -> dict[str, torch.Tensor]:
    """Cameras perturbed around random training cameras within ``scale`` x the RL trajectory ranges (Meta-World rig).

    K (B,3,3) is reused for every jittered camera; returns K, w2c, c2w (B,views,...)."""
    if seed is None:
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
    rng = np.random.default_rng(seed)
    poses = [offsets_to_orbit(int(rng.integers(len(TRAIN_CAMERAS))), sample_offsets(rng, scale))
             for _ in range(batch_size * views)]
    rig = orbit_rig(poses, height, width)
    like = {"device": K.device, "dtype": torch.float32}
    return {
        "K": K[:, None].expand(batch_size, views, 3, 3).contiguous(),
        "w2c": torch.as_tensor(rig["w2c"], **like).view(batch_size, views, 4, 4),
        "c2w": torch.as_tensor(rig["c2w"], **like).view(batch_size, views, 4, 4),
    }


@torch.no_grad()
def synthesize_views(batch: dict, cams: dict, near: float, far: float) -> dict[str, torch.Tensor]:
    """Fuse the training cameras' GT depth + RGB per (sample, time) and splat it into ``cams``.

    Returns images (B,T,n,3,H,W) in [0,1], depth (B,T,n,1,H,W) and covered (B,T,n,1,H,W)."""
    depth = batch["depth"][:, :, :, 0]  # (B,T,V,H,W)
    B, T, V, H, W = depth.shape
    colors = batch["images"].float().permute(0, 1, 2, 4, 5, 3) / 255.0  # (B,T,V,H,W,3)
    xyz = lift_depth(depth, batch["K"][:, None].expand(B, T, V, 3, 3), batch["c2w"][:, None].expand(B, T, V, 4, 4))
    keep = (depth > 0) & (depth <= far)
    rgb, cov, dep = [], [], []
    for b in range(B):
        for t in range(T):
            r, c, d = splat_depth(xyz[b, t][keep[b, t]], colors[b, t][keep[b, t]], cams["K"][b], cams["w2c"][b], H, W, near)
            rgb.append(r)
            cov.append(c)
            dep.append(d)
    n = cams["K"].shape[1]
    return {
        "images": torch.stack(rgb).view(B, T, n, 3, H, W),
        "covered": torch.stack(cov).view(B, T, n, 1, H, W),
        "depth": torch.stack(dep).view(B, T, n, 1, H, W),
    }


def coverage_visible(covered: torch.Tensor, patch: int, keep: int) -> torch.Tensor:
    """covered (R,T,1,H,W) -> visible-token mask (R,N) with exactly ``keep`` True per row: random among patches covered
    to >= PATCH_COVERAGE in every frame first, holes only when there are fewer such patches than ``keep``."""
    R, T = covered.shape[:2]
    frac = F.avg_pool2d(covered.flatten(0, 1).float(), patch).view(R, T, -1).amin(1)  # (R,N)
    priority = torch.where(frac >= PATCH_COVERAGE, torch.rand_like(frac), torch.rand_like(frac) - 2.0)
    ids = torch.topk(priority, keep, dim=1).indices
    visible = torch.zeros_like(frac, dtype=torch.bool)
    visible.scatter_(1, ids, True)
    return visible
