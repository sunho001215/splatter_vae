"""Occlusion / free-space loss on Gaussian centres and the Gaussian-usage diagnostics (directive item 1).

Every centre is projected into the training cameras and compared with the GT depth at its nearest pixel. A camera
counts for a centre when the centre is in front of the near plane, inside the image and lands on valid depth
(0 < D < far). Depths are camera z in metres, like the rendered expected depth.
"""

from __future__ import annotations

import torch

from s4d.geometry import project
from s4d.model.render import render_blending_weights

USED_WEIGHT = 1e-3
OPAQUE = 0.3  # the CD-centers opacity threshold (s4d.diag.heldout.CENTER_OPACITY)


def sample_depth(
    xyz: torch.Tensor, depth: torch.Tensor, w2c: torch.Tensor, K: torch.Tensor, near: float, far: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """xyz (B,S,N,3) centres per state, depth (B,S,V,1,H,W) GT depth per state and camera, cameras (B,V,…).

    Returns centre depth z, GT depth D at the nearest pixel and the validity mask, each (B,S,V,N).
    """
    B, S, N, _ = xyz.shape
    V, H, W = depth.shape[2], depth.shape[-2], depth.shape[-1]
    pts = xyz.float()[:, :, None].expand(B, S, V, N, 3)
    uv, z = project(pts, K.float()[:, None].expand(B, S, V, 3, 3), w2c.float()[:, None].expand(B, S, V, 4, 4))
    uv = torch.nan_to_num(uv.detach(), nan=-1.0, posinf=-1.0, neginf=-1.0).clamp(-1.0, float(max(H, W)) + 1.0)
    u, v = torch.floor(uv[..., 0]).long(), torch.floor(uv[..., 1]).long()  # pixel centres sit at +0.5
    inside = (z.detach() > near) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    index = (v.clamp(0, H - 1) * W + u.clamp(0, W - 1)).view(B, S, V, N)
    D = torch.gather(depth.float().reshape(B, S, V, H * W), 3, index)
    valid = inside & (D > 0) & (D < far)
    return z, D, valid


def occlusion_loss(
    xyz_seq: torch.Tensor,
    depth: torch.Tensor,
    w2c: torch.Tensor,
    K: torch.Tensor,
    near: float,
    far: float,
    margin: float,
) -> torch.Tensor:
    """Per-state loss (S,), averaged over Gaussians; gradient reaches the centres only.

    Behind: min over the counted cameras of relu(z - D - m), non-zero only for a centre that is behind the observed
    surface by more than m in every camera that sees its pixel. Front: relu(D - z - m) averaged over the cameras in
    which the centre lies in observed free space (in front of the surface by more than m).
    """
    z, D, valid = sample_depth(xyz_seq, depth, w2c, K, near, far)
    behind = torch.where(valid, torch.relu(z - D - margin), torch.full_like(z, float("inf"))).amin(dim=2)
    behind = torch.where(torch.isfinite(behind), behind, torch.zeros_like(behind))
    front = torch.relu(D - z - margin) * valid.float()
    free = (front > 0).float().sum(dim=2)
    front = front.sum(dim=2) / free.clamp_min(1.0)
    return (behind + front).mean(dim=(0, 2))


def usage_diagnostics(
    gs, depth0: torch.Tensor, w2c: torch.Tensor, K: torch.Tensor, near: float, far: float, margin: float
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Diagnostics at t0 for every sample: utilisation (fraction of all Gaussians with blending weight > 1e-3 in at
    least one training camera), hidden fraction (opaque Gaussians behind the surface by > m in every counted camera)
    and floater fraction (opaque Gaussians in front of the surface by > m in at least one camera). depth0 (B,V,1,H,W).

    Returns per-sample metrics (B,) and the utilisation mask (B,N).
    """
    H, W = depth0.shape[-2:]
    xyz = gs.xyz.detach().float()
    used = (render_blending_weights(gs, xyz, w2c, K, H, W, near, far) > USED_WEIGHT).any(dim=1)
    z, D, valid = sample_depth(xyz[:, None], depth0[:, None], w2c, K, near, far)
    z, D, valid = z[:, 0], D[:, 0], valid[:, 0]  # (B,V,N)
    hidden = valid.any(dim=1) & ((z - D > margin) | ~valid).all(dim=1)
    floater = (valid & (D - z > margin)).any(dim=1)
    opaque = gs.opacity.detach() > OPAQUE
    n_opaque = opaque.float().sum(dim=1)
    nan = torch.full_like(n_opaque, float("nan"))
    metrics = {
        "utilisation": used.float().mean(dim=1),
        "hidden_fraction": torch.where(n_opaque > 0, (hidden & opaque).float().sum(1) / n_opaque.clamp_min(1.0), nan),
        "floater_fraction": torch.where(n_opaque > 0, (floater & opaque).float().sum(1) / n_opaque.clamp_min(1.0), nan),
    }
    return metrics, used
