"""Training-time view augmentations for viewpoint robustness within the six fixed cameras (review item 2).

* ``random_resized_crop`` (2a): encoder input only; render targets stay untouched.
* ``random_shift`` (2026-10-17 attribution variant): DrM-style random shift in place of the crop; encoder input only.
* ``jitter_cameras`` + ``synthesize_views`` (2b): the training cameras' GT depth and colours are fused and splatted into
  cameras jittered within the RL trajectory ranges; holes become masked tokens through ``coverage_visible``.
* the same jittered cameras serve the self-rendered view consistency term (2c, in ``loop.forward_losses``).
* ``load_viewpoint_domain`` + ``domain_cameras`` (variant selfrender_replace_v2, 2026-10-29): cameras drawn uniformly over
  the frozen training-supported viewpoint domain (configs/metaworld/selfrender_domain_v1.yaml, scripts/selfrender_domain.py)
  for the decoded-Gaussian self-rendered synthetic views; variant selfrender_replace_v3 (2026-10-31) adds a radial
  distance jitter (``radius_jitter``, +-5 %) to those cameras.

Nothing here runs at evaluation or RL time.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from s4d.data.metaworld.cameras import LOOKAT, TRAIN_CAMERAS, offsets_to_orbit, orbit_rig, sample_offsets
from s4d.diag.heldout import fused_training_clouds, splat_batched

PATCH_COVERAGE = 0.9  # a synthetic-view patch is a token only if this fraction of its pixels is covered in every frame
REPO = Path(__file__).resolve().parents[2]
_DOMAINS: dict[str, dict] = {}


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


def random_shift(
    images: torch.Tensor, extra: torch.Tensor | None, prob: float, pad: int = 4
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """images (R,T,C,H,W), extra (R,T,1,H,W) or None: DrM-style random shift (``RandomShiftsAug``: replicate padding
    by ``pad`` pixels, then an integer crop back to H x W). With probability ``1 - prob`` a row is unchanged; otherwise
    one shift is drawn per row and applied to all T frames of the row and to its aligned ``extra`` map."""
    R, T, C, H, W = images.shape
    device = images.device
    shift = torch.rand(R, device=device) < prob
    if not bool(shift.any()):
        return images, extra
    offsets = torch.randint(0, 2 * pad + 1, (R, 2), device=device)
    offsets = torch.where(shift[:, None], offsets, torch.full_like(offsets, pad))  # pad, pad: the unshifted image
    rows = (offsets[:, 0, None] + torch.arange(H, device=device)).repeat_interleave(T, 0)  # (R*T, H)
    cols = (offsets[:, 1, None] + torch.arange(W, device=device)).repeat_interleave(T, 0)  # (R*T, W)
    index = torch.arange(R * T, device=device)[:, None, None]

    def apply(x):
        padded = F.pad(x.reshape(R * T, *x.shape[2:]).float(), (pad,) * 4, mode="replicate")
        out = padded.permute(0, 2, 3, 1)[index, rows[:, :, None], cols[:, None, :]].permute(0, 3, 1, 2)
        return out.reshape(x.shape).to(x.dtype)

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


def load_viewpoint_domain(path: str | Path) -> dict:
    """The frozen training-supported viewpoint domain (a YAML written by scripts/selfrender_domain.py; a relative path
    is resolved against the repository root). Returns the cell centres (n,2) (azimuth, MuJoCo elevation) in degrees,
    the half cell size, the orbit distance and the per-cell sampling probabilities (proportional to solid angle)."""
    p = Path(path)
    p = (p if p.is_absolute() else REPO / p).resolve()
    key = str(p)
    if key not in _DOMAINS:
        doc = yaml.safe_load(p.read_text())
        if not str(doc.get("version", "")).startswith("selfrender_domain_"):
            raise ValueError(f"{p} is not a self-render viewpoint domain")
        if not np.allclose(doc["rig"]["lookat"], LOOKAT):
            raise ValueError("the viewpoint domain must use the rig look-at point")
        cells = np.asarray([c[:2] for c in doc["cells"]], dtype=np.float64)
        half = float(doc["cell_deg"]) / 2.0
        if not len(cells) or np.any(cells[:, 1] - half < -90.0) or np.any(cells[:, 1] + half > 90.0):
            raise ValueError("domain cells must lie within elevation [-90, 90]")
        area = np.sin(np.radians(cells[:, 1] + half)) - np.sin(np.radians(cells[:, 1] - half))
        _DOMAINS[key] = {"version": doc["version"], "cells": cells, "half": half,
                         "distance": float(doc["rig"]["distance"]), "probs": area / area.sum()}
    return _DOMAINS[key]


def domain_cameras(
    batch_size: int, views: int, K: torch.Tensor, height: int, width: int, domain: dict, radius_jitter: float = 0.0
) -> dict[str, torch.Tensor]:
    """Cameras uniform on the sphere over the viewpoint ``domain`` (``load_viewpoint_domain``): a cell with probability
    proportional to its solid angle, then azimuth uniform and sin(elevation) uniform within the cell, at the domain's
    orbit distance, looking at LOOKAT (torch's default generator). One camera per (sample, view): the caller renders
    every timestep from it. K (B,3,3) is reused for every camera; returns K, w2c, c2w (B,views,...) and poses
    (B,views,3) = (azimuth, elevation, distance).

    ``radius_jitter`` j > 0 (variant selfrender_replace_v3, user 2026-10-31): the distance is the domain's times
    (1 + d), d uniform in [-j, j] per camera, as the point-cloud views' radius_frac offset (HELDOUT_RANGES, +-5 %);
    the direction is unchanged and the camera still looks at LOOKAT. j = 0 draws nothing more (v2 bit for bit)."""
    n = batch_size * views
    cell = torch.multinomial(torch.as_tensor(domain["probs"]), n, replacement=True).numpy()
    u = torch.rand(n, 2, dtype=torch.float64).numpy()
    centre, half = domain["cells"][cell], domain["half"]
    azimuth = centre[:, 0] + (2.0 * u[:, 0] - 1.0) * half
    s_lo, s_hi = np.sin(np.radians(centre[:, 1] - half)), np.sin(np.radians(centre[:, 1] + half))
    elevation = np.degrees(np.arcsin(np.clip(s_lo + u[:, 1] * (s_hi - s_lo), -1.0, 1.0)))
    distance = np.full(n, domain["distance"])
    if radius_jitter > 0.0:
        distance = distance * (1.0 + (2.0 * torch.rand(n, dtype=torch.float64).numpy() - 1.0) * radius_jitter)
    poses = np.stack((azimuth, elevation, distance), 1)
    rig = orbit_rig([tuple(p) for p in poses], height, width)
    like = {"device": K.device, "dtype": torch.float32}
    return {
        "K": K[:, None].expand(batch_size, views, 3, 3).contiguous(),
        "w2c": torch.as_tensor(rig["w2c"], **like).view(batch_size, views, 4, 4),
        "c2w": torch.as_tensor(rig["c2w"], **like).view(batch_size, views, 4, 4),
        "poses": torch.as_tensor(poses, dtype=torch.float64).view(batch_size, views, 3),
    }


@torch.no_grad()
def synthesize_views(batch: dict, cams: dict, near: float, far: float) -> dict[str, torch.Tensor]:
    """Fuse the training cameras' GT depth + RGB per (sample, time) and splat it into ``cams``.

    Returns images (B,T,n,3,H,W) in [0,1], depth (B,T,n,1,H,W) and covered (B,T,n,1,H,W)."""
    B, T = batch["depth"].shape[:2]
    H, W = batch["depth"].shape[-2:]
    points, colors, valid = fused_training_clouds(batch, far)
    n = cams["K"].shape[1]
    rgb, cov, dep = splat_batched(
        points, colors, valid, cams["K"][:, None].expand(B, T, n, 3, 3).reshape(B * T, n, 3, 3),
        cams["w2c"][:, None].expand(B, T, n, 4, 4).reshape(B * T, n, 4, 4), H, W, near
    )
    return {
        "images": rgb.view(B, T, n, 3, H, W),
        "covered": cov.view(B, T, n, 1, H, W),
        "depth": dep.view(B, T, n, 1, H, W),
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


def coverage_visible_tokens(covered: torch.Tensor, patch: int, keep: int) -> torch.Tensor:
    """Per-token form of ``coverage_visible`` for encoders with ``mask_mode`` "random": covered (R,T,1,H,W) ->
    visible-token mask (R,T*N) (frame-major) with exactly ``keep`` True per row, uniform among the tokens whose own
    frame's patch is covered to >= PATCH_COVERAGE; holes (insufficiently covered tokens) are used only when a row has
    fewer such tokens than ``keep`` (the same coverage exception as the shared spatial mask)."""
    R, T = covered.shape[:2]
    frac = F.avg_pool2d(covered.flatten(0, 1).float(), patch).view(R, -1)  # (R,T*N), frame-major
    priority = torch.where(frac >= PATCH_COVERAGE, torch.rand_like(frac), torch.rand_like(frac) - 2.0)
    ids = torch.topk(priority, keep, dim=1).indices
    visible = torch.zeros_like(frac, dtype=torch.bool)
    visible.scatter_(1, ids, True)
    return visible
