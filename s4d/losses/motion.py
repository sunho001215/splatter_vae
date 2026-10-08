"""3D motion loss on rendered per-pixel displacements. Shapes (B,P,V,C,H,W)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

PAIR_WEIGHTS = (0.4, 0.4, 0.2)
MOVING_THRESHOLD_M = 0.005
# normalization="moving" (review item 4e): the moving term sums over valid pixels with non-zero target motion,
# w = 1 + |target| / MOTION_WEIGHT_SCALE_M, divided by max(sum w, MIN_MOVING_FRACTION * valid pixels); a static term
# (weight STATIC_WEIGHT) averages over valid pixels with zero target motion.
MOTION_WEIGHT_SCALE_M = 0.03
MIN_MOVING_FRACTION = 0.01
STATIC_WEIGHT = 0.1
ZERO_MOTION_M = 1e-6


def expected_displacement(features: torch.Tensor, coverage: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Alpha-normalised displacement: features (B,P,V,3,H,W) / coverage (B,P,V,1,H,W)."""
    return torch.nan_to_num(features / coverage.clamp_min(eps), nan=0.0, posinf=0.0, neginf=0.0)


def motion_loss(
    pred: torch.Tensor,
    coverage: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    huber_delta: float = 0.01,
    coverage_threshold: float = 0.01,
    pair_weights=PAIR_WEIGHTS,
    normalization: str = "valid",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Huber loss between predicted and target displacements, weighted by motion_weight.

    ``normalization="valid"`` averages over every valid pixel; ``"moving"`` normalises over moving pixels (see the
    constants above). Returns the pair-weighted scalar loss and metrics: per-pair loss, EPE, relative EPE on moving
    pixels.
    """
    if normalization not in ("valid", "moving"):
        raise ValueError(f"unknown motion normalization {normalization!r}")
    valid = weight > 0
    w = weight * valid.float()
    per_pixel = F.smooth_l1_loss(pred.float(), target.float(), beta=huber_delta, reduction="none").sum(3, keepdim=True)
    err = (pred.float() - target.float()).norm(dim=3, keepdim=True)
    gt_mag = target.float().norm(dim=3, keepdim=True)
    moving = valid & (gt_mag > MOVING_THRESHOLD_M)
    losses, metrics = [], {}
    for p, name in enumerate(("01", "12", "02")):
        wp = w[:, p]
        denom = wp.sum().clamp_min(1.0)
        if normalization == "valid":
            lp = (per_pixel[:, p] * wp).sum() / denom
        else:
            mag = gt_mag[:, p]
            nonzero = (valid[:, p] & (mag > ZERO_MOTION_M)).float()
            zero = (valid[:, p] & (mag <= ZERO_MOTION_M)).float()
            wm = nonzero * (1.0 + mag / MOTION_WEIGHT_SCALE_M)
            floor = MIN_MOVING_FRACTION * valid[:, p].float().sum()
            moving_term = (per_pixel[:, p] * wm).sum() / torch.maximum(wm.sum(), floor.clamp_min(1.0))
            static_term = (per_pixel[:, p] * zero).sum() / zero.sum().clamp_min(1.0)
            lp = moving_term + STATIC_WEIGHT * static_term
        losses.append(lp)
        metrics[f"motion_loss_{name}"] = lp.detach()
        metrics[f"epe_{name}"] = ((err[:, p] * wp).sum() / denom).detach()
        mv = moving[:, p].float()

        def region_mean(values, region):
            count = region.sum()
            average = (values * region).sum() / count.clamp_min(1.0)
            return torch.where(count > 0, average, average.new_tensor(float("nan"))).detach()

        metrics[f"rel_epe_{name}"] = region_mean(err[:, p] / gt_mag[:, p].clamp_min(1e-6), mv)
        metrics[f"motion_moving_count_{name}"] = mv.sum().detach()
        metrics[f"motion_valid_weight_{name}"] = wp.sum().detach()
        static = (valid[:, p] & (gt_mag[:, p] <= MOVING_THRESHOLD_M)).float()
        metrics[f"motion_static_count_{name}"] = static.sum().detach()
        metrics[f"epe_static_{name}"] = region_mean(err[:, p], static)
        metrics[f"epe_moving_{name}"] = region_mean(err[:, p], mv)
    total = sum(float(pw) * lp for pw, lp in zip(pair_weights, losses))
    visible = valid & (coverage.detach() > coverage_threshold)
    metrics["motion_visible_fraction"] = (visible.float().sum() / valid.float().sum().clamp_min(1.0)).detach()
    return total, metrics
