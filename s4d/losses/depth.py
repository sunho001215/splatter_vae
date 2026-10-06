"""Depth supervision: teacher alignment, L1, multi-scale log-gradient matching, hard depth. Rows (R,1,H,W)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

MIN_ALIGN_PIXELS = 16


def align_teacher(
    rendered: torch.Tensor,
    teacher: torch.Tensor,
    alpha: torch.Tensor,
    mode: str,
    scale_bounds: tuple[float, float] = (0.5, 2.0),
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Align the teacher depth to the (detached) rendered depth per image.

    mode: "none" (identity), "scale" (median log ratio), "scale_shift" (weighted least squares).
    Returns the aligned teacher (0 where the teacher is invalid) and {"scale", "shift"} of shape (R,).
    """
    rendered = rendered.detach().float()
    teacher = teacher.float()
    R = teacher.shape[0]
    valid = (teacher > 0) & (alpha.detach() > 0.5) & (rendered > 0)
    count = valid.flatten(1).sum(1)
    scale = torch.ones(R, device=teacher.device)
    shift = torch.zeros(R, device=teacher.device)
    if mode == "scale":
        log_ratio = torch.where(
            valid,
            torch.log(rendered.clamp_min(1e-6)) - torch.log(teacher.clamp_min(1e-6)),
            torch.full_like(teacher, float("nan")),
        )
        med = torch.nanmedian(log_ratio.flatten(1), dim=1).values
        est = torch.exp(torch.nan_to_num(med, nan=0.0)).clamp(*scale_bounds)
        scale = torch.where(count >= MIN_ALIGN_PIXELS, est, scale)
    elif mode == "scale_shift":
        confidence = torch.ones_like(teacher) if weights is None else weights.detach().float()
        w = (valid.float() * confidence).flatten(1)
        t = teacher.flatten(1)
        d = rendered.flatten(1)
        sw = w.sum(1).clamp_min(1.0)
        mt = (w * t).sum(1) / sw
        md = (w * d).sum(1) / sw
        var = (w * (t - mt[:, None]) ** 2).sum(1) / sw
        cov = (w * (t - mt[:, None]) * (d - md[:, None])).sum(1) / sw
        a = (cov / var.clamp_min(1e-8)).clamp(*scale_bounds)
        b = md - a * mt
        ok = (count >= MIN_ALIGN_PIXELS) & (var > 1e-8)
        scale = torch.where(ok, a, scale)
        shift = torch.where(ok, b, shift)
    elif mode != "none":
        raise ValueError(f"unknown depth alignment mode {mode!r}")
    aligned = teacher * scale[:, None, None, None] + shift[:, None, None, None]
    aligned = torch.where(teacher > 0, aligned.clamp_min(1e-6), torch.zeros_like(aligned))
    return aligned, {"scale": scale, "shift": shift}


def depth_l1(rendered: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    diff = (rendered.float() - target.float()).abs() * weights * valid.float()
    return diff.sum(dim=(1, 2, 3)) / valid.float().sum(dim=(1, 2, 3)).clamp_min(1.0)


def _downsample(x: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return F.avg_pool2d(x, 2), -F.max_pool2d(-valid.float(), 2) > 0.5  # valid only if all four are valid


def depth_gradient_loss(
    rendered: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, levels: int = 4, weights: torch.Tensor | None = None
) -> torch.Tensor:
    """Multi-scale gradient matching on the log-depth difference, per render (R,)."""
    r = torch.log(rendered.float().clamp_min(1e-4)) - torch.log(target.float().clamp_min(1e-4))
    r = torch.where(valid, r, torch.zeros_like(r))
    weights = torch.ones_like(r) if weights is None else weights.detach().float()
    total = torch.zeros(r.shape[0], device=r.device)
    for level in range(levels):
        vx = valid[..., :, 1:] & valid[..., :, :-1]
        vy = valid[..., 1:, :] & valid[..., :-1, :]
        wx = (weights[..., :, 1:] + weights[..., :, :-1]) * 0.5
        wy = (weights[..., 1:, :] + weights[..., :-1, :]) * 0.5
        gx = (r[..., :, 1:] - r[..., :, :-1]).abs() * vx.float() * wx
        gy = (r[..., 1:, :] - r[..., :-1, :]).abs() * vy.float() * wy
        denom = (vx.float().sum(dim=(1, 2, 3)) + vy.float().sum(dim=(1, 2, 3))).clamp_min(1.0)
        total = total + (gx.sum(dim=(1, 2, 3)) + gy.sum(dim=(1, 2, 3))) / denom
        if level + 1 < levels and min(r.shape[-2:]) >= 4:
            r, valid = _downsample(r, valid)
            weights = F.avg_pool2d(weights, 2)
    return total / levels


def abs_rel(rendered: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    rel = ((rendered.float() - target.float()).abs() / target.float().clamp_min(1e-6)) * valid.float()
    return rel.sum(dim=(1, 2, 3)) / valid.float().sum(dim=(1, 2, 3)).clamp_min(1.0)
