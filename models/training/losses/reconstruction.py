from __future__ import annotations

import torch
from fused_ssim import FusedSSIMMap

DYNAMIC_FLOW_SCALE_PIXELS = 3.0


def _as_render_rows(values: torch.Tensor) -> torch.Tensor:
    return values.reshape(-1, *values.shape[-3:])


def normalized_dynamic_weights(
    dynamic_score: torch.Tensor | None,
    valid_mask: torch.Tensor,
    dynamic_region_weight: float,
) -> torch.Tensor:
    valid = valid_mask.to(dtype=torch.float32)
    if dynamic_score is None or float(dynamic_region_weight) == 0.0:
        return torch.ones_like(valid)
    score = dynamic_score.detach().float().clamp(0.0, 1.0)
    if score.shape != valid.shape:
        raise ValueError(f"Dynamic score and valid mask must match, got {score.shape} and {valid.shape}.")
    weights = 1.0 + float(dynamic_region_weight) * score
    dims = tuple(range(1, weights.dim()))
    valid_count = valid.sum(dim=dims, keepdim=True)
    valid_mean = (weights * valid).sum(dim=dims, keepdim=True) / valid_count.clamp_min(1.0)
    return weights / valid_mean.clamp_min(1.0e-8)


def _forward_warp_score(score: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Detached bilinear forward splat of a scalar source score."""
    score = score.detach().float()
    flow = flow.detach().float()
    if score.dim() != 5 or score.shape[2] != 1:
        raise ValueError("Expected score as (B,A,1,H,W).")
    if flow.shape != (score.shape[0], score.shape[1], 2, score.shape[3], score.shape[4]):
        raise ValueError("Flow must align with score and contain two channels.")
    batch_views = score.shape[0] * score.shape[1]
    height, width = score.shape[-2:]
    values = score.reshape(batch_views, -1)
    vector = flow.reshape(batch_views, 2, -1)
    ys, xs = torch.meshgrid(
        torch.arange(height, device=score.device, dtype=torch.float32),
        torch.arange(width, device=score.device, dtype=torch.float32),
        indexing="ij",
    )
    target_x = xs.flatten()[None] + vector[:, 0]
    target_y = ys.flatten()[None] + vector[:, 1]
    x0 = target_x.floor(); y0 = target_y.floor()
    output = score.new_zeros(batch_views, height * width)
    coverage = score.new_zeros(batch_views, height * width)
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        xi = x0 + dx; yi = y0 + dy
        weight = (1.0 - (target_x - xi).abs()) * (1.0 - (target_y - yi).abs())
        valid = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height) & torch.isfinite(weight)
        index = (yi.clamp(0, height - 1) * width + xi.clamp(0, width - 1)).long()
        contribution = values * weight.clamp_min(0.0) * valid
        output.scatter_add_(1, index, contribution)
        coverage.scatter_add_(1, index, weight.clamp_min(0.0) * valid)
    warped = output / coverage.clamp_min(1.0e-8)
    return warped.view(score.shape[0], score.shape[1], 1, height, width).detach()


def build_target_dynamic_scores(optical_flows: torch.Tensor) -> torch.Tensor:
    """Build detached target-aligned motion scores for t0, t1, and t2."""
    if optical_flows.dim() != 6 or optical_flows.shape[1:4:2] != (3, 2):
        raise ValueError(f"Expected optical flows as (B,3,A,2,H,W), got {tuple(optical_flows.shape)}.")
    flow = torch.nan_to_num(
        optical_flows.detach().float(), nan=0.0, posinf=0.0, neginf=0.0
    )
    scores = (flow.square().sum(dim=3, keepdim=True).sqrt() / DYNAMIC_FLOW_SCALE_PIXELS).clamp(0.0, 1.0)
    score01, score12, score02 = scores.unbind(1)
    flow01, flow12, flow02 = flow.unbind(1)
    score0 = torch.maximum(score01, score02)
    score1 = torch.maximum(_forward_warp_score(score01, flow01), score12)
    score2 = torch.maximum(_forward_warp_score(score02, flow02), _forward_warp_score(score12, flow12))
    return torch.stack((score0, score1, score2), dim=1).detach()


def compute_reconstruction_loss(
    predicted: torch.Tensor,
    ground_truth: torch.Tensor,
    ssim_weight: float = 0.2,
    loss_mask: torch.Tensor | None = None,
    dynamic_score: torch.Tensor | None = None,
    dynamic_region_weight: float = 0.0,
    return_per_render: bool = False,
):
    """Masked L1+D-SSIM with optional normalized dynamic redistribution."""
    predicted = _as_render_rows(predicted.float())
    ground_truth = _as_render_rows(ground_truth.float())
    if predicted.shape != ground_truth.shape or predicted.shape[1] != 3:
        raise ValueError("Predicted and target RGB tensors must match as (R,3,H,W).")
    if loss_mask is None:
        mask = torch.ones_like(predicted[:, :1])
    else:
        mask = _as_render_rows(loss_mask.float())
        if mask.shape != predicted[:, :1].shape:
            raise ValueError("RGB loss mask must be one-channel and align with rendered RGB.")
        mask = mask.clamp(0.0, 1.0)
    score = None if dynamic_score is None else _as_render_rows(dynamic_score.float())
    weights = normalized_dynamic_weights(score, mask, dynamic_region_weight)
    weighted_mask = mask * weights
    denominator = weighted_mask.flatten(1).sum(-1)
    l1_map = (predicted - ground_truth).abs().mean(dim=1, keepdim=True)
    l1 = (l1_map * weighted_mask).flatten(1).sum(-1) / denominator.clamp_min(1.0)
    if float(ssim_weight) > 0.0:
        ssim_map = 1.0 - FusedSSIMMap.apply(
            0.01**2, 0.03**2, predicted.contiguous(), ground_truth.contiguous(), "same", True, 2
        ).mean(dim=1, keepdim=True)
        dssim = (ssim_map * weighted_mask).flatten(1).sum(-1) / denominator.clamp_min(1.0)
        per_render = (1.0 - float(ssim_weight)) * l1 + float(ssim_weight) * dssim
    else:
        per_render = l1
    valid = (denominator > 0).to(per_render.dtype)
    reduction_weight = denominator * valid
    if return_per_render:
        return per_render, reduction_weight
    return (per_render * reduction_weight).sum() / reduction_weight.sum().clamp_min(1.0)


def compute_balanced_silhouette_loss(
    rendered_alpha: torch.Tensor,
    target_mask: torch.Tensor,
    dynamic_score: torch.Tensor | None = None,
    dynamic_region_weight: float = 0.0,
    return_per_render: bool = False,
):
    alpha = _as_render_rows(rendered_alpha.float()).clamp(1.0e-6, 1.0 - 1.0e-6)
    mask = _as_render_rows(target_mask.float()).clamp(0.0, 1.0)
    if alpha.shape != mask.shape or alpha.shape[1] != 1:
        raise ValueError("Alpha and target mask must align as (R,1,H,W).")
    score = None if dynamic_score is None else _as_render_rows(dynamic_score.float())
    foreground_weights = normalized_dynamic_weights(score, mask, dynamic_region_weight)
    foreground_denominator = mask.flatten(1).sum(-1)
    foreground = -(mask * foreground_weights * torch.log(alpha)).flatten(1).sum(-1)
    foreground = foreground / (mask * foreground_weights).flatten(1).sum(-1).clamp_min(1.0)
    background_mask = 1.0 - mask
    background_denominator = background_mask.flatten(1).sum(-1)
    background = -(background_mask * torch.log1p(-alpha)).flatten(1).sum(-1)
    background = background / background_denominator.clamp_min(1.0)
    foreground_valid = (foreground_denominator > 0).to(alpha.dtype)
    background_valid = (background_denominator > 0).to(alpha.dtype)
    total = 0.5 * (foreground + background)
    if return_per_render:
        return foreground, background, total, foreground_valid, background_valid
    foreground_loss = (foreground * foreground_valid).sum() / foreground_valid.sum().clamp_min(1.0)
    background_loss = (background * background_valid).sum() / background_valid.sum().clamp_min(1.0)
    return foreground_loss, background_loss, 0.5 * (foreground_loss + background_loss)
