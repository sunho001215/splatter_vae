from __future__ import annotations

import torch


def compute_depth_l1_loss(
    rendered_depth: torch.Tensor,
    target_depth: torch.Tensor,
    supervision_mask: torch.Tensor,
    *,
    return_per_render: bool = False,
):
    """Compute metric L1 depth over supervised pixels with valid target depth."""
    if (
        rendered_depth.shape != target_depth.shape
        or rendered_depth.shape != supervision_mask.shape
        or rendered_depth.dim() < 3
        or rendered_depth.shape[-3] != 1
    ):
        raise ValueError(
            "Rendered depth, target depth, and supervision mask must match with one depth channel."
        )
    predicted = rendered_depth.float().reshape(
        -1, 1, *rendered_depth.shape[-2:]
    )
    target = target_depth.float().reshape_as(predicted)
    mask = supervision_mask.reshape_as(predicted).bool()

    target_valid = torch.isfinite(target) & (target > 0.0)
    valid = mask & target_valid
    safe_predicted = torch.nan_to_num(
        predicted, nan=0.0, posinf=0.0, neginf=0.0
    )
    safe_target = torch.where(target_valid, target, torch.zeros_like(target))
    weights = valid.to(dtype=safe_predicted.dtype)
    valid_count = weights.flatten(1).sum(-1)
    per_render = (
        (safe_predicted - safe_target).abs() * weights
    ).flatten(1).sum(-1) / valid_count.clamp_min(1.0)

    if return_per_render:
        return per_render, valid_count
    return (per_render * valid_count).sum() / valid_count.sum().clamp_min(1.0)
