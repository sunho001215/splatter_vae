from __future__ import annotations

import torch
import torch.nn.functional as F

from .reconstruction import normalized_dynamic_weights


def _masked_standardize(values: torch.Tensor, valid: torch.Tensor, eps: float = 1.0e-6):
    weights = valid.to(values.dtype)
    count = weights.sum(-1, keepdim=True)
    mean = (values * weights).sum(-1, keepdim=True) / count.clamp_min(1.0)
    centered = values - mean
    variance = (centered.square() * weights).sum(-1, keepdim=True) / count.clamp_min(1.0)
    return centered / variance.clamp_min(eps).sqrt(), count.squeeze(-1), variance.squeeze(-1)


def _patchify(values: torch.Tensor, patch_size: int) -> tuple[torch.Tensor, int]:
    rows, _, height, width = values.shape
    size = int(patch_size)
    pad_h = (size - height % size) % size
    pad_w = (size - width % size) % size
    if pad_h or pad_w:
        values = F.pad(values, (0, pad_w, 0, pad_h))
    h_blocks = values.shape[-2] // size
    w_blocks = values.shape[-1] // size
    patches = values[:, 0].reshape(rows, h_blocks, size, w_blocks, size)
    patches = patches.permute(0, 1, 3, 2, 4).reshape(rows, h_blocks * w_blocks, size * size)
    return patches, h_blocks * w_blocks


def compute_global_local_depth_loss(
    rendered_depth: torch.Tensor,
    target_depth: torch.Tensor,
    foreground_mask: torch.Tensor,
    patch_size: int,
    min_valid_pixels: int,
    dynamic_score: torch.Tensor | None = None,
    dynamic_region_weight: float = 0.0,
    return_per_render: bool = False,
):
    """Scale/shift-invariant FP32 depth losses with weighted final reductions."""
    predicted = rendered_depth.float().reshape(-1, 1, *rendered_depth.shape[-2:])
    target_depth = target_depth.float().reshape_as(predicted)
    foreground = foreground_mask.reshape_as(predicted).bool()
    if predicted.shape != target_depth.shape:
        raise ValueError("Rendered depth, target depth, and mask must have identical shapes.")
    target_valid = torch.isfinite(target_depth) & (target_depth > 0.0)
    valid = foreground & target_valid
    predicted = torch.nan_to_num(predicted, nan=0.0, posinf=0.0, neginf=0.0)
    target = torch.where(target_valid, target_depth, torch.zeros_like(target_depth))
    score = None if dynamic_score is None else dynamic_score.float().reshape_as(predicted)

    flat_predicted = predicted.flatten(1)
    flat_target = target.flatten(1)
    flat_valid = valid.flatten(1)
    pred_global, global_count, _ = _masked_standardize(flat_predicted, flat_valid)
    target_global, _, _ = _masked_standardize(flat_target, flat_valid)
    global_penalty = F.smooth_l1_loss(pred_global, target_global, beta=1.0, reduction="none")
    global_weights = normalized_dynamic_weights(
        None if score is None else score.flatten(1), flat_valid, dynamic_region_weight
    ) * flat_valid
    global_per_render = (global_penalty * global_weights).sum(-1) / global_weights.sum(-1).clamp_min(1.0)
    global_valid = (global_count >= 2).to(predicted.dtype)

    size = max(1, int(patch_size))
    min_pixels = max(2, int(min_valid_pixels))
    pred_patches, patch_count = _patchify(predicted, size)
    target_patches, _ = _patchify(target, size)
    valid_patches, _ = _patchify(valid.to(predicted.dtype), size)
    valid_patches = valid_patches.bool()
    score_patches = None
    if score is not None:
        score_patches, _ = _patchify(score, size)
    pred_local, local_count, _ = _masked_standardize(pred_patches, valid_patches)
    target_local, _, target_variance = _masked_standardize(target_patches, valid_patches)
    valid_patch = (local_count >= min_pixels) & (target_variance > 1.0e-6)
    local_penalty = F.smooth_l1_loss(pred_local, target_local, beta=1.0, reduction="none")
    flat_patch_valid = valid_patches.reshape(-1, valid_patches.shape[-1])
    flat_patch_score = None if score_patches is None else score_patches.reshape_as(flat_patch_valid)
    local_weights = normalized_dynamic_weights(
        flat_patch_score, flat_patch_valid, dynamic_region_weight
    ).view_as(valid_patches) * valid_patches
    local_per_patch = (local_penalty * local_weights).sum(-1) / local_weights.sum(-1).clamp_min(1.0)
    valid_patch_float = valid_patch.to(predicted.dtype)
    local_valid_count = valid_patch_float.sum(-1)
    local_per_render = (local_per_patch * valid_patch_float).sum(-1) / local_valid_count.clamp_min(1.0)

    if return_per_render:
        return global_per_render, local_per_render, global_valid, local_valid_count
    global_loss = (global_per_render * global_valid).sum() / global_valid.sum().clamp_min(1.0)
    local_loss = (local_per_render * local_valid_count).sum() / local_valid_count.sum().clamp_min(1.0)
    return global_loss, local_loss
