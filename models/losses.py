import torch
import torch.nn.functional as F

from fused_ssim import FusedSSIMMap


def compute_reconstruction_loss(
    predicted: torch.Tensor,
    ground_truth: torch.Tensor,
    ssim_weight: float = 0.2,
    loss_mask: torch.Tensor | None = None,
):
    """
    Compute combined L1 + D-SSIM reconstruction loss over a masked image region.

    predicted, ground_truth: (B,3,H,W), values in [0,1]
    ssim_weight: weight for D-SSIM term in [0,1]
    loss_mask: optional (B,1,H,W) or (B,H,W) evaluation mask.
    """
    mask = None
    if loss_mask is not None:
        mask = loss_mask.to(device=predicted.device, dtype=predicted.dtype).clamp(0.0, 1.0)
        if mask.ndim == predicted.ndim - 1:
            mask = mask.unsqueeze(1)
        if mask.shape[0] != predicted.shape[0] or mask.shape[-2:] != predicted.shape[-2:]:
            raise ValueError(
                f"loss_mask must match batch/spatial dimensions of predicted, got "
                f"{tuple(mask.shape)} and {tuple(predicted.shape)}."
            )
        if mask.shape[1] != 1:
            raise ValueError(f"loss_mask must have one channel, got {mask.shape[1]}.")

    if mask is None:
        l1_loss = F.l1_loss(predicted, ground_truth)
    else:
        mask_rgb = mask.expand_as(predicted)
        l1_loss = ((predicted - ground_truth).abs() * mask_rgb).sum() / mask_rgb.sum().clamp_min(1.0)

    if ssim_weight <= 0.0:
        return l1_loss

    # ``fused_ssim`` reduces the spatial map to a scalar internally, which
    # cannot be masked after the fact. Call its public autograd function
    # directly, reduce the RGB channels, and only then apply the exact spatial
    # foreground mask.
    ssim_map = FusedSSIMMap.apply(
        0.01**2,
        0.03**2,
        predicted.contiguous(),
        ground_truth.contiguous(),
        "same",
        True,
        2,
    ).mean(dim=1)
    ssim_loss_map = 1.0 - ssim_map
    if mask is None:
        ssim_loss = ssim_loss_map.mean()
    else:
        mask_2d = mask[:, 0]
        ssim_loss = (ssim_loss_map * mask_2d).sum() / mask_2d.sum().clamp_min(1.0)

    total_loss = (1 - ssim_weight) * l1_loss + ssim_weight * ssim_loss
    return total_loss


def compute_balanced_silhouette_loss(
    rendered_alpha: torch.Tensor,
    target_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Balanced foreground/background BCE using the learned-opacity alpha."""
    alpha = rendered_alpha.clamp(1.0e-6, 1.0 - 1.0e-6)
    mask = target_mask.to(device=alpha.device, dtype=alpha.dtype).clamp(0.0, 1.0)
    if alpha.shape != mask.shape:
        raise ValueError(f"Alpha and mask shapes must match, got {tuple(alpha.shape)} and {tuple(mask.shape)}.")

    if alpha.dim() < 2:
        raise ValueError(f"Expected alpha and mask with spatial dimensions, got {tuple(alpha.shape)}.")

    # Treat every item in the leading dimensions as an independent render.
    # Separately normalize its foreground and background pixels, then average
    # over renders containing that region. This prevents large silhouettes or
    # backgrounds from assigning a larger weight to a camera/timestep. The
    # weighted reductions also return graph-connected zeros when a region is
    # absent from every render.
    alpha_per_render = alpha.reshape(-1, alpha.shape[-2] * alpha.shape[-1])
    mask_per_render = mask.reshape_as(alpha_per_render)
    foreground_count = mask_per_render.sum(dim=-1)
    foreground_per_render = -(mask_per_render * torch.log(alpha_per_render)).sum(dim=-1)
    foreground_per_render = foreground_per_render / foreground_count.clamp_min(1.0)
    foreground_valid = (foreground_count > 0).to(dtype=alpha.dtype)
    foreground_loss = (foreground_per_render * foreground_valid).sum() / foreground_valid.sum().clamp_min(1.0)

    background_mask = 1.0 - mask_per_render
    background_count = background_mask.sum(dim=-1)
    background_per_render = -(background_mask * torch.log1p(-alpha_per_render)).sum(dim=-1)
    background_per_render = background_per_render / background_count.clamp_min(1.0)
    background_valid = (background_count > 0).to(dtype=alpha.dtype)
    background_loss = (background_per_render * background_valid).sum() / background_valid.sum().clamp_min(1.0)
    silhouette_loss = 0.5 * (foreground_loss + background_loss)
    return foreground_loss, background_loss, silhouette_loss


def _masked_standardize(
    values: torch.Tensor,
    valid: torch.Tensor,
    eps: float = 1.0e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    weights = valid.to(dtype=values.dtype)
    count = weights.sum(dim=-1, keepdim=True)
    mean = (values * weights).sum(dim=-1, keepdim=True) / count.clamp_min(1.0)
    centered = values - mean
    variance = (centered.square() * weights).sum(dim=-1, keepdim=True) / count.clamp_min(1.0)
    normalized = centered / variance.clamp_min(float(eps)).sqrt()
    return normalized, count.squeeze(-1), variance.squeeze(-1)


def compute_global_local_depth_loss(
    rendered_depth: torch.Tensor,
    target_depth: torch.Tensor | None,
    foreground_mask: torch.Tensor,
    patch_size: int,
    min_valid_pixels: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scale/shift-invariant Smooth-L1 depth supervision at global and patch scales."""
    zero = torch.nan_to_num(rendered_depth).sum() * 0.0
    if target_depth is None:
        return zero, zero
    if rendered_depth.shape != target_depth.shape or rendered_depth.shape != foreground_mask.shape:
        raise ValueError(
            "Rendered depth, target depth, and foreground mask must share shape, got "
            f"{tuple(rendered_depth.shape)}, {tuple(target_depth.shape)}, and {tuple(foreground_mask.shape)}."
        )

    predicted = torch.nan_to_num(rendered_depth, nan=0.0, posinf=0.0, neginf=0.0)
    target_is_valid = torch.isfinite(target_depth) & (target_depth > 0.0)
    target = torch.where(target_is_valid, target_depth, torch.zeros_like(target_depth))
    foreground = foreground_mask.to(device=predicted.device, dtype=torch.bool)
    valid = foreground & target_is_valid

    flat_predicted = predicted.reshape(-1, predicted.shape[-2] * predicted.shape[-1])
    flat_target = target.reshape_as(flat_predicted)
    flat_valid = valid.reshape_as(flat_predicted)
    pred_global, global_count, _ = _masked_standardize(flat_predicted, flat_valid)
    target_global, _, _ = _masked_standardize(flat_target, flat_valid)
    global_penalty = F.smooth_l1_loss(pred_global, target_global, beta=1.0, reduction="none")
    global_per_image = (global_penalty * flat_valid).sum(dim=-1) / global_count.clamp_min(1.0)
    valid_images = (global_count >= 2).to(dtype=predicted.dtype)
    global_loss = (global_per_image * valid_images).sum() / valid_images.sum().clamp_min(1.0)

    size = max(1, int(patch_size))
    min_pixels = max(2, int(min_valid_pixels))
    height, width = predicted.shape[-2:]
    pad_h = (size - height % size) % size
    pad_w = (size - width % size) % size
    predicted_2d = predicted.reshape(-1, 1, height, width)
    target_2d = target.reshape(-1, 1, height, width)
    valid_2d = valid.reshape(-1, 1, height, width)
    if pad_h or pad_w:
        padding = (0, pad_w, 0, pad_h)
        predicted_2d = F.pad(predicted_2d, padding)
        target_2d = F.pad(target_2d, padding)
        valid_2d = F.pad(valid_2d, padding, value=False)

    padded_height, padded_width = predicted_2d.shape[-2:]
    height_blocks = padded_height // size
    width_blocks = padded_width // size

    def non_overlapping_patches(values: torch.Tensor) -> torch.Tensor:
        values = values[:, 0].reshape(-1, height_blocks, size, width_blocks, size)
        return values.permute(0, 1, 3, 2, 4).reshape(-1, height_blocks * width_blocks, size * size)

    pred_patches = non_overlapping_patches(predicted_2d)
    target_patches = non_overlapping_patches(target_2d)
    valid_patches = non_overlapping_patches(valid_2d).to(dtype=torch.bool)

    pred_local, local_count, _ = _masked_standardize(pred_patches, valid_patches)
    target_local, _, target_variance = _masked_standardize(target_patches, valid_patches)
    valid_patch = (local_count >= min_pixels) & (target_variance > 1.0e-6)
    local_penalty = F.smooth_l1_loss(pred_local, target_local, beta=1.0, reduction="none")
    local_per_patch = (local_penalty * valid_patches).sum(dim=-1) / local_count.clamp_min(1.0)
    valid_patch_weights = valid_patch.to(dtype=predicted.dtype)
    local_loss = (local_per_patch * valid_patch_weights).sum() / valid_patch_weights.sum().clamp_min(1.0)
    return global_loss, local_loss


def _masked_multi_positive_nce_from_normalized(
    normalized_features: torch.Tensor,
    positive_mask: torch.Tensor,
    negative_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    logits = (normalized_features @ normalized_features.t()) / float(temperature)
    positive_mask = positive_mask.to(device=logits.device, dtype=torch.bool)
    denominator_mask = positive_mask | negative_mask.to(device=logits.device, dtype=torch.bool)
    valid_queries = positive_mask.any(dim=1) & denominator_mask.any(dim=1)

    # Give rows without a valid positive a finite, detached fallback entry.
    # Their final weight is zero, but avoiding all-masked logsumexp rows also
    # prevents NaNs in backward for single-state or single-camera batches.
    fallback = torch.eye(logits.shape[0], device=logits.device, dtype=torch.bool)
    invalid_rows = ~valid_queries
    safe_positive_mask = positive_mask | (fallback & invalid_rows[:, None])
    safe_denominator_mask = denominator_mask | (fallback & invalid_rows[:, None])
    neg_inf = torch.finfo(logits.dtype).min
    log_positive = torch.logsumexp(logits.masked_fill(~safe_positive_mask, neg_inf), dim=1)
    log_denominator = torch.logsumexp(logits.masked_fill(~safe_denominator_mask, neg_inf), dim=1)
    loss_per_query = torch.where(
        valid_queries,
        -(log_positive - log_denominator),
        torch.zeros_like(log_positive),
    )
    valid_weights = valid_queries.to(dtype=logits.dtype)
    return loss_per_query.sum() / valid_weights.sum().clamp_min(1.0)


def masked_multi_positive_nce(
    features: torch.Tensor,
    positive_mask: torch.Tensor,
    negative_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Multi-positive InfoNCE with caller-provided positive/negative masks."""
    if temperature <= 0.0:
        raise ValueError("temperature must be > 0.")
    normalized = F.normalize(
        torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0),
        dim=-1,
        eps=1e-6,
    )
    return _masked_multi_positive_nce_from_normalized(
        normalized_features=normalized,
        positive_mask=positive_mask,
        negative_mask=negative_mask,
        temperature=temperature,
    )


def compute_view_structured_representation_losses(
    s_inv_by_view: torch.Tensor,
    z_dep_by_view: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Contrastive and normalized-L1 consistency losses from one encoder pass.

    Invariant positives are the same state under different views. Dependent
    positives are the same camera index across different states. Camera order
    is therefore assumed to be globally aligned across the batch.
    """
    if temperature <= 0.0:
        raise ValueError("temperature must be > 0.")
    if s_inv_by_view.dim() != 3 or z_dep_by_view.dim() != 3:
        raise ValueError(
            "Expected s_inv_by_view and z_dep_by_view as (B,A,D), got "
            f"{tuple(s_inv_by_view.shape)} and {tuple(z_dep_by_view.shape)}."
        )
    if s_inv_by_view.shape[:2] != z_dep_by_view.shape[:2]:
        raise ValueError(
            "Invariant/dependent view grids must share (B,A), got "
            f"{tuple(s_inv_by_view.shape[:2])} and {tuple(z_dep_by_view.shape[:2])}."
        )

    batch, views = s_inv_by_view.shape[:2]
    num_items = batch * views
    device = s_inv_by_view.device
    sample_ids = torch.arange(batch, device=device).repeat_interleave(views)
    view_ids = torch.arange(views, device=device).repeat(batch)
    diagonal = torch.eye(num_items, device=device, dtype=torch.bool)
    same_sample = sample_ids[:, None] == sample_ids[None, :]
    same_view = view_ids[:, None] == view_ids[None, :]
    invariant_positive = same_sample & ~diagonal
    dependent_positive = same_view & ~diagonal

    invariant_features = F.normalize(
        torch.nan_to_num(s_inv_by_view.reshape(num_items, -1), nan=0.0, posinf=0.0, neginf=0.0),
        dim=-1,
        eps=1e-6,
    )
    dependent_features = F.normalize(
        torch.nan_to_num(z_dep_by_view.reshape(num_items, -1), nan=0.0, posinf=0.0, neginf=0.0),
        dim=-1,
        eps=1e-6,
    )

    invariant_contrastive = _masked_multi_positive_nce_from_normalized(
        normalized_features=invariant_features,
        positive_mask=invariant_positive,
        negative_mask=~same_sample,
        temperature=temperature,
    )
    dependent_contrastive = _masked_multi_positive_nce_from_normalized(
        normalized_features=dependent_features,
        positive_mask=dependent_positive,
        negative_mask=~same_view,
        temperature=temperature,
    )

    invariant_pair_distance = (
        invariant_features[:, None, :] - invariant_features[None, :, :]
    ).abs().mean(dim=-1)
    dependent_pair_distance = (
        dependent_features[:, None, :] - dependent_features[None, :, :]
    ).abs().mean(dim=-1)
    invariant_weights = invariant_positive.to(dtype=invariant_pair_distance.dtype)
    dependent_weights = dependent_positive.to(dtype=dependent_pair_distance.dtype)
    invariant_consistency = (
        invariant_pair_distance * invariant_weights
    ).sum() / invariant_weights.sum().clamp_min(1.0)
    dependent_consistency = (
        dependent_pair_distance * dependent_weights
    ).sum() / dependent_weights.sum().clamp_min(1.0)
    return (
        invariant_contrastive,
        dependent_contrastive,
        invariant_consistency,
        dependent_consistency,
    )
