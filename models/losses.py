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
    valid_images = global_count >= 2
    global_loss = global_per_image.masked_select(valid_images).mean() if bool(valid_images.any()) else zero

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

    pred_patches = F.unfold(predicted_2d, kernel_size=size, stride=size).transpose(1, 2)
    target_patches = F.unfold(target_2d, kernel_size=size, stride=size).transpose(1, 2)
    valid_patches = F.unfold(valid_2d.to(dtype=predicted.dtype), kernel_size=size, stride=size)
    valid_patches = valid_patches.transpose(1, 2) > 0.5

    pred_local, local_count, _ = _masked_standardize(pred_patches, valid_patches)
    target_local, _, target_variance = _masked_standardize(target_patches, valid_patches)
    valid_patch = (local_count >= min_pixels) & (target_variance > 1.0e-6)
    local_penalty = F.smooth_l1_loss(pred_local, target_local, beta=1.0, reduction="none")
    local_per_patch = (local_penalty * valid_patches).sum(dim=-1) / local_count.clamp_min(1.0)
    local_loss = local_per_patch.masked_select(valid_patch).mean() if bool(valid_patch.any()) else zero
    return global_loss, local_loss


def masked_multi_positive_nce(
    features: torch.Tensor,
    positive_mask: torch.Tensor,
    negative_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Multi-positive InfoNCE with caller-provided positive/negative masks."""
    if temperature <= 0.0:
        raise ValueError("temperature must be > 0.")

    features = F.normalize(
        torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0),
        dim=-1,
        eps=1e-6,
    )
    logits = (features @ features.t()) / float(temperature)

    positive_mask = positive_mask.to(device=features.device, dtype=torch.bool)
    negative_mask = negative_mask.to(device=features.device, dtype=torch.bool)
    denominator_mask = positive_mask | negative_mask

    valid_queries = positive_mask.any(dim=1) & denominator_mask.any(dim=1)
    if not bool(valid_queries.any()):
        return features.new_zeros(())

    neg_inf = torch.finfo(logits.dtype).min
    positive_logits = logits.masked_fill(~positive_mask, neg_inf)
    denominator_logits = logits.masked_fill(~denominator_mask, neg_inf)

    log_positive = torch.logsumexp(positive_logits[valid_queries], dim=1)
    log_denominator = torch.logsumexp(denominator_logits[valid_queries], dim=1)
    return -(log_positive - log_denominator).mean()


def compute_state_consistency_loss(s_inv_a: torch.Tensor, s_inv_b: torch.Tensor) -> torch.Tensor:
    """Align two masked/view-augmented encodings of the same sequence."""
    if s_inv_a.shape != s_inv_b.shape or s_inv_a.dim() != 2:
        raise ValueError(f"Expected matching state vectors (B,D), got {tuple(s_inv_a.shape)} and {tuple(s_inv_b.shape)}.")
    a = F.normalize(torch.nan_to_num(s_inv_a, nan=0.0, posinf=0.0, neginf=0.0), dim=-1, eps=1e-6)
    b = F.normalize(torch.nan_to_num(s_inv_b, nan=0.0, posinf=0.0, neginf=0.0), dim=-1, eps=1e-6)
    return (a - b).abs().mean()


def compute_dependent_view_consistency_loss(z_dep_a: torch.Tensor, z_dep_b: torch.Tensor) -> torch.Tensor:
    """Weakly align first-timestep dependent anchors under different mask samples."""
    if z_dep_a.shape != z_dep_b.shape or z_dep_a.dim() != 3:
        raise ValueError(
            f"Expected matching dependent anchors (B,A,D), got {tuple(z_dep_a.shape)} and {tuple(z_dep_b.shape)}."
        )
    a = F.normalize(torch.nan_to_num(z_dep_a, nan=0.0, posinf=0.0, neginf=0.0), dim=-1, eps=1e-6)
    b = F.normalize(torch.nan_to_num(z_dep_b, nan=0.0, posinf=0.0, neginf=0.0), dim=-1, eps=1e-6)
    return (a - b).abs().mean()


def compute_view_structured_contrastive_losses(
    s_inv_by_view: torch.Tensor,
    z_dep_by_view: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Multi-positive contrastive losses over ``B x A`` encoded view items.

    Invariant positives are different viewpoints of the same batch sample.
    Dependent positives are the same viewpoint across different batch samples.
    Everything outside the positive group is treated as a negative.
    """
    if s_inv_by_view.dim() != 3 or z_dep_by_view.dim() != 3:
        raise ValueError(
            f"Expected s_inv_by_view and z_dep_by_view as (B,A,D), got "
            f"{tuple(s_inv_by_view.shape)} and {tuple(z_dep_by_view.shape)}."
        )
    if s_inv_by_view.shape[:2] != z_dep_by_view.shape[:2]:
        raise ValueError(
            f"Invariant/dependent view grids must share (B,A), got "
            f"{tuple(s_inv_by_view.shape[:2])} and {tuple(z_dep_by_view.shape[:2])}."
        )

    bsz, num_views = s_inv_by_view.shape[:2]
    device = s_inv_by_view.device
    num_items = bsz * num_views
    sample_ids = torch.arange(bsz, device=device).repeat_interleave(num_views)
    view_ids = torch.arange(num_views, device=device).repeat(bsz)
    eye = torch.eye(num_items, device=device, dtype=torch.bool)

    same_sample = sample_ids[:, None] == sample_ids[None, :]
    same_view = view_ids[:, None] == view_ids[None, :]

    inv_features = s_inv_by_view.reshape(num_items, -1)
    dep_features = z_dep_by_view.reshape(num_items, -1)
    inv_loss = masked_multi_positive_nce(
        features=inv_features,
        positive_mask=same_sample & ~eye,
        negative_mask=~same_sample,
        temperature=temperature,
    )
    dep_loss = masked_multi_positive_nce(
        features=dep_features,
        positive_mask=same_view & ~eye,
        negative_mask=~same_view,
        temperature=temperature,
    )
    return inv_loss, dep_loss
