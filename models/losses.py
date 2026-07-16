import torch
import torch.nn.functional as F

from fused_ssim import fused_ssim


def compute_reconstruction_loss(
    predicted: torch.Tensor,
    ground_truth: torch.Tensor,
    ssim_weight: float = 0.2,
    loss_mask: torch.Tensor | None = None,
):
    """
    Compute combined L1 + SSIM reconstruction loss over a masked image region.

    predicted, ground_truth: (B,3,H,W), values in [0,1]
    ssim_weight: weight for SSIM term in [0,1]
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

    ssim_map = fused_ssim(predicted, ground_truth)  # (B,H,W)
    ssim_loss_map = 1.0 - ssim_map
    if mask is None:
        ssim_loss = ssim_loss_map.mean()
    else:
        mask_2d = mask[:, 0]
        ssim_loss = (ssim_loss_map * mask_2d).sum() / mask_2d.sum().clamp_min(1.0)

    total_loss = (1 - ssim_weight) * l1_loss + ssim_weight * ssim_loss
    return total_loss


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
