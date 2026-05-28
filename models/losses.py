import torch
import torch.nn.functional as F

from fused_ssim import fused_ssim


def infonce_loss(query, positive_keys, negative_keys=None, temperature=0.1, negative_mode='unpaired'):
    """
    Compute InfoNCE loss between queries and positives, with optional negatives.

    L2-normalizes inputs and uses dot-product similarities scaled by 1/temperature.
    negative_mode: 'unpaired' ((M,D) negatives, uses batch-wise negatives),
                   'paired' ((B,M,D) negatives per-query),
                   'mixed' (both; average of batch and paired losses).

    Args:
        query (B,D), positive_keys (B,D), negative_keys: None or (M,D) or (B,M,D).
    Returns:
        Scalar loss tensor.
    Raises:
        ValueError on invalid input shapes.
    """
    if temperature <= 0.0:
        raise ValueError("temperature must be > 0.")

    # Validate input dimensions
    if query.dim() != 2 or positive_keys.dim() != 2:
        raise ValueError("query/positive must be 2D [B, D]")
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("negative_keys must be (M, D) when negative_mode='unpaired'")
        if negative_mode in ('paired', 'mixed') and negative_keys.dim() != 3:
            raise ValueError("negative_keys must be (B, M, D) when negative_mode in ['paired','mixed']")

    # Defensive sanitization keeps training alive when upstream tensors briefly explode.
    query = torch.nan_to_num(query, nan=0.0, posinf=0.0, neginf=0.0)
    positive_keys = torch.nan_to_num(positive_keys, nan=0.0, posinf=0.0, neginf=0.0)

    # L2 normalize the query and keys
    q = F.normalize(query, dim=-1, eps=1e-6)
    kpos = F.normalize(positive_keys, dim=-1, eps=1e-6)
    if negative_keys is not None:
        negative_keys = torch.nan_to_num(negative_keys, nan=0.0, posinf=0.0, neginf=0.0)
        knegs = F.normalize(negative_keys, dim=-1, eps=1e-6)

    if negative_mode == 'mixed':
        # Logits for all pairs in the batch
        logits_full = q @ kpos.t()
        labels_full = torch.arange(q.size(0), device=q.device)
        loss_full = F.cross_entropy(logits_full / temperature, labels_full, reduction='mean')

        # Logits for explicitly paired negatives
        pos_logit = (q * kpos).sum(dim=1, keepdim=True)      # (B, 1)
        # q: (B, D), knegs: (B, M, D) -> (B, M)
        neg_logits = torch.einsum('bd,bmd->bm', q, knegs)
        logits_mix = torch.cat([pos_logit, neg_logits], dim=1)   # (B, 1+M)
        labels_mix = torch.zeros(q.size(0), dtype=torch.long, device=q.device)
        loss_mix = F.cross_entropy(logits_mix / temperature, labels_mix, reduction='mean')

        return 0.5 * (loss_full + loss_mix)

    elif negative_keys is not None and negative_mode == 'paired':
        # Paired negatives
        pos_logit = (q * kpos).sum(dim=1, keepdim=True)      # (B, 1)
        neg_logits = torch.einsum('bd,bmd->bm', q, knegs)    # (B, M)
        logits = torch.cat([pos_logit, neg_logits], dim=1)   # (B, 1+M)
        labels = torch.zeros(q.size(0), dtype=torch.long, device=q.device)
        return F.cross_entropy(logits / temperature, labels, reduction='mean')

    else:
        # Basic case with no negatives
        logits = q @ kpos.t()                                # (B, B)
        labels = torch.arange(q.size(0), device=q.device)
        return F.cross_entropy(logits / temperature, labels, reduction='mean')


def compute_reconstruction_loss(
    predicted: torch.Tensor,
    ground_truth: torch.Tensor,
    ssim_weight: float = 0.2,
):
    """
    Compute combined MSE + SSIM reconstruction loss.

    predicted, ground_truth: (B,3,H,W), values in [0,1]
    ssim_weight: weight for SSIM term in [0,1]
    """
    mse_loss = F.mse_loss(predicted, ground_truth)

    if ssim_weight <= 0.0:
        return mse_loss

    ssim_map = fused_ssim(predicted, ground_truth)  # (B,H,W)
    ssim_loss = 1.0 - ssim_map.mean()

    total_loss = (1 - ssim_weight) * mse_loss + ssim_weight * ssim_loss
    return total_loss


def flatten_latent_tokens(z: torch.Tensor) -> torch.Tensor:
    """Flatten a token tensor to ``(num_items, feature_dim)`` with NaN guards."""
    z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    return z.reshape(z.shape[0], -1)


def compute_latent_consistency_loss(z: torch.Tensor, mode: str) -> torch.Tensor:
    """ReViWo-style mean-alignment loss over normalized latent features.

    Args:
        z: Latent tensor with shape ``(B, A, N_tokens, D)``.
        mode: ``"state"`` aligns all camera views from the same state, used for
            view-invariant features. ``"view"`` aligns the same camera index
            across different batch states, used for view-dependent features.
    """
    bsz, num_views = z.shape[:2]
    features = flatten_latent_tokens(z.reshape(bsz * num_views, *z.shape[2:]))
    features = F.normalize(features, dim=-1, eps=1e-6).reshape(bsz, num_views, -1)

    if mode == "state":
        return (features.mean(dim=1, keepdim=True) - features).abs().mean()
    if mode == "view":
        return (features.mean(dim=0, keepdim=True) - features).abs().mean()
    raise ValueError(f"Unknown consistency mode: {mode}")


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


def compute_all_camera_contrastive_losses(
    z_inv: torch.Tensor,
    z_dep: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute ReViWo-style all-camera invariant and view-dependent losses.

    ``z_inv`` and ``z_dep`` have shape ``(B, A, N_tokens, D)``.  Invariant
    positives are views from the same state.  Dependent positives are the same
    camera index from other states, matching ReViWo's view branch objective.
    """
    bsz, num_views = z_inv.shape[:2]
    device = z_inv.device

    state_ids = torch.arange(bsz, device=device).repeat_interleave(num_views)
    view_ids = torch.arange(num_views, device=device).repeat(bsz)
    eye = torch.eye(bsz * num_views, device=device, dtype=torch.bool)

    same_state = state_ids[:, None] == state_ids[None, :]
    same_view = view_ids[:, None] == view_ids[None, :]

    inv_features = flatten_latent_tokens(z_inv.reshape(bsz * num_views, *z_inv.shape[2:]))
    inv_contrastive_loss = masked_multi_positive_nce(
        features=inv_features,
        positive_mask=same_state & ~eye,
        negative_mask=~same_state,
        temperature=temperature,
    )

    dep_features = flatten_latent_tokens(z_dep.reshape(bsz * num_views, *z_dep.shape[2:]))
    dep_contrastive_loss = masked_multi_positive_nce(
        features=dep_features,
        positive_mask=same_view & ~same_state,
        negative_mask=~same_view,
        temperature=temperature,
    )

    return inv_contrastive_loss, dep_contrastive_loss


def compute_batched_reconstruction_losses(
    predicted: torch.Tensor,
    ground_truth: torch.Tensor,
    ssim_weight: float = 0.0,
) -> torch.Tensor:
    """Compute one reconstruction loss per leading variant.

    Args:
        predicted: ``(V, B, T, 3, H, W)`` rendered target-only images.
        ground_truth: ``(V, B, T, 3, H, W)`` matching target images.
        ssim_weight: if positive, fall back to per-variant SSIM calls; otherwise
            use a vectorized per-image MSE path.

    Returns:
        ``(V,)`` tensor, one loss for each reconstruction variant.
    """
    if predicted.shape != ground_truth.shape:
        raise ValueError(
            f"predicted and ground_truth must have the same shape, got "
            f"{tuple(predicted.shape)} and {tuple(ground_truth.shape)}."
        )
    if predicted.dim() != 6:
        raise ValueError(f"Expected (V,B,T,3,H,W), got {tuple(predicted.shape)}.")

    num_variants, bsz, num_targets = predicted.shape[:3]
    if ssim_weight <= 0.0:
        mse_per_image = F.mse_loss(predicted, ground_truth, reduction="none").mean(dim=(3, 4, 5))
        return mse_per_image.mean(dim=(1, 2))

    losses = []
    for variant_idx in range(num_variants):
        losses.append(
            compute_reconstruction_loss(
                predicted[variant_idx].reshape(bsz * num_targets, *predicted.shape[3:]),
                ground_truth[variant_idx].reshape(bsz * num_targets, *ground_truth.shape[3:]),
                ssim_weight=ssim_weight,
            )
        )
    return torch.stack(losses, dim=0)
