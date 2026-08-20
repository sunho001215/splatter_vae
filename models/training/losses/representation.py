import torch
import torch.nn.functional as F

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


def compute_view_structured_invariant_losses(
    s_inv_by_view: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Invariant contrastive and normalized-L1 consistency from one encoder pass.

    Positives are the same state under different views; negatives are different
    batch/state indices. The deleted dependent representation is not emulated.
    """
    if temperature <= 0.0:
        raise ValueError("temperature must be > 0.")
    if s_inv_by_view.dim() != 3:
        raise ValueError(
            f"Expected s_inv_by_view as (B,A,D), got {tuple(s_inv_by_view.shape)}."
        )
    batch, views = s_inv_by_view.shape[:2]
    num_items = batch * views
    device = s_inv_by_view.device
    sample_ids = torch.arange(batch, device=device).repeat_interleave(views)
    diagonal = torch.eye(num_items, device=device, dtype=torch.bool)
    same_sample = sample_ids[:, None] == sample_ids[None, :]
    invariant_positive = same_sample & ~diagonal
    invariant_features = F.normalize(
        torch.nan_to_num(
            s_inv_by_view.reshape(num_items, -1), nan=0.0, posinf=0.0, neginf=0.0
        ),
        dim=-1,
        eps=1e-6,
    )
    invariant_contrastive = _masked_multi_positive_nce_from_normalized(
        normalized_features=invariant_features,
        positive_mask=invariant_positive,
        negative_mask=~same_sample,
        temperature=temperature,
    )
    invariant_grid = invariant_features.view(batch, views, -1)
    pair_distance = (
        invariant_grid[:, :, None, :] - invariant_grid[:, None, :, :]
    ).abs().mean(dim=-1)
    pair_mask = ~torch.eye(views, device=device, dtype=torch.bool)[None]
    weights = pair_mask.to(pair_distance.dtype)
    invariant_consistency = (
        pair_distance * weights
    ).sum() / weights.expand(batch, -1, -1).sum().clamp_min(1.0)
    return invariant_contrastive, invariant_consistency
