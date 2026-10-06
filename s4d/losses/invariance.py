"""Cross-view invariance: multi-positive InfoNCE (DDP-safe negatives) and per-slot cosine consistency."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from s4d.train.ddp import all_gather_with_grad, rank, world_size


def multi_positive_info_nce(
    slots_mean: torch.Tensor, temperature: float = 0.1, *, distributed: bool = True
) -> tuple[torch.Tensor, dict]:
    """slots_mean (B,V,D): positives are other views of the same sample, negatives other samples (all ranks)."""
    B, V, D = slots_mean.shape
    local = F.normalize(slots_mean.float().reshape(B * V, D), dim=-1)
    if V < 2:
        zero = local.sum() * 0.0
        return zero, {"positive_cosine": zero.detach(), "negative_cosine": zero.detach()}
    global_feats = all_gather_with_grad(local) if distributed else local
    current_rank = rank() if distributed else 0
    world = world_size() if distributed else 1
    n_local = B * V
    offset = current_rank * n_local
    local_ids = torch.arange(n_local, device=local.device)
    sample_local = local_ids // V + current_rank * B
    sample_global = torch.arange(world * n_local, device=local.device) // V
    same_sample = sample_local[:, None] == sample_global[None, :]
    self_mask = torch.zeros_like(same_sample)
    self_mask[local_ids, offset + local_ids] = True
    positive = same_sample & ~self_mask
    denominator = ~self_mask
    logits = local @ global_feats.t() / temperature
    neg_inf = torch.finfo(logits.dtype).min
    log_pos = torch.logsumexp(logits.masked_fill(~positive, neg_inf), dim=1)
    log_den = torch.logsumexp(logits.masked_fill(~denominator, neg_inf), dim=1)
    loss = (log_den - log_pos).mean()
    with torch.no_grad():
        cos = local @ global_feats.t()
        pos_cos = cos[positive].mean()
        neg_mask = ~same_sample
        neg_cos = cos[neg_mask].mean() if neg_mask.any() else cos.new_zeros(())
    return loss, {"positive_cosine": pos_cos, "negative_cosine": neg_cos}


def slot_consistency(slots: torch.Tensor) -> torch.Tensor:
    """Mean (1 - cosine) between the same slot seen from different views. slots (B,V,K,D)."""
    B, V, K, D = slots.shape
    if V < 2:
        return slots.sum() * 0.0
    s = F.normalize(slots.float(), dim=-1)
    sim = torch.einsum("bvkd,bwkd->bvwk", s, s)
    off = ~torch.eye(V, dtype=torch.bool, device=slots.device)
    return (1.0 - sim[:, off]).mean()


def state_statistics(slots_mean: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-dimension std and effective rank (exp entropy of normalised singular values) of (N,D) states."""
    x = slots_mean.detach().float().reshape(-1, slots_mean.shape[-1])
    x = x - x.mean(0, keepdim=True)
    std = x.std(0, unbiased=False)
    if x.shape[0] < 2:
        return {"state_std_mean": std.mean(), "effective_rank": torch.zeros((), device=x.device)}
    s = torch.linalg.svdvals(x)
    p = s / s.sum().clamp_min(1e-12)
    erank = torch.exp(-(p * torch.log(p.clamp_min(1e-12))).sum())
    return {"state_std_mean": std.mean(), "state_std_min": std.min(), "effective_rank": erank}
