from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F


class _AllGatherWithGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, local: torch.Tensor) -> torch.Tensor:
        if not dist.is_available() or not dist.is_initialized():
            ctx.world_size = 1
            ctx.rank = 0
            return local
        ctx.world_size = dist.get_world_size()
        ctx.rank = dist.get_rank()
        gathered = [torch.empty_like(local) for _ in range(ctx.world_size)]
        dist.all_gather(gathered, local.contiguous())
        return torch.cat(gathered, dim=0)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> tuple[torch.Tensor]:
        if ctx.world_size == 1:
            return (gradient,)
        local_gradient = gradient.chunk(ctx.world_size, dim=0)[ctx.rank].contiguous()
        dist.all_reduce(local_gradient, op=dist.ReduceOp.SUM)
        return (local_gradient,)


def autograd_safe_all_gather(features: torch.Tensor) -> torch.Tensor:
    return _AllGatherWithGradient.apply(features)


def _assert_equal_local_counts(local_count: int, device: torch.device) -> None:
    if not dist.is_available() or not dist.is_initialized():
        return
    count = torch.tensor([int(local_count)], device=device, dtype=torch.long)
    gathered = [torch.empty_like(count) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, count)
    values = [int(item.item()) for item in gathered]
    if len(set(values)) != 1:
        raise RuntimeError(
            f"Distributed contrastive batches must have equal size, got {values}."
        )


def cross_view_info_nce(
    projected_by_view: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Local paired queries against autograd-safe global DDP negatives."""
    if projected_by_view.dim() != 3 or projected_by_view.shape[1] != 2:
        raise ValueError("Projected positives must have shape (B,2,D).")
    if float(temperature) <= 0.0:
        raise ValueError("InfoNCE temperature must be positive.")
    local = F.normalize(projected_by_view.float(), dim=-1, eps=1.0e-6).flatten(0, 1)
    _assert_equal_local_counts(local.shape[0], local.device)
    global_features = autograd_safe_all_gather(local)
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    offset = rank * local.shape[0]
    local_indices = torch.arange(local.shape[0], device=local.device)
    self_indices = offset + local_indices
    positive_indices = offset + torch.bitwise_xor(
        local_indices, torch.ones_like(local_indices)
    )
    logits = (local @ global_features.t()) / float(temperature)
    logits.scatter_(1, self_indices[:, None], torch.finfo(logits.dtype).min)
    loss = F.cross_entropy(logits, positive_indices)
    cosine = local @ global_features.detach().t()
    positive_cosine = cosine.gather(1, positive_indices[:, None]).mean()
    negative_mask = torch.ones_like(cosine, dtype=torch.bool)
    negative_mask.scatter_(1, self_indices[:, None], False)
    negative_mask.scatter_(1, positive_indices[:, None], False)
    negative_cosine = (
        cosine.masked_select(negative_mask).mean()
        if negative_mask.any()
        else cosine.new_zeros(())
    )
    return loss, {
        "positive_cosine_similarity": positive_cosine.detach(),
        "negative_cosine_similarity": negative_cosine.detach(),
    }
