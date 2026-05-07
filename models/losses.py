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
