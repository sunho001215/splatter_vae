import torch
import torch.nn.functional as F


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
    # Validate input dimensions
    if query.dim() != 2 or positive_keys.dim() != 2:
        raise ValueError("query/positive must be 2D [B, D]")
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("negative_keys must be (M, D) when negative_mode='unpaired'")
        if negative_mode in ('paired', 'mixed') and negative_keys.dim() != 3:
            raise ValueError("negative_keys must be (B, M, D) when negative_mode in ['paired','mixed']")

    # L2 normalize the query and keys
    q = F.normalize(query, dim=-1)
    kpos = F.normalize(positive_keys, dim=-1)
    if negative_keys is not None:
        knegs = F.normalize(negative_keys, dim=-1)
    
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

def masked_mse(pred, target, mask):
    # pred, target: (B,3,H,W), mask: (B,1,H,W)
    diff2 = (pred - target) ** 2
    diff2 = diff2 * mask
    denom = mask.sum() * 3.0 + 1e-8  # 3 channels
    return diff2.sum() / denom