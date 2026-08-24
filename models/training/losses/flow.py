from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F


def compute_optical_flow_loss(
    predicted_flows: torch.Tensor,
    target_flows: torch.Tensor,
    rendered_coverage: torch.Tensor,
    predicted_validity: torch.Tensor,
    target_validity: torch.Tensor,
    target_confidence: torch.Tensor | None = None,
    *,
    pair_weights: Sequence[float] = (0.4, 0.4, 0.2),
    alpha_threshold: float = 0.01,
    smooth_l1_beta: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compare rendered Gaussian flow to WAFT without semantic masks."""
    if predicted_flows.shape != target_flows.shape:
        raise ValueError("Predicted and WAFT flow tensors must have identical shapes.")
    if (
        predicted_flows.dim() != 6
        or predicted_flows.shape[1] != 3
        or predicted_flows.shape[3] != 2
    ):
        raise ValueError("Flow must have shape (B,3,A,2,H,W) for pairs 01, 12, and 02.")
    auxiliary_shape = (*predicted_flows.shape[:3], 1, *predicted_flows.shape[-2:])
    for name, value in (
        ("rendered_coverage", rendered_coverage),
        ("predicted_validity", predicted_validity),
        ("target_validity", target_validity),
    ):
        if value.shape != auxiliary_shape:
            raise ValueError(
                f"Expected {name} as {auxiliary_shape}, got {tuple(value.shape)}."
            )
    if target_confidence is not None and target_confidence.shape != auxiliary_shape:
        raise ValueError(
            f"Expected target_confidence as {auxiliary_shape}, got {tuple(target_confidence.shape)}."
        )
    if len(pair_weights) != 3 or sum(float(value) for value in pair_weights) <= 0.0:
        raise ValueError("Exactly three nonnegative flow-pair weights are required.")
    prediction = predicted_flows.float()
    teacher = target_flows.detach().float()
    coverage = rendered_coverage.detach().float().clamp(0.0, 1.0)
    valid = (
        predicted_validity.detach().bool()
        & target_validity.detach().bool()
        & torch.isfinite(prediction).all(dim=3, keepdim=True)
        & torch.isfinite(teacher).all(dim=3, keepdim=True)
        & (coverage > float(alpha_threshold))
    )
    teacher_confidence = (
        torch.ones_like(coverage)
        if target_confidence is None
        else target_confidence.detach().float().clamp(0.0, 1.0)
    )
    weights = coverage * teacher_confidence * valid.to(coverage.dtype)
    safe_prediction = torch.nan_to_num(prediction, nan=0.0, posinf=0.0, neginf=0.0)
    safe_teacher = torch.nan_to_num(teacher, nan=0.0, posinf=0.0, neginf=0.0)
    component = F.smooth_l1_loss(
        safe_prediction, safe_teacher, beta=float(smooth_l1_beta), reduction="none"
    ).mean(dim=3, keepdim=True)
    endpoint = (safe_prediction - safe_teacher).square().sum(dim=3, keepdim=True).sqrt()
    pair_losses = []
    metrics: dict[str, torch.Tensor] = {}
    for pair, name in enumerate(("01", "12", "02")):
        pair_weight = weights[:, pair]
        denominator = pair_weight.sum().clamp_min(1.0)
        pair_losses.append((component[:, pair] * pair_weight).sum() / denominator)
        metrics[f"flow_epe_{name}"] = (
            (endpoint[:, pair] * pair_weight).sum() / denominator
        ).detach()
    normalized_weights = prediction.new_tensor(
        tuple(float(value) for value in pair_weights)
    )
    normalized_weights = normalized_weights / normalized_weights.sum()
    loss = (torch.stack(pair_losses) * normalized_weights).sum()
    metrics["flow_valid_fraction"] = valid.float().mean().detach()
    metrics["flow_mean_confidence"] = (
        (teacher_confidence * valid).sum() / valid.float().sum().clamp_min(1.0)
    ).detach()
    return loss, metrics
