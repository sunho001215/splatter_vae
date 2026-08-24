from __future__ import annotations

import torch


def _validate_depth_inputs(
    predicted: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor,
    validity: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if (
        predicted.shape != target.shape
        or predicted.shape != confidence.shape
        or predicted.shape != validity.shape
    ):
        raise ValueError(
            "Predicted depth, target depth, confidence, and validity must have identical shapes."
        )
    if predicted.dim() < 4 or predicted.shape[-3] != 1:
        raise ValueError("Depth inputs must have one channel and end in (1,H,W).")
    prediction = predicted.float()
    teacher = target.detach().float()
    teacher_confidence = confidence.detach().float()
    valid = validity.detach().bool()
    finite = (
        torch.isfinite(prediction)
        & torch.isfinite(teacher)
        & torch.isfinite(teacher_confidence)
    )
    valid = (
        valid
        & finite
        & (prediction > 0.0)
        & (teacher > 0.0)
        & (teacher_confidence > 0.0)
    )
    weights = torch.where(
        valid,
        torch.nan_to_num(teacher_confidence, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(
            0.0
        ),
        torch.zeros_like(teacher_confidence),
    )
    return prediction, teacher, weights, valid


def confidence_weighted_metric_depth_l1(
    predicted: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor,
    validity: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    prediction, teacher, weights, valid = _validate_depth_inputs(
        predicted, target, confidence, validity
    )
    penalty = (prediction - teacher).abs()
    loss = (penalty * weights).sum() / weights.sum().clamp_min(1.0)
    metrics = {
        "metric_depth_mae": loss.detach(),
        "depth_valid_fraction": valid.float().mean().detach(),
        "depth_mean_confidence": (
            weights.sum() / valid.float().sum().clamp_min(1.0)
        ).detach(),
    }
    return loss, metrics


def scale_invariant_log_depth_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor,
    validity: torch.Tensor,
    *,
    mean_weight: float = 1.0,
    epsilon: float = 1.0e-6,
) -> torch.Tensor:
    """Confidence-weighted SI log-depth, reduced independently per rendered view."""
    if not 0.0 <= float(mean_weight) <= 1.0:
        raise ValueError("mean_weight must lie in [0,1].")
    prediction, teacher, weights, _valid = _validate_depth_inputs(
        predicted, target, confidence, validity
    )
    rows = prediction.reshape(-1, prediction.shape[-2] * prediction.shape[-1])
    targets = teacher.reshape_as(rows)
    row_weights = weights.reshape_as(rows)
    safe_prediction = rows.clamp_min(float(epsilon))
    safe_target = targets.clamp_min(float(epsilon))
    difference = torch.log(safe_prediction) - torch.log(safe_target)
    denominator = row_weights.sum(dim=-1)
    mean = (difference * row_weights).sum(dim=-1) / denominator.clamp_min(1.0)
    second_moment = (difference.square() * row_weights).sum(
        dim=-1
    ) / denominator.clamp_min(1.0)
    per_render = torch.sqrt(
        (second_moment - float(mean_weight) * mean.square()).clamp_min(0.0)
        + float(epsilon)
    )
    render_weights = (denominator > 0).to(per_render.dtype)
    return (per_render * render_weights).sum() / render_weights.sum().clamp_min(1.0)
