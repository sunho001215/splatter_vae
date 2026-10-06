from __future__ import annotations

import torch


def _validate_depth_inputs(
    predicted: torch.Tensor,
    target: torch.Tensor,
    validity: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if predicted.shape != target.shape or predicted.shape != validity.shape:
        raise ValueError("Predicted depth, target depth, and validity must align.")
    if predicted.dim() < 4 or predicted.shape[-3] != 1:
        raise ValueError("Depth inputs must have one channel and end in (1,H,W).")
    prediction = predicted.float()
    teacher = target.detach().float()
    valid = (
        validity.detach().bool()
        & torch.isfinite(prediction)
        & torch.isfinite(teacher)
        & (prediction > 0.0)
        & (teacher > 0.0)
    )
    return prediction, teacher, valid


def metric_depth_l1(
    predicted: torch.Tensor,
    target: torch.Tensor,
    validity: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    prediction, teacher, valid = _validate_depth_inputs(predicted, target, validity)
    weights = valid.to(prediction.dtype)
    penalty = (prediction - teacher).abs()
    loss = (penalty * weights).sum() / weights.sum().clamp_min(1.0)
    return loss, {
        "metric_depth_mae": loss.detach(),
        "depth_valid_fraction": valid.float().mean().detach(),
    }


def scale_invariant_log_depth_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    validity: torch.Tensor,
    *,
    mean_weight: float = 1.0,
    epsilon: float = 1.0e-6,
) -> torch.Tensor:
    """Validity-masked SI log-depth, reduced independently per rendered view."""

    if not 0.0 <= float(mean_weight) <= 1.0:
        raise ValueError("mean_weight must lie in [0,1].")
    prediction, teacher, valid = _validate_depth_inputs(predicted, target, validity)
    rows = prediction.reshape(-1, prediction.shape[-2] * prediction.shape[-1])
    targets = teacher.reshape_as(rows)
    row_weights = valid.reshape_as(rows).to(rows.dtype)
    difference = torch.log(rows.clamp_min(float(epsilon))) - torch.log(
        targets.clamp_min(float(epsilon))
    )
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
