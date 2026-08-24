from __future__ import annotations

import torch
import torch.nn.functional as F


def _validate_rgb(
    predicted: torch.Tensor,
    target: torch.Tensor,
    validity: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if predicted.shape != target.shape or predicted.shape[-3] != 3:
        raise ValueError("Predicted and target RGB must match and end in (3,H,W).")
    expected_mask = (*predicted.shape[:-3], 1, *predicted.shape[-2:])
    if validity.shape != expected_mask:
        raise ValueError(
            f"Expected geometric image-validity mask {expected_mask}, got {tuple(validity.shape)}."
        )
    return (
        predicted.float(),
        target.detach().float(),
        validity.detach().float().clamp(0.0, 1.0),
    )


def _ssim_map(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    x = predicted.reshape(-1, 3, *predicted.shape[-2:])
    y = target.reshape_as(x)
    kernel = 11
    padding = kernel // 2
    mean_x = F.avg_pool2d(x, kernel, stride=1, padding=padding)
    mean_y = F.avg_pool2d(y, kernel, stride=1, padding=padding)
    variance_x = (
        F.avg_pool2d(x.square(), kernel, stride=1, padding=padding) - mean_x.square()
    )
    variance_y = (
        F.avg_pool2d(y.square(), kernel, stride=1, padding=padding) - mean_y.square()
    )
    covariance = (
        F.avg_pool2d(x * y, kernel, stride=1, padding=padding) - mean_x * mean_y
    )
    c1, c2 = 0.01**2, 0.03**2
    ssim = ((2.0 * mean_x * mean_y + c1) * (2.0 * covariance + c2)) / (
        (mean_x.square() + mean_y.square() + c1) * (variance_x + variance_y + c2)
    ).clamp_min(1.0e-8)
    return (
        ssim.clamp(-1.0, 1.0)
        .mean(dim=1, keepdim=True)
        .view(*predicted.shape[:-3], 1, *predicted.shape[-2:])
    )


def masked_rgb_reconstruction_losses(
    predicted: torch.Tensor,
    target: torch.Tensor,
    image_validity: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    prediction, teacher, mask = _validate_rgb(predicted, target, image_validity)
    denominator = mask.sum().clamp_min(1.0)
    l1_map = (prediction - teacher).abs().mean(dim=-3, keepdim=True)
    l1 = (l1_map * mask).sum() / denominator
    dssim_map = 0.5 * (1.0 - _ssim_map(prediction, teacher))
    dssim = (dssim_map * mask).sum() / denominator
    metrics = {
        "rgb_l1": l1.detach(),
        "dssim": dssim.detach(),
        "image_valid_fraction": mask.mean().detach(),
    }
    return l1, dssim, metrics
