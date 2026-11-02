"""RGB reconstruction (L1 + D-SSIM), coverage loss, and PSNR metrics. Inputs are (R,C,H,W) render rows."""

from __future__ import annotations

import torch
import torch.nn.functional as F

SSIM_RADIUS = 5  # fused_ssim: 11 x 11 Gaussian window


def pixel_weights(score: torch.Tensor, lambda_dyn: float, observed: torch.Tensor | None = None) -> torch.Tensor:
    """w = 1 + lambda * score, normalised to mean 1 per image. score (R,1,H,W) in [0,1].

    ``observed`` (R,1,H,W) (square-crop padding, optional): w is 0 on non-observed pixels and the normalisation runs
    over the whole image, so a mean of ``loss * w`` is the weighted mean over observed pixels only."""
    w = 1.0 + float(lambda_dyn) * score.detach().float().clamp(0, 1)
    if observed is not None:
        w = w * observed.float()
        return w / w.mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
    return w / w.mean(dim=(1, 2, 3), keepdim=True)


def ssim_support(observed: torch.Tensor, radius: int = SSIM_RADIUS) -> torch.Tensor:
    """(R,1,H,W) -> float mask of the pixels whose whole SSIM window lies in observed pixels (the image border itself is
    not padding: windows are only cut by non-observed pixels)."""
    return -F.max_pool2d(-observed.float(), 2 * radius + 1, stride=1, padding=radius)


def _ssim_map(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    from fused_ssim import FusedSSIMMap  # noqa: PLC0415

    return FusedSSIMMap.apply(0.01**2, 0.03**2, pred.contiguous(), target.contiguous(), "same", True, 2)


def rgb_loss(
    pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor, ssim_weight: float = 0.2,
    ssim_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-render loss (R,): weighted L1 + ssim_weight * weighted D-SSIM (``ssim_weights``, default ``weights``: the
    square-crop path zeroes pixels whose SSIM window reaches padding)."""
    pred, target = pred.float(), target.float()
    l1 = (pred - target).abs().mean(1, keepdim=True)
    loss = (l1 * weights).mean(dim=(1, 2, 3))
    if ssim_weight > 0:
        dssim = (1.0 - _ssim_map(pred, target).mean(1, keepdim=True)) * 0.5
        loss = loss + ssim_weight * (dssim * (weights if ssim_weights is None else ssim_weights)).mean(dim=(1, 2, 3))
    return loss


def coverage_loss(alpha: torch.Tensor, valid: torch.Tensor, weights: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """-log(alpha) on pixels with a valid target depth, per render (R,)."""
    nll = -torch.log(alpha.float().clamp_min(eps)) * weights * valid.float()
    return nll.sum(dim=(1, 2, 3)) / valid.float().sum(dim=(1, 2, 3)).clamp_min(1.0)


def masked_psnr(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """PSNR per render over mask (R,1,H,W); NaN when the mask is empty."""
    se = ((pred.float() - target.float()) ** 2).mean(1, keepdim=True)
    count = mask.float().sum(dim=(1, 2, 3))
    mse = (se * mask.float()).sum(dim=(1, 2, 3)) / count.clamp_min(1.0)
    psnr = -10.0 * torch.log10(mse.clamp_min(1e-10))
    return torch.where(count > 0, psnr, torch.full_like(psnr, float("nan")))


def masked_ssim(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average SSIM map over each region, with NaN for absent regions."""
    values = _ssim_map(pred.float(), target.float()).mean(1, keepdim=True)
    count = mask.float().sum(dim=(1, 2, 3))
    result = (values * mask.float()).sum(dim=(1, 2, 3)) / count.clamp_min(1)
    return torch.where(count > 0, result, torch.full_like(result, float("nan")))
