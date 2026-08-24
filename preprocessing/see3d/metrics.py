from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import uniform_filter


def rgb_metrics(
    predicted: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, float]:
    prediction = np.asarray(predicted, dtype=np.float64) / 255.0
    reference = np.asarray(target, dtype=np.float64) / 255.0
    valid = (
        np.ones(prediction.shape[:2], dtype=bool)
        if mask is None
        else np.asarray(mask, dtype=bool)
    )
    if not valid.any():
        return {"psnr": float("nan"), "ssim": float("nan")}
    mse = float(np.square(prediction - reference).mean(axis=-1)[valid].mean())
    psnr = -10.0 * np.log10(max(mse, 1.0e-12))
    mu_x = uniform_filter(prediction, size=(11, 11, 1), mode="reflect")
    mu_y = uniform_filter(reference, size=(11, 11, 1), mode="reflect")
    var_x = (
        uniform_filter(prediction * prediction, size=(11, 11, 1), mode="reflect")
        - mu_x**2
    )
    var_y = (
        uniform_filter(reference * reference, size=(11, 11, 1), mode="reflect")
        - mu_y**2
    )
    covariance = (
        uniform_filter(prediction * reference, size=(11, 11, 1), mode="reflect")
        - mu_x * mu_y
    )
    c1, c2 = 0.01**2, 0.03**2
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * covariance + c2)) / np.maximum(
        (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2), 1.0e-12
    )
    return {"psnr": float(psnr), "ssim": float(ssim_map.mean(axis=-1)[valid].mean())}


def depth_metrics(
    predicted: np.ndarray,
    target: np.ndarray,
    validity: np.ndarray,
) -> dict[str, float]:
    prediction = np.asarray(predicted, dtype=np.float64)
    reference = np.asarray(target, dtype=np.float64)
    valid = (
        np.asarray(validity, dtype=bool)
        & np.isfinite(prediction)
        & np.isfinite(reference)
        & (reference > 0)
    )
    if not valid.any():
        return {"abs_rel": float("nan"), "rmse": float("nan")}
    difference = prediction[valid] - reference[valid]
    return {
        "abs_rel": float((np.abs(difference) / reference[valid]).mean()),
        "rmse": float(np.sqrt(np.square(difference).mean())),
    }


class PerceptualMetrics:
    def __init__(self, *, device: str = "cuda:0", dinov2_repository: str | None = None):
        import lpips

        self.device = torch.device(device)
        self.lpips = lpips.LPIPS(net="alex").eval().to(self.device)
        if dinov2_repository:
            self.dino = torch.hub.load(
                dinov2_repository, "dinov2_vits14", source="local", pretrained=True
            )
        else:
            self.dino = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vits14", pretrained=True
            )
        self.dino.eval().to(self.device)

    @torch.inference_mode()
    def __call__(
        self,
        predicted: np.ndarray,
        target: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> dict[str, float]:
        prediction = (
            torch.from_numpy(np.asarray(predicted)).permute(2, 0, 1)[None].float()
            / 255.0
        )
        reference = (
            torch.from_numpy(np.asarray(target)).permute(2, 0, 1)[None].float() / 255.0
        )
        if mask is not None:
            valid = torch.from_numpy(np.asarray(mask, dtype=np.float32))[None, None]
            prediction = prediction * valid
            reference = reference * valid
        prediction = prediction.to(self.device)
        reference = reference.to(self.device)
        lpips_value = self.lpips(prediction * 2.0 - 1.0, reference * 2.0 - 1.0).mean()
        resized_prediction = F.interpolate(
            prediction, (224, 224), mode="bilinear", align_corners=False
        )
        resized_reference = F.interpolate(
            reference, (224, 224), mode="bilinear", align_corners=False
        )
        mean = prediction.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
        std = prediction.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
        feature_prediction = self.dino((resized_prediction - mean) / std)
        feature_reference = self.dino((resized_reference - mean) / std)
        similarity = F.cosine_similarity(feature_prediction, feature_reference).mean()
        return {
            "lpips": float(lpips_value.item()),
            "dino_similarity": float(similarity.item()),
        }
