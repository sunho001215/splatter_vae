"""Actual installed fused-SSIM CUDA loss checks, without any source/JIT build."""

import torch

from s4d.losses.rgb import _ssim_map, masked_ssim, pixel_weights, rgb_loss


def test_native_ssim_perfect_image_and_weighted_dssim():
    image = torch.rand(2, 3, 32, 32, device="cuda")
    weights = pixel_weights(torch.rand(2, 1, 32, 32, device="cuda"), 1)
    values = _ssim_map(image, image)
    assert values.shape == image.shape and torch.isfinite(values).all()
    torch.testing.assert_close(values, torch.ones_like(values), atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(rgb_loss(image, image, weights), torch.zeros(2, device="cuda"), atol=1e-5, rtol=0)
    mask = torch.ones_like(weights, dtype=torch.bool)
    torch.testing.assert_close(masked_ssim(image, image, mask), torch.ones(2, device="cuda"), atol=2e-5, rtol=2e-5)


def test_native_ssim_nonperfect_images_have_finite_nonzero_gradients():
    image = torch.rand(2, 3, 32, 32, device="cuda", requires_grad=True)
    target = torch.rand_like(image)
    weights = torch.ones_like(image[:, :1])
    actual = rgb_loss(image, target, weights)
    expected = (image - target).abs().mean((1, 2, 3)) + 0.2 * (1 - _ssim_map(image, target)).mean((1, 2, 3)) / 2
    torch.testing.assert_close(actual, expected)
    assert torch.isfinite(actual).all() and (actual > 0).all()
    actual.sum().backward()
    assert torch.isfinite(image.grad).all() and image.grad.abs().sum() > 0
