from __future__ import annotations

import numpy as np

from preprocessing.see3d.metrics import depth_metrics, rgb_metrics
from preprocessing.see3d.official import letterbox_droid_image, unletterbox_droid_image


def test_see3d_letterbox_preserves_droid_aspect_ratio() -> None:
    image = np.full((180, 320, 3), 127, np.uint8)
    boxed, padding = letterbox_droid_image(image)
    assert boxed.shape == (512, 512, 3)
    assert padding == (0, 0, 112, 112)
    restored = unletterbox_droid_image(boxed, padding)
    assert restored.shape == image.shape
    assert np.max(np.abs(restored.astype(int) - image.astype(int))) == 0


def test_see3d_numeric_metrics() -> None:
    image = np.full((16, 16, 3), 100, np.uint8)
    assert rgb_metrics(image, image)["psnr"] > 100.0
    depth = np.ones((16, 16), np.float32)
    result = depth_metrics(depth, depth, np.ones_like(depth, dtype=bool))
    assert result == {"abs_rel": 0.0, "rmse": 0.0}
