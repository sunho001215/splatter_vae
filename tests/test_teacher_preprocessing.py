from __future__ import annotations

import numpy as np

from preprocessing.xlens.official import pad_pinhole_scene_to_patch_multiple


def test_xlens_symmetric_patch_padding_updates_intrinsics() -> None:
    images = [np.zeros((180, 320, 3), np.uint8) for _ in range(2)]
    K = np.array([[200.0, 0.0, 160.0], [0.0, 200.0, 90.0], [0.0, 0.0, 1.0]], np.float32)
    padded, intrinsics, padding = pad_pinhole_scene_to_patch_multiple(images, [K, K])
    assert padding == (1, 1, 1, 1)
    assert padded[0].shape == (182, 322, 3)
    assert intrinsics[0][0, 2] == 161.0
    assert intrinsics[0][1, 2] == 91.0
