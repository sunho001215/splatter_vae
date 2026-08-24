from __future__ import annotations

import numpy as np

from preprocessing.see3d.geometry import (
    fuse_two_warps,
    interpolate_camera_pose,
    interpolate_intrinsics,
    warp_rgbd_to_camera,
)


def test_pose_interpolation_uses_linear_translation_and_slerp() -> None:
    a = np.eye(4)
    b = np.eye(4)
    b[:3, :3] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    b[:3, 3] = (2.0, 0.0, 0.0)
    middle = interpolate_camera_pose(a, b, 0.5)
    np.testing.assert_allclose(middle[:3, 3], (1.0, 0.0, 0.0))
    np.testing.assert_allclose(
        middle[:3, :3].T @ middle[:3, :3], np.eye(3), atol=1.0e-8
    )
    assert np.linalg.det(middle[:3, :3]) > 0.999


def test_identity_rgbd_warp_and_fusion() -> None:
    height, width = 4, 5
    rgb = np.arange(height * width * 3, dtype=np.uint8).reshape(height, width, 3)
    depth = np.ones((height, width), np.float32)
    confidence = np.ones_like(depth)
    K = np.array([[10.0, 0.0, 2.5], [0.0, 10.0, 2.0], [0.0, 0.0, 1.0]])
    warp = warp_rgbd_to_camera(
        rgb, depth, confidence, depth > 0, K, np.eye(4), K, np.eye(4)
    )
    assert warp["validity"].all()
    np.testing.assert_allclose(warp["rgb"], rgb)
    fused = fuse_two_warps(warp, warp)
    assert fused["overlap"].all()
    assert np.all(fused["source_camera_ids"] == 2)
    np.testing.assert_allclose(fused["rgb"], rgb)


def test_virtual_intrinsics_do_not_invent_focal_length() -> None:
    a = np.diag((200.0, 210.0, 1.0))
    b = np.diag((220.0, 230.0, 1.0))
    middle = interpolate_intrinsics(a, b, 0.5)
    assert middle[0, 0] == 210.0
    assert middle[1, 1] == 220.0
