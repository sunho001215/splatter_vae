"""Synthetic geometry checks isolate conventions before empirical camera checks."""

from __future__ import annotations

import math

import torch

from s4d.geometry import invert_se3, lift_depth, pixel_centers, project, quat_wxyz_to_matrix, rigid_body_displacement


def test_pixel_centers_are_half_integer_continuous_coordinates():
    uv = pixel_centers(2, 3, dtype=torch.float64)
    expected = torch.tensor(
        [[[0.5, 0.5], [1.5, 0.5], [2.5, 0.5]], [[0.5, 1.5], [1.5, 1.5], [2.5, 1.5]]], dtype=torch.float64
    )
    torch.testing.assert_close(uv, expected, atol=0, rtol=0)


def test_lift_project_roundtrip_with_batched_rotated_cameras():
    H, W = 5, 7
    K = torch.tensor([[8.0, 0.0, W / 2], [0.0, 11.0, H / 2], [0.0, 0.0, 1.0]], dtype=torch.float64)
    K = K.expand(2, 3, 3).clone()
    c2w = torch.eye(4, dtype=torch.float64).expand(2, 4, 4).clone()
    c2w[0, :3, :3] = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    c2w[0, :3, 3] = torch.tensor([0.4, -0.2, 0.7])
    c2w[1, :3, 3] = torch.tensor([-0.1, 0.6, -0.2])
    depth = torch.linspace(0.4, 2.0, 2 * H * W, dtype=torch.float64).reshape(2, H, W)
    xyz = lift_depth(depth, K, c2w)
    uv, projected_depth = project(xyz.flatten(1, 2), K, invert_se3(c2w))
    torch.testing.assert_close(
        uv, pixel_centers(H, W, dtype=torch.float64).flatten(0, 1).expand(2, -1, -1), atol=1e-12, rtol=1e-12
    )
    torch.testing.assert_close(projected_depth.reshape_as(depth), depth, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(
        invert_se3(c2w) @ c2w, torch.eye(4, dtype=torch.float64).expand(2, -1, -1), atol=1e-12, rtol=1e-12
    )


def test_wxyz_quaternion_rotation_is_correct_and_sign_invariant():
    q = torch.tensor([[math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]], dtype=torch.float64)
    expected = torch.tensor([[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]], dtype=torch.float64)
    torch.testing.assert_close(quat_wxyz_to_matrix(q), expected, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(quat_wxyz_to_matrix(-3 * q), expected, atol=1e-12, rtol=1e-12)


def test_rigid_lift_matches_known_body_rotation_and_translation():
    xyz = torch.tensor([[[0.2, 0.4, 0.6], [1.2, 0.0, 0.0], [1.0, 0.3, 0.1], [-0.2, 0.1, 0.7]]], dtype=torch.float64)
    ids = torch.tensor([[0, 1, 1, 0]])
    pa = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=torch.float64)
    pb = torch.tensor([[[0.0, 0.0, 0.0], [0.1, 0.5, -0.2]]], dtype=torch.float64)
    qa = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]], dtype=torch.float64)
    qb = qa.clone()
    qb[:, 1] = torch.tensor([math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)], dtype=torch.float64)
    displacement = rigid_body_displacement(xyz, ids, pa, qa, pb, qb)
    expected_target = xyz.clone()
    expected_target[0, 1] = torch.tensor([0.1, 0.7, -0.2], dtype=torch.float64)
    expected_target[0, 2] = torch.tensor([-0.2, 0.5, -0.1], dtype=torch.float64)
    torch.testing.assert_close(displacement, expected_target - xyz, atol=1e-12, rtol=1e-12)
    assert torch.count_nonzero(displacement[ids == 0]).item() == 0


def test_world_body_has_bitwise_exact_zero_motion_for_all_pixels():
    generator = torch.Generator().manual_seed(17)
    xyz = torch.randn(3, 20, 3, generator=generator)
    ids = torch.zeros(3, 20, dtype=torch.long)
    pa = torch.zeros(3, 2, 3)
    pb = pa.clone()
    pb[:, 1] = torch.tensor([0.5, -0.3, 0.2])
    qa = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(3, 2, 4).clone()
    qb = qa.clone()
    qb[:, 1] = torch.tensor([0.70710678, 0.70710678, 0.0, 0.0])
    displacement = rigid_body_displacement(xyz, ids, pa, qa, pb, qb)
    assert torch.equal(displacement, torch.zeros_like(xyz))
