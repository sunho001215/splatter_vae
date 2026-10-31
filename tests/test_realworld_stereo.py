"""Stereo conventions on synthetic rigs: disparity -> metric depth, flip-and-swap right reference, rectification,
and image/depth/intrinsics alignment under resize/crop. CPU numpy only; no model is involved."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from s4d.realworld.geometry import (
    PixelAffine,
    crop_affine,
    hflip_affine,
    make_T,
    project,
    rectify_map,
    rectify_points,
    resize_affine,
    stereo_rectify,
    transform_points,
    unproject,
    warp_affine,
)
from s4d.realworld.stereo import (
    block_match,
    depth_from_pair,
    disparity_to_depth,
    flip_and_swap_right_disparity,
    rescale_disparity,
    visible_in_other,
)
from s4d.realworld.synthetic import rectified_pair


def test_disparity_to_depth_scale_and_doffs():
    fx, B = 600.0, 0.055
    z = np.array([[0.4, 1.0], [2.5, 7.0]])
    d = fx * B / z
    depth, valid = disparity_to_depth(d, fx, B)
    assert valid.all() and np.allclose(depth, z, rtol=1e-6)
    depth, valid = disparity_to_depth(d + 3.0, fx, B, doffs=3.0)
    assert np.allclose(depth, z, rtol=1e-6)
    depth, valid = disparity_to_depth(np.array([[0.0, -1.0], [np.nan, 5.0]]), fx, B)
    assert valid.tolist() == [[False, False], [False, True]] and depth[0, 0] == 0
    with pytest.raises(ValueError):
        disparity_to_depth(d, fx, float("nan"))  # missing stereo geometry never gets an invented scale


def _interior(depth, margin=3):
    """Pixels away from depth edges and image borders (block matching is unreliable there)."""
    edge = np.zeros_like(depth, bool)
    for axis in (0, 1):
        jump = np.abs(np.diff(depth, axis=axis)) > 1e-6
        idx = np.nonzero(jump)
        for k in range(-margin, margin + 2):
            shifted = list(idx)
            shifted[axis] = np.clip(idx[axis] + k, 0, depth.shape[axis] - 1)
            edge[tuple(shifted)] = True
    edge[:margin] = edge[-margin:] = True
    edge[:, :16] = edge[:, -16:] = True
    return ~edge


def test_left_reference_depth_matches_ground_truth():
    left, right, depth_l, _, K = rectified_pair()
    res = depth_from_pair(left, right, K, K, 0.1, lambda a, b: block_match(a, b, 12), "left")
    m = _interior(depth_l) & res.valid
    assert m.sum() > 1000
    gt_disp = K[0, 0] * 0.1 / depth_l
    assert np.median(np.abs(res.disparity[m] - gt_disp[m])) < 0.15
    assert np.median(np.abs(res.depth[m] - depth_l[m]) / depth_l[m]) < 0.03


def test_right_reference_flip_and_swap_matches_ground_truth_and_naive_swap_fails():
    left, right, _, depth_r, K = rectified_pair()
    matcher = lambda a, b: block_match(a, b, 12)  # noqa: E731
    res = depth_from_pair(left, right, K, K, 0.1, matcher, "right")
    m = _interior(depth_r) & res.valid
    gt_disp = K[0, 0] * 0.1 / depth_r  # right-image grid, x_left - x_right > 0
    assert np.median(np.abs(res.disparity[m] - gt_disp[m])) < 0.15
    assert np.median(np.abs(res.depth[m] - depth_r[m]) / depth_r[m]) < 0.03
    naive = matcher(right, left)  # swapping without flipping asks for negative disparities
    assert np.median(np.abs(naive[m] - gt_disp[m])) > 1.0


def test_flip_and_swap_is_exact_for_an_exact_matcher():
    """With a matcher that is exact by construction, flip-and-swap returns exactly the right-grid disparity."""
    H, W = 6, 20
    d_right = np.tile(np.linspace(1, 4, W), (H, 1))

    def oracle(a, b):
        # Inputs are (flip(right), flip(left)); the left-reference disparity of that pair is flip(d_right).
        assert a.shape == (H, W)
        return d_right[:, ::-1]

    out = flip_and_swap_right_disparity(np.zeros((H, W)), np.zeros((H, W)), oracle)
    assert np.array_equal(out, d_right)


def test_visibility_in_other_image():
    d = np.full((1, 4), 1.5)
    assert visible_in_other(d, "left").tolist() == [[False, True, True, True]]  # x=.5 -> -1 < 0
    assert visible_in_other(d, "right").tolist() == [[True, True, True, False]]  # x=3.5 -> 5 > 4


def test_bouguet_rectification_aligns_rows_and_recovers_depth():
    rng = np.random.default_rng(0)
    hw = (120, 160)
    K1 = np.array([[150.0, 0, 81.0], [0, 152.0, 59.0], [0, 0, 1]])
    K2 = np.array([[149.0, 0, 79.0], [0, 151.0, 61.0], [0, 0, 1]])
    R = Rotation.from_euler("xyz", [0.01, -0.02, 0.015]).as_matrix()
    T_rl = make_T(R, R @ -np.array([0.05, 0.002, -0.001]))  # right camera centre at ~(+5 cm, 0, 0) of the left
    dist1, dist2 = (0.05, -0.02, 0.001, -0.0005, 0.0), (0.03, -0.01, 0.0, 0.0005, 0.0)
    rect = stereo_rectify(K1, K2, T_rl, hw)
    P = rng.uniform([-0.3, -0.2, 0.6], [0.3, 0.2, 2.0], size=(200, 3))  # left raw camera frame

    def raw_pixels(K, dist, Pc):
        from s4d.realworld.geometry import distort_normalized

        xy = distort_normalized(Pc[:, :2] / Pc[:, 2:], "opencv_plumb_bob", dist)
        return xy @ K[:2, :2].T + K[:2, 2]

    uv1 = raw_pixels(K1, dist1, P)
    uv2 = raw_pixels(K2, dist2, transform_points(T_rl, P))
    r1 = rectify_points(uv1, K1, "opencv_plumb_bob", dist1, rect.R1, rect.K_rect)
    r2 = rectify_points(uv2, K2, "opencv_plumb_bob", dist2, rect.R2, rect.K_rect)
    assert np.abs(r1[:, 1] - r2[:, 1]).max() < 1e-6
    disp = r1[:, 0] - r2[:, 0]
    z_rect = (P @ rect.R1.T)[:, 2]
    depth, valid = disparity_to_depth(disp, rect.K_rect[0, 0], rect.baseline_m)
    assert valid.all() and np.allclose(depth, z_rect, rtol=1e-6)
    # The dense remap is the inverse of the point map.
    maps = rectify_map(K1, "opencv_plumb_bob", dist1, rect.R1, rect.K_rect, hw)
    back = rectify_points(maps.reshape(-1, 2), K1, "opencv_plumb_bob", dist1, rect.R1, rect.K_rect)
    from s4d.realworld.geometry import pixel_centers

    assert np.abs(back - pixel_centers(*hw).reshape(-1, 2)).max() < 1e-6


def test_intrinsics_track_resize_crop_flip_chain():
    rng = np.random.default_rng(1)
    K = np.array([[300.0, 0, 161.3], [0, 310.0, 119.7], [0, 0, 1]])
    P = rng.uniform([-0.4, -0.3, 0.5], [0.4, 0.3, 2.0], size=(100, 3))
    chain = resize_affine((240, 320), (144, 256)).then(crop_affine(top=2, left=2)).then(hflip_affine(252))
    uv_a = chain.apply_uv(project(P, K)[0])
    uv_b, _ = project(P, chain.apply_K(K))
    assert np.allclose(uv_a, uv_b)
    assert np.allclose(chain.then(chain.inverse()).matrix(), np.eye(3))


def test_depth_and_image_alignment_after_resize_and_crop():
    """A tilted plane rendered at native resolution, then resized/cropped, matches a direct render with the tracked K."""
    K = np.array([[200.0, 0, 80.0], [0, 200.0, 60.0], [0, 0, 1]])
    n, c = np.array([0.2, -0.1, 1.0]), 1.2  # plane n . X = c

    def plane_depth(Kx, hw):
        from s4d.realworld.geometry import pixel_centers

        rays = unproject(pixel_centers(*hw), np.ones(hw), Kx)
        return (c / (rays @ n)).astype(np.float32)

    native = plane_depth(K, (120, 160))
    aff = resize_affine((120, 160), (60, 80)).then(crop_affine(top=5, left=3))
    out_hw = (50, 70)
    warped = warp_affine(native, aff, out_hw, order=0)
    direct = plane_depth(aff.apply_K(K), out_hw)
    # nearest sampling: within the depth change across one native pixel
    step = np.abs(np.diff(native, axis=1)).max() + np.abs(np.diff(native, axis=0)).max()
    assert np.abs(warped - direct).max() <= step + 1e-6
    # disparity measured on the resized copy converts back to native pixels by the x scale only
    d_model = np.full((2, 2), 3.0)
    assert np.allclose(rescale_disparity(d_model, PixelAffine(0.5, 0.5, -3, -5)), 6.0)


def test_depth_transfer_to_a_second_camera_matches_direct_render():
    """IR-left depth forward-warped into a colour camera 15 mm to the side equals that camera's own render where
    both see the surface; nothing is invented in disocclusions."""
    from s4d.realworld.geometry import invert_T, reproject_depth
    from s4d.realworld.synthetic import look_at_T_cam_base, render_box_scene

    K = np.array([[150.0, 0, 72.0], [0, 150.0, 48.0], [0, 0, 1]])
    hw = (96, 144)
    T_ir = look_at_T_cam_base([1.25, 0.55, 0.65], [0.5, 0.0, 0.1])
    T_rgb = make_T(Rotation.from_euler("y", 0.5, degrees=True).as_matrix(), [-0.015, 0.0, 0.0]) @ T_ir
    _, d_ir, _, _ = render_box_scene(T_ir, K, hw, 1.0)
    _, d_rgb, _, _ = render_box_scene(T_rgb, K, hw, 1.0)
    warped = reproject_depth(d_ir, K, T_rgb @ invert_T(T_ir), K, hw, splat=2)
    m = (warped > 0) & (d_rgb > 0)
    assert m.mean() > 0.8
    rel = np.abs(warped[m] - d_rgb[m]) / d_rgb[m]
    assert np.median(rel) < 0.005 and np.percentile(rel, 95) < 0.03
