"""Pure stereo geometry and image/depth/intrinsics alignment. CPU numpy only; no model is involved."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from s4d.realworld.geometry import (
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
    assert np.allclose(disp * z_rect, rect.K_rect[0, 0] * rect.baseline_m, rtol=1e-6)
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


def test_pixel_distortion_round_trip():
    from s4d.realworld.geometry import distort_pixels, undistort_pixels

    K = np.array([[300.0, 0, 160.0], [0, 300.0, 120.0], [0, 0, 1]])
    uv = np.random.default_rng(0).uniform([5, 5], [315, 235], (200, 2))
    for model, coeffs in (("opencv_plumb_bob", (0.08, -0.03, 0.001, -0.001, 0.0)),
                          ("opencv_rational", (0.05, -0.02, 0.0005, 0.0, 0.0, 0.01, 0.0, 0.0))):
        assert np.abs(undistort_pixels(distort_pixels(uv, K, model, coeffs), K, model, coeffs) - uv).max() < 1e-6
    with pytest.raises(NotImplementedError):
        undistort_pixels(uv, K, "kannala_brandt4", (0.1, 0, 0, 0))
