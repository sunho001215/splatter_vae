"""Restored stereo conventions (2026-11-05, optional FoundationStereo backend for device IR pairs): disparity ->
metric depth, flip-and-swap right reference, left-right consistency, depth registration with source pixels, and the
IR disparity -> colour-grid kernel. CPU numpy only; the block matcher is a test oracle, never model output."""

import numpy as np
import pytest

from s4d.realworld.geometry import make_T, reproject_depth
from s4d.realworld.stereo import (
    block_match,
    depth_from_pair,
    disparity_to_depth,
    flip_and_swap_right_disparity,
    lr_consistency,
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




def test_lr_consistency_is_zero_for_consistent_disparities_and_flags_mismatches():
    H, W = 4, 40
    d = np.full((H, W), 5.0, np.float32)
    err = lr_consistency(d, d)  # constant disparity: right match x-5 has the same disparity
    assert np.allclose(err[:, 6:], 0) and np.isinf(err[:, :5]).all()  # no right match for x < 5
    d_right = d.copy()
    d_right[:, 20] = 9.0  # x_left = 25.5 maps exactly onto right column 20 -> mismatch of 4 px
    err = lr_consistency(d, d_right)
    assert err[0, 25] == pytest.approx(4.0) and err[0, 30] == pytest.approx(0.0)


def test_lr_consistency_with_block_matching_on_a_synthetic_pair():
    left, right, depth_l, _, K = rectified_pair()
    matcher = lambda a, b: block_match(a, b, 12)  # noqa: E731
    dl = matcher(left, right)
    dr = flip_and_swap_right_disparity(left, right, matcher)
    err = lr_consistency(dl, dr)
    m = _interior(depth_l) & np.isfinite(err)
    assert np.median(err[m]) < 0.5


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf, 0.5])
def test_lr_consistency_rejects_either_invalid_interpolation_tap(bad):
    dl = np.full((1, 20), 5.5, np.float32)
    dr = np.full_like(dl, 11.0)
    for col in (4, 5):  # left column 10 samples halfway between right columns 4 and 5
        right = dr.copy()
        right[0, col] = bad
        assert np.isinf(lr_consistency(dl, right, min_disparity=0.5)[0, 10])
    dr[0, 4:6] = [4.0, 7.0]
    assert lr_consistency(dl, dr, min_disparity=0.5)[0, 10] == 0


def test_stereo_stage_persists_depth_and_valid_only(tmp_path, monkeypatch):
    """Output layout 2: the per-pixel diagnostics (source_index, disparity_left, lr_error, ir_valid) are summarised,
    never persisted; depth is stored bitwise, valid = depth > 0, the kernel statistics reach the stage summary."""
    from types import SimpleNamespace

    import h5py

    from s4d.realworld import depth_stages, ir_stereo_depth, pipeline
    from s4d.realworld import io as rio
    from s4d.realworld.workers import WorkerSpec

    cam = "color"
    g = SimpleNamespace(rig_id="rig", left_stream="left", right_stream="right", baseline_m=0.05,
                        T_color=lambda: np.eye(4))
    stream = SimpleNamespace(intrinsics=SimpleNamespace(height=2, width=3))
    meta = SimpleNamespace(rig_rgbd=lambda _: SimpleNamespace(color_stream=cam), stream=lambda _: stream,
                           stream_timestamps_ns={"left": [0], cam: [0]})
    monkeypatch.setattr(pipeline, "load_meta", lambda *_: meta)
    monkeypatch.setattr(pipeline, "read_frames", lambda *args: np.zeros((1, 2, 3), np.uint8))
    monkeypatch.setattr(ir_stereo_depth, "ir_rigs", lambda _: [g])
    monkeypatch.setattr(depth_stages, "view_K", lambda *_: np.eye(3))
    rio.run_stage(rio.episode_dir(tmp_path, "ep") / "normalized", stage="ingest", version="test",
                  config={}, inputs={}, build=lambda _: {})
    error = np.array([[[1.0004, 0.9998, np.inf], [0.0, 0.0, 0.0]]], np.float32)
    depth = np.array([[[0.5, 0.0, 1.25], [np.float32(1 / 3), 2.0, 0.0]]], np.float32)
    out = {"depth": depth, "source_index": np.arange(6).reshape(1, 2, 3),
           "lr_error": error, "ir_valid": np.ones((1, 2, 3), bool)}
    disp = {"disparity_left": np.ones((1, 2, 3), np.float32), "disparity_right": np.ones((1, 2, 3), np.float32)}
    stats = {"frames": [{"ir_valid_fraction": 0.75, "registered_valid_fraction": 4 / 6, "lr_error_median_px": 0.5}]}
    runner = SimpleNamespace(worker=lambda *a, **k: ({"model": {}, "meta": {}}, disp, None),
                             kernel=lambda *a, **k: (out, stats, None), ack=lambda _: None, describe=lambda: {})
    rec = ir_stereo_depth.ir_stereo_depth(tmp_path, "ep", WorkerSpec("mock", "python"), runner)
    src = rio.stage_dir(tmp_path, "ep", "foundation_stereo_depth") / "color.h5"
    with h5py.File(src) as f:
        assert set(f) == {"depth", "valid"}
        assert f["depth"].dtype == np.float32 and f["valid"].dtype == bool
        assert f["depth"][:].tobytes() == depth.tobytes()
        np.testing.assert_array_equal(f["valid"][:], depth > 0)
        assert f.attrs["output_layout"] == ir_stereo_depth.OUTPUT_LAYOUT == "2"
        assert list(f.attrs["ir_hw"]) == [2, 3]
    s = rec["summary"]
    assert s["output_layout"] == "2" and s["persisted_datasets"] == ["depth", "valid"]
    assert s["color"]["lr_error_median_px"] == 0.5 and s["color"]["ir_valid_fraction_mean"] == 0.75
    assert rec["version"] == ir_stereo_depth.STAGE_VERSION == "2"  # computation version: layout-1 outputs stay valid
    again = ir_stereo_depth.ir_stereo_depth(tmp_path, "ep", WorkerSpec("mock", "python"), runner)
    assert again["skipped"] and again["build_id"] == rec["build_id"]


def test_reproject_depth_source_index_keeps_the_default_output_and_names_the_winners():
    rng = np.random.default_rng(0)
    depth = rng.uniform(0.5, 2.0, (24, 32))
    depth[5:9, 5:9] = 0
    K = np.array([[30.0, 0, 16.0], [0, 30.0, 12.0], [0, 0, 1]])
    T = make_T(np.eye(3), np.array([0.015, 0.0, 0.0]))
    plain = reproject_depth(depth, K, T, K, (24, 32), splat=2)
    out, src = reproject_depth(depth, K, T, K, (24, 32), splat=2, return_source=True)
    np.testing.assert_array_equal(plain, out)
    has = src >= 0
    assert ((out > 0) == has).all()
    from s4d.realworld.geometry import depth_to_points, transform_points

    pts, _ = depth_to_points(depth, K)
    z_src = transform_points(T, pts.reshape(-1, 3))[:, 2]
    np.testing.assert_allclose(out[has], z_src[src[has]], rtol=1e-6)  # each value is its winner's depth


def test_ir_disparity_to_color_kernel_registers_like_native_depth():
    from s4d.realworld.kernels import ir_disparity_to_color

    H, W = 30, 40
    K_ir = np.array([[40.0, 0, 20.0], [0, 40.0, 15.0], [0, 0, 1]])
    B = 0.05
    z = np.full((H, W), 1.0)
    z[10:20, 10:25] = 0.6
    dl = (K_ir[0, 0] * B / z).astype(np.float32)[None]
    T = make_T(np.eye(3), np.array([0.015, 0.0, 0.0]))
    out, meta = ir_disparity_to_color({"color_hw": [H, W], "min_disparity": 0.5, "min_depth_m": 0.1, "max_depth_m": 6,
                                       "splat": 2, "baseline_m": B},
                                      {"disparity_left": dl, "disparity_right": dl, "K_ir": K_ir, "K_color": K_ir,
                                       "T_color_left": T})
    ok = out["ir_valid"][0]
    z_from_disp, _ = disparity_to_depth(dl[0], K_ir[0, 0], B, 0.0, 0.5)  # the kernel's own depth (float32 disparity)
    np.testing.assert_allclose(z_from_disp[ok], z[ok], rtol=1e-5)
    depth_ir = np.where(ok, z_from_disp, 0.0)
    ref = reproject_depth(depth_ir, K_ir, T, K_ir, (H, W), splat=2)
    np.testing.assert_allclose(out["depth"][0], ref, rtol=1e-6)  # same registration as native depth
    assert not ok[:, :1].any()  # x - d < 0: no match in the right image
    assert meta["frames"][0]["registered_valid_fraction"] > 0.5
