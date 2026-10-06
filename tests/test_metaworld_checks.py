"""Controlled planar scenes expose empty-support and residual-based filtering."""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from scripts.check_metaworld import check_dataset, cloud_zbuffer, metric_summary, visible_surface


def _write_fixture(path, *, moving=True, bad_depth=False, disjoint_bodies=False):
    T, V, H, W, unit = 19, 3, 32, 32, 1e-4
    K = np.tile(np.array([[32.0, 0.0, 16.0], [0.0, 32.0, 16.0], [0.0, 0.0, 1.0]], dtype=np.float32), (V, 1, 1))
    c2w = np.tile(np.eye(4, dtype=np.float32), (V, 1, 1))
    c2w[:, 0, 3] = [0.0, 1 / 32, -1 / 32]
    rgb = np.zeros((T, V, H, W, 3), dtype=np.uint8)
    rgb[..., 0] = np.arange(W, dtype=np.uint8)[None, None, None, :] * 8
    depth = np.full((T, V, H, W), 10000, dtype=np.uint16)
    if bad_depth:
        depth[:, 1] += 200  # Exactly 2 cm, well above both fixed acceptance gates.
    body = np.zeros((T, 2, H, W), dtype=np.uint16)
    body[..., H // 2 :, :] = 1
    if disjoint_bodies:
        body[:, 1, H // 2 :, :] = 2
    xpos = np.zeros((T, 3, 3), dtype=np.float32)
    if moving:
        xpos[:, 1, 0] = np.arange(T) / 96.0  # One source pixel of lateral motion at stride 3.
    quat = np.zeros((T, 3, 4), dtype=np.float32)
    quat[..., 0] = 1
    with h5py.File(path, "w") as file:
        file.attrs.update(
            {
                "task": "synthetic-plane",
                "depth_unit_m": unit,
                "dt_seconds": 0.05,
                "background_body": 65535,
                "body_names": json.dumps(["world", "moving-plane", "other-plane"]),
                "complete": True,
            }
        )
        cams = file.create_group("cameras")
        for key, values in {
            "K": K,
            "c2w": c2w,
            "w2c": np.linalg.inv(c2w),
            "is_train": np.array([True, True, False]),
        }.items():
            cams.create_dataset(key, data=values)
        cams.create_dataset("names", data=np.asarray(["train0", "train1", "eval0"], dtype=h5py.string_dtype()))
        episode = file.create_group("episodes").create_group("ep000")
        episode.attrs["length"] = T
        for key, values in {
            "rgb": rgb,
            "depth": depth,
            "body_id": body,
            "xpos": xpos,
            "xquat": quat,
            "obs": np.zeros((T, 39), dtype=np.float32),
            "qpos": np.zeros((T, 3), dtype=np.float32),
            "qvel": np.zeros((T, 3), dtype=np.float32),
            "action": np.zeros((T, 4), dtype=np.float32),
            "reward": np.zeros(T, dtype=np.float32),
            "success": np.zeros(T, dtype=np.uint8),
        }.items():
            episode.create_dataset(key, data=values)
    return path


def test_complete_planar_fixture_passes_both_unchanged_geometry_gates(tmp_path):
    path = _write_fixture(tmp_path / "plane.hdf5")
    report = check_dataset(path, tmp_path / "checks", windows=3, points_per_view=1024)
    assert report["passed"]
    assert report["d2"]["threshold_mm"] == 3.0 and report["d3"]["threshold_mm"] == 5.0
    assert report["d2"]["count"] > 0 and report["d2"]["median_mm"] < 0.001
    assert report["d3"]["count"] > 0 and report["d3"]["median_mm"] < 0.001
    assert report["world_body_zero"]["pixels"] > 0 and report["world_body_zero"]["nonzero_pixels"] == 0
    assert not report["contract_errors"]
    assert (tmp_path / "checks" / "summary.json").is_file()
    assert (tmp_path / "checks" / "sanity.png").is_file()
    reasons = (
        "nonfinite_or_behind",
        "outside_interpolation_support",
        "invalid_target_depth_support",
        "target_body_or_boundary_mismatch",
        "independent_cloud_occlusion",
        "retained",
    )
    for window in report["windows"]:
        for comparison in window["d2_comparisons"]:
            assert sum(comparison[reason] for reason in reasons) == comparison["comparisons"]


def test_two_centimeter_depth_error_is_retained_and_fails_not_filtered_away(tmp_path):
    path = _write_fixture(tmp_path / "bad_depth.hdf5", bad_depth=True)
    report = check_dataset(path, tmp_path / "checks", windows=3, points_per_view=1024)
    assert not report["passed"]
    assert report["d2"]["count"] > 0 and report["d2"]["median_mm"] > 15
    assert report["d3"]["count"] > 0 and report["d3"]["median_mm"] > 15
    assert not report["d2"]["passed"] and not report["d3"]["passed"]


def test_static_only_windows_do_not_claim_success_on_empty_moving_support(tmp_path):
    path = _write_fixture(tmp_path / "static.hdf5", moving=False)
    report = check_dataset(path, tmp_path / "checks", windows=2, points_per_view=1024)
    assert not report["passed"]
    assert report["d2"]["count"] == 0 and report["d2"]["median_mm"] is None
    assert report["d2"]["status"] == "insufficient_support"


def test_body_filter_empty_support_is_not_a_zero_error_pass(tmp_path):
    path = _write_fixture(tmp_path / "disjoint.hdf5", disjoint_bodies=True)
    report = check_dataset(path, tmp_path / "checks", windows=2, points_per_view=1024)
    assert not report["passed"]
    assert report["d2"]["count"] == 0 and not report["d2"]["passed"]
    excluded = sum(
        comparison["target_body_or_boundary_mismatch"]
        for window in report["windows"]
        for comparison in window["d2_comparisons"]
    )
    assert excluded > 0


def test_visibility_never_uses_the_evaluated_depth_residual_to_accept_points():
    K = np.array([[32.0, 0.0, 16.0], [0.0, 32.0, 16.0], [0.0, 0.0, 1.0]])
    points = np.array([[-7.5 / 32, -7.5 / 32, 1.0]])
    target = np.full((32, 32), 1.04)
    body = np.ones((32, 32), dtype=np.int64)
    independent = cloud_zbuffer(points, K, np.eye(4), 32, 32)
    keep, observed, z, counts = visible_surface(points, np.ones(1, dtype=np.int64), K, np.eye(4), target, body, independent)
    assert keep.tolist() == [True] and counts["retained"] == 1
    np.testing.assert_allclose(np.abs(observed[keep] - z[keep]), [0.04], atol=1e-12)


def test_independent_front_surface_occludes_same_body_back_surface():
    K = np.array([[32.0, 0.0, 16.0], [0.0, 32.0, 16.0], [0.0, 0.0, 1.0]])
    points = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.05]])
    depth = np.ones((32, 32))
    body = np.ones((32, 32), dtype=np.int64)
    independent = cloud_zbuffer(points, K, np.eye(4), 32, 32)
    keep, _, _, counts = visible_surface(points, np.ones(2, dtype=np.int64), K, np.eye(4), depth, body, independent)
    assert keep.tolist() == [True, False]
    assert counts["independent_cloud_occlusion"] == 1


@pytest.mark.parametrize(
    "corruption,reason", [("body", "target_body_or_boundary_mismatch"), ("depth", "invalid_target_depth_support")]
)
def test_bilinear_support_requires_all_four_same_body_valid_neighbors(corruption, reason):
    K = np.array([[32.0, 0.0, 16.0], [0.0, 32.0, 16.0], [0.0, 0.0, 1.0]])
    # Projection is exactly (8.5, 8.5), so the half-pixel convention gives (8, 8).
    point = np.array([[-7.5 / 32, -7.5 / 32, 1.0]])
    depth = np.ones((32, 32))
    body = np.ones((32, 32), dtype=np.int64)
    if corruption == "body":
        body[9, 9] = 2
    else:
        depth[9, 9] = 0
    independent = cloud_zbuffer(point, K, np.eye(4), 32, 32)
    keep, _, _, counts = visible_surface(point, np.ones(1, dtype=np.int64), K, np.eye(4), depth, body, independent)
    assert keep.tolist() == [False] and counts[reason] == 1
    assert counts["comparisons"] == 1 and counts["retained"] == 0


def test_tiny_nonzero_world_body_motion_fails_exact_zero_gate(tmp_path, monkeypatch):
    from scripts.check_metaworld import MetaworldWindowDataset

    path = _write_fixture(tmp_path / "plane.hdf5")
    original = MetaworldWindowDataset.__getitem__

    def corrupt_world_pixel(dataset, index):
        sample = original(dataset, index)
        sample["motion3d"][:, 0, 0, 0, 0] = 1e-12
        return sample

    monkeypatch.setattr(MetaworldWindowDataset, "__getitem__", corrupt_world_pixel)
    report = check_dataset(path, tmp_path / "checks", windows=2, points_per_view=1024)
    assert report["d2"]["passed"] and report["d3"]["passed"]
    assert report["world_body_zero"]["nonzero_pixels"] > 0
    assert not report["world_body_zero"]["passed"] and not report["passed"]


def test_empty_metric_support_has_null_median_and_fails():
    result = metric_summary([], threshold_mm=3.0)
    assert result["count"] == 0 and result["median_mm"] is None and not result["passed"]


def test_check_outputs_cannot_escape_new_repository(tmp_path):
    path = _write_fixture(tmp_path / "plane.hdf5")
    with pytest.raises(ValueError, match="new repository"):
        check_dataset(path, "/home/ws/data/droid/forbidden-check-output")


def test_split_manifest_is_deterministic_and_bounded_before_input_reads(tmp_path):
    from s4d.data.metaworld.dataset import split_episodes

    path = tmp_path / "episodes.hdf5"
    with h5py.File(path, "w") as file:
        for i in range(5):
            file.create_group(f"episodes/ep{i:03d}")
    first = split_episodes(path, seed=3)
    assert first == split_episodes(path, seed=3)
    assert len(first[0]) == 4 and len(first[1]) == 1
    manifest = json.loads((tmp_path / "splits/episodes_seed3.json").read_text())
    assert manifest["train"] == first[0] and manifest["validation"] == first[1]
    with pytest.raises(ValueError, match="escapes"):
        split_episodes("/nonexistent", manifest_dir="/home/ws/data/droid/splits")
    escape = tmp_path / "escape"
    escape.symlink_to("/home/ws/data/droid", target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        split_episodes(path, manifest_dir=escape / "splits")
