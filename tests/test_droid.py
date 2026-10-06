"""Focused CPU checks for real-schema DROID conversion and the shared contract."""

from __future__ import annotations

import importlib.util
import io
import json
import tarfile
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from s4d.data.contract import collate, validate_batch
from s4d.data.droid.convert import RESOLUTIONS, resize_image
from s4d.data.droid.dataset import DroidCacheDataset
from s4d.data.droid.gripper import STROKE_M, gripper_points
from s4d.data.droid.pointworld import (
    clip_windows,
    depth_test,
    nearest_timestamps,
    project,
    resize_intrinsics,
    sparse_targets,
)
from s4d.data.droid.rlds import IMAGE_KEYS, save_matched_episode, scene_suffix, verify_path, verify_states


def test_episode_suffix_and_exact_both_paths():
    scene = "gs://new/bucket/AUTOLab/success/2023-10-21/recording/"
    metadata = {
        "file_path": "gs://old/AUTOLab/success/2023-10-21/recording/trajectory.h5",
        "recording_folderpath": "gs://old/AUTOLab/success/2023-10-21/recording/recordings/MP4",
    }
    verify_path(metadata, scene)
    assert scene_suffix(scene) == "AUTOLab/success/2023-10-21/recording"
    with pytest.raises(ValueError, match="mismatch"):
        verify_path(metadata | {"recording_folderpath": "gs://other/AUTOLab/failure/different/recording"}, scene)


def test_half_open_windows_never_cross_clip():
    windows = clip_windows("5:16")
    assert len(windows) == 5
    assert windows[0] == (5, 8, 11)
    assert windows[-1] == (9, 12, 15)
    assert clip_windows("0:6") == []
    with pytest.raises(ValueError):
        clip_windows("2:1")


def test_camera_nearest_timestamp_ties_earlier():
    assert nearest_timestamps(np.array([10, 20, 31]), np.array([0, 15, 19, 30, 50])).tolist() == [0, 0, 1, 2, 2]
    with pytest.raises(ValueError):
        nearest_timestamps(np.array([3, 2]), np.array([2]))


def test_integer_to_continuous_intrinsics_independent_axis_resize():
    K = np.array([[200.0, 0, 159.5], [0, 210.0, 89.5], [0, 0, 1]], np.float32)
    out = resize_intrinsics(K, 252, 140)
    assert out.dtype == np.float32
    assert out[0, 0] == pytest.approx(200 * 252 / 320)
    assert out[1, 1] == pytest.approx(210 * 140 / 180)
    assert out[0, 2] == pytest.approx(126)
    assert out[1, 2] == pytest.approx(70)
    assert K[0, 2] == 159.5


def test_world_to_camera_projection_direction_and_native_depth():
    K = np.eye(3, dtype=np.float32)
    w2c = np.eye(4, dtype=np.float32)
    w2c[0, 3] = -0.5
    p = np.array([[0.5, 0.0, 1.0]], np.float32)
    uv, z = project(p, K, w2c)
    np.testing.assert_allclose(uv, [[0, 0]])
    assert z[0] == 1
    depth = np.ones((3, 3), np.float32)
    assert depth_test(p, K, w2c, depth)[0][0]
    assert not depth_test(p + [0, 0, 0.1], K, w2c, depth)[0][0]
    assert not depth_test(p + [0, 0, -0.1], K, w2c, depth)[0][0]
    assert depth_test(p + [0, 0, -0.1], K, w2c, depth, surface=False)[0][0]


def motion_case():
    K = np.array([[10, 0, 3.5], [0, 10, 3.5], [0, 0, 1]], np.float32)
    points = np.array([[[0.0, 0, 1]], [[0.2, 0, 1]], [[0.4, 0, 1]]], np.float32)
    return points, np.ones((3, 1), bool), K, np.eye(4, dtype=np.float32), np.ones((3, 8, 8), np.float32)


def test_pairs_source_grid_and_world_position_difference():
    motion, weight, score = sparse_targets(*motion_case())
    np.testing.assert_allclose(motion[0, :, 3, 3], [0.2, 0, 0])
    np.testing.assert_allclose(motion[1, :, 3, 5], [0.2, 0, 0])
    np.testing.assert_allclose(motion[2, :, 3, 3], [0.4, 0, 0])
    assert weight.sum() == 3
    assert weight[1, 0, 3, 3] == 0
    assert score[0, 0, 3, 3] == score[1, 0, 3, 5] == score[2, 0, 3, 7] == 1


def test_motion_score_maximum_includes_02_at_middle_time():
    points, mask, K, w2c, depth = motion_case()
    points[1, 0, 0], points[2, 0, 0] = 0.01, 0.02
    _, _, score = sparse_targets(points, mask, K, w2c, depth)
    np.testing.assert_allclose(score[:, 0, 3, 3], [0.02 / 0.03] * 3, atol=1e-6)


def test_scene_and_gripper_collision_nearest_z_stable_tie():
    points, mask, K, w2c, depth = motion_case()
    points = np.concatenate([points, points], axis=1)
    points[:, 0, 2] = 1.005
    points[:, 1, 2] = 0.995
    points[1, 1, 0] = 0.1
    points[2, 1, 0] = 0.1
    mask = np.ones((3, 2), bool)
    motion, weight, _ = sparse_targets(points, mask, K, w2c, depth)
    assert weight[0].sum() == 1
    assert motion[0, 0, 3, 3] == pytest.approx(0.1)
    points[:, :, 2] = 1
    motion, _, _ = sparse_targets(points, mask, K, w2c, depth)
    assert motion[0, 0, 3, 3] == pytest.approx(0.2)


def test_source_visibility_validity_and_two_centimeter_surface_test():
    points, mask, K, w2c, depth = motion_case()
    mask[0] = False
    _, weight, _ = sparse_targets(points, mask, K, w2c, depth)
    assert weight[0].sum() == weight[2].sum() == 0
    assert weight[1].sum() == 1
    points[:, :, 2] = 1.03
    _, weight, _ = sparse_targets(points, np.ones_like(mask), K, w2c, depth)
    assert weight.sum() == 0
    points[:, :, 2] = 1
    depth[:] = 0
    _, weight, _ = sparse_targets(points, np.ones_like(mask), K, w2c, depth)
    assert weight.sum() == 0


def test_gripper_sign_stroke_points_and_rigid_transform():
    pose = np.zeros((2, 6))
    points = gripper_points(pose, np.array([0, 1]))
    assert points.shape == (2, 32, 3)
    assert points.dtype == np.float32
    assert abs(points[0, 20, 1] - points[0, 8, 1]) == pytest.approx(STROKE_M)
    assert points[1, 20, 1] == points[1, 8, 1]
    pose[:, :3] = [1, 2, 3]
    pose[:, 5] = np.pi / 2
    moved = gripper_points(pose, np.array([0, 1]), offset_m=0.012)
    expected = points @ Rotation.from_euler("z", np.pi / 2).as_matrix().T + [1, 2, 3.012]
    np.testing.assert_allclose(moved, expected, atol=1e-6)
    with pytest.raises(ValueError):
        gripper_points(pose, np.array([-1, 0]))


def test_raw_euler_quaternion_gripper_and_timeline_verification(tmp_path):
    pose = np.zeros((22, 6))
    pose[:, 0] = np.arange(22) / 100
    pose[:, 3:] = [0.1, 0.2, -0.3]
    closure = np.arange(22).reshape(-1, 1) / 30
    path = tmp_path / "flow.h5"
    with h5py.File(path, "w") as flow:
        group = flow.create_group("0:11")
        group["gripper_pose"] = np.c_[pose[::2, :3], Rotation.from_euler("xyz", pose[::2, 3:]).as_quat()]
        group["gripper_positions"] = closure[::2, 0] * 0.725
        result = verify_states({"cartesian_position": pose, "gripper_position": closure}, flow)
        assert result["xyz_max_m"] == 0
        assert result["gripper_direct_max"] < 1e-7
        with pytest.raises(ValueError, match="check failed"):
            verify_states({"cartesian_position": pose + 0.1, "gripper_position": closure}, flow)


def test_raw_cache_never_retains_wrist(tmp_path):
    raw = {k: np.zeros((1, 2, 2, 3), np.uint8) for k in (*IMAGE_KEYS, "wrist_image_left")}
    raw.update(cartesian_position=np.zeros((1, 6)), gripper_position=np.zeros((1, 1)), joint_position=np.zeros((1, 7)))
    save_matched_episode(tmp_path / "raw.npz", raw, {"checked": True})
    with np.load(tmp_path / "raw.npz", allow_pickle=False) as saved:
        assert "wrist_image_left" not in saved.files
    assert (tmp_path / "raw.json").is_file()


def test_metric_depth_nearest_resize_preserves_zero():
    depth = np.array([[0.0, 0.3], [0.7, 1]], np.float32)
    resized = resize_image(depth, 6, 8, depth=True)
    assert resized.dtype == np.float32
    np.testing.assert_array_equal(np.unique(resized), np.unique(depth))


@pytest.mark.parametrize("height,width", RESOLUTIONS.values())
def test_cache_contract_roundtrip_both_backbone_sizes(tmp_path, height, width):
    backbone = next(key for key, value in RESOLUTIONS.items() if value == (height, width))
    manifest = {
        "episode": "fixture",
        "camera_serials": ["ext1", "ext2"],
        "windows": [
            {"split": "train", "clip": "0:11", "canonical_indices": [0, 3, 6], "raw_indices": [0, 6, 12]},
            {"split": "validation", "clip": "20:31", "canonical_indices": [20, 23, 26], "raw_indices": [40, 46, 52]},
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    K = np.array([[100, 0, width / 2], [0, 100, height / 2], [0, 0, 1]], np.float32)
    values = {
        "images": np.zeros((3, 2, 3, height, width), np.uint8),
        "depth": np.ones((3, 2, 1, height, width), np.float32),
        "motion3d": np.zeros((3, 2, 3, height, width), np.float32),
        "motion_weight": np.ones((3, 2, 1, height, width), np.float32),
        "motion_score": np.zeros((3, 2, 1, height, width), np.float32),
        "K": np.stack([K, K]),
        "w2c": np.stack([np.eye(4, dtype=np.float32)] * 2),
        "c2w": np.stack([np.eye(4, dtype=np.float32)] * 2),
    }
    with h5py.File(tmp_path / f"{backbone}.h5", "w") as cache:
        for i in range(2):
            group = cache.create_group(f"windows/{i:05d}")
            for key, value in values.items():
                group[key] = value
    for split in ("train", "validation"):
        dataset = DroidCacheDataset(tmp_path, split=split, with_eval=True, image_height=height, image_width=width)
        assert len(dataset) == 1
        batch = collate([dataset[0], dataset[0]])
        assert validate_batch(batch) == {"B": 2, "T": 3, "V": 2, "H": height, "W": width}
        assert batch["images"].dtype == torch.uint8
        assert "eval_images" not in batch
        assert dataset.__getstate__()["_file"] is None
        dataset.close()


def test_download_copies_only_exact_regular_member_without_path_extraction(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/download_droid_sample.py"
    spec = importlib.util.spec_from_file_location("safe_sample_download", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w") as archive:
        for name, contents in (("../../planted.py", b"never execute"), ("droid/selected.h5", b"actual data")):
            info = tarfile.TarInfo(name)
            info.size = len(contents)
            archive.addfile(info, io.BytesIO(contents))
    data.seek(0)
    destination = tmp_path / "selected.h5"
    record = module.copy_member(data, "droid/selected.h5", destination)
    assert destination.read_bytes() == b"actual data"
    assert record["member"] == "droid/selected.h5"
    assert list(tmp_path.iterdir()) == [destination]
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w") as archive:
        info = tarfile.TarInfo("droid/symlink.h5")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        archive.addfile(info)
    data.seek(0)
    with pytest.raises(ValueError, match="regular"):
        module.copy_member(data, "droid/symlink.h5", tmp_path / "symlink.h5")


def test_output_boundary_rejects_protected_and_symlink_escapes(tmp_path):
    from s4d.data import DROID_CACHE_ROOT, POINTWORLD_ROOT, writable_path

    assert writable_path(tmp_path / "allowed", DROID_CACHE_ROOT) == tmp_path / "allowed"
    assert writable_path(DROID_CACHE_ROOT / "new", DROID_CACHE_ROOT) == DROID_CACHE_ROOT / "new"
    assert writable_path(POINTWORLD_ROOT / "new", POINTWORLD_ROOT) == POINTWORLD_ROOT / "new"
    for path in ("/home/ws/data/droid/empty", "/home/ws/ws/droid_training/empty", "/home/ws/ws/hierarchical_splatter/empty"):
        with pytest.raises(ValueError, match="escapes"):
            writable_path(path, DROID_CACHE_ROOT)
    symlink = tmp_path / "escape"
    symlink.symlink_to("/home/ws/data/droid", target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        writable_path(symlink / "empty", DROID_CACHE_ROOT)


def test_partial_gripper_calibration_candidates_are_strict_json_serializable(monkeypatch):
    import s4d.data.droid.gripper as grip

    monkeypatch.setattr(grip, "gripper_points", lambda pose, closure, offset: offset)
    monkeypatch.setattr(
        grip, "gripper_depth_residuals", lambda offset, *a: np.full(40, abs(offset)) if offset >= 0 else np.empty(0)
    )
    result = grip.calibrate_offset(None, None, None, None, None)
    assert result["offset_m"] == 0 and result["count"] == 40
    assert any(c["median_residual_m"] is None and not c["valid"] for c in result["candidates"])
    json.dumps(result, allow_nan=False)
    monkeypatch.setattr(grip, "gripper_depth_residuals", lambda *a: np.empty(0))
    with pytest.raises(ValueError, match="not enough"):
        grip.calibrate_offset(None, None, None, None, None)


def test_converter_rejects_output_boundary_before_any_input_reads():
    from s4d.data.droid.convert import convert_sample

    with pytest.raises(ValueError, match="escapes"):
        convert_sample(Path("/nonexistent"), Path("/home/ws/data/droid/empty"), rlds_root=Path("/nonexistent"))


def test_dense_cross_camera_validation_uses_depth_not_supplied_tracks():
    from s4d.data.droid.inspect import dense_depth_alignment

    depths = np.ones((2, 2, 8, 8), np.float32)
    K = np.tile(np.array([[8, 0, 4], [0, 8, 4], [0, 0, 1]], np.float32), (2, 1, 1))
    cameras = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
    report, snapshots = dense_depth_alignment(depths, K, cameras)
    assert report["passed"] and report["median_relative_error"] == 0
    assert report["selected_target_pixels"] == 256 and len(snapshots) == 2
    depths[:, 1] = 1.1
    report, _ = dense_depth_alignment(depths, K, cameras)
    assert not report["passed"] and report["median_relative_error"] > 0.03
    assert report["unfiltered_target_pixels"] > report["selected_target_pixels"]
    empty, _ = dense_depth_alignment(np.zeros_like(depths), K, cameras)
    assert empty["median_relative_error"] is None and not empty["passed"]
