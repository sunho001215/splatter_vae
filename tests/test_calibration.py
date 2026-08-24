from __future__ import annotations

import json

import numpy as np

from dataset.droid.calibration import (
    CalibrationThresholds,
    EpisodePathMatcher,
    RLDSEpisodeMetadata,
    build_calibration_entry,
    cam2base_pose_matrix,
    infer_cam2cam_direction,
    normalize_episode_path,
    parse_intrinsics,
    prepare_calibration_manifest,
    released_ext2_from_ext1,
    resize_intrinsics,
)

EPISODE_ID = "Lab+collector+2023-08-01-12h-00m-00s"
EPISODE_PATH = "Lab/success/2023-08-01/Tue_Aug__1_12:00:00_2023"


def _pose(tx: float, ty: float, tz: float) -> list[float]:
    return [tx, ty, tz, 0.0, 0.0, 0.0]


def _relative_entry(base_from_a: np.ndarray, base_from_b: np.ndarray) -> dict:
    b_from_a = np.linalg.inv(base_from_b) @ base_from_a
    return {
        "left_cam": {"pose": b_from_a.tolist()},
        "right_cam": {"pose": np.eye(4).tolist()},
    }


def _official(*, include_b_direct: bool = True) -> dict:
    base_from_a = cam2base_pose_matrix(_pose(0.4, -0.3, 0.8))
    base_from_b = cam2base_pose_matrix(_pose(-0.4, -0.3, 0.8))
    direct = {"111": _pose(0.4, -0.3, 0.8)}
    if include_b_direct:
        direct["222"] = _pose(-0.4, -0.3, 0.8)
    return {
        "episode_id_to_path": {EPISODE_ID: EPISODE_PATH},
        "camera_serials": {
            EPISODE_ID: {
                "ext1_cam_serial": "111",
                "ext2_cam_serial": "222",
                "wrist_cam_serial": "333",
            }
        },
        "intrinsics": {
            EPISODE_ID: {
                "111": {
                    "cameraMatrix": [800.0, 640.0, 800.0, 360.0],
                    "width": 1280,
                    "height": 720,
                },
                "222": {
                    "cameraMatrix": [804.0, 638.0, 802.0, 358.0],
                    "width": 1280,
                    "height": 720,
                },
            }
        },
        "cam2base_extrinsic_superset": {EPISODE_ID: direct},
        "cam2base_extrinsics": {},
        "cam2cam_extrinsics": {EPISODE_ID: _relative_entry(base_from_a, base_from_b)},
    }


def _metadata() -> RLDSEpisodeMetadata:
    return RLDSEpisodeMetadata(
        rlds_split="train",
        rlds_ordinal=7,
        file_path=(
            f"gs://gresearch/robotics/droid_raw/1.0.1/{EPISODE_PATH}/trajectory.h5"
        ),
        recording_folderpath=f"/mnt/r2d2-data-full/{EPISODE_PATH}/recordings/MP4",
        num_steps=100,
    )


def test_rlds_path_to_episode_id_matching() -> None:
    matcher = EpisodePathMatcher({EPISODE_ID: EPISODE_PATH})
    episode_id, normalized = matcher.match(_metadata().file_path, "")
    assert episode_id == EPISODE_ID
    assert normalized == EPISODE_PATH
    assert (
        normalize_episode_path(f"/tmp/r2d2-data-full/{EPISODE_PATH}/trajectory.h5")
        == EPISODE_PATH
    )


def test_intrinsics_parse_and_scale_to_rlds() -> None:
    K, width, height = parse_intrinsics(
        {"cameraMatrix": [800.0, 640.0, 820.0, 360.0], "width": 1280, "height": 720}
    )
    resized = resize_intrinsics(K, (width, height), (320, 180))
    np.testing.assert_allclose(
        resized,
        np.array(((200.0, 0.0, 160.0), (0.0, 205.0, 90.0), (0.0, 0.0, 1.0))),
    )


def test_manifest_rejects_unexpected_official_intrinsic_resolution() -> None:
    official = _official()
    official["intrinsics"][EPISODE_ID]["111"]["width"] = 320
    official["intrinsics"][EPISODE_ID]["111"]["height"] = 180
    entry = build_calibration_entry(
        _metadata(),
        official,
        EpisodePathMatcher(official["episode_id_to_path"]),
        CalibrationThresholds(direction_minimum_samples=1),
    )
    assert not entry["valid"]
    assert entry["failure_reason"] == "unexpected_official_intrinsic_resolution"


def test_cam2base_is_c2w_and_w2c_is_inverse() -> None:
    c2w = cam2base_pose_matrix([0.1, -0.2, 0.3, 0.2, -0.1, 0.4])
    np.testing.assert_allclose(np.linalg.inv(c2w) @ c2w, np.eye(4), atol=1.0e-10)
    np.testing.assert_allclose(c2w[:3, 3], (0.1, -0.2, 0.3))


def test_cam2cam_direction_is_ext2_from_ext1() -> None:
    official = _official()
    thresholds = CalibrationThresholds(direction_minimum_samples=1)
    result = infer_cam2cam_direction(
        official["cam2base_extrinsic_superset"],
        official["cam2cam_extrinsics"],
        official["camera_serials"],
        thresholds,
    )
    assert result["num_comparisons"] == 1
    assert result["forward_median_translation_m"] < 1.0e-10
    base_a = cam2base_pose_matrix(_pose(0.4, -0.3, 0.8))
    base_b = cam2base_pose_matrix(_pose(-0.4, -0.3, 0.8))
    np.testing.assert_allclose(
        released_ext2_from_ext1(official["cam2cam_extrinsics"][EPISODE_ID]),
        np.linalg.inv(base_b) @ base_a,
    )


def test_episode_camera_serial_mapping_and_direct_poses() -> None:
    official = _official()
    entry = build_calibration_entry(
        _metadata(),
        official,
        EpisodePathMatcher(official["episode_id_to_path"]),
        CalibrationThresholds(),
    )
    assert entry["valid"]
    assert [camera["serial"] for camera in entry["exterior_cameras"]] == ["111", "222"]
    assert [camera["rlds_image_key"] for camera in entry["exterior_cameras"]] == [
        "exterior_image_1_left",
        "exterior_image_2_left",
    ]
    assert all(camera["pose_flag"] == "direct" for camera in entry["exterior_cameras"])
    for camera in entry["exterior_cameras"]:
        np.testing.assert_allclose(
            np.asarray(camera["w2c"]) @ np.asarray(camera["c2w"]), np.eye(4)
        )


def test_second_pose_is_derived_without_guessing_direction() -> None:
    official = _official(include_b_direct=False)
    entry = build_calibration_entry(
        _metadata(),
        official,
        EpisodePathMatcher(official["episode_id_to_path"]),
        CalibrationThresholds(),
    )
    assert entry["valid"]
    assert entry["exterior_cameras"][0]["pose_flag"] == "direct"
    assert entry["exterior_cameras"][1]["pose_flag"] == "derived"
    expected = cam2base_pose_matrix(_pose(-0.4, -0.3, 0.8))
    np.testing.assert_allclose(
        entry["exterior_cameras"][1]["c2w"], expected, atol=1.0e-10
    )


def test_prepare_persists_configured_episode_split(tmp_path) -> None:
    calibration_dir = tmp_path / "official"
    calibration_dir.mkdir()
    for name, payload in _official().items():
        (calibration_dir / f"{name}.json").write_text(json.dumps(payload))
    manifest = tmp_path / "derived" / "calibration.jsonl.gz"
    split_manifest = tmp_path / "separate" / "episode_splits.json"
    result = prepare_calibration_manifest(
        [_metadata()],
        calibration_dir,
        manifest,
        thresholds=CalibrationThresholds(direction_minimum_samples=1),
        validation_fraction=0.5,
        split_seed=17,
        split_output_path=split_manifest,
        droid_root=tmp_path / "source",
    )
    assert manifest.is_file()
    assert split_manifest.is_file()
    split = json.loads(split_manifest.read_text())
    assert split["seed"] == 17
    assert split["validation"] == [EPISODE_ID]
    assert result["statistics"]["calibration_valid_episodes"] == 1
