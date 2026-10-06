from __future__ import annotations

import json

import numpy as np
import pytest

import dataset.droid.records as record_codec
from dataset.droid.codecs import (
    JPEGCodecConfig,
    depth_meters_to_u16,
    encode_jpeg,
    encode_numeric_array,
    flow_pixels_to_i16,
    pack_byte_strings,
    pack_support_masks,
)
from dataset.droid.dataset import (
    DROIDDatasetConfig,
    DROIDPreprocessedDataset,
    droid_collate,
)
from dataset.droid.integrity import (
    IntegrityConfig,
    _validate_pose_against_depth,
    _validate_pose_metadata,
    verify_stage0_dataset,
)
from dataset.droid.preprocessed_manifest import (
    build_stage0_manifest,
    sample_key,
    shard_path,
)
from dataset.droid.records import (
    pack_lager_record,
    pack_real_rgb_record,
    pack_timestep_metadata,
    unpack_lager_record,
)
from dataset.droid.sampling import EpisodeGroupedDistributedSampler, MotionCropConfig
from dataset.droid.shards import IndexedTarReader, IndexedTarWriter


def _entry() -> dict:
    cameras = []
    for index, x in enumerate((0.4, -0.4), start=1):
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, 3] = (x, -0.3, 0.8)
        cameras.append(
            {
                "logical_id": f"exterior_{index}",
                "serial": str(index) * 3,
                "intrinsics_rlds": [
                    [200.0, 0.0, 160.0],
                    [0.0, 200.0, 90.0],
                    [0.0, 0.0, 1.0],
                ],
                "c2w": c2w.tolist(),
                "w2c": np.linalg.inv(c2w).tolist(),
            }
        )
    return {
        "episode_id": "episode-test",
        "rlds_split": "train",
        "rlds_ordinal": 0,
        "rlds_path": "LAB/success/date/episode-test",
        "num_steps": 13,
        "dataset_split": "validation",
        "valid": True,
        "exterior_cameras": cameras,
    }


def _lager_metadata() -> dict[str, np.ndarray]:
    poses = np.broadcast_to(np.eye(4, dtype=np.float32), (4, 4, 4)).copy()
    alpha = np.array((0.2, 0.3, 0.7, 0.8), dtype=np.float32)
    return {
        "target_c2w": poses,
        "target_w2c": poses,
        "base_c2w": poses,
        "alpha": alpha,
        "baseline": np.ones(4, dtype=np.float32),
        "translation_perturbation": np.zeros((4, 3), dtype=np.float32),
        "translation_perturbation_magnitude": np.zeros(4, dtype=np.float32),
        "rotation_perturbation_degrees": np.zeros(4, dtype=np.float32),
        "source_coverage": np.ones(4, dtype=np.float32),
        "minimum_source_coverage_threshold": np.full(4, 0.60, dtype=np.float32),
        "minimum_geometry_distance": np.ones(4, dtype=np.float32),
        "geometry_clearance_distance": np.ones(4, dtype=np.float32),
        "minimum_geometry_distance_threshold": np.full(4, 0.08, dtype=np.float32),
        "source_clearance_reference": np.full(4, 0.10, dtype=np.float32),
        "target_scene_center": np.zeros((4, 3), dtype=np.float32),
        "distance_from_camera_a": np.ones(4, dtype=np.float32),
        "distance_from_camera_b": np.ones(4, dtype=np.float32),
        "rejected_candidates": np.zeros(4, dtype=np.int16),
        "alpha_resample_count": np.zeros(4, dtype=np.int16),
        "fallback_used": np.zeros(4, dtype=np.uint8),
        "scene_centered_arc_used": np.ones(4, dtype=np.uint8),
        "geometry_scene_center_used": np.zeros(4, dtype=np.uint8),
        "safety_translation_limit_escalated": np.zeros(4, dtype=np.uint8),
        "safety_thresholds_relaxed": np.zeros(4, dtype=np.uint8),
    }


def _plane_lager_metadata(depth: float) -> dict[str, np.ndarray]:
    """Calibrated synthetic cameras observing the same constant-z plane."""
    metadata = _lager_metadata()
    centers = np.asarray([camera["c2w"] for camera in _entry()["exterior_cameras"]])[
        :, :3, 3
    ]
    alpha = metadata["alpha"]
    targets = (1 - alpha[:, None]) * centers[0] + alpha[:, None] * centers[1]
    poses = metadata["target_c2w"].copy()
    poses[:, :3, 3] = targets
    metadata["target_c2w"] = poses
    metadata["target_w2c"] = np.linalg.inv(poses)
    metadata["base_c2w"] = poses.copy()
    metadata["baseline"][:] = np.linalg.norm(centers[1] - centers[0])
    y, x = np.mgrid[0:180:2, 0:320:2]
    camera_points = np.stack(
        (
            (x + 0.5 - 160) / 200 * depth,
            (y + 0.5 - 90) / 200 * depth,
            np.full_like(x, depth, dtype=float),
        ),
        axis=-1,
    ).reshape(-1, 3)
    world = np.concatenate([camera_points + center for center in centers])
    source_clearance = [
        np.quantile(np.linalg.norm(world - center, axis=-1), 0.01) for center in centers
    ]
    metadata["source_clearance_reference"] = (1 - alpha) * source_clearance[
        0
    ] + alpha * source_clearance[1]
    for index, center in enumerate(targets):
        distances = np.linalg.norm(world - center, axis=-1)
        metadata["minimum_geometry_distance"][index] = distances.min()
        metadata["geometry_clearance_distance"][index] = np.quantile(distances, 0.01)
    metadata["distance_from_camera_a"] = np.linalg.norm(targets - centers[0], axis=-1)
    metadata["distance_from_camera_b"] = np.linalg.norm(targets - centers[1], axis=-1)
    return metadata


def _cached_dataset(tmp_path) -> DROIDPreprocessedDataset:
    source = tmp_path / "source"
    source.mkdir()
    calibration = tmp_path / "calibration.jsonl"
    calibration.write_text(json.dumps(_entry()) + "\n", encoding="utf-8")
    root = tmp_path / "cached"
    manifest = build_stage0_manifest(
        calibration,
        root,
        droid_root=source,
        target_retained_per_shard=4096,
    )
    episode = manifest["episodes"][0]
    path = shard_path(root, "final", 0)
    jpeg = JPEGCodecConfig(quality=95, subsampling="4:4:4")
    support = np.ones((4, 1, 256, 256), dtype=np.bool_)
    lager_jpegs = tuple(
        encode_jpeg(np.full((256, 256, 3), 20 + view * 40, dtype=np.uint8), jpeg)
        for view in range(4)
    )
    with IndexedTarWriter(
        path,
        stage="final",
        shard_id=0,
        schema_signature=manifest["schema_signature"],
    ) as writer:
        for retained_index in range(5):
            key = sample_key(retained_index)
            real = []
            for camera in range(2):
                image = np.zeros((180, 320, 3), dtype=np.uint8)
                image[..., camera] = retained_index * 30 + 10
                real.append(encode_jpeg(image, jpeg))
            writer.add(
                key,
                "meta",
                pack_timestep_metadata(
                    global_retained_index=retained_index,
                    episode_index=0,
                    episode_id="episode-test",
                    retained_index=retained_index,
                    raw_timestep=retained_index * 3,
                ),
            )
            writer.add(key, "rgb", pack_real_rgb_record(real))
            writer.add(
                key,
                "depth",
                encode_numeric_array(
                    depth_meters_to_u16(
                        np.full((2, 180, 320), 1.0 + retained_index * 0.01)
                    )
                ),
            )
            if retained_index + 2 < 5:
                flow = np.zeros((2, 2, 180, 320), dtype=np.float32)
                flow[:, 0, 80:100, 140:180] = float(retained_index + 1)
                writer.add(
                    key,
                    "flow",
                    encode_numeric_array(flow_pixels_to_i16(flow)),
                )
            writer.add(
                key,
                "lager",
                pack_lager_record(
                    lager_jpegs,
                    _plane_lager_metadata(1.0 + retained_index * 0.01),
                    support,
                ),
            )
    assert episode["training_window_count"] == 1
    return DROIDPreprocessedDataset(
        DROIDDatasetConfig(
            preprocessed_root=str(root),
            split="validation",
            motion_crop=MotionCropConfig(min_size=180, max_size=180),
        )
    )


def test_lager_record_stores_safety_tiers_and_decodes_legacy_as_strict() -> None:
    metadata = _lager_metadata()
    metadata["safety_translation_limit_escalated"][1] = 1
    metadata["safety_thresholds_relaxed"][2] = 1
    metadata["minimum_source_coverage_threshold"][2] = 0.40
    metadata["minimum_geometry_distance_threshold"][2] = 0.05
    support = np.ones((4, 1, 256, 256), dtype=np.bool_)
    jpegs = (b"a", b"b", b"c", b"d")

    _images, decoded, _support = unpack_lager_record(
        pack_lager_record(jpegs, metadata, support)
    )
    assert decoded["safety_translation_limit_escalated"].tolist() == [
        False,
        True,
        False,
        False,
    ]
    assert decoded["safety_thresholds_relaxed"].tolist() == [
        False,
        False,
        True,
        False,
    ]
    np.testing.assert_allclose(
        decoded["minimum_source_coverage_threshold"], [0.60, 0.60, 0.40, 0.60]
    )
    np.testing.assert_allclose(decoded["source_clearance_reference"], 0.10)

    v3_floats = np.concatenate(
        [
            np.asarray(metadata[name], dtype="<f4").reshape(-1)
            for name, _shape in record_codec._V3_FLOAT_FIELDS
        ]
    )
    v3_ints = np.concatenate(
        [
            np.asarray(metadata[name], dtype="<i2").reshape(-1)
            for name, _shape in record_codec._INT16_FIELDS
        ]
    )
    v3_flags = np.concatenate(
        [
            np.asarray(metadata[name], dtype=np.uint8).reshape(-1)
            for name, _shape in record_codec._UINT8_FIELDS
        ]
    )
    v3_metadata = (
        record_codec._LAGER_HEADER.pack(
            record_codec.LAGER_V3_METADATA_MAGIC,
            v3_floats.size,
            v3_ints.size,
        )
        + v3_floats.tobytes()
        + v3_ints.tobytes()
        + v3_flags.tobytes()
    )
    v3_record = pack_byte_strings(
        (*jpegs, v3_metadata, pack_support_masks(support)),
        magic=record_codec.LAGER_V3_RECORD_MAGIC,
    )
    _images, v3_decoded, _support = unpack_lager_record(v3_record)
    assert "source_clearance_reference" not in v3_decoded
    assert v3_decoded["safety_thresholds_relaxed"].tolist() == [
        False,
        False,
        True,
        False,
    ]
    np.testing.assert_allclose(
        v3_decoded["minimum_geometry_distance_threshold"],
        [0.08, 0.08, 0.05, 0.08],
    )

    legacy_floats = np.concatenate(
        [
            np.asarray(metadata[name], dtype="<f4").reshape(-1)
            for name, _shape in record_codec._LEGACY_FLOAT_FIELDS
        ]
    )
    legacy_ints = np.concatenate(
        [
            np.asarray(metadata[name], dtype="<i2").reshape(-1)
            for name, _shape in record_codec._INT16_FIELDS
        ]
    )
    legacy_flags = np.concatenate(
        [
            np.asarray(metadata[name], dtype=np.uint8).reshape(-1)
            for name, _shape in record_codec._LEGACY_UINT8_FIELDS
        ]
    )
    legacy_metadata = (
        record_codec._LAGER_HEADER.pack(
            record_codec.LEGACY_LAGER_METADATA_MAGIC,
            legacy_floats.size,
            legacy_ints.size,
        )
        + legacy_floats.tobytes()
        + legacy_ints.tobytes()
        + legacy_flags.tobytes()
    )
    legacy_record = pack_byte_strings(
        (*jpegs, legacy_metadata, pack_support_masks(support)),
        magic=record_codec.LEGACY_LAGER_RECORD_MAGIC,
    )
    _images, legacy_decoded, _support = unpack_lager_record(legacy_record)
    assert not legacy_decoded["safety_translation_limit_escalated"].any()
    assert not legacy_decoded["safety_thresholds_relaxed"].any()
    np.testing.assert_allclose(
        legacy_decoded["minimum_source_coverage_threshold"], 0.60
    )
    np.testing.assert_allclose(
        legacy_decoded["minimum_geometry_distance_threshold"], 0.08
    )


def test_v4_lager_integrity_recomputes_source_calibrated_clearance() -> None:
    metadata = _lager_metadata()
    metadata["safety_thresholds_relaxed"][2] = 1
    metadata["minimum_source_coverage_threshold"][2] = 0.40
    metadata["source_clearance_reference"][2] = 0.04
    metadata["minimum_geometry_distance_threshold"][2] = 0.032
    _validate_pose_metadata(metadata, record_magic=record_codec.LAGER_RECORD_MAGIC)

    metadata["minimum_geometry_distance_threshold"][2] = 0.031
    with pytest.raises(ValueError, match="clearance-threshold metadata"):
        _validate_pose_metadata(metadata, record_magic=record_codec.LAGER_RECORD_MAGIC)


def test_cached_dataset_reconstructs_exact_gap6_history_and_modalities(
    tmp_path,
) -> None:
    dataset = _cached_dataset(tmp_path)
    item = dataset[0]
    assert item["history_raw_timesteps"].tolist() == [0, 6, 12]
    assert item["history_retained_indices"].tolist() == [0, 2, 4]
    assert int(item["retained_raw_stride"]) == 3
    assert int(item["temporal_gap_raw"]) == 6
    assert item["representation_histories"].shape == (2, 3, 3, 224, 224)
    assert item["representation_middle_motion"].shape == (2, 1, 224, 224)
    assert item["target_rgb"].shape == (3, 2, 3, 224, 224)
    assert item["target_depth"].shape == (3, 2, 1, 224, 224)
    assert item["target_flow"].shape == (2, 2, 2, 224, 224)
    assert item["target_flow_validity"].shape == (2, 2, 1, 224, 224)
    assert item["novel_rgb"].shape == (3, 4, 3, 256, 256)
    assert item["novel_c2w"].shape == (3, 4, 4, 4)
    assert item["novel_support_mask"].shape == (3, 4, 1, 256, 256)
    assert item["novel_pose_metadata"]["alpha"].shape == (3, 4)
    assert item["novel_pose_metadata"]["safety_thresholds_relaxed"].shape == (3, 4)
    assert item["representation_validity"].any()
    assert not item["representation_validity"].all()
    batch = droid_collate([item, item])
    assert batch["target_rgb"].shape == (2, 3, 2, 3, 224, 224)
    assert batch["novel_rgb"].shape == (2, 3, 4, 3, 256, 256)
    assert batch["episode_id"] == ["episode-test", "episode-test"]


def test_pose_matrix_changes_cannot_hide_behind_unchanged_metadata() -> None:
    metadata = _lager_metadata()
    metadata["target_c2w"] = metadata["target_c2w"].copy()
    metadata["target_c2w"][0, 0, 3] += 0.10
    metadata["target_w2c"] = np.linalg.inv(metadata["target_c2w"])
    with pytest.raises(ValueError, match="translation metadata"):
        _validate_pose_metadata(metadata, record_magic=record_codec.LAGER_RECORD_MAGIC)
    metadata = _lager_metadata()
    angle = np.deg2rad(2)
    c, s = np.cos(angle), np.sin(angle)
    metadata["target_c2w"] = metadata["target_c2w"].copy()
    metadata["target_c2w"][0, :3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    metadata["target_w2c"] = np.linalg.inv(metadata["target_c2w"])
    with pytest.raises(ValueError, match="rotation metadata"):
        _validate_pose_metadata(metadata, record_magic=record_codec.LAGER_RECORD_MAGIC)


def test_self_consistent_clearance_metadata_must_agree_with_depth() -> None:
    metadata = _plane_lager_metadata(1.0)
    depth = np.ones((2, 180, 320), dtype=np.float32)
    _validate_pose_against_depth(metadata, depth, _entry())
    metadata["source_clearance_reference"] *= 0.5
    metadata["safety_thresholds_relaxed"][:] = True
    metadata["minimum_source_coverage_threshold"][:] = 0.4
    metadata["minimum_geometry_distance_threshold"][:] = np.clip(
        0.8 * metadata["source_clearance_reference"], 0.02, 0.05
    )
    _validate_pose_metadata(metadata, record_magic=record_codec.LAGER_RECORD_MAGIC)
    with pytest.raises(ValueError, match="reference disagrees with cached depth"):
        _validate_pose_against_depth(metadata, depth, _entry())


def test_cached_storage_contains_timeline_records_not_duplicated_windows(
    tmp_path,
) -> None:
    dataset = _cached_dataset(tmp_path)
    assert len(dataset) == 1
    reader = IndexedTarReader(shard_path(dataset.config.preprocessed_root, "final", 0))
    # Five retained records appear once each; only the first three carry flow.
    assert sum(name.endswith(".meta") for name in reader.entries) == 5
    assert sum(name.endswith(".rgb") for name in reader.entries) == 5
    assert sum(name.endswith(".depth") for name in reader.entries) == 5
    assert sum(name.endswith(".lager") for name in reader.entries) == 5
    assert sum(name.endswith(".flow") for name in reader.entries) == 3
    assert not any("window" in name for name in reader.entries)
    reader.close()


def test_episode_grouped_distributed_sampler_has_equal_disjoint_rank_shards() -> None:
    class _Dataset:
        episode_index_ranges = ((0, 4), (4, 7), (7, 12))

        def __len__(self):
            return 12

    dataset = _Dataset()
    first = EpisodeGroupedDistributedSampler(
        dataset, num_replicas=2, rank=0, shuffle=True, seed=9, drop_last=True
    )
    second = EpisodeGroupedDistributedSampler(
        dataset, num_replicas=2, rank=1, shuffle=True, seed=9, drop_last=True
    )
    first.set_epoch(3)
    second.set_epoch(3)
    rank_zero = list(first)
    rank_one = list(second)
    assert len(rank_zero) == len(rank_one) == 6
    assert set(rank_zero).isdisjoint(rank_one)
    assert sorted(rank_zero + rank_one) == list(range(12))


def test_final_integrity_checker_scans_payloads_and_actual_loader(tmp_path) -> None:
    dataset = _cached_dataset(tmp_path)
    report = verify_stage0_dataset(
        IntegrityConfig(
            root=dataset.config.preprocessed_root,
            random_samples=5,
            full_payload_scan=True,
            decode_all_jpegs=True,
            loader_windows=1,
        )
    )
    assert report["status"] == "passed"
    assert report["samples_payload_validated"] == 5
    assert report["actual_shards"] == 1
    assert report["lagernvs"]["independent_depth_geometry_timestamps_checked"] == 5
    assert report["loader"]["windows_checked"] == 1
