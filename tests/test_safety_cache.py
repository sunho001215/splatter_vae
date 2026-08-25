from __future__ import annotations

import os

import numpy as np
import pytest

from dataset.droid.cache import (
    CacheProvenance,
    HDF5CacheReader,
    HDF5ShardWriter,
    cache_item_key,
    calibration_manifest_version,
    sequence_cache_key,
)
from dataset.droid.safety import DEFAULT_DROID_ROOT, validate_derived_root
from preprocessing.common import configure_external_model_caches


def test_derived_outputs_cannot_be_inside_droid_source(tmp_path) -> None:
    source = tmp_path / "droid"
    source.mkdir()
    with pytest.raises(ValueError):
        validate_derived_root(source / "cache", source)
    with pytest.raises(ValueError):
        validate_derived_root(tmp_path, source)
    assert (
        validate_derived_root(tmp_path / "derived", source)
        == (tmp_path / "derived").resolve()
    )


def test_default_read_only_source_is_the_actual_droid_mount() -> None:
    assert str(DEFAULT_DROID_ROOT) == "/home/ws/data/droid"
    with pytest.raises(ValueError):
        validate_derived_root("/home/ws/data/droid/manifests")


def test_foundation_model_caches_are_forced_under_external_derived_root(
    tmp_path, monkeypatch
) -> None:
    names = ("TORCH_HOME", "HF_HOME", "XDG_CACHE_HOME")
    previous = {name: os.environ.get(name) for name in names}
    source = tmp_path / "droid"
    source.mkdir()
    locations = configure_external_model_caches(tmp_path / "derived", source)
    assert all(str(tmp_path / "derived") in value for value in locations.values())
    with pytest.raises(ValueError):
        configure_external_model_caches(source / "cache", source)
    for name, value in previous.items():
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)


def test_sharded_cache_roundtrip_and_provenance(tmp_path) -> None:
    source = tmp_path / "droid"
    source.mkdir()
    cache_root = tmp_path / "derived" / "xlens"
    provenance = CacheProvenance(
        teacher_name="X-Lens",
        checkpoint="xlens-test.safetensors",
        teacher_version="test-revision",
        calibration_version="test-calibration",
        preprocessing_version="xlens-droid-v1",
        resolution=(320, 180),
    )
    key = cache_item_key("episode", "111", 4)
    writer = HDF5ShardWriter(
        cache_root,
        provenance,
        shard_prefix="depth",
        items_per_shard=1,
        droid_root=source,
    )
    writer.add(
        key,
        {
            "metric_depth": np.ones((180, 320), dtype=np.float16),
            "confidence": np.full((180, 320), 0.8, dtype=np.float16),
            "validity": np.ones((180, 320), dtype=np.uint8),
        },
        {"episode_id": "episode", "camera_id": "111", "frame_index": 4},
    )
    index = writer.close()
    reader = HDF5CacheReader(index, expected=provenance)
    item = reader.read_depth("episode", "111", 4)
    assert item["metric_depth"].shape == (180, 320)
    assert np.all(item["validity"] == 1)
    wrong = CacheProvenance(
        teacher_name="X-Lens",
        checkpoint="different",
        teacher_version="test-revision",
        calibration_version="test-calibration",
        preprocessing_version="xlens-droid-v1",
        resolution=(320, 180),
    )
    with pytest.raises(ValueError, match="provenance mismatch"):
        HDF5CacheReader(index, expected=wrong)


def test_sequence_cache_slices_frames_without_per_frame_index_entries(tmp_path) -> None:
    provenance = CacheProvenance(
        teacher_name="WAFT",
        checkpoint="waft.pth",
        teacher_version="test",
        calibration_version="test",
        preprocessing_version="test",
        resolution=(320, 180),
    )
    cache_root = tmp_path / "waft"
    with HDF5ShardWriter(cache_root, provenance, shard_prefix="flow") as writer:
        writer.add(
            sequence_cache_key("episode", "exterior_1"),
            {
                "forward_flow_gap_3": np.ones((4, 180, 320, 2), np.float16),
                "validity_gap_3": np.ones((4, 180, 320), np.uint8),
            },
            {"episode_id": "episode", "logical_camera_id": "exterior_1"},
        )
    reader = HDF5CacheReader(cache_root / "flow-index.json")
    item = reader.read_flow("episode", "exterior_1", 2, 3)
    assert item["forward_flow"].shape == (180, 320, 2)
    assert len(reader.items) == 1


def test_manifest_identity_and_cache_compatibility_are_enforced(tmp_path) -> None:
    manifest = tmp_path / "calibration.jsonl"
    manifest.write_text('{"episode_id":"one"}\n', encoding="utf-8")
    version = calibration_manifest_version(manifest)
    provenance = CacheProvenance(
        teacher_name="X-Lens",
        checkpoint="checkpoint:sha256:test",
        teacher_version="revision",
        calibration_version=version,
        preprocessing_version="test",
        resolution=(320, 180),
    )
    with HDF5ShardWriter(
        tmp_path / "cache", provenance, shard_prefix="depth"
    ) as writer:
        writer.add("item", {"value": np.ones(1)}, {})
    reader = HDF5CacheReader(tmp_path / "cache" / "depth-index.json")
    reader.require_compatible(
        teacher_name="X-Lens",
        calibration_version=version,
        checkpoint="checkpoint:sha256:test",
    )
    manifest.write_text('{"episode_id":"two"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="refusing to mix provenance"):
        reader.require_compatible(
            teacher_name="X-Lens",
            calibration_version=calibration_manifest_version(manifest),
        )
