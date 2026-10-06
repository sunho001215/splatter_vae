from __future__ import annotations

import json

import numpy as np
import pytest

from dataset.droid.codecs import (
    FLOW_INVALID_I16,
    JPEGCodecConfig,
    NumericCodecConfig,
    decode_jpeg,
    decode_numeric_array,
    depth_meters_to_u16,
    depth_u16_to_meters,
    encode_jpeg,
    encode_numeric_array,
    flow_i16_to_pixels,
    flow_pixels_to_i16,
    pack_byte_strings,
    pack_support_masks,
    unpack_byte_strings,
    unpack_support_masks,
)
from dataset.droid.integrity import (
    BoundedPrioritySample,
    classify_partial_artifacts,
    write_final_manifest,
)
from dataset.droid.preprocessed_manifest import (
    build_stage0_manifest,
    load_stage0_manifest,
    resolve_window,
    retained_raw_indices,
)
from dataset.droid.shards import (
    IndexedTarReader,
    IndexedTarWriter,
    quarantine_incomplete_shard,
    shard_is_complete,
    validate_indexed_shard,
)
from preprocessing.lagernvs.pose import pose_sampler_contract
from scripts.audit_lagernvs_pose_safety import _report_is_complete
from scripts.preprocess_droid_stage0 import _check_lagernvs_pose_audit


def _camera(serial: str) -> dict:
    identity = np.eye(4).tolist()
    return {
        "serial": serial,
        "intrinsics_rlds": [[200.0, 0.0, 160.0], [0.0, 200.0, 90.0], [0.0, 0.0, 1.0]],
        "c2w": identity,
        "w2c": identity,
        "pose_flag": "direct",
    }


def test_jpeg_codec_dimensions_and_metadata_are_deterministic() -> None:
    y, x = np.mgrid[:180, :320]
    image = np.stack((x % 256, y % 256, (x + y) % 256), axis=-1).astype(np.uint8)
    config = JPEGCodecConfig(quality=95, subsampling="4:4:4")
    first = encode_jpeg(image, config)
    second = encode_jpeg(image, config)
    assert first == second
    assert decode_jpeg(first).shape == (180, 320, 3)
    metadata = config.metadata()
    assert metadata["quality"] == 95
    assert metadata["subsampling"] == "4:4:4"


def test_depth_uint16_mm_and_bitshuffle_zstd_round_trip() -> None:
    depth = np.array([[np.nan, -1.0, 0.0, 0.001, 1.2344, 99.0]], dtype=np.float32)
    encoded = depth_meters_to_u16(depth)
    assert encoded.tolist() == [[0, 0, 0, 1, 1234, 65535]]
    blob = encode_numeric_array(encoded, NumericCodecConfig(compression_level=3))
    recovered = decode_numeric_array(blob)
    np.testing.assert_array_equal(recovered, encoded)
    meters, valid = depth_u16_to_meters(recovered)
    assert valid.tolist() == [[False, False, False, True, True, True]]
    np.testing.assert_allclose(meters[0, 3:5], [0.001, 1.234], atol=1e-7)


def test_flow_int16_fixed_point_invalid_and_round_trip_error() -> None:
    rng = np.random.default_rng(7)
    flow = rng.uniform(-100.0, 100.0, size=(2, 180, 320)).astype(np.float32)
    flow[:, 10, 20] = np.nan
    encoded = flow_pixels_to_i16(flow)
    assert np.all(encoded[:, 10, 20] == FLOW_INVALID_I16)
    blob = encode_numeric_array(encoded)
    decoded, valid = flow_i16_to_pixels(decode_numeric_array(blob))
    assert decoded.shape == (2, 180, 320)
    assert valid.shape == (1, 180, 320)
    assert not valid[0, 10, 20]
    error = np.abs(decoded[:, valid[0]] - flow[:, valid[0]])
    assert float(error.max()) <= 1.0 / 128.0 + 1e-6


def test_framed_jpegs_and_bitpacked_support_masks_round_trip() -> None:
    values = (b"jpeg-a", b"jpeg-b", b"jpeg-c", b"jpeg-d")
    payload = pack_byte_strings(values, magic=b"ST0LJPG1")
    assert unpack_byte_strings(payload, magic=b"ST0LJPG1", expected_count=4) == values
    masks = np.zeros((4, 1, 256, 256), dtype=np.bool_)
    masks[0, :, 17:91, 12:201] = True
    masks[3, :, ::3, ::5] = True
    np.testing.assert_array_equal(
        unpack_support_masks(pack_support_masks(masks)), masks
    )


def test_indexed_tar_is_atomic_indexed_and_restart_verifiable(tmp_path) -> None:
    tar_path = tmp_path / "stage0-00000.tar"
    with IndexedTarWriter(
        tar_path, stage="final", shard_id=0, schema_signature="unit-test"
    ) as writer:
        writer.add("0000000000", "rgb", b"rgb-bytes")
        writer.add("0000000000", "depth", b"depth-bytes")
    assert tar_path.is_file()
    assert not list(tmp_path.glob("*.partial"))
    completion = validate_indexed_shard(tar_path, deep=True)
    assert completion["sample_count"] == 1
    reader = IndexedTarReader(tar_path)
    assert reader.read("0000000000", "rgb", verify=True) == b"rgb-bytes"
    reader.close()
    with np.testing.assert_raises(FileExistsError):
        # A pre-existing partial is never silently overwritten on restart.
        partial = tar_path.with_suffix(".tar.partial")
        partial.write_bytes(b"unfinished")
        IndexedTarWriter(
            tar_path, stage="final", shard_id=0, schema_signature="unit-test"
        )


def test_partial_artifact_classification_ignores_archived_child_trees(tmp_path) -> None:
    active = tmp_path / "staging" / "da3" / "shards" / "da3-00000.tar.partial"
    recovery = tmp_path / "recovery" / "old" / "da3-00000.tar.partial"
    pilot = tmp_path / "pilot" / "staging" / "rgb-00000.tar.partial"
    for path in (active, recovery, pilot):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"unfinished")
    unexpected, archived = classify_partial_artifacts(tmp_path)
    assert unexpected == ["staging/da3/shards/da3-00000.tar.partial"]
    assert archived == [
        "pilot/staging/rgb-00000.tar.partial",
        "recovery/old/da3-00000.tar.partial",
    ]


def test_incomplete_or_stale_shard_is_quarantined_and_rebuilt_on_resume(
    tmp_path,
) -> None:
    tar_path = tmp_path / "shards" / "stage0-00000.tar"
    with IndexedTarWriter(
        tar_path, stage="final", shard_id=0, schema_signature="stale"
    ) as writer:
        writer.add("0000000000", "rgb", b"old")
    assert shard_is_complete(tar_path, schema_signature="stale", verify_checksums=True)
    moved = quarantine_incomplete_shard(
        tar_path,
        recovery_root=tmp_path / "recovery",
        schema_signature="current",
    )
    assert len(moved) == 3
    assert not tar_path.exists()

    with IndexedTarWriter(
        tar_path, stage="final", shard_id=0, schema_signature="current"
    ) as writer:
        writer.add("0000000000", "rgb", b"new")
    assert shard_is_complete(
        tar_path, schema_signature="current", verify_checksums=True
    )
    with pytest.raises(ValueError, match="Refusing to quarantine"):
        quarantine_incomplete_shard(
            tar_path,
            recovery_root=tmp_path / "recovery",
            schema_signature="current",
        )


def test_integrity_distribution_sample_is_bounded_and_deterministic() -> None:
    first = BoundedPrioritySample(capacity=257, seed=19, flush_size=101)
    second = BoundedPrioritySample(capacity=257, seed=19, flush_size=101)
    for offset in range(20):
        values = np.arange(offset * 100, (offset + 1) * 100, dtype=np.float32)
        first.add(values)
        second.add(values)
    assert first.result().shape == (257,)
    np.testing.assert_array_equal(first.result(), second.result())


def test_integrity_distribution_sample_supports_vector_items_and_per_item_limit() -> None:
    values = np.arange(300, dtype=np.float32).reshape(100, 3)
    first = BoundedPrioritySample(
        capacity=20, seed=23, item_shape=(3,), flush_size=7
    )
    second = BoundedPrioritySample(
        capacity=20, seed=23, item_shape=(3,), flush_size=7
    )
    first.add(values, maximum_per_item=10)
    second.add(values, maximum_per_item=10)
    assert first.result().shape == (10, 3)
    np.testing.assert_array_equal(first.result(), second.result())
    input_rows = {tuple(row) for row in values.tolist()}
    assert all(tuple(row) in input_rows for row in first.result().tolist())
    # Stratification spans the full item instead of selecting one contiguous region.
    assert float(first.result()[-1, 0] - first.result()[0, 0]) > 250.0


def test_integrity_distribution_sample_rejects_wrong_vector_shape() -> None:
    sample = BoundedPrioritySample(capacity=10, seed=5, item_shape=(3,))
    with pytest.raises(ValueError, match="trailing shape"):
        sample.add(np.zeros((4, 2), dtype=np.float32))


def test_lagernvs_launch_gate_requires_current_complete_pose_audits(tmp_path) -> None:
    manifest = {
        "schema_signature": "dataset-signature",
        "shards": [
            {"shard_id": 0, "retained_count": 7},
            {"shard_id": 1, "retained_count": 9},
        ],
    }
    report_root = tmp_path / "reports" / "lagernvs_pose_safety"
    report_root.mkdir(parents=True)
    signature = pose_sampler_contract()["signature"]
    for shard in manifest["shards"]:
        retained = int(shard["retained_count"])
        report = {
            "pose_audit_schema_version": 2,
            "schema_signature": "dataset-signature",
            "pose_contract_signature": signature,
            "status": "pass",
            "shard_id": int(shard["shard_id"]),
            "statistics": {
                "retained_timestamps": retained,
                "targets": retained * 4,
                "fallback_targets": 1,
                "exceptional_translation_targets": 2,
                "exceptional_safety_threshold_targets": 3,
            },
        }
        (report_root / f"shard-{int(shard['shard_id']):05d}.json").write_text(
            json.dumps(report), encoding="utf-8"
        )
    result = _check_lagernvs_pose_audit(tmp_path, manifest)
    assert result["audited_shards"] == 2
    assert result["statistics"]["retained_timestamps"] == 16
    assert result["statistics"]["exceptional_safety_threshold_targets"] == 6

    stale = report_root / "shard-00001.json"
    payload = json.loads(stale.read_text(encoding="utf-8"))
    payload["shard_id"] = 0
    stale.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="stale or failed"):
        _check_lagernvs_pose_audit(tmp_path, manifest)
    payload["shard_id"] = 1
    payload["pose_contract_signature"] = "stale"
    stale.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="stale or failed"):
        _check_lagernvs_pose_audit(tmp_path, manifest)


def test_pose_audit_resume_rejects_incomplete_shard_accounting(tmp_path) -> None:
    signature = pose_sampler_contract()["signature"]
    path = tmp_path / "shard-00007.json"
    report = {
        "pose_audit_schema_version": 2,
        "schema_signature": "dataset-signature",
        "pose_contract_signature": signature,
        "status": "pass",
        "shard_id": 7,
        "statistics": {"retained_timestamps": 11, "targets": 44},
    }
    path.write_text(json.dumps(report), encoding="utf-8")
    assert _report_is_complete(
        path,
        "dataset-signature",
        signature,
        shard_id=7,
        retained_timestamps=11,
    )

    report["statistics"]["targets"] = 40
    path.write_text(json.dumps(report), encoding="utf-8")
    assert not _report_is_complete(
        path,
        "dataset-signature",
        signature,
        shard_id=7,
        retained_timestamps=11,
    )


def test_manifest_uses_stage0_only_stride_three_and_gap_six(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    calibration = tmp_path / "calibration.jsonl"
    entries = [
        {
            "episode_id": "valid-a",
            "rlds_split": "train",
            "rlds_ordinal": 2,
            "rlds_path": "LAB/success/date/a",
            "num_steps": 17,
            "valid": True,
            "dataset_split": "train",
            "exterior_cameras": [_camera("a"), _camera("b")],
        },
        {
            "episode_id": "invalid",
            "rlds_split": "train",
            "rlds_ordinal": 3,
            "num_steps": 99,
            "valid": False,
            "failure_reason": "invalid_intrinsics",
        },
    ]
    calibration.write_text("".join(json.dumps(value) + "\n" for value in entries))
    output = tmp_path / "derived"
    manifest = build_stage0_manifest(
        calibration,
        output,
        droid_root=source,
        target_retained_per_shard=4,
    )
    assert manifest["counts"]["eligible_episodes"] == 1
    assert manifest["counts"]["raw_timesteps"] == 17
    assert manifest["counts"]["retained_timesteps"] == 6
    assert manifest["counts"]["training_windows"] == 2
    assert retained_raw_indices(17) == (0, 3, 6, 9, 12, 15)
    episode = manifest["episodes"][0]
    window = resolve_window(episode, 1)
    assert window.retained_indices == (1, 3, 5)
    assert window.raw_timesteps == (3, 9, 15)


def test_manifest_signatures_bind_codec_and_reject_tampering(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    calibration = tmp_path / "calibration.jsonl"
    entry = {
        "episode_id": "valid-a",
        "rlds_split": "train",
        "rlds_ordinal": 0,
        "rlds_path": "LAB/success/date/a",
        "num_steps": 13,
        "valid": True,
        "dataset_split": "train",
        "exterior_cameras": [_camera("a"), _camera("b")],
    }
    calibration.write_text(json.dumps(entry) + "\n", encoding="utf-8")
    root = tmp_path / "derived"
    q95 = build_stage0_manifest(calibration, root, droid_root=source, jpeg_quality=95)
    q97 = build_stage0_manifest(
        calibration,
        tmp_path / "q97",
        droid_root=source,
        jpeg_quality=97,
    )
    assert q95["schema_signature"] != q97["schema_signature"]
    assert q95["pipeline_signature"] != q97["pipeline_signature"]
    with pytest.raises(FileExistsError, match="incompatible dataset manifest"):
        build_stage0_manifest(calibration, root, droid_root=source, jpeg_quality=97)

    manifest_path = root / "manifest.json"
    tampered = json.loads(manifest_path.read_text(encoding="utf-8"))
    tampered["encoding"]["real_rgb"]["quality"] = 97
    manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest signature is invalid"):
        load_stage0_manifest(root)


def test_full_finalization_requires_environment_and_source_provenance(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    calibration = tmp_path / "calibration.jsonl"
    entry = {
        "episode_id": "valid-a",
        "rlds_split": "train",
        "rlds_ordinal": 0,
        "rlds_path": "LAB/success/date/a",
        "num_steps": 13,
        "valid": True,
        "dataset_split": "train",
        "exterior_cameras": [_camera("a"), _camera("b")],
    }
    calibration.write_text(json.dumps(entry) + "\n", encoding="utf-8")
    root = tmp_path / "derived"
    manifest = build_stage0_manifest(calibration, root, droid_root=source, mode="full")
    metadata = root / "metadata"
    metadata.mkdir(exist_ok=True)
    (metadata / "lagernvs-pose-safety-contract.json").write_text(
        json.dumps({"dataset_schema_signature": manifest["schema_signature"]}),
        encoding="utf-8",
    )
    integrity = {
        "status": "passed",
        "dataset_schema_signature": manifest["schema_signature"],
    }
    with pytest.raises(FileNotFoundError, match="environment inventories"):
        write_final_manifest(root, integrity)

    environment_root = metadata / "environments"
    environment_root.mkdir()
    for name in ("training", "da3", "megaflow", "lagernvs"):
        (environment_root / f"{name}.json").write_text(
            json.dumps({"stage_environment": name}), encoding="utf-8"
        )
    with pytest.raises(FileNotFoundError, match="source fingerprints"):
        write_final_manifest(root, integrity)

    fingerprint = {
        "root": str(source.resolve()),
        "mode": "metadata",
        "file_count": 0,
        "total_bytes": 0,
        "sha256": "unchanged",
    }
    for name in ("source-tree-before-full-run", "source-tree-after-full-run"):
        (metadata / f"{name}.json").write_text(
            json.dumps(fingerprint), encoding="utf-8"
        )
    final_path = write_final_manifest(root, integrity)
    final = json.loads(final_path.read_text(encoding="utf-8"))
    assert set(final["environment_provenance"]) == {
        "training",
        "da3",
        "megaflow",
        "lagernvs",
    }
    assert final["source_read_only_verification"]["unchanged"] is True
