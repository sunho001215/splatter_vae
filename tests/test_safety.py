from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from dataset.droid.safety import (
    DEFAULT_DERIVED_ROOT,
    DEFAULT_DROID_ROOT,
    assert_no_preprocessing_quality_hold,
    assert_source_fingerprint_unchanged,
    source_tree_fingerprint,
    validate_derived_root,
    write_source_fingerprint,
)
from preprocessing.common import configure_external_model_caches
from preprocessing.stage0.workflow import Stage0WorkerConfig, run_stage_worker


def test_quality_hold_blocks_launcher_and_direct_worker_before_loading(tmp_path):
    from scripts.preprocess_droid_stage0 import _check_full_gate

    assert_no_preprocessing_quality_hold(tmp_path)
    report = tmp_path / "reports" / "preprocessing_quality_hold.json"
    report.parent.mkdir()
    report.write_text("malformed reports also fail closed")
    with pytest.raises(RuntimeError, match="quality hold is active"):
        assert_no_preprocessing_quality_hold(tmp_path)
    with pytest.raises(RuntimeError, match="quality hold is active"):
        _check_full_gate(
            SimpleNamespace(root=str(tmp_path), allow_missing_pilot_gate=True),
            {"mode": "full"},
        )
    with pytest.raises(RuntimeError, match="quality hold is active"):
        run_stage_worker(SimpleNamespace(root=str(tmp_path)))


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
    assert str(DEFAULT_DERIVED_ROOT) == "/home/ws/data/droid_stage0_preprocessed"
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


def test_stage0_worker_rejects_outputs_and_model_cache_inside_source(tmp_path) -> None:
    source = tmp_path / "droid"
    source.mkdir()
    with pytest.raises(ValueError, match="inside the read-only DROID source"):
        Stage0WorkerConfig(
            root=str(source / "derived"),
            droid_root=str(source),
            stage="rgb",
        )
    with pytest.raises(ValueError, match="inside the read-only DROID source"):
        Stage0WorkerConfig(
            root=str(tmp_path / "derived"),
            droid_root=str(source),
            model_cache=str(source / "model-cache"),
            stage="rgb",
        )


def test_source_fingerprint_detects_mutation_and_is_written_atomically(
    tmp_path,
) -> None:
    source = tmp_path / "droid"
    source.mkdir()
    item = source / "episode.data"
    item.write_bytes(b"original")
    before = source_tree_fingerprint(source)
    output = tmp_path / "derived" / "source-fingerprint.json"
    assert write_source_fingerprint(output, source) == before
    assert output.is_file()
    assert not output.with_suffix(".json.partial").exists()

    item.write_bytes(b"changed-size")
    after = source_tree_fingerprint(source)
    with pytest.raises(RuntimeError, match="source-tree fingerprint changed"):
        assert_source_fingerprint_unchanged(before, after)
