"""Viewer synchronization follows the stored episode policy, then the root's pipeline config."""

from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from s4d.realworld import io as rio
from s4d.realworld import pipeline
from s4d.realworld.viewer_data import describe, load_episode


@pytest.fixture
def cameras(monkeypatch):
    # 15 Hz, camera B offset by 22.9 ms and missing frame 2: strict 22.2 ms pairing fails.
    ref = np.arange(6, dtype=np.int64) * 66_666_667
    stamps = {"a": ref, "b": np.delete(ref, 2) + 22_900_000, "c": ref.copy()}
    meta = SimpleNamespace(stream_timestamps_ns=stamps,
                           views=lambda: [SimpleNamespace(stream_id=s) for s in stamps])
    monkeypatch.setattr(pipeline, "load_meta", lambda *_: meta)
    monkeypatch.setattr(pipeline, "accepted_calibration", lambda *_: np.eye(4))
    return meta


def test_viewer_uses_config_then_record_and_reports_excluded_frames(tmp_path, cameras):
    policy = {"max_sync_ns": 30_000_000, "allow_unmatched": True}
    path = tmp_path / "pipeline.yaml"
    path.write_text(yaml.safe_dump({"stages": {"mapanything_depth": policy}}))
    ve = load_episode(tmp_path, "ep")
    np.testing.assert_array_equal(ve.sets["index"], [[0, 0, 0], [1, 1, 1], [3, 2, 3], [4, 3, 4], [5, 4, 5]])
    assert describe(ve)["unmatched_frames"] == {"a": [2], "b": [], "c": [2]}
    assert ve.n_sets == 5
    rio.run_stage(rio.stage_dir(tmp_path, "ep", "mapanything_depth"), stage="mapanything_depth", version="test",
                  config=policy, inputs={}, build=lambda _: {})
    path.write_text(yaml.safe_dump({"stages": {"mapanything_depth": {"max_sync_ns": 1}}}))
    np.testing.assert_array_equal(load_episode(tmp_path, "ep").sets["index"], ve.sets["index"])


def test_viewer_defaults_to_strict_and_honours_strict_record_tolerance(tmp_path, cameras):
    with pytest.raises(ValueError, match="frames do not pair 1:1"):
        load_episode(tmp_path, "ep")
    cameras.stream_timestamps_ns["b"] = cameras.stream_timestamps_ns["a"] + 22_900_000
    rio.run_stage(rio.stage_dir(tmp_path, "ep", "mapanything_depth"), stage="mapanything_depth", version="test",
                  config={"max_sync_ns": 30_000_000}, inputs={}, build=lambda _: {})
    assert load_episode(tmp_path, "ep").n_sets == 6


def test_viewer_falls_back_to_export_tolerance_for_roots_without_mapanything(tmp_path, cameras):
    # sensor/stereo depth variant roots keep no mapanything_depth config: the export's tolerance (mutual pairing) applies
    (tmp_path / "pipeline.yaml").write_text(yaml.safe_dump({"stages": {"export": {"max_sync_ns": 30_000_000}}}))
    ve = load_episode(tmp_path, "ep")
    np.testing.assert_array_equal(ve.sets["index"], [[0, 0, 0], [1, 1, 1], [3, 2, 3], [4, 3, 4], [5, 4, 5]])
