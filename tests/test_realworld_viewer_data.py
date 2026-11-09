"""Viewer synchronization follows the root's pipeline config (the export's frame-pairing tolerance)."""

from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from s4d.realworld import pipeline
from s4d.realworld.stages.depth import configured_sets_policy
from s4d.realworld.viewer_data import describe, load_episode


@pytest.fixture
def cameras(monkeypatch):
    # 15 Hz, camera B offset by 22.9 ms and missing frame 2: strict 22.2 ms pairing fails.
    ref = np.arange(6, dtype=np.int64) * 66_666_667
    stamps = {"a": ref, "b": np.delete(ref, 2) + 22_900_000, "c": ref.copy()}
    meta = SimpleNamespace(stream_timestamps_ns=stamps,
                           views=lambda: [SimpleNamespace(stream_id=s, kind="rgbd") for s in stamps])
    monkeypatch.setattr(pipeline, "load_meta", lambda *_: meta)
    monkeypatch.setattr(pipeline, "accepted_calibration", lambda *_: np.eye(4))
    return meta


def test_viewer_uses_the_export_tolerance_with_mutual_pairing_and_reports_excluded_frames(tmp_path, cameras):
    (tmp_path / "pipeline.yaml").write_text(yaml.safe_dump({"stages": {"export": {"max_sync_ns": 30_000_000}}}))
    assert configured_sets_policy(tmp_path, "ep") == {"max_sync_ns": 30_000_000, "allow_unmatched": True}
    ve = load_episode(tmp_path, "ep")
    np.testing.assert_array_equal(ve.sets["index"], [[0, 0, 0], [1, 1, 1], [3, 2, 3], [4, 3, 4], [5, 4, 5]])
    assert describe(ve)["unmatched_frames"] == {"a": [2], "b": [], "c": [2]}
    assert ve.n_sets == 5
    # an explicit config argument (e.g. the contract checks' pipeline config) takes precedence over the root's file
    assert configured_sets_policy(tmp_path, "ep", {"stages": {}}) == {}


def test_viewer_defaults_to_strict_pairing(tmp_path, cameras):
    with pytest.raises(ValueError, match="frames do not pair 1:1"):
        load_episode(tmp_path, "ep")  # no pipeline.yaml
    (tmp_path / "pipeline.yaml").write_text(yaml.safe_dump({"stages": {"export": {}}}))
    with pytest.raises(ValueError, match="frames do not pair 1:1"):
        load_episode(tmp_path, "ep")  # no export tolerance configured
    cameras.stream_timestamps_ns["b"] = cameras.stream_timestamps_ns["a"] + 20_000_000
    assert load_episode(tmp_path, "ep").n_sets == 6  # within the default third of a period
