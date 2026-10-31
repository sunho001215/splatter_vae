"""Episode contract round trip/validation, worker subprocess protocol (with the flagged MOCK worker only),
completion records, and the isolation rule (no existing module imports s4d.realworld)."""

import dataclasses
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from s4d.realworld import io as rio
from s4d.realworld.adapters import get_adapter
from s4d.realworld.contract import EpisodeContractError, meta_from_json, meta_to_json, validate_episode
from s4d.realworld.synthetic import rectified_pair
from s4d.realworld.workers import WorkerError, WorkerSpec, call_worker

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def meta(tmp_path):
    return get_adapter("synthetic", {"asset_dir": str(tmp_path / "a"), "frames": 4}).read_meta("synthetic_ep000")


def test_contract_round_trip_and_conventions(meta):
    doc, arrays = meta_to_json(meta)
    back = meta_from_json(doc, arrays)
    assert back.streams == meta.streams and back.stereo == meta.stereo and back.robot == meta.robot
    assert np.array_equal(back.trajectory.joint_positions, meta.trajectory.joint_positions)
    assert validate_episode(back).ok
    # integer-centre intrinsics become continuous (+0.5), the repository convention
    k = meta.streams[0].intrinsics
    assert meta.streams[0].intrinsics.K()[0, 2] == pytest.approx(k.cx + 0.5)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda m: dataclasses.replace(m, streams=(dataclasses.replace(m.streams[0], modality="rgb"),) + m.streams[1:]),
         "rgb streams need 3 channels"),
        (lambda m: dataclasses.replace(m, stereo=(dataclasses.replace(m.stereo[0], baseline_m=0.2),) + m.stereo[1:]),
         "baseline_m disagrees"),
        (lambda m: dataclasses.replace(m, stream_timestamps_ns={**m.stream_timestamps_ns,
                                                                 "rig0/left": np.array([3, 2, 1, 0])}),
         "not strictly increasing"),
        (lambda m: dataclasses.replace(m, schema_version="s4d.realworld.episode/0.9"), "schema"),
    ],
)
def test_contract_rejects_inconsistent_metadata(meta, mutate, message):
    report = validate_episode(mutate(meta))
    assert not report.ok and any(message in e for e in report.errors), report.errors
    with pytest.raises(EpisodeContractError):
        report.raise_if_errors()


def test_mock_worker_subprocess_protocol(tmp_path):
    left, right, depth, _, K = rectified_pair()
    spec = WorkerSpec("mock", sys.executable, timeout_s=120)
    with pytest.raises(WorkerError, match="mock output refused"):
        call_worker(spec, "stereo_disparity", {"max_disparity": 12},
                    {"left": left[None].astype(np.uint8), "right": right[None].astype(np.uint8)}, tmp_path / "w0")
    resp, out = call_worker(spec, "stereo_disparity", {"max_disparity": 12},
                            {"left": left[None], "right": right[None]}, tmp_path / "w1", allow_mock=True)
    assert resp["model"]["is_mock"] is True and out["disparity"].shape == (1, *left.shape)
    m = np.isfinite(out["disparity"][0])
    assert np.median(np.abs(out["disparity"][0][m] - K[0, 0] * 0.1 / depth[m])) < 0.3
    pinned = WorkerSpec("mock", sys.executable, expected_commit="0" * 40, timeout_s=120)
    with pytest.raises(WorkerError, match="pinned"):
        call_worker(pinned, "stereo_disparity", {}, {"left": left[None], "right": right[None]}, tmp_path / "w2",
                    allow_mock=True)
    with pytest.raises(WorkerError, match="unknown task"):
        call_worker(spec, "nope", {}, {"x": np.zeros(1)}, tmp_path / "w3", allow_mock=True)


def test_run_stage_is_atomic_and_restartable(tmp_path):
    calls = []

    def build(tmp):
        calls.append(1)
        (tmp / "a.txt").write_text("x")
        return {"n": 1}

    d = tmp_path / "stage"
    assert not rio.run_stage(d, stage="s", version="1", config={"a": 1}, inputs={}, build=build)["skipped"]
    assert rio.run_stage(d, stage="s", version="1", config={"a": 1}, inputs={}, build=build)["skipped"]

    def broken(tmp):
        (tmp / "partial").write_text("half")
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        rio.run_stage(d, stage="s", version="1", config={"a": 2}, inputs={}, build=broken)
    assert rio.read_record(d)["config"] == {"a": 1} and not (d / "partial").exists()  # old result intact
    assert not list(tmp_path.glob(".stage.building.*"))
    assert len(calls) == 1


def test_existing_code_does_not_import_realworld():
    out = subprocess.run(["git", "-C", str(REPO), "grep", "-l", "s4d.realworld", "--", "s4d", "scripts"],
                         capture_output=True, text=True).stdout.split()
    assert all(p.startswith("s4d/realworld/") for p in out), out


def test_upstream_worker_scripts_compile_and_never_import_upstream_at_module_level():
    """Upstream packages may only be imported inside the request handler (they live in other environments)."""
    import ast

    upstream = {"core", "omegaconf", "nets", "sam3", "torch"}
    for path in sorted((REPO / "s4d/realworld/workers").glob("*_worker.py")):
        tree = ast.parse(path.read_text())
        compile(tree, str(path), "exec")
        top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        names = {(n.module or "").split(".")[0] if isinstance(n, ast.ImportFrom) else n.names[0].name.split(".")[0]
                 for n in top}
        assert not names & upstream, (path.name, names)


def test_stage_lock_blocks_concurrent_builders_and_recovers_stale(tmp_path):
    import os

    d = tmp_path / "stage"
    lock = d.with_name(".stage.lock")
    lock.write_text(str(os.getpid()))  # a live builder (this process)
    with pytest.raises(RuntimeError, match="being built"):
        rio.run_stage(d, stage="s", version="1", config={}, inputs={}, build=lambda t: {})
    lock.write_text("999999999")  # dead pid -> stale lock is taken over
    assert not rio.run_stage(d, stage="s", version="1", config={}, inputs={}, build=lambda t: {})["skipped"]
    assert not lock.exists()


def test_configs_parse_and_pins_are_full_shas():
    import yaml

    pins = yaml.safe_load((REPO / "configs/realworld/upstream_pins.yaml").read_text())
    for name in ("foundation_stereo", "alltracker", "sam3", "dino_snerl"):
        sha = pins[name]["commit"]
        assert len(sha) == 40 and all(c in "0123456789abcdef" for c in sha)
    for name in ("pipeline_example", "template_adapter"):
        assert isinstance(yaml.safe_load((REPO / f"configs/realworld/{name}.yaml").read_text()), dict)
