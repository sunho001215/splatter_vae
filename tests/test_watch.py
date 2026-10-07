"""Event watcher and heartbeat: exactly one line per event, cursor survives restarts, errors surface."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / f"scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_watcher_reports_each_event_once_across_restarts(tmp_path, monkeypatch):
    watch = load("watch_events")
    for name, value in {
        "REPO": tmp_path,
        "REGISTRY": tmp_path / "registry.jsonl",
        "STATE": tmp_path / "state.json",
        "BUILD_LOG": tmp_path / "build.log",
        "DISK_LOW_GB": 1e9,
        "DISK_CLEAR_GB": 2e9,
    }.items():
        monkeypatch.setattr(watch, name, value)
    gpu = watch.ALLOWED[0]
    smi = {"out": f"{watch.ALLOWED[0]}, 1000, 97887\n{watch.ALLOWED[1]}, 2000, 97887\nGPU-other, 97000, 97887\n"}

    class FakeRun:
        def __init__(self, *args, **kwargs):
            self.stdout = smi["out"]

    monkeypatch.setattr(watch.subprocess, "run", FakeRun)  # hermetic: the real GPUs may be busy
    rows = [
        {"event": "launched", "id": "a", "pid": 1, "gpu": gpu, "attempt": 1},
        {"event": "exited", "id": "a", "code": 0},
        {"event": "launched", "id": "b", "pid": 2, "gpu": gpu, "attempt": 1},
        {"event": "exited", "id": "b", "code": 3},
        {"event": "launched", "id": "b", "pid": 3, "gpu": gpu, "attempt": 2},
    ]
    (tmp_path / "registry.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows) + '{"event": "lau')
    (tmp_path / "runs/b").mkdir(parents=True)
    (tmp_path / "runs/b/console.log").write_text("old line\n")
    (tmp_path / "build.log").write_text("Built gsplat\nEXIT=0\n")
    state = watch.load_state()
    events = watch.poll(state)
    watch.save_state(state)
    kinds = [e.split()[0] for e in events]
    assert kinds == ["JOB_COMPLETED", "JOB_CRASHED", "JOB_RESTARTED", "BUILD_FINISHED", "DISK_LOW"]
    assert f"gpu={gpu}" in events[1] and "code=3" in events[1] and "runs/b/console.log" in events[2]
    with open(tmp_path / "runs/b/console.log", "a") as f:
        f.write("Traceback (most recent call last):\n  File x\nRuntimeError: boom\n")
    with open(tmp_path / "registry.jsonl", "a") as f:  # completes the partial line written earlier
        f.write('nched", "id": "c", "pid": 4, "gpu": "' + gpu + '", "attempt": 1}\n')
        f.write(json.dumps({"event": "failed", "id": "b"}) + "\n")
    state = watch.load_state()  # a restarted watcher resumes from the cursor
    events = watch.poll(state)
    assert [e.split()[0] for e in events] == ["JOB_FAILED", "JOB_LOG_ERROR"]
    assert "line=Traceback" in events[1]
    watch.save_state(state)
    assert watch.poll(watch.load_state()) == []
    smi["out"] = f"{watch.ALLOWED[1]}, 91000, 97887\n"  # 93% of memory on an authorized GPU
    state = watch.load_state()
    assert [e.split()[0] for e in watch.poll(state)] == ["GPU_MEMORY_HIGH"] and watch.poll(state) == []

    class SlowRun:
        def __init__(self, *args, **kwargs):
            raise watch.subprocess.TimeoutExpired("nvidia-smi", 60)

    monkeypatch.setattr(watch.subprocess, "run", SlowRun)  # a hung nvidia-smi is skipped, not fatal
    assert [watch.poll(state) for _ in range(4)] == [[]] * 4
    assert [e.split()[0] for e in watch.poll(state)] == ["GPU_QUERY_FAILING"] and watch.poll(state) == []


def test_watcher_internal_error_prints_marker(tmp_path):
    env = {**os.environ, "PATH": str(tmp_path)}  # no nvidia-smi on PATH -> internal error
    result = subprocess.run(
        [sys.executable, "-I", str(REPO / "scripts/watch_events.py"), "--once"],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    assert result.returncode == 1 and result.stdout.startswith("WATCHER_ERROR: ")


def test_heartbeat_flags_stale_running_jobs_and_dead_watcher(tmp_path, monkeypatch, capsys):
    beat = load("heartbeat")
    monkeypatch.setattr(beat, "REPO", tmp_path)
    (tmp_path / "experiments").mkdir()
    (tmp_path / "runs/fresh").mkdir(parents=True)
    (tmp_path / "runs/stale").mkdir(parents=True)
    (tmp_path / "runs/fresh/console.log").write_text("x")
    (tmp_path / "runs/stale/console.log").write_text("x")
    old = time.time() - 3 * 3600
    os.utime(tmp_path / "runs/stale/console.log", (old, old))
    rows = [{"event": "launched", "id": job, "pid": os.getpid(), "gpu": "g", "attempt": 1} for job in ("fresh", "stale")]
    (tmp_path / "experiments/registry.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(sys, "argv", ["heartbeat.py"])
    with pytest.raises(SystemExit) as exit_:
        beat.main()
    assert exit_.value.code == 1
    output = capsys.readouterr().out
    assert "OK    job fresh" in output and "STUCK job stale" in output and "DEAD  watcher" in output
    assert "HEARTBEAT_PROBLEM 2" in output
