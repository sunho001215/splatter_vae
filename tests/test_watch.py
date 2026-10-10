"""Event watcher and heartbeat: exactly one line per event, cursor survives restarts, errors surface."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

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

    def fake_query(command, state_dir, timeout):
        assert command[0] == "nvidia-smi" and state_dir == tmp_path / "experiments"
        assert timeout == watch.GPU_QUERY_TIMEOUT
        return smi["out"]

    monkeypatch.setattr(watch, "query_output", fake_query)
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

    def unavailable_query(command, state_dir, timeout):
        raise watch.subprocess.TimeoutExpired(command, timeout)

    monkeypatch.setattr(watch, "query_output", unavailable_query)
    assert [watch.poll(state) for _ in range(4)] == [[]] * 4
    assert [e.split()[0] for e in watch.poll(state)] == ["GPU_QUERY_FAILING"] and watch.poll(state) == []


def test_hung_query_survives_watcher_rearm_without_blocking_other_events(tmp_path, monkeypatch):
    watch = load("watch_events")
    for name, value in {
        "REPO": tmp_path,
        "REGISTRY": tmp_path / "registry.jsonl",
        "STATE": tmp_path / "state.json",
        "BUILD_LOG": tmp_path / "build.log",
    }.items():
        monkeypatch.setattr(watch, name, value)
    gpu = watch.ALLOWED[0]
    watch.REGISTRY.write_text(json.dumps({"event": "launched", "id": "a", "gpu": gpu, "attempt": 1}) + "\n")
    network = tmp_path / "runs/remote/network.jsonl"
    network.parent.mkdir(parents=True)
    disk = {"free": 500e9, "polls": 0}

    def disk_usage(path):
        disk["polls"] += 1
        return SimpleNamespace(free=disk["free"])

    monkeypatch.setattr(watch.shutil, "disk_usage", disk_usage)
    created, deferred_kills, actual_kills = [], [], []
    real_popen = subprocess.Popen

    def deferred_popen(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        created.append(proc)
        actual_kills.append(proc.kill)
        proc.kill = lambda: deferred_kills.append(proc.pid)
        return proc

    monkeypatch.setattr(subprocess, "Popen", deferred_popen)
    command = [sys.executable, "-I", "-c", "import os,signal;os.closerange(3,65536);print('held',flush=True);signal.pause()"]

    def cpu_query(command_ignored, state_dir, timeout):
        return load("_nvidia_query").query_output(command, state_dir, timeout=0.5)

    monkeypatch.setattr(watch, "query_output", cpu_query)
    try:
        state = watch.load_state()
        assert watch.poll(state) == []
        stuck = created[0]
        assert deferred_kills == [stuck.pid] and stuck.poll() is None
        assert json.loads((tmp_path / "experiments/local_gpu_query.json").read_text())["pid"] == stuck.pid
        with watch.REGISTRY.open("a") as stream:
            stream.write(json.dumps({"event": "exited", "id": "a", "code": 0}) + "\n")
        network.write_text(json.dumps({"event": "REMOTE_UNREACHABLE", "time": 1000}) + "\n")
        disk["free"] = 300e9
        assert [event.split()[0] for event in watch.poll(state)] == [
            "JOB_COMPLETED", "REMOTE_UNREACHABLE", "DISK_LOW"
        ]
        assert state["registry_offset"] == watch.REGISTRY.stat().st_size
        assert state["remote_network_offset"] == network.stat().st_size
        with watch.REGISTRY.open("a") as stream:
            stream.write(json.dumps({"event": "launched", "id": "a", "gpu": gpu, "attempt": 2}) + "\n")
        with network.open("a") as stream:
            stream.write(json.dumps({
                "event": "REMOTE_RECONNECTED", "time": 1900, "outage_started": 1000, "duration_seconds": 900
            }) + "\n")
        disk["free"] = 500e9
        assert [event.split()[0] for event in watch.poll(state)] == ["JOB_RESTARTED", "REMOTE_RECONNECTED"]
        assert not state["disk_low"] and disk["polls"] == 3
        watch.save_state(state)
        watch = load("watch_events")
        for name, value in {
            "REPO": tmp_path, "REGISTRY": tmp_path / "registry.jsonl",
            "STATE": tmp_path / "state.json", "BUILD_LOG": tmp_path / "build.log",
        }.items():
            monkeypatch.setattr(watch, name, value)
        monkeypatch.setattr(watch, "query_output", cpu_query)
        state = watch.load_state()
        assert watch.poll(state) == []
        assert [event.split()[0] for event in watch.poll(state)] == ["GPU_QUERY_FAILING"]
        assert watch.poll(state) == []
        assert len(created) == 1 and deferred_kills == [stuck.pid] and disk["polls"] == 6
        actual_kills[0]()
        stuck.wait(timeout=5)
        command = [sys.executable, "-I", "-c", f"print({f'{gpu}, 95000, 100000'!r})"]
        assert [event.split()[0] for event in watch.poll(state)] == ["GPU_MEMORY_HIGH"]
        assert len(created) == 2 and state["gpu_query_failures"] == 0 and disk["polls"] == 7
    finally:
        for proc, kill in zip(created, actual_kills):
            if proc.poll() is None:
                kill()
            proc.wait(timeout=5)


def test_remote_events_are_unknown_until_results_return(tmp_path, monkeypatch):
    watch = load("watch_events")
    monkeypatch.setattr(watch, "REPO", tmp_path)
    monkeypatch.setattr(watch, "REGISTRY", tmp_path / "registry.jsonl")
    rows = [
        {"event": "launching", "id": "remote-a", "host": "remote", "attempt": 1},
        {"event": "launched", "id": "remote-a", "host": "remote", "attempt": 1, "gpu": "remote-gpu"},
        {"event": "host_unreachable", "id": "__host_remote__", "host": "remote"},
        {"event": "host_query_failed", "id": "__host_remote__", "host": "remote"},
        {"event": "host_disk_low", "id": "__host_remote__", "host": "remote"},
        {"event": "unknown", "id": "remote-a", "host": "remote"},
        {"event": "host_reconnected", "id": "__host_remote__", "host": "remote", "duration_seconds": 900},
        {"event": "remote_exited", "id": "remote-a", "host": "remote", "code": 0},
    ]
    watch.REGISTRY.write_text("".join(json.dumps(row) + "\n" for row in rows))
    state = {"registry_offset": 0, "jobs": {}, "log_offsets": {}}
    events = watch.registry_events(state)
    assert not any("JOB_COMPLETED" in event or "JOB_CRASHED" in event for event in events)
    assert [event.split()[0] for event in events] == [
        "REMOTE_HOST_UNREACHABLE", "REMOTE_HOST_QUERY_FAILED", "REMOTE_HOST_DISK_LOW",
        "REMOTE_JOB_UNKNOWN", "REMOTE_HOST_RECONNECTED", "REMOTE_RESULTS_PENDING"
    ]
    with watch.REGISTRY.open("a") as stream:
        for row in [
            {"event": "results_synced", "id": "remote-a", "host": "remote", "final": True},
            {"event": "exited", "id": "remote-a", "host": "remote", "code": 0},
        ]:
            stream.write(json.dumps(row) + "\n")
    events = watch.registry_events(state)
    assert [event.split()[0] for event in events] == ["REMOTE_RESULTS_SYNCED", "JOB_COMPLETED"]
    assert events[-1].endswith("host=remote")
    assert watch.registry_events(state) == []


def test_remote_network_cursor_preserves_outage_duration(tmp_path, monkeypatch):
    watch = load("watch_events")
    monkeypatch.setattr(watch, "REPO", tmp_path)
    path = tmp_path / "runs/remote/network.jsonl"
    path.parent.mkdir(parents=True)
    row = {"event": "REMOTE_UNREACHABLE", "time": 1000.0}
    path.write_text(json.dumps(row) + "\n" + '{"event": "REMOTE_RE')
    state = {}
    assert watch.remote_network_events(state) == ["REMOTE_UNREACHABLE host=remote started=1000.000"]
    assert watch.remote_network_events(state) == []
    with path.open("a") as stream:
        stream.write('CONNECTED", "time": 1900, "outage_started": 1000, "duration_seconds": 900}\n')
    assert watch.remote_network_events(state) == [
        "REMOTE_RECONNECTED host=remote started=1000.000 ended=1900.000 duration_seconds=900.0"
    ]
    assert watch.remote_network_events(state) == []


def test_watcher_internal_error_prints_marker(tmp_path):
    # Run a copy whose REPO is tmp_path, so the live experiments/ state, pid file and beacon are never touched.
    (tmp_path / "scripts").mkdir()
    script = tmp_path / "scripts" / "watch_events.py"
    script.write_text((REPO / "scripts/watch_events.py").read_text())
    (tmp_path / "scripts/_nvidia_query.py").write_text((REPO / "scripts/_nvidia_query.py").read_text())
    (tmp_path / "experiments").mkdir()
    (tmp_path / "experiments" / "watch_state.json").write_text("{")
    env = {**os.environ, "PATH": str(tmp_path)}
    result = subprocess.run(
        [sys.executable, "-I", str(script), "--once"],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    assert result.returncode == 1 and result.stdout.startswith("WATCHER_ERROR: JSONDecodeError: ")
    assert not (tmp_path / "experiments/local_gpu_query.lock").exists()


def test_heartbeat_never_uses_local_pid_for_remote_jobs(tmp_path, monkeypatch, capsys):
    beat = load("heartbeat")
    monkeypatch.setattr(beat, "REPO", tmp_path)
    (tmp_path / "experiments").mkdir()
    rows = [
        {"event": "launching", "id": "remote-a", "host": "remote", "container": "s4d-remote-a-a1"},
        {"event": "launched", "id": "remote-a", "host": "remote", "gpu": "r", "container": "s4d-remote-a-a1"},
        {"event": "unknown", "id": "remote-a", "host": "remote"},
    ]
    (tmp_path / "experiments/registry.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))

    def forbidden(*args):
        raise AssertionError("Remote PID must never be inspected in the local namespace")

    monkeypatch.setattr(beat, "alive", forbidden)
    monkeypatch.setattr(sys, "argv", ["heartbeat.py"])
    with pytest.raises(SystemExit):
        beat.main()
    output = capsys.readouterr().out
    assert "UNKNOWN job remote-a" in output and "registry_status=unknown" in output
    assert "STUCK job remote-a" not in output


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
    (tmp_path / "runs/fresh-eval").mkdir()
    (tmp_path / "runs/fresh-eval/console.log").write_text("x")
    os.utime(tmp_path / "runs/fresh-eval/console.log", (old, old))
    (tmp_path / "runs/fresh/eval.jsonl").write_text("{}")  # the companion writes here
    jobs = ("fresh", "stale", "fresh-eval")
    rows = [{"event": "launched", "id": job, "pid": os.getpid(), "gpu": "g", "attempt": 1} for job in jobs]
    (tmp_path / "experiments/registry.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(sys, "argv", ["heartbeat.py"])
    with pytest.raises(SystemExit) as exit_:
        beat.main()
    assert exit_.value.code == 1
    output = capsys.readouterr().out
    assert "OK    job fresh " in output and "STUCK job stale" in output and "DEAD  watcher" in output
    assert "OK    job fresh-eval" in output, "an evaluation companion is active through its run's eval.jsonl"
    assert "HEARTBEAT_PROBLEM 2" in output
