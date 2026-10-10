"""Scheduler: queue validation, registry replay, admission control, guard check and real detached launches."""

from __future__ import annotations

import importlib.util
import json
import os
import time
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("jobs", REPO / "scripts/jobs.py")
jobs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(jobs)
GPU4, GPU5 = jobs.ALLOWED_GPU_UUIDS
LIMITS = {GPU4: {"max_jobs": 2, "max_mem_gb": 40}, GPU5: {"max_jobs": 1, "max_mem_gb": 40}}


def write_queue(path: Path, job_list, limits=LIMITS) -> Path:
    path.write_text(yaml.safe_dump({"limits": limits, "jobs": job_list}))
    return path


def test_queue_validation_and_registry_replay(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        jobs.load_queue(write_queue(tmp_path / "q.yaml", [{"id": "a"}, {"id": "a"}]))
    with pytest.raises(ValueError, match="non-authorized"):
        jobs.load_queue(write_queue(tmp_path / "q.yaml", [], {"GPU-6b33eef2-80c5-547e-6a61-7ab81c14d63b": {}}))
    registry = tmp_path / "r.jsonl"
    jobs.record({"event": "launched", "id": "a", "pid": 1, "gpu": GPU4, "attempt": 1}, registry)
    jobs.record({"event": "exited", "id": "a", "code": 3}, registry)
    jobs.record({"event": "launched", "id": "b", "pid": 2, "gpu": GPU5, "attempt": 1}, registry)
    state = jobs.read_registry(registry)
    assert state["a"]["status"] == "crashed" and state["a"]["attempts"] == 1
    assert state["b"]["status"] == "running" and state["b"]["gpu"] == GPU5
    jobs.record({"event": "reset", "id": "a"}, registry)
    assert jobs.read_registry(registry)["a"] == {**state["a"], "status": "pending", "attempts": 0}


def test_session_id_matches_the_kernel():
    assert jobs.session_id(os.getpid()) == os.getsid(os.getpid())
    assert jobs.session_id(2**22 + 12345) is None


def test_eligibility_priority_dependencies_and_admission():
    queue = [
        {"id": "late", "priority": 2},
        {"id": "first", "priority": 1, "deps": ["done-dep"]},
        {"id": "blocked", "priority": 0, "deps": ["running-dep"]},
    ]
    state = {"done-dep": {"status": "done"}, "running-dep": {"status": "running"}}
    assert [j["id"] for j in jobs.eligible(queue, state)] == ["first", "late"]
    running = [{"gpu": GPU4, "mem_gb": 30.0}]
    assert jobs.choose_gpu({"id": "x", "gpu": "any", "mem_gb": 5}, LIMITS, running, set()) == GPU5
    assert jobs.choose_gpu({"id": "x", "gpu": "any", "mem_gb": 5}, LIMITS, running, {GPU5}) == GPU4
    assert jobs.choose_gpu({"id": "x", "gpu": GPU4, "mem_gb": 20}, LIMITS, running, set()) is None
    assert jobs.choose_gpu({"id": "x", "gpu": "any", "mem_gb": 5}, LIMITS, running * 2, {GPU5}) is None
    with pytest.raises(ValueError, match="non-authorized"):
        jobs.choose_gpu({"id": "x", "gpu": "GPU-6b33eef2-80c5-547e-6a61-7ab81c14d63b", "mem_gb": 1}, LIMITS, [], set())


def test_launches_wait_for_host_ram(tmp_path):
    uuid = GPU4
    queue = tmp_path / "q.yaml"
    queue.write_text(
        yaml.safe_dump(
            {
                "limits": LIMITS,
                "host_ram_reserve_gb": 40,
                "jobs": [
                    {"id": "big", "script": "tests/_job_worker.py", "args": [0], "gpu": uuid, "mem_gb": 1, "ram_gb": 30},
                ],
            }
        )
    )
    registry, runs = tmp_path / "registry.jsonl", tmp_path / "runs"
    messages = jobs.tick(queue, registry, runs, usage={}, available_gb=60.0)
    assert any("waiting for host RAM" in m for m in messages) and not registry.exists()
    assert jobs.host_available_gb() > 0


@pytest.mark.parametrize("failure", [
    jobs.subprocess.CalledProcessError(255, ["nvidia-smi"]),
    jobs.subprocess.TimeoutExpired(["nvidia-smi"], 60),
])
def test_gpu_query_failure_reconciles_exits_but_launches_nothing(tmp_path, monkeypatch, failure):
    queue = write_queue(tmp_path / "q.yaml", [
        {"id": "done", "script": "tests/_job_worker.py", "args": [0], "gpu": GPU4, "mem_gb": 1},
        {"id": "next", "script": "tests/_job_worker.py", "args": [0], "gpu": GPU4, "mem_gb": 1, "deps": ["done"]},
    ])
    registry, runs = tmp_path / "registry.jsonl", tmp_path / "runs"
    jobs.record({"event": "launched", "id": "done", "pid": 2**30, "gpu": GPU4, "attempt": 1}, registry)
    (runs / "done").mkdir(parents=True)
    (runs / "done/exit_code").write_text("0")
    probes = iter([failure, {GPU4: [], GPU5: []}])

    def query():
        result = next(probes)
        if isinstance(result, Exception):
            raise result
        return result

    launched = []

    def launch(job, gpu, attempt, runs):
        launched.append(job["id"])
        return 2**30

    monkeypatch.setattr(jobs, "gpu_usage", query)
    monkeypatch.setattr(jobs, "launch", launch)
    messages = jobs.tick(queue, registry, runs, available_gb=100)
    assert any("done exited with 0" in message for message in messages)
    assert any("GPU process query failed" in message and "no jobs launched" in message for message in messages)
    assert not launched
    assert jobs.read_registry(registry)["done"]["status"] == "done"
    assert "next" not in jobs.read_registry(registry)
    jobs.tick(queue, registry, runs, available_gb=100)
    assert launched == ["next"]
    assert jobs.read_registry(registry)["next"]["status"] == "running"


def test_recently_launched_jobs_still_count_against_host_ram(tmp_path):
    """A job that started a minute ago has not allocated its memory yet; admission must not count it as free."""
    queue = tmp_path / "q.yaml"
    jobs_list = [
        {"id": "ramping", "script": "tests/_job_worker.py", "args": [0], "gpu": GPU4, "mem_gb": 1, "ram_gb": 30},
        {"id": "next", "script": "tests/_job_worker.py", "args": [0], "gpu": GPU4, "mem_gb": 1, "ram_gb": 30},
    ]
    queue.write_text(yaml.safe_dump({"limits": LIMITS, "host_ram_reserve_gb": 40, "jobs": jobs_list}))
    registry, runs = tmp_path / "registry.jsonl", tmp_path / "runs"
    own = os.getpid()  # a live session so reconciliation keeps the job running
    registry.write_text(json.dumps({"event": "launched", "id": "ramping", "pid": os.getsid(own), "gpu": GPU4,
                                    "attempt": 1, "time": time.time() - 60}) + "\n")
    messages = jobs.tick(queue, registry, runs, usage={}, available_gb=90.0)  # 90 - 30 (ramping) - 30 < 40
    assert any("next: waiting for host RAM (60 GB available" in m for m in messages)
    assert "next" not in registry.read_text()
    registry.write_text(json.dumps({"event": "launched", "id": "ramping", "pid": os.getsid(own), "gpu": GPU4,
                                    "attempt": 1, "time": time.time() - 3600}) + "\n")
    messages = jobs.tick(queue, registry, runs, usage={}, available_gb=50.0)  # ramp over: 50 - 30 < 40 still waits
    assert any("next: waiting for host RAM (50 GB available" in m for m in messages)


def test_only_guarded_repository_scripts_launch(tmp_path):
    assert jobs.check_guarded("scripts/train_rl.py").name == "train_rl.py"
    assert jobs.check_guarded("scripts/train.py").name == "train.py"
    assert jobs.check_guarded("scripts/check_metaworld.py").name == "check_metaworld.py"  # enforce_allowed_gpus()
    with pytest.raises(ValueError, match="GPU guard"):
        jobs.check_guarded("scripts/jobs.py")
    with pytest.raises(ValueError, match="inside the repository"):
        jobs.check_guarded("../outside.py")


def wait_for(queue, registry, runs, predicate, timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        jobs.tick(queue, registry, runs, usage={})
        state = jobs.read_registry(registry)
        if predicate(state):
            return state
        time.sleep(1.0)
    raise TimeoutError(json.dumps(state))


def test_exit_terminates_leftover_processes_of_the_job_session(tmp_path):
    import subprocess as sp

    # A session leader that exits while a child keeps running, like an OOM-killed trainer and its loader workers.
    leader = sp.Popen(["setsid", "bash", "-c", "sleep 300 & echo $! > child.pid; exit 9"], cwd=tmp_path)
    leader.wait()
    child = int((tmp_path / "child.pid").read_text())
    registry, runs = tmp_path / "registry.jsonl", tmp_path / "runs"
    queue = write_queue(
        tmp_path / "q.yaml",
        [{"id": "job", "script": "tests/_job_worker.py", "args": [0], "gpu": GPU4, "mem_gb": 1, "max_restarts": 0}],
    )
    jobs.record({"event": "launched", "id": "job", "pid": leader.pid, "gpu": GPU4, "attempt": 1}, registry)
    (runs / "job").mkdir(parents=True)
    (runs / "job" / "exit_code").write_text("9")
    (tmp_path / "HOLD").touch()
    messages = jobs.tick(queue, registry, runs, usage={})
    assert any("terminated leftover" in m for m in messages)
    deadline = time.time() + 10
    while jobs.pid_alive(child) and time.time() < deadline:
        time.sleep(0.2)
    status = Path(f"/proc/{child}/status")
    assert not jobs.pid_alive(child), status.read_text()[:400] if status.exists() else "gone"


def test_detached_launch_success_restart_once_then_fail(tmp_path):
    uuid = os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0]
    limits = {uuid: {"max_jobs": 2, "max_mem_gb": 10}}
    queue = write_queue(
        tmp_path / "q.yaml",
        [
            {"id": "ok", "script": "tests/_job_worker.py", "args": [0, 300], "gpu": uuid, "mem_gb": 1, "oom_score_adj": 300},
            {"id": "bad", "script": "tests/_job_worker.py", "args": [7], "gpu": uuid, "mem_gb": 1},
            {"id": "after-ok", "script": "tests/_job_worker.py", "args": [0], "gpu": "any", "mem_gb": 1, "deps": ["ok"]},
        ],
        limits,
    )
    registry, runs = tmp_path / "registry.jsonl", tmp_path / "runs"
    (tmp_path / "HOLD").touch()
    assert jobs.tick(queue, registry, runs, usage={}) == [] and not registry.exists(), "HOLD launches nothing"
    (tmp_path / "HOLD").unlink()
    state = wait_for(
        queue,
        registry,
        runs,
        lambda s: s.get("after-ok", {}).get("status") == "done" and s.get("bad", {}).get("status") == "failed",
    )
    assert state["ok"]["status"] == "done" and state["bad"]["attempts"] == 2
    log = (runs / "bad" / "console.log").read_text()
    assert log.count("=== attempt") == 2 and f"[gpu_guard] cuda:0 -> {uuid}" in log
    assert (runs / "ok" / "exit_code").read_text().strip() == "0"
