"""Experiment event watcher: one flushed line per meaningful event.

Sources: ``experiments/registry.jsonl`` (scheduler events), the environment build log, the console logs of
running jobs, nvidia-smi memory of the two authorized GPUs, and free disk on /home/ws. Event lines:

    EVENT JOB_COMPLETED id=<id> gpu=<uuid> log=<path>
    EVENT JOB_CRASHED id=<id> gpu=<uuid> code=<exit> log=<path>
    EVENT JOB_RESTARTED id=<id> gpu=<uuid> attempt=<n> log=<path>
    EVENT JOB_FAILED id=<id> gpu=<uuid> log=<path>
    EVENT JOB_LOG_ERROR id=<id> gpu=<uuid> log=<path> line=<traceback line>
    EVENT BUILD_FINISHED log=<path>  |  EVENT BUILD_FAILED code=<exit> log=<path>
    EVENT GPU_MEMORY_HIGH gpu=<uuid> used_gb=<x> total_gb=<y>
    EVENT DISK_LOW free_gb=<x>

A cursor (``experiments/watch_state.json``) records what has been reported, so restarting never repeats or
drops an event. ``--once`` exits after the first poll that produced events (used as a wake-up). Any internal
error prints ``WATCHER_ERROR: <reason>`` and exits non-zero. Liveness: ``experiments/watcher.alive`` is
touched every poll. No CUDA context is created (nvidia-smi only).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ALLOWED = ("GPU-76871c16-ff1d-f7b1-cdf5-fabbaf9df8ce", "GPU-d09f0338-71b9-d915-3c7f-e99754a3b639")
REGISTRY = REPO / "experiments/registry.jsonl"
STATE = REPO / "experiments/watch_state.json"
ALIVE = REPO / "experiments/watcher.alive"
BUILD_LOG = REPO / "runs/setup/uv_sync.log"
GPU_HIGH, GPU_CLEAR = 0.92, 0.85
DISK_LOW_GB, DISK_CLEAR_GB = 400.0, 420.0


def load_state() -> dict:
    state = json.loads(STATE.read_text()) if STATE.is_file() else {}
    state.setdefault("registry_offset", 0)
    state.setdefault("jobs", {})  # id -> {"gpu", "attempt"}
    state.setdefault("log_offsets", {})
    state.setdefault("build_reported", False)
    state.setdefault("gpu_high", {})
    state.setdefault("disk_low", False)
    return state


def save_state(state: dict) -> None:
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1))
    os.replace(tmp, STATE)


def log_path(job_id: str) -> str:
    return str(REPO / "runs" / job_id / "console.log")


def registry_events(state: dict) -> list[str]:
    if not REGISTRY.is_file():
        return []
    events = []
    with open(REGISTRY, "rb") as f:
        f.seek(state["registry_offset"])
        data = f.read()
    complete = data[: data.rfind(b"\n") + 1]  # never parse a partially written line
    state["registry_offset"] += len(complete)
    for line in complete.decode().splitlines():
        event = json.loads(line)
        job_id = event["id"]
        job = state["jobs"].setdefault(job_id, {"gpu": "-", "attempt": 0})
        kind = event["event"]
        if kind == "launched":
            job.update(gpu=event["gpu"], attempt=event["attempt"])
            state["log_offsets"][job_id] = Path(log_path(job_id)).stat().st_size if Path(log_path(job_id)).is_file() else 0
            if event["attempt"] > 1:
                events.append(
                    f"JOB_RESTARTED id={job_id} gpu={job['gpu']} attempt={event['attempt']} log={log_path(job_id)}"
                )
        elif kind == "exited":
            state["log_offsets"].pop(job_id, None)
            if event["code"] == 0:
                events.append(f"JOB_COMPLETED id={job_id} gpu={job['gpu']} log={log_path(job_id)}")
            else:
                events.append(f"JOB_CRASHED id={job_id} gpu={job['gpu']} code={event['code']} log={log_path(job_id)}")
        elif kind == "failed":
            events.append(f"JOB_FAILED id={job_id} gpu={job['gpu']} log={log_path(job_id)}")
    return events


def log_error_events(state: dict) -> list[str]:
    """First new traceback line in the console log of each running job (once per launch)."""
    events = []
    for job_id, offset in list(state["log_offsets"].items()):
        path = Path(log_path(job_id))
        if offset < 0 or not path.is_file() or path.stat().st_size <= offset:
            continue
        with open(path, "rb") as f:
            f.seek(offset)
            chunk = f.read()
        text = chunk[: chunk.rfind(b"\n") + 1].decode(errors="replace")
        state["log_offsets"][job_id] = offset + len(text.encode())
        for line in text.splitlines():
            if line.startswith("Traceback"):
                gpu = state["jobs"].get(job_id, {}).get("gpu", "-")
                events.append(f"JOB_LOG_ERROR id={job_id} gpu={gpu} log={path} line={line.strip()[:160]}")
                state["log_offsets"][job_id] = -1  # report once per launch; the exit event follows
                break
    return events


def build_events(state: dict) -> list[str]:
    if state["build_reported"] or not BUILD_LOG.is_file():
        return []
    for line in reversed(BUILD_LOG.read_text(errors="replace").splitlines()[-20:]):
        if line.startswith("EXIT="):
            state["build_reported"] = True
            code = int(line.split("=", 1)[1])
            return [f"BUILD_FINISHED log={BUILD_LOG}" if code == 0 else f"BUILD_FAILED code={code} log={BUILD_LOG}"]
    return []


def gpu_events(state: dict) -> list[str]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,memory.used,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    ).stdout
    events = []
    for line in out.splitlines():
        uuid, used, total = (part.strip() for part in line.split(","))
        if uuid not in ALLOWED:
            continue
        share = float(used) / float(total)
        if share >= GPU_HIGH and not state["gpu_high"].get(uuid):
            state["gpu_high"][uuid] = True
            events.append(f"GPU_MEMORY_HIGH gpu={uuid} used_gb={float(used) / 1024:.1f} total_gb={float(total) / 1024:.1f}")
        elif share < GPU_CLEAR:
            state["gpu_high"][uuid] = False
    return events


def disk_events(state: dict) -> list[str]:
    free_gb = shutil.disk_usage("/home/ws").free / 1e9
    if free_gb < DISK_LOW_GB and not state["disk_low"]:
        state["disk_low"] = True
        return [f"DISK_LOW free_gb={free_gb:.0f}"]
    if free_gb > DISK_CLEAR_GB:
        state["disk_low"] = False
    return []


def poll(state: dict) -> list[str]:
    events = registry_events(state) + log_error_events(state) + build_events(state) + gpu_events(state)
    return events + disk_events(state)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--once", action="store_true", help="exit after the first poll that produced events")
    ap.add_argument("--interval", type=float, default=20.0)
    args = ap.parse_args()
    try:
        STATE.parent.mkdir(parents=True, exist_ok=True)
        (REPO / "experiments/watcher.pid").write_text(str(os.getpid()))
        while True:
            state = load_state()
            events = poll(state)
            save_state(state)
            ALIVE.touch()
            for event in events:
                print(f"EVENT {event}", flush=True)
            if events and args.once:
                return
            time.sleep(args.interval)
    except Exception as exc:  # noqa: BLE001  (any failure must wake the agent)
        reason = f"{type(exc).__name__}: {exc}".replace("\n", " ")
        print(f"WATCHER_ERROR: {reason} | {traceback.format_exc().splitlines()[-2].strip()}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
