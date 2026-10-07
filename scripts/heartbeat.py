"""Hourly safety net for silent stalls (stdlib only, no CUDA context).

For every job the registry reports as running, the newest modification time among its console log,
metrics/train/eval logs and latest checkpoint must be within ``--max-age`` minutes. Also checks that the
event watcher touched ``experiments/watcher.alive`` recently and that the scheduler daemon (if it was
started) is alive. Prints one line per check and ``HEARTBEAT_OK`` or ``HEARTBEAT_PROBLEM <n>``; exits 1 on
any problem.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ACTIVITY = ("console.log", "metrics.jsonl", "train.jsonl", "eval.jsonl", "log.txt", "checkpoints/latest.pt")


def running_jobs() -> dict[str, dict]:
    state: dict[str, dict] = {}
    path = REPO / "experiments/registry.jsonl"
    for line in path.read_text().splitlines() if path.is_file() else []:
        event = json.loads(line)
        if event["event"] == "launched":
            state[event["id"]] = {"pid": event["pid"], "gpu": event["gpu"]}
        elif event["event"] in ("exited", "failed"):
            state.pop(event["id"], None)
    return state


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    stat = Path(f"/proc/{pid}/stat")
    return stat.is_file() and stat.read_text().rsplit(")", 1)[1].split()[0] != "Z"


def newest_activity(job_id: str) -> float:
    run = REPO / "runs" / job_id
    times = [(run / name).stat().st_mtime for name in ACTIVITY if (run / name).exists()]
    times += [p.stat().st_mtime for p in run.glob("**/checkpoints/latest.pt")]
    return max(times, default=0.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-age", type=float, default=60.0, help="minutes")
    args = ap.parse_args()
    now, problems = time.time(), 0
    for job_id, job in sorted(running_jobs().items()):
        age = (now - newest_activity(job_id)) / 60
        ok = alive(job["pid"]) and age <= args.max_age
        problems += not ok
        print(
            f"{'OK   ' if ok else 'STUCK'} job {job_id} gpu={job['gpu']} pid={job['pid']} alive={alive(job['pid'])} "
            f"last_activity_min={age:.1f} log={REPO / 'runs' / job_id / 'console.log'}"
        )
    beacon = REPO / "experiments/watcher.alive"
    watcher_age = (now - beacon.stat().st_mtime) / 60 if beacon.exists() else float("inf")
    watcher_ok = watcher_age <= 5
    problems += not watcher_ok
    print(f"{'OK   ' if watcher_ok else 'DEAD '} watcher last_poll_min={watcher_age:.1f}")
    pidfile = REPO / "experiments/daemon.pid"
    if pidfile.is_file():
        daemon_ok = alive(int(pidfile.read_text()))
        problems += not daemon_ok
        print(f"{'OK   ' if daemon_ok else 'DEAD '} scheduler daemon pid={pidfile.read_text().strip()}")
    print(f"INFO  free_disk_gb={shutil.disk_usage('/home/ws').free / 1e9:.0f}")
    print("HEARTBEAT_OK" if problems == 0 else f"HEARTBEAT_PROBLEM {problems}")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
