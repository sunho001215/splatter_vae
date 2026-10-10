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
        kind = event["event"]
        if kind in ("launching", "launched"):
            state[event["id"]] = {
                "host": event.get("host", "local"), "pid": event.get("pid"),
                "gpu": event.get("gpu", "-"), "container": event.get("container"), "status": kind,
            }
        elif kind == "remote_exited" and event["id"] in state:
            state[event["id"]]["status"] = "sync_pending"
        elif kind == "unknown" and event["id"] in state:
            state[event["id"]]["status"] = "unknown"
        elif kind in ("exited", "failed"):
            state.pop(event["id"], None)
    return state


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:  # the process may exit between the signal check and this read
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except (FileNotFoundError, ProcessLookupError):  # reading /proc of an exiting process can fail with ESRCH
        return False


def newest_activity(job_id: str) -> float:
    run = REPO / "runs" / job_id
    times = [(run / name).stat().st_mtime for name in ACTIVITY if (run / name).exists()]
    times += [p.stat().st_mtime for p in run.glob("**/checkpoints/latest.pt")]
    if job_id.endswith("-eval"):  # evaluation companions write into their training run directory
        trained = REPO / "runs" / job_id.removesuffix("-eval")
        times += [p.stat().st_mtime for p in (trained / "eval.jsonl", trained / "snapshots") if p.exists()]
    return max(times, default=0.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-age", type=float, default=60.0, help="minutes")
    args = ap.parse_args()
    now, problems = time.time(), 0
    snapshot_path = REPO / "runs/remote/host_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text()) if snapshot_path.is_file() else {}
    for job_id, job in sorted(running_jobs().items()):
        if job["host"] == "remote":
            observed_age = (now - snapshot.get("checked_at", 0)) / 60
            inspect = snapshot.get("containers", {}).get(job["container"], {})
            remote_known = (
                snapshot.get("reachable", False) and snapshot.get("query_ok", False)
                and observed_age <= 5 and bool(inspect)
            )
            if not remote_known:
                status = "UNKNOWN"
            elif inspect.get("State", {}).get("Running"):
                status = "RUNNING"
            elif inspect.get("State", {}).get("ExitCode") == 0:
                status = "SYNC_PENDING"
            else:
                status = "OBSERVED_EXIT"
                problems += 1
            print(
                f"{status} job {job_id} host=remote gpu={job['gpu']} container={job['container']} "
                f"registry_status={job['status']} snapshot_age_min={observed_age:.1f}"
            )
            continue
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
