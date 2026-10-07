"""File-based job scheduler for long GPU work on the two authorized GPUs.

``experiments/queue.yaml`` lists jobs::

    limits:   {<gpu uuid>: {max_jobs: 6, max_mem_gb: 90}, ...}
    jobs:
      - {id: pt-hammer, script: scripts/train.py, args: [...], gpu: any, mem_gb: 20, priority: 1, deps: []}

``tick`` reconciles running jobs with ``experiments/registry.jsonl`` (append-only events) and launches
eligible jobs detached (``setsid nohup``) with exactly one allowed UUID in ``CUDA_VISIBLE_DEVICES``.
Logs go to ``runs/<id>/console.log`` and the exit status to ``runs/<id>/exit_code``. A job whose
process dies or exits non-zero is relaunched once (job scripts resume from their latest checkpoint);
a second failure marks it ``failed`` for diagnosis. ``daemon`` repeats ``tick`` until stopped.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from s4d.gpu_guard import ALLOWED_GPU_UUIDS  # noqa: E402  (pure constants; no CUDA import)

QUEUE = REPO / "experiments/queue.yaml"
REGISTRY = REPO / "experiments/registry.jsonl"
RUNS = REPO / "runs"
PYTHON = REPO / ".venv/bin/python"
GUARD_CALL = re.compile(r"^(\w+\s*=\s*)?(guard_gpus|enforce_allowed_gpus)\(\)\s*$", re.MULTILINE)  # module level
DONE, FAILED, RUNNING, PENDING = "done", "failed", "running", "pending"


def load_queue(path: Path = QUEUE) -> dict:
    queue = yaml.safe_load(path.read_text()) or {}
    ids = [job["id"] for job in queue.get("jobs", [])]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate job ids in queue")
    for uuid in queue.get("limits", {}):
        if uuid not in ALLOWED_GPU_UUIDS:
            raise ValueError(f"queue limit for a non-authorized GPU {uuid}")
    return queue


def read_registry(path: Path = REGISTRY) -> dict[str, dict]:
    """Replay events into the latest state per job id."""
    state: dict[str, dict] = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            event = json.loads(line)
            job = state.setdefault(event["id"], {"status": PENDING, "attempts": 0})
            if event["event"] == "launched":
                job.update(
                    status=RUNNING, pid=event["pid"], gpu=event["gpu"], attempts=event["attempt"], since=event["time"]
                )
            elif event["event"] == "exited":
                job.update(status=DONE if event["code"] == 0 else "crashed", code=event["code"])
            elif event["event"] == "failed":
                job.update(status=FAILED)
            elif event["event"] == "reset":
                job.update(status=PENDING, attempts=0)
    return state


def record(event: dict, path: Path = REGISTRY) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps({**event, "time": time.time()}) + "\n")


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    stat = Path(f"/proc/{pid}/stat")
    return stat.is_file() and stat.read_text().rsplit(")", 1)[1].split()[0] != "Z"


def session_id(pid: int) -> int | None:
    try:
        return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[4])
    except (FileNotFoundError, IndexError, ValueError):
        return None


def gpu_usage() -> dict[str, list[tuple[int, float]]]:
    """Compute processes per authorized GPU as (pid, used MiB)."""
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    usage: dict[str, list[tuple[int, float]]] = {uuid: [] for uuid in ALLOWED_GPU_UUIDS}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3 and parts[0] in usage and parts[1].isdigit():
            usage[parts[0]].append((int(parts[1]), float(parts[2]) if parts[2].replace(".", "").isdigit() else 0.0))
    return usage


def check_guarded(script: str) -> Path:
    path = (REPO / script).resolve()
    if REPO not in path.parents or not path.is_file():
        raise ValueError(f"job script {script} is not a file inside the repository")
    if not GUARD_CALL.search(path.read_text()):
        raise ValueError(f"job script {script} does not call the GPU guard")
    return path


def eligible(jobs: list[dict], state: dict[str, dict]) -> list[dict]:
    """Pending jobs whose dependencies are done, by priority then queue order."""
    ready = []
    for order, job in enumerate(jobs):
        if state.get(job["id"], {}).get("status", PENDING) != PENDING:
            continue
        if all(state.get(dep, {}).get("status") == DONE for dep in job.get("deps", [])):
            ready.append((job.get("priority", 5), order, job))
    return [job for _, _, job in sorted(ready, key=lambda item: item[:2])]


def choose_gpu(job: dict, limits: dict, running: list[dict], foreign: set[str]) -> str | None:
    """Admission control: per-GPU job count and declared-memory budget; never a GPU with foreign work."""
    candidates = list(limits) if job.get("gpu", "any") == "any" else [job["gpu"]]
    best = None
    for uuid in candidates:
        if uuid not in ALLOWED_GPU_UUIDS:
            raise ValueError(f"job {job['id']} requests non-authorized GPU {uuid}")
        if uuid in foreign:
            continue
        here = [r for r in running if r["gpu"] == uuid]
        mem = sum(r["mem_gb"] for r in here) + float(job["mem_gb"])
        if len(here) < limits[uuid]["max_jobs"] and mem <= limits[uuid]["max_mem_gb"]:
            if best is None or mem < best[0]:
                best = (mem, uuid)
    return None if best is None else best[1]


def launch(job: dict, gpu: str, attempt: int, runs: Path = RUNS) -> int:
    script = check_guarded(job["script"])
    run_dir = runs / job["id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "exit_code").unlink(missing_ok=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl"}
    env.pop("MUJOCO_EGL_DEVICE_ID", None)
    wrapper = '"$@"; echo $? > "$S4D_EXIT_FILE"'
    env["S4D_EXIT_FILE"] = str(run_dir / "exit_code")
    argv = [str(PYTHON), "-I", str(script), *map(str, job.get("args", []))]
    with open(run_dir / "console.log", "a") as log:
        log.write(f"\n=== attempt {attempt} on {gpu} at {time.ctime()} ===\n{' '.join(argv)}\n")
        log.flush()
        proc = subprocess.Popen(
            ["setsid", "nohup", "bash", "-c", wrapper, "_", *argv],
            cwd=REPO,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
        )
    return proc.pid


def tick(queue_path: Path = QUEUE, registry: Path = REGISTRY, runs: Path = RUNS, usage=None) -> list[str]:
    """One reconcile-and-launch pass, serialized by a lock so concurrent ticks never double-launch."""
    registry.parent.mkdir(parents=True, exist_ok=True)
    with open(registry.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _tick(queue_path, registry, runs, usage)


def _tick(queue_path: Path, registry: Path, runs: Path, usage) -> list[str]:
    queue = load_queue(queue_path)
    jobs = {job["id"]: job for job in queue.get("jobs", [])}
    state = read_registry(registry)
    messages = []
    for job_id, job_state in state.items():
        if job_state["status"] != RUNNING:
            continue
        exit_file = runs / job_id / "exit_code"
        if exit_file.is_file() and exit_file.read_text().strip():
            code = int(exit_file.read_text().strip())
        elif not pid_alive(job_state["pid"]):
            code = -1  # process died without writing an exit status (killed or machine restart)
        else:
            continue
        record({"event": "exited", "id": job_id, "code": code}, registry)
        messages.append(f"{job_id} exited with {code}")
    state = read_registry(registry)
    for job_id, job_state in state.items():
        if job_state["status"] == "crashed":
            if job_state["attempts"] <= int(jobs.get(job_id, {}).get("max_restarts", 1)):
                job_state["status"] = PENDING  # relaunch below; the script resumes from its checkpoint
            else:
                record({"event": "failed", "id": job_id}, registry)
                job_state["status"] = FAILED
                messages.append(f"{job_id} failed twice; diagnose before retrying")
    running = [
        {"gpu": s["gpu"], "mem_gb": float(jobs.get(i, {}).get("mem_gb", 0)), "pid": s["pid"]}
        for i, s in state.items()
        if s["status"] == RUNNING
    ]
    sessions = {r["pid"] for r in running}
    usage = gpu_usage() if usage is None else usage
    foreign = {uuid for uuid, procs in usage.items() if any(session_id(pid) not in sessions for pid, _ in procs)}
    for uuid in foreign:
        messages.append(f"foreign processes on {uuid}: {usage[uuid]}; not launching there")
    for job in eligible(list(jobs.values()), state):
        gpu = choose_gpu(job, queue["limits"], running, foreign)
        if gpu is None:
            continue
        attempt = state.get(job["id"], {}).get("attempts", 0) + 1
        pid = launch(job, gpu, attempt, runs)
        record({"event": "launched", "id": job["id"], "pid": pid, "gpu": gpu, "attempt": attempt}, registry)
        running.append({"gpu": gpu, "mem_gb": float(job["mem_gb"]), "pid": pid})
        messages.append(f"launched {job['id']} on {gpu} (pid {pid}, attempt {attempt})")
    return messages


def status() -> str:
    queue = load_queue()
    state = read_registry()
    rows = [f"{'id':40s} {'status':8s} {'gpu':14s} {'pid':>8s} {'tries':>5s}"]
    for job in queue.get("jobs", []):
        s = state.get(job["id"], {"status": PENDING, "attempts": 0})
        rows.append(
            f"{job['id']:40s} {s['status']:8s} {s.get('gpu', '-')[:14]:14s} {s.get('pid', '-')!s:>8s} {s['attempts']:>5d}"
        )
    return "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("tick")
    sub.add_parser("status")
    daemon = sub.add_parser("daemon")
    daemon.add_argument("--interval", type=float, default=60.0)
    reset = sub.add_parser("reset", help="re-queue a failed job after diagnosis")
    reset.add_argument("job_id")
    args = ap.parse_args()
    if args.command == "tick":
        print("\n".join(tick()))
    elif args.command == "status":
        print(status())
    elif args.command == "reset":
        record({"event": "reset", "id": args.job_id})
    else:
        pidfile = REPO / "experiments/daemon.pid"
        if pidfile.is_file() and pid_alive(int(pidfile.read_text())):
            raise SystemExit(f"a scheduler daemon is already running (pid {pidfile.read_text()})")
        pidfile.write_text(str(os.getpid()))
        while True:
            for message in tick():
                print(time.strftime("%F %T"), message, flush=True)
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
