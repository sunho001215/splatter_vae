"""Authoritative file-based scheduler with independent local and remote resource admission."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from s4d.gpu_guard import APPROVED_HOST_GPUS, HOST_CONFIG_PATH  # noqa: E402
from s4d.remote_access import sha256 as file_sha256  # noqa: E402
from s4d.remote_jobs import RemoteBackend, RemoteJobError, RemoteUnreachable, validate_container  # noqa: E402
from scripts._nvidia_query import query_output  # noqa: E402

QUEUE = REPO / "experiments/queue.yaml"
REGISTRY = REPO / "experiments/registry.jsonl"
RUNS = REPO / "runs"
PYTHON = sys.executable
GIB = 2**30
GUARD_CALL = re.compile(r"^(\w+\s*=\s*)?(guard_gpus|enforce_allowed_gpus)\(\)\s*$", re.MULTILINE)
DONE, FAILED, RUNNING, PENDING = "done", "failed", "running", "pending"
ACTIVE = {RUNNING, "launching", "sync_pending"}
HOST_EVENT_ID = "__host_remote__"


def load_queue(path: Path = QUEUE) -> dict:
    queue = yaml.safe_load(path.read_text()) or {}
    ids = [job["id"] for job in queue.get("jobs", [])]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate job ids in queue")
    hosts = queue.setdefault("hosts", {})
    if set(hosts) - {"local", "remote"}:
        raise ValueError("queue has an unknown host")
    local = hosts.setdefault("local", {})
    if not isinstance(local, dict):
        raise ValueError("host local configuration must be a mapping")
    for key in ("limits", "host_ram_reserve_gb", "host_ram_ramp_minutes", "disk_floor_gb"):
        if key in queue:
            local.setdefault(key, queue[key])
    local.setdefault("limits", {})
    local["disk_floor_gb"] = max(300.0, float(local.get("disk_floor_gb", 300)))
    for name, host in hosts.items():
        if not isinstance(host, dict):
            raise ValueError(f"host {name} configuration must be a mapping")
        for uuid, limit in host.get("limits", {}).items():
            if uuid not in APPROVED_HOST_GPUS[name]:
                raise ValueError(f"queue limit for a non-authorized {name} GPU {uuid}")
            if int(limit.get("max_jobs", 0)) < 1 or float(limit.get("max_mem_gb", 0)) <= 0:
                raise ValueError(f"invalid admission limits for {uuid}")
        for key in ("host_ram_reserve_gb", "host_ram_ramp_minutes", "disk_floor_gb", "disk_floor_fraction"):
            value = float(host.get(key, 0))
            if not math.isfinite(value) or value < 0 or (key == "disk_floor_fraction" and value > 1):
                raise ValueError(f"invalid {name} resource limit {key}")
    for job in queue.get("jobs", []):
        job_id = job["id"]
        if not isinstance(job_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", job_id):
            raise ValueError("unsafe job id")
        name = job.get("host", "local")
        if name not in hosts:
            raise ValueError(f"job {job_id} targets unconfigured host {name}")
        if name == "remote" and (any(key not in job for key in ("mem_gb", "ram_gb", "disk_gb"))
                                 or float(job.get("ram_gb", 0)) <= 0):
            raise ValueError(f"job {job_id}: remote memory/disk reservations must be explicit")
        if name == "remote":
            required = job.get("required_results")
            if (not isinstance(required, list) or not required
                    or not any(isinstance(value, str) and value not in {"exit.json", "provenance.json"}
                               for value in required)
                    or any(not isinstance(value, str)
                           or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*(?:/[A-Za-z0-9][A-Za-z0-9_.-]*)*", value)
                           or ".." in Path(value).parts
                           or Path(value).parts[0] in {"replay", "wandb", ".cache"} for value in required)):
                raise ValueError(f"job {job_id}: explicit run-relative required_results are mandatory")
        gpu = job.get("gpu", "any")
        if gpu != "any" and gpu not in APPROVED_HOST_GPUS[name]:
            raise ValueError(f"job {job_id} requests non-authorized {name} GPU {gpu}")
        for key in ("mem_gb", "ram_gb", "disk_gb"):
            value = float(job.get(key, 0))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"job {job_id}: invalid {key}")
    return queue


def read_registry(path: Path = REGISTRY) -> dict[str, dict]:
    state: dict[str, dict] = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            event = json.loads(line)
            kind = event["event"]
            job = state.setdefault(event["id"], {"status": PENDING, "attempts": 0, "failures": 0,
                                                "failed_attempts": [], "infrastructure_failures": []})
            if kind in {"launching", "launched"}:
                if job["status"] not in ACTIVE or event["attempt"] != job.get("attempt"):
                    job["attempt_since"] = event["time"]
                job.update({key: value for key, value in event.items() if key not in {"event", "id", "time", "attempt"}})
                job.update(status="launching" if kind == "launching" else RUNNING,
                           host=event.get("host", "local"), attempt=event["attempt"],
                           attempts=max(job["attempts"], event["attempt"]), since=event["time"], unknown=False)
                job.pop("sync", None)
            elif kind == "unknown":
                job.update(unknown=True, reason=event.get("reason", "unknown remote state"))
            elif kind == "remote_exited":
                job.update(status="sync_pending", code=event["code"], unknown=False)
            elif kind == "results_synced":
                job.update(sync=event["sync"], unknown=False)
            elif kind == "exited":
                synced = job.get("sync", {})
                verified = job.get("host", "local") != "remote" or (synced.get("final") is True
                                                                              and synced.get("verified") is True)
                job.update(status=(DONE if verified else "sync_pending") if event["code"] == 0 else "crashed",
                           code=event["code"], unknown=False)
                if event["code"] != 0:
                    failure = {"attempt": event.get("attempt", job.get("attempt", 0)),
                               "launch_time": job.get("attempt_since", event["time"]),
                               "time": event["time"], "code": event["code"]}
                    if not any(item["attempt"] == failure["attempt"] and item["launch_time"] == failure["launch_time"]
                               for item in job["failed_attempts"]):
                        job["failed_attempts"].append(failure)
                        job["failures"] += 1
            elif kind == "infrastructure_failure":
                start, end = event.get("incident_started"), event.get("incident_ended")
                if (event.get("reason") not in {"cuda_init_unavailable", "nvml_init_unavailable"}
                        or type(event.get("attempt")) is not int or event["attempt"] < 1
                        or type(start) not in {int, float} or not math.isfinite(start) or start < 0
                        or (end is not None and (type(end) not in {int, float} or not math.isfinite(end) or end < start))):
                    raise ValueError(f"invalid infrastructure failure event for {event['id']}")
                matching = [item for item in job["failed_attempts"] if item["attempt"] == event["attempt"]
                            and item["time"] >= start and (end is None or item["launch_time"] <= end)
                            and ("failure_time" not in event or item["time"] == event["failure_time"])]
                if len(matching) != 1:
                    raise ValueError(f"infrastructure failure must identify one observed outage failure for {event['id']}")
                failure = matching[0]
                if not any(item["attempt"] == failure["attempt"] and item["failure_time"] == failure["time"]
                           for item in job["infrastructure_failures"]):
                    job["infrastructure_failures"].append({**event, "failure_time": failure["time"]})
                    job["failures"] -= 1
                    if job["status"] == FAILED:
                        job["status"] = "crashed"
            elif kind == "failed":
                job.update(status=FAILED)
            elif kind == "reset":
                job.update(status=PENDING)
            elif kind == "host_unreachable":
                job.update(status="unreachable", host=event["host"], outage_started=event["time"])
            elif kind == "host_reconnected":
                job.update(status="reachable", host=event["host"])
    return state


def record(event: dict, path: Path = REGISTRY) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as stream:
        stream.write(json.dumps({**event, "time": time.time()}) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except (FileNotFoundError, ProcessLookupError):
        return False


def session_id(pid: int) -> int | None:
    try:
        return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[3])
    except (FileNotFoundError, IndexError, ValueError):
        return None


def gpu_usage() -> dict[str, list[tuple[int, float]]]:
    out = query_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"],
        state_dir=REPO / "experiments", timeout=60,
    )
    usage: dict[str, list[tuple[int, float]]] = {uuid: [] for uuid in APPROVED_HOST_GPUS["local"]}
    for line in out.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 3 and parts[0] in usage:
            if not parts[1].isdigit():
                raise ValueError(f"GPU process ownership unavailable for {parts[0]}")
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
    ready = []
    for order, job in enumerate(jobs):
        if state.get(job["id"], {}).get("status", PENDING) != PENDING:
            continue
        if all(state.get(dep, {}).get("status") == DONE for dep in job.get("deps", [])):
            ready.append((job.get("priority", 5), order, job))
    return [job for _, _, job in sorted(ready, key=lambda item: item[:2])]


def choose_gpu(job: dict, limits: dict, running: list[dict], foreign: set[str]) -> str | None:
    host = job.get("host", "local")
    candidates = list(limits) if job.get("gpu", "any") == "any" else [job["gpu"]]
    best = None
    for uuid in candidates:
        if uuid not in APPROVED_HOST_GPUS[host]:
            raise ValueError(f"job {job['id']} requests non-authorized GPU {uuid} on {host}")
        if uuid in foreign or uuid not in limits:
            continue
        here = [item for item in running if item.get("host", "local") == host and item["gpu"] == uuid]
        memory = sum(item["mem_gb"] for item in here) + float(job["mem_gb"])
        if len(here) < limits[uuid]["max_jobs"] and memory <= limits[uuid]["max_mem_gb"]:
            if best is None or memory < best[0]:
                best = (memory, uuid)
    return None if best is None else best[1]


def job_fingerprint(job: dict) -> str:
    payload = {"script": job["script"], "args": list(map(str, job.get("args", []))),
               "required_results": job.get("required_results", []),
               "expected_config_sha256": job.get("expected_config_sha256")}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def reservation(current: dict, job: dict, key: str) -> float:
    return max(float(current.get(key, 0)), float(job.get(key, 0)))


def source_identity() -> dict:
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()
    paths = [REPO / "mujoco_mig_setup.py"] + [
        path for folder in ("s4d", "scripts", "tests", "configs")
        for path in sorted((REPO / folder).rglob("*"))
        if path.is_file() and path.suffix in (".py", ".yaml", ".sh")
    ]
    fingerprints = {str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    return {"commit": commit, "source_sha256": fingerprints}


def launch(job: dict, gpu: str, attempt: int, runs: Path = RUNS) -> int:
    script = check_guarded(job["script"])
    run_dir = runs / job["id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "exit_code").unlink(missing_ok=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl",
           "S4D_HOST_CONFIG": str(HOST_CONFIG_PATH)}
    env.pop("MUJOCO_EGL_DEVICE_ID", None)
    adj = int(job.get("oom_score_adj", 0))
    if not 0 <= adj <= 1000:
        raise ValueError(f"job {job['id']}: oom_score_adj must be in [0, 1000]")
    wrapper = f'echo {adj} > /proc/self/oom_score_adj; "$@"; echo $? > "$S4D_EXIT_FILE"'
    env["S4D_EXIT_FILE"] = str(run_dir / "exit_code")
    argv = [str(PYTHON), "-I", str(script), *map(str, job.get("args", []))]
    with open(run_dir / "console.log", "a") as log:
        log.write(f"\n=== attempt {attempt} on {gpu} at {time.ctime()} ===\n{' '.join(argv)}\n")
        log.flush()
        proc = subprocess.Popen(["setsid", "nohup", "bash", "-c", wrapper, "_", *argv], cwd=REPO, env=env,
                                stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    return proc.pid


def host_available_gb() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2**20
    return 0.0


def disk_usage(path: Path) -> tuple[int, int]:
    while not path.exists():
        path = path.parent
    result = shutil.disk_usage(path)
    return result.total, result.free


def _host_reachability(registry: Path, reachable: bool, reason: str = "") -> None:
    previous = read_registry(registry).get(HOST_EVENT_ID, {})
    if not reachable and previous.get("status") != "unreachable":
        record({"event": "host_unreachable", "id": HOST_EVENT_ID, "host": "remote", "reason": reason}, registry)
    elif reachable and previous.get("status") == "unreachable":
        start = previous["outage_started"]
        record({"event": "host_reconnected", "id": HOST_EVENT_ID, "host": "remote", "outage_started": start,
                "duration_seconds": time.time() - start}, registry)


def _unknown(job_id: str, registry: Path, reason: str, messages: list[str]) -> None:
    previous = read_registry(registry).get(job_id, {})
    if not previous.get("unknown") or previous.get("reason") != reason:
        record({"event": "unknown", "id": job_id, "host": "remote", "reason": reason}, registry)
        messages.append(f"{job_id}: remote state unknown ({reason}); no relaunch")


def _verify_synced_exit(run_id: str, state: dict, result: dict, runs: Path) -> None:
    expected = {"host": "remote", "job_id": run_id, "attempt": state["attempts"],
                "git_commit": state["commit"], "image_digest": state["image_digest"], "gpu_uuids": [state["gpu"]]}
    required = state.get("required_results")
    if not required or not any(name not in {"exit.json", "provenance.json"} for name in required):
        raise RemoteJobError("Remote numerical result requirements are missing")
    root = (runs / run_id).resolve()
    for name in required:
        path = (root / name).resolve()
        if (root not in path.parents or not path.is_file() or path.stat().st_size == 0
                or (root / name).is_symlink()):
            raise RemoteJobError("Required numerical result is missing or outside its run")
        if file_sha256(path) != result.get("files", {}).get(name):
            raise RemoteJobError("Required numerical result checksum is missing or changed")
    for name in ("exit.json", "provenance.json"):
        path = runs / run_id / name
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != result.get("files", {}).get(name):
            raise RemoteJobError("Final remote exit/provenance checksums are missing or changed")
        report = json.loads(payload)
        if any(report.get(key) != value for key, value in expected.items()):
            raise RemoteJobError("Final remote exit/provenance identity mismatch")
        if name == "exit.json" and (type(report.get("code")) is not int or report["code"] != 0):
            raise RemoteJobError("Final remote exit code is not verified successful")
        if name == "provenance.json":
            if (report.get("data_id") != state["data_identity"] or report.get("script") != state.get("script")
                    or report.get("arguments") != state.get("args")):
                raise RemoteJobError("Final remote data/config argument identity mismatch")
            normalized = report.get("numeric_config")
            if Path(state.get("script", "")).name in {"train.py", "train_rl.py", "train_sincro.py", "train_reviwo.py"}:
                if not isinstance(normalized, dict) or not normalized:
                    raise RemoteJobError("Final training numerical config is missing")
            if normalized is not None:
                digest = hashlib.sha256(json.dumps(normalized, sort_keys=True, separators=(",", ":"),
                                                   allow_nan=False).encode()).hexdigest()
                if report.get("config_sha256") != digest:
                    raise RemoteJobError("Final remote numerical config checksum mismatch")
            if (state.get("expected_config_sha256") is not None
                    and report.get("config_sha256") != state["expected_config_sha256"]):
                raise RemoteJobError("Final remote numerical config differs from declared config")


def _reconcile_remote(backend, snapshot, state: dict, registry: Path, runs: Path,
                      messages: list[str]) -> tuple[set[int], bool]:
    owned = set()
    reachable = True
    for job_id, current in state.items():
        if current.get("host", "local") != "remote" or current["status"] not in ACTIVE:
            continue
        item = snapshot.containers.get(current["container"])
        if item is None:
            _unknown(job_id, registry, "container missing from confirmed snapshot", messages)
            continue
        try:
            validate_container(item, job_id, current)
        except (RemoteJobError, KeyError, ValueError, TypeError):
            _unknown(job_id, registry, "container identity/isolation mismatch", messages)
            continue
        observed = item.get("State", {}).get("Status")
        if observed in {"running", "restarting", "paused"}:
            owned.update(item.get("pids", set()))
            if current["status"] == "launching" or current.get("unknown"):
                record({"event": "launched", "id": job_id, "host": "remote", "gpu": current["gpu"],
                        "attempt": current["attempts"], "container": current["container"], "container_id": item["Id"],
                        "commit": current["commit"], "image_digest": current["image_digest"],
                        "gpu_backend": current["gpu_backend"]}, registry)
            try:
                backend.sync_results(job_id, final=False)
            except RemoteUnreachable:
                _host_reachability(registry, False, "RemoteUnreachable")
                reachable = False
                _unknown(job_id, registry, "result sync SSH unavailable", messages)
            except (RemoteJobError, OSError, ValueError, RuntimeError, subprocess.SubprocessError):
                messages.append(f"{job_id}: live result sync unavailable; job remains running")
            continue
        if observed != "exited" or type(item.get("State", {}).get("ExitCode")) is not int:
            _unknown(job_id, registry, f"unconfirmed container exit ({observed})", messages)
            continue
        code = item["State"]["ExitCode"]
        if code != 0:
            record({"event": "exited", "id": job_id, "host": "remote", "attempt": current["attempts"], "code": code,
                    "container_id": item["Id"]}, registry)
            messages.append(f"{job_id} exited with {code} on remote")
            continue
        if current["status"] != "sync_pending":
            record({"event": "remote_exited", "id": job_id, "host": "remote", "code": 0,
                    "container_id": item["Id"]}, registry)
        try:
            if any(Path(name).parts[0] == "checkpoints" for name in current.get("required_results", [])):
                synced = backend.sync_results(job_id, final=True, checkpoints=True)
            else:
                synced = backend.sync_results(job_id, final=True)
            _verify_synced_exit(job_id, current, synced, runs)
        except RemoteUnreachable:
            _host_reachability(registry, False, "RemoteUnreachable")
            reachable = False
            _unknown(job_id, registry, "final result sync SSH unavailable", messages)
            continue
        except (RemoteJobError, OSError, ValueError, RuntimeError, subprocess.SubprocessError):
            messages.append(f"{job_id}: final result checksums not verified; dependencies remain blocked")
            continue
        record({"event": "results_synced", "id": job_id, "host": "remote", "final": True, "sync": synced}, registry)
        record({"event": "exited", "id": job_id, "host": "remote", "attempt": current["attempts"], "code": 0}, registry)
        messages.append(f"{job_id} exited with 0 on remote; results checksum verified")
    return owned, reachable


def tick(queue_path: Path = QUEUE, registry: Path = REGISTRY, runs: Path = RUNS, usage=None,
         available_gb: float | None = None, disk_bytes: tuple[int, int] | None = None, remote_backend=None) -> list[str]:
    registry.parent.mkdir(parents=True, exist_ok=True)
    with open(registry.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _tick(queue_path, registry, runs, usage, available_gb, disk_bytes, remote_backend)


def _tick(queue_path: Path, registry: Path, runs: Path, usage, available_gb, disk_bytes, remote_backend) -> list[str]:
    queue = load_queue(queue_path)
    jobs = {job["id"]: job for job in queue.get("jobs", [])}
    state, messages = read_registry(registry), []
    for job_id, current in state.items():
        if current.get("host", "local") != "local" or current["status"] != RUNNING:
            continue
        exit_file = runs / job_id / "exit_code"
        if exit_file.is_file() and exit_file.read_text().strip():
            code = int(exit_file.read_text().strip())
        elif not pid_alive(current["pid"]):
            code = -1
        else:
            continue
        record({"event": "exited", "id": job_id, "host": "local", "attempt": current["attempt"], "code": code}, registry)
        messages.append(f"{job_id} exited with {code}")
        try:
            os.killpg(current["pid"], signal.SIGTERM)
            messages.append(f"{job_id}: terminated leftover processes of its session")
        except ProcessLookupError:
            pass
    resources = {}
    remote_host = queue["hosts"].get("remote")
    remote_candidates = read_registry(registry)
    for job_id, current in remote_candidates.items():
        if current["status"] == "crashed" and current["failures"] <= int(jobs.get(job_id, {}).get("max_restarts", 1)):
            current["status"] = PENDING
    remote_needed = any(current.get("host") == "remote" and current["status"] in ACTIVE
                        for current in state.values()) or (
        remote_host is not None and remote_host.get("enabled") is True and not remote_host.get("hold", False)
        and not (queue_path.parent / "HOLD").exists()
        and any(job.get("host", "local") == "remote" for job in eligible(list(jobs.values()), remote_candidates))
    )
    if remote_needed:
        try:
            remote_backend = remote_backend or RemoteBackend(REPO, runs)
            snapshot = remote_backend.snapshot()
            _host_reachability(registry, True)
            owned, reachable = _reconcile_remote(remote_backend, snapshot, read_registry(registry), registry, runs, messages)
            if reachable and snapshot.resources_ready:
                foreign = {uuid for uuid, procs in snapshot.usage.items() if any(pid not in owned for pid, _ in procs)}
                resources["remote"] = {"available": snapshot.available_gb, "disk_total": snapshot.disk_total_bytes,
                                       "disk_free": snapshot.disk_free_bytes, "foreign": foreign}
            else:
                messages.append("remote: resource evidence unavailable; no remote jobs launched")
        except RemoteUnreachable:
            _host_reachability(registry, False, "RemoteUnreachable")
            messages.append("remote unreachable; active jobs remain unknown, local scheduling continues")
            for job_id, current in read_registry(registry).items():
                if current.get("host") == "remote" and current["status"] in ACTIVE:
                    _unknown(job_id, registry, "SSH unavailable", messages)
        except (RemoteJobError, OSError, ValueError, RuntimeError, subprocess.SubprocessError):
            record({"event": "host_query_failed", "id": HOST_EVENT_ID, "host": "remote"}, registry)
            messages.append("remote resource/status query failed; no remote jobs launched")
            for job_id, current in read_registry(registry).items():
                if current.get("host") == "remote" and current["status"] in ACTIVE:
                    _unknown(job_id, registry, "remote snapshot unavailable", messages)
    state = read_registry(registry)
    for job_id, current in state.items():
        if job_id in jobs and "host" in current and jobs[job_id].get("host", "local") != current["host"]:
            raise ValueError(f"job {job_id} host changed; existing runs cannot be migrated implicitly")
        if current["status"] == "crashed":
            if current["failures"] <= int(jobs.get(job_id, {}).get("max_restarts", 1)):
                current["status"] = PENDING
            else:
                record({"event": "failed", "id": job_id, "host": current.get("host", "local")}, registry)
                current["status"] = FAILED
                messages.append(f"{job_id} exhausted restarts; diagnose before retrying")
    running = [{"host": current.get("host", "local"), "gpu": current["gpu"],
                "mem_gb": reservation(current, jobs.get(job_id, {}), "mem_gb"), "pid": current.get("pid")}
               for job_id, current in state.items() if current["status"] in ACTIVE]
    try:
        usage = gpu_usage() if usage is None else usage
        sessions = {item["pid"] for item in running if item["host"] == "local"}
        foreign = {uuid for uuid, procs in usage.items() if any(session_id(pid) not in sessions for pid, _ in procs)}
        total, free = disk_usage(runs) if disk_bytes is None else disk_bytes
        resources["local"] = {"available": host_available_gb() if available_gb is None else available_gb,
                              "disk_total": total, "disk_free": free, "foreign": foreign}
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError, ValueError) as exc:
        messages.append(f"GPU process query failed ({type(exc).__name__}); no jobs launched on local")
    if (queue_path.parent / "HOLD").exists():
        return messages
    now = time.time()
    for host_name, resource in resources.items():
        host = queue["hosts"].get(host_name, {})
        ramp = 60 * float(host.get("host_ram_ramp_minutes", 10))
        for job_id, current in state.items():
            if current.get("host", "local") == host_name and current["status"] in ACTIVE:
                if now - float(current.get("since", 0)) < ramp:
                    resource["available"] -= reservation(current, jobs.get(job_id, {}), "ram_gb")
                resource["disk_free"] -= reservation(current, jobs.get(job_id, {}), "disk_gb") * GIB
        for uuid in resource["foreign"]:
            messages.append(f"foreign processes on {host_name}/{uuid}; not launching there")
    evidence, remote_ready_checked, identity = None, False, None
    for job in eligible(list(jobs.values()), state):
        host_name = job.get("host", "local")
        host, resource = queue["hosts"][host_name], resources.get(host_name)
        if resource is None or host.get("enabled", host_name == "local") is not True or host.get("hold", False):
            continue
        gpu = choose_gpu(job, host.get("limits", {}), running, resource["foreign"])
        if gpu is None:
            continue
        need, reserve = float(job.get("ram_gb", 0)), float(host.get("host_ram_reserve_gb", 0))
        if resource["available"] - need < reserve:
            messages.append(f"{job['id']}: waiting for host RAM ({resource['available']:.0f} GB available, "
                            f"needs {need:.0f} + {reserve:.0f}) on {host_name}")
            continue
        floor = max(float(host.get("disk_floor_gb", 0)) * GIB,
                    resource["disk_total"] * (max(0.15, float(host.get("disk_floor_fraction", 0.15)))
                                              if host_name == "remote" else float(host.get("disk_floor_fraction", 0))))
        disk_need = float(job.get("disk_gb", 0)) * GIB
        if resource["disk_free"] - disk_need < floor:
            messages.append(f"{job['id']}: waiting for {host_name} disk floor/reservation")
            if host_name == "remote":
                record({"event": "host_disk_low", "id": HOST_EVENT_ID, "host": host_name,
                        "free_bytes": resource["disk_free"], "floor_bytes": floor}, registry)
            continue
        attempt = state.get(job["id"], {}).get("attempts", 0) + 1
        if host_name == "remote":
            if not remote_ready_checked:
                remote_ready_checked = True
                try:
                    evidence = remote_backend.ready(host)
                except (RemoteJobError, OSError, ValueError, RuntimeError, subprocess.SubprocessError):
                    messages.append("remote: acceptance/pushed-commit evidence invalid; admission held")
            if evidence is None:
                continue
            previous = state.get(job["id"], {})
            if (previous.get("commit", host["commit"]) != host["commit"]
                    or previous.get("image_digest", host["image_digest"]) != host["image_digest"]
                    or previous.get("job_sha256", job_fingerprint(job)) != job_fingerprint(job)
                    or previous.get("data_identity", evidence["data_identity"]) != evidence["data_identity"]):
                messages.append(f"{job['id']}: retry code/image/config/data changed; admission held")
                continue
            check_guarded(job["script"])
            path, spec = remote_backend.prepare(job, gpu, attempt, host, evidence)
            event = {"id": job["id"], "host": "remote", "gpu": gpu, "attempt": attempt,
                     "container": spec["container"], "commit": spec["commit"], "image_digest": spec["image_digest"],
                     "gpu_backend": spec["gpu_backend"],
                     "data_identity": evidence["data_identity"], "evidence_sha256": evidence["manifest_sha256"],
                     "spec_path": str(path), "required_results": spec["required_results"],
                     "script": job["script"], "args": list(map(str, job.get("args", []))),
                     "job_sha256": job_fingerprint(job), "expected_config_sha256": job.get("expected_config_sha256"),
                     "mem_gb": float(job["mem_gb"]), "ram_gb": need, "disk_gb": float(job.get("disk_gb", 0))}
            record({"event": "launching", **event}, registry)
            resource["available"] -= need
            resource["disk_free"] -= disk_need
            running.append({"host": "remote", "gpu": gpu, "mem_gb": float(job["mem_gb"]), "pid": None})
            try:
                launched = remote_backend.launch(path)
                record({"event": "launched", **event, "container_id": launched["container_id"]}, registry)
                messages.append(f"launched {job['id']} on remote/{gpu} ({spec['container']}, attempt {attempt})")
            except RemoteUnreachable:
                _host_reachability(registry, False, "RemoteUnreachable")
                _unknown(job["id"], registry, "launch response unknown", messages)
                resources.pop("remote", None)
            except (RemoteJobError, OSError, ValueError, RuntimeError, subprocess.SubprocessError):
                _unknown(job["id"], registry, "launch outcome requires reconciliation", messages)
                resources.pop("remote", None)
        else:
            identity = source_identity() if identity is None else identity
            pid = launch(job, gpu, attempt, runs)
            record({"event": "launched", "id": job["id"], "host": "local", "pid": pid, "gpu": gpu,
                    "attempt": attempt, "mem_gb": float(job["mem_gb"]), "ram_gb": need,
                    "disk_gb": float(job.get("disk_gb", 0)), "job_sha256": job_fingerprint(job), **identity}, registry)
            resource["available"] -= need
            resource["disk_free"] -= disk_need
            running.append({"host": "local", "gpu": gpu, "mem_gb": float(job["mem_gb"]), "pid": pid})
            messages.append(f"launched {job['id']} on {gpu} (pid {pid}, attempt {attempt})")
    return messages


def status() -> str:
    queue, state = load_queue(), read_registry()
    rows = [f"{'id':40s} {'host':6s} {'status':12s} {'gpu':14s} {'process/container':24s} {'tries':>5s} {'fails':>5s}"]
    for job in queue.get("jobs", []):
        current = state.get(job["id"], {"status": PENDING, "attempts": 0})
        host = current.get("host", job.get("host", "local"))
        status_name = "unknown" if current.get("unknown") else current["status"]
        process = current.get("container", "-") if host == "remote" else current.get("pid", "-")
        rows.append(f"{job['id']:40s} {host:6s} {status_name:12s} {current.get('gpu', '-')[:14]:14s} "
                    f"{str(process):24s} {current['attempts']:>5d} {current.get('failures', 0):>5d}")
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("tick")
    sub.add_parser("status")
    daemon = sub.add_parser("daemon")
    daemon.add_argument("--interval", type=float, default=60.0)
    reset = sub.add_parser("reset", help="re-queue a failed job after diagnosis")
    reset.add_argument("job_id")
    args = parser.parse_args()
    if args.command == "tick":
        print("\n".join(tick()))
    elif args.command == "status":
        print(status())
    elif args.command == "reset":
        with open(REGISTRY.with_suffix(".lock"), "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            current = read_registry().get(args.job_id, {})
            if current.get("host") == "remote" and current.get("status") in ACTIVE:
                raise SystemExit("Cannot reset a remote job whose execution/sync outcome remains active or unknown")
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
