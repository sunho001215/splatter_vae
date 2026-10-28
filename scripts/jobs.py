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
from s4d.gpu_guard import APPROVED_HOST_GPUS, HOST_CONFIG_PATH, MIG_DEVICES  # noqa: E402
from s4d.remote_access import sha256 as file_sha256  # noqa: E402
from s4d.remote_jobs import ROOT as REMOTE_ROOT  # noqa: E402
from s4d.remote_jobs import (  # noqa: E402
    ContainerRemovalRefused,
    RemoteBackend,
    RemoteJobError,
    RemoteUnreachable,
    validate_container,
)
from s4d.remote_verification import validate_report, validate_verification, verification_resources  # noqa: E402
from s4d.third_party import working_tree_identity as third_party_identity  # noqa: E402
from scripts._nvidia_query import mig_query  # noqa: E402

QUEUE = REPO / "experiments/queue.yaml"
REGISTRY = REPO / "experiments/registry.jsonl"
RUNS = REPO / "runs"
PYTHON = sys.executable
GIB = 2**30
GUARD_CALL = re.compile(r"^(\w+\s*=\s*)?(guard_gpus|enforce_allowed_gpus)\(\)\s*$", re.MULTILINE)
DONE, FAILED, RUNNING, PENDING = "done", "failed", "running", "pending"
ACTIVE = {RUNNING, "launching", "sync_pending"}
# Capacity is reserved only while a job can still execute: a remote job whose container exited ("sync_pending": compute
# complete, awaiting artifact verification or local backup) holds no GPU, RAM or disk-growth reservation (2026-10-27).
RESERVING = {RUNNING, "launching"}
# Remote result syncs inside the scheduler tick copy only what advances dependencies (status, completion records,
# metrics, checkpoints, exports); bulk evaluation media follow in scripts/sync_media.py (2026-10-27).
ESSENTIAL = "essential"
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
        validate_verification(job, queued=True)
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
                job.pop("remote_verified", None)  # recorded only after an exit, so a launch always starts a new attempt
                job.pop("remote_verification_failed", None)
                if kind == "launched" and event.get("container_id"):
                    job.setdefault("attempt_containers", {})[str(event["attempt"])] = {
                        "container": event.get("container"), "container_id": event["container_id"]}
            elif kind == "unknown":
                job.update(unknown=True, reason=event.get("reason", "unknown remote state"))
            elif kind == "remote_exited":
                job.update(status="sync_pending", code=event["code"], unknown=False)
            elif kind == "results_synced":
                job.update(sync=event["sync"], unknown=False)
            elif kind == "remote_verified":
                # compute complete and the artifacts verified in place on the remote host (small records only): releases
                # same-host dependents; the job becomes done only after its local backup is verified (2026-10-27)
                job["remote_verified"] = {key: event[key] for key in ("attempt", "files", "checks") if key in event}
                job["unknown"] = False
                job.pop("remote_verification_failed", None)
            elif kind == "remote_verification_failed":  # content check failed: the job cannot complete until resolved
                job["remote_verification_failed"] = {"attempt": event.get("attempt"), "time": event["time"],
                                                     "reason": event.get("reason", "")}
            elif kind == "container_removed":
                job.setdefault("containers_removed", []).append(event.get("container_id"))
            elif kind == "container_kept":
                job.setdefault("containers_kept", []).append(event.get("container_id"))
            elif kind == "container_kept_cleared":  # a refusal traced to a defect of the check itself, since fixed
                job["containers_kept"] = [i for i in job.get("containers_kept", []) if i != event.get("container_id")]
            elif kind == "container_absent":
                job.setdefault("containers_absent", []).append(event.get("container_id"))
            elif kind == "exited":
                synced = job.get("sync", {})
                verified = job.get("host", "local") != "remote" or (synced.get("final") is True
                                                                              and synced.get("verified") is True)
                job.update(status=(DONE if verified else "sync_pending") if event["code"] == 0 else "crashed",
                           code=event["code"], unknown=False)
                attempt = event.get("attempt", job.get("attempt", 0))
                job.setdefault("attempt_codes", {})[str(attempt)] = event["code"]
                if event["code"] != 0 and any(stop["attempt"] == attempt for stop in job.get("intentional_stops", [])):
                    pass  # an owned, verified checkpoint pause: restartable from its checkpoint, not a failure
                elif event["code"] != 0:
                    failure = {"attempt": attempt,
                               "launch_time": job.get("attempt_since", event["time"]),
                               "time": event["time"], "code": event["code"]}
                    if not any(item["attempt"] == failure["attempt"] and item["launch_time"] == failure["launch_time"]
                               for item in job["failed_attempts"]):
                        job["failed_attempts"].append(failure)
                        job["failures"] += 1
            elif kind in {"checkpoint_paused", "intentional_stopped"} and event.get("intentional_stop") is True:
                # recorded by runs/setup/tools/pause_*_at_checkpoint.py after the checkpoint was published and
                # verified; the following non-zero exit of the same attempt is then not charged as a failure. The
                # checkpoint path, container id and exit code (remote pauses) identify a paused attempt's container
                # for cleanup (2026-10-28).
                stop = {key: event[key] for key in ("attempt", "checkpoint_step", "checkpoint_sha256", "reason",
                                                    "checkpoint", "container_id", "container_exit_code")
                        if key in event}
                if stop not in job.setdefault("intentional_stops", []):
                    job["intentional_stops"].append(stop)
                    # The daemon can observe the signal exit before the pause tool finishes hashing the checkpoint.
                    matching = [failure for failure in job["failed_attempts"] if failure["attempt"] == stop["attempt"]]
                    job["failed_attempts"] = [failure for failure in job["failed_attempts"]
                                              if failure["attempt"] != stop["attempt"]]
                    job["failures"] -= len(matching)
                    if matching and job["status"] == FAILED:
                        job["status"] = "crashed"
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
                reconciliation = event.get("reconciliation", {})
                if reconciliation.get("conclusion") == "launch refused before docker run; deterministic container never existed":
                    checks = reconciliation.get("checks", [])
                    if (job["status"] != "launching" or job.get("host") != "remote"
                            or reconciliation.get("attempt") != job.get("attempt") or len(checks) != 2
                            or any(check.get("containers_with_name") != 0 or check.get("remote_output") != "absent"
                                   for check in checks)):
                        raise ValueError(f"invalid refused-launch reconciliation for {event['id']}")
                    # There was no execution or checkpoint: a corrected launch specification is not a run retry.
                    job.pop("job_sha256", None)
                job.update(status=PENDING)
            elif kind == "migrated":
                # an owned run stopped at a recorded checkpoint pause moves to another host and resumes there from that
                # checkpoint under the same job id; its sources are identical in the destination release (user 2026-10-24)
                stops = [stop for stop in job.get("intentional_stops", []) if stop.get("attempt") == job.get("attempt")]
                if (job["status"] in ACTIVE or job.get("host", "local") != event.get("from_host") or not stops
                        or stops[-1].get("checkpoint_step") != event.get("checkpoint_step")
                        or event.get("host") not in {"local", "remote"} or event.get("host") == event.get("from_host")
                        or event.get("sources_identical") is not True
                        or not all(isinstance(event.get(key), str) and event[key] for key in ("commit", "image_digest"))):
                    raise ValueError(f"invalid migration event for {event['id']}")
                job.update(status=PENDING, host=event["host"], commit=event["commit"], image_digest=event["image_digest"],
                           migrated_from={"host": event["from_host"], "attempt": job["attempt"],
                                          "checkpoint_step": event["checkpoint_step"]})
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
    # Local devices are MIG instances: the compute-apps CSV query cannot attribute their processes, the per-GPU
    # detailed query can (GPU instance ID + PID).
    # The detailed per-GPU query lists every process; with ~60 local DrM processes it took 70 s (2026-10-18)
    found = mig_query(MIG_DEVICES, state_dir=REPO / "experiments", timeout=180)
    return {uuid: found[uuid]["processes"] for uuid in APPROVED_HOST_GPUS["local"]}


def check_guarded(script: str) -> Path:
    path = (REPO / script).resolve()
    if REPO not in path.parents or not path.is_file():
        raise ValueError(f"job script {script} is not a file inside the repository")
    if not GUARD_CALL.search(path.read_text()):
        raise ValueError(f"job script {script} does not call the GPU guard")
    return path


def dependency_ready(job: dict, dep: dict) -> bool:
    """A dependency is satisfied once it is done (results verified locally, the only case for a local job), or, for a
    remote job, once a remote dependency's artifacts are verified in place on the remote host for its final attempt
    (``remote_verified``): the files the dependent reads are there; the local backup follows separately."""
    if dep.get("status") == DONE:
        return True
    return (job.get("host", "local") == "remote" and dep.get("host") == "remote" and dep.get("status") == "sync_pending"
            and (dep.get("remote_verified") or {}).get("attempt") == dep.get("attempts"))


def eligible(jobs: list[dict], state: dict[str, dict]) -> list[dict]:
    ready = []
    for order, job in enumerate(jobs):
        if state.get(job["id"], {}).get("status", PENDING) != PENDING:
            continue
        if all(dependency_ready(job, state.get(dep, {})) for dep in job.get("deps", [])):
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
    purpose = validate_verification(job, queued=True)
    if purpose is not None:
        payload.update(verification_purpose=purpose, campaign_eligible=False,
                       verification_timeout_seconds=verification_resources(job, purpose),
                       verification_resources={key: job.get(key, 8 if key == "cpu_threads" else None)
                                               for key in ("ram_gb", "mem_gb", "disk_gb", "cpu_threads")})
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
    fingerprints.update(third_party_identity(REPO))  # pinned submodule commits, .gitmodules, patch files
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
    purpose = validate_verification({**state, "id": run_id, "attempt": state["attempts"]})
    campaign_eligible = purpose is None
    if (("verification_purpose" in result and (purpose is None or result["verification_purpose"] != purpose))
            or result.get("campaign_eligible", campaign_eligible) is not campaign_eligible):
        raise RemoteJobError("Remote result synchronization job kind differs")
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
    _check_exit_records(run_id, state, purpose, {name: (runs / run_id / name).read_bytes()
                                                 for name in ("exit.json", "provenance.json")}, result.get("files", {}))
    if purpose is not None:
        if not isinstance(state.get("source_sha256"), dict) or not state["source_sha256"]:
            raise RemoteJobError("Verification source identity is missing")
        name = "tests.json" if purpose == "native_suite" else "gpu_isolation.json"
        validate_report(purpose, json.loads((root / "evidence" / name).read_text()), state["source_sha256"],
                        state["commit"], state["image_digest"], state["gpu"])


def _check_exit_records(run_id: str, state: dict, purpose: str | None, payloads: dict[str, bytes],
                        files: dict[str, str]) -> None:
    """Identity checks of a finished remote attempt's exit.json and provenance.json (payload bytes and their recorded
    checksums), shared by the local-backup and the in-place remote verification."""
    expected = {"host": "remote", "job_id": run_id, "attempt": state["attempts"],
                "git_commit": state["commit"], "image_digest": state["image_digest"], "gpu_uuids": [state["gpu"]]}
    if purpose is not None:
        expected.update(verification_purpose=purpose, campaign_eligible=False)
    for name in ("exit.json", "provenance.json"):
        payload = payloads[name]
        if hashlib.sha256(payload).hexdigest() != files.get(name):
            raise RemoteJobError("Final remote exit/provenance checksums are missing or changed")
        report = json.loads(payload)
        if purpose is None and ("verification_purpose" in report or report.get("campaign_eligible", True) is not True):
            raise RemoteJobError("Verification results cannot complete a campaign job")
        if any(report.get(key) != value for key, value in expected.items()):
            raise RemoteJobError("Final remote exit/provenance identity mismatch")
        if name == "exit.json" and (type(report.get("code")) is not int or report["code"] != 0):
            raise RemoteJobError("Final remote exit code is not verified successful")
        if name == "provenance.json":
            if (report.get("data_id") != state["data_identity"] or report.get("script") != state.get("script")
                    or report.get("arguments") != state.get("args")):
                raise RemoteJobError("Final remote data/config argument identity mismatch")
            normalized = report.get("numeric_config")
            if Path(state.get("script", "")).name in {
                "train.py", "train_rl.py", "train_sincro.py", "train_reviwo.py",
                "train_snerl.py", "train_dynamo.py", "train_mvp3d.py",
            }:
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


WEIGHTS_ONLY_GLOBALS = re.compile(r"collections\.OrderedDict|torch\._utils\._rebuild_tensor_v2|torch\.[A-Za-z0-9]+Storage")
REMOTE_RECHECK_SECONDS = 300  # a failed in-place verification of the same attempt is retried at most this often
STORAGE_NAMES = {"float32": "FloatStorage", "float64": "DoubleStorage", "float16": "HalfStorage",
                 "bfloat16": "BFloat16Storage", "int64": "LongStorage", "int32": "IntStorage", "bool": "BoolStorage",
                 "uint8": "ByteStorage"}


def remote_check_names(state: dict) -> list[str]:
    """Files inspected in place: the required results, the exit / provenance records, and the completion record the
    script's budget check needs (pretraining completion.json, RL final.json)."""
    script = Path(state.get("script", "")).name
    mandatory = {"train.py": {"completion.json"}, "train_rl.py": {"final.json"}}.get(script, set())
    return sorted(set(state["required_results"]) | {"exit.json", "provenance.json"} | mandatory)


def _override(args: list, prefix: str) -> str | None:
    values = [str(value).split("=", 1)[1] for value in args if str(value).startswith(prefix)]
    return values[-1] if values else None


def _option(args: list, flag: str) -> str | None:
    args = list(map(str, args))
    return args[args.index(flag) + 1] if flag in args and args.index(flag) + 1 < len(args) else None


def encoder_signature(encoder_config: dict) -> dict[str, list]:
    """{parameter / buffer name: [storage dtype, shape]} of the encoder an export's configuration builds (CPU; the
    signature a strict ``load_state_dict`` of the export requires)."""
    import torch  # noqa: F401  (lazy: only exports need it)

    from s4d.model.encoder import Encoder, EncoderConfig
    config = {**encoder_config, "backbone": "scratch", "pretrained_path": None}
    module = Encoder(EncoderConfig(**config))
    return {name: [STORAGE_NAMES[str(value.dtype).removeprefix("torch.")], list(value.shape)]
            for name, value in module.state_dict().items()}


def signature_sha256(tensors: dict) -> str:
    """Digest of {name: [storage dtype, shape]} (inspection report form)."""
    found = {name: [value["dtype"], list(value["shape"])] for name, value in tensors.items()}
    return hashlib.sha256(json.dumps(found, sort_keys=True).encode()).hexdigest()


def _verify_remote_export(state: dict, archive: dict, registry_state: dict | None = None) -> dict:
    args = list(map(str, state.get("args", [])))
    checkpoint, run = _option(args, "--checkpoint"), _option(args, "--run")
    match = re.search(r"step_(\d+)\.pt$", checkpoint or "")
    if match is None or run is None:
        raise RemoteJobError("Export row does not name its source run and step checkpoint")
    if (archive.get("zip_ok") is not True or archive.get("format") != "splatter4d-encoder-v1"
            or not all(WEIGHTS_ONLY_GLOBALS.fullmatch(name) for name in archive.get("globals", [""]))):
        raise RemoteJobError("Remote export is not an intact weights-only splatter4d encoder export")
    if (archive.get("step") != int(match.group(1)) or archive.get("source_checkpoint") != checkpoint
            or archive.get("run_name") != Path(run).name):
        raise RemoteJobError("Remote export step / source checkpoint / run identity differs")
    tensors = archive.get("state_dict") or {}
    found = {name: [value["dtype"], list(value["shape"])] for name, value in tensors.items()}
    if not found or archive.get("state_dict_storage_problems") or found != encoder_signature(archive["encoder_config"]):
        raise RemoteJobError("Remote export tensors do not match the encoder its configuration builds")
    digest = signature_sha256(tensors)
    if registry_state is not None:
        # Codex review 2026-10-27: the export must come from its producer's verified checkpoint at the producer's
        # completed budget, checked before any consumer is released (not only at the later local closeout).
        producer = registry_state.get(Path(run).name) or {}
        relative = checkpoint.split("/", 2)[2] if checkpoint.startswith(f"runs/{Path(run).name}/") else None
        verified = producer.get("remote_verified") or {}
        evidence = (verified.get("files") if verified.get("attempt") == producer.get("attempts")
                    else (producer.get("sync") or {}).get("files") if producer.get("status") == DONE else None) or {}
        produced = (verified.get("checks") or {}).get("checkpoints", {}).get(relative, {})
        declared = _override(producer.get("args", []), "train.steps=")
        budget = (verified.get("checks") or {}).get("steps") or (int(declared) if declared else None)
        if relative is None or relative not in evidence or budget != archive["step"]:
            raise RemoteJobError("Remote export does not read its producer's verified checkpoint at the producer's budget")
        if produced.get("encoder_signature_sha256") not in (None, digest):
            raise RemoteJobError("Remote export tensors differ from its source checkpoint's encoder")
    return {"step": archive["step"], "tensors": len(found), "signature_sha256": digest,
            "policy_frames": archive.get("policy_frames"), "encoder_config": archive["encoder_config"]}


def _verify_remote_exit(run_id: str, state: dict, report: dict, registry_state: dict | None = None) -> dict:
    """In-place verification of a finished remote campaign attempt from the remote inspection report
    (``RemoteBackend.verify_remote_artifacts``): every required result present and nonempty with its checksum, the
    exit / provenance identity checks of the local path, the configured training budget (pretraining: completion.json
    steps = train.steps; RL: final.json completed at train.num_train_steps), training-checkpoint archive integrity and
    step, and for an encoder export its archive integrity, weights-only content, step / source / run identity and a
    tensor signature equal to the encoder its configuration builds. Returns the summary recorded with the event."""
    if validate_verification({**state, "id": run_id, "attempt": state["attempts"]}) is not None:
        raise RemoteJobError("Verification jobs complete through their own evidence synchronization")
    names = remote_check_names(state)
    files = report.get("files") or {}
    if report.get("root") != f"{REMOTE_ROOT}/runs/{run_id}" or report.get("problems") or set(files) != set(names):
        raise RemoteJobError("Remote artifact inspection reported missing or unsafe files")
    if any(not isinstance(value, dict) or not re.fullmatch(r"[0-9a-f]{64}", str(value.get("sha256")))
           or type(value.get("bytes")) is not int or value["bytes"] <= 0 for value in files.values()):
        raise RemoteJobError("Remote artifact checksums are incomplete")
    digests = {name: value["sha256"] for name, value in files.items()}
    texts = report.get("payloads") or {}

    def record_of(name: str) -> dict:
        text = texts.get(name)
        if not isinstance(text, str) or hashlib.sha256(text.encode()).hexdigest() != digests.get(name):
            raise RemoteJobError(f"Remote {name} is missing or differs from its checksum")
        return json.loads(text)
    _check_exit_records(run_id, state, None, {name: str(texts.get(name, "")).encode()
                                              for name in ("exit.json", "provenance.json")}, digests)
    script, args, checks = Path(state.get("script", "")).name, state.get("args", []), {}
    if script == "train.py":
        target, completion = _override(args, "train.steps="), record_of("completion.json")
        if target is None or completion.get("status") != "completed" or completion.get("steps") != int(target):
            raise RemoteJobError("Remote pretraining did not complete its configured budget")
        checks["steps"] = int(target)
    elif script == "train_rl.py":
        target, final = _override(args, "train.num_train_steps="), record_of("final.json")
        if target is None or final.get("completed") is not True or final.get("steps") != int(target):
            raise RemoteJobError("Remote RL training did not complete its configured budget")
        checks["steps"] = int(target)
    archives = report.get("archives") or {}
    for name in names:
        if name.startswith("checkpoints/") and name.endswith(".pt"):
            archive = archives.get(name) or {}
            named = re.search(r"step_(\d+)\.pt$", name)
            expected = int(named.group(1)) if named else checks.get("steps")
            if (archive.get("zip_ok") is not True or archive.get("encoder_storage_problems")
                    or (script == "train.py" and not archive.get("encoder"))
                    or (expected is not None and archive.get("step") != expected)):
                raise RemoteJobError(f"Remote checkpoint {name} failed its integrity / step check")
            checks.setdefault("checkpoints", {})[name] = {
                "step": archive.get("step"), "encoder_tensors": len(archive.get("encoder") or {}),
                "encoder_signature_sha256": signature_sha256(archive["encoder"]) if archive.get("encoder") else None}
    if script == "export_encoder.py":
        if "encoder.pt" not in names:
            raise RemoteJobError("Export rows must require encoder.pt")
        checks["export"] = _verify_remote_export(state, archives.get("encoder.pt") or {}, registry_state)
    return checks


def _verify_backup(run_id: str, state: dict, backup: dict, runs: Path) -> None:
    """A local backup completes a remote job: a checksum-verified final essential copy (scripts/sync_media.py) of an
    attempt verified in place on the remote, carrying the same checksums and passing the local checks."""
    if (backup.get("run_id") != run_id or backup.get("host") != "remote" or backup.get("verified") is not True
            or backup.get("final") is not True or backup.get("scope") != ESSENTIAL):
        raise RemoteJobError("No verified final local backup")
    verified = state.get("remote_verified")
    if verified is None or verified.get("attempt") != state["attempts"]:
        raise RemoteJobError("The attempt was not verified in place on the remote; a backup alone cannot complete it")
    if any(backup.get("files", {}).get(name) != digest for name, digest in verified.get("files", {}).items()):
        raise RemoteJobError("Local backup differs from the artifacts verified on the remote")
    _verify_synced_exit(run_id, state, backup, runs)


def _reconcile_remote(backend, snapshot, state: dict, registry: Path, runs: Path,
                      messages: list[str]) -> tuple[set[int], bool]:
    owned = set()
    reachable = True
    for job_id, current in state.items():
        if current.get("host", "local") != "remote" or current["status"] not in ACTIVE:
            continue
        item = snapshot.containers.get(current["container"])
        if item is None and _retired_after_verification(current):
            # 2026-10-28: removed by the cleanup once verified in place, before the local backup: not unknown; the job
            # completes from the bulk worker's backup as before
            _complete_from_backup(backend, job_id, current, registry, runs, messages)
            continue
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
            if current.get("verification_purpose") is None:
                continue  # campaign runs: live copies are made by scripts/sync_media.py, never inside the tick
            try:
                backend.sync_verification_results(job_id, current, final=False)
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
        if current.get("verification_purpose") is not None:
            try:
                synced = backend.sync_verification_results(job_id, current, final=True)
                _verify_synced_exit(job_id, current, synced, runs)
                synced = {**synced, "verification_purpose": current["verification_purpose"], "campaign_eligible": False}
            except RemoteUnreachable:
                _host_reachability(registry, False, "RemoteUnreachable")
                reachable = False
                _unknown(job_id, registry, "final result sync SSH unavailable", messages)
                continue
            except (RemoteJobError, OSError, ValueError, RuntimeError, subprocess.SubprocessError):
                messages.append(f"{job_id}: final result checksums not verified; dependencies remain blocked")
                continue
            record({"event": "results_synced", "id": job_id, "host": "remote", "final": True, "sync": synced}, registry)
            record({"event": "exited", "id": job_id, "host": "remote", "attempt": current["attempts"], "code": 0},
                   registry)
            messages.append(f"{job_id} exited with 0 on remote; results checksum verified")
            continue
        # Campaign job, compute complete (2026-10-27): (1) its artifacts are verified in place on the remote host from
        # one small inspection report, which releases same-host dependents; (2) it is done only once scripts/sync_media.py
        # has made a checksum-verified final local backup (cross-host dependents, closeouts, container cleanup).
        # Neither step downloads results inside the tick.
        failed = current.get("remote_verification_failed") or {}
        recheck = (failed.get("attempt") != current["attempts"]
                   or time.time() - float(failed.get("time", 0)) >= REMOTE_RECHECK_SECONDS)
        if (current.get("remote_verified") or {}).get("attempt") != current["attempts"] and recheck:
            try:
                report = backend.verify_remote_artifacts(job_id, remote_check_names(current))
                checks = _verify_remote_exit(job_id, current, report, state)
            except RemoteUnreachable:
                _host_reachability(registry, False, "RemoteUnreachable")
                reachable = False
                _unknown(job_id, registry, "remote artifact check SSH unavailable", messages)
                continue
            except (RemoteJobError, OSError, ValueError, KeyError, TypeError, RuntimeError,
                    subprocess.SubprocessError) as exc:
                checks = None
                reason = f"{type(exc).__name__}: {str(exc)[:300]}"
                record({"event": "remote_verification_failed", "id": job_id, "host": "remote",
                        "attempt": current["attempts"], "reason": reason}, registry)
                messages.append(f"{job_id}: artifacts not verified on the remote ({reason[:160]}); the job cannot "
                                "complete and its dependents remain blocked (rechecked every 5 min)")
            if checks is not None:
                files = {name: value["sha256"] for name, value in report["files"].items()}
                record({"event": "remote_verified", "id": job_id, "host": "remote", "attempt": current["attempts"],
                        "container_id": item["Id"], "files": files, "checks": checks}, registry)
                current = {**current, "remote_verified": {"attempt": current["attempts"], "files": files,
                                                          "checks": checks}}
                messages.append(f"{job_id} exited with 0 on remote; artifacts verified on the remote "
                                "(same-host dependents released; local backup pending)")
        _complete_from_backup(backend, job_id, current, registry, runs, messages)
    return owned, reachable


def _retired_after_verification(current: dict) -> bool:
    """A compute-complete campaign job (sync_pending after exit 0) whose final attempt was verified in place: its
    outputs are in persistent remote storage, so a container missing from a confirmed snapshot (removed by the cleanup,
    also when the removal's reply was lost, or confirmed absent) does not make it unknown (2026-10-28)."""
    return (current.get("status") == "sync_pending" and current.get("code") == 0
            and current.get("verification_purpose") is None and current.get("campaign_eligible") is not False
            and (current.get("remote_verified") or {}).get("attempt") == current["attempts"]
            and bool(current.get("container_id")))


def _complete_from_backup(backend, job_id: str, current: dict, registry: Path, runs: Path, messages: list[str]) -> None:
    """Done once scripts/sync_media.py's verified final local backup matches the in-place verification."""
    try:
        backup = backend.backup_record(job_id)
        if backup is None:
            return
        _verify_backup(job_id, current, backup, runs)
    except (RemoteJobError, OSError, ValueError, KeyError, TypeError):
        return  # no verified final backup yet: the job stays compute-complete (sync_pending), reserving nothing
    record({"event": "results_synced", "id": job_id, "host": "remote", "final": True, "sync": backup}, registry)
    record({"event": "exited", "id": job_id, "host": "remote", "attempt": current["attempts"], "code": 0}, registry)
    messages.append(f"{job_id}: local backup checksum verified; done")


CONTAINER_CLEANUP_PER_TICK = 10
FINAL_REMOTE_FILES = ("exit.json", "provenance.json", "console.log")
# A paused attempt's container (exit e.g. 143 after the pause signal) keeps its outputs in the bind-mounted run
# directory: its removal requires the recorded checkpoint, provenance.json and console.log. scripts/remote/entry.py also
# writes exit.json on SIGTERM, but a resumed attempt rewrites it (and a SIGKILL leaves none), so it is not required.
PAUSED_REMOTE_FILES = ("provenance.json", "console.log")


def _run_relative(job_id: str, path) -> str | None:
    """A recorded run file path (``runs/<id>/...`` or under the remote root) relative to its run directory, or None."""
    for prefix in (f"runs/{job_id}/", f"{REMOTE_ROOT}/runs/{job_id}/"):
        if isinstance(path, str) and path.startswith(prefix):
            relative = path[len(prefix):]
            parts = relative.split("/")
            if (re.fullmatch(r"[A-Za-z0-9._/-]{1,255}", relative)
                    and all(part not in {"", ".", ".."} for part in parts)):  # canonical, relative, no traversal
                return relative
    return None


def _remote_completed(current: dict) -> bool:
    """The final attempt is complete for container cleanup: it exited 0 and was verified in place on the remote
    (status sync_pending or done; user directive 2026-10-28: the local bulk backup no longer delays removal, the
    required files are in persistent remote storage), or the job is done with a verified final sync (records that
    predate the in-place verification)."""
    final = int(current.get("attempt", current["attempts"]))
    verified = current.get("remote_verified") or {}
    return ((current.get("status") == DONE and (current.get("sync") or {}).get("final") is True)
            or (current.get("status") in {"sync_pending", DONE} and current.get("code") == 0
                and verified.get("attempt") == current["attempts"] == final))


def _cleanup_targets(job_id: str, current: dict) -> list[tuple[int, str, str, int, tuple[str, ...]]]:
    """Containers of a remote campaign job that automatic cleanup may remove, as (attempt, name, id, expected exit code,
    required run files):
    - once the final attempt is complete (``_remote_completed``), its container (exit 0) and the containers of its
      earlier, failed or interrupted attempts (their recovery is complete), each only if its id was recorded at launch
      and, for earlier attempts, its exit code was recorded; required: exit / provenance / console and the job's
      required results;
    - (2026-10-28) the container of an intentionally paused attempt (a ``checkpoint_paused`` / ``intentional_stopped``
      record with ``intentional_stop`` true and the checkpoint path), whatever the job's state, once its non-zero exit
      code was recorded (and equals the pause record's, when it carries one): its outputs are in the bind-mounted run
      directory and a resume always starts a new attempt and container; required: ``PAUSED_REMOTE_FILES`` and the
      recorded checkpoint.
    Verification jobs and containers already removed, kept or found absent are excluded (verification containers whose
    registry rows predate the purpose field are recognised by their labels in ``_cleanup_completed_containers`` and in
    the removal command)."""
    if (current.get("host") != "remote" or current.get("verification_purpose") is not None
            or current.get("campaign_eligible") is False):
        return []
    targets: dict[str, tuple[int, str, str, int, tuple[str, ...]]] = {}
    codes = current.get("attempt_codes", {})
    launched_containers = current.get("attempt_containers", {})
    if _remote_completed(current) and current.get("container_id"):
        final = int(current.get("attempt", current["attempts"]))
        required = tuple(sorted(set(FINAL_REMOTE_FILES) | {str(value) for value in current.get("required_results", [])}))
        targets[current["container_id"]] = (final, current["container"], current["container_id"], 0, required)
        for key, launched in launched_containers.items():
            if int(key) != final and key in codes and launched.get("container") == f"s4d-{job_id}-a{int(key)}":
                targets.setdefault(launched["container_id"], (int(key), launched["container"], launched["container_id"],
                                                              int(codes[key]), required))
    for stop in current.get("intentional_stops", []):
        key, launched = str(stop.get("attempt")), launched_containers.get(str(stop.get("attempt"))) or {}
        checkpoint = _run_relative(job_id, stop.get("checkpoint"))
        if (key in codes and codes[key] != 0 and checkpoint is not None and launched.get("container_id")
                and launched.get("container") == f"s4d-{job_id}-a{key}"
                and stop.get("container_id", launched["container_id"]) == launched["container_id"]
                and stop.get("container_exit_code", codes[key]) == codes[key]
                and not (int(key) == current["attempts"] and current["status"] in ACTIVE)):
            required = tuple(sorted({*PAUSED_REMOTE_FILES, checkpoint}))
            targets.setdefault(launched["container_id"], (int(key), launched["container"], launched["container_id"],
                                                          int(codes[key]), required))
    settled = {value for key in ("containers_removed", "containers_kept", "containers_absent")
               for value in current.get(key, [])}
    return sorted(target for target in targets.values() if target[2] not in settled)


def _cleanup_completed_containers(backend, snapshot, state: dict, registry: Path, messages: list[str]) -> bool:
    """Remove the containers of completed remote jobs and of intentionally paused attempts (``_cleanup_targets``; user
    instructions 2026-10-21 and 2026-10-28), each by its recorded id, stopped with the expected exit code and labelled
    ours, after preserving its Docker log and inspect record in the remote run directory
    (``RemoteBackend.remove_container``, which holds the run's result-sync lock). Containers whose identity or state
    differs from the record, or whose run lacks required files, are kept and recorded once (``container_kept``);
    failed or interrupted containers of unfinished jobs are never targets. At most CONTAINER_CLEANUP_PER_TICK removal
    calls per tick; returns False if the host became unreachable (retried on a later tick)."""
    calls = 0
    for job_id, current in state.items():
        for attempt, name, container_id, code, required in _cleanup_targets(job_id, current):
            if calls >= CONTAINER_CLEANUP_PER_TICK:
                return True
            base = {"event": "container_removed", "id": job_id, "host": "remote", "attempt": attempt,
                    "container": name, "container_id": container_id}
            item = snapshot.containers.get(name)
            labels = ((item or {}).get("Config") or {}).get("Labels") or {}
            if labels.get("s4d.verification_purpose") or labels.get("s4d.campaign_eligible") == "false":
                continue  # verification containers are kept (early ones predate the registry's purpose field)
            if item is not None and (item.get("Id") != container_id or item.get("State", {}).get("Status") != "exited"
                                     or item.get("State", {}).get("ExitCode") != code):
                record({**base, "event": "container_kept", "reason": "identity or state differs from the record"},
                       registry)
                messages.append(f"{job_id}: container {name} kept (identity or state differs from the record)")
                continue
            calls += 1
            try:  # a container missing from the snapshot is checked by the command itself (absent, or refused)
                result = backend.remove_container(job_id, attempt, name, container_id, code, list(required))
            except RemoteUnreachable:
                _host_reachability(registry, False, "RemoteUnreachable")
                return False
            except ContainerRemovalRefused as exc:
                record({**base, "event": "container_kept", "reason": exc.reason}, registry)
                messages.append(f"{job_id}: container {name} kept ({exc.reason})")
                continue
            except (RemoteJobError, OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
                messages.append(f"{job_id}: container {name} kept for now (removal failed: {type(exc).__name__})")
                continue
            if result.get("absent"):  # gone already: with our receipt's hashes if an interrupted run removed it
                record({**base, **result, "event": "container_absent"}, registry)
                messages.append(f"{job_id}: container {name} already absent"
                                + (" (removal receipt found)" if "log_sha256" in result else ""))
            else:
                record({**base, **result}, registry)
                messages.append(f"{job_id}: container {name} removed (Docker log and inspect record kept)")
    return True


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
        if job_id in jobs and "attempt" in current:
            purpose = validate_verification(jobs[job_id], queued=True)
            if (purpose != current.get("verification_purpose")
                    or (purpose is not None and current.get("campaign_eligible") is not False)):
                raise ValueError(f"job {job_id} verification/campaign identity changed")
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
        remote_host is not None and remote_host.get("cleanup_completed_containers", True) is True
        and any(_cleanup_targets(job_id, current) for job_id, current in state.items())) or (
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
            if reachable and remote_host is not None and remote_host.get("cleanup_completed_containers", True) is True:
                reachable = _cleanup_completed_containers(remote_backend, snapshot, read_registry(registry), registry,
                                                          messages)
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
               for job_id, current in state.items() if current["status"] in RESERVING]
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
            if current.get("host", "local") == host_name and current["status"] in RESERVING:
                if now - float(current.get("since", 0)) < ramp:
                    resource["available"] -= reservation(current, jobs.get(job_id, {}), "ram_gb")
                resource["disk_free"] -= reservation(current, jobs.get(job_id, {}), "disk_gb") * GIB
        for uuid in resource["foreign"]:
            messages.append(f"foreign processes on {host_name}/{uuid}; not launching there")
    campaign_evidence, remote_ready_checked, identity = None, False, None
    verification_evidence = {}
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
            purpose = validate_verification(job, queued=True)
            if purpose is not None:
                if purpose not in verification_evidence:
                    verification_evidence[purpose] = None
                    try:
                        verification_evidence[purpose] = remote_backend.ready_verification(host, job)
                    except (RemoteJobError, OSError, ValueError, RuntimeError, subprocess.SubprocessError):
                        messages.append(f"remote: {purpose} bootstrap/pushed-commit evidence invalid; admission held")
                evidence = verification_evidence[purpose]
            else:
                if not remote_ready_checked:
                    remote_ready_checked = True
                    try:
                        campaign_evidence = remote_backend.ready(host)
                    except (RemoteJobError, OSError, ValueError, RuntimeError, subprocess.SubprocessError):
                        messages.append("remote: acceptance/pushed-commit evidence invalid; admission held")
                evidence = campaign_evidence
            if evidence is None:
                continue
            previous = state.get(job["id"], {})
            if (previous.get("commit", host["commit"]) != host["commit"]
                    or previous.get("image_digest", host["image_digest"]) != host["image_digest"]
                    or previous.get("job_sha256", job_fingerprint(job)) != job_fingerprint(job)
                    or previous.get("data_identity", evidence["data_identity"]) != evidence["data_identity"]):
                messages.append(f"{job['id']}: retry code/image/config/data changed; admission held")
                continue
            if purpose is None:
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
            if purpose is not None:
                event.update(verification_purpose=purpose, campaign_eligible=False, args=spec["args"],
                             verification_timeout_seconds=spec["verification_timeout_seconds"],
                             cpu_threads=spec.get("cpu_threads", 8), source_sha256=evidence["source_sha256"])
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
            # Device provenance: the MIG instance and its parent physical GPU (CUDA reports the parent's UUID).
            record({"event": "launched", "id": job["id"], "host": "local", "pid": pid, "gpu": gpu,
                    "gpu_parent": MIG_DEVICES[gpu][0] if gpu in MIG_DEVICES else gpu,
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
