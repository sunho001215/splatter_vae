from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from s4d.gpu_guard import APPROVED_HOST_GPUS
from s4d.remote_access import RemoteClient, RemoteUnreachable

GIB = 2**30
RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
COMMIT = re.compile(r"[0-9a-f]{40}\Z")
DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
ROOT = "/home/compu/kaist/sunho"


class RemoteJobError(RuntimeError):
    pass


@dataclass
class RemoteSnapshot:
    usage: dict[str, list[tuple[int, float]]]
    available_gb: float
    disk_total_bytes: int
    disk_free_bytes: int
    containers: dict[str, dict]
    resources_ready: bool = True


def container_name(run_id: str, attempt: int) -> str:
    if not RUN_ID.fullmatch(run_id) or len(run_id) > 128 or int(attempt) < 1:
        raise ValueError("Unsafe remote job identity")
    return f"s4d-{run_id}-a{int(attempt)}"


def labels(run_id: str, attempt: int, commit: str, image_digest: str) -> dict:
    return {"s4d.managed": "true", "s4d.run_id": run_id, "s4d.attempt": str(attempt),
            "s4d.host": "remote", "s4d.commit": commit, "s4d.image_digest": image_digest}


def snapshot_command(root: str) -> str:
    if root != ROOT:
        raise ValueError("Remote snapshot root is not authorized")
    return (
        "set -eu; export LC_ALL=C; "
        "printf '__S4D_GPUS__\\n'; "
        "if ! nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits; then printf '__S4D_QUERY_ERROR__\\n'; fi; "
        "printf '__S4D_USAGE__\\n'; "
        "if ! nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits; "
        "then printf '__S4D_QUERY_ERROR__\\n'; fi; "
        "printf '__S4D_GRAPHICS__\\n'; if ! nvidia-smi; then printf '__S4D_QUERY_ERROR__\\n'; fi; "
        "printf '__S4D_MEMORY__\\n'; cat /proc/meminfo; "
        "printf '__S4D_DISK__\\n'; df -B1 --output=size,avail -- " + shlex.quote(root) + "; "
        "printf '__S4D_CONTAINERS__\\n'; "
        "ids=$(docker ps -aq --no-trunc --filter label=s4d.managed=true --filter label=s4d.host=remote);"
        "for id in $ids; do docker inspect --format '{{json .}}' \"$id\" || true; "
        "printf '__S4D_TOP__ %s\\n' \"$id\"; docker top \"$id\" -eo pid 2>/dev/null || true; "
        "printf '__S4D_END_TOP__\\n'; done; printf '__S4D_END__\\n'"
    )


def parse_snapshot(output: str) -> RemoteSnapshot:
    sections: dict[str, list[str]] = {}
    tops: dict[str, set[int]] = {}
    section, top = "", None
    for line in output.splitlines():
        if line.startswith("__S4D_TOP__ "):
            top = line.split()[1]
            if not SHA256.fullmatch(top):
                raise RemoteJobError("Invalid Docker container id in snapshot")
            tops[top] = set()
        elif line == "__S4D_END_TOP__":
            top = None
        elif line in {"__S4D_GPUS__", "__S4D_USAGE__", "__S4D_GRAPHICS__", "__S4D_MEMORY__",
                      "__S4D_DISK__", "__S4D_CONTAINERS__", "__S4D_END__"}:
            section = line
            sections.setdefault(section, [])
        elif top is not None:
            if line.strip().isdigit():
                tops[top].add(int(line.strip()))
        elif section:
            sections[section].append(line)
    required = {"__S4D_GPUS__", "__S4D_USAGE__", "__S4D_GRAPHICS__", "__S4D_MEMORY__",
                "__S4D_DISK__", "__S4D_CONTAINERS__", "__S4D_END__"}
    if not required <= sections.keys():
        raise RemoteJobError("Incomplete remote snapshot")
    ready = not any("__S4D_QUERY_ERROR__" in rows for rows in sections.values())
    indices = {}
    for line in sections["__S4D_GPUS__"]:
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit():
            indices[int(parts[0])] = parts[1]
    if not set(APPROVED_HOST_GPUS["remote"]) <= set(indices.values()):
        ready = False
    usage = {uuid: [] for uuid in APPROVED_HOST_GPUS["remote"]}
    for line in sections["__S4D_USAGE__"]:
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 3 and parts[0] in usage:
            if not parts[1].isdigit():
                ready = False
                continue
            memory = float(parts[2]) if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", parts[2]) else 0.0
            usage[parts[0]].append((int(parts[1]), memory))
    for line in sections["__S4D_GRAPHICS__"]:
        match = re.match(r"\|\s*(\d+)\s+\S+\s+\S+\s+(\d+)\s+([CG+]+)\s", line)
        if match and indices.get(int(match[1])) in usage:
            uuid, pid = indices[int(match[1])], int(match[2])
            if not any(existing == pid for existing, _ in usage[uuid]):
                usage[uuid].append((pid, 0.0))
    memory = next((line.split()[1] for line in sections["__S4D_MEMORY__"]
                   if line.startswith("MemAvailable:")), "0")
    if not memory.isdigit() or int(memory) <= 0:
        ready = False
        memory = "0"
    disk = [line.split() for line in sections["__S4D_DISK__"] if re.fullmatch(r"\s*\d+\s+\d+\s*", line)]
    if len(disk) != 1 or int(disk[0][0]) <= 0 or not 0 <= int(disk[0][1]) <= int(disk[0][0]):
        raise RemoteJobError("Invalid remote disk snapshot")
    containers = {}
    for line in sections["__S4D_CONTAINERS__"]:
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            name, identity = item["Name"].removeprefix("/"), item["Id"]
            if not name.startswith("s4d-") or not SHA256.fullmatch(identity) or name in containers:
                raise ValueError("Invalid container identity")
            item["pids"] = tops.get(identity, set())
            containers[name] = item
        except (KeyError, TypeError, ValueError) as exc:
            raise RemoteJobError("Invalid Docker inspect snapshot") from exc
    return RemoteSnapshot(usage, int(memory) / 2**20, int(disk[0][0]), int(disk[0][1]), containers, ready)


def validate_container(item: dict, run_id: str, state: dict) -> None:
    expected = labels(run_id, state["attempts"], state["commit"], state["image_digest"])
    actual = item.get("Config", {}).get("Labels", {}) or {}
    env = dict(value.split("=", 1) for value in item.get("Config", {}).get("Env", []) if "=" in value)
    config = item.get("HostConfig", {})
    requests = config.get("DeviceRequests", []) or []
    backend = state.get("gpu_backend")
    expected_device = f"nvidia.com/gpu={state['gpu']}" if backend == "cdi" else state["gpu"]
    request_ok = len(requests) == 1 and requests[0].get("DeviceIDs") == [expected_device]
    if backend == "cdi":
        request_ok = (request_ok and requests[0].get("Driver") == "cdi"
                      and requests[0].get("Count") == 0
                      and not requests[0].get("Capabilities") and not requests[0].get("Options")
                      and not config.get("Devices") and env.get("NVIDIA_VISIBLE_DEVICES") == "void")
    elif backend == "legacy":
        request_ok = (request_ok and requests[0].get("Driver") in ("", "nvidia")
                      and requests[0].get("Count", 0) in (0, 1)
                      and env.get("NVIDIA_VISIBLE_DEVICES") == state["gpu"])
    else:
        request_ok = False
    if (item.get("Name", "").removeprefix("/") != state["container"]
            or any(actual.get(key) != value for key, value in expected.items())
            or item.get("Image") != state["image_digest"]
            or not SHA256.fullmatch(str(item.get("Id", "")))
            or (state.get("container_id") and item["Id"] != state["container_id"])
            or state["gpu"] not in APPROVED_HOST_GPUS["remote"]
            or env.get("CUDA_VISIBLE_DEVICES") != state["gpu"] or not request_ok
            or not {"compute", "utility", "graphics"} <= set(env.get("NVIDIA_DRIVER_CAPABILITIES", "").split(","))):
        raise RemoteJobError(f"Remote container identity/isolation mismatch for {run_id}")
    if (config.get("Privileged") is not False or config.get("ReadonlyRootfs") is not True
            or config.get("PidMode") != "host" or config.get("ShmSize") != 8 * GIB):
        raise RemoteJobError(f"Remote container runtime isolation mismatch for {run_id}")
    repository = "/workspace/splatter4d"
    runtime = f"{ROOT}/runtime"
    acceptance = f"{runtime}/acceptance/{state['commit']}-{state['image_digest'][7:]}"
    proof = f"{ROOT}/releases/{state['commit']}/provenance.json"
    expected_mounts = {
        repository: (f"{ROOT}/releases/{state['commit']}/code", False),
        f"{repository}/runs": (f"{ROOT}/runs", True),
        f"{repository}/outputs": (f"{ROOT}/outputs", True),
        f"{repository}/.cache": (f"{runtime}/cache", True),
        runtime: (runtime, True),
        acceptance: (acceptance, False),
        proof: (proof, False),
        "/home/ws/data/metaworld/splatter4d_v1": (f"{ROOT}/data/metaworld/splatter4d_v1", False),
    }
    mounts = item.get("Mounts", [])
    if (len(mounts) != len(expected_mounts)
            or {mount.get("Destination") for mount in mounts} != set(expected_mounts)):
        raise RemoteJobError(f"Remote container mount mapping mismatch for {run_id}")
    for mount in mounts:
        source, writable = expected_mounts[mount["Destination"]]
        if mount.get("Type") != "bind" or mount.get("Source") != source or mount.get("RW") is not writable:
            raise RemoteJobError(f"Remote container mount identity/permissions mismatch for {run_id}")


def _artifact(reference: dict, repo: Path, commit: str, image_digest: str) -> dict:
    if not isinstance(reference, dict) or not SHA256.fullmatch(str(reference.get("sha256", ""))):
        raise RemoteJobError("Acceptance artifact requires a SHA256")
    path = Path(reference.get("path", ""))
    path = (repo / path).resolve() if not path.is_absolute() else path.resolve()
    if repo.resolve() not in path.parents or not path.is_file():
        raise RemoteJobError("Acceptance artifact must be inside the authoritative repository")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != reference["sha256"]:
        raise RemoteJobError("Acceptance artifact checksum mismatch")
    result = json.loads(payload)
    if (result.get("commit", result.get("expected_commit")) != commit
            or result.get("image_digest", result.get("expected_image_digest")) != image_digest):
        raise RemoteJobError("Acceptance artifact code/image identity mismatch")
    return result.get("report", result)


def commit_source_sha256(repo: Path, commit: str) -> dict[str, str]:
    tree = subprocess.run(["git", "-C", str(repo), "ls-tree", "-r", "-z", commit,
                           "mujoco_mig_setup.py", "s4d", "scripts", "tests", "configs"],
                          capture_output=True, check=True).stdout
    entries = []
    for entry in tree.split(b"\0"):
        if not entry:
            continue
        metadata, path = entry.split(b"\t", 1)
        _, kind, blob = metadata.split()
        name = path.decode()
        if kind == b"blob" and Path(name).suffix in (".py", ".yaml", ".sh"):
            entries.append((name, blob))
    if not entries:
        raise RemoteJobError("Remote commit has no method sources")
    data = subprocess.run(["git", "-C", str(repo), "cat-file", "--batch"],
                          input=b"\n".join(blob for _, blob in entries) + b"\n",
                          capture_output=True, check=True).stdout
    offset, fingerprints = 0, {}
    for name, _ in entries:
        end = data.index(b"\n", offset)
        fields = data[offset:end].split()
        if len(fields) != 3 or fields[1] != b"blob":
            raise RemoteJobError("Cannot read remote commit sources")
        size = int(fields[2])
        payload = data[end + 1:end + 1 + size]
        fingerprints[name] = hashlib.sha256(payload).hexdigest()
        offset = end + size + 2
    return fingerprints


def acceptance_manifest(host: dict, repo: Path) -> dict:
    commit, image = str(host.get("commit", "")), str(host.get("image_digest", ""))
    if not COMMIT.fullmatch(commit) or not DIGEST.fullmatch(image):
        raise RemoteJobError("Remote launch requires explicit commit and immutable image digest")
    path = Path(host.get("evidence_manifest", ""))
    path = (repo / path).resolve() if not path.is_absolute() else path.resolve()
    if repo.resolve() not in path.parents or not path.is_file():
        raise RemoteJobError("Remote acceptance manifest is missing")
    manifest_bytes = path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("commit") != commit or manifest.get("image_digest") != image:
        raise RemoteJobError("Remote acceptance manifest code/image identity mismatch")
    suite = _artifact(manifest.get("full_suite"), repo, commit, image)
    counts = suite.get("counts", {})
    if (suite.get("full_suite_passed") is not True or suite.get("exit_code") != 0
            or counts.get("passed", 0) <= 0 or any(counts.get(key, -1) != 0 for key in ("failed", "error", "skipped"))
            or not suite.get("source_sha256") or suite.get("host") != "remote"
            or suite.get("source_changed_during_suite") is not False
            or suite.get("source_sha256_before") != suite.get("source_sha256")):
        raise RemoteJobError("Remote full native suite is not accepted")
    if suite["source_sha256"] != commit_source_sha256(repo, commit):
        raise RemoteJobError("Remote suite source fingerprints do not match the pushed commit")
    isolation = manifest.get("gpu_isolation", {})
    if set(isolation) != set(APPROVED_HOST_GPUS["remote"]):
        raise RemoteJobError("Remote GPU isolation evidence must cover every approved GPU")
    for uuid, reference in isolation.items():
        report = _artifact(reference, repo, commit, image)
        matches = [item for item in report.get("results", []) if item.get("uuid") == uuid]
        rejected = {item.get("cuda_visible_devices") for item in report.get("results", [])
                    if item.get("rejected") is True}
        child = matches[0].get("child", {}) if len(matches) == 1 else {}
        if (report.get("passed") is not True or report.get("host") != "remote"
                or report.get("source_sha256") != suite["source_sha256"]
                or report.get("source_changed_during_check") is not False
                or [item.get("uuid") for item in report.get("parent_mapping", [])] != [uuid]
                or len(matches) != 1 or matches[0].get("passed") is not True
                or matches[0].get("gpus_seen") != [uuid]
                or [item.get("uuid") for item in child.get("mapping", [])] != [uuid]
                or child.get("native_cuda_passed") is not True or child.get("native_egl_passed") is not True
                or not {"0", None, APPROVED_HOST_GPUS["local"][0]} <= rejected):
            raise RemoteJobError("Remote GPU isolation is not accepted")
    data = _artifact(manifest.get("data"), repo, commit, image)
    if (not SHA256.fullmatch(str(data.get("identity_sha256", ""))) or not data.get("files")
            or any(not SHA256.fullmatch(str(item.get("sha256", ""))) for item in data["files"])):
        raise RemoteJobError("Remote data identity is not verified")
    equivalence = _artifact(manifest.get("cross_host"), repo, commit, image)
    if (any(equivalence.get(key) is not True for key in ("passed", "step0_identical", "short_run_within_noise"))
            or equivalence.get("data_identity") != data["identity_sha256"] or not equivalence.get("protocol")):
        raise RemoteJobError("Cross-host equivalence is not accepted")
    return {**manifest, "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "data_identity": data["identity_sha256"]}


class RemoteBackend:
    def __init__(self, repo: Path, runs: Path, client: RemoteClient | None = None):
        self.repo, self.runs = repo.resolve(), runs.resolve()
        self.client = client or RemoteClient(repo=self.repo)
        if self.client.root != ROOT:
            raise ValueError("Remote backend root is not authorized")

    def snapshot(self) -> RemoteSnapshot:
        try:
            result = self.client.ssh(snapshot_command(self.client.root), wait=False, idempotent=True, timeout=60)
        except RemoteUnreachable:
            self._save_snapshot(reachable=False, query_ok=False)
            raise
        if result.returncode:
            self._save_snapshot(reachable=True, query_ok=False)
            raise RemoteJobError(f"Remote resource/status query exited {result.returncode}")
        try:
            snapshot = parse_snapshot(result.stdout)
        except RemoteJobError:
            self._save_snapshot(reachable=True, query_ok=False)
            raise
        self._save_snapshot(reachable=True, query_ok=True, snapshot=snapshot)
        return snapshot

    def _save_snapshot(self, *, reachable: bool, query_ok: bool, snapshot: RemoteSnapshot | None = None) -> None:
        path = self.runs / "remote/host_snapshot.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.loads(path.read_text()) if path.exists() else {"containers": {}}
        payload.update(checked_at=time.time(), reachable=reachable, query_ok=query_ok)
        if snapshot is not None:
            payload.update(containers=snapshot.containers, available_gb=snapshot.available_gb,
                           disk_total_bytes=snapshot.disk_total_bytes, disk_free_bytes=snapshot.disk_free_bytes,
                           resources_ready=snapshot.resources_ready)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, default=lambda value: sorted(value), sort_keys=True) + "\n")
        os.replace(temporary, path)

    def ready(self, host: dict) -> dict:
        evidence = acceptance_manifest(host, self.repo)
        result = subprocess.run(
            ["git", "-C", str(self.repo), "-c", "credential.helper=", "-c",
             "credential.helper=!gh auth git-credential", "ls-remote", "origin", "refs/heads/splatter4d"],
            capture_output=True, text=True, timeout=60, check=True,
        )
        advertised = result.stdout.split()
        if len(advertised) != 2 or not COMMIT.fullmatch(advertised[0]):
            raise RemoteJobError("Cannot verify origin/splatter4d")
        subprocess.run(["git", "-C", str(self.repo), "merge-base", "--is-ancestor", host["commit"], advertised[0]],
                       capture_output=True, text=True, check=True, timeout=60)
        return evidence

    def prepare(self, job: dict, gpu: str, attempt: int, host: dict, evidence: dict) -> tuple[Path, dict]:
        name = container_name(job["id"], attempt)
        if gpu not in APPROVED_HOST_GPUS["remote"]:
            raise ValueError("Remote launch GPU is not authorized")
        script = Path(job["script"])
        if script.is_absolute() or ".." in script.parts or not (self.repo / script).is_file():
            raise ValueError("Remote job script must be repository-relative")
        args = list(map(str, job.get("args", [])))
        if any("\x00" in value for value in args):
            raise ValueError("NUL in remote job arguments")
        adj = int(job.get("oom_score_adj", 0))
        if not 0 <= adj <= 1000:
            raise ValueError("oom_score_adj must be in [0, 1000]")
        backend = host.get("gpu_backend", "cdi")
        if backend not in ("cdi", "legacy"):
            raise ValueError("Remote GPU backend must be explicit CDI or legacy")
        spec = {"version": 1, "id": job["id"], "host": "remote", "attempt": attempt, "container": name,
                "gpu": gpu, "gpu_backend": backend, "commit": host["commit"], "image_digest": host["image_digest"],
                "root": self.client.root, "host_config": "configs/hosts/remote.yaml", "script": str(script),
                "args": args, "required_results": job["required_results"], "oom_score_adj": adj,
                "ram_gb": float(job.get("ram_gb", 0)), "mem_gb": float(job["mem_gb"]),
                "disk_gb": float(job.get("disk_gb", 0)), "disk_floor_gb": float(host.get("disk_floor_gb", 0)),
                "disk_floor_fraction": max(0.15, float(host.get("disk_floor_fraction", 0.15))),
                "max_jobs": int(host["limits"][gpu]["max_jobs"]),
                "max_mem_gb": float(host["limits"][gpu]["max_mem_gb"]),
                "host_ram_reserve_gb": float(host.get("host_ram_reserve_gb", 0)),
                "host_ram_ramp_minutes": float(host.get("host_ram_ramp_minutes", 10)),
                "evidence": evidence, "labels": labels(job["id"], attempt, host["commit"], host["image_digest"])}
        if any(not math.isfinite(spec[key]) or spec[key] < 0 for key in ("ram_gb", "disk_gb", "disk_floor_gb")):
            raise ValueError("Invalid remote resource reservation")
        folder = self.runs / "remote/jobs"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{job['id']}-a{attempt}.json"
        payload = json.dumps(spec, indent=2, sort_keys=True) + "\n"
        if path.exists() and path.read_text() != payload:
            raise RemoteJobError("Remote launch specification changed for an existing attempt")
        if not path.exists():
            with path.open("x") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        return path, spec

    def launch(self, path: Path) -> dict:
        try:
            result = subprocess.run(["bash", str(self.repo / "scripts/remote/docker_run.sh"), str(path)],
                                    capture_output=True, text=True, timeout=1200,
                                    env={**os.environ, "S4D_REMOTE_NO_WAIT": "1"})
        except subprocess.TimeoutExpired:
            self._save_snapshot(reachable=False, query_ok=False)
            raise RemoteUnreachable("Remote launch response timed out; reconcile its deterministic container") from None
        if result.returncode == 75:
            self._save_snapshot(reachable=False, query_ok=False)
            raise RemoteUnreachable("Remote launch response unknown; reconcile its deterministic container")
        if result.returncode:
            raise RemoteJobError(f"Remote launcher exited {result.returncode}; launch outcome requires reconciliation")
        try:
            payload = json.loads(result.stdout)
        except ValueError:
            payload = {"container_id": result.stdout.strip()}
        if not isinstance(payload, dict) or not SHA256.fullmatch(str(payload.get("container_id", ""))):
            raise RemoteJobError("Remote launcher returned no verifiable container id")
        return payload

    def sync_results(self, run_id: str, *, final: bool = False, checkpoints: bool = False) -> dict:
        try:
            result = self.client.sync_results(run_id, checkpoints=checkpoints, final=final, wait=False)
        except RemoteUnreachable:
            self._save_snapshot(reachable=False, query_ok=False)
            raise
        if (result.get("run_id") != run_id or result.get("host") != "remote" or result.get("final") is not final
                or result.get("verified") is not True or not result.get("files")
                or any(not SHA256.fullmatch(value) for value in result["files"].values())):
            raise RemoteJobError("Remote result synchronization returned no checksum evidence")
        return result
