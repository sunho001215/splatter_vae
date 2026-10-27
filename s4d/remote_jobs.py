from __future__ import annotations

import base64
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

from s4d import third_party
from s4d.gpu_guard import APPROVED_HOST_GPUS, LOCAL_MIG_PARENTS
from s4d.remote_access import RemoteClient, RemoteUnreachable
from s4d.remote_verification import (
    POLICIES,
    canonical_arguments,
    entry_verification,
    validate_verification,
    verification_artifacts,
    verification_output,
    verification_resources,
)

GIB = 2**30
RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
COMMIT = re.compile(r"[0-9a-f]{40}\Z")
DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
RESULT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*(?:/[A-Za-z0-9][A-Za-z0-9_.-]*)*\Z")
ROOT = "/home/compu/kaist/sunho"
# A local device ID (MIG instance or its parent GPU) that the remote isolation check must reject.
LOCAL_DEVICES = set(APPROVED_HOST_GPUS["local"]) | set(LOCAL_MIG_PARENTS)


class RemoteJobError(RuntimeError):
    pass


class ContainerRemovalRefused(RemoteJobError):
    """``removal_command`` refused for a lasting reason: the container differs from the record (exit 3) or required run
    files are missing (exit 4). The container is kept."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


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


# Project-ownership labels on every container created from 2026-10-21 on (the remote Docker host is shared). Containers
# created earlier lack them; their identity rests on the core labels above and the registry's recorded container id.
OWNERSHIP_LABELS = {"s4d.project": "splatter4d", "s4d.owner": "kaist-sunho"}


def container_labels(run_id: str, attempt: int, commit: str, image_digest: str) -> dict:
    return {**labels(run_id, attempt, commit, image_digest), **OWNERSHIP_LABELS}


_INSPECT_IDENTITY = ('{{.Id}}|{{.State.Status}}|{{.State.ExitCode}}|{{index .Config.Labels "s4d.managed"}}|'
                     '{{index .Config.Labels "s4d.host"}}|{{index .Config.Labels "s4d.run_id"}}|'
                     '{{index .Config.Labels "s4d.attempt"}}|{{.Name}}|{{.HostConfig.ReadonlyRootfs}}|'
                     # optional labels: Docker prints "<no value>" for a missing key in this position, so the value is
                     # printed only when present and non-empty (empty otherwise)
                     '{{with index .Config.Labels "s4d.verification_purpose"}}{{.}}{{end}}|'
                     '{{with index .Config.Labels "s4d.campaign_eligible"}}{{.}}{{end}}')
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}")


_REMOVAL_SCRIPT = r"""set -u; export LC_ALL=C
id=@ID@; tmp_log=; tmp_record=
trap 'rm -f -- "$tmp_log" "$tmp_record"' EXIT
gone() {
  if [ -s @RECEIPT@ ] && read -r rid rlog rinspect < @RECEIPT@ && [ "$rid" = "$id" ]; then
    printf '{"absent": true, "log_sha256": "%s", "inspect_sha256": "%s"}\n' "$rlog" "$rinspect"
  else
    printf '{"absent": true}\n'
  fi
  exit 0
}
check() {
  if ! info="$(docker inspect --format @FORMAT@ "$id" 2>&1)"; then
    case "$info" in
      *"No such object"*|*"No such container"*) gone;;
      *) printf 'inspect failed %s\n' "$info" >&2; exit 7;;
    esac
  fi
  test "$info" = @EXPECTED@ || { printf 'refused %s\n' "$info" >&2; exit 3; }
}
digest() {
  d="$(sha256sum < "$1")" || exit 5
  d="${d%% *}"
  case "$d" in *[!0-9a-f]*|"") exit 5;; esac
  [ "${#d}" -eq 64 ] || exit 5
}
check
test -d @RUN_DIR@ || exit 4
@REQUIRED@
mkdir -p @KEEP@ || exit 5
exec 9> @LOCK@ || exit 5
flock -w 60 9 || exit 8
check
tmp_log="$(mktemp @LOG_TMP@)" || exit 5
tmp_record="$(mktemp @RECORD_TMP@)" || exit 5
docker logs "$id" > "$tmp_log" 2>&1 || exit 5
docker inspect "$id" > "$tmp_record" || exit 5
digest "$tmp_log"; logsum="$d"
digest "$tmp_record"; inspectsum="$d"
mv -f -- "$tmp_log" @LOG@ || exit 5
tmp_log=
mv -f -- "$tmp_record" @RECORD@ || exit 5
tmp_record=
printf '%s %s %s\n' "$id" "$logsum" "$inspectsum" > @RECEIPT_TMP@ || exit 5
mv -f -- @RECEIPT_TMP@ @RECEIPT@ || exit 5
check
docker rm "$id" > /dev/null || exit 6
printf '{"removed": true, "log_sha256": "%s", "inspect_sha256": "%s"}\n' "$logsum" "$inspectsum"
"""


def removal_command(root: str, run_id: str, attempt: int, container: str, container_id: str, expected_code: int,
                    required: list[str]) -> str:
    """Remove one completed container of ours by its explicit id, after preserving its Docker log and inspect record
    under the run directory. Refuses (non-zero exit) unless the container still is exactly that id, stopped
    ("exited") with the expected exit code, labelled ours for this run and attempt, with a read-only root filesystem,
    not labelled as a verification container, and every required run file is present and non-empty under the authorized root. Identity and state are checked
    first, again under a per-container lock (overlapping executions, e.g. an SSH retry while a timed-out first
    execution still runs, are serialized), and once more immediately before ``docker rm`` (no force, no volume
    removal). The archive goes to unique temporary files, is checksummed (a checksum failure refuses), moved in place
    and described by a receipt before removal. Output: {"removed": true, hashes} when this execution removed it;
    {"absent": true, hashes} when it is gone and our receipt for this id exists (an earlier, interrupted execution
    removed it); {"absent": true} when it is gone without our receipt. Exit 3: identity or state differs; 4: required
    files missing; 5: archive failed; 6: docker rm failed; 7: inspect failed; 8: lock busy."""
    if root != ROOT:
        raise ValueError("Remote removal root is not authorized")
    if (not _RUN_ID.fullmatch(run_id) or container != f"s4d-{run_id}-a{int(attempt)}" or not SHA256.fullmatch(container_id)
            or any(not re.fullmatch(r"[A-Za-z0-9._/-]{1,255}", name) or ".." in name for name in required)):
        raise ValueError("Invalid container removal request")
    q = shlex.quote
    run_dir = f"{root}/runs/{run_id}"
    keep = f"{run_dir}/containers"
    log, record = f"{keep}/{container}.docker.log", f"{keep}/{container}.inspect.json"
    receipt = f"{keep}/{container}.removal-receipt"
    # read-only root filesystem: every output was written to the bind-mounted run directory, none only inside it;
    # verification containers (s4d.verification_purpose / s4d.campaign_eligible labels) are never removed
    expected = f"{container_id}|exited|{int(expected_code)}|true|remote|{run_id}|{int(attempt)}|/{container}|true||"
    values = {
        "@ID@": q(container_id), "@FORMAT@": q(_INSPECT_IDENTITY), "@EXPECTED@": q(expected), "@RUN_DIR@": q(run_dir),
        "@KEEP@": q(keep), "@LOCK@": q(f"{keep}/.{container}.lock"), "@LOG@": q(log), "@RECORD@": q(record),
        "@LOG_TMP@": q(log + ".tmp.") + "XXXXXX", "@RECORD_TMP@": q(record + ".tmp.") + "XXXXXX",
        "@RECEIPT_TMP@": q(receipt + ".tmp"), "@RECEIPT@": q(receipt),
        "@REQUIRED@": "\n".join(f"test -s {q(run_dir + '/' + name)} || {{ printf 'missing %s\\n' {q(name)} >&2; exit 4; }}"
                                for name in required),
    }
    script = _REMOVAL_SCRIPT
    for key, value in values.items():
        script = script.replace(key, value)
    return script


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
            or any(key in actual and actual[key] != value for key, value in OWNERSHIP_LABELS.items())
            or item.get("Image") != state["image_digest"]
            or not SHA256.fullmatch(str(item.get("Id", "")))
            or (state.get("container_id") and item["Id"] != state["container_id"])
            or state["gpu"] not in APPROVED_HOST_GPUS["remote"]
            or env.get("CUDA_VISIBLE_DEVICES") != state["gpu"] or not request_ok
            or not {"compute", "utility", "graphics"} <= set(env.get("NVIDIA_DRIVER_CAPABILITIES", "").split(","))):
        raise RemoteJobError(f"Remote container identity/isolation mismatch for {run_id}")
    try:
        purpose = validate_verification({**state, "id": run_id, "attempt": state["attempts"]})
        if purpose is None:
            if ("s4d.verification_purpose" in actual or any(key.startswith("S4D_VERIFICATION_") for key in env)
                    or actual.get("s4d.campaign_eligible", "true") != "true"):
                raise ValueError("Verification container cannot reconcile as a campaign job")
        else:
            arguments = canonical_arguments(purpose, run_id, state["attempts"], state["gpu"])
            bound = entry_verification(env, POLICIES[purpose][0], arguments)
            if (bound is None or bound["id"] != run_id or bound["attempt"] != state["attempts"]
                    or bound["verification_purpose"] != purpose
                    or actual.get("s4d.verification_purpose") != purpose or actual.get("s4d.campaign_eligible") != "false"
                    or env.get("S4D_EXPECTED_COMMIT") != state["commit"]
                    or env.get("S4D_IMAGE_DIGEST") != state["image_digest"]
                    or env.get("S4D_DATA_ID") != state["data_identity"]
                    or actual.get("s4d.gpu") != state["gpu"]
                    or config.get("Memory") != math.ceil(state["ram_gb"] * GIB)
                    or config.get("MemorySwap") != config["Memory"]
                    or config.get("Memory") != int(env["S4D_VERIFICATION_RAM_BYTES"])
                    or config.get("NanoCpus") != state.get("cpu_threads", 8) * 10**9
                    or bound["verification_timeout_seconds"] != state["verification_timeout_seconds"]
                    or bound["cpu_threads"] != state.get("cpu_threads", 8)
                    or int(env["S4D_VERIFICATION_GPU_BYTES"]) != math.ceil(state["mem_gb"] * GIB)
                    or int(env["S4D_VERIFICATION_DISK_BYTES"]) != math.ceil(state["disk_gb"] * GIB)
                    or any(actual.get("s4d." + label) != env["S4D_VERIFICATION_" + suffix]
                           for label, suffix in (("ram_bytes", "RAM_BYTES"), ("mem_bytes", "GPU_BYTES"),
                                                 ("disk_bytes", "DISK_BYTES")))
                    or item["Config"].get("Entrypoint") != ["python"]
                    or item["Config"].get("Cmd") != ["-I", "/workspace/splatter4d/scripts/remote/entry.py",
                                                   "--script", POLICIES[purpose][0], "--", *arguments]):
                raise ValueError("Verification container bound identity differs")
    except (ValueError, KeyError, TypeError) as exc:
        raise RemoteJobError("Remote verification/campaign identity or bounds mismatch") from exc
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
    fingerprints.update(third_party.commit_identity(repo, commit))  # pinned submodule commits, patch files
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
    if "verification_purpose" in manifest or manifest.get("campaign_eligible", True) is not True:
        raise RemoteJobError("Verification bootstrap cannot replace campaign acceptance")
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
                or not {"0", None} <= rejected or not LOCAL_DEVICES & rejected):
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

    def ready_verification(self, host: dict, job: dict) -> dict:
        purpose = validate_verification(job, queued=True)
        if purpose is None:
            raise RemoteJobError("Verification readiness requires an explicit typed purpose")
        commit, image = host.get("commit", ""), host.get("image_digest", "")
        if not isinstance(commit, str) or not COMMIT.fullmatch(commit) or not isinstance(image, str) or not DIGEST.fullmatch(image):
            raise RemoteJobError("Verification requires an immutable commit and image")
        manifests = host.get("verification_manifests", {})
        if not isinstance(manifests, dict):
            raise RemoteJobError("Verification manifests must be purpose-specific")
        reference = manifests.get(purpose)
        if not isinstance(reference, str) or not reference:
            raise RemoteJobError("Purpose-specific verification manifest is missing")
        raw = Path(reference)
        path = raw if raw.is_absolute() else self.repo / raw
        if self.repo not in path.parents or path.resolve() != path or not path.is_file():
            raise RemoteJobError("Verification manifest must be a symlink-free repository file")
        payload = path.read_bytes()
        manifest = json.loads(payload)
        if not isinstance(manifest, dict) or manifest.get("commit") != commit or manifest.get("image_digest") != image:
            raise RemoteJobError("Verification manifest immutable identity differs")
        _, identity, _ = verification_artifacts(manifest, purpose, self.repo, commit_source_sha256(self.repo, commit))
        result = subprocess.run(
            ["git", "-C", str(self.repo), "-c", "credential.helper=", "-c",
             "credential.helper=!gh auth git-credential", "ls-remote", "origin", "refs/heads/splatter4d"],
            capture_output=True, text=True, timeout=60, check=True,
        )
        advertised = result.stdout.split()
        if len(advertised) != 2 or not COMMIT.fullmatch(advertised[0]):
            raise RemoteJobError("Cannot verify pushed verification source")
        subprocess.run(["git", "-C", str(self.repo), "merge-base", "--is-ancestor", commit, advertised[0]],
                       capture_output=True, text=True, timeout=60, check=True)
        return {**manifest, "manifest_sha256": hashlib.sha256(payload).hexdigest(), "data_identity": identity}

    def prepare(self, job: dict, gpu: str, attempt: int, host: dict, evidence: dict) -> tuple[Path, dict]:
        purpose = validate_verification(job, queued=True)
        if purpose is None:
            if "verification_purpose" in evidence or evidence.get("campaign_eligible", True) is not True:
                raise RemoteJobError("Campaign preparation cannot reuse bootstrap evidence")
        else:
            if evidence.get("commit") != host["commit"] or evidence.get("image_digest") != host["image_digest"]:
                raise RemoteJobError("Verification preparation immutable identity differs")
            verification_artifacts(evidence, purpose, self.repo, commit_source_sha256(self.repo, host["commit"]))
        name = container_name(job["id"], attempt)
        if gpu not in APPROVED_HOST_GPUS["remote"]:
            raise ValueError("Remote launch GPU is not authorized")
        script = Path(job["script"])
        if script.is_absolute() or ".." in script.parts or not (self.repo / script).is_file():
            raise ValueError("Remote job script must be repository-relative")
        args = (canonical_arguments(purpose, job["id"], attempt, gpu) if purpose is not None
                else list(map(str, job.get("args", []))))
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
                "evidence": evidence, "labels": container_labels(job["id"], attempt, host["commit"], host["image_digest"])}
        if "cpu_threads" in job:
            spec["cpu_threads"] = job["cpu_threads"]
        if purpose is not None:
            spec.update(verification_purpose=purpose, campaign_eligible=False,
                        verification_timeout_seconds=verification_resources(job, purpose))
            spec["labels"].update({"s4d.verification_purpose": purpose, "s4d.campaign_eligible": "false"})
            validate_verification(spec)
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
            diagnostic = {"returncode": result.returncode,
                          "stdout": self.client._redact(result.stdout)[-8000:],
                          "stderr": self.client._redact(result.stderr)[-8000:]}
            path.with_suffix(".launch-error.json").write_text(json.dumps(diagnostic, indent=2) + "\n")
            raise RemoteJobError(f"Remote launcher exited {result.returncode}; launch outcome requires reconciliation")
        try:
            payload = json.loads(result.stdout)
        except ValueError:
            payload = {"container_id": result.stdout.strip()}
        if not isinstance(payload, dict) or not SHA256.fullmatch(str(payload.get("container_id", ""))):
            raise RemoteJobError("Remote launcher returned no verifiable container id")
        return payload

    def remove_container(self, run_id: str, attempt: int, container: str, container_id: str, expected_code: int,
                         required: list[str]) -> dict:
        """``removal_command`` over SSH; returns {"removed": True, "log_sha256", "inspect_sha256"} or {"absent": True}."""
        command = removal_command(self.client.root, run_id, attempt, container, container_id, expected_code, required)
        try:
            response = self.client.ssh(command, wait=False, idempotent=True, timeout=120)
        except RemoteUnreachable:
            self._save_snapshot(reachable=False, query_ok=False)
            raise
        if response.returncode == 3:
            raise ContainerRemovalRefused("identity or state differs from the record")
        if response.returncode == 4:
            raise ContainerRemovalRefused("required run files are missing")
        if response.returncode:
            raise RemoteJobError(f"Container removal exited {response.returncode}")
        try:
            payload = json.loads(response.stdout.strip().splitlines()[-1])
        except (IndexError, ValueError) as exc:
            raise RemoteJobError("Container removal returned no result") from exc
        hashes = SHA256.fullmatch(str(payload.get("log_sha256"))) and SHA256.fullmatch(str(payload.get("inspect_sha256")))
        if payload == {"absent": True} or (set(payload) == {"absent", "log_sha256", "inspect_sha256"}
                                           and payload["absent"] is True and hashes) or (
                set(payload) == {"removed", "log_sha256", "inspect_sha256"} and payload["removed"] is True and hashes):
            return payload
        raise RemoteJobError("Container removal returned an unexpected result")

    def sync_results(self, run_id: str, *, final: bool = False, checkpoints: bool = False, scope: str = "all",
                     only_checkpoints: list[str] | None = None) -> dict:
        try:
            result = self.client.sync_results(run_id, checkpoints=checkpoints, final=final, wait=False, scope=scope,
                                              only_checkpoints=only_checkpoints)
        except RemoteUnreachable:
            self._save_snapshot(reachable=False, query_ok=False)
            raise
        if (result.get("run_id") != run_id or result.get("host") != "remote" or result.get("final") is not final
                or result.get("verified") is not True or not result.get("files")
                or any(not SHA256.fullmatch(value) for value in result["files"].values())):
            raise RemoteJobError("Remote result synchronization returned no checksum evidence")
        return result

    def verify_remote_artifacts(self, run_id: str, names: list[str]) -> dict:
        """Inspect a finished run in place on the remote host, read-only: s4d/remote_artifact_check.py (standard library
        only) runs under the host's python3 and returns one small JSON report (checksums and sizes, the small records'
        text, archive integrity and tensor structure of .pt files). No result file is transferred."""
        if not RUN_ID.fullmatch(run_id) or not names or any(
                not isinstance(name, str) or not RESULT_NAME.fullmatch(name) or ".." in Path(name).parts for name in names):
            raise ValueError("Unsafe run id or artifact name")
        source = (Path(__file__).with_name("remote_artifact_check.py")).read_bytes()
        loader = ("import base64,sys;exec(compile(base64.b64decode('" + base64.b64encode(source).decode()
                  + "'),'remote_artifact_check','exec'))")
        run_dir = f"{ROOT}/runs/{run_id}"
        command = (f"set -eu; test -d {shlex.quote(run_dir)}; test ! -L {shlex.quote(run_dir)}; exec python3 -I -c "
                   + shlex.quote(loader) + " " + shlex.quote(run_dir) + " " + " ".join(shlex.quote(n) for n in names))
        try:
            response = self.client.checked_ssh(command, wait=False, idempotent=True, timeout=300)
        except RemoteUnreachable:
            self._save_snapshot(reachable=False, query_ok=False)
            raise
        report = json.loads(response.stdout)
        if not isinstance(report, dict) or report.get("root") != run_dir:
            raise RemoteJobError("Remote artifact inspection returned an unexpected report")
        report["checker_sha256"] = hashlib.sha256(source).hexdigest()
        return report

    def backup_record(self, run_id: str) -> dict | None:
        """The local essential-scope result record of ``run_id`` (written by RemoteClient.sync_results, e.g. from the
        bulk worker scripts/sync_media.py), or None."""
        if not RUN_ID.fullmatch(run_id):
            raise ValueError("Unsafe run id")
        path = self.client.network_dir / "results" / f"{run_id}.json"
        try:
            return json.loads(path.read_text())
        except FileNotFoundError:
            return None

    def sync_verification_results(self, run_id: str, state: dict, *, final: bool = False) -> dict:
        purpose = validate_verification({**state, "id": run_id, "attempt": state["attempts"]})
        if purpose is None:
            raise RemoteJobError("Verification sync cannot handle campaign jobs")
        output = verification_output(purpose, run_id, state["attempts"])
        names = ["provenance.json", "console.log"]
        if final:
            names += ["exit.json", *state["required_results"]]
        sources = {name: (f"{output}/{Path(name).name}" if name.startswith("evidence/")
                          else f"{ROOT}/runs/{run_id}/{name}") for name in names}
        lines = ["set -eu"]
        for source in sources.values():
            quoted = shlex.quote(source)
            lines += [f'test "$(realpath -e -- {quoted})" = {quoted}', f"test -f {quoted}",
                      f"sha256sum -- {quoted}"]
        command = "\n".join(lines)

        def checksums():
            response = self.client.checked_ssh(command, wait=False, idempotent=True, timeout=60)
            found = {}
            for line in response.stdout.splitlines():
                parts = line.split()
                if len(parts) != 2 or not SHA256.fullmatch(parts[0]) or parts[1] not in sources.values() or parts[1] in found:
                    raise RemoteJobError("Verification sync checksum identity differs")
                found[parts[1]] = parts[0]
            if set(found) != set(sources.values()):
                raise RemoteJobError("Verification sync checksums are incomplete")
            return {name: found[source] for name, source in sources.items()}

        try:
            before = checksums()
            for name, source in sources.items():
                target = self.runs / run_id / name
                if target.resolve() != target:
                    raise RemoteJobError("Verification sync destination must be symlink-free")
                target.parent.mkdir(parents=True, exist_ok=True)
                self.client.rsync_download(source, target, wait=False)
                if target.resolve() != target or not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != before[name]:
                    raise RemoteJobError("Downloaded verification result checksum differs")
            if checksums() != before:
                raise RemoteJobError("Remote verification results changed during synchronization")
        except RemoteUnreachable:
            self._save_snapshot(reachable=False, query_ok=False)
            raise
        return {"run_id": run_id, "host": "remote", "final": final, "verified": True, "files": before,
                "verification_purpose": purpose, "campaign_eligible": False}
