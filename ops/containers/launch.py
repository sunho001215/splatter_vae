"""Validated remote container launcher.

Exit 20 carries a checked guard refusal envelope, emitted inside admission.lock
before docker run/create and only after both deterministic-name existence checks.
Exit 75 remains an unknown transport/outcome; every malformed refusal fails closed.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import sys
from pathlib import Path, PurePosixPath

from ops.resources.gpu_guard import APPROVED_HOST_GPUS, LOCAL_MIG_PARENTS, REMOTE_MIG_DEVICES
from ops.remote import launch_hold
from ops.remote.access import (ORIGIN_REF, RELEASE_ORIGIN_REFS, REMOTE_ROOT, RemoteClient, RemoteCommandError,
                               RemoteUnreachable, sha256)
from ops.jobs.host_roles import job_class
from ops.jobs.retired import retired_reason
from ops.jobs.remote_backend import (COTENANT_GPU_MARGIN_GB, COTENANT_LABELS, COTENANT_ROLES, OWNERSHIP_LABELS,
                                     commit_source_sha256, AdmissionRefused, GUARD_REFUSED_EXIT, validate_guard_refusal)
from ops.containers import suite_partition
from ops.containers.slot_guard import SOURCE as SLOT_GUARD_SOURCE
from ops.containers.verification import validate_verification, verification_artifacts, verification_output
from ops.containers.mounts import REALWORLD_READ_ROOTS, realworld_mounts

CONTAINER_REPO = "/workspace/splatter4d"
RAM_RESERVE_BYTES = 120 * 1024**3
SHM_SIZE_BYTES = 8 * 1024**3
DEFAULT_CPU_THREADS = 8  # per container; overridable per job with `cpu_threads`
MAX_CPU_THREADS = 64
ORIGIN_URL = "https://github.com/sunho001215/splatter_vae.git"
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")
# A local device ID (MIG instance or its parent GPU) that the remote isolation check must reject.
LOCAL_DEVICES = set(APPROVED_HOST_GPUS["local"]) | set(LOCAL_MIG_PARENTS)


def _text(value, name: str) -> str:
    if not isinstance(value, str) or any(ord(char) < 32 for char in value):
        raise ValueError(f"Invalid {name}")
    return value


def _number(value, name: str, minimum: float = 0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < minimum:
        raise ValueError(f"Invalid {name}")
    return float(value)


def _option(arguments: list[str], flag: str) -> str | None:
    values = []
    for index, argument in enumerate(arguments):
        if argument == flag:
            if index + 1 >= len(arguments) or arguments[index + 1].startswith("--"):
                raise ValueError(f"Missing value for {flag}")
            values.append(arguments[index + 1])
        elif argument.startswith(flag + "="):
            values.append(argument[len(flag) + 1:])
    if len(values) > 1:
        raise ValueError(f"Repeated {flag} is not accepted")
    return values[0] if values else None


def _output_path(value: str | None) -> str:
    if value is None:
        raise ValueError("Remote training requires an explicit result directory")
    path = PurePosixPath(value)
    if ".." in path.parts:
        raise ValueError("Remote result paths cannot traverse parents")
    return str(path if path.is_absolute() else PurePosixPath(CONTAINER_REPO) / path)


def validate_output_arguments(script: str, arguments: list[str], job_id: str) -> None:
    if script in (
        "scripts/train.py", "scripts/train_sincro.py", "scripts/train_reviwo.py",
        "scripts/train_snerl.py", "scripts/train_dynamo.py", "scripts/train_mvp3d.py",
    ):
        if (_option(arguments, "--name") != job_id
                or _output_path(_option(arguments, "--output-root")) != CONTAINER_REPO + "/runs"):
            raise ValueError("Remote pretraining artifacts must be under runs/<job_id>")
    if script in ("scripts/train_rl.py", "scripts/train_sac.py", "scripts/train_ppo.py", "scripts/train_dreamer.py",
                  "scripts/train_flashsac.py"):
        if _output_path(_option(arguments, "--run-dir")) != f"{CONTAINER_REPO}/runs/{job_id}":
            raise ValueError("Remote RL artifacts must be under runs/<job_id>")
    if script == "scripts/eval_rl.py":
        if _output_path(_option(arguments, "--output-dir")) != f"{CONTAINER_REPO}/runs/{job_id}":
            raise ValueError("Remote RL evaluation results must be written under runs/<job_id>")


def validate_spec(spec: dict) -> dict:
    if not isinstance(spec, dict) or spec.get("version") != 1 or spec.get("host") != "remote":
        raise ValueError("Expected a version-1 remote job spec")
    purpose = validate_verification(spec)
    job_id = _text(spec.get("id"), "job id")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,179}", job_id):
        raise ValueError("Unsafe job id")
    attempt = spec.get("attempt")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
        raise ValueError("Invalid attempt")
    name = f"s4d-{job_id}-a{attempt}"
    if spec.get("container") != name:
        raise ValueError("Container name must identify the exact job attempt")
    if spec.get("root") != REMOTE_ROOT or spec.get("host_config") != "configs/hosts/remote.yaml":
        raise ValueError("Unauthorized remote root or host config")
    if spec.get("gpu") not in APPROVED_HOST_GPUS["remote"]:
        raise ValueError("Unauthorized remote GPU")
    gpu_backend = spec.get("gpu_backend", "cdi")
    if gpu_backend not in ("cdi", "legacy"):
        raise ValueError("GPU backend must be cdi or explicitly declared legacy")
    commit, image = spec.get("commit"), spec.get("image_digest")
    if not isinstance(commit, str) or not HEX40.fullmatch(commit):
        raise ValueError("Expected an immutable commit")
    if not isinstance(image, str) or not image.startswith("sha256:") or not HEX64.fullmatch(image[7:]):
        raise ValueError("Expected an immutable Docker image ID")
    script = PurePosixPath(_text(spec.get("script"), "script"))
    if (not script.parts or script.is_absolute() or ".." in script.parts
            or script.parts[0] != "scripts" or script.suffix != ".py"):
        raise ValueError("Entry script must be a repository Python script")
    arguments = spec.get("args")
    if not isinstance(arguments, list):
        raise ValueError("Job arguments must be a list")
    for argument in arguments:
        _text(argument, "job argument")
    reason = retired_reason(spec)
    if reason is not None:
        raise ValueError(f"Launch refused: {reason}")
    classification = job_class(spec)
    if classification.cls in ("invalid", "ambiguous"):
        raise ValueError(f"Invalid or ambiguous remote job role: {classification.evidence}")
    validate_output_arguments(str(script), arguments, job_id)
    oom = spec.get("oom_score_adj", 500)
    if isinstance(oom, bool) or not isinstance(oom, int) or not 0 <= oom <= 1000:
        raise ValueError("Invalid OOM score")
    expected_labels = {
        "s4d.managed": "true", "s4d.run_id": job_id, "s4d.attempt": str(attempt), "s4d.host": "remote",
        "s4d.commit": commit, "s4d.image_digest": image, **OWNERSHIP_LABELS,
    }
    if purpose is not None:
        expected_labels.update({"s4d.verification_purpose": purpose, "s4d.campaign_eligible": "false"})
    if spec.get("labels") != expected_labels:
        raise ValueError("Container ownership labels differ from the job identity")
    if "enforce_limits" in spec and (spec["enforce_limits"] is not True or purpose is not None):
        raise ValueError("enforce_limits is an explicit flag of campaign attempts")
    if "repinned_from" in spec:  # a recorded release re-pin (ops/jobs/release_repin.py) of a campaign run
        source = spec["repinned_from"]
        if (purpose is not None or not isinstance(source, dict) or set(source) != {"commit", "image_digest"}
                or not isinstance(source["commit"], str) or not HEX40.fullmatch(source["commit"])
                or not isinstance(source["image_digest"], str) or not source["image_digest"].startswith("sha256:")
                or not HEX64.fullmatch(source["image_digest"][7:])
                or (source["commit"], source["image_digest"]) == (commit, image)):
            raise ValueError("Invalid release re-pin source")
    if not isinstance(spec.get("evidence"), dict):
        raise ValueError("Native acceptance evidence is required")
    result_files = spec.get("required_results")
    if not isinstance(result_files, list) or not result_files:
        raise ValueError("Numerical completion artifacts must be declared")
    for result_file in result_files:
        path = PurePosixPath(_text(result_file, "required result path"))
        if (not path.parts or path.is_absolute() or ".." in path.parts
                or path.parts[0] in ("replay", "wandb", ".cache")
                or str(path) in ("exit.json", "exit_code", "console.log", "provenance.json", "config.yaml")):
            raise ValueError("Numerical completion cannot rely on wrapper-only or unsafe artifacts")
    ram = _number(spec.get("ram_gb"), "RAM declaration", 0.001)
    memory = _number(spec.get("mem_gb"), "GPU-memory declaration", 0.001)
    disk = _number(spec.get("disk_gb"), "disk declaration")
    floor = _number(spec.get("disk_floor_gb"), "disk floor")
    fraction = _number(spec.get("disk_floor_fraction"), "disk fraction", 0.15)
    reserve = _number(spec.get("host_ram_reserve_gb"), "host RAM reserve", 120)
    ramp = _number(spec.get("host_ram_ramp_minutes"), "host RAM ramp", 10)
    cap = _number(spec.get("max_mem_gb"), "GPU-memory cap", 0.001)
    slots = spec.get("max_jobs")
    if (fraction > 1 or cap > 90 or memory > cap or isinstance(slots, bool)
            or not isinstance(slots, int) or not 1 <= slots <= 20):
        raise ValueError("Invalid host resource ceilings")
    if type(spec.get("exclusive_gpu", False)) is not bool:
        raise ValueError("Invalid exclusive_gpu")
    if spec.get("gpu") in REMOTE_MIG_DEVICES and cap > 21.5:
        raise ValueError("MIG capacity must preserve headroom (max 21.5 GiB)")
    threads = spec.get("cpu_threads", DEFAULT_CPU_THREADS)
    if isinstance(threads, bool) or not isinstance(threads, int) or not 1 <= threads <= MAX_CPU_THREADS:
        raise ValueError("Invalid CPU thread limit")
    return {**spec, "exclusive_gpu": spec.get("exclusive_gpu", False) or purpose is not None,
            "gpu_backend": gpu_backend, "cpu_threads": threads,
            "ram_bytes": math.ceil(ram * 1024**3), "gpu_bytes": math.ceil(memory * 1024**3),
            "gpu_cap_bytes": math.floor(cap * 1024**3), "reserve_bytes": math.ceil(reserve * 1024**3),
            "ramp_seconds": math.ceil(ramp * 60), "disk_bytes": math.ceil(disk * 1024**3),
            "floor_bytes": math.ceil(floor * 1024**3), "oom_score_adj": oom}


def evidence_artifacts(spec: dict, repository: Path) -> tuple[list[tuple[Path, str, str]], str, dict]:
    evidence = spec["evidence"]
    if evidence.get("commit") != spec["commit"] or evidence.get("image_digest") != spec["image_digest"]:
        raise ValueError("Acceptance identity differs from the job")
    purpose = validate_verification(spec)
    if purpose is not None:
        return verification_artifacts(evidence, purpose, repository, commit_source_sha256(repository, spec["commit"]))
    if "verification_purpose" in evidence or evidence.get("campaign_eligible", True) is not True:
        raise ValueError("Verification bootstrap cannot replace campaign acceptance")
    records = [(evidence.get("full_suite"), "tests.json"), (evidence.get("data"), "data.json"),
               (evidence.get("cross_host"), "cross_host.json")]
    isolation = evidence.get("gpu_isolation")
    if not isinstance(isolation, dict) or set(isolation) != set(APPROVED_HOST_GPUS["remote"]):
        raise ValueError("Isolation evidence must cover all four approved GPUs")
    records.extend((isolation[uuid], f"isolation-{uuid}.json") for uuid in APPROVED_HOST_GPUS["remote"])
    verified, payloads = [], {}
    for record, name in records:
        if (not isinstance(record, dict) or not isinstance(record.get("sha256"), str)
                or not HEX64.fullmatch(record["sha256"])):
            raise ValueError("Acceptance artifacts require SHA-256 identities")
        raw = Path(_text(record.get("path"), "evidence path"))
        path = (raw if raw.is_absolute() else repository / raw).resolve()
        if repository.resolve() not in path.parents or raw.is_symlink() or not path.is_file():
            raise ValueError("Acceptance evidence must be a repository file")
        if sha256(path) != record["sha256"]:
            raise ValueError("Acceptance artifact checksum differs")
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise ValueError("Acceptance artifacts must contain JSON objects")
        for key in ("commit", "image_digest"):
            if payload.get(key) != spec[key]:
                raise ValueError("Acceptance artifact provenance differs")
        report = payload.get("report", payload)
        if not isinstance(report, dict):
            raise ValueError("Acceptance report must be a JSON object")
        verified.append((path, name, record["sha256"]))
        payloads[name] = report
    suite = payloads["tests.json"]
    counts = suite.get("counts", {})
    if (suite.get("full_suite_passed") is not True or suite.get("exit_code") != 0 or not isinstance(counts, dict)
            or not isinstance(counts.get("passed"), int) or counts["passed"] <= 0):
        raise ValueError("Full native suite has not passed")
    if (suite.get("host") != "remote" or any(counts.get(kind, 1) for kind in ("failed", "error", "skipped"))
            or not isinstance(suite.get("source_sha256"), dict) or not suite["source_sha256"]):
        raise ValueError("Full native suite contains skipped or failing tests")
    if suite.get("partition", "all") != "all":  # user 2026-11-12: a partitioned suite is complete only when combined
        try:
            suite_partition.check_combined(suite)
        except ValueError as exc:
            raise ValueError(f"Full native suite partitions are inconsistent ({exc})") from None
    for uuid in APPROVED_HOST_GPUS["remote"]:
        report = payloads[f"isolation-{uuid}.json"]
        if (report.get("passed") is not True or report.get("host") != "remote"
                or report.get("source_sha256") != suite["source_sha256"]
                or report.get("pid_matching", {}).get("host_pid_namespace_asserted") is not True
                or [item.get("uuid") for item in report.get("parent_mapping", [])] != [uuid]):
            raise ValueError("GPU isolation acceptance is incomplete or stale")
        results = report.get("results", [])
        positives = [item for item in results if item.get("uuid") == uuid]
        if len(positives) != 1:
            raise ValueError("Expected one native CUDA/EGL isolation result per GPU")
        positive = positives[0]
        child = positive.get("child", {})
        if (positive.get("passed") is not True or positive.get("gpus_seen") != [uuid]
                or child.get("native_cuda_passed") is not True or child.get("native_egl_passed") is not True
                or [item.get("uuid") for item in child.get("mapping", [])] != [uuid]):
            raise ValueError("Native CUDA/EGL isolation has not passed")
        rejections = [item.get("cuda_visible_devices") for item in results if item.get("rejected") is True]
        if "0" not in rejections or None not in rejections or not LOCAL_DEVICES.intersection(rejections):
            raise ValueError("GPU isolation rejection checks are incomplete")
    data = payloads["data.json"]
    identity, files = data.get("identity_sha256"), data.get("files")
    if not isinstance(identity, str) or not HEX64.fullmatch(identity) or not isinstance(files, list) or not files:
        raise ValueError("Verified data identity is required")
    for item in files:
        if (not isinstance(item, dict) or not isinstance(item.get("sha256"), str) or not HEX64.fullmatch(item["sha256"])
                or not str(item.get("target", "")).startswith(REMOTE_ROOT + "/data/")):
            raise ValueError("Verified data manifest contains invalid files")
        _bound_path(item["target"])
    cross_host = payloads["cross_host.json"]
    if (any(cross_host.get(key) is not True for key in ("passed", "step0_identical", "short_run_within_noise"))
            or cross_host.get("data_identity") != identity or not cross_host.get("protocol")):
        raise ValueError("Cross-host equivalence has not passed for this code, data and protocol")
    if "data_identity" in evidence and evidence["data_identity"] != identity:
        raise ValueError("Acceptance manifest data identity is inconsistent")
    return verified, identity, payloads


def validate_release_proof(proof: dict, commit: str) -> str:
    if not isinstance(proof, dict) or proof.get("commit") != commit or proof.get("verified_ancestor") is not True:
        raise ValueError("Release is not verified against the pushed branch")
    tip = proof.get("origin_commit")
    if not isinstance(tip, str) or not HEX40.fullmatch(tip):
        raise ValueError("Release origin commit is invalid")
    # Either the single branch (main) or the legacy name a release was proven under; its recorded ref must match it.
    branch = proof.get("origin_branch")
    ref = RELEASE_ORIGIN_REFS.get(branch) if isinstance(branch, str) else None
    if ref is None or proof.get("origin_ref") != ref or proof.get("origin_url") != ORIGIN_URL:
        raise ValueError("Release origin is not the authorized branch")
    if "verified_origin_commit" in proof and proof["verified_origin_commit"] != tip:
        raise ValueError("Release origin proof is inconsistent")
    if not isinstance(proof.get("bundle_sha256"), str) or not HEX64.fullmatch(proof["bundle_sha256"]):
        raise ValueError("Release bundle identity is missing")
    if not proof.get("verified_at"):
        raise ValueError("Release verification time is missing")
    return tip


def _bound_path(path: str) -> str:
    candidate = PurePosixPath(path)
    if not candidate.is_absolute() or ".." in candidate.parts or PurePosixPath(REMOTE_ROOT) not in candidate.parents:
        raise ValueError("Bind source is outside the authorized root")
    return str(candidate)


def build_remote_command(spec: dict, data_identity: str, origin_commit: str, proof_sha256: str,
                         acceptance_sha256: dict[str, str], data_files: list[tuple[str, str]], *,
                         origin_ref: str = ORIGIN_REF) -> str:
    spec = validate_spec(spec)
    if (not HEX64.fullmatch(data_identity) or not HEX40.fullmatch(origin_commit) or not HEX64.fullmatch(proof_sha256)
            or origin_ref not in RELEASE_ORIGIN_REFS.values()):
        raise ValueError("Invalid launch provenance")
    root, name, commit, image = REMOTE_ROOT, spec["container"], spec["commit"], spec["image_digest"]
    code = f"{root}/releases/{commit}/code"
    proof = f"{root}/releases/{commit}/provenance.json"
    runtime = f"{root}/runtime"
    cache = f"{runtime}/cache"
    acceptance = f"{runtime}/acceptance/{commit}-{image[7:]}"
    data = f"{root}/data/metaworld/splatter4d_v1"
    # user 2026-11-12: the Meta-World development data was removed from the remote. Only a Meta-World job (its class
    # domain, ops/jobs/host_roles.job_class) needs, checks, checksum-verifies and mounts it; verification jobs (the
    # remote native suite deselects the tests that read it; they run in the release chain's local partition) and
    # real-world / DROID jobs do not.
    needs_metaworld = job_class(spec).domain == "metaworld"
    mounts = [(code, CONTAINER_REPO, True), (f"{root}/runs", f"{CONTAINER_REPO}/runs", False),
              (f"{root}/outputs", f"{CONTAINER_REPO}/outputs", False),
              (cache, f"{CONTAINER_REPO}/.cache", False), (runtime, runtime, False),
              (acceptance, acceptance, True), (proof, proof, True)]
    if needs_metaworld:
        mounts.append((data, "/home/ws/data/metaworld/splatter4d_v1", True))
    mounts += realworld_mounts(spec)
    docker = ["docker", "run", "-d", "--name", name, "--read-only", "--pid=host", "--user", "1000:1000",
              "--oom-score-adj", str(spec["oom_score_adj"]),
              "--workdir", CONTAINER_REPO, "--init", "--shm-size", "8g"]
    purpose = spec.get("verification_purpose")
    if purpose is not None or spec.get("enforce_limits") is True or spec["gpu"] in REMOTE_MIG_DEVICES:  # verification, or ARCH Phase 2 campaign limits
        docker += ["--memory", str(spec["ram_bytes"]), "--memory-swap", str(spec["ram_bytes"]),
                   "--cpus", str(spec["cpu_threads"])]
    if spec["gpu_backend"] == "cdi":
        docker += ["--device", f"nvidia.com/gpu={spec['gpu']}"]
    else:
        docker += ["--gpus", f"device={spec['gpu']}"]
    for key, value in spec["labels"].items():
        docker += ["--label", f"{key}={value}"]
    docker += ["--label", f"s4d.ram_bytes={spec['ram_bytes']}", "--label", f"s4d.disk_bytes={spec['disk_bytes']}",
               "--label", f"s4d.mem_bytes={spec['gpu_bytes']}", "--label", f"s4d.gpu={spec['gpu']}"]
    if spec["exclusive_gpu"] or spec["gpu"] in REMOTE_MIG_DEVICES:
        docker += ["--label", "s4d.exclusive_gpu=" + str(spec["exclusive_gpu"]).lower()]
    for source, target, readonly in mounts:
        _bound_path(source)
        docker += ["--mount", f"type=bind,src={source},dst={target}" + (",readonly" if readonly else "")]
    environment = {
        "CUDA_VISIBLE_DEVICES": spec["gpu"], "NVIDIA_DRIVER_CAPABILITIES": "compute,utility,graphics",
        "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl", "S4D_HOST_CONFIG": f"{CONTAINER_REPO}/configs/hosts/remote.yaml",
        "S4D_EXPECTED_COMMIT": commit, "S4D_IMAGE_DIGEST": image, "S4D_DATA_ID": data_identity,
        "S4D_JOB_ID": spec["id"], "S4D_ATTEMPT": str(spec["attempt"]), "S4D_RELEASE_PROOF": proof,
        "S4D_CACHE_ROOT": cache, "S4D_TEST_EVIDENCE_DIR": acceptance, "TMPDIR": f"{cache}/tmp",
        "HOME": f"{cache}/home", "LC_ALL": "C", "PYTHONDONTWRITEBYTECODE": "1",
        # No W&B credentials exist on the remote: runs log offline into their own run folder and are synced from main.
        "WANDB_MODE": "offline",
        # The remote host is shared: without a limit every PyTorch/OpenMP pool spans all host cores, and a few jobs
        # oversubscribe the CPU (ReViWo measured ~130 busy cores each). Thread count is not a hyperparameter.
        **{name: str(spec["cpu_threads"]) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
    }
    if "repinned_from" in spec:  # runtime provenance accepts exactly this transition (ops/jobs/run_identity.py)
        environment.update(S4D_REPINNED_FROM_COMMIT=spec["repinned_from"]["commit"],
                           S4D_REPINNED_FROM_IMAGE=spec["repinned_from"]["image_digest"],
                           S4D_REPINNED_TO_COMMIT=spec["commit"], S4D_REPINNED_TO_IMAGE=spec["image_digest"])
    if purpose is not None:
        environment.update({
            "S4D_TEST_EVIDENCE_DIR": verification_output(purpose, spec["id"], spec["attempt"]),
            "S4D_VERIFICATION_PURPOSE": purpose,
            "S4D_VERIFICATION_TIMEOUT_SECONDS": str(spec["verification_timeout_seconds"]),
            "S4D_VERIFICATION_RAM_BYTES": str(spec["ram_bytes"]),
            "S4D_VERIFICATION_GPU_BYTES": str(spec["gpu_bytes"]),
            "S4D_VERIFICATION_DISK_BYTES": str(spec["disk_bytes"]),
            "S4D_VERIFICATION_CPU_THREADS": str(spec["cpu_threads"]),
            "S4D_VERIFICATION_ACCEPTANCE_DIR": acceptance,
            "S4D_VERIFICATION_DATA_SHA256": spec["evidence"]["data"]["sha256"],
        })
        if purpose == "gpu_isolation":
            environment["S4D_VERIFICATION_SUITE_SHA256"] = spec["evidence"]["full_suite"]["sha256"]
    if spec["gpu_backend"] == "cdi":
        environment["NVIDIA_VISIBLE_DEVICES"] = "void"
    else:
        environment["NVIDIA_VISIBLE_DEVICES"] = spec["gpu"]
    for key, value in environment.items():
        docker += ["--env", f"{key}={value}"]
    docker += ["--entrypoint", "python", image, "-I", f"{CONTAINER_REPO}/scripts/remote/entry.py",
               "--script", spec["script"], "--", *spec["args"]]
    q = shlex.quote
    assignments = {"root": root, "container": name, "commit": commit, "image": image, "code": code, "proof": proof,
                   "gpu": spec["gpu"], "runtime": runtime, "acceptance": acceptance}
    lines = ["set -eu", "export LC_ALL=C GIT_OPTIONAL_LOCKS=0", *(f"{key}={q(value)}" for key, value in assignments.items()),
             'test ! -L "$root"; test "$(realpath -m -- "$root")" = "$root"',
             'bounded() { case "$1" in "$root"/*) ;; *) exit 1;; esac; '
             'test "$(realpath -m -- "$1")" = "$1"; '
             'current="$1"; while [ "$current" != "$root" ]; do '
             'test ! -L "$current"; current="$(dirname -- "$current")"; done; }',
             'existing() { names="$(docker ps -a --format \'{{.Names}}\')" || exit 1; '
             'if ! printf \'%s\\n\' "$names" | grep -Fxq -- "$container"; then return 1; fi;']
    for key, value in spec["labels"].items():
        template = '{{index .Config.Labels "' + key + '"}}'
        lines.append(f'  test "$(docker inspect --format {q(template)} "$container")" = {q(value)} || exit 1')
    if purpose is not None:
        for key, value in (("ram_bytes", spec["ram_bytes"]), ("mem_bytes", spec["gpu_bytes"]),
                           ("disk_bytes", spec["disk_bytes"]), ("gpu", spec["gpu"])):
            template = '{{index .Config.Labels "s4d.' + key + '"}}'
            lines.append('  test "$(docker inspect --format ' + q(template) + ' "$container")" = '
                         + q(str(value)) + ' || exit 1')
        for key, value in environment.items():
            if key.startswith("S4D_"):
                lines.append('  docker inspect --format \'{{range .Config.Env}}{{println .}}{{end}}\' "$container" '
                             '| grep -Fxq -- ' + q(f"{key}={value}") + ' || exit 1')
        for field, value in (("HostConfig.Memory", str(spec["ram_bytes"])),
                             ("HostConfig.MemorySwap", str(spec["ram_bytes"])),
                             ("HostConfig.NanoCpus", str(spec["cpu_threads"] * 10**9)),
                             ("Config.Entrypoint", json.dumps(["python"], separators=(",", ":"))),
                             ("Config.Cmd", json.dumps(["-I", f"{CONTAINER_REPO}/scripts/remote/entry.py",
                                                       "--script", spec["script"], "--", *spec["args"]],
                                                      separators=(",", ":")))):
            template = "{{json ." + field + "}}" if field.startswith("Config.") else "{{." + field + "}}"
            lines.append('  test "$(docker inspect --format ' + q(template) + ' "$container")" = '
                         + q(value) + ' || exit 1')
    if purpose is None and spec.get("enforce_limits") is True:  # lost-response reuse re-checks the enforced limits
        for field, value in (("HostConfig.Memory", str(spec["ram_bytes"])),
                             ("HostConfig.MemorySwap", str(spec["ram_bytes"])),
                             ("HostConfig.NanoCpus", str(spec["cpu_threads"] * 10**9))):
            lines.append('  test "$(docker inspect --format ' + q("{{." + field + "}}") + ' "$container")" = '
                         + q(value) + ' || exit 1')
    expected_mounts = "\n".join(sorted(f"bind|{source}|{target}|{'false' if readonly else 'true'}"
                                      for source, target, readonly in mounts))
    mount_template = '{{range .Mounts}}{{printf "%s|%s|%s|%t\\n" .Type .Source .Destination .RW}}{{end}}'
    lines.append('  actual_mounts="$(docker inspect --format ' + q(mount_template) + ' "$container" | sort)" || exit 1')
    if needs_metaworld:
        lines.append('  test "$actual_mounts" = ' + q(expected_mounts) + ' || exit 1')
    else:  # a container created before 2026-11-12 also carries the read-only Meta-World data mount (as reconciliation)
        legacy_mounts = "\n".join(sorted(f"bind|{source}|{target}|{'false' if readonly else 'true'}"
                                         for source, target, readonly in
                                         mounts + [(data, "/home/ws/data/metaworld/splatter4d_v1", True)]))
        lines.append('  { test "$actual_mounts" = ' + q(expected_mounts) + ' || test "$actual_mounts" = '
                     + q(legacy_mounts) + '; } || exit 1')
    lines += ['  test "$(docker inspect --format \'{{.Image}}\' "$container")" = "$image" || exit 1',
              '  test "$(docker inspect --format \'{{.HostConfig.Privileged}}\' "$container")" = false || exit 1',
              '  test "$(docker inspect --format \'{{.HostConfig.ReadonlyRootfs}}\' "$container")" = true || exit 1',
              '  test "$(docker inspect --format \'{{.HostConfig.PidMode}}\' "$container")" = host || exit 1',
              '  test "$(docker inspect --format \'{{.HostConfig.ShmSize}}\' "$container")" '
              f'= {SHM_SIZE_BYTES} || exit 1']
    if spec["gpu_backend"] == "cdi":
        request_template = ('{{range .HostConfig.DeviceRequests}}{{.Driver}}|{{.Count}}|'
                            '{{len .Capabilities}}|{{len .Options}}|{{range .DeviceIDs}}{{println .}}{{end}}{{end}}')
        lines += ['  test "$(docker inspect --format \'{{len .HostConfig.DeviceRequests}}\' "$container")" = 1 || exit 1',
                  '  test "$(docker inspect --format \'{{len .HostConfig.Devices}}\' "$container")" = 0 || exit 1',
                  '  test "$(docker inspect --format ' + q(request_template) +
                  ' "$container")" = "cdi|0|0|0|nvidia.com/gpu=$gpu" || exit 1',
                  '  docker inspect --format \'{{range .Config.Env}}{{println .}}{{end}}\' "$container" '
                  '| grep -Fxq -- NVIDIA_VISIBLE_DEVICES=void || exit 1']
    else:
        lines += ['  test "$(docker inspect --format \'{{range .HostConfig.DeviceRequests}}'
                  '{{range .DeviceIDs}}{{println .}}{{end}}{{end}}\' "$container")" = "$gpu" || exit 1']
    lines += ['  docker inspect --format \'{{range .Config.Env}}{{println .}}{{end}}\' "$container" '
              '| grep -Fxq -- "CUDA_VISIBLE_DEVICES=$gpu" || exit 1',
              '  cid="$(docker inspect --format \'{{.Id}}\' "$container")" || exit 1',
              '  printf \'{"container_id":"%s","container":"%s"}\\n\' "$cid" "$container"; return 0; }',
              'if existing; then exit 0; fi', 'bounded "$runtime/launch"; mkdir -p -- "$runtime/launch"',
              'bounded "$runtime/launch/admission.lock"; exec 9>"$runtime/launch/admission.lock"; flock -x 9',
              'if existing; then exit 0; fi',
              # durable launch hold (remote MIG migration, 2026-11-11): inside the admission lock, after the idempotent
              # check (an existing container is reported, never re-created), before anything is started
              launch_hold.check_shell(root)]
    for path in (code, proof, *([data] if needs_metaworld else []), acceptance, *(source for source, _, _ in mounts)):
        lines.append("bounded " + q(path))
    lines += ['test -d "$code/.git"' + ("; test -d " + q(data) if needs_metaworld else ""),
              'test "$(git -C "$code" rev-parse HEAD)" = "$commit"',
              'test -z "$(git -C "$code" status --porcelain --untracked-files=all)"',
              'test "$(git -C "$code" rev-parse ' + q(origin_ref) + ')" = ' + q(origin_commit),
              'git -C "$code" merge-base --is-ancestor "$commit" ' + q(origin_commit),
              'test "$(sha256sum -- "$proof" | cut -d\' \' -f1)" = ' + q(proof_sha256),
              "bounded " + q(f"{code}/{spec['script']}"), "test -f " + q(f"{code}/{spec['script']}"),
              'test "$(docker image inspect --format \'{{.Id}}\' "$image")" = "$image"',
              'test "$(docker image inspect --format \'{{index .Config.Labels "s4d.managed"}}\' "$image")" = true',
              'docker image inspect --format \'{{range .RepoTags}}{{println .}}{{end}}\' "$image" | grep -q \'^s4d-\'']
    for artifact, digest in acceptance_sha256.items():
        if (not re.fullmatch(r"(?:binding-)?(?:tests|data|cross_host|isolation-(?:GPU|MIG)-[0-9a-f-]+)\.json", artifact)
                or not HEX64.fullmatch(digest)):
            raise ValueError("Invalid acceptance checksum")
        lines.append("bounded " + q(f"{acceptance}/{artifact}"))
        lines.append('test "$(sha256sum -- ' + q(f"{acceptance}/{artifact}") + " | cut -d' ' -f1)\" = " + q(digest))
    if needs_metaworld and not data_files:
        raise ValueError("Dataset files must be checksum-verified before launch")
    for path, digest in (data_files if needs_metaworld else []):
        _bound_path(path)
        if not path.startswith(root + "/data/") or not HEX64.fullmatch(digest):
            raise ValueError("Invalid data checksum")
        lines.append("bounded " + q(path))
        lines.append('test "$(sha256sum -- ' + q(path) + " | cut -d' ' -f1)\" = " + q(digest))
    if spec["gpu"] in REMOTE_MIG_DEVICES:
        parent, instance, device = REMOTE_MIG_DEVICES[spec["gpu"]]
        guard_input = {"uuid": spec["gpu"], "parent": parent, "gpu_instance_id": instance, "device": device,
                       "max_jobs": spec["max_jobs"], "exclusive_gpu": spec["exclusive_gpu"],
                       "cpu_threads": spec["cpu_threads"], "cpu_thread_budget": spec.get("cpu_thread_budget", 320),
                       "idle_mib": 256, "gpu_bytes": spec["gpu_bytes"],
                       "capacity_bytes": spec["gpu_cap_bytes"], "ram_bytes": spec["ram_bytes"],
                       "reserve_bytes": spec["reserve_bytes"], "disk_bytes": spec["disk_bytes"],
                       "floor_bytes": spec["floor_bytes"], "disk_floor_fraction": spec["disk_floor_fraction"],
                       "root": root}
        # Only this branch emits exit 20 and the versioned refusal protocol. No
        # docker run/create has occurred; existing() ran again inside admission.lock.
        prefix = json.dumps({'version': 1, 'container': spec['container']})[:-1] + ', "guard": '
        lines += ["if verdict=$(python3 -I -c " + q(SLOT_GUARD_SOURCE) + " " + q(json.dumps(guard_input)) + "); then",
                  '  printf \'%s\\n\' "$verdict" >&2',
                  'else', '  guard_code=$?',
                  '  if [ "$guard_code" -eq 20 ]; then',
                  "    printf '%s%s}\\n' " + q(prefix) + ' "$verdict"; exit 20',
                  '  fi', '  printf \'%s\\n\' "$verdict" >&2; exit 1', 'fi']
    else:
        cotenant_filters = " ".join("--filter " + q(f"label={key}={value}")
                                    for key, value in sorted(COTENANT_LABELS.items()))
        # name|role|enforced RAM limit|s4d.gpu|s4d.mem_bytes|other devices|device requests (driver,count,capabilities,
        # options,ids;/ each); a label value is printed only when present. Only a record of exactly this shape, with
        # values in the character sets of remote_backend.identify_cotenant, is read (no separator or newline injection).
        label = '{{with index .Config.Labels "s4d.%s"}}{{.}}{{end}}'
        cotenant_format = ("{{.Name}}|" + label % "role" + "|{{.HostConfig.Memory}}|" + label % "gpu" + "|"
                           + label % "mem_bytes" + "|{{len .HostConfig.Devices}}|"
                           "{{range .HostConfig.DeviceRequests}}{{.Driver}},{{.Count}},{{len .Capabilities}},"
                           "{{len .Options}},{{range .DeviceIDs}}{{.}};{{end}}/{{end}}")
        value, device = "[A-Za-z0-9_.-]*", "[A-Za-z0-9_.=/:-]*"
        count = "[1-9][0-9]{0,14}"  # byte counts: canonical, bounded decimals (a leading zero would be read as octal)
        cotenant_record = (f"/[A-Za-z0-9][A-Za-z0-9_.-]*\\|{value}\\|(0|{count})\\|{value}\\|({count})?\\|[0-9]+\\|"
                           f"({device},-?[0-9]+,[0-9]+,[0-9]+,({device};)*/)*")
        if not all(re.fullmatch(r"[a-z][a-z-]*", role) for role in COTENANT_ROLES):
            raise ValueError("Invalid co-tenant role")
        probe = gpu_probe_lines(spec["gpu"])
        lines += probe[:-1]  # the selected device's processes (the graphics memory line follows the owned listing)
        lines += ['owned="$(docker ps --filter label=s4d.managed=true --filter label=s4d.host=remote --format \'{{.Names}}\')"',
                  probe[-1],
                  # GPU memory (bytes) used on the selected GPU by the processes in $1 (whitespace-separated PIDs)
                  'gpu_used() { printf \'%s\\n\' "$apps" "$graphics_memory" | awk -F, -v gpu="$gpu" -v pids=" $(printf \'%s \' $1)" '
                  '\'{gsub(/ /,"",$1); gsub(/ /,"",$2); gsub(/ /,"",$3); '
                  'if ($1==gpu && $3 ~ /^[0-9]+$/ && index(pids, " " $2 " ") && $3>mem[$2]) mem[$2]=$3} '
                  'END {for(p in mem)s+=mem[p]; printf "%.0f", s*1048576}\'; }',
                  'owned_pids=""; reserved_disk=0; ramp_ram=0; gpu_jobs=0; gpu_reserved=0; now="$(date +%s)"',
                  'own_pending=0; own_extra=0; co_here=0; co_pending=0',
                  'for own in $owned; do case "$own" in s4d-*) ;; *) exit 1;; esac; '
                  'pids="$(docker top "$own" -eo pid | awk \'NR>1 && $1 ~ /^[0-9]+$/ {print $1}\')"; '
                  'owned_pids="$owned_pids $pids"; '
                  'declared="$(docker inspect --format \'{{index .Config.Labels "s4d.disk_bytes"}}\' "$own")"; '
                  'case "$declared" in ""|*[!0-9]*) exit 1;; esac; reserved_disk=$((reserved_disk + declared)); '
                  'own_gpu="$(docker inspect --format \'{{index .Config.Labels "s4d.gpu"}}\' "$own")"; '
                  'if [ "$own_gpu" = "$gpu" ]; then gpu_jobs=$((gpu_jobs+1)); '
                  'declared="$(docker inspect --format \'{{index .Config.Labels "s4d.mem_bytes"}}\' "$own")"; '
                  'case "$declared" in ""|*[!0-9]*) exit 1;; esac; gpu_reserved=$((gpu_reserved+declared)); '
                  'used="$(gpu_used "$pids")"; '
                  'if [ "$declared" -gt "$used" ]; then own_pending=$((own_pending+declared-used)); '
                  'else own_extra=$((own_extra+used-declared)); fi; fi; '
                  'started="$(docker inspect --format \'{{.State.StartedAt}}\' "$own")"; start="$(date -d "$started" +%s)"; '
                  f'if [ "$((now-start))" -lt {spec["ramp_seconds"]} ]; then '
                  'declared="$(docker inspect --format \'{{index .Config.Labels "s4d.ram_bytes"}}\' "$own")"; '
                  'case "$declared" in ""|*[!0-9]*) exit 1;; esac; measured=0; '
                  'for pid in $pids; do if [ -r "/proc/$pid/smaps_rollup" ]; then '
                  'pss="$(awk \'$1=="Pss:" {printf "%.0f",$2*1024}\' "/proc/$pid/smaps_rollup")"; '
                  'case "$pss" in ""|*[!0-9]*) measured=0; break;; esac; measured=$((measured+pss)); '
                  'else measured=0; break; fi; done; '
                  'if [ "$declared" -gt "$measured" ]; then ramp_ram=$((ramp_ram+declared-measured)); fi; fi; done',
                  # Campaign-owned co-tenants (remote_backend.identify_cotenant; the real-world preprocessing containers,
                  # s4d.managed=false): host RAM = their RAM limit less measured anon + shmem; on the selected GPU, only when
                  # bound to it exactly (CDI request = s4d.gpu label, no other device, declared s4d.mem_bytes), their
                  # processes are known and they count as a job holding max(measured, declared) GPU memory. Any other
                  # process on the GPU stays foreign.
                  'co_ids="$(docker ps -q --no-trunc ' + cotenant_filters + ')"',
                  'for co in $co_ids; do case "$co" in ""|*[!0-9a-f]*) exit 1;; esac',
                  # one inspect per co-tenant: a container removed since the listing holds nothing (its processes, if
                  # still in $apps, stay foreign); any other inspect failure refuses; a malformed record: unidentified
                  '  if ! info="$(docker inspect --format ' + q(cotenant_format) + ' "$co" 2>/dev/null)"; then '
                  'gone="$(docker inspect --format \'{{.Id}}\' "$co" 2>&1 >/dev/null)" || true; '
                  'case "$gone" in *"No such object"*|*"No such container"*) continue;; esac; '
                  'printf \'Co-tenant inspect failed\\n\' >&2; exit 1; fi',
                  '  [ "$(printf \'%s\\n\' "$info" | grep -c \'\')" = 1 ] && '
                  'printf \'%s\\n\' "$info" | grep -Eqx ' + q(cotenant_record) + ' || continue',
                  "  IFS='|' read -r co_name role limit co_gpu declared co_devices co_request <<S4D_COTENANT",
                  '$info',
                  'S4D_COTENANT',
                  '  case "$role" in ' + "|".join(sorted(COTENANT_ROLES)) + ') ;; *) continue;; esac',
                  '  case "$co_name" in /s4d-*) ;; *) continue;; esac',
                  # an enforced Docker memory limit is required (a label bounds nothing)
                  '  case "$limit" in ""|0|*[!0-9]*) printf \'Co-tenant without a RAM limit\\n\' >&2; exit 1;; esac',
                  '  stat="/sys/fs/cgroup/system.slice/docker-$co.scope/memory.stat"; used=0',
                  '  if [ -r "$stat" ]; then '
                  'used="$(awk \'$1=="anon" || $1=="shmem" {s+=$2} END {printf "%.0f", s}\' "$stat")"; fi',
                  '  case "$used" in ""|*[!0-9]*) used=0;; esac',
                  '  if [ "$limit" -gt "$used" ]; then ramp_ram=$((ramp_ram+limit-used)); fi',
                  '  test "$co_gpu" = "$gpu" && test "$co_devices" = 0 || continue',
                  '  test "$co_request" = "cdi,0,0,0,nvidia.com/gpu=$gpu;/" || continue',
                  '  case "$declared" in ""|0|*[!0-9]*) continue;; esac',
                  '  pids="$(docker top "$co" -eo pid | awk \'NR>1 && $1 ~ /^[0-9]+$/ {print $1}\')"',
                  '  used="$(gpu_used "$pids")"',
                  '  owned_pids="$owned_pids $pids"; co_here=$((co_here+1)); gpu_jobs=$((gpu_jobs+1))',
                  '  if [ "$used" -gt "$declared" ]; then gpu_reserved=$((gpu_reserved+used)); '
                  'else gpu_reserved=$((gpu_reserved+declared)); co_pending=$((co_pending+declared-used)); fi',
                  'done',
                  'owned_pids="$(printf \'%s\\n\' "$owned_pids" | tr \'\\n\' \' \')"',
                  'foreign="$(printf \'%s\\n\' "$apps" | awk -F, -v gpu="$gpu" '
                  '\'{gsub(/ /,"",$1); gsub(/ /,"",$2); if ($1==gpu) print $2}\')"',
                  'for pid in $foreign $graphics; do case "$pid" in ""|*[!0-9]*) exit 1;; esac; '
                  'case " $owned_pids " in *" $pid "*) ;; *) '
                  'printf \'Foreign process on selected GPU\\n\' >&2; exit 1;; esac; done',
                  'disk="$(df -B1 --output=size,avail -- "$root" | awk \'NR==2 {print $1, $2}\')"; set -- $disk; '
                  'test "$#" -eq 2; total="$1"; free="$2"',
                  'floor="$(awk -v total="$total" -v fraction=' + q(str(spec["disk_floor_fraction"])) +
                  ' -v minimum=' + q(str(spec["floor_bytes"])) + ' \'BEGIN {x=total*fraction; x=int(x)+(x>int(x)); '
                  'if(x<minimum)x=minimum; printf "%.0f",x}\')"',
                  f'test "$((free-reserved_disk-{spec["disk_bytes"]}))" -ge "$floor"',
                  'available="$(awk \'$1=="MemAvailable:" {printf "%.0f", $2*1024}\' /proc/meminfo)"; test -n "$available"',
                  f'test "$((available-ramp_ram-{spec["ram_bytes"]}))" -ge {spec["reserve_bytes"]}',
                  f'test "$gpu_jobs" -lt {spec["max_jobs"]}',
                  # Keep admission unchanged without co-tenants; with one, charge max(measured, declared) for own jobs.
                  'if [ "$co_here" -gt 0 ]; then gpu_reserved=$((gpu_reserved+own_extra)); fi',
                  f'test "$((gpu_reserved+{spec["gpu_bytes"]}))" -le {spec["gpu_cap_bytes"]}',
                  # with co-tenants on the GPU: the measured free memory less every known tenant's declared but not yet
                  # allocated reservation must hold this job plus the margin (scheduler.choose_gpu applies the same rule)
                  # a MIG instance: the measured free framebuffer is checked for every launch (1g.24gb leaves no slack)
                  ('if true; then ' if spec["gpu"] in REMOTE_MIG_DEVICES else 'if [ "$co_here" -gt 0 ]; then ')
                  + gpu_free_command(spec["gpu"]) +
                  'case "$gpu_free" in ""|*[!0-9]*) exit 1;; esac; '
                  f'test "$((gpu_free-co_pending-own_pending-{math.ceil(COTENANT_GPU_MARGIN_GB * 1024**3)}))" '
                  f'-ge {spec["gpu_bytes"]} || {{ printf \'Co-tenant GPU memory\\n\' >&2; exit 1; }}; fi']
    if purpose is not None:
        output = verification_output(purpose, spec["id"], spec["attempt"])
        lines += ["bounded " + q(output), "mkdir -p -- " + q(str(PurePosixPath(output).parent))]
    for directory in (f"{root}/runs", f"{root}/outputs", cache, f"{cache}/home", f"{cache}/tmp"):
        lines.append("bounded " + q(directory))
        lines.append("mkdir -p -- " + q(directory))
    lines += ["cid=$(" + shlex.join(docker) + ")", 'test "${#cid}" -eq 64',
              'case "$cid" in *[!0-9a-f]*) exit 1;; esac',
              'printf \'{"container_id":"%s","container":"%s"}\\n\' "$cid" "$container"']
    return "\n".join(lines)


# Remote MIG (2026-11-11): the instance's evidence is read from ``nvidia-smi -q -i <parent>`` on the host (python3 -I,
# stdlib only): exit != 0 when the process listing is unreadable, a process lacks its GPU instance, or the device index
# no longer carries the allowlisted GPU instance; otherwise "MEM,<total MiB>,<used MiB>" and one "APP,<pid>,<MiB>" per
# process on this instance (C and G alike).
MIG_PROBE = r"""
import re, sys
gi, dev, uuid, parent = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
listing, sep, query = sys.stdin.read().partition("\n__S4D_QUERY__\n")
if not sep or listing.count("(UUID: " + uuid + ")") != 1:
    sys.exit(6)
owner = None
for line in listing.splitlines():
    if re.match(r"GPU \d+: .* \(UUID: GPU-[0-9a-f-]+\)\s*$", line):
        owner = line.rsplit("(UUID: ", 1)[1].rstrip(") ")
    elif ("(UUID: " + uuid + ")") in line:
        m = re.match(r"\s+MIG \S+\s+Device\s+(\d+): \(UUID: ", line)
        if owner != parent or not m or int(m.group(1)) != dev:
            sys.exit(6)
sec = sub = cur = proc = found = None
apps, listed, modes = [], False, {}
for line in query.splitlines():
    if not line.strip():
        continue
    ind = len(line) - len(line.lstrip())
    k, sep, v = (x.strip() for x in line.partition(":"))
    if ind == 0:
        sec = None
        continue
    if ind <= 4:
        if proc is not None:
            sys.exit(3)
        sec, sub, proc = k, None, None
        if k == "MIG Device":
            cur = {}
        if k == "Processes":
            if v not in ("", "None"):
                sys.exit(2)
            listed = True
        continue
    if sec == "MIG Mode" and k in ("Current", "Pending"):
        modes[k] = v
    if sec == "MIG Device" and cur is not None:
        if ind == 8 and not sep:
            sub = k
        elif ind == 8 and k in ("Index", "GPU Instance ID"):
            cur[k] = int(v)
        elif sub == "FB Memory Usage" and k in ("Total", "Used"):
            cur[k] = int(v.split()[0])
        if cur.get("Index") == dev:
            found = cur
    elif sec == "Processes":
        if k == "GPU instance ID":
            if proc is not None:
                sys.exit(3)
            proc = [v]
        elif k == "Process ID":
            if proc is None or len(proc) != 1:
                sys.exit(3)
            proc.append(v)
        elif k == "Used GPU Memory" and proc is not None and len(proc) == 2:
            if not (proc[0].isdigit() and proc[1].isdigit()):
                sys.exit(3)
            if int(proc[0]) == gi:
                apps.append((proc[1], (v.split() or ["N/A"])[0]))
            proc = None
if proc is not None:
    sys.exit(3)
if modes != {"Current": "Enabled", "Pending": "Enabled"}:
    sys.exit(7)
if not listed:
    sys.exit(5)
if not found or found.get("GPU Instance ID") != gi or "Total" not in found or "Used" not in found:
    sys.exit(4)
print("MEM,%d,%d" % (found["Total"], found["Used"]))
for pid, mib in apps:
    print("APP,%s,%s" % (pid, mib))
"""


def gpu_probe_lines(gpu: str) -> list[str]:
    """Shell lines that set ``apps`` ("<device>,<pid>,<MiB>" per process on the selected device), ``graphics`` (PIDs
    of graphics-only processes) and ``graphics_memory``. A physical GPU keeps the established lines byte for byte; a
    remote MIG instance is read from its parent's detailed query (the compute-apps CSV names only the parent). A
    physical GPU whose MIG mode is (being) enabled is refused earlier, by the scheduler's snapshot
    (remote_backend.parse_snapshot: resources not ready)."""
    q = shlex.quote
    if gpu in REMOTE_MIG_DEVICES:
        parent, instance, device = REMOTE_MIG_DEVICES[gpu]
        return ['parent=' + q(parent),
                'listing="$(nvidia-smi -L)"; mig_query="$(nvidia-smi -q -i "$parent")"',
                # fresh evidence: the UUID is listed exactly once, under the allowlisted parent and device index, whose
                # GPU instance is the allowlisted one; its processes are complete records (else exit != 0: refused)
                'mig_probe="$(printf \'%s\\n__S4D_QUERY__\\n%s\\n\' "$listing" "$mig_query" | python3 -I -c ' + q(MIG_PROBE)
                + f' {int(instance)} {int(device)} "$gpu" "$parent")" '
                '|| { printf \'MIG instance evidence refused\\n\' >&2; exit 1; }',
                'apps="$(printf \'%s\\n\' "$mig_probe" | awk -F, -v gpu="$gpu" \'$1=="APP" {print gpu "," $2 "," $3}\')"',
                'graphics=""', 'graphics_memory=""']
    return ['apps="$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits)"',
            'indices="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits)"',
            'selected_index="$(printf \'%s\\n\' "$indices" | awk -F, -v gpu="$gpu" '
            '\'{gsub(/ /,"",$2); if($2==gpu)print $1}\')"; test -n "$selected_index"',
            'table="$(nvidia-smi)"; graphics="$(printf \'%s\\n\' "$table" | awk -v gpu="$selected_index" '
            '\'$1=="|" && $2==gpu && $5 ~ /^[0-9]+$/ && $6 ~ /^[CG+]+$/ {print $5}\')"',
            # Include graphics-only memory; C+G processes can appear in both sources, so use the larger
            # per-PID reading before summing each container's memory.
            'graphics_memory="$(printf \'%s\\n\' "$table" | awk -v gpu="$gpu" -v idx="$selected_index" '
            '\'$1=="|" && $2==idx && $5 ~ /^[0-9]+$/ && $6 ~ /^[CG+]+$/ '
            '{m=$(NF-1); sub(/MiB$/,"",m); if(m ~ /^[0-9]+$/) print gpu "," $5 "," m}\')"']


def gpu_free_command(gpu: str) -> str:
    """Shell fragment assigning the selected device's measured free framebuffer (bytes) to ``gpu_free``."""
    if gpu in REMOTE_MIG_DEVICES:
        return ('gpu_free="$(printf \'%s\\n\' "$mig_probe" | awk -F, \'$1=="MEM" && $2 ~ /^[0-9]+$/ && $3 ~ /^[0-9]+$/ '
                '{printf "%.0f", ($2-$3)*1048576}\')"; ')
    return ('gpu_free="$(nvidia-smi --query-gpu=uuid,memory.total,memory.used --format=csv,noheader,nounits '
            '| awk -F, -v gpu="$gpu" \'{gsub(/ /,"",$1); gsub(/ /,"",$2); gsub(/ /,"",$3); '
            'if ($1==gpu && $2 ~ /^[0-9]+$/ && $3 ~ /^[0-9]+$/) printf "%.0f", ($2-$3)*1048576}\')"; ')


def recorded_repin(repo: Path, spec: dict) -> bool:
    """Codex review 3 (6): a spec carrying ``repinned_from`` launches only when the local registry holds exactly that
    release_repinned transition (job, source and destination release, scope identity, authorization)."""
    registry = Path(repo) / "experiments/registry.jsonl"
    source = spec["repinned_from"]
    if not registry.is_file():
        return False
    for line in registry.read_text().splitlines():
        if '"release_repinned"' not in line:
            continue
        event = json.loads(line)
        if (event.get("event") == "release_repinned" and event.get("id") == spec["id"]
                and event.get("from_commit") == source["commit"] and event.get("from_image_digest") == source["image_digest"]
                and event.get("commit") == spec["commit"] and event.get("image_digest") == spec["image_digest"]
                and event.get("sources_identical") is True and str(event.get("authorization", "")).strip()):
            return True
    return False


MIG_MODE_QUERY = "nvidia-smi --query-gpu=uuid,mig.mode.current,mig.mode.pending --format=csv,noheader,nounits"
PHYSICAL_MODES = {("Disabled", "Disabled"), ("[N/A]", "[N/A]")}


def mig_mode_preflight(client, gpu: str, *, wait: bool = False) -> None:
    """Independent pre-launch check (Codex #12, 2026-11-11), outside the byte-identical launch command: a physical GPU
    is launched on only while its MIG mode is current = pending = Disabled (or [N/A]); a remote MIG instance only while
    its parent is current = pending = Enabled. An unreadable or incomplete reading refuses (ValueError: nothing was
    started; the scheduler's lost-launch reconciliation applies)."""
    output = client.checked_ssh(MIG_MODE_QUERY, wait=wait, idempotent=True, timeout=120).stdout
    modes = {}
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 3 and parts[0].startswith("GPU-"):
            modes[parts[0]] = (parts[1], parts[2])
    if gpu in REMOTE_MIG_DEVICES:
        parent = REMOTE_MIG_DEVICES[gpu][0]
        if modes.get(parent) != ("Enabled", "Enabled"):
            raise ValueError(f"MIG instance {gpu}: parent {parent} MIG mode {modes.get(parent)} is not Enabled/Enabled")
    elif modes.get(gpu) not in PHYSICAL_MODES:
        raise ValueError(f"physical GPU {gpu}: MIG mode {modes.get(gpu)} (MIG enabled, pending or unreadable)")


def launch(spec_path: Path, client: RemoteClient | None = None, *, wait: bool | None = None) -> dict:
    client = client or RemoteClient()
    wait = os.environ.get("S4D_REMOTE_NO_WAIT") != "1" if wait is None else wait
    spec = validate_spec(json.loads(Path(spec_path).read_text()))
    if "repinned_from" in spec and not recorded_repin(client.main_repo, spec):
        raise ValueError("Release re-pin of this attempt is not recorded in the registry (release_repinned)")
    client.check_tunnel(wait=wait)
    mig_mode_preflight(client, spec["gpu"], wait=wait)
    artifacts, data_identity, payloads = evidence_artifacts(spec, client.main_repo)
    proof_path = f"{REMOTE_ROOT}/releases/{spec['commit']}/provenance.json"
    local_proof = client.state_dir / "proofs" / (spec["commit"] + ".json")
    client.rsync_download(proof_path, local_proof, wait=wait)
    proof = json.loads(local_proof.read_text())
    origin_commit = validate_release_proof(proof, spec["commit"])
    acceptance = f"{REMOTE_ROOT}/runtime/acceptance/{spec['commit']}-{spec['image_digest'][7:]}"
    checksums = {}
    for path, name, digest in artifacts:
        client.rsync_upload(path, f"{acceptance}/binding-{name}", wait=wait)
        checksums[f"binding-{name}"] = digest
        extracted = client.state_dir / "acceptance" / f"{spec['commit']}-{spec['image_digest'][7:]}" / name
        extracted.parent.mkdir(parents=True, exist_ok=True)
        extracted.write_text(json.dumps(payloads[name], indent=2, sort_keys=True) + "\n")
        client.rsync_upload(extracted, f"{acceptance}/{name}", wait=wait)
        checksums[name] = sha256(extracted)
    data_files = [(item["target"], item["sha256"]) for item in payloads["data.json"]["files"]]
    command = build_remote_command(spec, data_identity, origin_commit, sha256(local_proof), checksums, data_files,
                                   origin_ref=proof["origin_ref"])
    try:
        result = client.checked_ssh(command, wait=wait, idempotent=False, timeout=900)
    except RemoteCommandError as exc:
        if exc.result is not None and exc.result.returncode == GUARD_REFUSED_EXIT:
            try:
                proof = validate_guard_refusal(json.loads(exc.result.stdout), spec['container'], spec['gpu'])
            except (ValueError, KeyError, TypeError):
                raise RemoteUnreachable('Remote refusal evidence is ambiguous; reconcile the deterministic container') from None
            raise AdmissionRefused(proof) from None
        raise
    try:
        launched = json.loads(result.stdout)
    except (ValueError, TypeError):
        raise RemoteUnreachable("Remote launch response is ambiguous; reconcile the deterministic container") from None
    if (not isinstance(launched, dict) or launched.get("container") != spec["container"]
            or not isinstance(launched.get("container_id"), str) or not HEX64.fullmatch(launched["container_id"])):
        raise RemoteUnreachable("Remote launch identity is ambiguous; reconcile the deterministic container")
    return launched


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(launch(args.spec), sort_keys=True))
        return 0
    except AdmissionRefused as exc:
        print(json.dumps(exc.proof, sort_keys=True))
        return GUARD_REFUSED_EXIT
    except RemoteUnreachable as exc:
        print(str(exc), file=sys.stderr)
        return exc.exit_code
    except (ValueError, OSError, RemoteCommandError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
