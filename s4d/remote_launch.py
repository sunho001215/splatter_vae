from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import sys
from pathlib import Path, PurePosixPath

from s4d.gpu_guard import APPROVED_HOST_GPUS
from s4d.remote_access import REMOTE_ROOT, RemoteClient, RemoteCommandError, RemoteUnreachable, sha256

CONTAINER_REPO = "/workspace/splatter4d"
RAM_RESERVE_BYTES = 120 * 1024**3
SHM_SIZE_BYTES = 8 * 1024**3
ORIGIN_URL = "https://github.com/sunho001215/splatter_vae.git"
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")


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
    if script in ("scripts/train.py", "scripts/train_sincro.py", "scripts/train_reviwo.py"):
        if (_option(arguments, "--name") != job_id
                or _output_path(_option(arguments, "--output-root")) != CONTAINER_REPO + "/runs"):
            raise ValueError("Remote pretraining artifacts must be under runs/<job_id>")
    if script == "scripts/train_rl.py":
        if _output_path(_option(arguments, "--run-dir")) != f"{CONTAINER_REPO}/runs/{job_id}":
            raise ValueError("Remote RL artifacts must be under runs/<job_id>")


def validate_spec(spec: dict) -> dict:
    if not isinstance(spec, dict) or spec.get("version") != 1 or spec.get("host") != "remote":
        raise ValueError("Expected a version-1 remote job spec")
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
    validate_output_arguments(str(script), arguments, job_id)
    oom = spec.get("oom_score_adj", 500)
    if isinstance(oom, bool) or not isinstance(oom, int) or not 0 <= oom <= 1000:
        raise ValueError("Invalid OOM score")
    expected_labels = {
        "s4d.managed": "true", "s4d.run_id": job_id, "s4d.attempt": str(attempt), "s4d.host": "remote",
        "s4d.commit": commit, "s4d.image_digest": image,
    }
    if spec.get("labels") != expected_labels:
        raise ValueError("Container ownership labels differ from the job identity")
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
    return {**spec, "gpu_backend": gpu_backend,
            "ram_bytes": math.ceil(ram * 1024**3), "gpu_bytes": math.ceil(memory * 1024**3),
            "gpu_cap_bytes": math.floor(cap * 1024**3), "reserve_bytes": math.ceil(reserve * 1024**3),
            "ramp_seconds": math.ceil(ramp * 60), "disk_bytes": math.ceil(disk * 1024**3),
            "floor_bytes": math.ceil(floor * 1024**3), "oom_score_adj": oom}


def evidence_artifacts(spec: dict, repository: Path) -> tuple[list[tuple[Path, str, str]], str, dict]:
    evidence = spec["evidence"]
    if evidence.get("commit") != spec["commit"] or evidence.get("image_digest") != spec["image_digest"]:
        raise ValueError("Acceptance identity differs from the job")
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
        if "0" not in rejections or None not in rejections or not set(APPROVED_HOST_GPUS["local"]).intersection(rejections):
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
    if proof.get("origin_branch") != "splatter4d" or proof.get("origin_url") != ORIGIN_URL:
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
                         acceptance_sha256: dict[str, str], data_files: list[tuple[str, str]]) -> str:
    spec = validate_spec(spec)
    if not HEX64.fullmatch(data_identity) or not HEX40.fullmatch(origin_commit) or not HEX64.fullmatch(proof_sha256):
        raise ValueError("Invalid launch provenance")
    root, name, commit, image = REMOTE_ROOT, spec["container"], spec["commit"], spec["image_digest"]
    code = f"{root}/releases/{commit}/code"
    proof = f"{root}/releases/{commit}/provenance.json"
    runtime = f"{root}/runtime"
    cache = f"{runtime}/cache"
    acceptance = f"{runtime}/acceptance/{commit}-{image[7:]}"
    data = f"{root}/data/metaworld/splatter4d_v1"
    mounts = [(code, CONTAINER_REPO, True), (f"{root}/runs", f"{CONTAINER_REPO}/runs", False),
              (f"{root}/outputs", f"{CONTAINER_REPO}/outputs", False),
              (cache, f"{CONTAINER_REPO}/.cache", False), (runtime, runtime, False),
              (acceptance, acceptance, True), (proof, proof, True),
              (data, "/home/ws/data/metaworld/splatter4d_v1", True)]
    docker = ["docker", "run", "-d", "--name", name, "--read-only", "--pid=host", "--user", "1000:1000",
              "--oom-score-adj", str(spec["oom_score_adj"]),
              "--workdir", CONTAINER_REPO, "--init", "--shm-size", "8g"]
    if spec["gpu_backend"] == "cdi":
        docker += ["--device", f"nvidia.com/gpu={spec['gpu']}"]
    else:
        docker += ["--gpus", f"device={spec['gpu']}"]
    for key, value in spec["labels"].items():
        docker += ["--label", f"{key}={value}"]
    docker += ["--label", f"s4d.ram_bytes={spec['ram_bytes']}", "--label", f"s4d.disk_bytes={spec['disk_bytes']}",
               "--label", f"s4d.mem_bytes={spec['gpu_bytes']}", "--label", f"s4d.gpu={spec['gpu']}"]
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
    }
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
    expected_mounts = "\n".join(sorted(f"bind|{source}|{target}|{'false' if readonly else 'true'}"
                                      for source, target, readonly in mounts))
    mount_template = '{{range .Mounts}}{{printf "%s|%s|%s|%t\\n" .Type .Source .Destination .RW}}{{end}}'
    lines.append('  actual_mounts="$(docker inspect --format ' + q(mount_template) + ' "$container" | sort)" || exit 1')
    lines.append('  test "$actual_mounts" = ' + q(expected_mounts) + ' || exit 1')
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
              'if existing; then exit 0; fi']
    for path in (code, proof, data, acceptance, *(source for source, _, _ in mounts)):
        lines.append("bounded " + q(path))
    lines += ['test -d "$code/.git"; test -d ' + q(data), 'test "$(git -C "$code" rev-parse HEAD)" = "$commit"',
              'test -z "$(git -C "$code" status --porcelain --untracked-files=all)"',
              'test "$(git -C "$code" rev-parse refs/remotes/origin/splatter4d)" = ' + q(origin_commit),
              'git -C "$code" merge-base --is-ancestor "$commit" ' + q(origin_commit),
              'test "$(sha256sum -- "$proof" | cut -d\' \' -f1)" = ' + q(proof_sha256),
              "bounded " + q(f"{code}/{spec['script']}"), "test -f " + q(f"{code}/{spec['script']}"),
              'test "$(docker image inspect --format \'{{.Id}}\' "$image")" = "$image"',
              'test "$(docker image inspect --format \'{{index .Config.Labels "s4d.managed"}}\' "$image")" = true',
              'docker image inspect --format \'{{range .RepoTags}}{{println .}}{{end}}\' "$image" | grep -q \'^s4d-\'']
    for artifact, digest in acceptance_sha256.items():
        if (not re.fullmatch(r"(?:binding-)?(?:tests|data|cross_host|isolation-GPU-[0-9a-f-]+)\.json", artifact)
                or not HEX64.fullmatch(digest)):
            raise ValueError("Invalid acceptance checksum")
        lines.append("bounded " + q(f"{acceptance}/{artifact}"))
        lines.append('test "$(sha256sum -- ' + q(f"{acceptance}/{artifact}") + " | cut -d' ' -f1)\" = " + q(digest))
    if not data_files:
        raise ValueError("Dataset files must be checksum-verified before launch")
    for path, digest in data_files:
        _bound_path(path)
        if not path.startswith(root + "/data/") or not HEX64.fullmatch(digest):
            raise ValueError("Invalid data checksum")
        lines.append("bounded " + q(path))
        lines.append('test "$(sha256sum -- ' + q(path) + " | cut -d' ' -f1)\" = " + q(digest))
    lines += ['apps="$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits)"',
              'indices="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits)"',
              'selected_index="$(printf \'%s\\n\' "$indices" | awk -F, -v gpu="$gpu" '
              '\'{gsub(/ /,"",$2); if($2==gpu)print $1}\')"; test -n "$selected_index"',
              'table="$(nvidia-smi)"; graphics="$(printf \'%s\\n\' "$table" | awk -v gpu="$selected_index" '
              '\'$1=="|" && $2==gpu && $5 ~ /^[0-9]+$/ && $6 ~ /^[CG+]+$/ {print $5}\')"',
              'owned="$(docker ps --filter label=s4d.managed=true --filter label=s4d.host=remote --format \'{{.Names}}\')"',
              'owned_pids=""; reserved_disk=0; ramp_ram=0; gpu_jobs=0; gpu_reserved=0; now="$(date +%s)"',
              'for own in $owned; do case "$own" in s4d-*) ;; *) exit 1;; esac; '
              'pids="$(docker top "$own" -eo pid | awk \'NR>1 && $1 ~ /^[0-9]+$/ {print $1}\')"; '
              'owned_pids="$owned_pids $pids"; '
              'declared="$(docker inspect --format \'{{index .Config.Labels "s4d.disk_bytes"}}\' "$own")"; '
              'case "$declared" in ""|*[!0-9]*) exit 1;; esac; reserved_disk=$((reserved_disk + declared)); '
              'own_gpu="$(docker inspect --format \'{{index .Config.Labels "s4d.gpu"}}\' "$own")"; '
              'if [ "$own_gpu" = "$gpu" ]; then gpu_jobs=$((gpu_jobs+1)); '
              'declared="$(docker inspect --format \'{{index .Config.Labels "s4d.mem_bytes"}}\' "$own")"; '
              'case "$declared" in ""|*[!0-9]*) exit 1;; esac; gpu_reserved=$((gpu_reserved+declared)); fi; '
              'started="$(docker inspect --format \'{{.State.StartedAt}}\' "$own")"; start="$(date -d "$started" +%s)"; '
              f'if [ "$((now-start))" -lt {spec["ramp_seconds"]} ]; then '
              'declared="$(docker inspect --format \'{{index .Config.Labels "s4d.ram_bytes"}}\' "$own")"; '
              'case "$declared" in ""|*[!0-9]*) exit 1;; esac; measured=0; '
              'for pid in $pids; do if [ -r "/proc/$pid/smaps_rollup" ]; then '
              'pss="$(awk \'$1=="Pss:" {printf "%.0f",$2*1024}\' "/proc/$pid/smaps_rollup")"; '
              'case "$pss" in ""|*[!0-9]*) measured=0; break;; esac; measured=$((measured+pss)); '
              'else measured=0; break; fi; done; '
              'if [ "$declared" -gt "$measured" ]; then ramp_ram=$((ramp_ram+declared-measured)); fi; fi; done',
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
              f'test "$((gpu_reserved+{spec["gpu_bytes"]}))" -le {spec["gpu_cap_bytes"]}']
    for directory in (f"{root}/runs", f"{root}/outputs", cache, f"{cache}/home", f"{cache}/tmp"):
        lines.append("bounded " + q(directory))
        lines.append("mkdir -p -- " + q(directory))
    lines += ["cid=$(" + shlex.join(docker) + ")", 'test "${#cid}" -eq 64',
              'case "$cid" in *[!0-9a-f]*) exit 1;; esac',
              'printf \'{"container_id":"%s","container":"%s"}\\n\' "$cid" "$container"']
    return "\n".join(lines)


def launch(spec_path: Path, client: RemoteClient | None = None, *, wait: bool | None = None) -> dict:
    client = client or RemoteClient()
    wait = os.environ.get("S4D_REMOTE_NO_WAIT") != "1" if wait is None else wait
    spec = validate_spec(json.loads(Path(spec_path).read_text()))
    client.check_tunnel(wait=wait)
    artifacts, data_identity, payloads = evidence_artifacts(spec, client.main_repo)
    proof_path = f"{REMOTE_ROOT}/releases/{spec['commit']}/provenance.json"
    local_proof = client.state_dir / "proofs" / (spec["commit"] + ".json")
    client.rsync_download(proof_path, local_proof, wait=wait)
    origin_commit = validate_release_proof(json.loads(local_proof.read_text()), spec["commit"])
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
    command = build_remote_command(spec, data_identity, origin_commit, sha256(local_proof), checksums, data_files)
    result = client.checked_ssh(command, wait=wait, idempotent=False, timeout=900)
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
    except RemoteUnreachable as exc:
        print(str(exc), file=sys.stderr)
        return exc.exit_code
    except (ValueError, OSError, RemoteCommandError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
