from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path

import pytest

from s4d.gpu_guard import APPROVED_HOST_GPUS
from s4d.remote_access import REMOTE_ROOT, RemoteCommandError, RemoteUnreachable, sha256
from s4d.remote_launch import (
    RAM_RESERVE_BYTES,
    SHM_SIZE_BYTES,
    build_remote_command,
    evidence_artifacts,
    launch,
    validate_release_proof,
    validate_spec,
)

COMMIT = "a" * 40
IMAGE = "sha256:" + "b" * 64
DATA_ID = "c" * 64


def job_spec():
    return {
        "version": 1, "id": "m3d-hammer-seed0", "host": "remote", "attempt": 1,
        "container": "s4d-m3d-hammer-seed0-a1", "gpu": APPROVED_HOST_GPUS["remote"][0], "gpu_backend": "cdi",
        "commit": COMMIT, "image_digest": IMAGE, "root": REMOTE_ROOT,
        "host_config": "configs/hosts/remote.yaml", "script": "scripts/train.py",
        "args": ["--name", "m3d-hammer-seed0", "--output-root", "runs", "--set", "train.steps=200000"],
        "oom_score_adj": 500, "ram_gb": 13, "disk_gb": 4, "disk_floor_gb": 0,
        "disk_floor_fraction": 0.15, "evidence": {}, "mem_gb": 6, "max_jobs": 20, "max_mem_gb": 90,
        "host_ram_reserve_gb": 120, "host_ram_ramp_minutes": 10,
        "required_results": ["completion.json", "metrics.jsonl"],
        "labels": {"s4d.managed": "true", "s4d.run_id": "m3d-hammer-seed0", "s4d.attempt": "1",
                   "s4d.host": "remote", "s4d.commit": COMMIT, "s4d.image_digest": IMAGE},
    }


def make_evidence(directory: Path):
    directory.mkdir()
    fingerprints = {"s4d/example.py": "f" * 64}
    suite = {"full_suite_passed": True, "exit_code": 0, "host": "remote", "source_sha256": fingerprints,
             "counts": {"passed": 300, "error": 0, "failed": 0, "skipped": 0}}
    reports = {"full_suite": suite,
               "data": {"identity_sha256": DATA_ID, "files": [{"sha256": "d" * 64,
                         "target": REMOTE_ROOT + "/data/metaworld/splatter4d_v1/hammer.hdf5"}]},
               "cross_host": {"passed": True, "step0_identical": True, "short_run_within_noise": True,
                              "data_identity": DATA_ID, "protocol": "same config and fixed seed"}}
    evidence = {"commit": COMMIT, "image_digest": IMAGE, "gpu_isolation": {}}
    for key, report in reports.items():
        path = directory / (key + ".json")
        path.write_text(json.dumps({"commit": COMMIT, "image_digest": IMAGE, "report": report}))
        evidence[key] = {"path": str(path), "sha256": sha256(path)}
    for uuid in APPROVED_HOST_GPUS["remote"]:
        report = {"passed": True, "host": "remote", "source_sha256": fingerprints,
                  "parent_mapping": [{"uuid": uuid}], "pid_matching": {"host_pid_namespace_asserted": True},
                  "results": [{"uuid": uuid, "gpus_seen": [uuid], "passed": True,
                               "child": {"native_cuda_passed": True, "native_egl_passed": True,
                                         "mapping": [{"uuid": uuid}]}},
                              {"cuda_visible_devices": "0", "rejected": True},
                              {"cuda_visible_devices": None, "rejected": True},
                              {"cuda_visible_devices": APPROVED_HOST_GPUS["local"][0], "rejected": True}]}
        path = directory / (uuid + ".json")
        path.write_text(json.dumps({"commit": COMMIT, "image_digest": IMAGE, "report": report}))
        evidence["gpu_isolation"][uuid] = {"path": str(path), "sha256": sha256(path)}
    return evidence


def release_proof():
    return {"commit": COMMIT, "origin_commit": "e" * 40, "verified_origin_commit": "e" * 40,
            "origin_branch": "splatter4d", "origin_url": "https://github.com/sunho001215/splatter_vae.git",
            "verified_ancestor": True, "bundle_sha256": "f" * 64, "verified_at": "2026-10-10"}


@pytest.mark.parametrize("key,value", [
    ("root", "/home/compu"), ("gpu", APPROVED_HOST_GPUS["local"][0]), ("attempt", True),
    ("container", "foreign-container"), ("commit", "main"), ("image_digest", "s4d-runtime:latest"),
    ("script", "../scripts/train.py"), ("script", ""), ("script", "/scripts/train.py"),
    ("args", ["bad\nargument"]), ("disk_floor_fraction", 0.14), ("ram_gb", 0), ("disk_gb", float("nan")),
    ("oom_score_adj", -1), ("labels", {"s4d.managed": "true"}), ("host_ram_reserve_gb", 119),
    ("host_ram_ramp_minutes", 9), ("max_jobs", 21), ("max_mem_gb", 91), ("mem_gb", 100),
    ("required_results", []), ("required_results", ["exit.json"]), ("required_results", ["../metrics.jsonl"]),
    ("gpu_backend", "auto"), ("gpu_backend", "gpus"), ("gpu_backend", None),
])
def test_rejects_unsafe_remote_launch_spec(key, value):
    spec = job_spec()
    spec[key] = value
    with pytest.raises(ValueError):
        validate_spec(spec)


@pytest.mark.parametrize("script,args", [
    ("scripts/train.py", ["--name", "other", "--output-root", "runs"]),
    ("scripts/train.py", ["--name", "m3d-hammer-seed0"]),
    ("scripts/train_sincro.py", ["--name", "m3d-hammer-seed0", "--output-root", "runs/pretrain"]),
    ("scripts/train_reviwo.py", ["--name", "m3d-hammer-seed0", "--output-root", "runs/../runs"]),
    ("scripts/train.py", ["--name", "m3d-hammer-seed0", "--name", "other", "--output-root", "runs"]),
    ("scripts/train_rl.py", ["--run-dir", "runs/another"]),
    ("scripts/train_rl.py", ["--run-dir"]),
])
def test_training_artifacts_cannot_escape_the_synced_run_directory(script, args):
    spec = job_spec()
    spec.update(script=script, args=args)
    with pytest.raises(ValueError):
        validate_spec(spec)


def test_launch_spec_preserves_numeric_arguments():
    spec = job_spec()
    validated = validate_spec(spec)
    assert validated["args"] == spec["args"]
    assert validated["ram_bytes"] == 13 * 1024**3
    assert validated["disk_bytes"] == 4 * 1024**3


def test_native_acceptance_requires_all_four_gpus_and_checksum_identity(tmp_path):
    spec = job_spec()
    spec["evidence"] = make_evidence(tmp_path / "acceptance")
    artifacts, identity, payloads = evidence_artifacts(spec, tmp_path)
    assert len(artifacts) == 7 and identity == DATA_ID
    assert payloads["tests.json"]["full_suite_passed"] is True
    spec["evidence"]["gpu_isolation"].pop(APPROVED_HOST_GPUS["remote"][3])
    with pytest.raises(ValueError, match="all four"):
        evidence_artifacts(spec, tmp_path)


@pytest.mark.parametrize("report_key,mutation", [
    ("full_suite", {"counts": {"passed": 299, "error": 0, "failed": 0, "skipped": 1}}),
    ("full_suite", {"full_suite_passed": False}),
    ("cross_host", {"passed": False}),
    ("cross_host", {"data_identity": "0" * 64}),
    ("cross_host", {"step0_identical": False}),
])
def test_rejects_incomplete_or_mismatched_native_acceptance(tmp_path, report_key, mutation):
    spec = job_spec()
    spec["evidence"] = make_evidence(tmp_path / "acceptance")
    record = spec["evidence"][report_key]
    path = Path(record["path"])
    payload = json.loads(path.read_text())
    payload["report"].update(mutation)
    path.write_text(json.dumps(payload))
    record["sha256"] = sha256(path)
    with pytest.raises(ValueError):
        evidence_artifacts(spec, tmp_path)


def test_changed_acceptance_bytes_fail_closed(tmp_path):
    spec = job_spec()
    spec["evidence"] = make_evidence(tmp_path / "acceptance")
    Path(spec["evidence"]["full_suite"]["path"]).write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        evidence_artifacts(spec, tmp_path)


def test_native_isolation_cannot_be_an_operator_only_assertion(tmp_path):
    spec = job_spec()
    spec["evidence"] = make_evidence(tmp_path / "acceptance")
    record = spec["evidence"]["gpu_isolation"][APPROVED_HOST_GPUS["remote"][0]]
    path = Path(record["path"])
    payload = json.loads(path.read_text())
    payload["report"]["results"][0]["child"]["native_egl_passed"] = False
    path.write_text(json.dumps(payload))
    record["sha256"] = sha256(path)
    with pytest.raises(ValueError, match="CUDA/EGL"):
        evidence_artifacts(spec, tmp_path)


@pytest.mark.parametrize("mutation", [
    {"verified_ancestor": False}, {"origin_branch": "main"}, {"origin_url": "https://example.invalid/repo"},
    {"commit": "0" * 40}, {"verified_origin_commit": "0" * 40}, {"bundle_sha256": ""},
])
def test_release_proof_rejects_unpushed_or_other_code(mutation):
    proof = {**release_proof(), **mutation}
    with pytest.raises(ValueError):
        validate_release_proof(proof, COMMIT)


@pytest.mark.parametrize("gpu_backend", ["cdi", "legacy"])
def test_generated_shell_has_detachment_single_uuid_rooted_mounts_and_syntax(gpu_backend):
    spec = job_spec()
    spec["gpu_backend"] = gpu_backend
    command = build_remote_command(spec, DATA_ID, "e" * 40, "f" * 64, {"tests.json": "1" * 64},
                                   [(REMOTE_ROOT + "/data/metaworld/splatter4d_v1/hammer.hdf5", "d" * 64)])
    assert subprocess.run(["bash", "-n"], input=command, text=True, capture_output=True).returncode == 0
    assert "docker run -d --name s4d-m3d-hammer-seed0-a1 --read-only --pid=host --user 1000:1000" in command
    run_line = next(line for line in command.splitlines() if line.startswith("cid=$(docker run "))
    docker_args = shlex.split(run_line[len("cid=$("):-1])
    assert docker_args[docker_args.index("--shm-size") + 1] == "8g"
    assert docker_args.count("--shm-size") == 1 and SHM_SIZE_BYTES == 8 * 2**30
    assert "--ipc" not in docker_args and "--ipc=host" not in docker_args
    if gpu_backend == "cdi":
        assert docker_args[docker_args.index("--device") + 1] == "nvidia.com/gpu=" + spec["gpu"]
        assert docker_args.count("--device") == 1 and "--gpus" not in docker_args
        assert "NVIDIA_VISIBLE_DEVICES=void" in docker_args
    else:
        assert docker_args[docker_args.index("--gpus") + 1] == "device=" + spec["gpu"]
        assert docker_args.count("--gpus") == 1 and "--device" not in docker_args
        assert "NVIDIA_VISIBLE_DEVICES=void" not in docker_args
        assert "NVIDIA_VISIBLE_DEVICES=" + spec["gpu"] in docker_args
    assert "CUDA_VISIBLE_DEVICES=" + spec["gpu"] in command
    assert "NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics" in command
    assert f"src={REMOTE_ROOT}/releases/{COMMIT}/code,dst=/workspace/splatter4d,readonly" in command
    assert "type=volume" not in command and "--volume" not in command
    assert str(RAM_RESERVE_BYTES) in command and "reserved_disk" in command and "ramp_ram" in command
    assert command.index("if existing; then exit 0; fi") < command.index("nvidia-smi --query-compute-apps")
    assert "git -C \"$code\" status --porcelain --untracked-files=all" in command
    assert "flock -x 9" in command and "native" not in spec["args"]


def test_cdi_is_the_default_backend_without_legacy_fallback():
    spec = job_spec()
    spec.pop("gpu_backend")
    assert validate_spec(spec)["gpu_backend"] == "cdi"
    command = build_remote_command(spec, DATA_ID, "e" * 40, "f" * 64, {"tests.json": "1" * 64},
                                   [(REMOTE_ROOT + "/data/hammer.hdf5", "d" * 64)])
    assert "--device nvidia.com/gpu=" + spec["gpu"] in command and "--gpus" not in command
    assert "CDIDevices" not in command


def test_generated_shell_quotes_arguments_without_evaluation():
    spec = job_spec()
    spec["args"] += ["--set", "run.note=a'; touch /not-authorized; '"]
    command = build_remote_command(spec, DATA_ID, "e" * 40, "f" * 64, {"tests.json": "1" * 64},
                                   [(REMOTE_ROOT + "/data/hammer.hdf5", "d" * 64)])
    assert subprocess.run(["bash", "-n"], input=command, text=True, capture_output=True).returncode == 0
    assert "'run.note=a'\"'\"'; touch /not-authorized; '\"'\"''" in command


@pytest.mark.parametrize("gpu_backend,mutation,foreign_label,shm_size", [
    ("cdi", None, False, 8 * 2**30), ("legacy", None, False, 8 * 2**30),
    ("cdi", None, True, 8 * 2**30), ("cdi", None, False, 64 * 2**20),
    ("cdi", None, False, 4 * 2**30), ("cdi", None, False, 16 * 2**30),
    *[("cdi", mutation, False, 8 * 2**30) for mutation in (
        "wrong_uuid", "all", "numeric", "count_all", "extra_request", "extra_device", "extra_id",
        "legacy_request", "wrong_nvv", "missing_request", "extra_capabilities", "extra_options",
        "privileged", "writable_root", "pid_namespace", "writable_code", "writable_data", "missing_mount",
    )],
])
def test_existing_attempt_is_returned_without_second_launch_or_foreign_changes(
    tmp_path, gpu_backend, mutation, foreign_label, shm_size,
):
    spec = job_spec()
    spec["gpu_backend"] = gpu_backend
    acceptance = f"{REMOTE_ROOT}/runtime/acceptance/{COMMIT}-{IMAGE[7:]}"
    code = f"{REMOTE_ROOT}/releases/{COMMIT}/code"
    proof = f"{REMOTE_ROOT}/releases/{COMMIT}/provenance.json"
    runtime = f"{REMOTE_ROOT}/runtime"
    cache = runtime + "/cache"
    mounts = [(code, "/workspace/splatter4d", False), (REMOTE_ROOT + "/runs", "/workspace/splatter4d/runs", True),
              (REMOTE_ROOT + "/outputs", "/workspace/splatter4d/outputs", True),
              (cache, "/workspace/splatter4d/.cache", True), (runtime, runtime, True),
              (acceptance, acceptance, False), (proof, proof, False),
              (REMOTE_ROOT + "/data/metaworld/splatter4d_v1", "/home/ws/data/metaworld/splatter4d_v1", False)]
    if mutation == "writable_code":
        mounts[0] = (mounts[0][0], mounts[0][1], True)
    if mutation == "writable_data":
        mounts[-1] = (mounts[-1][0], mounts[-1][1], True)
    if mutation == "missing_mount":
        mounts.pop()
    mount_listing = "\n".join(f"bind|{source}|{target}|{str(writable).lower()}" for source, target, writable in mounts)
    qualified_id = "nvidia.com/gpu=" + spec["gpu"]
    request = {"Driver": "cdi", "Count": 0, "DeviceIDs": [qualified_id], "Capabilities": [], "Options": {}}
    devices, requests, nvv = [], [request], "void"
    if gpu_backend == "legacy" or mutation == "legacy_request":
        request.update(Driver="", DeviceIDs=[spec["gpu"]], Capabilities=[["gpu"]])
    if mutation == "wrong_uuid":
        request["DeviceIDs"] = ["nvidia.com/gpu=" + APPROVED_HOST_GPUS["remote"][1]]
    if mutation in ("all", "numeric"):
        request["DeviceIDs"] = ["nvidia.com/gpu=" + ("all" if mutation == "all" else "0")]
    if mutation == "count_all":
        request["Count"] = -1
    if mutation == "extra_request":
        requests.append({**request, "DeviceIDs": ["nvidia.com/gpu=" + APPROVED_HOST_GPUS["remote"][1]]})
    if mutation == "extra_device":
        devices.append({"PathOnHost": "/dev/nvidia1", "PathInContainer": "/dev/nvidia1", "CgroupPermissions": "rwm"})
    if mutation == "extra_id":
        request["DeviceIDs"].append("nvidia.com/gpu=" + APPROVED_HOST_GPUS["remote"][1])
    if mutation == "wrong_nvv":
        nvv = "all"
    if mutation == "missing_request":
        requests = []
    if mutation == "extra_capabilities":
        request["Capabilities"] = [["gpu"]]
    if mutation == "extra_options":
        request["Options"] = {"extra": "value"}
    request_listing = "".join(
        f"{item['Driver']}|{item['Count']}|{len(item['Capabilities'])}|{len(item['Options'])}|"
        + "\n".join(item["DeviceIDs"]) + "\n" for item in requests
    )
    request_template = ('{{range .HostConfig.DeviceRequests}}{{.Driver}}|{{.Count}}|'
                        '{{len .Capabilities}}|{{len .Options}}|{{range .DeviceIDs}}{{println .}}{{end}}{{end}}')
    responses = {"{{.Image}}": IMAGE, "{{.Id}}": "7" * 64,
                 "{{.HostConfig.Privileged}}": "true" if mutation == "privileged" else "false",
                 "{{.HostConfig.ReadonlyRootfs}}": "false" if mutation == "writable_root" else "true",
                 "{{.HostConfig.PidMode}}": "" if mutation == "pid_namespace" else "host",
                 "{{.HostConfig.ShmSize}}": str(shm_size), "{{len .HostConfig.DeviceRequests}}": str(len(requests)),
                 "{{len .HostConfig.Devices}}": str(len(devices)), request_template: request_listing,
                 "{{range .HostConfig.DeviceRequests}}{{range .DeviceIDs}}{{println .}}{{end}}{{end}}":
                 "\n".join(device for item in requests for device in item["DeviceIDs"]),
                 "{{range .Config.Env}}{{println .}}{{end}}":
                 "CUDA_VISIBLE_DEVICES=" + spec["gpu"] + "\nNVIDIA_VISIBLE_DEVICES=" + nvv,
                 '{{range .Mounts}}{{printf "%s|%s|%s|%t\\n" .Type .Source .Destination .RW}}{{end}}': mount_listing}
    for key, value in spec["labels"].items():
        responses['{{index .Config.Labels "' + key + '"}}'] = value
    if foreign_label:
        responses['{{index .Config.Labels "s4d.managed"}}'] = "false"
    cases = []
    for template, response in responses.items():
        args = f"inspect --format {template} {spec['container']}"
        cases.append(shlex.quote(args) + ") printf '%s\\n' " + shlex.quote(response) + ";;")
    log = tmp_path / "docker-calls.log"
    stub = tmp_path / "docker"
    stub.write_text("#!/usr/bin/env bash\nset -eu\nprintf '%s\\n' \"$*\" >> " + shlex.quote(str(log))
                    + "\ncase \"$*\" in\n" + shlex.quote("ps -a --format {{.Names}}")
                    + ") printf '%s\\n' " + shlex.quote(spec["container"]) + ";;\n"
                    + "\n".join(cases) + "\n*) exit 97;;\nesac\n")
    stub.chmod(0o700)
    command = build_remote_command(spec, DATA_ID, "e" * 40, "f" * 64, {"tests.json": "1" * 64},
                                   [(REMOTE_ROOT + "/data/hammer.hdf5", "d" * 64)])
    result = subprocess.run(["bash", "-c", command], capture_output=True, text=True,
                            env={**os.environ, "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"]})
    if mutation is not None or foreign_label or shm_size != SHM_SIZE_BYTES:
        assert result.returncode != 0 and not result.stdout
    else:
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout) == {"container_id": "7" * 64, "container": spec["container"]}
    assert not any(line.startswith("run ") or line.startswith("rm ") or line.startswith("stop ")
                   for line in log.read_text().splitlines())


class FakeClient:
    def __init__(self, root: Path, response):
        self.main_repo = root
        self.state_dir = root / "access"
        self.response = response
        self.calls = []

    def check_tunnel(self, wait):
        self.calls.append(("check", wait))

    def rsync_download(self, remote, local, wait):
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text(json.dumps(release_proof()))
        self.calls.append(("download", remote, wait))

    def rsync_upload(self, local, remote, wait):
        self.calls.append(("upload", str(local), remote, wait))

    def checked_ssh(self, command, **kwargs):
        self.calls.append(("ssh", command, kwargs))
        if isinstance(self.response, Exception):
            raise self.response
        return subprocess.CompletedProcess([], 0, self.response, "")


def test_launcher_mirrors_native_report_separately_from_provenance_binding(tmp_path):
    spec = job_spec()
    spec["evidence"] = make_evidence(tmp_path / "acceptance")
    path = tmp_path / "job.json"
    path.write_text(json.dumps(spec))
    expected = {"container_id": "7" * 64, "container": spec["container"]}
    client = FakeClient(tmp_path, json.dumps(expected))
    assert launch(path, client, wait=False) == expected
    uploads = [call for call in client.calls if call[0] == "upload"]
    assert len(uploads) == 14
    raw = [call for call in uploads if call[2].endswith("/tests.json")][0]
    binding = [call for call in uploads if call[2].endswith("/binding-tests.json")][0]
    assert json.loads(Path(raw[1]).read_text())["full_suite_passed"] is True
    assert json.loads(Path(binding[1]).read_text())["report"]["full_suite_passed"] is True
    ssh = [call for call in client.calls if call[0] == "ssh"][0]
    assert ssh[2]["idempotent"] is False and ssh[2]["wait"] is False


def test_unsupported_cdi_fails_closed_without_legacy_retry(tmp_path):
    spec = job_spec()
    spec["evidence"] = make_evidence(tmp_path / "acceptance")
    path = tmp_path / "job.json"
    path.write_text(json.dumps(spec))
    client = FakeClient(tmp_path, RemoteCommandError("CDI device injection failed: unresolved device"))
    with pytest.raises(RemoteCommandError, match="CDI device injection failed"):
        launch(path, client, wait=False)
    commands = [call[1] for call in client.calls if call[0] == "ssh"]
    assert len(commands) == 1
    assert "--device nvidia.com/gpu=" + spec["gpu"] in commands[0] and "--gpus" not in commands[0]


@pytest.mark.parametrize("response", [
    "not-json", "{}", RemoteUnreachable("response lost"),
    json.dumps({"container_id": "7" * 64, "container": "another-container"}),
])
def test_lost_or_ambiguous_launch_response_is_unknown_and_never_retried(tmp_path, response):
    spec = job_spec()
    spec["evidence"] = make_evidence(tmp_path / "acceptance")
    path = tmp_path / "job.json"
    path.write_text(json.dumps(spec))
    client = FakeClient(tmp_path, response)
    with pytest.raises(RemoteUnreachable):
        launch(path, client, wait=False)
    assert sum(call[0] == "ssh" for call in client.calls) == 1
