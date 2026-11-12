from __future__ import annotations

import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from ops.resources.gpu_guard import APPROVED_HOST_GPUS
from ops.remote.access import REMOTE_ROOT, RemoteCommandError, RemoteUnreachable, sha256
from ops.containers.launch import (
    RAM_RESERVE_BYTES,
    SHM_SIZE_BYTES,
    REALWORLD_READ_ROOTS,
    _bound_path,
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
                   "s4d.host": "remote", "s4d.commit": COMMIT, "s4d.image_digest": IMAGE,
                   "s4d.project": "splatter4d", "s4d.owner": "kaist-sunho"},
    }


@pytest.mark.parametrize("script", ["train_ppo.py", "train_dreamer.py", "train_flashsac.py"])
def test_retired_trainer_specs_are_refused_even_with_renamed_ids(script):
    spec = job_spec()
    spec.update(script="scripts/" + script, args=["--run-dir", "runs/" + spec["id"]])
    with pytest.raises(ValueError, match="Launch refused: .*retired on 2026-11-09"):
        validate_spec(spec)


@pytest.mark.parametrize("prefix", ["ds-ppo-", "ds-dreamer-", "ds-flashsac-"])
@pytest.mark.parametrize("target", ["train-job", "run-dir", "absolute-run-dir", "deps"])
def test_retired_evaluator_specs_are_refused(prefix, target):
    spec = job_spec()
    run_id = prefix + "cnn-hammer-s2000"
    spec.update(script="scripts/eval_rl.py", args=["--output-dir", "runs/" + spec["id"]])
    if target == "train-job":
        spec["args"] += ["--train-job=" + run_id]
    elif target == "run-dir":
        spec["args"] += ["--run-dir", "runs/" + run_id]
    elif target == "absolute-run-dir":
        spec["args"] += ["--run-dir=/workspace/splatter4d/runs/" + run_id]
    else:
        spec["deps"] = [run_id]
    with pytest.raises(ValueError, match="Launch refused: .*retired on 2026-11-09"):
        validate_spec(spec)


@pytest.mark.parametrize("script", ["scripts/train_rl.py", "scripts/train_sac.py", "scripts/eval_rl.py"])
def test_supported_rl_specs_keep_their_validation(script):
    spec = job_spec()
    spec.update(script=script, args=["--output-dir" if script == "scripts/eval_rl.py" else "--run-dir",
                                   "runs/" + spec["id"]])
    if script == "scripts/eval_rl.py":
        spec["args"] += ["--run-dir=runs/ds-sac-cnn-hammer-s2000", "--train-job", "ds-sac-cnn-hammer-s2000"]
    assert validate_spec(spec)["script"] == script
    spec["args"][1] = "runs/other"
    with pytest.raises(ValueError, match="under runs/<job_id>"):
        validate_spec(spec)


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
            "origin_branch": "main", "origin_ref": "refs/remotes/origin/main",
            "origin_url": "https://github.com/sunho001215/splatter_vae.git",
            "verified_ancestor": True, "bundle_sha256": "f" * 64, "verified_at": "2026-10-10"}


# A release synced before the 2026-11-09 branch rename records the legacy branch and ref.
LEGACY = {"origin_branch": "splatter4d", "origin_ref": "refs/remotes/origin/splatter4d"}


@pytest.mark.parametrize("key,value", [
    ("root", "/home/compu"), ("gpu", APPROVED_HOST_GPUS["local"][0]), ("attempt", True),
    ("container", "foreign-container"), ("commit", "main"), ("image_digest", "s4d-runtime:latest"),
    ("script", "../scripts/train.py"), ("script", ""), ("script", "/scripts/train.py"),
    ("args", ["bad\nargument"]), ("disk_floor_fraction", 0.14), ("ram_gb", 0), ("disk_gb", float("nan")),
    ("oom_score_adj", -1), ("labels", {"s4d.managed": "true"}), ("host_ram_reserve_gb", 119),
    ("host_ram_ramp_minutes", 9), ("max_jobs", 21), ("max_mem_gb", 91), ("mem_gb", 100),
    ("required_results", []), ("required_results", ["exit.json"]), ("required_results", ["../metrics.jsonl"]),
    ("gpu_backend", "auto"), ("gpu_backend", "gpus"), ("gpu_backend", None),
    ("cpu_threads", 0), ("cpu_threads", 65), ("cpu_threads", True), ("cpu_threads", 8.0),
])
def test_rejects_unsafe_remote_launch_spec(key, value):
    spec = job_spec()
    spec[key] = value
    with pytest.raises(ValueError):
        validate_spec(spec)


@pytest.mark.parametrize("script", ["train_snerl.py", "train_dynamo.py", "train_mvp3d.py"])
@pytest.mark.parametrize("args", [
    ["--name", "other", "--output-root", "runs"],
    ["--name", "m3d-hammer-seed0"],
    ["--name", "m3d-hammer-seed0", "--output-root", "runs/pretrain"],
    ["--name", "m3d-hammer-seed0", "--output-root", "runs/../runs"],
])
def test_new_baseline_artifacts_cannot_escape_the_synced_run_directory(script, args):
    spec = job_spec()
    spec.update(script=f"scripts/{script}", args=args)
    with pytest.raises(ValueError):
        validate_spec(spec)


@pytest.mark.parametrize("script", ["train_snerl.py", "train_dynamo.py", "train_mvp3d.py"])
def test_new_baseline_launch_spec_preserves_valid_output_and_numeric_arguments(script):
    spec = job_spec()
    spec["script"] = f"scripts/{script}"
    assert validate_spec(spec)["args"] == spec["args"]


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
    {"verified_ancestor": False}, {"origin_branch": "feature-x"}, {"origin_url": "https://example.invalid/repo"},
    {"commit": "0" * 40}, {"verified_origin_commit": "0" * 40}, {"bundle_sha256": ""},
    {"origin_branch": "feature-x", "origin_ref": "refs/remotes/origin/feature-x"}, {"origin_ref": None},
    {"origin_branch": "splatter4d"}, {"origin_ref": "refs/remotes/origin/splatter4d"}, {"origin_ref": "refs/heads/main"},
    {**LEGACY, "verified_origin_commit": "0" * 40},
])
def test_release_proof_rejects_unpushed_or_other_code(mutation):
    proof = {**release_proof(), **mutation}
    with pytest.raises(ValueError):
        validate_release_proof(proof, COMMIT)


@pytest.mark.parametrize("recorded", [{}, LEGACY])
def test_release_proof_accepts_main_and_legacy_branch_releases(recorded):
    assert validate_release_proof({**release_proof(), **recorded}, COMMIT) == "e" * 40


def test_main_release_command_differs_from_legacy_only_in_the_origin_ref():
    def command(**kwargs):
        return build_remote_command(job_spec(), DATA_ID, "e" * 40, "f" * 64, {"tests.json": "1" * 64},
                                    [(REMOTE_ROOT + "/data/hammer.hdf5", "d" * 64)], **kwargs)

    current, legacy = command(), command(origin_ref=LEGACY["origin_ref"])
    assert 'test "$(git -C "$code" rev-parse refs/remotes/origin/main)" = ' + "e" * 40 in current.splitlines()
    assert current == legacy.replace("rev-parse refs/remotes/origin/splatter4d)", "rev-parse refs/remotes/origin/main)")
    for ref in ("refs/remotes/origin/feature-x", "refs/heads/main", "origin/main", "refs/remotes/origin/main; id"):
        with pytest.raises(ValueError, match="launch provenance"):
            command(origin_ref=ref)


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
    assert docker_args.count("WANDB_MODE=offline") == 1
    assert docker_args[docker_args.index("WANDB_MODE=offline") - 1] == "--env"
    threads = str(spec.get("cpu_threads", 8))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        setting = f"{name}={threads}"
        assert docker_args.count(setting) == 1 and docker_args[docker_args.index(setting) - 1] == "--env"
    assert f"src={REMOTE_ROOT}/releases/{COMMIT}/code,dst=/workspace/splatter4d,readonly" in command
    assert "type=volume" not in command and "--volume" not in command
    assert str(RAM_RESERVE_BYTES) in command and "reserved_disk" in command and "ramp_ram" in command
    assert command.index("if existing; then exit 0; fi") < command.index("nvidia-smi --query-compute-apps")
    assert "git -C \"$code\" status --porcelain --untracked-files=all" in command
    assert "flock -x 9" in command and "native" not in spec["args"]


def without_launch_hold(command: str) -> str:
    """The command minus the one line added on purpose on 2026-11-11 (the durable remote launch hold, checked inside the
    admission lock): every other byte stays pinned by the golden hashes below."""
    from ops.remote import launch_hold

    line = "\n" + launch_hold.check_shell(REMOTE_ROOT)
    assert command.count(line) == 1
    return command.replace(line, "")


@pytest.mark.parametrize("gpu_backend,script,original_sha256", [
    # Full command fingerprints captured from e9338cb before adding real-world mounts (including all shell guards).
    ("cdi", "train.py", "5da26a0e5191d58318971603400e1f2f53953a632f6496daf4d01076dbcefcfb"),
    ("cdi", "train_reviwo.py", "6d074b20f7103cf18bdbcdb4d2274e00efe025583c09272428d55b0d0a566b23"),
    ("cdi", "train_dynamo.py", "27c9eabac474ba58de42875069eaf87f94f867e56e5c34373ba9a275688f9af9"),
    ("legacy", "train.py", "2aacdb661f56e7b306bc005153cfc4aeea0eb3fc577a0bab995bcc45706d64be"),
    ("legacy", "train_reviwo.py", "96024f15d4349ef6d4787151b1f82dbaa1ca26e10359006391777f0f853c4c95"),
    ("legacy", "train_dynamo.py", "97d5bad8a907f478c8682e2220f2a684ae2683701817165fd074e4cf9678bbc7"),
])
def test_metaworld_command_is_byte_identical_to_before_realworld_mounts(gpu_backend, script, original_sha256):
    spec = job_spec()
    spec.update(role="metaworld", gpu_backend=gpu_backend, script="scripts/" + script)
    spec["args"] += ["--config", "configs/metaworld/tasks/hammer.yaml"]
    command = build_remote_command(spec, DATA_ID, "e" * 40, "f" * 64, {"tests.json": "1" * 64},
                                   [(REMOTE_ROOT + "/data/metaworld/splatter4d_v1/hammer.hdf5", "d" * 64)],
                                   origin_ref=LEGACY["origin_ref"])  # a legacy-branch release: byte-identical
    assert hashlib.sha256(without_launch_hold(command).encode()).hexdigest() == original_sha256


@pytest.mark.parametrize("gpu_backend,script,original_sha256", [
    # Full commands captured from 447be17 before sharing the real-world mount rule with reconciliation; re-pinned
    # 2026-11-12 (user) when real-world launches stopped checking, checksum-verifying and mounting the Meta-World
    # development data (test_metaworld_data_is_required_only_by_metaworld_jobs pins that difference).
    ("cdi", "train.py", "4cb17a27d4ecebf798bbabaeb7c358ed28a1ce2d98491f5179d6b768976ca65e"),
    ("cdi", "train_reviwo.py", "a9028e4d61c938a3eff238bde4cfb4aabbe11008f6ed23bceb6ab82b2a172b7b"),
    ("cdi", "train_dynamo.py", "62e70f959de669b71819d21fea50db2d532d7f2d6235eff6f698fa8e8f42f9b3"),
    ("cdi", "realworld_fql_dataset.py", "ece1419c68594a0f3222c13ab2025ef7f99d1a912f286bc47f536c32dd7b846e"),
    ("cdi", "train_realworld_fql.py", "8d43c5e7083c1a11040780ad78c9acabb21e69348b0f7ef270e7f3bbe4c0c10a"),
    ("cdi", "realworld_preprocess.py", "cd294299d8615295dd49114ae2d669d9dc67856be344ccf94ca1f3e78c1e81b5"),
    ("legacy", "train.py", "b40e8fbcbf4afdabf62e6e39cf9716b33477623d2c08f60f1b7bbf142ffe3e62"),
    ("legacy", "train_reviwo.py", "810e8922fef072287083c5d16ff2fa5d4c2380e71d10da4d99d9c1491777fd77"),
    ("legacy", "train_dynamo.py", "555dd9eeb3192d5669cd358f34939cea5742ec93c3dbf93cfce7f3cc7d957fac"),
    ("legacy", "realworld_fql_dataset.py", "6ef3d6b8ec25c3a6f98284eb72bb09d2bf6db219d4d8b6a83eefdcac1c114798"),
    ("legacy", "train_realworld_fql.py", "9c32ad0cd3871c849da058f7ee0b392c35987485aac4b0a222b7bd9c86ef71d0"),
    ("legacy", "realworld_preprocess.py", "a50355dec4546bb38330a04fa6f446cd4cb74d13408758c9e617e212a75b5994"),
])
def test_realworld_command_is_byte_identical_before_mount_refactor(gpu_backend, script, original_sha256):
    spec = job_spec()
    spec.update(gpu_backend=gpu_backend, script="scripts/" + script)
    spec["args"] += ["--config", "configs/realworld/baselines/reviwo/drawer-open.yaml"]
    command = build_remote_command(spec, DATA_ID, "e" * 40, "f" * 64, {"tests.json": "1" * 64},
                                   [(REMOTE_ROOT + "/data/hammer.hdf5", "d" * 64)],
                                   origin_ref=LEGACY["origin_ref"])  # a legacy-branch release: byte-identical
    assert hashlib.sha256(without_launch_hold(command).encode()).hexdigest() == original_sha256


@pytest.mark.parametrize("script", ["train.py", "export_encoder.py", "train_reviwo.py", "train_dynamo.py",
                                   "train_sincro.py", "train_snerl.py", "train_mvp3d.py",
                                   "realworld_fql_dataset.py", "train_realworld_fql.py", "realworld_preprocess.py"])
@pytest.mark.parametrize("explicit_role", [False, True])
def test_realworld_roles_mount_input_roots_readonly_at_identical_paths(script, explicit_role):
    spec = job_spec()
    spec["script"] = "scripts/" + script
    spec["args"] += ["--config", "configs/realworld/baselines/reviwo/drawer-open.yaml"]
    if explicit_role:
        spec["role"] = ("realworld_fql" if "fql" in script else
                        "realworld_preprocess" if "preprocess" in script else "realworld_pretrain")
    assert validate_spec(spec)["script"] == spec["script"]  # Also the scheduler's local-side launcher validation.
    command = build_remote_command(spec, DATA_ID, "e" * 40, "f" * 64, {"tests.json": "1" * 64},
                                   [(REMOTE_ROOT + "/data/hammer.hdf5", "d" * 64)])
    assert subprocess.run(["bash", "-n"], input=command, text=True, capture_output=True).returncode == 0
    docker_line = next(line for line in command.splitlines() if line.startswith("cid=$(docker run "))
    docker_args = shlex.split(docker_line[len("cid=$("):-1])
    mounts = [docker_args[i + 1] for i, arg in enumerate(docker_args) if arg == "--mount"]
    assert REALWORLD_READ_ROOTS == (REMOTE_ROOT + "/realworld/learning_inputs", REMOTE_ROOT + "/realworld/processed",
                                    REMOTE_ROOT + "/data/real_world")
    for path in REALWORLD_READ_ROOTS:
        assert mounts.count(f"type=bind,src={path},dst={path},readonly") == 1
        assert f"bind|{path}|{path}|false" in command  # existing-attempt inspection enforces read-only, too
        assert "bounded " + shlex.quote(path) in command
    # 2026-11-12 (user): real-world jobs no longer mount, check or checksum the Meta-World development data
    assert len(mounts) == 10 and not any("/data/metaworld/splatter4d_v1" in mount for mount in mounts)
    assert "metaworld/splatter4d_v1" not in command.split("existing() {")[0] + command.split("bounded \"$runtime/launch\"")[-1]


def test_metaworld_data_is_required_only_by_metaworld_jobs():
    """User 2026-11-12: the data were removed from the remote; only a Meta-World job checks, checksum-verifies and
    mounts them (a real-world job and a typed verification row do not)."""
    data_files = [(REMOTE_ROOT + "/data/metaworld/splatter4d_v1/hammer.hdf5", "d" * 64)]

    def build(spec):
        return build_remote_command(spec, DATA_ID, "e" * 40, "f" * 64, {"tests.json": "1" * 64}, data_files)
    meta = job_spec()
    meta["args"] += ["--config", "configs/metaworld/tasks/hammer.yaml"]
    real = job_spec()
    real["args"] += ["--config", "configs/realworld/baselines/reviwo/drawer-open.yaml"]
    real["role"] = "realworld_pretrain"
    meta_command, real_command = build(meta), build(real)
    target = "dst=/home/ws/data/metaworld/splatter4d_v1"
    assert target in meta_command and "test -d " + shlex.quote(REMOTE_ROOT + "/data/metaworld/splatter4d_v1") in meta_command
    assert "sha256sum -- " + shlex.quote(data_files[0][0]) in meta_command
    assert target not in real_command and "sha256sum -- " + shlex.quote(data_files[0][0]) not in real_command
    assert "test -d " + shlex.quote(REMOTE_ROOT + "/data/metaworld/splatter4d_v1") not in real_command
    with pytest.raises(ValueError):  # a Meta-World job still requires checksum-verified data files
        build_remote_command(meta, DATA_ID, "e" * 40, "f" * 64, {"tests.json": "1" * 64}, [])
    build_remote_command(real, DATA_ID, "e" * 40, "f" * 64, {"tests.json": "1" * 64}, [])  # not required


@pytest.mark.parametrize("role,script,args", [
    (None, "train.py", []),
    (None, "realworld_viewer.py", []),  # a real-world name without a recognized role class grants no inputs
    ("metaworld", "train_reviwo.py", ["--config", "configs/metaworld/tasks/hammer.yaml"]),
    ("droid_pretrain", "train_dynamo.py", ["--config", "configs/droid/base.yaml"]),
    ("droid_preprocess", "droid_preprocess.py", []),
])
def test_non_realworld_roles_do_not_gain_input_mounts(role, script, args):
    spec = job_spec()
    spec.update(script="scripts/" + script)
    spec["args"] += args
    if role is not None:
        spec["role"] = role
    command = build_remote_command(spec, DATA_ID, "e" * 40, "f" * 64, {"tests.json": "1" * 64},
                                   [(REMOTE_ROOT + "/data/hammer.hdf5", "d" * 64)])
    assert all(path not in command for path in REALWORLD_READ_ROOTS)


@pytest.mark.parametrize("role,script,configs", [
    (None, "train_dynamo.py", ["configs/realworld/base.yaml", "configs/droid/base.yaml"]),
    ("realworld_pretrain", "train_dynamo.py", ["configs/metaworld/base.yaml"]),
    ("realworld_fql", "train_rl.py", []),
    ("realworld_typo", "train_dynamo.py", []),
])
def test_realworld_mounts_fail_closed_on_ambiguous_or_invalid_roles(role, script, configs):
    spec = job_spec()
    spec["script"] = "scripts/" + script
    spec["args"] += ["--config", *configs] if configs else []
    if role is not None:
        spec["role"] = role
    with pytest.raises(ValueError, match="remote job role"):
        build_remote_command(spec, DATA_ID, "e" * 40, "f" * 64, {"tests.json": "1" * 64},
                             [(REMOTE_ROOT + "/data/hammer.hdf5", "d" * 64)])


@pytest.mark.parametrize("root", REALWORLD_READ_ROOTS)
def test_realworld_input_paths_keep_bind_source_bounds(root):
    assert _bound_path(root) == root
    assert _bound_path(root + "/drawer-open/episode.h5") == root + "/drawer-open/episode.h5"
    for path in (root + "/../outside", root.lstrip("/"), "/home/compu/kaist/sunho-other/realworld/processed",
                 "/home/compu/kaist", REMOTE_ROOT):
        with pytest.raises(ValueError, match="authorized root"):
            _bound_path(path)


def test_remote_rl_evaluation_writes_only_into_its_own_run_directory():
    spec = job_spec()
    spec.update(id="cnn-hammer-s2000-eval", container="s4d-cnn-hammer-s2000-eval-a1", script="scripts/eval_rl.py",
                required_results=["eval.jsonl"],
                labels={**spec["labels"], "s4d.run_id": "cnn-hammer-s2000-eval"})
    for output in (None, "runs/cnn-hammer-s2000", "runs/other-eval", "runs/cnn-hammer-s2000-eval/../x"):
        spec["args"] = ["--run-dir", "runs/cnn-hammer-s2000", "--train-job", "cnn-hammer-s2000"]
        if output is not None:
            spec["args"] += ["--output-dir", output]
        with pytest.raises(ValueError):
            validate_spec(spec)
    for output in ("runs/cnn-hammer-s2000-eval", "/workspace/splatter4d/runs/cnn-hammer-s2000-eval"):
        spec["args"] = ["--run-dir", "runs/cnn-hammer-s2000", "--train-job", "cnn-hammer-s2000", "--output-dir", output]
        assert validate_spec(spec)["script"] == "scripts/eval_rl.py"


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
    ("cdi", "without_metaworld", False, 8 * 2**30),
])
@pytest.mark.parametrize("realworld", [False, True])
def test_existing_attempt_is_returned_without_second_launch_or_foreign_changes(
    tmp_path, gpu_backend, mutation, foreign_label, shm_size, realworld,
):
    spec = job_spec()
    spec["gpu_backend"] = gpu_backend
    if realworld:
        spec["role"] = "realworld_pretrain"
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
    if realworld:
        mounts += [(path, path, False) for path in REALWORLD_READ_ROOTS]
    if mutation == "writable_code":
        mounts[0] = (mounts[0][0], mounts[0][1], True)
    if mutation == "writable_data":
        mounts[-1] = (mounts[-1][0], mounts[-1][1], True)
    if mutation == "missing_mount":  # a required mount (the release proof) is missing
        mounts.pop(6)
    if mutation == "without_metaworld":  # 2026-11-12: a non-Meta-World job may lack the Meta-World data mount
        mounts.pop(7)
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
    if (mutation not in (None, "without_metaworld")) or foreign_label or shm_size != SHM_SIZE_BYTES:
        assert result.returncode != 0 and not result.stdout
    else:
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout) == {"container_id": "7" * 64, "container": spec["container"]}
    assert not any(line.startswith("run ") or line.startswith("rm ") or line.startswith("stop ")
                   for line in log.read_text().splitlines())


class FakeClient:
    def __init__(self, root: Path, response, proof: dict | None = None):
        self.main_repo = root
        self.state_dir = root / "access"
        self.response = response
        self.proof = release_proof() if proof is None else proof
        self.calls = []

    def check_tunnel(self, wait):
        self.calls.append(("check", wait))

    def rsync_download(self, remote, local, wait):
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text(json.dumps(self.proof))
        self.calls.append(("download", remote, wait))

    def rsync_upload(self, local, remote, wait):
        self.calls.append(("upload", str(local), remote, wait))

    def checked_ssh(self, command, **kwargs):
        if command.startswith("nvidia-smi --query-gpu=uuid,mig.mode"):  # the launcher's MIG-mode preflight
            modes = "".join(f"{uuid}, [N/A], [N/A]\n" for uuid in APPROVED_HOST_GPUS["remote"])
            return subprocess.CompletedProcess([], 0, modes, "")
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


@pytest.mark.parametrize("recorded", [{}, LEGACY])
def test_launch_checks_the_origin_ref_its_release_proof_records(tmp_path, recorded):
    """New releases are proven against origin/main; one proven under the legacy branch name stays launchable."""
    spec = job_spec()
    spec["evidence"] = make_evidence(tmp_path / "acceptance")
    path = tmp_path / "job.json"
    path.write_text(json.dumps(spec))
    proof = {**release_proof(), **recorded}
    expected = {"container_id": "7" * 64, "container": spec["container"]}
    client = FakeClient(tmp_path, json.dumps(expected), proof)
    assert launch(path, client, wait=False) == expected
    command = [call[1] for call in client.calls if call[0] == "ssh"][0].splitlines()
    checks = [line for line in command if "rev-parse refs/remotes/origin/" in line]
    assert checks == ['test "$(git -C "$code" rev-parse ' + proof["origin_ref"] + ')" = ' + "e" * 40]


def test_launch_refuses_a_release_proven_on_another_branch(tmp_path):
    spec = job_spec()
    spec["evidence"] = make_evidence(tmp_path / "acceptance")
    path = tmp_path / "job.json"
    path.write_text(json.dumps(spec))
    proof = {**release_proof(), "origin_branch": "feature-x", "origin_ref": "refs/remotes/origin/feature-x"}
    client = FakeClient(tmp_path, json.dumps({"container_id": "7" * 64, "container": spec["container"]}), proof)
    with pytest.raises(ValueError, match="authorized branch"):
        launch(path, client, wait=False)
    assert not any(call[0] in ("ssh", "upload") for call in client.calls)


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


def test_cpu_thread_limit_defaults_to_eight_and_follows_the_job():
    assert validate_spec(job_spec())["cpu_threads"] == 8
    spec = job_spec()
    spec["cpu_threads"] = 16
    assert validate_spec(spec)["cpu_threads"] == 16

def _rv_launch_spec(purpose="native_suite"):
    from ops.resources.gpu_guard import APPROVED_HOST_GPUS
    from ops.jobs.remote_backend import container_labels
    from ops.containers.verification import POLICIES, ROOT, canonical_arguments

    script, timeout, ram, memory, reports = POLICIES[purpose]
    gpu, job_id = next(iter(APPROVED_HOST_GPUS["remote"])), "verify-" + purpose
    commit, image = "a" * 40, "sha256:" + "b" * 64
    return {"version": 1, "host": "remote", "id": job_id, "attempt": 1, "container": f"s4d-{job_id}-a1",
            "root": ROOT, "host_config": "configs/hosts/remote.yaml", "commit": commit, "image_digest": image,
            "gpu": gpu, "gpu_backend": "cdi", "script": script,
            "args": canonical_arguments(purpose, job_id, 1, gpu),
            "verification_purpose": purpose, "campaign_eligible": False, "verification_timeout_seconds": timeout,
            "ram_gb": ram, "mem_gb": memory, "cpu_threads": 32 if purpose == "native_suite" else 8,
            "disk_gb": 1, "disk_floor_gb": 0, "disk_floor_fraction": 0.15,
            "host_ram_reserve_gb": 120, "host_ram_ramp_minutes": 10, "max_mem_gb": 90, "max_jobs": 20,
            "required_results": ["evidence/" + name for name in reports],
            "labels": {**container_labels(job_id, 1, commit, image),
                       "s4d.verification_purpose": purpose, "s4d.campaign_eligible": "false"},
            "evidence": {"commit": commit, "image_digest": image, "verification_purpose": purpose,
                         "campaign_eligible": False, "data": {"sha256": "d" * 64},
                         **({"full_suite": {"sha256": "e" * 64}} if purpose == "gpu_isolation" else {})}}


@pytest.mark.parametrize("purpose", ["native_suite", "gpu_isolation"])
@pytest.mark.parametrize("mutation", [
    {"verification_purpose": "other"}, {"verification_purpose": None}, {"campaign_eligible": True},
    {"campaign_eligible": 0}, {"script": "scripts/train.py"}, {"host": "local"},
    {"args": []}, {"required_results": ["exit.json"]}, {"verification_timeout_seconds": 0},
    {"verification_timeout_seconds": True}, {"verification_timeout_seconds": 7201},
    {"ram_gb": 1024}, {"mem_gb": 90}, {"disk_gb": 17}, {"cpu_threads": 33},
])
def test_verification_launch_spec_is_a_fixed_bounded_allowlist(purpose, mutation):
    from ops.containers.launch import validate_spec

    spec = _rv_launch_spec(purpose)
    validate_spec(spec)
    spec.update(mutation)
    with pytest.raises(ValueError):
        validate_spec(spec)


@pytest.mark.parametrize("purpose", ["native_suite", "gpu_isolation"])
def test_verification_requires_explicit_non_campaign_flag_and_reports(purpose):
    from ops.containers.launch import validate_spec
    from ops.containers.verification import validate_verification

    spec = _rv_launch_spec(purpose)
    for missing in ("campaign_eligible", "verification_timeout_seconds", "required_results"):
        incomplete = {key: value for key, value in spec.items() if key != missing}
        with pytest.raises(ValueError):
            validate_spec(incomplete)
    queued = {**spec, "args": []}
    assert validate_verification(queued, queued=True) == purpose
    queued["args"] = ["--other"]
    with pytest.raises(ValueError):
        validate_verification(queued, queued=True)
    spec["args"].append("--other")
    with pytest.raises(ValueError):
        validate_spec(spec)


@pytest.mark.parametrize("direction", ["bootstrap_as_campaign", "campaign_as_bootstrap"])
def test_launch_evidence_cannot_change_job_kind(tmp_path, monkeypatch, direction):
    import ops.containers.launch as launcher

    spec = _rv_launch_spec()
    monkeypatch.setattr(launcher, "commit_source_sha256", lambda *_: {"s4d/example.py": "c" * 64})
    if direction == "bootstrap_as_campaign":
        for key in ("verification_purpose", "campaign_eligible", "verification_timeout_seconds"):
            spec.pop(key)
    else:
        spec["evidence"] = {"commit": spec["commit"], "image_digest": spec["image_digest"],
                            "full_suite": {}, "gpu_isolation": {}, "data": {}, "cross_host": {}}
    with pytest.raises(ValueError):
        launcher.evidence_artifacts(spec, tmp_path)


@pytest.mark.parametrize("purpose", ["native_suite", "gpu_isolation"])
def test_verification_command_binds_output_limits_and_child_environment(purpose):
    import shlex

    from ops.containers.launch import build_remote_command
    from ops.containers.verification import ROOT, entry_verification

    spec = _rv_launch_spec(purpose)
    command = build_remote_command(spec, "1" * 64, "2" * 40, "3" * 64, {"binding-data.json": "d" * 64},
                                   [(ROOT + "/data/metaworld/splatter4d_v1/hammer.hdf5", "4" * 64)])
    docker_line = next(line for line in command.splitlines() if line.startswith("cid=$("))
    argv = shlex.split(docker_line[len("cid=$("):-1])
    environment = dict(argv[index + 1].split("=", 1) for index, value in enumerate(argv) if value == "--env")
    assert entry_verification(environment, spec["script"], spec["args"])["verification_purpose"] == purpose
    assert argv[argv.index("--memory") + 1] == str(spec["ram_gb"] * 2**30)
    assert argv[argv.index("--memory-swap") + 1] == str(spec["ram_gb"] * 2**30)
    assert argv[argv.index("--cpus") + 1] == str(spec["cpu_threads"])
    assert environment["S4D_TEST_EVIDENCE_DIR"].endswith(f"/{spec['id']}/a1/evidence")
    assert f"s4d.verification_purpose={purpose}" in argv and "s4d.campaign_eligible=false" in argv
    assert argv[-len(spec["args"]):] == spec["args"]
    if purpose == "gpu_isolation":
        assert spec["args"][-3:] == ["--gpu", spec["gpu"], "--host-pid-namespace"]


def test_campaign_specs_cannot_use_the_verification_namespace():
    from ops.containers.launch import build_remote_command

    spec = _rv_launch_spec()
    for key in ("verification_purpose", "campaign_eligible", "verification_timeout_seconds"):
        spec.pop(key)
    with pytest.raises(ValueError, match="reserved"):
        build_remote_command(spec, "1" * 64, "2" * 40, "3" * 64, {}, [])


@pytest.mark.parametrize("campaign_launch", [False])
def test_existing_container_of_other_kind_is_refused_before_reuse(campaign_launch):
    import shlex
    import subprocess

    from ops.containers.launch import build_remote_command
    from ops.containers.verification import ROOT

    spec = _rv_launch_spec()
    actual_labels = dict(spec["labels"])
    target = spec["labels"] if campaign_launch else actual_labels
    for key in ("s4d.verification_purpose", "s4d.campaign_eligible"):
        target.pop(key)
    if campaign_launch:
        for key in ("verification_purpose", "campaign_eligible", "verification_timeout_seconds"):
            spec.pop(key)
    command = build_remote_command(spec, "1" * 64, "2" * 40, "3" * 64, {"binding-data.json": "d" * 64},
                                   [(ROOT + "/data/hammer.hdf5", "4" * 64)])
    existing = command[command.index("existing()"):command.index("if existing; then exit 0; fi")]
    cases = []
    for key in ("s4d.managed", "s4d.run_id", "s4d.attempt", "s4d.host", "s4d.commit",
                "s4d.image_digest", "s4d.project", "s4d.owner", "s4d.verification_purpose", "s4d.campaign_eligible"):
        template = '{{index .Config.Labels "' + key + '"}}'
        cases.append(shlex.quote(template) + ") printf '%s\\n' " + shlex.quote(actual_labels.get(key, "")) + ";;")
    shell = ("set -eu\ncontainer=" + shlex.quote(spec["container"]) + "\n"
             + "docker() {\nif [ \"$1\" = ps ]; then printf '%s\\n' \"$container\"; return; fi\n"
             + 'case "$3" in\n' + "\n".join(cases)
             + "\n*'range $key'*) printf '%s\\n' verificationeligible:false;;\n"
             + "*) printf '%s\\n' unexpected-later-inspect >&2; return 99;;\nesac\n}\n"
             + existing + "\nexisting\n")
    result = subprocess.run(["bash", "-c", shell], capture_output=True, text=True)
    assert result.returncode != 0 and "unexpected-later-inspect" not in result.stderr


def test_ordinary_campaign_docker_flags_environment_and_arguments_are_preserved():
    import shlex

    from ops.containers.launch import build_remote_command
    from ops.containers.verification import ROOT

    spec = _rv_launch_spec()
    for key in ("verification_purpose", "campaign_eligible", "verification_timeout_seconds"):
        spec.pop(key)
    for key in ("s4d.verification_purpose", "s4d.campaign_eligible"):
        spec["labels"].pop(key)
    spec["cpu_threads"] = 8
    spec["id"] = "campaign-job"
    spec["container"] = f"s4d-campaign-job-a{spec['attempt']}"
    spec["labels"]["s4d.run_id"] = spec["id"]
    spec["script"] = "scripts/train.py"
    spec["args"] = ["--name", spec["id"], "--output-root", "runs"]
    spec["evidence"] = {}
    command = build_remote_command(spec, "1" * 64, "2" * 40, "3" * 64, {"tests.json": "5" * 64},
                                   [(ROOT + "/data/hammer.hdf5", "4" * 64)])
    docker_line = next(line for line in command.splitlines() if line.startswith("cid=$("))
    argv = shlex.split(docker_line[len("cid=$("):-1])
    assert "--memory" not in argv and "--memory-swap" not in argv and "--cpus" not in argv
    assert not any(value.startswith("S4D_VERIFICATION_") for value in argv)
    assert "WANDB_MODE=offline" in argv and "OMP_NUM_THREADS=8" in argv
    assert "NVIDIA_VISIBLE_DEVICES=void" in argv and argv[-len(spec["args"]):] == spec["args"]
    assert "S4D_TEST_EVIDENCE_DIR=" + ROOT + "/runtime/acceptance/" + spec["commit"] + "-" + spec["image_digest"][7:] in argv


def test_verification_child_timeout_terminates_its_process_group(monkeypatch):
    import signal
    import subprocess
    from pathlib import Path

    from scripts.remote import entry

    waits, signals = [], []
    class Child:
        pid = 123
        def wait(self, timeout=None):
            waits.append(timeout)
            if len(waits) == 1:
                raise subprocess.TimeoutExpired("verification", timeout)
            return -signal.SIGTERM
    monkeypatch.setattr(entry.subprocess, "Popen", lambda *args, **kwargs: Child())
    monkeypatch.setattr(entry.signal, "signal", lambda *args: None)
    monkeypatch.setattr(entry.os, "killpg", lambda pid, signum: signals.append((pid, signum)))
    assert entry.run_child(Path("scripts/run_tests.py"), [], None, timeout_seconds=1, environment={}) == 124
    assert waits == [1, 10] and signals == [(123, signal.SIGTERM)]


# ---- campaign-owned co-tenants in the remote launch guard (2026-11-06) ----
GIB = 2**30
CO_ID = "e" * 64


def _guard(tmp_path, *, apps, cotenant=None, inspect_fails=None, memory_stat=None, gpu_used_mib=40000,
           available_gib=1000, graphics=(), own_pids=(11,), own_declared_gib=5):
    """Runs the generated admission section (from the GPU queries to the last capacity test) against stubbed docker,
    nvidia-smi, df, /proc/meminfo and cgroup files: one scheduler job (s4d-own-a1, pid 11, 5 GiB declared, 4 GiB
    used) is running on the selected GPU; ``cotenant`` is the co-tenant's inspect line (or None)."""
    spec = job_spec()
    gpu = spec["gpu"]
    command = build_remote_command(spec, DATA_ID, "e" * 40, "f" * 64, {"tests.json": "1" * 64},
                                   [(REMOTE_ROOT + "/data/hammer.hdf5", "d" * 64)])
    lines = command.splitlines()
    head = lines[:next(i for i, line in enumerate(lines) if line.startswith('test ! -L "$root"'))]
    start = next(i for i, line in enumerate(lines) if line.startswith('apps="$(nvidia-smi'))
    end = next(i for i, line in enumerate(lines) if i > start and line.startswith("bounded "))
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(f"MemTotal: 1 kB\nMemAvailable: {available_gib * 1024 * 1024} kB\n")
    cgroup = tmp_path / "cgroup"
    if memory_stat is not None:
        (cgroup / f"docker-{CO_ID}.scope").mkdir(parents=True)
        (cgroup / f"docker-{CO_ID}.scope/memory.stat").write_text(
            f"anon {memory_stat[0]}\nfile 999999999999\nkernel 5\nshmem {memory_stat[1]}\n")
    body = "\n".join(lines[start:end]).replace("/proc/meminfo", str(meminfo)).replace(
        "/sys/fs/cgroup/system.slice", str(cgroup))
    q = shlex.quote
    own = {"s4d.disk_bytes": str(GIB), "s4d.gpu": gpu, "s4d.mem_bytes": str(own_declared_gib * GIB),
           "s4d.ram_bytes": str(GIB)}
    listing = "printf '%s\\n' " + CO_ID if cotenant is not None else "true"
    stubs = ("docker() {\ncase \"$1\" in\n"
             "ps) case \"$*\" in *s4d.managed=true*) printf '%s\\n' s4d-own-a1;; "
             "*s4d.managed=false*) " + listing + ";; "
             "*) return 97;; esac;;\n"
             "top) case \"$2\" in s4d-own-a1) printf '%s\\n' PID " + " ".join(map(str, own_pids))
             + ";; " + CO_ID + ") printf 'PID\\n777\\n778\\n';; "
             "*) return 97;; esac;;\n"
             "inspect) case \"$4\" in s4d-own-a1) case \"$3\" in "
             + " ".join(f"*{q(key)}*) printf '%s\\n' {q(value)};;" for key, value in own.items())
             + " *StartedAt*) printf '%s\\n' 2020-01-01T00:00:00Z;; *) return 97;; esac;; "
             + CO_ID + {None: f") printf '%s\\n' {q(cotenant or '')};; ",
                        "gone": ") printf 'Error: No such object: %s\\n' \"$4\" >&2; return 1;; ",
                        "daemon": ") printf 'Error response from daemon: busy\\n' >&2; return 1;; "}[inspect_fails]
             + "*) return 97;; esac;;\n*) return 97;; esac\n}\n"
             "nvidia-smi() {\ncase \"$*\" in\n"
             f"*query-compute-apps*) printf '%s\\n' {' '.join(q(row) for row in apps)};;\n"
             f"*index,uuid*) printf '%s\\n' {q('0, ' + gpu)} {q('1, ' + APPROVED_HOST_GPUS['remote'][1])};;\n"
             f"*memory.total*) printf '%s\\n' {q(f'{gpu}, 97887, {gpu_used_mib}')};;\n"
             "'') printf '%s\\n' " + " ".join(q(row) for row in graphics or ("no graphics",))
             + ";;\n*) return 97;; esac\n}\n"
             "df() { printf '1B-blocks Avail\\n%s %s\\n' 6900000000000 2200000000000; }\n")
    script = "\n".join(["set -eu", *head[1:], stubs, body, "printf 'ADMITTED\\n'"])
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True)


def _co_line(name="/s4d-rwprep-x-a1", role="rwprep", limit=48 * GIB, gpu=None, declared=34 * GIB, devices=0,
             request=None):
    gpu = gpu or APPROVED_HOST_GPUS["remote"][0]
    request = f"cdi,0,0,0,nvidia.com/gpu={gpu};/" if request is None else request
    return "|".join(map(str, (name, role, limit, gpu, declared, devices, request)))


def _apps(cotenant_mib=(15000, 15000)):
    gpu = APPROVED_HOST_GPUS["remote"][0]
    return [f"{gpu}, 11, 4096", *(f"{gpu}, {pid}, {mib}" for pid, mib in zip((777, 778), cotenant_mib))]


def test_launch_guard_admits_next_to_a_verified_cotenant_only():
    import tempfile

    other = APPROVED_HOST_GPUS["remote"][1]
    needed = (34 * 1024 - 30000) + 1024 + 4 * 1024 + 6 * 1024  # MiB
    cases = {  # name -> (keyword arguments, expected stderr marker or None for admitted)
        "no_cotenant": (dict(apps=_apps(())), None),
        "unlisted_process": (dict(apps=_apps()), "Foreign process"),
        "cotenant": (dict(apps=_apps(), cotenant=_co_line(), memory_stat=(10 * GIB, GIB)), None),
        "cpu_only_cotenant": (dict(apps=_apps(()), cotenant=_co_line(role="rwprep-maint", gpu="none", request="",
                                                                     declared="")), None),
        "role": (dict(apps=_apps(), cotenant=_co_line(role="rwprep-fetch")), "Foreign process"),
        "name": (dict(apps=_apps(), cotenant=_co_line(name="/kimi")), "Foreign process"),
        "other_gpu": (dict(apps=_apps(), cotenant=_co_line(gpu=other)), "Foreign process"),
        "label_differs": (dict(apps=_apps(), cotenant=_co_line(
            request=f"cdi,0,0,0,nvidia.com/gpu={other};/")), "Foreign process"),
        "legacy_request": (dict(apps=_apps(), cotenant=_co_line(
            request=f"nvidia,0,1,0,{APPROVED_HOST_GPUS['remote'][0]};/")), "Foreign process"),
        "devices": (dict(apps=_apps(), cotenant=_co_line(devices=1)), "Foreign process"),
        "undeclared": (dict(apps=_apps(), cotenant=_co_line(declared="")), "Foreign process"),
        "vanished": (dict(apps=_apps(()), cotenant=_co_line(), inspect_fails="gone"), None),
        "vanished_process_left": (dict(apps=_apps(), cotenant=_co_line(), inspect_fails="gone"), "Foreign process"),
        "inspect_error": (dict(apps=_apps(()), cotenant=_co_line(), inspect_fails="daemon"), "inspect failed"),
        "unbounded": (dict(apps=_apps(), cotenant=_co_line(limit=0)), "without a RAM limit"),
        # Codex review: label values carrying separators or newlines must not forge the binding fields
        "forged_newline": (dict(apps=_apps(), cotenant=_co_line(gpu=other) + "\n" + _co_line()), "Foreign process"),
        "forged_separator": (dict(apps=_apps(), cotenant=_co_line(role="rwprep|x")), "Foreign process"),
        # Codex review round 2: no octal / unbounded byte counts reach the shell arithmetic
        "octal_declared": (dict(apps=_apps(), cotenant=_co_line(declared="085899345920")), "Foreign process"),
        "zero_declared": (dict(apps=_apps(), cotenant=_co_line(declared="00")), "Foreign process"),
        "huge_limit": (dict(apps=_apps(), cotenant=_co_line(limit=10**15)), "Foreign process"),
        "octal_limit": (dict(apps=_apps(), cotenant=_co_line(limit="0123")), "Foreign process"),
        "forged_request": (dict(apps=_apps(), cotenant=_co_line(
            request=f"cdi,0,0,0,nvidia.com/gpu={APPROVED_HOST_GPUS['remote'][0]};/|")), "Foreign process"),
        "static_cap": (dict(apps=_apps(), cotenant=_co_line(declared=80 * GIB)), ""),  # 80 + 5 + 6 > 90
        "measured_cap": (dict(apps=_apps((60000, 25000)), cotenant=_co_line(), gpu_used_mib=90000), ""),
        # free memory needed: co-tenant pending 34 GiB - 30000 MiB, own job pending 1 GiB, 4 GiB margin, 6 GiB job
        "free": (dict(apps=_apps(), cotenant=_co_line(), gpu_used_mib=97887 - needed + 1), "Co-tenant GPU memory"),
        "free_exact": (dict(apps=_apps(), cotenant=_co_line(), gpu_used_mib=97887 - needed), None),
        "ram": (dict(apps=_apps(), cotenant=_co_line(limit=900 * GIB)), ""),  # 1000 - 900 - 13 < 120
        "ram_in_use": (dict(apps=_apps(), cotenant=_co_line(limit=900 * GIB), memory_stat=(880 * GIB, 10 * GIB)),
                       None),
    }
    for name, (kwargs, marker) in cases.items():
        with tempfile.TemporaryDirectory() as directory:
            result = _guard(Path(directory), **kwargs)
        if marker is None:
            assert result.returncode == 0 and result.stdout == "ADMITTED\n", (name, result.stderr)
        else:
            assert result.returncode != 0 and "ADMITTED" not in result.stdout, (name, result.stdout)
            assert marker in result.stderr, (name, result.stderr)


@pytest.mark.parametrize("cotenant", [False, True])
@pytest.mark.parametrize("graphics_only", [False, True])
def test_scheduler_over_declaration_is_charged_with_cotenants(tmp_path, cotenant, graphics_only):
    gpu = APPROVED_HOST_GPUS["remote"][0]
    # 54 GiB measured scheduler job + 34 GiB co-tenant + 6 GiB new job = 94 GiB > 90 GiB cap.
    apps = [f"{gpu}, 11, {4 * 1024 if graphics_only else 54 * 1024}"]
    if cotenant:
        apps += [f"{gpu}, 777, {34 * 1024}"]
    graphics = ["| 0 N/A N/A 12 G /usr/bin/renderer 51200MiB |"] if graphics_only else []
    result = _guard(tmp_path, apps=apps, graphics=graphics, own_pids=(11, 12),
                    cotenant=_co_line() if cotenant else None, gpu_used_mib=88 * 1024)
    assert (result.returncode == 0) is (not cotenant), result.stderr


def test_graphics_duplicate_pid_is_counted_once_and_foreign_gpu_is_ignored(tmp_path):
    gpu = APPROVED_HOST_GPUS["remote"][0]
    result = _guard(tmp_path, apps=[f"{gpu}, 11, {50 * 1024}", f"{gpu}, 777, {34 * 1024}"],
                    cotenant=_co_line(), gpu_used_mib=84 * 1024,
                    graphics=["| 0 N/A N/A 11 C+G /usr/bin/train 51200MiB |",
                              "| 1 N/A N/A 999 G /usr/bin/foreign 51200MiB |"])
    assert result.returncode == 0 and result.stdout == "ADMITTED\n", result.stderr  # 50 + 34 + 6 == 90


def test_no_cotenant_keeps_the_existing_declared_reservation_policy(tmp_path):
    gpu = APPROVED_HOST_GPUS["remote"][0]
    # Without a co-tenant, even a managed job exceeding its declaration retains the old admission behavior.
    result = _guard(tmp_path, apps=[f"{gpu}, 11, {88 * 1024}"], gpu_used_mib=88 * 1024)
    assert result.returncode == 0 and result.stdout == "ADMITTED\n", result.stderr


def test_native_depth_stage_alone_gains_the_read_only_configs_mount():
    from ops.containers.mounts import SCRIPT_READ_ROOTS, realworld_mounts

    configs = REMOTE_ROOT + "/realworld/configs"
    assert SCRIPT_READ_ROOTS == {"scripts/realworld_native_depth.py": (configs,)}
    stage = {"id": "rwcal-20261110-nativedepth-mvp3d-drawer-open", "host": "remote",
             "script": "scripts/realworld_native_depth.py", "role": "realworld_fql", "args": ["build"]}
    mounts = realworld_mounts(stage)  # the registered rows carry role realworld_fql
    assert (configs, configs, True) in mounts
    fql = {**stage, "script": "scripts/train_realworld_fql.py"}
    assert all(src != configs for src, _, _ in realworld_mounts(fql))


@pytest.mark.parametrize("script", ["./scripts/realworld_native_depth.py", "scripts//realworld_native_depth.py"])
def test_aliased_native_depth_launch_mounts_match_reconciliation(tmp_path, script):
    from types import SimpleNamespace

    from ops.jobs.remote_backend import RemoteBackend, validate_container

    queued = {**job_spec(), "script": script, "role": "realworld_fql", "args": ["build"]}
    gpu = queued["gpu"]
    host = {**queued, "limits": {gpu: {"max_jobs": 20, "max_mem_gb": 90}}}
    backend = RemoteBackend(Path(__file__).resolve().parents[1], tmp_path / "runs", SimpleNamespace(root=REMOTE_ROOT))
    _, spec = backend.prepare(queued, gpu, 1, host, {})
    assert spec["script"] == "scripts/realworld_native_depth.py"
    command = build_remote_command(spec, DATA_ID, "e" * 40, "f" * 64, {"tests.json": "1" * 64},
                                   [(REMOTE_ROOT + "/data/hammer.hdf5", "d" * 64)])
    docker_line = next(line for line in command.splitlines() if line.startswith("cid=$(docker run "))
    docker_args = shlex.split(docker_line[len("cid=$("):-1])
    mounts = []
    for i, arg in enumerate(docker_args):
        if arg == "--mount":
            fields = dict(field.split("=", 1) if "=" in field else (field, "")
                          for field in docker_args[i + 1].split(","))
            mounts.append({"Type": fields["type"], "Source": fields["src"], "Destination": fields["dst"],
                           "RW": "readonly" not in fields})
    configs = REMOTE_ROOT + "/realworld/configs"
    assert {"Type": "bind", "Source": configs, "Destination": configs, "RW": False} in mounts
    item = {"Name": "/" + spec["container"], "Id": "7" * 64, "Image": IMAGE, "Mounts": mounts,
            "Config": {"Labels": spec["labels"], "Env": ["CUDA_VISIBLE_DEVICES=" + gpu,
                       "NVIDIA_VISIBLE_DEVICES=void", "NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics"]},
            "HostConfig": {"DeviceRequests": [{"Driver": "cdi", "Count": 0,
                           "DeviceIDs": ["nvidia.com/gpu=" + gpu], "Capabilities": [], "Options": {}}],
                           "Privileged": False, "ReadonlyRootfs": True, "PidMode": "host", "ShmSize": SHM_SIZE_BYTES}}
    validate_container(item, queued["id"], {**spec, "attempts": 1, "script": queued["script"]})


@pytest.mark.parametrize("purpose,threads,accepted", [
    ("native_suite", 32, True), ("native_suite", 33, False),
    ("gpu_isolation", 8, True), ("gpu_isolation", 9, False),
])
def test_verification_cpu_budget_is_specific_to_purpose(purpose, threads, accepted):
    from ops.containers.launch import validate_spec

    spec = {**_rv_launch_spec(purpose), "cpu_threads": threads}
    if accepted:
        assert validate_spec(spec)["cpu_threads"] == threads
    else:
        with pytest.raises(ValueError):
            validate_spec(spec)


def test_release_repin_source_reaches_the_container_environment_only_when_recorded():
    """Codex #6 (2026-11-11): a re-pinned attempt carries its source release so that runtime provenance accepts
    exactly that transition; validation refuses malformed or no-op sources; other launches are unchanged."""
    spec = job_spec()
    plain = build_remote_command(spec, DATA_ID, "e" * 40, "f" * 64, {"tests.json": "1" * 64},
                                 [(REMOTE_ROOT + "/data/metaworld/splatter4d_v1/hammer.hdf5", "d" * 64)])
    assert "S4D_REPINNED_FROM" not in plain
    spec["repinned_from"] = {"commit": "9" * 40, "image_digest": IMAGE}
    validate_spec(spec)
    command = build_remote_command(spec, DATA_ID, "e" * 40, "f" * 64, {"tests.json": "1" * 64},
                                   [(REMOTE_ROOT + "/data/metaworld/splatter4d_v1/hammer.hdf5", "d" * 64)])
    assert f"S4D_REPINNED_FROM_COMMIT={'9' * 40}" in command and f"S4D_REPINNED_FROM_IMAGE={IMAGE}" in command
    for bad in ({"commit": COMMIT, "image_digest": IMAGE}, {"commit": "9" * 40}, {"commit": "x", "image_digest": IMAGE},
                {"commit": "9" * 40, "image_digest": "sha256:zz"}, "9" * 40):
        with pytest.raises(ValueError, match="re-pin"):
            validate_spec({**spec, "repinned_from": bad})


def test_mig_mode_preflight_refuses_physical_gpus_in_mig_mode_and_unreadable_modes(monkeypatch):
    from ops.containers import launch as module

    gpu = APPROVED_HOST_GPUS["remote"][0]

    class Modes:
        def __init__(self, text):
            self.text = text

        def checked_ssh(self, command, **kwargs):
            assert command == module.MIG_MODE_QUERY
            return subprocess.CompletedProcess([], 0, self.text, "")

    module.mig_mode_preflight(Modes(f"{gpu}, [N/A], [N/A]\n"), gpu)
    module.mig_mode_preflight(Modes(f"{gpu}, Disabled, Disabled\n"), gpu)
    for text in (f"{gpu}, Enabled, Enabled\n", f"{gpu}, Disabled, Enabled\n", "", "garbage"):
        with pytest.raises(ValueError, match="MIG mode"):
            module.mig_mode_preflight(Modes(text), gpu)
    mig = "MIG-00000000-0000-5000-8000-000000000000"
    monkeypatch.setattr(module, "REMOTE_MIG_DEVICES", {mig: (gpu, 3, 0)})
    module.mig_mode_preflight(Modes(f"{gpu}, Enabled, Enabled\n"), mig)
    with pytest.raises(ValueError, match="not Enabled"):
        module.mig_mode_preflight(Modes(f"{gpu}, Enabled, Disabled\n"), mig)


def test_enforced_limits_reach_docker_run_and_the_reuse_check():
    """ARCH Phase 2: --memory = --memory-swap and --cpus on a campaign attempt that carries enforce_limits; the
    lost-response reuse check verifies them; verification-only flags are refused on campaign rows."""
    spec = job_spec()
    spec.update(enforce_limits=True, cpu_threads=6)
    checked = validate_spec(spec)
    command = build_remote_command(checked, DATA_ID, "e" * 40, "f" * 64, {"tests.json": "1" * 64},
                                   [(REMOTE_ROOT + "/data/metaworld/splatter4d_v1/hammer.hdf5", "d" * 64)])
    ram = str(math.ceil(13 * 1024**3))
    assert f"--memory {ram} --memory-swap {ram} --cpus 6" in command
    assert "{{.HostConfig.NanoCpus}}" in command and "6000000000" in command
    for bad in (False, "yes", 1):
        with pytest.raises(ValueError, match="enforce_limits"):
            validate_spec({**spec, "enforce_limits": bad})




def test_mig_launch_uses_shared_slot_guard_under_existing_lock(monkeypatch):
    from ops.containers import launch as launcher
    from ops.containers.slot_guard import SOURCE

    uuid = 'MIG-00000000-0000-5000-8000-000000000000'
    parent = 'GPU-4391fcee-537f-d408-d138-10d8a3866eea'
    monkeypatch.setattr(launcher, 'APPROVED_HOST_GPUS', {**launcher.APPROVED_HOST_GPUS, 'remote': (uuid,)})
    monkeypatch.setattr(launcher, 'REMOTE_MIG_DEVICES', {uuid: (parent, 3, 0)})
    spec = job_spec()
    spec.update(gpu=uuid, max_jobs=4, max_mem_gb=21.5)
    # Spec path overrides must never reach the remote guard.
    spec['test_paths'] = {'docker': '/untrusted'}
    command = build_remote_command(spec, DATA_ID, 'e' * 40, 'f' * 64, {'tests.json': '1' * 64},
                                   [(REMOTE_ROOT + '/data/hammer.hdf5', 'd' * 64)])
    guard = 'python3 -I -c ' + shlex.quote(SOURCE)
    start = command.index(guard)
    invocation = command[start:command.index('); then', start)]
    argv = shlex.split(invocation)
    data = json.loads(argv[-1])
    assert argv[:3] == ['python3', '-I', '-c'] and argv[3] == SOURCE
    assert data['uuid'] == uuid and data['gpu_instance_id'] == 3 and data['device'] == 0
    assert data['max_jobs'] == 4 and data['idle_mib'] == 256 and 'test_paths' not in data
    assert command.index('flock -x 9') < start < command.index('cid=$(docker run')
    for obsolete in ('owned_pids=', 'co_pending=', 'gpu_used()', 'mig_probe=', 'gpu_free='):
        assert obsolete not in command
    assert subprocess.run(['bash', '-n'], input=command, text=True, capture_output=True).returncode == 0


def test_a_repinned_spec_launches_only_with_its_recorded_registry_transition(tmp_path):
    from ops.containers import launch as module

    spec = job_spec()
    spec["repinned_from"] = {"commit": "9" * 40, "image_digest": IMAGE}
    assert not module.recorded_repin(tmp_path, spec)  # no registry at all
    (tmp_path / "experiments").mkdir()
    event = {"event": "release_repinned", "id": spec["id"], "host": "remote", "from_commit": "9" * 40,
             "from_image_digest": IMAGE, "commit": COMMIT, "image_digest": IMAGE, "sources_identical": True,
             "authorization": "user: resume"}
    registry = tmp_path / "experiments/registry.jsonl"
    registry.write_text(json.dumps({**event, "commit": "8" * 40}) + "\n")  # another destination
    assert not module.recorded_repin(tmp_path, spec)
    registry.write_text(json.dumps(event) + "\n")
    assert module.recorded_repin(tmp_path, spec)
    path = tmp_path / "job.json"
    path.write_text(json.dumps(spec))
    registry.write_text("")
    with pytest.raises(ValueError, match="not recorded"):
        launch(path, FakeClient(tmp_path, "{}"), wait=False)


@pytest.mark.parametrize('valid', [True, False])
def test_launcher_maps_only_verified_final_guard_refusal(tmp_path, monkeypatch, valid):
    from ops.containers import launch as launcher
    from ops.jobs.remote_backend import AdmissionRefused
    spec = job_spec()
    spec['evidence'] = make_evidence(tmp_path / 'acceptance')
    path = tmp_path / 'job.json'
    path.write_text(json.dumps(spec))
    proof = {'version': 1, 'container': spec['container'], 'guard': {
        'admit': False, 'uuid': spec['gpu'], 'reasons': ['host RAM reserve'],
        'containers': 0, 'used_mib': 67, 'gpu_reserved_bytes': 0, 'gpu_pending_bytes': 0,
        'pending_ram_bytes': 0, 'reserved_disk_bytes': 0}}
    response = subprocess.CompletedProcess([], 20, json.dumps(proof) if valid else '{}', '')
    client = FakeClient(tmp_path, RemoteCommandError('guard refused', response))
    with pytest.raises(AdmissionRefused if valid else RemoteUnreachable):
        launcher.launch(path, client, wait=False)
    monkeypatch.setattr(launcher, 'launch', lambda _: (_ for _ in ()).throw(
        AdmissionRefused(proof) if valid else RemoteUnreachable('ambiguous')))
    monkeypatch.setattr(sys, 'argv', ['launcher', str(path)])
    assert launcher.main() == (20 if valid else 75)


@pytest.mark.parametrize('guard_code', [0, 1, 20])
def test_mig_shell_maps_only_guard_refusal_before_create(tmp_path, monkeypatch, guard_code):
    """Execute only the admission fragment with a fake Python guard; no Docker."""
    from ops.containers import launch as launcher
    from ops.containers.slot_guard import SOURCE
    uuid = 'MIG-00000000-0000-5000-8000-000000000000'
    monkeypatch.setattr(launcher, 'APPROVED_HOST_GPUS', {**launcher.APPROVED_HOST_GPUS, 'remote': (uuid,)})
    monkeypatch.setattr(launcher, 'REMOTE_MIG_DEVICES', {uuid: ('GPU-4391fcee-537f-d408-d138-10d8a3866eea', 3, 0)})
    spec = job_spec()
    spec.update(gpu=uuid, max_mem_gb=21.5, max_jobs=4)
    command = build_remote_command(spec, DATA_ID, 'e' * 40, 'f' * 64, {'tests.json': '1' * 64},
                                   [(REMOTE_ROOT + '/data/hammer.hdf5', 'd' * 64)])
    start = command.index('if verdict=$(')
    fragment = command[start:command.index('\nfi', start) + 3]
    invocation = fragment[len('if verdict=$('):fragment.index('); then')]
    verdict = {'admit': guard_code == 0, 'reasons': ['host RAM reserve'], 'uuid': uuid,
               'containers': 0, 'used_mib': 67, 'gpu_reserved_bytes': 0, 'gpu_pending_bytes': 0,
               'pending_ram_bytes': 0, 'reserved_disk_bytes': 0}
    fake = shlex.join([sys.executable, '-c', f'import sys; print({json.dumps(verdict)!r}); sys.exit({guard_code})'])
    fragment = fragment.replace(invocation, fake)
    result = subprocess.run(['bash', '-c', 'set -eu\n' + fragment + '\nprintf CREATED'],
                            capture_output=True, text=True)
    assert result.returncode == guard_code
    assert ('CREATED' in result.stdout) is (guard_code == 0)
    if guard_code == 20:
        assert json.loads(result.stdout) == {'version': 1, 'container': spec['container'], 'guard': verdict}
    elif guard_code == 1:
        assert result.stdout == ''
