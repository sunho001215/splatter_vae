from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import time
import traceback
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

from s4d.gpu_guard import APPROVED_HOST_GPUS, LOCAL_MIG_PARENTS

ROOT = "/home/compu/kaist/sunho"
GIB = 2**30
# Script, maximum seconds, fixed RAM/GPU GiB, required original reports.
POLICIES = {
    "native_suite": ("scripts/run_tests.py", 7200, 32, 12, ("tests.json", "tests.log", "tests.junit.xml")),
    "gpu_isolation": ("scripts/check_gpu_isolation.py", 1200, 8, 4, ("gpu_isolation.json",)),
}
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
LOCAL_DEVICES = set(APPROVED_HOST_GPUS["local"]) | set(LOCAL_MIG_PARENTS)


def verification_output(purpose: str, job_id: str, attempt: int) -> str:
    if (purpose not in POLICIES or not isinstance(job_id, str)
            or not re.fullmatch(r"verify-[A-Za-z0-9_.-]{1,121}", job_id)):
        raise ValueError("Verification requires a verify-prefixed job identity")
    if type(attempt) is not int or attempt < 1:
        raise ValueError("Invalid verification attempt")
    return f"{ROOT}/runtime/verification/{job_id}/a{attempt}/evidence"


def canonical_arguments(purpose: str, job_id: str, attempt: int, gpu: str) -> list[str]:
    output = verification_output(purpose, job_id, attempt)
    if gpu not in APPROVED_HOST_GPUS["remote"]:
        raise ValueError("Verification requires one authorized remote GPU UUID")
    arguments = ["--output-dir", output]
    if purpose == "gpu_isolation":
        arguments += ["--gpu", gpu, "--host-pid-namespace"]
    return arguments


def verification_resources(value: Mapping, purpose: str) -> int:
    _, maximum, ram, gpu, _ = POLICIES[purpose]
    timeout = value.get("verification_timeout_seconds", maximum)
    if type(timeout) is not int or not 0 < timeout <= maximum:
        raise ValueError("Invalid bounded verification timeout")
    for key, fixed in (("ram_gb", ram), ("mem_gb", gpu)):
        if type(value.get(key)) not in (int, float) or value[key] != fixed:
            raise ValueError(f"Verification requires exactly {fixed} GiB {key}")
    disk, threads = value.get("disk_gb"), value.get("cpu_threads", 8)
    if (type(disk) not in (int, float) or not math.isfinite(disk) or not 0 < disk <= 16
            or type(threads) is not int or not 1 <= threads <= 8):
        raise ValueError("Verification disk/CPU reservations exceed their bounds")
    return timeout


def validate_verification(value: Mapping, *, queued: bool = False) -> str | None:
    if "verification_purpose" not in value:
        if value.get("campaign_eligible", True) is not True or "verification_timeout_seconds" in value:
            raise ValueError("Non-campaign execution requires an explicit verification purpose")
        if str(value.get("id", "")).startswith("verify-"):
            raise ValueError("The verify- job namespace is reserved for typed verification")
        return None
    purpose = value["verification_purpose"]
    if not isinstance(purpose, str) or purpose not in POLICIES:
        raise ValueError("Unknown remote verification purpose")
    verification_output(purpose, value.get("id"), 1 if queued else value.get("attempt"))
    if value.get("host") != "remote" or value.get("script") != POLICIES[purpose][0]:
        raise ValueError("Verification cannot execute an arbitrary script or host")
    if value.get("campaign_eligible") is not False:
        raise ValueError("Verification must explicitly declare campaign_eligible=False")
    if not queued and "verification_timeout_seconds" not in value:
        raise ValueError("Verification spec must declare its bounded timeout")
    expected = [] if queued else canonical_arguments(purpose, value["id"], value["attempt"], value.get("gpu"))
    if value.get("args", []) != expected:
        raise ValueError("Verification arguments differ from the fixed allowlist")
    if value.get("required_results") != ["evidence/" + name for name in POLICIES[purpose][4]]:
        raise ValueError("Verification must return its exact report artifacts")
    verification_resources(value, purpose)
    return purpose


def validate_report(purpose: str, report: dict, source: dict, commit: str, image: str, gpu: str) -> None:
    if not isinstance(report, dict) or not source or report.get("host") != "remote":
        raise ValueError("Verification requires an actual remote report")
    if purpose == "native_suite":
        counts = report.get("counts", {})
        if (report.get("full_suite_passed") is not True or type(report.get("exit_code")) is not int
                or report["exit_code"] != 0 or not isinstance(counts, dict)
                or type(counts.get("passed")) is not int or counts["passed"] < 1
                or any(type(counts.get(key)) is not int or counts[key] != 0 for key in ("failed", "error", "skipped"))
                or report.get("source_sha256") != source or report.get("source_sha256_before") != source
                or report.get("source_changed_during_suite") is not False
                or report.get("expected_commit", report.get("commit")) != commit
                or report.get("expected_image_digest", report.get("image_digest")) != image):
            raise ValueError("Verification full suite is incomplete, failing or stale")
        return
    if purpose != "gpu_isolation":
        raise ValueError("Unknown remote verification purpose")
    results, mapping = report.get("results"), report.get("parent_mapping")
    if (not isinstance(results, list) or not all(isinstance(item, dict) for item in results)
            or not isinstance(mapping, list) or not all(isinstance(item, dict) for item in mapping)):
        raise ValueError("Invalid verification isolation results")
    positives = [item for item in results if item.get("uuid") == gpu]
    child = positives[0].get("child", {}) if len(positives) == 1 else {}
    rejected = [item.get("cuda_visible_devices") for item in results if item.get("rejected") is True]
    child_mapping = child.get("mapping", []) if isinstance(child, dict) else []
    pid_matching = report.get("pid_matching", {})
    if (report.get("passed") is not True or report.get("source_sha256") != source
            or report.get("source_changed_during_check") is not False
            or not isinstance(pid_matching, dict) or pid_matching.get("host_pid_namespace_asserted") is not True
            or [item.get("uuid") for item in mapping] != [gpu]
            or len(positives) != 1 or positives[0].get("passed") is not True
            or positives[0].get("gpus_seen") != [gpu] or not isinstance(child, dict)
            or child.get("native_cuda_passed") is not True or child.get("native_egl_passed") is not True
            or not isinstance(child_mapping, list) or not all(isinstance(item, dict) for item in child_mapping)
            or [item.get("uuid") for item in child_mapping] != [gpu]
            or "0" not in rejected or None not in rejected or not any(device in rejected for device in LOCAL_DEVICES)):
        raise ValueError("Verification isolation report is incomplete, failing or stale")


def validate_development_data(data: dict) -> str:
    identity, files = data.get("identity_sha256"), data.get("files")
    if not isinstance(identity, str) or not HEX64.fullmatch(identity) or not isinstance(files, list):
        raise ValueError("Checksummed development data is mandatory for verification")
    base = PurePosixPath(ROOT) / "data/metaworld/splatter4d_v1"
    expected = {str(base / relative) for task in ("hammer", "pick-place")
                for relative in (f"{task}.hdf5", f"heldout_sets/{task}.hdf5",
                                 f"splits/{task}_seed0.json", f"workspace_stats/{task}.json")}
    normalized, targets = [], set()
    for item in files:
        if (not isinstance(item, dict) or not isinstance(item.get("target"), str)
                or item["target"] not in expected or item["target"] in targets
                or not isinstance(item.get("sha256"), str) or not HEX64.fullmatch(item["sha256"])
                or type(item.get("bytes")) is not int or item["bytes"] < 1):
            raise ValueError("Verification requires all eight checksummed development files")
        targets.add(item["target"])
        normalized.append({**{key: item[key] for key in ("sha256", "bytes")},
                           "target": str(PurePosixPath(item["target"]).relative_to(ROOT))})
    normalized.sort(key=lambda item: item["target"])
    if targets != expected or hashlib.sha256(json.dumps(normalized, sort_keys=True).encode()).hexdigest() != identity:
        raise ValueError("Development data identity differs from its files")
    return identity


def verification_artifacts(evidence: dict, purpose: str, repository: Path, source: dict) -> tuple[list, str, dict]:
    keys = {"schema_version", "verification_purpose", "campaign_eligible", "commit", "image_digest", "source_sha256", "data"}
    if purpose == "gpu_isolation":
        keys.add("full_suite")
    if (purpose not in POLICIES or set(evidence) - {"manifest_sha256", "data_identity"} != keys
            or type(evidence.get("schema_version")) is not int or evidence["schema_version"] != 1
            or evidence.get("verification_purpose") != purpose or evidence.get("campaign_eligible") is not False
            or not source or evidence.get("source_sha256") != source
            or not re.fullmatch(r"[0-9a-f]{40}", str(evidence.get("commit", "")))
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", str(evidence.get("image_digest", "")))):
        raise ValueError("Invalid purpose-bound verification manifest")
    records = [(evidence["data"], "data.json")]
    if purpose == "gpu_isolation":
        records.append((evidence["full_suite"], "tests.json"))
    artifacts, payloads = [], {}
    repository = repository.resolve()
    for reference, name in records:
        if (not isinstance(reference, dict) or set(reference) != {"path", "sha256"}
                or not isinstance(reference["path"], str) or not isinstance(reference["sha256"], str)
                or not HEX64.fullmatch(reference["sha256"])):
            raise ValueError("Verification artifact requires a path and checksum")
        raw = Path(reference["path"])
        path = raw if raw.is_absolute() else repository / raw
        if repository not in path.parents or path.resolve() != path or not path.is_file():
            raise ValueError("Verification artifacts must be symlink-free repository files")
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != reference["sha256"]:
            raise ValueError("Verification artifact checksum differs")
        binding = json.loads(payload)
        if (not isinstance(binding, dict) or binding.get("commit") != evidence["commit"]
                or binding.get("image_digest") != evidence["image_digest"] or not isinstance(binding.get("report"), dict)):
            raise ValueError("Verification artifact commit/image binding differs")
        artifacts.append((path, name, reference["sha256"]))
        payloads[name] = binding["report"]
    identity = validate_development_data(payloads["data.json"])
    if "data_identity" in evidence and evidence["data_identity"] != identity:
        raise ValueError("Verification data identity differs")
    if purpose == "gpu_isolation":
        validate_report("native_suite", payloads["tests.json"], source, evidence["commit"], evidence["image_digest"], "")
    return artifacts, identity, payloads


def entry_verification(env: Mapping[str, str], script: str, arguments: list[str]) -> dict | None:
    prefix = {key for key in env if key.startswith("S4D_VERIFICATION_")}
    if "S4D_VERIFICATION_PURPOSE" not in env:
        if prefix:
            raise ValueError("Incomplete verification environment")
        return None
    purpose = env["S4D_VERIFICATION_PURPOSE"]
    if purpose not in POLICIES:
        raise ValueError("Unknown remote verification purpose")
    numbers = {}
    for suffix in ("TIMEOUT_SECONDS", "RAM_BYTES", "GPU_BYTES", "DISK_BYTES", "CPU_THREADS"):
        key = "S4D_VERIFICATION_" + suffix
        if not re.fullmatch(r"[1-9][0-9]*", env.get(key, "")):
            raise ValueError("Invalid verification environment bound")
        numbers[suffix] = int(env[key])
    if not re.fullmatch(r"[1-9][0-9]*", env.get("S4D_ATTEMPT", "")):
        raise ValueError("Invalid verification attempt")
    spec = {"host": "remote", "id": env.get("S4D_JOB_ID"), "attempt": int(env["S4D_ATTEMPT"]),
            "verification_purpose": purpose, "campaign_eligible": False, "script": script, "args": arguments,
            "gpu": env.get("CUDA_VISIBLE_DEVICES"), "ram_gb": numbers["RAM_BYTES"] / GIB,
            "mem_gb": numbers["GPU_BYTES"] / GIB, "disk_gb": numbers["DISK_BYTES"] / GIB,
            "cpu_threads": numbers["CPU_THREADS"], "verification_timeout_seconds": numbers["TIMEOUT_SECONDS"],
            "required_results": ["evidence/" + name for name in POLICIES[purpose][4]]}
    validate_verification(spec)
    commit, image = env.get("S4D_EXPECTED_COMMIT", ""), env.get("S4D_IMAGE_DIGEST", "")
    if not re.fullmatch(r"[0-9a-f]{40}", commit) or not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
        raise ValueError("Verification requires immutable commit/image identity")
    if (env.get("S4D_TEST_EVIDENCE_DIR") != verification_output(purpose, spec["id"], spec["attempt"])
            or env.get("S4D_VERIFICATION_ACCEPTANCE_DIR") != f"{ROOT}/runtime/acceptance/{commit}-{image[7:]}"):
        raise ValueError("Verification directories differ from their fixed identities")
    checksums = {"S4D_VERIFICATION_DATA_SHA256"}
    if purpose == "gpu_isolation":
        checksums.add("S4D_VERIFICATION_SUITE_SHA256")
    expected = {"S4D_VERIFICATION_PURPOSE", "S4D_VERIFICATION_ACCEPTANCE_DIR", *checksums,
                *("S4D_VERIFICATION_" + suffix for suffix in numbers)}
    if prefix != expected or any(not HEX64.fullmatch(env.get(key, "")) for key in checksums):
        raise ValueError("Verification prerequisite checksums/environment differ")
    return spec


def verify_entry_prerequisites(env: Mapping[str, str], purpose: str, source: dict) -> None:
    directory = Path(env["S4D_VERIFICATION_ACCEPTANCE_DIR"])
    if directory.resolve() != directory:
        raise ValueError("Verification prerequisites must be symlink-free")
    records = [("data", "S4D_VERIFICATION_DATA_SHA256")]
    if purpose == "gpu_isolation":
        records.append(("tests", "S4D_VERIFICATION_SUITE_SHA256"))
    for name, key in records:
        path = directory / f"binding-{name}.json"
        if path.is_symlink() or not path.is_file():
            raise ValueError("Missing verification prerequisite")
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != env[key]:
            raise ValueError("Verification prerequisite checksum differs")
        binding = json.loads(payload)
        if (not isinstance(binding, dict) or binding.get("commit") != env["S4D_EXPECTED_COMMIT"]
                or binding.get("image_digest") != env["S4D_IMAGE_DIGEST"] or not isinstance(binding.get("report"), dict)):
            raise ValueError("Verification prerequisite identity differs")
        if name == "data":
            if validate_development_data(binding["report"]) != env.get("S4D_DATA_ID"):
                raise ValueError("Verification data identity differs")
        else:
            validate_report("native_suite", binding["report"], source, env["S4D_EXPECTED_COMMIT"],
                            env["S4D_IMAGE_DIGEST"], "")


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.is_symlink() or temporary.is_symlink():
        raise ValueError("Symlinked verification wrapper report")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def run_verification(repository: Path, spec: dict, run_child) -> int:
    # Keep bootstrap independent of campaign identity/acceptance helpers.
    from s4d import third_party
    from s4d.gpu_guard import enforce_allowed_gpus
    from s4d.remote_jobs import commit_source_sha256
    from s4d.remote_launch import validate_release_proof

    purpose = validate_verification(spec)
    if purpose is None:
        raise ValueError("Verification entry requires a typed purpose")
    env = os.environ
    commit, image = env["S4D_EXPECTED_COMMIT"], env["S4D_IMAGE_DIGEST"]
    def git(*arguments):
        return subprocess.run(["git", "-C", str(repository), *arguments], capture_output=True,
                              text=True, check=True, timeout=60).stdout.strip()

    proof = Path(f"{ROOT}/releases/{commit}/provenance.json")
    if env.get("S4D_RELEASE_PROOF") != str(proof) or proof.resolve() != proof:
        raise ValueError("Verification release proof path differs")
    origin = validate_release_proof(json.loads(proof.read_text()), commit)
    if git("rev-parse", "HEAD") != commit or git("status", "--porcelain", "--untracked-files=all", "--ignore-submodules=none"):
        raise ValueError("Verification release code differs from its clean commit")
    git("merge-base", "--is-ancestor", commit, origin)
    if third_party.working_tree_identity(repository) != third_party.commit_identity(repository, commit):
        raise ValueError("Verification third-party release identity differs")
    script = repository / spec["script"]
    if script.resolve() != script or not script.is_file():
        raise ValueError("Verification script must be a symlink-free release file")
    git("ls-files", "--error-unmatch", "--", spec["script"])
    source = commit_source_sha256(repository, commit)
    verify_entry_prerequisites(env, purpose, source)
    output = Path(verification_output(purpose, spec["id"], spec["attempt"]))
    run_dir = repository / "runs" / spec["id"]
    if output.resolve() != output or run_dir.resolve() != run_dir:
        raise ValueError("Verification output/run paths must be symlink-free")
    output.mkdir(parents=True, exist_ok=False)
    run_dir.mkdir(parents=True, exist_ok=True)
    identity = {"host": "remote", "job_id": spec["id"], "attempt": spec["attempt"],
                "git_commit": commit, "image_digest": image, "gpu_uuids": [spec["gpu"]],
                "verification_purpose": purpose, "campaign_eligible": False}
    _write_json(run_dir / "provenance.json", {**identity, "data_id": env["S4D_DATA_ID"],
                "script": spec["script"], "arguments": spec["args"], "source_sha256": source})
    (run_dir / "exit.json").unlink(missing_ok=True)
    (run_dir / "exit_code").unlink(missing_ok=True)
    code = 1
    if (run_dir / "console.log").is_symlink():
        raise ValueError("Symlinked verification console")
    with (run_dir / "console.log").open("a", buffering=1) as console:
        try:
            enforce_allowed_gpus()
            child_env = dict(env)
            if purpose == "native_suite":
                child_env.pop("S4D_JOB_ID", None)
            else:
                child_env["S4D_TEST_EVIDENCE_DIR"] = env["S4D_VERIFICATION_ACCEPTANCE_DIR"]
            code = run_child(script, spec["args"], console, timeout_seconds=spec["verification_timeout_seconds"],
                             environment=child_env)
            if code == 0:
                report_name = "tests.json" if purpose == "native_suite" else "gpu_isolation.json"
                validate_report(purpose, json.loads((output / report_name).read_text()), source, commit, image, spec["gpu"])
                for name in POLICIES[purpose][4]:
                    original = output / name
                    if original.is_symlink() or not original.is_file() or original.stat().st_size == 0:
                        raise ValueError("Required original verification report is missing or unsafe")
        except BaseException:
            traceback.print_exc(file=console)
            code = 1
        finally:
            _write_json(run_dir / "exit.json", {**identity, "code": code, "finished_at": time.time()})
    return code
