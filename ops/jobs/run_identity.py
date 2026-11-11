"""Run provenance and clean, pushed remote-release enforcement."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

if __name__ == "__main__" and not __package__:  # retain isolated script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ops.provenance.hashing import json_bytes  # noqa: E402
from ops.provenance.hashing import sha256_file as _sha256_file  # noqa: E402

COMMIT_RE = re.compile(r"[0-9a-f]{40}")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
IMAGE_RE = re.compile(r"sha256:[0-9a-f]{64}")
JOB_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
GENERATED_ROOTS = frozenset({"runs", "outputs", ".cache", ".pytest_tmp", ".pytest_cache", ".ruff_cache"})
# Accepted (origin_branch -> origin_ref) of a release proof (as ops/remote/access.py RELEASE_ORIGIN_REFS): the single
# branch `main` (since 2026-11-09) and the legacy `splatter4d` name earlier releases were proven under.
RELEASE_ORIGIN_REFS = {"main": "refs/remotes/origin/main", "splatter4d": "refs/remotes/origin/splatter4d"}


class RunIdentityError(RuntimeError):
    pass


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=30, check=False
    )
    if result.returncode:
        raise RunIdentityError(f"Git release verification failed: {args[0]}")
    return result.stdout.rstrip("\n")


def _required(env: Mapping[str, str], key: str, pattern: re.Pattern | None = None) -> str:
    value = env.get(key, "")
    if not value or (pattern is not None and pattern.fullmatch(value) is None):
        raise RunIdentityError(f"Missing or invalid {key}")
    return value


def _json_bytes(payload: dict | int) -> bytes:
    return json_bytes(payload)


def _atomic_json(path: Path, payload: dict | int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(_json_bytes(payload) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def verify_clean_release(repo: Path, commit: str) -> None:
    repo = Path(repo).resolve()
    if Path(_git(repo, "rev-parse", "--show-toplevel")).resolve() != repo:
        raise RunIdentityError("Remote release must be the checkout root")
    if _git(repo, "rev-parse", "HEAD") != commit:
        raise RunIdentityError("Remote HEAD differs from S4D_EXPECTED_COMMIT")
    # The generated roots are the container's bind mounts; runs/ also holds tracked campaign records that the mount
    # hides. The launcher checks the whole release clean on the host before docker run.
    mounted = [f":(exclude){root}" for root in sorted(GENERATED_ROOTS)]
    if _git(repo, "diff", "--name-only", "HEAD", "--", ".", *mounted):
        raise RunIdentityError("Remote release has tracked changes")
    if _git(repo, "diff", "--name-only", "--", ".", *mounted):
        raise RunIdentityError("Remote release has unstaged changes")
    untracked = _git(repo, "ls-files", "--others", "--exclude-standard", "-z")
    ignored = _git(repo, "ls-files", "--others", "--ignored", "--exclude-standard", "-z")
    for relative in (untracked + ignored).split("\0"):
        if relative and Path(relative).parts[0] not in GENERATED_ROOTS:
            raise RunIdentityError(f"Remote release contains untracked files: {relative}")


def verify_remote_identity(repo: Path, env: Mapping[str, str] | None = None) -> dict:
    from ops.resources.gpu_guard import load_host_config, validate_runtime_path

    repo = Path(repo).resolve()
    env = os.environ if env is None else env
    host_config = Path(_required(env, "S4D_HOST_CONFIG"))
    if not host_config.is_absolute():
        raise RunIdentityError("S4D_HOST_CONFIG must be absolute")
    selected_host = load_host_config(host_config)
    if selected_host["name"] != "remote":
        raise RunIdentityError("Remote entry requires the explicit remote host config")
    uuids = _required(env, "CUDA_VISIBLE_DEVICES").split(",")
    if any(uuid not in selected_host["gpu_uuids"] for uuid in uuids) or len(uuids) != len(set(uuids)):
        raise RunIdentityError("Remote entry requires authorized remote GPU UUIDs")
    commit = _required(env, "S4D_EXPECTED_COMMIT", COMMIT_RE)
    image_digest = _required(env, "S4D_IMAGE_DIGEST", IMAGE_RE)
    repinned_from = None
    if "S4D_REPINNED_FROM_COMMIT" in env or "S4D_REPINNED_FROM_IMAGE" in env:  # both or neither (launch.py)
        repinned_from = {"git_commit": _required(env, "S4D_REPINNED_FROM_COMMIT", COMMIT_RE),
                         "image_digest": _required(env, "S4D_REPINNED_FROM_IMAGE", IMAGE_RE)}
    data_id = _required(env, "S4D_DATA_ID", SHA256_RE)
    job_id = _required(env, "S4D_JOB_ID", JOB_RE)
    attempt = _required(env, "S4D_ATTEMPT", re.compile(r"[1-9][0-9]*"))
    evidence_dir = Path(_required(env, "S4D_TEST_EVIDENCE_DIR"))
    if not evidence_dir.is_absolute() or repo in evidence_dir.resolve().parents:
        raise RunIdentityError("Remote native test evidence must be external to the release")
    validate_runtime_path(evidence_dir)
    proof_path = Path(_required(env, "S4D_RELEASE_PROOF"))
    if (
        not proof_path.is_absolute()
        or proof_path.resolve() != proof_path
        or repo in proof_path.resolve().parents
    ):
        raise RunIdentityError("Release proof must be an external, symlink-free file outside the checkout")
    validate_runtime_path(proof_path)
    try:
        proof_bytes = proof_path.read_bytes()
        proof = json.loads(proof_bytes)
    except (OSError, ValueError) as exc:
        raise RunIdentityError("Cannot read the external release proof") from exc
    if not isinstance(proof, dict):
        raise RunIdentityError("Release proof must be a mapping")
    origin_commit = proof.get("verified_origin_commit")
    origin_branch = proof.get("origin_branch")
    origin_ref = RELEASE_ORIGIN_REFS.get(origin_branch) if isinstance(origin_branch, str) else None
    if (
        proof.get("commit") != commit
        or not isinstance(origin_commit, str)
        or COMMIT_RE.fullmatch(origin_commit) is None
        or proof.get("origin_commit", origin_commit) != origin_commit
        or origin_ref is None
        or proof.get("origin_ref") != origin_ref
        or proof.get("verified_ancestor") is not True
        or not isinstance(proof.get("bundle_sha256"), str)
        or SHA256_RE.fullmatch(proof["bundle_sha256"]) is None
        or not isinstance(proof.get("verified_at"), str)
        or not proof["verified_at"]
    ):
        raise RunIdentityError("Release proof does not establish a pushed main commit")
    verify_clean_release(repo, commit)
    if _git(repo, "rev-parse", origin_ref) != origin_commit:
        raise RunIdentityError("Verified origin commit differs from the checkout origin ref")
    if _git(repo, "remote", "get-url", "origin") != proof.get("origin_url"):
        raise RunIdentityError("Release proof origin URL differs from the checkout")
    _git(repo, "merge-base", "--is-ancestor", commit, origin_commit)
    return {
        "schema_version": 1,
        "host": "remote",
        "campaign_eligible": True,
        "git_commit": commit,
        "image_digest": image_digest,
        "data_id": data_id,
        "gpu_uuids": uuids,
        "job_id": job_id,
        "attempt": int(attempt),
        "verified_origin_commit": origin_commit,
        "release_proof_sha256": hashlib.sha256(proof_bytes).hexdigest(),
        "host_config_sha256": _sha256_file(host_config),
        **({"repinned_from": repinned_from} if repinned_from else {}),
    }


def numeric_config(cfg: dict) -> dict:
    normalized = copy.deepcopy(cfg)
    normalized.pop("run", None)
    normalized.pop("wandb", None)
    if isinstance(normalized.get("data"), dict) and "root" in normalized["data"]:
        normalized["data"]["root"] = "<checksummed-data>"
    for section, key in (("model", "anchor_stats"), ("vision", "export_path")):
        node = normalized.get(section)
        if isinstance(node, dict) and node.get(key) is not None:
            node[key] = {"sha256": _sha256_file(Path(node[key]))}
    return normalized


def _repinned(previous: dict, identity: dict) -> bool:
    """True when this launch moves the run from exactly its previous release to the one recorded by the scheduler's
    ``release_repinned`` event (ops/jobs/release_repin.py: same execution scope except the GPU-allowlist files, user
    authorization recorded): the container receives the source release as S4D_REPINNED_FROM_COMMIT/_IMAGE."""
    source = identity.get("repinned_from")
    return (isinstance(source, dict) and "git_commit" in previous and "image_digest" in previous
            and previous["git_commit"] == source.get("git_commit")
            and previous["image_digest"] == source.get("image_digest")
            and (previous["git_commit"], previous["image_digest"]) != (identity.get("git_commit"),
                                                                       identity.get("image_digest")))


def _check_previous(previous: dict, identity: dict) -> None:
    moved = _repinned(previous, identity)
    for key in ("host", "campaign_eligible", "git_commit", "image_digest", "data_id", "config_sha256"):
        if moved and key in ("git_commit", "image_digest"):
            continue  # the recorded re-pin transition (data, config, host and eligibility must still match)
        if key in previous and key in identity and previous[key] != identity[key]:
            raise RunIdentityError(f"Existing remote run has a different {key}")


def _release_history(previous: dict, identity: dict) -> list:
    history = list(previous.get("release_history", []))
    if _repinned(previous, identity):
        history.append({"from_commit": previous["git_commit"], "from_image_digest": previous["image_digest"],
                        "to_commit": identity["git_commit"], "to_image_digest": identity["image_digest"],
                        "attempt": identity.get("attempt"), "at": datetime.now(timezone.utc).isoformat()})
    return history


def _native_diagnostic_identity(repo: Path, run_dir: Path, steps: int | None) -> dict | None:
    from ops.resources.gpu_guard import validate_runtime_path

    if type(steps) is not int or not 0 < steps <= 1000 or os.environ.get("S4D_JOB_ID"):
        return None
    current_test = os.environ.get("PYTEST_CURRENT_TEST", "")
    if "tests/" not in current_test or not current_test.endswith((" (setup)", " (call)", " (teardown)")):
        return None
    temporary = Path(os.environ.get("S4D_PYTEST_TMP", ""))
    evidence = Path(os.environ.get("S4D_TEST_EVIDENCE_DIR", ""))
    if not temporary.is_absolute() or not evidence.is_absolute() or temporary.resolve() != temporary:
        return None
    if Path(repo).resolve() in evidence.resolve().parents:
        return None
    temporary = validate_runtime_path(temporary)
    evidence = validate_runtime_path(evidence)
    if evidence not in temporary.parents or temporary not in run_dir.parents:
        return None
    fingerprints = {
        str(path.relative_to(repo)): _sha256_file(path)
        for folder in ("s4d", "ops", "scripts", "tests", "configs")
        for path in sorted((Path(repo) / folder).rglob("*"))
        if path.is_file() and path.suffix in (".py", ".yaml", ".sh")
    }
    return {
        "schema_version": 1,
        "host": "remote",
        "campaign_eligible": False,
        "purpose": "native_test_diagnostic",
        "git_commit": _git(repo, "rev-parse", "HEAD"),
        "source_dirty": bool(_git(repo, "diff", "--name-only", "HEAD", "--")),
        "source_sha256": fingerprints,
        "native_diagnostic_steps": steps,
        "pytest_context": current_test,
    }


def record_run_identity(
    cfg: dict, repo: Path, run_dir: Path, *, native_diagnostic_steps: int | None = None
) -> dict:
    from ops.resources.gpu_guard import HOST_NAME, validate_runtime_path

    run_dir = validate_runtime_path(run_dir, repository=repo)
    if HOST_NAME == "remote":
        identity = _native_diagnostic_identity(repo, run_dir, native_diagnostic_steps)
        if identity is None:
            identity = verify_remote_identity(repo)
    else:
        identity = {
            "schema_version": 1,
            "host": "local",
            "git_commit": _git(repo, "rev-parse", "HEAD"),
            "source_dirty": bool(_git(repo, "diff", "--name-only", "HEAD", "--")),
        }
    normalized = numeric_config(cfg)
    identity.update(numeric_config=normalized, config_sha256=hashlib.sha256(_json_bytes(normalized)).hexdigest())
    metadata = {key: value for key, value in identity.items() if key not in ("numeric_config", "schema_version")}
    cfg.setdefault("run", {}).update(metadata)
    destinations = [run_dir / "provenance.json"]
    if identity["host"] == "remote" and identity["campaign_eligible"]:
        destinations.append(Path(repo) / "runs" / identity["job_id"] / "provenance.json")
    for path in dict.fromkeys(destinations):
        previous = json.loads(path.read_text()) if path.is_file() else {}
        history = None
        if identity["host"] == "remote" and identity["campaign_eligible"]:
            _check_previous(previous, identity)
            history = _release_history(previous, identity)
        elif previous.get("campaign_eligible") is True:
            raise RunIdentityError("A native diagnostic cannot overwrite campaign provenance")
        _atomic_json(path, {**previous, **identity, **({"release_history": history} if history else {})})
    return identity


def start_remote_run(repo: Path, identity: dict, script: str, arguments: list[str]) -> Path:
    from ops.resources.gpu_guard import validate_runtime_path

    path = validate_runtime_path(Path(repo) / "runs" / identity["job_id"], repository=repo) / "provenance.json"
    previous = json.loads(path.read_text()) if path.is_file() else {}
    _check_previous(previous, identity)
    history = _release_history(previous, identity)
    _atomic_json(
        path,
        {
            **previous,
            **identity,
            **({"release_history": history} if history else {}),
            "script": script,
            "arguments": arguments,
            "started_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return path.parent


def record_remote_exit(run_dir: Path, identity: dict, code: int) -> None:
    _atomic_json(
        run_dir / "exit.json",
        {
            "host": "remote",
            "job_id": identity["job_id"],
            "attempt": identity["attempt"],
            "git_commit": identity["git_commit"],
            "image_digest": identity["image_digest"],
            "gpu_uuids": identity["gpu_uuids"],
            "code": code,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    _atomic_json(run_dir / "exit_code", code)
