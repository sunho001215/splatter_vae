"""Pushed-release verification and numerical run provenance."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from s4d import gpu_guard
from s4d.run_identity import (
    RunIdentityError,
    _native_diagnostic_identity,
    numeric_config,
    record_remote_exit,
    record_run_identity,
    start_remote_run,
    verify_clean_release,
    verify_remote_identity,
)

REPO = Path(__file__).resolve().parents[1]


def git(repo, *args):
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Release test",
        "GIT_AUTHOR_EMAIL": "release-test@example.invalid",
        "GIT_COMMITTER_NAME": "Release test",
        "GIT_COMMITTER_EMAIL": "release-test@example.invalid",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
    }
    return subprocess.run(
        ["git", "-C", str(repo), *args], env=env, text=True, capture_output=True, check=True
    ).stdout.strip()


@pytest.fixture
def release(tmp_path):
    repo = tmp_path / "release"
    repo.mkdir()
    git(repo, "init", "--initial-branch=splatter4d")
    (repo / "scripts").mkdir()
    (repo / "scripts" / "train.py").write_text("from _bootstrap import guard_gpus\nguard_gpus()\n")
    (repo / ".gitignore").write_text(".cache/\n.venv/\noutputs/\n")
    git(repo, "add", "scripts/train.py", ".gitignore")
    git(repo, "commit", "-m", "Release fixture")
    commit = git(repo, "rev-parse", "HEAD")
    git(repo, "remote", "add", "origin", "https://github.com/sunho001215/splatter_vae.git")
    git(repo, "update-ref", "refs/remotes/origin/splatter4d", commit)
    proof = {
        "commit": commit,
        "verified_origin_commit": commit,
        "origin_commit": commit,
        "origin_branch": "splatter4d",
        "origin_ref": "refs/remotes/origin/splatter4d",
        "origin_url": git(repo, "remote", "get-url", "origin"),
        "verified_ancestor": True,
        "verified_at": "2026-10-10T09:30:00+00:00",
        "bundle_sha256": "a" * 64,
    }
    proof_path = tmp_path / "provenance.json"
    proof_path.write_text(json.dumps(proof))
    env = {
        "S4D_HOST_CONFIG": str(REPO / "configs/hosts/remote.yaml"),
        "S4D_EXPECTED_COMMIT": commit,
        "S4D_IMAGE_DIGEST": "sha256:" + "b" * 64,
        "S4D_DATA_ID": "c" * 64,
        "S4D_JOB_ID": "remote-test",
        "S4D_ATTEMPT": "1",
        "S4D_RELEASE_PROOF": str(proof_path),
        "S4D_TEST_EVIDENCE_DIR": str(REPO / "runs" / "remote" / "evidence"),
        "CUDA_VISIBLE_DEVICES": gpu_guard.APPROVED_HOST_GPUS["remote"][0],
    }
    return repo, env, proof, proof_path


def test_verified_release_records_actual_origin_ancestry(release):
    repo, env, _, _ = release
    identity = verify_remote_identity(repo, env)
    assert identity["host"] == "remote"
    assert identity["git_commit"] == identity["verified_origin_commit"] == env["S4D_EXPECTED_COMMIT"]
    assert identity["image_digest"] == env["S4D_IMAGE_DIGEST"]
    assert identity["data_id"] == env["S4D_DATA_ID"]
    assert identity["attempt"] == 1
    assert len(identity["release_proof_sha256"]) == 64


@pytest.mark.parametrize(
    "key,value",
    [
        ("S4D_HOST_CONFIG", ""),
        ("S4D_EXPECTED_COMMIT", "short"),
        ("S4D_IMAGE_DIGEST", "mutable-tag"),
        ("S4D_DATA_ID", "not-checksummed"),
        ("S4D_JOB_ID", "../other-run"),
        ("S4D_ATTEMPT", "0"),
        ("S4D_RELEASE_PROOF", ""),
        ("S4D_TEST_EVIDENCE_DIR", ""),
        ("CUDA_VISIBLE_DEVICES", "0"),
    ],
)
def test_remote_identity_rejects_missing_or_invalid_environment(release, key, value):
    repo, env, _, _ = release
    with pytest.raises(RunIdentityError):
        verify_remote_identity(repo, {**env, key: value})


def test_remote_identity_rejects_local_host_config(release):
    repo, env, _, _ = release
    env["S4D_HOST_CONFIG"] = str(REPO / "configs/hosts/local.yaml")
    with pytest.raises(RunIdentityError, match="explicit remote"):
        verify_remote_identity(repo, env)


@pytest.mark.parametrize("staged", [False, True])
def test_remote_release_rejects_tracked_changes(release, staged):
    repo, env, _, _ = release
    (repo / "scripts" / "train.py").write_text("changed = True\n")
    if staged:
        git(repo, "add", "scripts/train.py")
    with pytest.raises(RunIdentityError, match="tracked changes"):
        verify_remote_identity(repo, env)


@pytest.mark.parametrize("path", ["scripts/untracked.py", ".venv/untracked.py"])
def test_remote_release_rejects_untracked_and_ignored_sources(release, path):
    repo, env, _, _ = release
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("untracked = True\n")
    with pytest.raises(RunIdentityError, match="untracked files"):
        verify_remote_identity(repo, env)


def test_remote_release_allows_only_generated_runtime_roots(release):
    repo, env, _, _ = release
    for folder in ("runs/remote-test", "outputs/train", ".cache"):
        target = repo / folder / "config.yaml"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("generated: true\n")
    verify_clean_release(repo, env["S4D_EXPECTED_COMMIT"])
    assert verify_remote_identity(repo, env)["host"] == "remote"


@pytest.mark.parametrize(
    "key,value",
    [
        ("commit", "d" * 40),
        ("verified_origin_commit", "e" * 40),
        ("origin_branch", "main"),
        ("origin_ref", "refs/heads/splatter4d"),
        ("origin_url", "different-origin"),
        ("verified_ancestor", False),
        ("bundle_sha256", "unverified"),
    ],
)
def test_remote_release_rejects_inconsistent_proof(release, key, value):
    repo, env, proof, proof_path = release
    proof_path.write_text(json.dumps({**proof, key: value}))
    with pytest.raises(RunIdentityError):
        verify_remote_identity(repo, env)


def test_remote_release_checks_ancestry_not_a_claimed_boolean(release):
    repo, env, proof, proof_path = release
    (repo / "scripts" / "train.py").write_text("from _bootstrap import guard_gpus\nguard_gpus()\nnew = True\n")
    git(repo, "add", "scripts/train.py")
    git(repo, "commit", "-m", "Unpushed descendant")
    descendant = git(repo, "rev-parse", "HEAD")
    proof_path.write_text(json.dumps({**proof, "commit": descendant}))
    env["S4D_EXPECTED_COMMIT"] = descendant
    with pytest.raises(RunIdentityError, match="merge-base"):
        verify_remote_identity(repo, env)


def test_remote_release_accepts_pushed_ancestor_of_tip(release):
    repo, env, proof, proof_path = release
    git(repo, "commit", "--allow-empty", "-m", "Later pushed tip")
    tip = git(repo, "rev-parse", "HEAD")
    git(repo, "update-ref", "refs/remotes/origin/splatter4d", tip)
    git(repo, "switch", "--detach", env["S4D_EXPECTED_COMMIT"])
    proof_path.write_text(json.dumps({**proof, "verified_origin_commit": tip, "origin_commit": tip}))
    assert verify_remote_identity(repo, env)["verified_origin_commit"] == tip


def test_release_proof_must_remain_outside_checkout(release):
    repo, env, proof, _ = release
    proof_path = repo / "runs" / "proof.json"
    proof_path.parent.mkdir()
    proof_path.write_text(json.dumps(proof))
    env["S4D_RELEASE_PROOF"] = str(proof_path)
    with pytest.raises(RunIdentityError, match="external.*file"):
        verify_remote_identity(repo, env)


def test_numeric_identity_normalizes_paths_not_content_or_seed(tmp_path):
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    first.write_text('{"mean": [1, 2, 3]}')
    second.write_bytes(first.read_bytes())
    cfg = {
        "data": {"root": "/local/data", "task": "hammer", "strides": [2, 4, 6]},
        "train": {"seed": 0, "steps": 200000, "stop_step": 100000},
        "model": {"anchor_stats": str(first)},
        "run": {"host": "local", "dir": "/local/run", "name": "one"},
        "wandb": {"enabled": False},
    }
    original = copy.deepcopy(cfg)
    variant = copy.deepcopy(cfg)
    variant["run"].update(host="remote", dir="/remote/run", name="two")
    variant["wandb"]["enabled"] = True
    variant["data"]["root"] = "/remote/data"
    variant["model"]["anchor_stats"] = str(second)
    assert numeric_config(cfg) == numeric_config(variant)
    assert cfg == original
    variant["train"]["seed"] = 1
    assert numeric_config(cfg) != numeric_config(variant)
    variant["train"]["seed"] = 0
    second.write_text('{"mean": [3, 2, 1]}')
    assert numeric_config(cfg) != numeric_config(variant)


def test_remote_config_and_exit_evidence_are_atomic_and_consistent(release, monkeypatch):
    repo, env, _, _ = release
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(gpu_guard, "HOST_NAME", "remote")
    identity = verify_remote_identity(repo, env)
    job_dir = start_remote_run(repo, identity, "scripts/train.py", ["--name", "train-test"])
    cfg = {"train": {"seed": 0}, "run": {"name": "train-test", "gpus": []}}
    identity = record_run_identity(cfg, repo, repo / "outputs" / "train-test")
    evidence = json.loads((job_dir / "provenance.json").read_text())
    assert evidence["arguments"] == ["--name", "train-test"]
    assert evidence["numeric_config"] == {"train": {"seed": 0}}
    assert cfg["run"]["name"] == "train-test" and cfg["run"]["host"] == "remote"
    assert cfg["run"]["config_sha256"] == evidence["config_sha256"]
    record_remote_exit(job_dir, identity, 7)
    assert json.loads((job_dir / "exit.json").read_text())["code"] == 7
    assert (job_dir / "exit_code").read_text() == "7\n"
    assert not list(job_dir.glob("*.tmp"))
    cfg["train"]["seed"] = 1
    with pytest.raises(RunIdentityError, match="config_sha256"):
        record_run_identity(cfg, repo, repo / "outputs" / "train-test")


def test_native_diagnostic_is_bounded_and_never_campaign_eligible(release, tmp_path, monkeypatch):
    repo, _, _, _ = release
    evidence = tmp_path / "native-evidence"
    temporary = evidence / "tmp"
    run_dir = temporary / "rl-test"
    run_dir.mkdir(parents=True)
    monkeypatch.setenv("S4D_TEST_EVIDENCE_DIR", str(evidence))
    monkeypatch.setenv("S4D_PYTEST_TMP", str(temporary))
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests/test_rl_cli.py::test_native_rl (call)")
    monkeypatch.delenv("S4D_JOB_ID", raising=False)
    identity = _native_diagnostic_identity(repo, run_dir, 60)
    assert identity["campaign_eligible"] is False
    assert identity["purpose"] == "native_test_diagnostic"
    assert identity["native_diagnostic_steps"] == 60
    assert "scripts/train.py" in identity["source_sha256"]
    assert _native_diagnostic_identity(repo, run_dir, 90)["native_diagnostic_steps"] == 90
    assert _native_diagnostic_identity(repo, run_dir, 1001) is None
    assert _native_diagnostic_identity(repo, tmp_path / "outside-tmp", 60) is None
    monkeypatch.setenv("S4D_JOB_ID", "campaign-container")
    assert _native_diagnostic_identity(repo, run_dir, 60) is None
    monkeypatch.delenv("S4D_JOB_ID")
    monkeypatch.delenv("PYTEST_CURRENT_TEST")
    assert _native_diagnostic_identity(repo, run_dir, 60) is None


def test_native_diagnostic_refuses_checkout_evidence(release, monkeypatch):
    repo, _, _, _ = release
    evidence = repo / "native-evidence"
    temporary = evidence / "tmp"
    run_dir = temporary / "rl-test"
    run_dir.mkdir(parents=True)
    monkeypatch.setenv("S4D_TEST_EVIDENCE_DIR", str(evidence))
    monkeypatch.setenv("S4D_PYTEST_TMP", str(temporary))
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests/test_rl_cli.py::test_native_rl (call)")
    monkeypatch.delenv("S4D_JOB_ID", raising=False)
    assert _native_diagnostic_identity(repo, run_dir, 60) is None


def test_wandb_tags_include_host_without_changing_numerical_config(monkeypatch):
    from s4d.diag import wandb_log

    captured = {}
    monkeypatch.setattr(gpu_guard, "HOST_NAME", "remote")
    monkeypatch.setattr(wandb_log, "_ENABLED", False)
    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(init=lambda **kwargs: captured.update(kwargs), Settings=lambda **kwargs: kwargs),
    )
    cfg = {"train": {"seed": 0}, "run": {"host": "remote"}}
    wandb_log.init_wandb(cfg, "test", "test", True, Path("runs/test"))
    assert captured["tags"] == ["host=remote"]
    assert captured["config"] is cfg


def test_entry_rejects_bad_identity_before_cuda_import(release):
    _, env, _, _ = release
    env["S4D_IMAGE_DIGEST"] = "mutable-tag"
    result = subprocess.run(
        [sys.executable, "-I", str(REPO / "scripts/remote/entry.py"), "--script", "scripts/train.py"],
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode != 0
    assert "S4D_IMAGE_DIGEST" in result.stderr
    assert "torch" not in result.stderr and "NVML" not in result.stderr


def test_entry_child_exit_and_output_are_retained(tmp_path):
    spec = importlib.util.spec_from_file_location("remote_entry", REPO / "scripts/remote/entry.py")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    child = tmp_path / "child.py"
    child.write_text("print('retained child output', flush=True)\nraise SystemExit(7)\n")
    log = tmp_path / "console.log"
    with log.open("a") as console:
        assert entry.run_child(child, [], console) == 7
    assert "retained child output" in log.read_text()
