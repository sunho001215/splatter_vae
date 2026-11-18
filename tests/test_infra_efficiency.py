"""Infrastructure workflow regressions using disposable repositories and fake process identities only."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts import test_baseline as baseline


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                           "-c", "commit.gpgsign=false", *args], capture_output=True, text=True, check=True).stdout.strip()


def commit(root, name="source"):
    git(root, "add", "-A")
    git(root, "commit", "-qm", name, "--allow-empty")
    return git(root, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q")
    git(root, "symbolic-ref", "HEAD", "refs/heads/main")
    (root / "s4d").mkdir()
    (root / "s4d/a.py").write_text("A = 1\n")
    commit(root)
    return root


def record_baseline(repo, cache, head, env, **overrides):
    store = baseline.test_gate.Baselines(cache)
    record = dict(version=1, commit=head, environment=env, nodes={"tests/test_a.py": ["tests/test_a.py::test_a"]},
                  source_digest=baseline.test_gate.source_digest(baseline.test_gate.commit_sources(repo, head)))
    store.root.mkdir(parents=True, exist_ok=True)
    store.path(head).write_text(json.dumps({**record, **overrides}))


def test_baseline_skips_missing_incompatible_and_tampered_records(repo, tmp_path):
    cache, env = tmp_path / "cache", {"host": "local"}
    first = git(repo, "rev-parse", "HEAD")
    record_baseline(repo, cache, first, env)
    wrong = commit(repo, "wrong environment")
    record_baseline(repo, cache, wrong, {"host": "remote"})
    tampered = commit(repo, "tampered digest")
    record_baseline(repo, cache, tampered, env, source_digest="bad")
    commit(repo, "no record")
    before = {p: p.read_bytes() for p in cache.rglob("*.json")}
    assert baseline.nearest_baseline(repo, cache, env) == first
    assert before == {p: p.read_bytes() for p in cache.rglob("*.json")}
    with pytest.raises(ValueError, match="no compatible accepted ancestor"):
        baseline.nearest_baseline(repo, cache, {"host": "unknown"})


def test_baseline_nearest_ancestor_includes_merge_parents_but_not_other_branches(repo, tmp_path):
    cache, env = tmp_path / "cache", {"host": "local"}
    first = git(repo, "rev-parse", "HEAD")
    record_baseline(repo, cache, first, env)
    git(repo, "checkout", "-qb", "side")
    side = commit(repo, "side baseline")
    record_baseline(repo, cache, side, env)
    git(repo, "checkout", "-q", "main")
    commit(repo, "main")
    assert baseline.nearest_baseline(repo, cache, env) == first
    git(repo, "merge", "--no-ff", "-qm", "merge", "side")
    assert baseline.nearest_baseline(repo, cache, env) == side


@pytest.fixture
def adopter():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "runs/setup/tools/merge_adopt_local_mig16_20261115.py"
    spec = importlib.util.spec_from_file_location("infra_adopter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evidence_files(repo, adopter):
    for name in adopter.EVIDENCE_FILES:
        path = repo / name
        path.parent.mkdir(exist_ok=True)
        path.write_text("original\n")
    return commit(repo, "original evidence")


@pytest.mark.local_host_data  # loads runs/setup/tools from the local checkout (masked by the runs/ mount remotely)
def test_evidence_only_candidate_binds_original_head_without_rewriting(repo, adopter):
    produced = evidence_files(repo, adopter)
    evidence = dict(head=produced, source_sha256=baseline.test_gate.commit_sources(repo, produced))
    (repo / "docs/tests.json").write_text("committed report\n")
    candidate = commit(repo, "evidence only")
    assert adopter.evidence_commit(repo, candidate, evidence) == produced
    assert evidence["head"] == produced
    with pytest.raises(ValueError, match="sources differ"):
        adopter.evidence_commit(repo, candidate, {**evidence, "source_sha256": {}})
    (repo / "s4d/a.py").write_text("A = 2\n")
    with pytest.raises(ValueError, match="outside the three evidence files"):
        adopter.evidence_commit(repo, commit(repo, "code change"), evidence)


@pytest.mark.local_host_data  # loads runs/setup/tools from the local checkout (masked by the runs/ mount remotely)
@pytest.mark.parametrize("change", ["other_doc", "rename", "delete", "symlink"])
def test_evidence_candidate_rejects_other_paths_and_nonregular_files(repo, adopter, change):
    produced = evidence_files(repo, adopter)
    evidence = dict(head=produced, source_sha256=baseline.test_gate.commit_sources(repo, produced))
    path = repo / "docs/tests.log"
    if change == "other_doc":
        (repo / "docs/notes.md").write_text("other doc")
    elif change == "rename":
        path.rename(repo / "docs/renamed.log")
    elif change == "delete":
        path.unlink()
    else:
        path.unlink()
        path.symlink_to("tests.json")
    with pytest.raises(ValueError, match="outside|regular"):
        adopter.evidence_commit(repo, commit(repo), evidence)


@pytest.mark.local_host_data  # loads runs/setup/tools from the local checkout (masked by the runs/ mount remotely)
def test_main_dirty_evidence_is_backed_up_and_restored_on_failure(repo, adopter):
    evidence_files(repo, adopter)
    paths = [repo / name for name in adopter.EVIDENCE_FILES]
    for path in paths:
        path.write_text("dirty " + path.name)
    paths[-1].unlink()  # preserve absent files too
    paths[0].chmod(0o600)
    original = {p: (p.read_bytes(), p.stat().st_mode) if p.exists() else None for p in paths}
    index = git(repo, "write-tree")
    with pytest.raises(RuntimeError, match="failed merge"):
        with adopter.preserve_main_evidence(repo) as backup:
            assert all(p.read_text() == "original\n" for p in paths)
            assert (backup / "tests.json").read_bytes() == original[paths[0]][0]
            raise RuntimeError("failed merge")
    assert original == {p: (p.read_bytes(), p.stat().st_mode) if p.exists() else None for p in paths}
    assert git(repo, "write-tree") == index


@pytest.mark.local_host_data  # loads runs/setup/tools from the local checkout (masked by the runs/ mount remotely)
def test_dirty_evidence_no_longer_blocks_ff_merge_and_backup_survives(repo, adopter):
    evidence_files(repo, adopter)
    git(repo, "checkout", "-qb", "candidate")
    (repo / "docs/tests.json").write_text("candidate report\n")
    candidate = commit(repo)
    git(repo, "checkout", "-q", "main")
    (repo / "docs/tests.json").write_text("dirty main report\n")
    with adopter.preserve_main_evidence(repo) as backup:
        git(repo, "merge", "--ff-only", candidate)
        (repo / "docs/tests.json").write_text("adopted report\n")
    assert (backup / "tests.json").read_text() == "dirty main report\n"
    assert (repo / "docs/tests.json").read_text() == "adopted report\n"
    assert git(repo, "rev-parse", "HEAD") == candidate


@pytest.mark.local_host_data  # loads runs/setup/tools from the local checkout (masked by the runs/ mount remotely)
def test_staged_main_evidence_is_refused_without_changing_index(repo, adopter):
    evidence_files(repo, adopter)
    (repo / "docs/tests.json").write_text("staged report\n")
    git(repo, "add", "docs/tests.json")
    index = git(repo, "write-tree")
    with pytest.raises(ValueError, match="staged evidence"):
        with adopter.preserve_main_evidence(repo):
            pytest.fail("must not enter")
    assert git(repo, "write-tree") == index
    assert (repo / "docs/tests.json").read_text() == "staged report\n"


@pytest.mark.local_host_data  # loads runs/setup/tools from the local checkout (masked by the runs/ mount remotely)
def test_selective_adopter_reverifies_the_tested_head_of_evidence_only_candidate(repo, adopter, monkeypatch):
    import test_gate

    produced = evidence_files(repo, adopter)
    report = dict(head=produced, full_suite_passed=True, exit_code=0, host="local", source_changed_during_suite=False,
                  source_sha256=baseline.test_gate.commit_sources(repo, produced),
                  counts=dict(passed=1, failed=0, error=0, skipped=0),
                  gpu_mapping=[dict(uuid=adopter.APPROVED_HOST_GPUS["local"][0])],
                  release_gate=dict(accepted=True, mode="selective", candidate=dict(head=produced)))
    (repo / "docs/tests.json").write_text(json.dumps(report))
    candidate = commit(repo, "commit evidence")
    calls = []
    monkeypatch.setattr(test_gate, "adoption_check", lambda evidence, tree, head: calls.append(head) or [])
    assert adopter.verify(repo, candidate)["head"] == produced
    assert calls == [produced]
    monkeypatch.setattr(test_gate, "adoption_check", lambda *a: ["candidate gate refuses"])
    with pytest.raises(ValueError, match="candidate gate refuses"):
        adopter.verify(repo, candidate)


@pytest.fixture
def pauser(tmp_path, monkeypatch):
    from ops.jobs import pause_at_checkpoint as pause
    from ops.jobs import scheduler

    root = tmp_path / "local"
    checkpoint = root / "runs/job/checkpoints/latest.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"initial")
    registry = root / "experiments/registry.jsonl"
    registry.parent.mkdir()
    registry.write_text("")
    state = dict(status="running", host="local", pid=100, attempt=2, gpu="MIG-owned", script="scripts/train.py")
    processes = {pid: dict(pid=pid, group=100, session=100, start_ticks=pid * 10,
                           command=[b"python", b"--name", b"job"]) for pid in (100, 101)}
    monkeypatch.setattr(scheduler, "read_registry", lambda path: {"job": state})
    monkeypatch.setattr(pause, "members", lambda pid: processes)
    signals = []
    monkeypatch.setattr(pause.os, "killpg", lambda *args: signals.append(args))
    monkeypatch.setattr(pause, "checkpoint_info", lambda path: (42, "checkpoint-digest"))
    from types import SimpleNamespace
    return SimpleNamespace(module=pause, root=root, checkpoint=checkpoint, registry=registry,
                           state=state, processes=processes, signals=signals)


def test_pause_dry_run_never_waits_writes_or_signals(pauser, monkeypatch):
    p = pauser
    monkeypatch.setattr(p.module.time, "sleep", lambda _: pytest.fail("dry-run waited"))
    assert p.module.pause("job", "maintenance", root=p.root)["execute"] is False
    assert p.registry.read_bytes() == b""
    assert not p.registry.with_suffix(".lock").exists()
    assert p.signals == []


@pytest.mark.parametrize("mutation", ["remote", "stopped", "foreign", "not_session_leader", "single_process"])
def test_pause_refuses_unowned_process_groups(pauser, mutation):
    p = pauser
    if mutation == "remote":
        p.state["host"] = "remote"
    elif mutation == "stopped":
        p.state["status"] = "crashed"
    elif mutation == "foreign":
        p.processes[101]["command"][-1] = b"other-job"
    elif mutation == "not_session_leader":
        p.processes[100]["session"] = 99
    else:
        del p.processes[101]
    with pytest.raises(p.module.PauseError):
        p.module.pause("job", "maintenance", root=p.root, execute=True)
    assert p.signals == [] and p.registry.read_bytes() == b""


def test_pause_waits_for_fresh_checkpoint_and_records_before_signal(pauser, monkeypatch):
    p = pauser
    sleeps = []
    def sleep(seconds):
        sleeps.append(seconds)
        if seconds == 5:
            p.checkpoint.write_bytes(b"fresh checkpoint")
    monkeypatch.setattr(p.module.time, "sleep", sleep)
    def signal(pid, sig):
        assert json.loads(p.registry.read_text())["checkpoint_step"] == 42
        p.signals.append((pid, sig))
    monkeypatch.setattr(p.module.os, "killpg", signal)
    result = p.module.pause("job", "drain", root=p.root, execute=True)
    assert sleeps == [5, 20]
    assert result["reason"] == "drain" and result["attempt"] == 2
    assert p.signals == [(100, p.module.signal.SIGTERM)]


@pytest.mark.parametrize("change", ["attempt", "pid_reuse", "foreign", "checkpoint"])
def test_pause_rechecks_identity_and_checkpoint_immediately_before_signal(pauser, monkeypatch, change):
    p = pauser
    monkeypatch.setattr(p.module.time, "sleep", lambda seconds: p.checkpoint.write_bytes(b"fresh") if seconds == 5 else None)
    def info(path):
        if change == "attempt":
            p.state["attempt"] += 1
        elif change == "pid_reuse":
            p.processes[100]["start_ticks"] += 1
        elif change == "foreign":
            p.processes[101]["command"][-1] = b"foreign"
        else:
            p.checkpoint.write_bytes(b"newer checkpoint")
        return 42, "digest"
    monkeypatch.setattr(p.module, "checkpoint_info", info)
    with pytest.raises(p.module.PauseError, match="changed|foreign"):
        p.module.pause("job", "drain", root=p.root, execute=True)
    assert p.signals == [] and p.registry.read_bytes() == b""


def test_pause_timeout_sends_no_signal_and_writes_no_event(pauser):
    p = pauser
    with pytest.raises(TimeoutError):
        p.module.pause("job", "drain", root=p.root, execute=True, timeout=0)
    assert p.signals == [] and p.registry.read_bytes() == b""
