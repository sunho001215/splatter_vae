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
