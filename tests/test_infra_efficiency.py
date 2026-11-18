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
