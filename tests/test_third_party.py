"""Pinned third-party submodules and patch files are part of the source identity."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from s4d import third_party


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), "-c", "protocol.file.allow=always", "-c", "user.name=t",
                           "-c", "user.email=t@t", *args], capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def superproject(tmp_path):
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    git(upstream, "init", "-q")
    (upstream / "model.py").write_text("WIDTH = 1\n")
    git(upstream, "add", "model.py")
    git(upstream, "commit", "-qm", "first")
    first = git(upstream, "rev-parse", "HEAD")
    (upstream / "model.py").write_text("WIDTH = 2\n")
    git(upstream, "commit", "-qam", "second")
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "submodule", "add", "-q", str(upstream), "third_party/upstream")
    git(repo / "third_party/upstream", "checkout", "-q", first)
    (repo / "third_party/patches/upstream").mkdir(parents=True)
    (repo / "third_party/patches/upstream/fix.patch").write_text("--- a\n+++ b\n")
    git(repo, "add", "third_party")
    git(repo, "commit", "-qm", "pin")
    return repo, first


def test_clean_pinned_checkout_matches_the_commit_identity(superproject):
    repo, first = superproject
    tree = third_party.working_tree_identity(repo)
    assert tree["third_party/upstream@commit"] == first
    assert tree["third_party/patches/upstream/fix.patch"] == hashlib.sha256(b"--- a\n+++ b\n").hexdigest()
    assert ".gitmodules" in tree
    assert tree == third_party.commit_identity(repo, git(repo, "rev-parse", "HEAD"))


def test_modified_untracked_moved_or_missing_submodules_never_match(superproject):
    repo, first = superproject
    module = repo / "third_party/upstream"
    (module / "model.py").write_text("WIDTH = 3\n")
    assert third_party.working_tree_identity(repo)["third_party/upstream@commit"] == f"dirty:{first}"
    git(module, "checkout", "-q", "--", "model.py")
    (module / "scratch.txt").write_text("x")
    assert third_party.working_tree_identity(repo)["third_party/upstream@commit"] == f"dirty:{first}"
    (module / "scratch.txt").unlink()
    git(module, "checkout", "-q", "master" if git(module, "branch", "--list", "master") else "main")
    assert third_party.working_tree_identity(repo)["third_party/upstream@commit"].startswith("other:")
    git(repo, "submodule", "deinit", "-q", "-f", "third_party/upstream")
    assert third_party.working_tree_identity(repo)["third_party/upstream@commit"] == "missing"


def test_patch_files_count_by_content(superproject):
    repo, _ = superproject
    before = third_party.working_tree_identity(repo)
    (repo / "third_party/patches/upstream/fix.patch").write_text("--- a\n+++ c\n")
    assert third_party.working_tree_identity(repo) != before


def test_repository_without_third_party_code_adds_nothing(tmp_path):
    git(tmp_path, "init", "-q")
    (tmp_path / "a.py").write_text("")
    git(tmp_path, "add", "a.py")
    git(tmp_path, "commit", "-qm", "a")
    assert third_party.working_tree_identity(tmp_path) == {}
    assert third_party.commit_identity(tmp_path, git(tmp_path, "rev-parse", "HEAD")) == {}


def test_source_tree_is_the_checkout_or_a_patched_copy_never_an_edit(superproject, tmp_path):
    repo, first = superproject
    (repo / "third_party/patches/upstream/fix.patch").write_text(
        "--- a/model.py\n+++ b/model.py\n@@ -1 +1 @@\n-WIDTH = 1\n+WIDTH = 9\n"
    )
    tree = third_party.source_tree("upstream", repo, tmp_path / "cache")
    assert (tree / "model.py").read_text() == "WIDTH = 9\n"
    assert (repo / "third_party/upstream/model.py").read_text() == "WIDTH = 1\n"
    assert third_party.working_tree_identity(repo)["third_party/upstream@commit"] == first
    assert third_party.source_tree("upstream", repo, tmp_path / "cache") == tree  # cached
    (repo / "third_party/patches/upstream/fix.patch").unlink()
    assert third_party.source_tree("upstream", repo, tmp_path / "cache") == repo / "third_party/upstream"
    (repo / "third_party/upstream/model.py").write_text("WIDTH = 5\n")
    with pytest.raises(RuntimeError, match="clean checkout"):
        third_party.source_tree("upstream", repo, tmp_path / "cache")
