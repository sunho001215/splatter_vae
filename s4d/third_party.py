"""Pinned third-party code (git submodules under ``third_party/``) in the source identity and in remote releases.

A submodule counts by its pinned commit, recorded as ``third_party/<name>@commit``. A working checkout that is
missing, at another commit, or carries modified or untracked files is recorded as such, so the identity can never
match a commit's or the test evidence's. Required changes to upstream code live as patch files under
``third_party/patches/`` (applied to a copy, never in place) and count by content, like ``.gitmodules``.
Standard library only: the GPU guard imports this before any GPU-aware package.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

ROOT = "third_party"
PATCHES = "third_party/patches"
GITMODULES = ".gitmodules"


def _git(repo: Path, *args: str) -> bytes:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=True).stdout


def pinned_submodules(repo: Path, commit: str | None = None) -> dict[str, str]:
    """Submodule path -> pinned commit, from ``commit``'s tree or (default) the index."""
    if commit is None:
        listing = _git(repo, "ls-files", "-s", "-z", "--", ROOT)
        rows = [entry.split(b"\t", 1) for entry in listing.split(b"\0") if entry]
        return {path.decode(): meta.split()[1].decode() for meta, path in rows if meta.split()[0] == b"160000"}
    listing = _git(repo, "ls-tree", "-r", "-z", commit, "--", ROOT)
    rows = [entry.split(b"\t", 1) for entry in listing.split(b"\0") if entry]
    return {path.decode(): meta.split()[2].decode() for meta, path in rows if meta.split()[1] == b"commit"}


def checkout_state(path: Path, pinned: str) -> str:
    """``pinned`` for a clean checkout at the pinned commit, otherwise a value that cannot equal any commit."""
    if not (path / ".git").exists():  # an empty gitlink directory: rev-parse would answer for the superproject
        return "missing"
    try:
        head = _git(path, "rev-parse", "HEAD").decode().strip()
        dirty = _git(path, "status", "--porcelain", "--untracked-files=all").strip()
    except (OSError, subprocess.CalledProcessError):
        return "unreadable"
    if head != pinned:
        return f"other:{head}"
    return f"dirty:{head}" if dirty else pinned


def working_tree_identity(repo: Path) -> dict[str, str]:
    """Identity entries of the checked-out third-party code (empty when the repository has none)."""
    repo = Path(repo)
    out = {f"{path}@commit": checkout_state(repo / path, pinned) for path, pinned in pinned_submodules(repo).items()}
    files = [repo / GITMODULES] if (repo / GITMODULES).is_file() else []
    files += sorted(p for p in (repo / PATCHES).rglob("*") if p.is_file()) if (repo / PATCHES).is_dir() else []
    out.update({str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    return out


def commit_identity(repo: Path, commit: str) -> dict[str, str]:
    """The same entries as ``working_tree_identity`` for a clean checkout of ``commit``, read from git objects."""
    out = {f"{path}@commit": pinned for path, pinned in pinned_submodules(repo, commit).items()}
    listing = _git(repo, "ls-tree", "-r", "-z", commit, "--", GITMODULES, PATCHES)
    for entry in listing.split(b"\0"):
        if entry:
            meta, path = entry.split(b"\t", 1)
            if meta.split()[1] == b"blob":
                out[path.decode()] = hashlib.sha256(_git(repo, "cat-file", "blob", meta.split()[2].decode())).hexdigest()
    return out
