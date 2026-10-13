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


def source_tree(name: str, repo: Path | None = None, cache_root: Path | None = None) -> Path:
    """Importable tree of ``third_party/<name>``: the clean pinned checkout itself, or, when patch files exist under
    ``third_party/patches/<name>/``, a copy of the pinned commit with those patches applied, built once under the
    cache root (``S4D_CACHE_ROOT`` or ``<repo>/.cache``) and keyed by commit and patch contents. Never edits the
    checkout; refuses a checkout that is missing, moved or dirty."""
    import io
    import json
    import os
    import shutil
    import tarfile
    import tempfile

    repo = Path(repo) if repo is not None else Path(__file__).resolve().parents[1]
    path = f"{ROOT}/{name}"
    pinned = pinned_submodules(repo).get(path)
    if pinned is None:
        raise ValueError(f"{path} is not a pinned submodule")
    state = checkout_state(repo / path, pinned)
    if state != pinned:
        raise RuntimeError(f"{path} is not a clean checkout of its pin ({state}); run git submodule update --init")
    patches = sorted((repo / PATCHES / name).glob("*.patch"))
    if not patches:
        return repo / path
    digest = hashlib.sha256(b"".join(p.name.encode() + b"\0" + p.read_bytes() for p in patches)).hexdigest()
    cache = Path(cache_root or os.environ.get("S4D_CACHE_ROOT", str(repo / ".cache"))) / "third_party"
    target = cache / f"{name}-{pinned[:12]}-{digest[:12]}"
    record = {"submodule": path, "commit": pinned, "patches": [p.name for p in patches], "patches_sha256": digest}
    marker = target / ".s4d-patched.json"
    if marker.is_file() and json.loads(marker.read_text()) == record:
        return target
    cache.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{name}-", dir=cache))
    try:
        with tarfile.open(fileobj=io.BytesIO(_git(repo / path, "archive", "--format=tar", pinned))) as archive:
            archive.extractall(staging, filter="data")
        for patch in patches:
            subprocess.run(["git", "apply", "--whitespace=nowarn", str(patch)], cwd=staging, capture_output=True,
                           check=True)
        (staging / ".s4d-patched.json").write_text(json.dumps(record, indent=2))
        if target.exists():  # a stale or concurrent build: replace it atomically enough for single-writer use
            shutil.rmtree(target)
        os.replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return target
