from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from pathlib import Path

DEFAULT_DROID_ROOT = Path("/ws/data/ws/droid")
DEFAULT_DERIVED_ROOT = Path("/ws/data/ws/droid_splattervae")


def resolved(path: str | os.PathLike[str]) -> Path:
    """Resolve a path without requiring the final component to exist."""
    return Path(path).expanduser().resolve(strict=False)


def is_within(path: str | os.PathLike[str], parent: str | os.PathLike[str]) -> bool:
    candidate = resolved(path)
    root = resolved(parent)
    return candidate == root or root in candidate.parents


def validate_derived_root(
    derived_root: str | os.PathLike[str],
    droid_root: str | os.PathLike[str] = DEFAULT_DROID_ROOT,
) -> Path:
    """Reject any derived output that could mutate the source DROID tree."""
    output = resolved(derived_root)
    source = resolved(droid_root)
    if is_within(output, source):
        raise ValueError(
            f"Derived output {output} is inside the read-only DROID source {source}."
        )
    if is_within(source, output):
        raise ValueError(
            f"Derived output {output} contains the DROID source {source}; choose a sibling directory."
        )
    return output


def prepare_derived_layout(
    derived_root: str | os.PathLike[str],
    droid_root: str | os.PathLike[str] = DEFAULT_DROID_ROOT,
) -> dict[str, Path]:
    root = validate_derived_root(derived_root, droid_root)
    names = (
        "calibration",
        "manifests",
        "xlens",
        "waft",
        "see3d",
        "workspace_stats",
        "metadata",
        "logs",
    )
    paths = {name: root / name for name in names}
    root.mkdir(parents=True, exist_ok=True)
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def source_tree_fingerprint(
    droid_root: str | os.PathLike[str],
    *,
    include_content: bool = False,
) -> dict[str, object]:
    """Create a deterministic read-only source-tree inventory.

    Metadata mode is inexpensive enough for large RLDS trees. Content mode is
    intentionally explicit because hashing a complete DROID installation is
    expensive.
    """
    root = resolved(droid_root)
    if not root.is_dir():
        raise FileNotFoundError(f"DROID source directory does not exist: {root}")
    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    for path in sorted((p for p in root.rglob("*") if p.is_file()), key=str):
        stat = path.stat()
        relative = path.relative_to(root).as_posix()
        record = f"{relative}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode()
        digest.update(record)
        file_count += 1
        total_bytes += stat.st_size
        if include_content:
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
    return {
        "root": str(root),
        "mode": "content" if include_content else "metadata",
        "file_count": file_count,
        "total_bytes": total_bytes,
        "sha256": digest.hexdigest(),
    }


def write_source_fingerprint(
    output_path: str | os.PathLike[str],
    droid_root: str | os.PathLike[str],
    *,
    include_content: bool = False,
) -> dict[str, object]:
    output = (
        validate_derived_root(Path(output_path).parent, droid_root)
        / Path(output_path).name
    )
    payload = source_tree_fingerprint(droid_root, include_content=include_content)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def assert_paths_outside_source(
    paths: Iterable[str | os.PathLike[str]],
    droid_root: str | os.PathLike[str] = DEFAULT_DROID_ROOT,
) -> None:
    for path in paths:
        validate_derived_root(path, droid_root)
