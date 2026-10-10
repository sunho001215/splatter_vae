"""Shared data-output safety boundary. This check never grants filesystem permission."""

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
METAWORLD_ROOT = Path("/home/ws/data/metaworld/splatter4d_v1")
POINTWORLD_ROOT = Path("/home/ws/data/pointworld_droid_sample")
DROID_CACHE_ROOT = Path("/home/ws/data/droid_pointworld_cache_sample")


def writable_path(path: str | Path, data_root: Path) -> Path:
    """Allow the new repository or the operation's explicitly authorized data root.

    Resolve symlinks before testing ancestry, including for nonexistent leaves.
    Callers remain responsible for honoring permission denials and empty outputs.
    """
    resolved = Path(path).resolve()
    if any(resolved == root or root in resolved.parents for root in (REPO.resolve(), data_root)):
        return resolved
    from s4d.gpu_guard import GPUIsolationError, validate_runtime_path

    try:
        return validate_runtime_path(resolved, repository=REPO)
    except GPUIsolationError as exc:
        raise ValueError(f"output escapes the authorized repository/data root: {resolved}") from exc
