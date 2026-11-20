"""Compatibility alias: unchanged ingestion validator in ops.jobs.episode_validator."""
import sys
from pathlib import Path

_repo = str(Path(__file__).resolve().parents[4])
if _repo not in sys.path:
    sys.path.insert(0, _repo)

if __name__ == "__main__":
    import runpy

    runpy.run_module("ops.jobs.episode_validator", run_name="__main__")
else:
    from ops.jobs import episode_validator as _impl

    sys.modules[__name__] = _impl
