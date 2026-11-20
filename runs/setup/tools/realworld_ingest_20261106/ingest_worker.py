"""Compatibility entry point: the ingestion worker lives in ops.datasets.ingest_worker."""
import sys
from pathlib import Path

_repo = str(Path(__file__).resolve().parents[4])
if _repo not in sys.path:
    sys.path.insert(0, _repo)

if __name__ == "__main__":
    from ops.datasets.ingest_worker import main

    raise SystemExit(main())
else:
    from ops.datasets import ingest_worker as _impl
    from ops.datasets.ingest_worker import set_stage  # importlib file loaders also access this attribute

    sys.modules[__name__] = _impl
