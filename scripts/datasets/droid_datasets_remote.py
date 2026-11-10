"""Local control of the remote dataset downloader through ops.remote.access (deploy, latency probe, remote commands).

    .venv/bin/python -I scripts/datasets/droid_datasets_remote.py deploy --sha <commit>
    .venv/bin/python -I scripts/datasets/droid_datasets_remote.py latency [--n 8]
    .venv/bin/python -I scripts/datasets/droid_datasets_remote.py remote --sha <commit> -- status
    .venv/bin/python -I scripts/datasets/droid_datasets_remote.py stage-upload --kib 600 <file> [...]

See ops/datasets/control.py and docs/dataset_downloads.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ops.datasets.control import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
