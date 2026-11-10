"""Resumable downloads of DROID RLDS 1.0.1, PointWorld-DROID and DROID-3D onto the remote data disk (remote side).

Runs with the remote host's own python3 (standard library only) from a deployed copy:

    python3 -I /data_8T/kaist/sunho/tools/<sha>/scripts/datasets/droid_datasets.py {plan,storage,configure,start,run,
        status,stop,adopt,retry} [--root /data_8T/kaist/sunho/datasets]

See ops/datasets/cli.py and docs/dataset_downloads.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ops.datasets.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
