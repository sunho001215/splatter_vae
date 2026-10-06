"""Disabled deletion entrypoint. No filesystem removal is implemented or attempted.

The required recursive open-file safety tools are missing. Their installation
and the destructive-script rewrite were denied. The obsolete cache stays intact.
The previous helper also did not implement the prescribed find/du measurements
or a reliable recursive fuser check, so retaining an executable path was unsafe.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import guard_gpus

GPU_MAPPING = guard_gpus()


def main() -> int:
    parser = argparse.ArgumentParser(description="Deletion is blocked; this entrypoint never removes data")
    parser.add_argument("--execute", action="store_true", help="rejected; no deletion authorization is available")
    parser.parse_args()
    print(f"GPU mapping: {GPU_MAPPING}")
    print("ABORT: cache deletion is disabled. See docs/RESULTS.md for the permission and safety-tool blockers.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
