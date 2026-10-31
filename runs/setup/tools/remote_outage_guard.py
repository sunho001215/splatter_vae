"""Compatibility entry point: the implementation lives in ops/remote/outage_guard.py (same module object)."""

import sys
from pathlib import Path

_repo = str(Path(__file__).resolve().parents[3])  # repo on sys.path when imported
if _repo not in sys.path:
    sys.path.insert(0, _repo)

if __name__ == "__main__":  # the CLI runs the implementation as __main__ (sys.argv[0] keeps this path)
    import runpy

    runpy.run_module("ops.remote.outage_guard", run_name="__main__")
else:
    from ops.remote import outage_guard as _impl

    sys.modules[__name__] = _impl
