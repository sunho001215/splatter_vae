"""Compatibility entry point: the implementation lives in ops/resources/watch_events.py (same module object)."""

import sys
from pathlib import Path

_repo = str(Path(__file__).resolve().parents[1])  # repo on sys.path when imported
if _repo not in sys.path:
    sys.path.insert(0, _repo)

if __name__ == "__main__":  # the CLI runs the implementation as __main__ (sys.argv[0] keeps this path)
    import runpy

    runpy.run_module("ops.resources.watch_events", run_name="__main__")
else:
    from ops.resources import watch_events as _impl

    sys.modules[__name__] = _impl
