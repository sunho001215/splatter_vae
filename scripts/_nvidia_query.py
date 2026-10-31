"""Compatibility alias: the implementation lives in ops/resources/nvidia_query.py (same module object)."""

import sys
from pathlib import Path

_repo = str(Path(__file__).resolve().parents[1])  # repo on sys.path when imported
if _repo not in sys.path:
    sys.path.insert(0, _repo)

if __name__ == "__main__":  # the CLI runs the implementation as __main__ (sys.argv[0] keeps this path)
    import runpy

    runpy.run_module("ops.resources.nvidia_query", run_name="__main__")
else:
    from ops.resources import nvidia_query as _impl

    sys.modules[__name__] = _impl
