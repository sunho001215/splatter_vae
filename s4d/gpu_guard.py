"""Compatibility alias: the implementation lives in ops/resources/gpu_guard.py (same module object)."""

import sys

if __name__ == "__main__":  # direct execution (also under python -I) runs the implementation as __main__
    import runpy
    from pathlib import Path

    _repo = str(Path(__file__).resolve().parents[1])
    if _repo not in sys.path:
        sys.path.insert(0, _repo)
    runpy.run_module("ops.resources.gpu_guard", run_name="__main__")
else:
    from ops.resources import gpu_guard as _impl

    sys.modules[__name__] = _impl
