"""Compatibility alias: the implementation lives in s4d/realworld/stages/motion.py (same module object)."""

import sys

if __name__ == "__main__":
    import runpy
    from pathlib import Path

    _repo = str(Path(__file__).resolve().parents[2])
    if _repo not in sys.path:
        sys.path.insert(0, _repo)
    runpy.run_module("s4d.realworld.stages.motion", run_name="__main__")
else:
    from s4d.realworld.stages import motion as _impl

    sys.modules[__name__] = _impl
