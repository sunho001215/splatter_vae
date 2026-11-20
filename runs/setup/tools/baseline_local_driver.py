"""Compatibility entry point: the implementation lives in ops/campaign/baseline_local_driver.py (same module object)."""

import sys
from pathlib import Path

_repo = str(Path(__file__).resolve().parents[3])  # repo on sys.path when imported
if _repo not in sys.path:
    sys.path.insert(0, _repo)

if __name__ == "__main__":  # the CLI runs the implementation as __main__ (sys.argv[0] keeps this path)
    from runs.setup.tools.k0_isolated_release import install

    install()  # optional exact-nine grant; absent -> unchanged driver
    from runs.setup.tools.bh_isolated_release import install_all as install_bh

    install_bh()  # absent/invalid grant delegates unchanged
    import runpy

    runpy.run_module("ops.campaign.baseline_local_driver", run_name="__main__")
else:
    from ops.campaign import baseline_local_driver as _impl

    sys.modules[__name__] = _impl
