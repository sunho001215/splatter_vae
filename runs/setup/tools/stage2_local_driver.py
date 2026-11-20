"""Compatibility entry point: the implementation lives in ops/campaign/stage2_local_driver.py (same module object)."""

import sys
from pathlib import Path

_repo = str(Path(__file__).resolve().parents[3])  # repo on sys.path when imported
if _repo not in sys.path:
    sys.path.insert(0, _repo)

if __name__ == "__main__":  # the CLI runs the implementation as __main__ (sys.argv[0] keeps this path)
    from runs.setup.tools.k0_isolated_release import install

    install()  # validate before the implementation takes its admission lock
    from runs.setup.tools.bh_isolated_release import install_all as install_bh

    install_bh()  # absent/invalid grant delegates unchanged
    import runpy

    from ops.jobs.scheduler import input_verification_pass  # per-pass SHA reuse (2026-11-16)

    with input_verification_pass():
        runpy.run_module("ops.campaign.stage2_local_driver", run_name="__main__")
else:
    from ops.campaign import stage2_local_driver as _impl

    sys.modules[__name__] = _impl
