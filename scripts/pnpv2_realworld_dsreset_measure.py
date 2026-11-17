"""pick-and-place-v2 scene measurement (prep-dispatcher step): installs the new task, then the established measure."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from s4d.rwlearn.pnpv2_runtime import install  # noqa: E402

install()
import scripts.realworld_dsreset_measure as measure  # noqa: E402

if __name__ == "__main__":
    measure.main()
