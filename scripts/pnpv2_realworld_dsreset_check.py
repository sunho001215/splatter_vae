"""pick-and-place-v2 readiness barrier (wave gate row); --stage prep = RGB view, --stage calib = geometry view."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # python -I omits the script directory
from _bootstrap import guard_gpus  # noqa: E402

guard_gpus()
from s4d.rwlearn.pnpv2_entry import readiness_main  # noqa: E402

if __name__ == "__main__":
    readiness_main()
