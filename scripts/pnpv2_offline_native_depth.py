"""PNP offline native depth with the committed calibration acceptance override; no robot I/O."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import guard_gpus  # noqa: E402

guard_gpus()
from s4d.rwlearn.pnpv2_offline import native_main  # noqa: E402

if __name__ == '__main__':
    native_main()
