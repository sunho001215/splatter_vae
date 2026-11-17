"""pick-and-place-v2 entrypoint for scripts/train_sincro.py: installs the new task process-locally, same argv."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # python -I omits the script directory
from _bootstrap import guard_gpus  # noqa: E402

guard_gpus()
from s4d.rwlearn.pnpv2_entry import run  # noqa: E402

if __name__ == "__main__":
    run("scripts.train_sincro")
