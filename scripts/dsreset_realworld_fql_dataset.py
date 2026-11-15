"""Paired S/SF entrypoint; existing realworld_fql_dataset.py rows retain their original imports."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # python -I omits the script directory
from _bootstrap import guard_gpus  # noqa: E402

guard_gpus()
from s4d.rwlearn.dsreset_runtime import install
install()
from scripts.realworld_fql_dataset import main

if __name__ == "__main__":
    main()
