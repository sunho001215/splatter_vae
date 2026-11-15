"""Label-aware task evaluation for the new S/SF matrix only."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # python -I omits the script directory
from _bootstrap import guard_gpus  # noqa: E402

guard_gpus()
from s4d.rwlearn.dsreset_runtime import install
install()
from scripts.evaluate_ours_task_length import main

if __name__ == '__main__':
    main()
