"""Label-aware task evaluation for the new S/SF matrix only."""
from _bootstrap import guard_gpus

guard_gpus()
from s4d.rwlearn.dsreset_runtime import install
install()
from scripts.evaluate_ours_task_length import main

if __name__ == '__main__':
    main()
