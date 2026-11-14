"""Paired S/SF entrypoint; existing compute_workspace_stats.py rows retain their original imports."""
from _bootstrap import guard_gpus

guard_gpus()
from s4d.rwlearn.dsreset_runtime import install
install()
from scripts.compute_workspace_stats import main

if __name__ == "__main__":
    main()
