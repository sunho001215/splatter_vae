"""Paired S/SF entrypoint; existing realworld_native_depth.py rows retain their original imports."""
from _bootstrap import guard_gpus

guard_gpus()
from s4d.rwlearn.dsreset_runtime import install
install()
from scripts.realworld_native_depth import main

if __name__ == "__main__":
    main()
