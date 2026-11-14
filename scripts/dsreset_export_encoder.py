"""Paired S/SF entrypoint; existing export_encoder.py rows retain their original imports."""
from _bootstrap import guard_gpus

guard_gpus()
from s4d.rwlearn.dsreset_runtime import install
install()
from scripts.export_encoder import main

if __name__ == "__main__":
    main()
