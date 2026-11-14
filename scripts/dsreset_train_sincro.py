"""Paired S/SF entrypoint; existing train_sincro.py rows retain their original imports."""
from _bootstrap import guard_gpus

guard_gpus()
from s4d.rwlearn.dsreset_runtime import install
install()
from scripts.train_sincro import main

if __name__ == "__main__":
    main()
