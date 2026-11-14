"""Paired S/SF entrypoint; existing train_reviwo.py rows retain their original imports."""
from _bootstrap import guard_gpus

guard_gpus()
from s4d.rwlearn.dsreset_runtime import install
install()
from scripts.train_reviwo import main

if __name__ == "__main__":
    main()
