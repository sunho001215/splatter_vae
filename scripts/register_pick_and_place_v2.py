"""Stdout-only new-dataset registrar; never registers or launches jobs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ops.jobs.pnpv2_registration import main

if __name__ == "__main__":
    main()
