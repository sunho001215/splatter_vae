"""Command entrypoint for the host-local admitted preprocessing queue."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ops.jobs.rwprep_mig import main

if __name__ == '__main__':
    main()
