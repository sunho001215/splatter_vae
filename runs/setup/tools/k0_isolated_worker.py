"""Guarded child entry for exactly nine authorized isolated k0 rows."""
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from k0_isolated_release import configure_worker_temp, prepare_worker

if len(sys.argv) < 3:
    raise SystemExit('requires authorization, exact job ID and registered arguments')
script, arguments = prepare_worker(Path(sys.argv[1]), sys.argv[2], sys.argv[3:])
configure_worker_temp(sys.argv[2])  # prepare_worker imports the cache-setting _bootstrap first
from _bootstrap import guard_gpus

guard_gpus()

sys.argv = [str(script), *arguments]
runpy.run_path(str(script), run_name='__main__')
