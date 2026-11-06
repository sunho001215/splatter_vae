"""Fast-forward main 6a3afb3 -> 4fff862 (= d0c5c19 + REL-A b06a4a2 adapter review fixes; server roles + role-aware drivers/migration tooling + replay-cleanup fix; DROID
far4 profiles; consolidated real-world pipeline (B1 FoundationStereo + TrackEverything standard, raw-DROID estimation
route retired, adapter dropped-stereo pairing); compact 128x128 real-world training store + window loader + online SNeRL
DINO targets; real-world baseline adapters + offline FQL) and adopt its remote full-suite evidence atomically w.r.t. the
drivers (as for 31846a5 on 2026-11-05): under the local admission lock and the remote driver lock. Difference: the
server-role hold (runs/setup/remote-hold-admissions, 2026-11-06 02:3x) is ALREADY in place and must stay after the
merge (lifting it is a separate decision) — it is kept unchanged; only if no hold exists is a temporary one written and
removed. Running jobs are untouched (local jobs run in their own sessions; remote jobs in containers on pinned releases).

    .venv/bin/python -I merge_adopt_d0c5c19_20261106.py <verify job id>
"""
import fcntl
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/home/ws/ws/splatter4d')
COMMIT_PREFIX = 'c9fc448'
verify = sys.argv[1]

# Resolve beside this command, including under python -I, without changing sys.path.
from importlib import util as _import_util

_spec = _import_util.spec_from_file_location(
    '_release_adoption', Path(__file__).resolve().parents[3] / 'ops/jobs/_release_adoption.py')
_impl = _import_util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)
_impl.merge_adopt(ROOT, '2a35cfb', COMMIT_PREFIX, verify,
                  f'merge+adoption {COMMIT_PREFIX} 2026-11-06 19:2x (main ff-merge and suite evidence adoption)\n',
                  subprocess=subprocess, fcntl=fcntl, time=time, open=open, print=print)
