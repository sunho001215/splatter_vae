"""Fast-forward the main checkout (branch main) d9b759c -> f3964f4 (covering 98-episode gate amendment, recording-period
rig documents for the calibration release procedure, register_realworld_newtasks fix; user decisions 2026-11-10 ~13:00)
and adopt its remote full-suite evidence, atomically w.r.t. the drivers: under the local admission lock and the remote
driver lock; a pre-existing remote role hold is kept unchanged, otherwise a temporary one is written and removed.
Single-branch procedure: fetch only origin main, assert the candidate and the accepted base, fast-forward, adopt, assert
branch/upstream. Running jobs are untouched (local jobs run in their own sessions; remote jobs in containers on pinned
releases).

    .venv/bin/python -I runs/setup/tools/merge_adopt_f3964f4_20261110.py <verify job id>
"""
import fcntl
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/home/ws/ws/splatter4d')
COMMIT = 'f3964f406be829b27c5d4b35f60d60b8a9a9a05d'
BASE = 'd9b759c275ace6ec0fe06d085d3662a50b119e69'
verify = sys.argv[1]


def git(*args, check=True):
    return subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True, text=True, check=check).stdout.strip()


hold = ROOT / 'runs/setup/remote-hold-admissions'
existing = hold.read_text() if hold.exists() else None
if existing is None:
    hold.write_text('merge+adoption f3964f4 2026-11-10 (main ff-merge and suite evidence adoption)\n')
try:
    git('fetch', '-q', 'origin', 'main')
    assert git('rev-parse', 'refs/remotes/origin/main') == COMMIT, 'origin/main is not the released commit'
    with open(ROOT / 'runs/setup/stage2-driver.lock', 'w') as local_lock, \
            open(ROOT / 'runs/setup/remote-driver.lock', 'w') as remote_lock:
        t0 = time.time()
        fcntl.flock(local_lock, fcntl.LOCK_EX)
        fcntl.flock(remote_lock, fcntl.LOCK_EX)
        print(f'locks acquired after {time.time() - t0:.0f}s', flush=True)
        assert git('rev-parse', 'HEAD') == BASE, git('rev-parse', 'HEAD')
        assert git('symbolic-ref', '--short', 'HEAD') == 'main'
        assert subprocess.run(['git', '-C', str(ROOT), 'merge-base', '--is-ancestor', BASE, COMMIT]).returncode == 0
        git('merge', '--ff-only', '-q', COMMIT)
        print('merged', git('rev-parse', '--short', 'HEAD'), flush=True)
        r = subprocess.run([str(ROOT / '.venv/bin/python'), '-I', str(ROOT / 'runs/setup/tools/adopt_remote_suite_evidence.py'),
                            verify], capture_output=True, text=True)
        print(r.stdout[-2000:], r.stderr[-2000:], flush=True)
        if r.returncode != 0:  # fail-safe: the local drivers refuse admission while the evidence does not match
            raise SystemExit(f'adoption failed ({r.returncode}); main is at the merged commit; evidence unchanged; '
                             'local admissions stay refused until resolved')
        assert git('rev-parse', 'HEAD') == COMMIT and git('symbolic-ref', '--short', 'HEAD') == 'main'
        assert git('rev-parse', '--abbrev-ref', 'main@{upstream}') == 'origin/main'
        print('adopted at', time.strftime('%T'), flush=True)
finally:
    if existing is None:
        hold.unlink(missing_ok=True)
        print('temporary hold removed', flush=True)
    else:
        assert hold.read_text() == existing, 'the pre-existing role hold changed during the window'
        print('pre-existing role hold kept unchanged', flush=True)
