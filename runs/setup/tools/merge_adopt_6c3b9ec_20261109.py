"""Fast-forward main 587e2f2 -> 6c3b9ec (calibration release code rw-calib-rel-20261108: rig_loma release documents, rig-mode
fixedness exception, calibration_validated builder, per-camera depth scale downstream, 2F-85 gripper map, fail-closed visual meshes) and
adopt its remote full-suite evidence atomically w.r.t. the drivers (same procedure as c1f41ed on 2026-11-07): under the
local admission lock and the remote driver lock. A pre-existing remote role hold is kept unchanged; only if none exists is
a temporary one written and removed. Running jobs are untouched (local jobs run in their own sessions; remote jobs in
containers on pinned releases).

    .venv/bin/python -I runs/setup/tools/merge_adopt_6c3b9ec_20261109.py <verify job id>
"""
import fcntl
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/home/ws/ws/splatter4d')
COMMIT_PREFIX = '6c3b9ec'
verify = sys.argv[1]
hold = ROOT / 'runs/setup/remote-hold-admissions'
existing = hold.read_text() if hold.exists() else None
if existing is None:
    hold.write_text(f'merge+adoption {COMMIT_PREFIX} 2026-11-08 (main ff-merge and suite evidence adoption)\n')
try:
    commit = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'refs/remotes/origin/splatter4d'], capture_output=True,
                            text=True, check=True).stdout.strip()
    assert commit.startswith(COMMIT_PREFIX), commit
    with open(ROOT / 'runs/setup/stage2-driver.lock', 'w') as local_lock, \
            open(ROOT / 'runs/setup/remote-driver.lock', 'w') as remote_lock:
        t0 = time.time()
        fcntl.flock(local_lock, fcntl.LOCK_EX)
        fcntl.flock(remote_lock, fcntl.LOCK_EX)
        print(f'locks acquired after {time.time() - t0:.0f}s', flush=True)
        head = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], capture_output=True, text=True,
                              check=True).stdout.strip()
        assert head.startswith('587e2f2'), head
        subprocess.run(['git', '-C', str(ROOT), 'merge', '--ff-only', '-q', commit], check=True)
        print('merged', subprocess.run(['git', '-C', str(ROOT), 'rev-parse', '--short', 'HEAD'], capture_output=True,
                                       text=True).stdout.strip(), flush=True)
        r = subprocess.run([str(ROOT / '.venv/bin/python'), '-I', str(ROOT / 'runs/setup/tools/adopt_remote_suite_evidence.py'),
                            verify], capture_output=True, text=True)
        print(r.stdout[-2000:], r.stderr[-2000:], flush=True)
        if r.returncode != 0:  # fail-safe: the local drivers refuse admission while the evidence does not match
            raise SystemExit(f'adoption failed ({r.returncode}); main is at the merged commit; evidence unchanged; '
                             'local admissions stay refused until resolved')
        print('adopted at', time.strftime('%T'), flush=True)
finally:
    if existing is None:
        hold.unlink(missing_ok=True)
        print('temporary hold removed', flush=True)
    else:
        assert hold.read_text() == existing, 'the pre-existing role hold changed during the window'
        print('pre-existing role hold kept unchanged', flush=True)
