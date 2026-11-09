"""Fast-forward the main checkout a14bcfe -> 57127d7 (refactor4 integration: S2M2 and PPO/Dreamer/FlashSAC retired,
real-world calib/stages/robotics packages, single-branch `main` release tooling; merged with the unrelated old
origin/main history via -s ours), adopt its remote full-suite evidence, and move the checkout from branch `splatter4d`
to branch `main` at the same commit (identical tree: working files, local modifications and untracked runtime state stay
as they are). Same locking as every earlier adoption: the local admission lock and the remote driver lock; a
pre-existing remote role hold is kept unchanged, otherwise a temporary one is written and removed. Running jobs are
untouched (local jobs run in their own sessions; remote jobs in containers on pinned releases).

Extra steps of this adoption (integration review finding D1): the submodules third_party/{flashsac,dreamer-pytorch,
stable-baselines3} were removed from the index; a fast-forward leaves their checkouts as untracked directories, which
sync-code would refuse as dirty sources, so they are deleted here (their module repositories under .git/modules stay).

    .venv/bin/python -I runs/setup/tools/merge_adopt_57127d7_20261109.py <verify job id>
"""
import fcntl
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/home/ws/ws/splatter4d')
COMMIT = '57127d7b7afb4f51a0fa854763afcc8c923e74d9'
BASE = 'a14bcfe'
RETIRED_SUBMODULES = ('third_party/flashsac', 'third_party/dreamer-pytorch', 'third_party/stable-baselines3')
verify = sys.argv[1]


def git(*args, check=True):
    return subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True, text=True, check=check).stdout.strip()


hold = ROOT / 'runs/setup/remote-hold-admissions'
existing = hold.read_text() if hold.exists() else None
if existing is None:
    hold.write_text('merge+adoption 57127d7 2026-11-09 (ff-merge, suite evidence adoption, branch splatter4d -> main)\n')
try:
    git('fetch', '-q', 'origin', 'main', 'splatter4d')
    assert git('rev-parse', 'refs/remotes/origin/splatter4d') == COMMIT, 'origin/splatter4d is not the released commit'
    assert git('rev-parse', 'refs/remotes/origin/main') == COMMIT, 'origin/main is not the released commit'
    with open(ROOT / 'runs/setup/stage2-driver.lock', 'w') as local_lock, \
            open(ROOT / 'runs/setup/remote-driver.lock', 'w') as remote_lock:
        t0 = time.time()
        fcntl.flock(local_lock, fcntl.LOCK_EX)
        fcntl.flock(remote_lock, fcntl.LOCK_EX)
        print(f'locks acquired after {time.time() - t0:.0f}s', flush=True)
        head = git('rev-parse', 'HEAD')
        assert head.startswith(BASE), head
        assert git('symbolic-ref', '--short', 'HEAD') == 'splatter4d'
        git('merge', '--ff-only', '-q', COMMIT)
        print('merged', git('rev-parse', '--short', 'HEAD'), flush=True)
        for path in RETIRED_SUBMODULES:
            target = ROOT / path
            assert not git('ls-files', '--', path), f'{path} is still tracked'
            if target.exists():
                shutil.rmtree(target)
                print('removed untracked leftover', path, flush=True)
        dirty_sources = [line for line in git('status', '--porcelain', '--untracked-files=normal').splitlines()
                         if line.startswith('?? third_party/')]
        assert not dirty_sources, dirty_sources
        r = subprocess.run([str(ROOT / '.venv/bin/python'), '-I', str(ROOT / 'runs/setup/tools/adopt_remote_suite_evidence.py'),
                            verify], capture_output=True, text=True)
        print(r.stdout[-2000:], r.stderr[-2000:], flush=True)
        if r.returncode != 0:  # fail-safe: the local drivers refuse admission while the evidence does not match
            raise SystemExit(f'adoption failed ({r.returncode}); checkout is at the merged commit on splatter4d; evidence '
                             'unchanged; local admissions stay refused until resolved')
        print('adopted at', time.strftime('%T'), flush=True)
        # Same commit, same tree: switching branches rewrites no working file.
        git('branch', '-f', 'main', COMMIT)
        git('switch', '-q', 'main')
        git('branch', '--set-upstream-to=origin/main', 'main')
        assert git('rev-parse', 'HEAD') == COMMIT and git('symbolic-ref', '--short', 'HEAD') == 'main'
        print('checkout now on branch main at', COMMIT[:7], flush=True)
finally:
    if existing is None:
        hold.unlink(missing_ok=True)
        print('temporary hold removed', flush=True)
    else:
        assert hold.read_text() == existing, 'the pre-existing role hold changed during the window'
        print('pre-existing role hold kept unchanged', flush=True)
