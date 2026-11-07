"""Fast-forward main to a released commit and adopt LOCAL full-suite evidence (scripts/run_tests.py run in a clean
worktree checked out at that commit), atomically w.r.t. the drivers: under the local admission lock and the remote driver
lock (as merge_adopt_4fff862_20261106.py, which adopted remote evidence). The remote role hold is left exactly as it is.

Checks: origin/splatter4d == <commit>; main HEAD == <expected head>; the worktree is clean at <commit> apart from its
docs/ evidence; the evidence passed the full suite with zero failures/errors; after the merge, the main checkout's source
fingerprint (scripts/_bootstrap.source_fingerprints, the scheduler's tested-source rule) equals the evidence's
source_sha256. Fail-safe: on any mismatch the previous docs/tests.json is restored, so the drivers keep refusing local
admission until resolved (the daemon re-executes only onto tested sources).

    .venv/bin/python -I merge_adopt_local_20261106.py <commit> <expected main head prefix> <worktree>
"""
import fcntl
import importlib.util
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/home/ws/ws/splatter4d')
commit, head_prefix, worktree = sys.argv[1], sys.argv[2], Path(sys.argv[3])


def git(*args, cwd=ROOT):
    return subprocess.run(['git', '-C', str(cwd), *args], capture_output=True, text=True, check=True).stdout.strip()


evidence_path = worktree / 'docs/tests.json'
evidence = json.loads(evidence_path.read_text())
assert git('rev-parse', 'HEAD', cwd=worktree) == commit, 'worktree not at the commit'
status = subprocess.run(['git', '-C', str(worktree), 'status', '--porcelain'], capture_output=True, text=True,
                        check=True).stdout  # not stripped: porcelain lines start with a two-column status
dirty = [line for line in status.splitlines() if not line[3:].startswith('docs/')]
assert not dirty, dirty
assert evidence.get('full_suite_passed') is True, 'suite did not pass'
counts = evidence.get('counts', {})
assert counts.get('failed', 0) == 0 and counts.get('error', 0) == 0 and counts.get('passed', 0) > 1500, counts
assert git('rev-parse', 'refs/remotes/origin/splatter4d') == commit, 'origin/splatter4d is not the release commit'

with open(ROOT / 'runs/setup/stage2-driver.lock', 'w') as local_lock, \
        open(ROOT / 'runs/setup/remote-driver.lock', 'w') as remote_lock:
    t0 = time.time()
    fcntl.flock(local_lock, fcntl.LOCK_EX)
    fcntl.flock(remote_lock, fcntl.LOCK_EX)
    print(f'locks acquired after {time.time() - t0:.0f}s', flush=True)
    head = git('rev-parse', 'HEAD')
    assert head.startswith(head_prefix), head
    subprocess.run(['git', '-C', str(ROOT), 'merge', '--ff-only', '-q', commit], check=True)
    print('merged', git('rev-parse', '--short', 'HEAD'), flush=True)
    target = ROOT / 'docs/tests.json'
    backup = ROOT / f'runs/setup/tests.json.before-{commit[:7]}'
    shutil.copy2(target, backup)
    spec = importlib.util.spec_from_file_location('_bootstrap_main', ROOT / 'scripts/_bootstrap.py')
    sys.path.insert(0, str(ROOT / 'scripts'))
    sys.path.insert(0, str(ROOT))
    boot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(boot)
    current = boot.source_fingerprints()
    if current != evidence.get('source_sha256'):
        diff = sorted(k for k in set(current) | set(evidence.get('source_sha256', {}))
                      if current.get(k) != evidence['source_sha256'].get(k))
        raise SystemExit(f'fingerprint mismatch after merge ({len(diff)} files, e.g. {diff[:8]}); evidence NOT adopted; '
                         f'main is at {commit[:7]}; local admissions stay refused until resolved')
    tmp = target.with_suffix('.adopt.tmp')
    shutil.copy2(evidence_path, tmp)
    tmp.replace(target)
    for extra in sorted((worktree / 'docs').glob('tests*')):  # junit / companion evidence files written by the run
        if extra.name != 'tests.json' and extra.is_file():
            shutil.copy2(extra, ROOT / 'docs' / extra.name)
    print('adopted local evidence', counts, 'at', time.strftime('%T'), flush=True)
