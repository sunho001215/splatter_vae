"""Private merge/adoption sequence for the five historical remote-release commands.

Importing this module performs no operational work. Dependencies come from the
executing command, whose checkout also selects this implementation.
"""


def merge_adopt(root, expected_head_prefix, commit_prefix, verify, hold_text, *, subprocess, fcntl, time, open, print):
    hold = root / 'runs/setup/remote-hold-admissions'
    existing = hold.read_text() if hold.exists() else None
    if existing is None:
        hold.write_text(hold_text)
    try:
        commit = subprocess.run(['git', '-C', str(root), 'rev-parse', 'refs/remotes/origin/splatter4d'], capture_output=True,
                                text=True, check=True).stdout.strip()
        assert commit.startswith(commit_prefix), commit
        with open(root / 'runs/setup/stage2-driver.lock', 'w') as local_lock, \
                open(root / 'runs/setup/remote-driver.lock', 'w') as remote_lock:
            t0 = time.time()
            fcntl.flock(local_lock, fcntl.LOCK_EX)
            fcntl.flock(remote_lock, fcntl.LOCK_EX)
            print(f'locks acquired after {time.time() - t0:.0f}s', flush=True)
            head = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True,
                                  check=True).stdout.strip()
            assert head.startswith(expected_head_prefix), head
            subprocess.run(['git', '-C', str(root), 'merge', '--ff-only', '-q', commit], check=True)
            print('merged', subprocess.run(['git', '-C', str(root), 'rev-parse', '--short', 'HEAD'], capture_output=True,
                                           text=True).stdout.strip(), flush=True)
            r = subprocess.run([str(root / '.venv/bin/python'), '-I',
                                str(root / 'runs/setup/tools/adopt_remote_suite_evidence.py'), verify],
                               capture_output=True, text=True)
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
