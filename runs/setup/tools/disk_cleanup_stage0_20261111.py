"""Disk cleanup stage 0 (user directive 2026-11-11 ~17:30): dated copy of disk_cleanup_stage0_20261025.py (same rules,
tightened as in the 2026-11-06 CLEAN pass): verified disposable temporary files of the campaign, local host.

Candidates (exact paths, campaign root only, no symlink following):
  A. Live-sync partial downloads `runs/<run>/**/<file>.s4d-partial-<sha>` of runs whose registry status is `done`
     (exit 0, not unknown; remote runs additionally with a verified final result sync), whose published file exists,
     untouched >= 24 h (was 60 min). A partial is a temporary download abandoned when the source changed mid-transfer
     (ops/remote/access.py rsync_download); a finished run never reads it again.
  B. Stale pytest temporary trees (`.pytest_tmp`) of the main checkout and of worktrees whose newest file is >= 72 h
     old (was 1 h) and that hold >= 50 MB.
  Not done (10-25 category C): worktree `.cache` copies. The worktrees belong to other agents who may run tests in
  them at any time; deleting their offline caches could interfere (11-06: "agents' worktrees, not CLEAN-owned").
The registry is replayed by disk_cleanup_common_20261111.read_registry (the scheduler is not imported). Every candidate
is rechecked immediately before deletion (state, mtime, open files, cwd of processes).

    nice -n 19 ionice -c3 .venv/bin/python -I runs/setup/tools/disk_cleanup_stage0_20261111.py [--execute]
"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import disk_cleanup_common_20261111 as C  # noqa: E402

ROOT = C.ROOT
EXECUTE = '--execute' in sys.argv
ONLY = sys.argv[sys.argv.index('--category') + 1] if '--category' in sys.argv else None
PARTIAL_MIN_AGE = 24 * 3600
TREE_MIN_AGE_H = 72.0
MANIFEST = C.OUT / 'local-stage0-manifest.json'


def allocated(path):
    if path.is_symlink():
        return 0
    if path.is_file():
        return path.lstat().st_blocks * 512
    total = 0
    for dirpath, _, files in os.walk(path, followlinks=False):
        for name in files:
            try:
                total += os.lstat(os.path.join(dirpath, name)).st_blocks * 512
            except OSError:
                pass
    return total


def inside(path, root):
    real = os.path.realpath(path)
    return real == str(root) or real.startswith(str(root) + '/')


def done_ok(s):
    if s is None or s['status'] != 'done' or s.get('unknown') or s.get('code') not in (0, None):
        return False
    return s.get('host', 'local') != 'remote' or bool(s.get('sync', {}).get('final') and s.get('sync', {}).get('verified'))


def partial_candidates(state):
    rows, retained = [], {}
    now = time.time()
    for path in (ROOT / 'runs').glob('**/*.s4d-partial-*'):
        parts = path.relative_to(ROOT / 'runs').parts
        run = parts[1] if parts[0] == 'pretrain' else parts[0]
        s = state.get(run)
        reason = None
        if s is None:
            reason = 'run not in registry'
        elif not done_ok(s):
            reason = f"run status {s['status']} / sync not verified"
        elif not path.with_name(path.name.split('.s4d-partial-')[0]).exists():
            reason = 'published file missing'
        elif now - path.lstat().st_mtime < PARTIAL_MIN_AGE:
            reason = 'modified within 24 h'
        elif path.is_symlink() or not path.is_file() or not inside(path, ROOT / 'runs'):
            reason = 'symlink, not a file or outside root'
        if reason:
            retained.setdefault(reason, []).append(str(path))
            continue
        st = path.lstat()
        rows.append({'path': str(path), 'host': 'local', 'run': run, 'category': 'partial_download',
                     'bytes': allocated(path), 'owner_uid': st.st_uid,
                     'reason': 'abandoned live-sync partial of a finished run',
                     'evidence': {'registry_status': s['status'], 'code': s.get('code'), 'host': s.get('host', 'local'),
                                  'final_sync_verified': bool(s.get('sync', {}).get('verified')),
                                  'published_file_exists': True, 'age_hours': round((now - st.st_mtime) / 3600, 1),
                                  'mtime': st.st_mtime, 'size': st.st_size}})
    return rows, retained


def scan_tree(path):
    """(newest mtime incl. the root, allocated bytes) walked fail-closed: any walk error, a mount point inside or an
    entry not owned by the root's owner raises (Codex review 2026-11-11 finding 4)."""
    root = path.lstat()
    newest, total = root.st_mtime, root.st_blocks * 512
    if root.st_uid != os.getuid():
        raise ValueError(f'{path} not owned by the executing user')
    if root.st_dev != path.parent.lstat().st_dev or (m := C.mount_inside(str(path))):
        raise ValueError(f'mount point at or inside {path}')

    def fail(err):
        raise err
    for d, ds, fs in os.walk(path, onerror=fail, followlinks=False):
        for f in fs + ds:
            st = os.lstat(os.path.join(d, f))
            if st.st_dev != root.st_dev:
                raise ValueError(f'mount point inside {path}')
            if st.st_uid != os.getuid():
                raise ValueError(f'entry not owned by the executing user inside {path}')
            newest = max(newest, st.st_mtime)
            total += st.st_blocks * 512
    return newest, total


def worktree_in_use(tree):
    """Any process whose cwd is inside the worktree (or the main checkout for its own .pytest_tmp), that holds a file
    under it, or names it on its command line (Codex review finding 2: a test run from the worktree root)."""
    base = str(tree.parent)
    cmd, held, cwds = C.proc_scan(strict=True)
    if base == str(ROOT):  # main checkout: only the scratch tree itself
        base = str(tree)
    hits = [c for c in cwds if c == base or c.startswith(base + '/')]
    hits += [t for t in held if t == base or t.startswith(base + '/')]
    hits += [p for p, c in cmd.items() if base in c]
    return hits[:3]


def tree_candidates():
    rows = []
    for path in [ROOT / '.pytest_tmp'] + sorted((ROOT / '.worktrees').glob('*/.pytest_tmp')):
        if not path.exists() or path.is_symlink() or not inside(path, ROOT):
            continue
        try:
            newest, _ = scan_tree(path)
        except (OSError, ValueError):
            continue
        rows.append({'path': str(path), 'host': 'local',
                     'run': path.parent.name if path.parent.name != 'splatter4d' else 'main checkout',
                     'category': 'pytest_tmp', 'bytes': allocated(path), 'owner_uid': path.lstat().st_uid,
                     'reason': 'stale pytest temporary outputs (pytest basetemp; recreated by every test run)',
                     'evidence': {'newest_file_age_hours': round((time.time() - newest) / 3600, 1),
                                  'newest_mtime': newest}})
    return rows


def plan():
    state = C.read_registry()
    partials, retained = partial_candidates(state)
    trees = tree_candidates()
    keep_trees = [r for r in trees if not (r['evidence']['newest_file_age_hours'] >= TREE_MIN_AGE_H and r['bytes'] >= 50e6)]
    trees = [r for r in trees if r not in keep_trees]
    return {'recorded_local': C.now_iso(), 'stage': 0, 'host': 'local', 'tool': __file__,
            'free_bytes_before': shutil.disk_usage('/home/ws').free,
            'candidates': partials + trees,
            'retained_partials_by_reason': {k: len(v) for k, v in retained.items()},
            'retained_partials_examples': {k: v[:5] for k, v in retained.items()},
            'retained_trees': [{'path': r['path'], 'bytes': r['bytes'], 'age_h': r['evidence']['newest_file_age_hours']}
                               for r in keep_trees]}


LOCK = ROOT / 'runs/remote/access/download-budget.lock'  # ops/remote/access.py download_slot (every download)


def execute(manifest):
    """Recheck each candidate immediately before deleting it (Codex review 2026-11-11 findings 2 and 4): registry
    re-read at most every 60 s, a fresh process scan per item; partials are deleted while holding the shared download
    lock (no transfer can run), must still be regular files with the manifest's size and mtime next to their published
    file; scratch trees must be unchanged (newest mtime incl. root), on one filesystem, owned by one user, and no
    process may work in their worktree."""
    import fcntl
    state, t_state = C.read_registry(), time.time()
    deleted, skipped = [], []
    log = (C.OUT / 'executed-actions.jsonl').open('a')
    lock = LOCK.open('a')
    have_lock = False
    for _ in range(60 if ONLY in (None, 'partial_download') else 0):  # up to 2 min, then partials are skipped
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            have_lock = True
            break
        except BlockingIOError:
            time.sleep(2)
    try:
        for row in sorted(manifest['candidates'], key=lambda r: r['category'] != 'partial_download'):
            if ONLY and row['category'] != ONLY:
                continue
            if row['category'] != 'partial_download' and have_lock:  # hold the download lock only for partials
                fcntl.flock(lock, fcntl.LOCK_UN)
                have_lock = False
            if time.time() - t_state > 60:
                state, t_state = C.read_registry(), time.time()
            path = Path(row['path'])
            why = None
            if not os.path.lexists(path):
                why = 'gone'
            elif path.is_symlink() or not inside(path, ROOT):
                why = 'symlink or outside root'
            elif row['category'] == 'partial_download':
                st = path.lstat()
                _, held, _ = C.proc_scan(strict=True)
                if not have_lock:
                    why = 'download lock busy'
                elif not path.is_file() or st.st_size != row['evidence']['size'] or st.st_mtime != row['evidence']['mtime']:
                    why = 'not the same regular file (type, size or mtime changed)'
                elif not done_ok(state.get(row['run'])) or time.time() - st.st_mtime < PARTIAL_MIN_AGE:
                    why = 'state or age changed'
                elif not path.with_name(path.name.split('.s4d-partial-')[0]).is_file():
                    why = 'published file missing'
                elif str(path) in held:
                    why = 'open by a process'
            else:
                try:
                    newest, total = scan_tree(path)
                except (OSError, ValueError) as exc:
                    newest, total, why = None, None, f'scan failed: {exc}'[:200]
                if why is None and (newest > row['evidence']['newest_mtime'] + 1e-6
                                    or time.time() - newest < TREE_MIN_AGE_H * 3600):
                    why = 'tree modified since the manifest'
                if why is None and abs(total - row['bytes']) > 0.01 * row['bytes'] + 1048576:
                    why = f'allocated size differs from the manifest ({total} vs {row["bytes"]})'
                if why is None and (busy := worktree_in_use(path)):
                    why = f'worktree in use: {busy}'
            if why:
                skipped.append({**row, 'skipped': why})
                log.write(json.dumps({'time': time.time(), 'host': 'local', 'action': 'skip', 'category': row['category'],
                                      'path': row['path'], 'reason': why}) + '\n')
                continue
            errors = []
            if path.is_dir():
                shutil.rmtree(path, onerror=lambda fn, p, exc: errors.append(f'{p}: {exc[1]}'))
            else:
                path.unlink()
            action = 'delete' if not errors and not os.path.lexists(path) else 'partial'
            if action == 'delete':
                deleted.append(row['path'])
            else:
                skipped.append({**row, 'skipped': f'partial removal: {errors[:3]}'})
            log.write(json.dumps({'time': time.time(), 'host': 'local', 'action': action, 'category': row['category'],
                                  'path': row['path'], 'allocated_bytes': row['bytes'], 'owner_uid': row['owner_uid'],
                                  'reason': row['reason'], 'evidence': row['evidence'], 'errors': errors[:3]}) + '\n')
            log.flush()
    finally:
        if have_lock:
            fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
        log.close()
    return deleted, skipped


if __name__ == '__main__':
    C.OUT.mkdir(parents=True, exist_ok=True)
    if not EXECUTE:
        m = plan()
        cats = {}
        for r in m['candidates']:
            cats.setdefault(r['category'], [0, 0])
            cats[r['category']][0] += 1
            cats[r['category']][1] += r['bytes']
        print({k: (n, round(b / 1e9, 2)) for k, (n, b) in cats.items()})
        print('retained partials:', m['retained_partials_by_reason'])
        MANIFEST.write_text(json.dumps(m, indent=1) + '\n')
        raise SystemExit(0)
    m = json.loads(MANIFEST.read_text())
    free0 = shutil.disk_usage('/home/ws').free
    deleted, skipped = execute(m)
    m['candidates'] = [{**r, 'deleted': r['path'] in set(deleted)} for r in m['candidates']]
    m.update(executed_local=C.now_iso(), deleted=len(deleted),
             deleted_bytes=sum(r['bytes'] for r in m['candidates'] if r['deleted']), skipped=skipped,
             free_bytes_before_execute=free0, free_bytes_after=shutil.disk_usage('/home/ws').free)
    MANIFEST.write_text(json.dumps(m, indent=1) + '\n')
    print(f"deleted {len(deleted)} ({m['deleted_bytes'] / 1e9:.2f} GB allocated), skipped {len(skipped)}; "
          f"free {free0 / 1e9:.1f} -> {m['free_bytes_after'] / 1e9:.1f} GB")
