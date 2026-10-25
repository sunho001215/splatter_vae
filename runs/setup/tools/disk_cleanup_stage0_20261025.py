"""Disk cleanup stage 0 (user request 2026-10-25, same rules as the 2026-10-19 tool disk_cleanup_stage0_20261019.py):
verified disposable temporary files of the campaign, local host.

Candidates (exact paths, campaign root only, no symlink following):
  A. Live-sync partial downloads `runs/<run>/**/<file>.s4d-partial-<sha>` of runs whose registry status is `done`
     (remote runs additionally with a verified final result sync), whose published file exists, untouched > 60 min.
     A partial is a temporary download abandoned when the source changed mid-transfer (s4d/remote_access.py
     rsync_download); a finished run never reads it again.
  B. Stale pytest temporary trees (`.pytest_tmp`) of the main checkout and of worktrees, and worktree `.cache`
     copies (copies of the main checkout's third_party/torch caches made for test runs; recopyable, no download).
Every candidate is rechecked immediately before deletion (state, mtime, open files, cwd of processes).

    python -I disk_cleanup_stage0_20261025.py            # manifest only
    python -I disk_cleanup_stage0_20261025.py --execute  # recheck + delete + record
"""
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path('/home/ws/ws/splatter4d')
sys.path.insert(0, str(ROOT))
from scripts import jobs  # noqa: E402

OUT = ROOT / 'runs/setup/disk-cleanup-20261025'
EXECUTE = '--execute' in sys.argv
MIN_AGE = 3600


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


def open_paths():
    held, cwds = set(), set()
    for pid in os.listdir('/proc'):
        if not pid.isdigit():
            continue
        try:
            cwds.add(os.readlink(f'/proc/{pid}/cwd'))
        except OSError:
            pass
        try:
            for fd in os.listdir(f'/proc/{pid}/fd'):
                try:
                    held.add(os.readlink(f'/proc/{pid}/fd/{fd}'))
                except OSError:
                    pass
        except OSError:
            pass
    return held, cwds


def inside(path, root):
    real = os.path.realpath(path)
    return real == str(root) or real.startswith(str(root) + '/')


def partial_candidates(state):
    rows, retained = [], {}
    now = time.time()
    for path in (ROOT / 'runs').glob('*/**/*.s4d-partial-*'):
        run = path.relative_to(ROOT / 'runs').parts[0]
        s = state.get(run)
        reason = None
        if s is None:
            reason = 'run not in registry'
        elif s['status'] != 'done' or s.get('unknown'):
            reason = f"run status {s['status']}"
        elif s.get('host') == 'remote' and not (s.get('sync', {}).get('final') and s.get('sync', {}).get('verified')):
            reason = 'remote final sync not verified'
        elif not path.with_name(path.name.split('.s4d-partial-')[0]).exists():
            reason = 'published file missing'
        elif now - path.lstat().st_mtime < MIN_AGE:
            reason = 'modified within 60 min'
        elif path.is_symlink() or not inside(path, ROOT / 'runs'):
            reason = 'symlink or outside root'
        if reason:
            retained.setdefault(reason, []).append(str(path.relative_to(ROOT)))
            continue
        rows.append({'path': str(path.relative_to(ROOT)), 'host': 'local', 'run': run, 'category': 'partial_download',
                     'bytes': allocated(path), 'reason': 'abandoned live-sync partial of a finished run',
                     'evidence': {'registry_status': s['status'], 'host': s.get('host', 'local'),
                                  'final_sync_verified': bool(s.get('sync', {}).get('verified')),
                                  'published_file_exists': True,
                                  'age_hours': round((now - path.lstat().st_mtime) / 3600, 1)}})
    return rows, retained


def tree_candidates():
    rows = []
    trees = [ROOT / '.pytest_tmp'] + sorted((ROOT / '.worktrees').glob('*/.pytest_tmp')) + \
        sorted((ROOT / '.worktrees').glob('*/.cache'))
    for path in trees:
        if not path.exists() or path.is_symlink() or not inside(path, ROOT):
            continue
        kind = 'pytest_tmp' if path.name == '.pytest_tmp' else 'worktree_cache_copy'
        newest = max((os.lstat(os.path.join(d, f)).st_mtime for d, _, fs in os.walk(path, followlinks=False)
                      for f in fs), default=path.stat().st_mtime)
        rows.append({'path': str(path.relative_to(ROOT)), 'host': 'local',
                     'run': path.parent.name if path.parent.name != 'splatter4d' else 'main checkout',
                     'category': kind, 'bytes': allocated(path),
                     'reason': 'stale pytest temporary outputs' if kind == 'pytest_tmp' else
                     'per-worktree copy of main .cache (third_party/torch); recopyable, no download',
                     'evidence': {'newest_file_age_hours': round((time.time() - newest) / 3600, 1)}})
    return rows


state = jobs.read_registry()
partials, retained = partial_candidates(state)
trees = [r for r in tree_candidates() if r['evidence']['newest_file_age_hours'] >= 1.0 and r['bytes'] >= 50e6]
manifest = {'recorded_local': datetime.now().astimezone().isoformat(), 'stage': 0, 'host': 'local',
            'free_bytes_before': shutil.disk_usage('/home/ws').free,
            'candidates': partials + trees,
            'retained_partials_by_reason': {k: len(v) for k, v in retained.items()},
            'retained_partials_examples': {k: v[:5] for k, v in retained.items()}}
total = sum(r['bytes'] for r in manifest['candidates'])
print(f"candidates {len(manifest['candidates'])}: partials {len(partials)} "
      f"({sum(r['bytes'] for r in partials) / 1e9:.1f} GB), trees {len(trees)} "
      f"({sum(r['bytes'] for r in trees) / 1e9:.1f} GB); total {total / 1e9:.1f} GB")
print('retained partials:', manifest['retained_partials_by_reason'])
if not EXECUTE:
    (OUT / 'stage0-manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    raise SystemExit(0)

# recheck immediately before deletion
state = jobs.read_registry()
held, cwds = open_paths()
deleted, skipped = [], []
for row in manifest['candidates']:
    path = ROOT / row['path']
    why = None
    if not path.exists():
        why = 'gone'
    elif path.is_symlink() or not inside(path, ROOT):
        why = 'symlink or outside root'
    elif row['category'] == 'partial_download':
        s = state.get(row['run'], {})
        if s.get('status') != 'done' or s.get('unknown') or time.time() - path.lstat().st_mtime < MIN_AGE:
            why = 'state or mtime changed'
        elif str(path) in held:
            why = 'open by a process'
    else:
        if any(h.startswith(str(path) + '/') for h in held) or any(c == str(path) or c.startswith(str(path) + '/')
                                                                    for c in cwds):
            why = 'in use by a process'
    if why:
        skipped.append({**row, 'skipped': why})
        continue
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    deleted.append(row)
manifest.update(executed_local=datetime.now().astimezone().isoformat(), deleted=len(deleted),
                deleted_bytes=sum(r['bytes'] for r in deleted), skipped=skipped,
                free_bytes_after=shutil.disk_usage('/home/ws').free)
manifest['candidates'] = [{**r, 'deleted': r in deleted} for r in manifest['candidates']]
(OUT / 'stage0-manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
print(f"deleted {len(deleted)} ({manifest['deleted_bytes'] / 1e9:.1f} GB allocated), skipped {len(skipped)}; "
      f"free {manifest['free_bytes_before'] / 1e9:.1f} -> {manifest['free_bytes_after'] / 1e9:.1f} GB")
