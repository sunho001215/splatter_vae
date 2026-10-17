"""Evict page cache of files nothing reads again, so the local full cgroup charge (150 GB rule) does not grow with them.
Files are neither modified nor deleted; only their clean cached pages are released (POSIX_FADV_DONTNEED); a later
read simply comes from disk again.
1. Leftover live-sync partial downloads (runs/**/*.s4d-partial-*) not modified for 10 minutes.
2. (2026-10-15) Finished RL runs: the policy snapshots and the saved replay of runs whose trainer and evaluation
   companion are both done in the registry (runs/<id>/snapshots/*, runs/<id>/replay/*). Datasets, active runs and
   pretraining checkpoints are never touched.
3. (2026-10-16) Files nothing reads again after they are written: source worktrees and their test outputs
   (.worktrees/, runs/nv/, runs/rel/; their .venv links are skipped), release bundles (runs/remote/bundles/) and pulled
   offline W&B folders (runs/<id>/wandb/), which `wandb sync` re-reads from disk.
4. (2026-10-17) Active RL runs: policy snapshots whose step the evaluation companion has already recorded
   (runs/<id>/snapshots/step_N.pt with N <= the last eval.jsonl step), and any snapshot not written for 10 minutes
   (the trainer never re-reads it; a lagging evaluator reads it once from disk).
5. (2026-10-17) Meta-World datasets (/home/ws/data/metaworld/splatter4d_v1, never other data roots) while no active
   local job reads datasets (DrM trainers/evaluators simulate live); a later reader re-reads them from disk.
6. (2026-10-17) Only while the full charge exceeds the 130 GB steady target: the replay files of active local RL
   runs. Their clean cached pages are dropped; sampling re-reads the needed frames from disk (no data change)."""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path('/home/ws/ws/splatter4d')
sys.path.insert(0, str(ROOT))
from scripts import jobs  # noqa: E402


def evict(path):
    stat = path.stat()
    if not path.is_file() or path.is_symlink():
        return 0
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    return stat.st_size


before = int(Path('/sys/fs/cgroup/memory.current').read_text())
count = size = 0
for path in (ROOT / 'runs').rglob('*.s4d-partial-*'):
    try:
        if time.time() - path.stat().st_mtime < 600:
            continue
        size += evict(path)
        count += 1
    except OSError as exc:
        print(f'skip {path}: {exc}', file=sys.stderr)
state = jobs.read_registry()
finished = [jid for jid, s in state.items() if not jid.startswith('__') and not jid.endswith('-eval')
            and s.get('status') == jobs.DONE and state.get(jid + '-eval', {}).get('status') == jobs.DONE
            and (ROOT / 'runs' / jid / 'final.json').is_file()]
rl_files = rl_bytes = 0
for jid in finished:
    for sub in ('snapshots', 'replay'):
        folder = ROOT / 'runs' / jid / sub
        if not folder.is_dir() or folder.is_symlink():
            continue
        for path in folder.iterdir():
            try:
                rl_bytes += evict(path)
                rl_files += 1
            except OSError as exc:
                print(f'skip {path}: {exc}', file=sys.stderr)
stale_files = 0
for root in [ROOT / '.worktrees', ROOT / 'runs/nv', ROOT / 'runs/rel', ROOT / 'runs/remote/bundles',
             *(ROOT / 'runs').glob('*/wandb')]:
    for dirpath, dirnames, names in os.walk(root, followlinks=False):
        dirnames[:] = [d for d in dirnames if d != '.venv']
        for name in names:
            try:
                path = Path(dirpath) / name
                if not path.is_symlink():
                    evict(path)
                    stale_files += 1
            except OSError:
                pass
active_snapshot_files = 0
for jid, s in state.items():
    if s.get('status') not in jobs.ACTIVE or s.get('host', 'local') != 'local' or jid.endswith('-eval'):
        continue
    folder = ROOT / 'runs' / jid / 'snapshots'
    evals = [ROOT / 'runs' / f'{jid}-eval' / 'eval.jsonl', ROOT / 'runs' / jid / 'eval.jsonl']
    path = next((e for e in evals if e.is_file()), None)
    if not folder.is_dir() or folder.is_symlink():
        continue
    try:
        done = -1 if path is None else max(
            (json.loads(line)['step'] for line in path.read_text().splitlines() if line.strip()), default=-1)
    except (OSError, ValueError, KeyError):
        done = -1
    for snap in folder.glob('step_*.pt'):
        try:
            if int(snap.stem.split('_')[1]) <= done or time.time() - snap.stat().st_mtime > 600:
                evict(snap)
                active_snapshot_files += 1
        except (OSError, ValueError):
            pass
DATA_READERS = {'scripts/train.py', 'scripts/evaluate.py', 'scripts/train_dynamo.py', 'scripts/train_mvp3d.py',
                'scripts/train_snerl.py', 'scripts/train_sincro.py', 'scripts/train_reviwo.py'}
queue = {j['id']: j for j in jobs.load_queue()['jobs']}
readers = [jid for jid, s in state.items() if s.get('status') in jobs.ACTIVE and s.get('host', 'local') == 'local'
           and queue.get(jid, {}).get('script', 'scripts/train.py') in DATA_READERS]
dataset_files = 0
if not readers:
    for path in Path('/home/ws/data/metaworld/splatter4d_v1').rglob('*.hdf5'):
        try:
            evict(path)
            dataset_files += 1
        except OSError:
            pass
STEADY_TARGET = 130_000_000_000
active_replay_files = 0
if int(Path('/sys/fs/cgroup/memory.current').read_text()) > STEADY_TARGET:
    for jid, s in state.items():
        if s.get('status') not in jobs.ACTIVE or s.get('host', 'local') != 'local' or jid.endswith('-eval'):
            continue
        folder = ROOT / 'runs' / jid / 'replay'
        if not folder.is_dir() or folder.is_symlink():
            continue
        for path in folder.iterdir():
            try:
                evict(path)
                active_replay_files += 1
            except OSError:
                pass
after = int(Path('/sys/fs/cgroup/memory.current').read_text())
print(json.dumps({'time': time.strftime('%F %T'), 'files': count, 'bytes': size, 'finished_rl_runs': len(finished),
                  'rl_files': rl_files, 'rl_bytes': rl_bytes, 'stale_tree_files': stale_files, 'active_snapshot_files': active_snapshot_files, 'dataset_files': dataset_files, 'active_replay_files': active_replay_files, 'dataset_readers': len(readers), 'charge_before': before, 'charge_after': after}))
