"""Best-effort, targeted eviction of clean file-cache pages of completed or inactive files (revised 2026-10-21).

Files are never modified or deleted; only their clean cached pages are released with POSIX_FADV_DONTNEED (dirty pages
stay until written back; a later read simply comes from disk again). Shared memory (/dev/shm) is never touched and
host-wide drop_caches is never used. Global guards (uncertain cases are skipped): a file currently open by any process
(/proc/*/fd), or modified in the last 10 minutes, is never evicted.

Rules:
1. leftover live-sync partial downloads (runs/**/*.s4d-partial-*);
2. finished RL runs (trainer and evaluator done): policy snapshots and the saved replay;
3. files nothing reads again: source worktrees and their test outputs (.worktrees/, runs/nv/, runs/rel/; .venv links
   skipped), release bundles (runs/remote/bundles/), pulled offline W&B folders (runs/<id>/wandb/);
4. active RL runs: policy snapshots the evaluator has already recorded, or not written for 10 minutes;
5. Meta-World datasets (main file and held-out set) of tasks that no active local data-reading job uses; if the task of
   any active reader cannot be resolved, no dataset is evicted;
6. (removed 2026-10-21) the replay files of active RL runs are active training data and are never evicted;
7. finished trainers whose evaluator still runs: their replay (only the trainer reads it);
8. pretraining checkpoints: every checkpoint of a finished run; step checkpoints of active runs (read again only on a
   resume);
9. report and analysis outputs under reports/ (re-read only by an analysis that has finished).
Reports the raw charge, the working set (max(0, memory.current - inactive_file), the 150 GB budget metric) and the
cache actually released (resident bytes of the evicted files before minus after, measured with fincore).

    .venv/bin/python -I runs/setup/tools/evict_partial_cache.py
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/home/ws/ws/splatter4d')
DATA = Path('/home/ws/data/metaworld/splatter4d_v1')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/setup/tools'))
from scripts import jobs  # noqa: E402
import memory_budget as mb  # noqa: E402

RECENT_S = 600
DATA_READERS = {'scripts/train.py', 'scripts/evaluate.py', 'scripts/train_dynamo.py', 'scripts/train_mvp3d.py',
                'scripts/train_snerl.py', 'scripts/train_sincro.py', 'scripts/train_reviwo.py'}


def open_files():
    out = set()
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            for fd in (proc / 'fd').iterdir():
                try:
                    out.add(os.readlink(fd))
                except OSError:
                    pass
        except OSError:
            pass
    return out


OPEN = open_files()
NOW = time.time()


def resident(paths):
    paths = [str(p) for p in paths]
    total = 0
    for i in range(0, len(paths), 500):
        out = subprocess.run(['fincore', '--bytes', '--noheadings', '--output', 'RES', *paths[i:i + 500]],
                             capture_output=True, text=True).stdout.split()
        total += sum(int(x) for x in out if x.isdigit())
    return total


def eligible(path):
    try:
        if path.is_symlink() or not path.is_file():
            return False
        if str(path) in OPEN or str(path.resolve()) in OPEN:
            return False
        return NOW - path.stat().st_mtime >= RECENT_S
    except OSError:
        return False


def evict(paths, rule, report):
    chosen = [p for p in paths if eligible(p)]
    if not chosen:
        report[rule] = {'files': 0, 'released_bytes': 0}
        return
    before = resident(chosen)
    for path in chosen:
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
        except OSError:
            pass
    report[rule] = {'files': len(chosen), 'released_bytes': max(0, before - resident(chosen))}


def walk(root, skip_dirs=('.venv',)):
    for dirpath, dirnames, names in os.walk(root, followlinks=False):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for name in names:
            yield Path(dirpath) / name


def task_of(job):
    args = [str(a) for a in job.get('args', [])]
    if '--task' in args:
        return args[args.index('--task') + 1]
    cfgs = [a for a in args if a.startswith('configs/metaworld/tasks/')]
    if cfgs:
        return Path(cfgs[0]).stem
    for flag in ('--config', '--run'):
        if flag in args:
            cfg = ROOT / args[args.index(flag) + 1]
            cfg = cfg / 'config.yaml' if cfg.is_dir() else cfg
            try:
                text = cfg.read_text()
            except OSError:
                continue
            for line in text.splitlines():
                if line.strip().startswith('task:'):
                    return line.split(':', 1)[1].strip().strip('{}').split(',')[0].strip()
    return None


def main():
    before = mb.snapshot()
    report = {}
    state = jobs.read_registry()
    queue = {j['id']: j for j in jobs.load_queue()['jobs']}
    evict(list((ROOT / 'runs').rglob('*.s4d-partial-*')), '1_partial_downloads', report)
    finished = [jid for jid, s in state.items() if not jid.startswith('__') and not jid.endswith('-eval')
                and s.get('status') == jobs.DONE and state.get(jid + '-eval', {}).get('status') == jobs.DONE
                and (ROOT / 'runs' / jid / 'final.json').is_file()]
    paths = []
    for jid in finished:
        for sub in ('snapshots', 'replay'):
            folder = ROOT / 'runs' / jid / sub
            if folder.is_dir() and not folder.is_symlink():
                paths += list(folder.iterdir())
    evict(paths, '2_finished_rl_runs', report)
    paths = []
    for root in [ROOT / '.worktrees', ROOT / 'runs/nv', ROOT / 'runs/rel', ROOT / 'runs/remote/bundles',
                 *(ROOT / 'runs').glob('*/wandb')]:
        if root.is_dir():
            paths += list(walk(root))
    evict(paths, '3_stale_trees', report)
    paths = []
    for jid, s in state.items():
        if s.get('status') not in jobs.ACTIVE or s.get('host', 'local') != 'local' or jid.endswith('-eval'):
            continue
        folder = ROOT / 'runs' / jid / 'snapshots'
        if not folder.is_dir() or folder.is_symlink():
            continue
        evals = [ROOT / 'runs' / f'{jid}-eval' / 'eval.jsonl', ROOT / 'runs' / jid / 'eval.jsonl']
        path = next((e for e in evals if e.is_file()), None)
        try:
            done = -1 if path is None else max(
                (json.loads(line)['step'] for line in path.read_text().splitlines() if line.strip()), default=-1)
        except (OSError, ValueError, KeyError):
            done = -1
        for snap in folder.glob('step_*.pt'):
            try:
                if int(snap.stem.split('_')[1]) <= done or NOW - snap.stat().st_mtime > RECENT_S:
                    paths.append(snap)
            except (OSError, ValueError):
                pass
    evict(paths, '4_active_rl_old_snapshots', report)
    readers = {jid: queue.get(jid, {}) for jid, s in state.items()
               if s.get('status') in jobs.ACTIVE and s.get('host', 'local') == 'local'
               and queue.get(jid, {}).get('script', 'scripts/train.py') in DATA_READERS}
    tasks_in_use = {task_of(job) for job in readers.values()}
    if None in tasks_in_use:
        report['5_unused_datasets'] = {'files': 0, 'released_bytes': 0, 'skipped': 'an active reader task is unknown'}
    else:
        paths = [p for p in DATA.glob('*.hdf5') if p.stem.removesuffix('_pilot') not in tasks_in_use]
        paths += [p for p in (DATA / 'heldout_sets').glob('*.hdf5') if p.stem not in tasks_in_use]
        evict(paths, '5_unused_datasets', report)
        report['5_unused_datasets']['tasks_in_use'] = sorted(tasks_in_use)
    paths = []
    for jid, s in state.items():
        if jid.endswith('-eval') or s.get('status') != jobs.DONE or s.get('host', 'local') != 'local':
            continue
        folder = ROOT / 'runs' / jid / 'replay'
        if folder.is_dir() and not folder.is_symlink() and (ROOT / 'runs' / jid / 'final.json').is_file():
            paths += list(folder.iterdir())
    evict(paths, '7_finished_trainer_replay', report)
    paths = []
    for run in (ROOT / 'runs/pretrain').glob('*'):
        folder = run / 'checkpoints'
        if folder.is_dir() and not folder.is_symlink():
            paths += [p for p in folder.glob('*.pt')]
    evict(paths, '8_pretraining_checkpoints', report)
    reports_root = ROOT / 'reports'
    evict(list(walk(reports_root)) if reports_root.is_dir() else [], '9_report_outputs', report)
    after = mb.snapshot()
    print(json.dumps({'time': time.strftime('%F %T'), 'rules': report,
                      'released_bytes_total': sum(r.get('released_bytes', 0) for r in report.values()),
                      'working_set_before': before['working_set'], 'working_set_after': after['working_set'],
                      'raw_before': before['current'], 'raw_after': after['current'],
                      'inactive_file_after': after['inactive_file'], 'active_file_after': after['active_file'],
                      'open_files_skipped_guard': len(OPEN)}))


if __name__ == '__main__':
    main()
