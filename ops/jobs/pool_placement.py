"""Host-neutral pool placement (user instruction 2026-10-23: the local and remote servers are one experiment pool).

Unstarted jobs may run on either host; running jobs stay where they are; host assignment never changes the scientific
experiment. A pending row that never launched (no registry entry, no run or output directory) is moved by rewriting
ONLY its host-specific execution fields (HOST_FIELDS) in experiments/queue.yaml, under the scheduler's registry lock
and the local admission lock (so no local or remote driver can admit it concurrently). Every move is appended to
runs/setup/pool-placement.json with the original and the effective row; the drivers apply ``effective_rows()`` as an
overlay to their registration files. RL trainers always move together with their evaluator (the evaluator reads the
trainer's run directory); exports of local pretrainings stay local (seconds of work); since
2026-10-27 the export of a remote pretraining moves to the remote and runs on its files verified there (no download). Remote results synchronise from
runs/<job id>, so a pretraining row moved to the remote writes there (--output-root runs instead of runs/pretrain) and
its local export row reads from there: an output-location rewrite only, recorded with both rows. Code per host: local jobs run the
main checkout, remote jobs the accepted release recorded in their dispatch (sources identical to main at acceptance);
the commit, configuration, data identity, export and seed of each run are recorded in the registry as before.

    python -I pool_placement.py status
    python -I pool_placement.py move --to remote|local <job id> [...]   (manual move; pairs added automatically)
"""
import argparse
import copy
import fcntl
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path('/home/ws/ws/splatter4d')
sys.path.insert(0, str(ROOT))
from ops.jobs import scheduler as jobs  # noqa: E402

SETUP = ROOT / 'runs/setup'
MANIFEST = SETUP / 'pool-placement.json'
ADMISSION_LOCK = SETUP / 'stage2-driver.lock'  # held by both local admission drivers for their whole pass
HOST_FIELDS = ('host', 'placement', 'required_results', 'disk_gb', 'cpu_threads')
SCRIPTS = ('scripts/train.py', 'scripts/train_rl.py', 'scripts/eval_rl.py', 'scripts/export_encoder.py')


def effective_rows() -> dict:
    return json.loads(MANIFEST.read_text())['effective_jobs'] if MANIFEST.is_file() else {}


def partner(job_id: str) -> str:
    return job_id.removesuffix('-eval') if job_id.endswith('-eval') else job_id + '-eval'


def output_dirs(job: dict) -> list[Path]:
    args = job['args']
    out = [ROOT / 'runs' / job['id']]
    for flag in ('--run-dir', '--output-dir'):
        if flag in args:
            out.append(ROOT / args[args.index(flag) + 1])
    if '--name' in args and '--output-root' in args:
        out.append(ROOT / args[args.index('--output-root') + 1] / args[args.index('--name') + 1])
    return out


def movable(job: dict, state: dict, by_id: dict) -> bool:
    """Never launched, an established pool script, no continuation seed, no verification, own result directories (an
    evaluator writes to its own --output-dir so results sync per job); RL pairs must both be movable."""
    if job['script'] not in SCRIPTS or job.get('verification_purpose') or job.get('continues'):
        return False
    if job['id'] in state or any(p.exists() for p in output_dirs(job)):
        return False
    if job['script'] == 'scripts/eval_rl.py' and '--output-dir' not in job['args']:
        return False
    if job['script'] in ('scripts/train_rl.py', 'scripts/eval_rl.py'):
        other = by_id.get(partner(job['id']))
        if other is None or other['id'] in state or other.get('host', 'local') != job.get('host', 'local'):
            return False
        if other['script'] == 'scripts/eval_rl.py' and '--output-dir' not in other['args']:
            return False
    return True


def output_rewrite(job: dict, host: str) -> dict:
    """{old path prefix: new path prefix} of a pretraining row's output location on ``host`` ({} if unchanged)."""
    if job['script'] != 'scripts/train.py' or '--output-root' not in job['args']:
        return {}
    root, name = job['args'][job['args'].index('--output-root') + 1], job['args'][job['args'].index('--name') + 1]
    want = 'runs' if host == 'remote' else root
    return {} if root == want else {f'{root}/{name}': f'{want}/{name}'}


def rewrite_args(args: list, mapping: dict, output_root: str | None = None) -> list:
    out = []
    for i, a in enumerate(args):
        if output_root is not None and i > 0 and args[i - 1] == '--output-root':
            out.append(output_root)
            continue
        for old, new in mapping.items():
            if a == old or a.startswith(old + '/'):
                a = new + a[len(old):]
        out.append(a)
    return out


def placed_row(job: dict, host: str, stamp: str, mapping: dict | None = None) -> dict:
    new = copy.deepcopy(job)
    new['host'] = host
    new['placement'] = f'pool {stamp} ({host}; runs/setup/pool-placement.json)'
    if mapping:  # output location only (module docstring)
        new['args'] = rewrite_args(job['args'], mapping, 'runs' if host == 'remote' else None)
    if host == 'remote':  # result synchronisation needs declared results (the remote driver closes out per job)
        defaults = {'scripts/train_rl.py': ['final.json', 'train.jsonl'], 'scripts/eval_rl.py': ['eval.jsonl']}
        if 'required_results' not in new and job['script'] in defaults:
            new['required_results'] = defaults[job['script']]
        new.setdefault('disk_gb', 30 if job['script'] == 'scripts/train.py' else 10)
    sci = lambda r: {k: v for k, v in r.items() if k not in HOST_FIELDS + ('args',)}  # noqa: E731
    assert sci(new) == sci(job), job['id']
    assert rewrite_args(new['args'], {v: k for k, v in (mapping or {}).items()},
                        job['args'][job['args'].index('--output-root') + 1] if mapping and '--output-root' in job['args']
                        else None) == job['args'], job['id']
    return new


def export_row(job: dict, mapping: dict, stamp: str) -> dict:
    """A local export row of a moved pretraining run: its input paths follow the output location."""
    new = copy.deepcopy(job)
    new['args'] = rewrite_args(job['args'], mapping)
    new['placement'] = f'pool {stamp} (input paths of a moved pretraining run; runs/setup/pool-placement.json)'
    assert {k: v for k, v in new.items() if k not in ('args', 'placement')} == \
        {k: v for k, v in job.items() if k not in ('args', 'placement')}
    return new


def move(ids: list[str], host: str, reason: str, blocking: bool = False) -> list[str]:
    """Move never-launched rows (and their RL partners) to ``host``; returns the moved ids ([] if the local admission
    lock is busy and not ``blocking``)."""
    with open(ADMISSION_LOCK, 'w') as admission:
        try:
            fcntl.flock(admission, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError:
            return []
        with jobs.REGISTRY.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            queue = jobs.load_queue()
            by_id = {j['id']: j for j in queue['jobs']}
            state = jobs.read_registry()
            wanted = set()
            for i in ids:
                if i in by_id:
                    wanted.add(i)
                    if by_id[i]['script'] in ('scripts/train_rl.py', 'scripts/eval_rl.py'):
                        wanted.add(partner(i))
            chosen = [i for i in sorted(wanted) if i in by_id and by_id[i].get('host', 'local') != host
                      and movable(by_id[i], state, by_id)]
            if not chosen:
                return []
            stamp = datetime.now().astimezone().isoformat(timespec='seconds')
            effective = {}
            for i in chosen:
                mapping = output_rewrite(by_id[i], host)
                effective[i] = placed_row(by_id[i], host, stamp, mapping)
                if mapping:
                    for j in queue['jobs']:
                        if i in j.get('deps', []) and j['script'] == 'scripts/export_encoder.py':
                            assert j['id'] not in state and j['id'] not in effective, j['id']
                            effective[j['id']] = export_row(j, mapping, stamp)
            before = jobs.QUEUE.read_bytes()
            text = before.decode()
            node = next(value for key, value in yaml.compose(text).value if key.value == 'jobs')
            new_jobs = [effective.get(j['id'], j) for j in queue['jobs']]
            staged = SETUP / 'pool-placement-queue-candidate.yaml'
            staged.write_text(text[:node.start_mark.index] + yaml.safe_dump(json.loads(json.dumps(new_jobs)), sort_keys=False)
                              + text[node.end_mark.index:])
            assert jobs.load_queue(staged)['jobs'] == new_jobs
            assert jobs.load_queue(staged)['hosts'] == queue['hosts']
            tmp = jobs.QUEUE.with_suffix('.pool.tmp')
            tmp.write_bytes(staged.read_bytes())
            os.replace(tmp, jobs.QUEUE)
            record = json.loads(MANIFEST.read_text()) if MANIFEST.is_file() else {
                'purpose': __doc__.split('\n\n')[0], 'moves': [], 'effective_jobs': {}}
            record['moves'].append({'time': stamp, 'to': host, 'reason': reason, 'ids': chosen,
                                    'rewritten_exports': sorted(set(effective) - set(chosen)),
                                    'original': {i: by_id[i] for i in effective},
                                    'queue_sha256_before': hashlib.sha256(before).hexdigest(),
                                    'queue_sha256_after': hashlib.sha256(jobs.QUEUE.read_bytes()).hexdigest()})
            record['effective_jobs'].update(effective)
            tmp = MANIFEST.with_suffix('.tmp')
            tmp.write_text(json.dumps(record, indent=1) + '\n')
            os.replace(tmp, MANIFEST)
            return chosen


def set_priority(ids: list[str], priority: int, reason: str) -> list[str]:
    """Queue-order change only (user instruction 2026-10-23: vision-only before the low-priority SNeRL RL expansion):
    rewrite the ``priority`` of never-launched rows, nothing else; recorded like a move (original / effective rows) and
    applied by the drivers through ``effective_rows()``."""
    with open(ADMISSION_LOCK, 'w') as admission:
        fcntl.flock(admission, fcntl.LOCK_EX)
        with jobs.REGISTRY.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            queue = jobs.load_queue()
            by_id = {j['id']: j for j in queue['jobs']}
            state = jobs.read_registry()
            chosen = [i for i in ids if i in by_id and i not in state and by_id[i].get('priority') != priority]
            if not chosen:
                return []
            stamp = datetime.now().astimezone().isoformat(timespec='seconds')
            effective = {}
            for i in chosen:
                new = copy.deepcopy(by_id[i])
                new['priority'] = priority
                assert {k: v for k, v in new.items() if k != 'priority'} == {k: v for k, v in by_id[i].items() if k != 'priority'}
                effective[i] = new
            before = jobs.QUEUE.read_bytes()
            text = before.decode()
            node = next(value for key, value in yaml.compose(text).value if key.value == 'jobs')
            new_jobs = [effective.get(j['id'], j) for j in queue['jobs']]
            staged = SETUP / 'pool-placement-queue-candidate.yaml'
            staged.write_text(text[:node.start_mark.index] + yaml.safe_dump(json.loads(json.dumps(new_jobs)), sort_keys=False)
                              + text[node.end_mark.index:])
            assert jobs.load_queue(staged)['jobs'] == new_jobs and jobs.load_queue(staged)['hosts'] == queue['hosts']
            tmp = jobs.QUEUE.with_suffix('.pool.tmp')
            tmp.write_bytes(staged.read_bytes())
            os.replace(tmp, jobs.QUEUE)
            record = json.loads(MANIFEST.read_text()) if MANIFEST.is_file() else {
                'purpose': __doc__.split('\n\n')[0], 'moves': [], 'effective_jobs': {}}
            record['moves'].append({'time': stamp, 'priority': priority, 'reason': reason, 'ids': chosen,
                                    'original': {i: by_id[i] for i in chosen},
                                    'queue_sha256_before': hashlib.sha256(before).hexdigest(),
                                    'queue_sha256_after': hashlib.sha256(jobs.QUEUE.read_bytes()).hexdigest()})
            record['effective_jobs'].update(effective)
            tmp = MANIFEST.with_suffix('.tmp')
            tmp.write_text(json.dumps(record, indent=1) + '\n')
            os.replace(tmp, MANIFEST)
            return chosen


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('status')
    mv = sub.add_parser('move')
    mv.add_argument('--to', choices=('local', 'remote'), required=True)
    mv.add_argument('ids', nargs='+')
    args = ap.parse_args()
    if args.cmd == 'status':
        record = json.loads(MANIFEST.read_text()) if MANIFEST.is_file() else {'moves': [], 'effective_jobs': {}}
        print(json.dumps({'moves': len(record['moves']), 'placed': {i: r['host'] for i, r in record['effective_jobs'].items()}},
                         indent=1))
    else:
        print(json.dumps(move(args.ids, args.to, 'manual', blocking=True)))


if __name__ == '__main__':
    main()
