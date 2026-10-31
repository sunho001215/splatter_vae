"""Lossless checkpoint pause of owned local pretraining runs for the local 150 GB ceiling (standing user authorization,
2026-10-10). Signals a run only after (1) step_<N>.pt and latest.pt are published at the next save boundary N,
(2) the scheduled validation at N is in metrics.jsonl, and (3) no training step after N has been logged. Resume via
scripts/jobs.py (--resume auto) restores step N exactly; nothing logged is lost or duplicated.

    python -I pause_pretrain_at_checkpoint.py [--reason <reason>] <job id> [...]

`--reason` (default local_campaign_150GB_ceiling) is recorded in the pause events, which also carry
intentional_stop: true, so the following non-zero exit is identifiable as an intentional stop, not a crash.
"""
import fcntl
import hashlib
import itertools
import json
import os
import pickletools
import re
import signal
import sys
import time
import zipfile
from pathlib import Path

import yaml

ROOT = Path('/home/ws/ws/splatter4d')
REGISTRY = ROOT / 'experiments/registry.jsonl'
BOOT = Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def identity(pid):
    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    return {'pid': pid, 'group': int(fields[2]), 'session': int(fields[3]), 'start_ticks': int(fields[19]),
            'state': fields[0]}


def record(event):
    with REGISTRY.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with REGISTRY.open('a') as stream:
            stream.write(json.dumps({**event, 'time': time.time()}) + '\n')
            stream.flush()
            os.fsync(stream.fileno())


def checkpoint_step(path):
    with zipfile.ZipFile(path) as archive:
        name = next(n for n in archive.namelist() if n.endswith('/data.pkl'))
        ops = list(itertools.islice(pickletools.genops(archive.read(name)), 7))
    if ops[-3][1] != 'step' or ops[-1][0].name not in {'BININT', 'BININT1', 'BININT2'}:
        raise RuntimeError('unexpected checkpoint step metadata')
    return ops[-1][1]


def logged_step(run):
    path = run / 'log.txt'
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size - 20000))
        matches = re.findall(rb'\] step (\d+) loss', stream.read())
    return int(matches[-1]) if matches else 0


def validated(run, step):
    path = run / 'metrics.jsonl'
    with path.open('rb') as stream:
        offset = max(0, path.stat().st_size - 400000)
        stream.seek(offset)
        lines = stream.read().splitlines()[1 if offset else 0:]
    for line in reversed(lines):
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get('step') == step and any(k.startswith('val/') for k in row):
            return True
    return False


def session_alive(pid):
    for entry in Path('/proc').iterdir():
        if entry.name.isdigit():
            try:
                status = identity(int(entry.name))
            except (FileNotFoundError, ProcessLookupError, IndexError):
                continue
            if status['session'] == pid and status['state'] not in {'Z', 'X'}:
                return True
    return False


ARGS = sys.argv[1:]
REASON = 'local_campaign_150GB_ceiling'
if '--reason' in ARGS:
    at = ARGS.index('--reason')
    REASON = ARGS[at + 1]
    del ARGS[at:at + 2]
assert (ROOT / 'experiments/HOLD').exists()
items = {}
for job_id in ARGS:
    events = [json.loads(line) for line in REGISTRY.read_text().splitlines() if f'"{job_id}"' in line]
    event = [e for e in events if e['id'] == job_id and e['event'] == 'launched' and e.get('host', 'local') == 'local'][-1]
    anchor = identity(event['pid'])
    command = Path(f"/proc/{event['pid']}/cmdline").read_bytes()
    if (anchor['session'] != event['pid'] or anchor['group'] != event['pid'] or anchor['state'] in {'Z', 'X'}
            or b'scripts/train.py' not in command or job_id.encode() not in command):
        raise RuntimeError(f'owned identity not verified for {job_id}')
    run = ROOT / 'runs/pretrain' / job_id
    cfg = yaml.safe_load((run / 'config.yaml').read_text())
    save, every = int(cfg['train']['save_every']), int(cfg['train']['eval_every'])
    assert save % every == 0
    target = (logged_step(run) // save + 1) * save
    assert target < int(cfg['train']['steps'])
    items[job_id] = {'event': event, 'anchor': anchor, 'run': run, 'target': target, 'next_log': target
                     + int(cfg['train']['log_every'])}
    record({'event': 'checkpoint_pause_requested', 'id': job_id, 'host': 'local', 'attempt': event['attempt'],
            'checkpoint_step': target, 'reason': REASON, 'intentional_stop': True, 'requested_signal': 'SIGTERM'})
print(json.dumps({'armed': {k: v['target'] for k, v in items.items()}}), flush=True)
deadline = time.monotonic() + 6 * 3600
while items:
    if time.monotonic() > deadline:
        raise TimeoutError(f'window exceeded; untouched: {list(items)}')
    for job_id, item in list(items.items()):
        run, target = item['run'], item['target']
        checkpoint = run / f'checkpoints/step_{target:07d}.pt'
        if not checkpoint.is_file() or not validated(run, target):
            continue
        logged = logged_step(run)
        latest = run / 'checkpoints/latest.pt'
        if logged >= item['next_log'] or checkpoint_step(latest) != target:
            print(json.dumps({'missed_boundary': job_id, 'target': target, 'logged': logged}), flush=True)
            item['target'] += int(yaml.safe_load((run / 'config.yaml').read_text())['train']['save_every'])
            item['next_log'] += item['target'] - target
            continue
        if identity(item['event']['pid'])['start_ticks'] != item['anchor']['start_ticks']:
            raise RuntimeError(f'owned identity changed for {job_id}')
        os.killpg(item['event']['pid'], signal.SIGTERM)
        began = time.monotonic()
        while session_alive(item['event']['pid']):
            if time.monotonic() - began > 900:
                raise RuntimeError(f'session did not exit; no further signals: {job_id}')
            time.sleep(1)
        after = logged_step(run)
        assert after < item['next_log'] and checkpoint_step(checkpoint) == target and checkpoint_step(latest) == target
        with zipfile.ZipFile(checkpoint) as archive:
            assert archive.testzip() is None
        paused = {'event': 'checkpoint_paused', 'id': job_id, 'host': 'local', 'attempt': item['event']['attempt'],
                  'pid': item['event']['pid'], 'start_ticks': item['anchor']['start_ticks'], 'boot_id': BOOT,
                  'reason': REASON, 'intentional_stop': True, 'requested_signal': 'SIGTERM', 'checkpoint_step': target,
                  'checkpoint': str(checkpoint.relative_to(ROOT)),
                  'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  'validation_at_checkpoint_logged': True, 'last_logged_step': after, 'no_steps_lost': True,
                  'processes_exited': True,
                  'memory_current_bytes': int(Path('/sys/fs/cgroup/memory.current').read_text())}
        record(paused)
        name = 'checkpoint_pause.json' if REASON == 'local_campaign_150GB_ceiling' else f'checkpoint_pause-{REASON}.json'
        with (ROOT / 'runs' / job_id / name).open('x') as stream:
            stream.write(json.dumps(paused, sort_keys=True) + '\n')
        print(json.dumps(paused), flush=True)
        del items[job_id]
    time.sleep(0.2)
print(json.dumps({'finished': True, 'memory_current_bytes': int(Path('/sys/fs/cgroup/memory.current').read_text())}),
      flush=True)
