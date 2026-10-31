"""(Copy for FlashSAC trainers, local shutdown 2026-10-31 23:00 KST, user authorization 2026-10-31.)
Lossless checkpoint pause of owned local DrM trainers (and then their evaluator companions) for the local 150 GB
ceiling (standing user authorization, 2026-10-10). A trainer is signalled only within 1 s of publishing latest.pt at
the next 50k boundary with no later train.jsonl step; its evaluator only after it has evaluated every snapshot up to
that checkpoint. Records checkpoint_paused events; resumes via scripts/jobs.py from latest.pt / eval.jsonl.

    python -I pause_rl_at_checkpoint.py <trainer job id> [...]
"""
import fcntl
import hashlib
import itertools
import json
import os
import pickletools
import signal
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path('/home/ws/ws/splatter4d')
REGISTRY = ROOT / 'experiments/registry.jsonl'
EVERY = 50000
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


def last_step(path):
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    return json.loads(lines[-1])['step'] if lines else 0


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


def owned(job_id, script):
    launch = [json.loads(line) for line in REGISTRY.read_text().splitlines() if f'"{job_id}"' in line]
    launch = [e for e in launch if e['id'] == job_id and e['event'] == 'launched' and e.get('host', 'local') == 'local']
    assert launch, job_id
    event = launch[-1]
    anchor = identity(event['pid'])
    command = Path(f"/proc/{event['pid']}/cmdline").read_bytes()
    # local evaluators name only their trainer (--train-job <trainer>), not their own job id
    needle = (b'--train-job\0' + job_id.removesuffix('-eval').encode() + b'\0') if job_id.endswith('-eval') \
        else job_id.encode()
    if (anchor['session'] != event['pid'] or anchor['group'] != event['pid'] or anchor['state'] in {'Z', 'X'}
            or script.encode() not in command or needle not in command):
        raise RuntimeError(f'owned identity not verified for {job_id}')
    return event, anchor


def pause(job_id, event, anchor, extra):
    os.killpg(event['pid'], signal.SIGTERM)
    began = time.monotonic()
    while session_alive(event['pid']):
        if time.monotonic() - began > 900:
            raise RuntimeError(f'session did not exit; no further signals: {job_id}')
        time.sleep(1)
    paused = {'event': 'checkpoint_paused', 'id': job_id, 'host': 'local', 'attempt': event['attempt'],
              'pid': event['pid'], 'start_ticks': anchor['start_ticks'], 'boot_id': BOOT,
              'reason': 'local_shutdown_20261031_maintenance', 'intentional_stop': True, 'requested_signal': 'SIGTERM', 'processes_exited': True,
              'memory_current_bytes': int(Path('/sys/fs/cgroup/memory.current').read_text()), **extra}
    record(paused)
    with (ROOT / 'runs' / job_id / 'checkpoint_pause-local_shutdown_20261031.json').open('x') as stream:
        stream.write(json.dumps(paused, sort_keys=True) + '\n')
    print(json.dumps(paused), flush=True)


assert (ROOT / 'experiments/HOLD').exists()
evaluators = {}
if sys.argv[1] == '--evaluators':  # trainers already paused: <evaluator id>=<trainer checkpoint step>
    evaluators = {k: int(v) for k, v in (a.split('=') for a in sys.argv[2:])}
    for job_id, step in evaluators.items():
        proof = json.loads((ROOT / 'runs' / job_id.removesuffix('-eval') / 'checkpoint_pause-local_shutdown_20261031.json').read_text())
        assert proof['checkpoint_step'] == step and proof['no_steps_lost'] is True, job_id
trainers = [] if evaluators else sys.argv[1:]
targets = {}
for job_id in trainers:
    event, anchor = owned(job_id, 'scripts/train_flashsac.py')
    run = ROOT / 'runs' / job_id
    target = (last_step(run / 'train.jsonl') // EVERY + 1) * EVERY
    targets[job_id] = {'event': event, 'anchor': anchor, 'target': target, 'run': run}
    record({'event': 'checkpoint_pause_requested', 'id': job_id, 'host': 'local', 'attempt': event['attempt'],
            'checkpoint_step': target, 'reason': 'local_shutdown_20261031_maintenance', 'requested_signal': 'SIGTERM'})
print(json.dumps({'armed': {k: v['target'] for k, v in targets.items()}}), flush=True)
deadline = time.monotonic() + 2.5 * 3600
while targets:
    if time.monotonic() > deadline:
        raise TimeoutError(f'window exceeded; untouched: {list(targets)}')
    for job_id, item in list(targets.items()):
        latest = item['run'] / 'checkpoints/latest.pt'
        if not latest.is_file() or time.time() - latest.stat().st_mtime > 1.0:
            continue
        step = checkpoint_step(latest)
        if step != item['target']:
            continue
        logged = last_step(item['run'] / 'train.jsonl')
        if logged > step:
            print(json.dumps({'missed_boundary': job_id, 'step': step, 'logged': logged}), flush=True)
            item['target'] += EVERY
            continue
        if identity(item['event']['pid'])['start_ticks'] != item['anchor']['start_ticks']:
            raise RuntimeError(f'owned identity changed for {job_id}')
        replay = item['run'] / 'replay/ram_snapshot.npz'
        pause(job_id, item['event'], item['anchor'],
              {'checkpoint_step': step, 'checkpoint': str(latest.relative_to(ROOT)),
               'checkpoint_sha256': hashlib.sha256(latest.read_bytes()).hexdigest(),
               'replay_snapshot_sha256': hashlib.sha256(replay.read_bytes()).hexdigest() if replay.is_file() else None,
               'last_logged_step': logged, 'no_steps_lost': True})
        assert checkpoint_step(latest) == step and last_step(item['run'] / 'train.jsonl') <= step
        evaluators[job_id + '-eval'] = step
        del targets[job_id]
    time.sleep(0.2)
while evaluators:
    for job_id, step in list(evaluators.items()):
        trainer = job_id.removesuffix('-eval')
        if not any(e.get('id') == job_id and e['event'] == 'launched' for e in
                   map(json.loads, REGISTRY.read_text().splitlines())):
            del evaluators[job_id]
            continue
        evaluated = last_step(ROOT / 'runs' / job_id / 'eval.jsonl')  # FlashSAC evaluators: --output-dir runs/<eval id>
        if evaluated < step:
            continue
        event, anchor = owned(job_id, 'scripts/eval_rl.py')
        pause(job_id, event, anchor, {'evaluated_through_step': evaluated, 'trainer_checkpoint_step': step,
                                      'no_steps_lost': True})
        assert last_step(ROOT / 'runs' / job_id / 'eval.jsonl') == evaluated
        del evaluators[job_id]
    time.sleep(5)
print(json.dumps({'finished': True, 'memory_current_bytes': int(Path('/sys/fs/cgroup/memory.current').read_text())}),
      flush=True)
