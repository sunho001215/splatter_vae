"""Stop the self-render v2 pilot (user instruction 2026-10-31: stop every running Gaussian self-rendering experiment,
only those, after verifying each job's identity and configuration; it is replaced by a new variant trained from
scratch). Nothing is deleted: run directories, checkpoints, metrics and W&B files stay as they are.

    python -I stop_selfrender_v2_20261031.py --pause            # admission pause for every v2 row (no admit/retry)
    python -I stop_selfrender_v2_20261031.py --verify           # identity + configuration checks only, no signal
    python -I stop_selfrender_v2_20261031.py --stop             # verify, record the intentional stop, SIGTERM the
                                                                  job's own process group, wait, record cancellation
    python -I stop_selfrender_v2_20261031.py --cancel-pending   # terminal cancellation of never-started v2 rows

Per running pretraining the checks are: registry state (running, local, attempt, pid, GPU, commit) and its launched
event; the pid is its own session and process-group leader, alive, started within 120 s of the launched event; the
command line is scripts/train.py with --name <job id> and selfrender_replace_v2.yaml; every process in the group belongs
to this job (no foreign process can receive the signal); the trainer's CUDA_VISIBLE_DEVICES is the registry's MIG; the
run's resolved config is the v2 self-render recipe. Only that process group is signalled (SIGTERM, no escalation).
"""
import fcntl
import hashlib
import json
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path('/home/ws/ws/splatter4d')
sys.path.insert(0, str(ROOT))
from scripts import jobs  # noqa: E402

TASKS = ('hammer', 'pick-place', 'bin-picking')
RUNNING = [f'sr-csw4-seed0-{t}-200k' for t in TASKS]
PENDING_IDS = ([f'sr-csw4-export-seed0-{t}-200k' for t in TASKS]
               + [f'ds-flashsac-sr-csw4-{t}-s{s}{e}' for t in TASKS for s in (2000, 2001, 2002) for e in ('', '-eval')])
PREFIXES = ['sr-csw4-', 'ds-flashsac-sr-csw4-']
V2 = 'configs/metaworld/variants/selfrender_replace_v2.yaml'
REASON = ('cancelled: user instruction 2026-10-31 -- stop all running Gaussian self-rendering experiments '
          '(selfrender_replace_v2); replaced by a new variant (+-5 % radial camera jitter and the 5 %-relaxed viewpoint '
          'domain) trained from scratch under new names')
PAUSES = ROOT / 'runs/setup/admission-pauses.json'
OUT = ROOT / 'runs/setup/selfrender-v2-cancellation-20261031.json'
HZ = os.sysconf('SC_CLK_TCK')
BTIME = next(int(line.split()[1]) for line in Path('/proc/stat').read_text().splitlines() if line.startswith('btime'))


def record(event):
    with jobs.REGISTRY.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with jobs.REGISTRY.open('a') as stream:
            stream.write(json.dumps({**event, 'time': time.time()}) + '\n')
            stream.flush()
            os.fsync(stream.fileno())


def stat(pid):
    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    return {'state': fields[0], 'group': int(fields[2]), 'session': int(fields[3]), 'start_ticks': int(fields[19])}


def group_members(pgid):
    out = []
    for entry in Path('/proc').iterdir():
        if entry.name.isdigit():
            try:
                s = stat(int(entry.name))
                if s['group'] == pgid and s['state'] not in {'Z', 'X'}:
                    out.append((int(entry.name), (entry / 'cmdline').read_bytes().split(b'\0')))
            except (FileNotFoundError, ProcessLookupError, IndexError):
                continue
    return out


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load_out():
    return json.loads(OUT.read_text()) if OUT.is_file() else {'reason': REASON, 'jobs': {}}


def save_out(doc):
    tmp = OUT.with_suffix('.tmp')
    tmp.write_text(json.dumps(doc, indent=2) + '\n')
    os.replace(tmp, OUT)


def verify(job_id):
    state = jobs.read_registry()[job_id]
    assert state['status'] == jobs.RUNNING and state.get('host', 'local') == 'local', (job_id, state['status'])
    pid, gpu, attempt = state['pid'], state['gpu'], state['attempt']
    launched = [json.loads(line) for line in jobs.REGISTRY.read_text().splitlines() if f'"{job_id}"' in line]
    launched = [e for e in launched if e['id'] == job_id and e['event'] == 'launched' and e.get('attempt') == attempt]
    assert len(launched) == 1 and launched[0]['pid'] == pid and launched[0]['gpu'] == gpu, job_id
    s = stat(pid)
    assert s['session'] == pid and s['group'] == pid and s['state'] not in {'Z', 'X'}, (job_id, s)
    started = BTIME + s['start_ticks'] / HZ
    assert abs(started - launched[0]['time']) < 120, (job_id, started, launched[0]['time'])
    command = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    assert b'/home/ws/ws/splatter4d/scripts/train.py' in command and V2.encode() in command, job_id
    assert command[command.index(b'--name') + 1] == job_id.encode(), job_id
    members = group_members(pid)
    trainers = []
    for member, cmd in members:
        if member == pid:
            continue
        assert b'/home/ws/ws/splatter4d/scripts/train.py' in cmd and cmd[cmd.index(b'--name') + 1] == job_id.encode(), \
            (job_id, member, cmd[:4])
        trainers.append(member)
    assert trainers, job_id
    for member in trainers:
        env = dict(item.split(b'=', 1) for item in Path(f'/proc/{member}/environ').read_bytes().split(b'\0') if b'=' in item)
        assert env.get(b'CUDA_VISIBLE_DEVICES', b'').decode() == gpu, (job_id, member, env.get(b'CUDA_VISIBLE_DEVICES'))
    cfg = yaml.safe_load((ROOT / 'runs' / job_id / 'config.yaml').read_text())
    task = job_id.removeprefix('sr-csw4-seed0-').removesuffix('-200k')
    assert cfg['aug'] == {'synth_views': 2, 'synth_render': False, 'synth_source': 'gaussian_self_render',
                          'synth_ramp_steps': [20000, 40000],
                          'synth_camera_domain': 'configs/metaworld/selfrender_domain_v1.yaml'}, (job_id, cfg['aug'])
    assert int(cfg['train']['seed']) == 0 and int(cfg['train']['steps']) == 200000, job_id
    assert cfg['data']['task'] == task, (job_id, cfg['data']['task'])
    ckpts = sorted((ROOT / 'runs' / job_id / 'checkpoints').glob('step_*.pt'))
    return {'pid': pid, 'gpu': gpu, 'attempt': attempt, 'commit': launched[0].get('commit'),
            'start_ticks': s['start_ticks'], 'group_pids': sorted(m for m, _ in members),
            'trainer_pids': sorted(trainers), 'aug': cfg['aug'],
            'last_checkpoint': str(ckpts[-1].relative_to(ROOT)) if ckpts else None}


def last_logged(job_id):
    lines = (ROOT / 'runs' / job_id / 'metrics.jsonl').read_text().splitlines()
    for line in reversed(lines):
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if 'step' in row:
            return row['step']
    return None


if '--pause' in sys.argv:
    doc = json.loads(PAUSES.read_text())
    assert not any(p['name'] == 'selfrender-v2-stopped-20261031' for p in doc['pauses'])
    doc['pauses'].append({'name': 'selfrender-v2-stopped-20261031', 'active': True,
                          'since': datetime.now().astimezone().isoformat(timespec='seconds'), 'reason': REASON,
                          'prefixes': PREFIXES, 'ids': RUNNING + PENDING_IDS})
    tmp = PAUSES.with_suffix('.tmp')
    tmp.write_text(json.dumps(doc, indent=1) + '\n')
    os.replace(tmp, PAUSES)
    sys.path.insert(0, str(ROOT / 'runs/setup/tools'))
    from admission_pause import paused  # noqa: E402
    queue_ids = [j['id'] for j in jobs.load_queue()['jobs']]
    hit = [i for i in queue_ids if paused(i)]
    v2 = set(RUNNING + PENDING_IDS)
    extra = [i for i in hit if i not in v2 and not i.startswith('enc-abl-')]
    assert v2 <= set(hit) and not extra, extra
    print(json.dumps({'paused_v2_rows': len(v2), 'other_rows_matched': extra}))

if '--verify' in sys.argv or '--stop' in sys.argv:
    checks = {job_id: verify(job_id) for job_id in RUNNING}
    print(json.dumps(checks, indent=1))

if '--stop' in sys.argv:
    sys.path.insert(0, str(ROOT / 'runs/setup/tools'))
    from admission_pause import paused  # noqa: E402
    assert all(paused(i) for i in RUNNING + PENDING_IDS), 'admission pause missing'
    doc = load_out()
    for job_id, info in checks.items():
        ckpt = ROOT / info['last_checkpoint']
        step = int(ckpt.stem.split('_')[1])
        digest = sha(ckpt)
        record({'event': 'intentional_stopped', 'id': job_id, 'host': 'local', 'attempt': info['attempt'],
                'intentional_stop': True, 'reason': REASON, 'requested_signal': 'SIGTERM',
                'checkpoint': info['last_checkpoint'], 'checkpoint_step': step, 'checkpoint_sha256': digest})
        if stat(info['pid'])['start_ticks'] != info['start_ticks']:
            raise RuntimeError(f'identity changed for {job_id}; not signalled')
        os.killpg(info['pid'], signal.SIGTERM)
        began = time.monotonic()
        while group_members(info['pid']):
            if time.monotonic() - began > 900:
                raise RuntimeError(f'process group did not exit; no further signals: {job_id}')
            time.sleep(1)
        exited_after = time.monotonic() - began
        began = time.monotonic()
        while True:  # the scheduler daemon records the exit; the terminal cancellation must follow it
            state = jobs.read_registry()[job_id]
            if state['status'] != jobs.RUNNING:
                break
            if time.monotonic() - began > 600:
                raise RuntimeError(f'scheduler did not record the exit of {job_id}')
            time.sleep(5)
        record({'event': 'failed', 'id': job_id, 'host': 'local', 'cancelled': True, 'superseded': True,
                'reason': REASON})
        doc['jobs'][job_id] = {'state_before': 'running', 'verified': info, 'signal': 'SIGTERM (own process group)',
                               'exit_seconds': round(exited_after, 1), 'scheduler_status_after_exit': state['status'],
                               'last_logged_step': last_logged(job_id), 'last_checkpoint_step': step,
                               'last_checkpoint_sha256': digest,
                               'cancelled_at': datetime.now().astimezone().isoformat(timespec='seconds')}
        save_out(doc)
        print(json.dumps({'stopped': job_id, 'exit_seconds': round(exited_after, 1),
                          'last_logged_step': doc['jobs'][job_id]['last_logged_step'], 'last_checkpoint_step': step}),
              flush=True)

if '--cancel-pending' in sys.argv:
    doc = load_out()
    state = jobs.read_registry()
    for job_id in PENDING_IDS:
        current = state.get(job_id, {})
        assert current.get('status', jobs.PENDING) == jobs.PENDING and not current.get('attempts'), (job_id, current)
        assert not (ROOT / 'runs' / job_id).exists(), job_id
        record({'event': 'failed', 'id': job_id, 'host': 'local', 'cancelled': True, 'superseded': True,
                'reason': REASON})
        doc['jobs'][job_id] = {'state_before': 'pending (never started)',
                               'cancelled_at': datetime.now().astimezone().isoformat(timespec='seconds')}
    save_out(doc)
    print(json.dumps({'cancelled_pending': len(PENDING_IDS)}))
