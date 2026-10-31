"""Local ledger of the remote campaign jobs, for interruptions such as maintenance shutdowns (read-only on the remote).

One bounded SSH call records, for every remote job that is active, unknown or queued in the registry:
- run identity: job id, attempt, container, GPU, release commit, image digest, launch spec path (local), local config;
- the remote run directory, its resumable checkpoint (checkpoints/latest.pt: size, mtime, sha256), completion marker,
  the last logged training step, and the container's Docker state;
plus host health (reachability, the four authorized GPUs' utilisation/memory, runtime-root disk, RAM, Docker).
Writes runs/setup/remote-ledger/latest.json (atomic) and appends one line to history.jsonl. When the host is
unreachable the last good ledger is kept and an `unreachable` line is appended; nothing is retried here (the drivers
own retries). Recovery after an outage compares the host's state with this ledger (docs/REMOTE.md, "Maintenance").

    .venv/bin/python -I runs/setup/tools/remote_ledger.py
"""
import json
import os
import shlex
import sys
import time
from pathlib import Path

ROOT = Path('/home/ws/ws/splatter4d')
sys.path.insert(0, str(ROOT))
from ops.remote.access import RemoteClient, RemoteUnreachable  # noqa: E402
from ops.jobs import scheduler as jobs  # noqa: E402

REMOTE_ROOT = '/home/compu/kaist/sunho'
OUT = ROOT / 'runs/setup/remote-ledger'
GPUS = ('GPU-4391fcee-537f-d408-d138-10d8a3866eea', 'GPU-6122c7a9-eaa6-b539-6005-f20e339da4ea',
        'GPU-9e9f1e97-b2ca-04e0-eb2a-8035398daa79', 'GPU-ca65e1b7-c0ac-6757-7d3f-816c3119a4c1')


def run_dir(job):
    a = job.get('args', [])
    if '--run-dir' in a:
        return a[a.index('--run-dir') + 1]
    if '--name' in a:
        root = a[a.index('--output-root') + 1] if '--output-root' in a else 'runs'
        return f"{root}/{a[a.index('--name') + 1]}"
    return f"runs/{job['id']}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    now = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    queue = {j['id']: j for j in jobs.load_queue()['jobs']}
    state = jobs.read_registry()
    tracked = []
    for jid, job in queue.items():
        if job.get('host') != 'remote':
            continue
        cur = state.get(jid, {})
        status = cur.get('status', jobs.PENDING)
        if status in jobs.ACTIVE or cur.get('unknown') or status in (jobs.PENDING, 'crashed'):
            tracked.append((jid, job, cur, status))
    parts = ['echo "@@host"; date -Is; free -g | sed -n 2p; df -B1 ' + shlex.quote(REMOTE_ROOT) + ' | tail -1',
             'echo "@@gpu"; nvidia-smi --query-gpu=uuid,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>&1',
             'echo "@@docker"; docker ps -a --filter name=s4d- --format "{{.Names}}|{{.State}}|{{.Status}}" 2>&1']
    for jid, job, cur, status in tracked:
        d = shlex.quote(f'{REMOTE_ROOT}/{run_dir(job)}')
        parts.append(f'echo "@@job {jid}"; cd {d} 2>/dev/null && '
                     '{ stat -c "ckpt %s %Y" checkpoints/latest.pt 2>/dev/null; sha256sum checkpoints/latest.pt 2>/dev/null; '
                     'test -f completion.json && echo "completion $(cat completion.json)"; '
                     'tail -c 4000 metrics.jsonl 2>/dev/null | tail -1 | cut -c1-40; }')
    record = {'checked_at': now, 'reachable': False, 'jobs': {}}
    try:
        client = RemoteClient()
        result = client.ssh(' ; '.join(parts), wait=False, idempotent=True, timeout=240)
        text = result.stdout
    except RemoteUnreachable as exc:
        with (OUT / 'history.jsonl').open('a') as f:
            f.write(json.dumps({'checked_at': now, 'reachable': False, 'reason': str(exc)[:200]}) + '\n')
        print('UNREACHABLE', now)
        return
    record['reachable'] = True
    sections, key = {}, None
    for line in text.splitlines():
        if line.startswith('@@'):
            key = line[2:].strip()
            sections[key] = []
        elif key:
            sections[key].append(line)
    host = sections.get('host', [])
    record['host'] = {'date': host[0] if host else None, 'mem_line': host[1] if len(host) > 1 else None,
                      'disk_line': host[2] if len(host) > 2 else None}
    record['gpus'] = {line.split(',')[0].strip(): line for line in sections.get('gpu', []) if line.startswith('GPU-')}
    record['authorized_gpus_visible'] = all(g in record['gpus'] for g in GPUS)
    containers = {line.split('|')[0]: line.split('|')[1:] for line in sections.get('docker', []) if '|' in line}
    for jid, job, cur, status in tracked:
        lines = sections.get(f'job {jid}', [])
        entry = {'status': status, 'unknown': bool(cur.get('unknown')), 'attempt': cur.get('attempts'),
                 'container': cur.get('container'), 'container_state': containers.get(cur.get('container')),
                 'gpu': cur.get('gpu'), 'commit': cur.get('commit'), 'image_digest': cur.get('image_digest'),
                 'spec_path': cur.get('spec_path'), 'remote_run_dir': f'{REMOTE_ROOT}/{run_dir(job)}',
                 'local_config': str(ROOT / run_dir(job) / 'config.yaml') if (ROOT / run_dir(job) / 'config.yaml').is_file() else None,
                 'script': job.get('script'), 'args': job.get('args'), 'max_restarts': job.get('max_restarts'),
                 'failures': cur.get('failures')}
        for line in lines:
            if line.startswith('ckpt '):
                _, size, mtime = line.split()
                entry['checkpoint'] = {'path': f"{entry['remote_run_dir']}/checkpoints/latest.pt", 'bytes': int(size),
                                       'mtime': int(mtime)}
            elif len(line) > 64 and line[64:66] == '  ' and line.endswith('checkpoints/latest.pt'):
                entry.setdefault('checkpoint', {})['sha256'] = line[:64]
            elif line.startswith('completion '):
                entry['completion'] = line[11:]
            elif line.startswith('{"step"'):
                entry['last_logged'] = line
        record['jobs'][jid] = entry
    tmp = OUT / 'latest.json.tmp'
    tmp.write_text(json.dumps(record, indent=1))
    os.replace(tmp, OUT / 'latest.json')
    with (OUT / 'history.jsonl').open('a') as f:
        f.write(json.dumps({'checked_at': now, 'reachable': True, 'gpus_ok': record['authorized_gpus_visible'],
                            'jobs': {j: {'status': e['status'], 'ckpt_mtime': e.get('checkpoint', {}).get('mtime'),
                                         'sha256': e.get('checkpoint', {}).get('sha256'),
                                         'container_state': e['container_state']} for j, e in record['jobs'].items()}}) + '\n')
    print('LEDGER', now, len(record['jobs']), 'jobs', 'gpus_ok', record['authorized_gpus_visible'])


if __name__ == '__main__':
    main()
