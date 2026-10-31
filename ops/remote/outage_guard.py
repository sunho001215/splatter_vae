"""Remote outage guard (user instruction 2026-10-20: the remote may shut down for maintenance). Runs every 5 minutes.

- Reachable and no guard hold: refresh the ledger (remote_ledger.py) when the last one is older than 30 minutes.
- Unreachable: record it; once the host has been unreachable for > 10 minutes (a maintenance-length outage, not the
  usual 1-3 minute VPN drops), create runs/setup/remote-hold-admissions (the remote driver then admits no remote
  job; reconciliation and result syncs continue), so nothing is relaunched before the host is verified.
- Reachable again while the guard's hold is in place: recovery check against the last pre-outage ledger:
  host health (the four authorized GPUs visible, runtime-root disk >= 15 % free, Docker answering); for every job
  that was running: container still running (survived), or the job's checkpoints/latest.pt present with the
  pre-outage sha256 or a newer complete file (size equal), so `--resume auto` continues from it; completed runs
  are left to the driver's result sync. All pass -> the hold is removed and the driver resumes the interrupted jobs
  from their checkpoints (new attempt, same run directory and identity). Any failure -> the hold stays and
  runs/setup/remote-ledger/recovery-blockers.json names the job and the blocker for the user.
The guard never starts, stops or deletes anything on the remote.

    .venv/bin/python -I runs/setup/tools/remote_outage_guard.py
"""
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/home/ws/ws/splatter4d')
sys.path.insert(0, str(ROOT))
from ops.remote.access import RemoteClient, RemoteUnreachable  # noqa: E402

LEDGER = ROOT / 'runs/setup/remote-ledger'
GUARD = LEDGER / 'guard.json'
HOLD = ROOT / 'runs/setup/remote-hold-admissions'
HOLD_AFTER_S, LEDGER_EVERY_S = 600, 1800
GPUS = ('GPU-4391fcee-537f-d408-d138-10d8a3866eea', 'GPU-6122c7a9-eaa6-b539-6005-f20e339da4ea',
        'GPU-9e9f1e97-b2ca-04e0-eb2a-8035398daa79', 'GPU-ca65e1b7-c0ac-6757-7d3f-816c3119a4c1')


def log(msg):
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {msg}"
    print(line, flush=True)
    with (LEDGER / 'guard.log').open('a') as f:
        f.write(line + '\n')


def save(state):
    tmp = GUARD.with_suffix('.tmp')
    tmp.write_text(json.dumps(state, indent=1))
    os.replace(tmp, GUARD)


def recovery_check(client):
    ledger = json.loads((LEDGER / 'pre-outage-ledger.json').read_text())
    running = {j: e for j, e in ledger['jobs'].items() if e['status'] == 'running'}
    cmds = ['echo "@@gpu"; nvidia-smi --query-gpu=uuid --format=csv,noheader 2>&1',
            'echo "@@disk"; df -P ' + shlex.quote('/home/compu/kaist/sunho') + ' | tail -1',
            'echo "@@docker"; docker ps -a --filter name=s4d- --format "{{.Names}}|{{.State}}|{{.Status}}" 2>&1']
    for j, e in running.items():
        d = shlex.quote(e['remote_run_dir'])
        cmds.append(f'echo "@@job {j}"; cd {d} && {{ stat -c "ckpt %s %Y" checkpoints/latest.pt; '
                    'sha256sum checkpoints/latest.pt; test -f completion.json && echo completion; } 2>&1')
    out = client.ssh(' ; '.join(cmds), wait=False, idempotent=True, timeout=300).stdout
    sec, key = {}, None
    for line in out.splitlines():
        if line.startswith('@@'):
            key = line[2:].strip()
            sec[key] = []
        elif key:
            sec[key].append(line)
    blockers, notes = {}, {}
    if not all(any(g in line for line in sec.get('gpu', [])) for g in GPUS):
        blockers['host'] = f"authorized GPUs not all visible: {sec.get('gpu')}"
    disk = (sec.get('disk') or [''])[0].split()
    if len(disk) >= 5 and disk[4].endswith('%') and int(disk[4][:-1]) > 85:
        blockers['host_disk'] = f'runtime root {disk[4]} used (floor: 15 % free)'
    containers = {l.split('|')[0]: l.split('|')[1:] for l in sec.get('docker', []) if '|' in l}
    if not sec.get('docker') or any('Cannot connect' in l or 'permission denied' in l for l in sec.get('docker', [])):
        blockers['docker'] = f"Docker not answering: {sec.get('docker')}"
    for j, e in running.items():
        lines = sec.get(f'job {j}', [])
        state = containers.get(e['container'])
        ck = next((l.split() for l in lines if l.startswith('ckpt ')), None)
        sha = next((l[:64] for l in lines if l.endswith('checkpoints/latest.pt') and len(l) > 64), None)
        before = e.get('checkpoint', {})
        if state and state[0] == 'running':
            notes[j] = 'container survived; running'
        elif any(l == 'completion' for l in lines):
            notes[j] = f'completed during the outage (container {state}); the driver syncs and verifies results'
        elif ck is None or sha is None:
            blockers[j] = f'no readable checkpoints/latest.pt (container {state}); --resume auto would restart from scratch'
        elif sha == before.get('sha256'):
            notes[j] = f'checkpoint unchanged since the pre-outage ledger (sha {sha[:12]}); resumable (container {state})'
        elif int(ck[1]) == before.get('bytes') and int(ck[2]) > before.get('mtime', 0):
            notes[j] = f'newer complete checkpoint (mtime {ck[2]}, sha {sha[:12]}); resumable (container {state})'
        else:
            blockers[j] = f'checkpoint differs from the pre-outage ledger and is not a newer full-size file: {ck} {sha}'
        if state is None and j not in blockers and 'completed' not in notes.get(j, ''):
            blockers[j] = (f'container {e["container"]} missing after the outage: the scheduler keeps the job unknown; '
                           'manual reconciliation with this evidence is needed before a resume (checkpoint verified: '
                           f'{notes.get(j)})')
    return blockers, notes


def main():
    LEDGER.mkdir(parents=True, exist_ok=True)
    state = json.loads(GUARD.read_text()) if GUARD.is_file() else {}
    now = time.time()
    client = RemoteClient()
    try:
        client.check_tunnel(wait=False)
        reachable = True
    except RemoteUnreachable:
        reachable = False
    if not reachable:
        state.setdefault('unreachable_since', now)
        down = now - state['unreachable_since']
        if down > HOLD_AFTER_S and not state.get('hold_by_guard'):
            if HOLD.exists():
                log(f'unreachable {down:.0f}s; a hold already exists (not the guard\'s); leaving it')
            else:
                if (LEDGER / 'latest.json').is_file():
                    (LEDGER / 'pre-outage-ledger.json').write_bytes((LEDGER / 'latest.json').read_bytes())
                HOLD.write_text(f'remote outage guard {time.strftime("%Y-%m-%dT%H:%M:%S%z")}: host unreachable since '
                                f'{time.strftime("%H:%M:%S", time.localtime(state["unreachable_since"]))}; resumes after the '
                                'recovery check (runs/setup/tools/remote_outage_guard.py)\n')
                state['hold_by_guard'] = True
                log(f'unreachable for {down:.0f}s: remote admissions held until the recovery check')
        save(state)
        return
    if state.pop('unreachable_since', None) is not None:
        log('reachable again')
    if state.get('hold_by_guard'):
        try:
            blockers, notes = recovery_check(client)
        except RemoteUnreachable:
            log('recovery check interrupted (unreachable); hold kept')
            save(state)
            return
        report = {'checked_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'blockers': blockers, 'notes': notes}
        (LEDGER / 'recovery-report.json').write_text(json.dumps(report, indent=1))
        if blockers:
            (LEDGER / 'recovery-blockers.json').write_text(json.dumps(report, indent=1))
            log(f'RECOVERY BLOCKED {sorted(blockers)}; hold kept')
        else:
            HOLD.rename(ROOT / f'runs/setup/remote-hold-admissions.cleared-{time.strftime("%Y%m%d-%H%M%S")}')
            state['hold_by_guard'] = False
            log(f'recovery check passed ({len(notes)} jobs): hold removed; the driver resumes interrupted jobs')
        save(state)
        return
    if now - state.get('last_ledger', 0) > LEDGER_EVERY_S:
        r = subprocess.run([sys.executable, '-I', str(ROOT / 'runs/setup/tools/remote_ledger.py')], capture_output=True,
                           text=True, timeout=300)
        log((r.stdout or r.stderr).strip()[-200:])
        if r.returncode == 0 and 'LEDGER' in r.stdout:
            state['last_ledger'] = now
    save(state)


if __name__ == '__main__':
    main()
