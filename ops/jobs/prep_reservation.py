"""Read-only capacity priority for the existing remote prep hold writer.

No persisted campaign reservations: prep's own admission remains unaffected.
Read durable intents again at planning/replay/dispatch; live containers are
already charged as co-tenants. Unreadable or ambiguous evidence fails closed.
"""
import json
import math
from pathlib import Path
import shlex

from ops.jobs import admission_pause
from ops.jobs.rwprep_mig import digest, validate


GATES = '''
import json,pathlib,sys
out=[]
for j in json.loads(sys.argv[1]):
    if not all(pathlib.Path(p).is_file() for p in j.get('requires',[])):continue
    if not all(pathlib.Path(p).is_file() and all(json.loads(pathlib.Path(p).read_text())['jobs'].get(i,{}).get('status')=='done' for i in ids) for p,ids in j.get('external_states',{}).items()):continue
    out.append(j['id'])
print(json.dumps(out))
'''


def ready_rows(snapshot, client=None):
    """Mirror rwprep_remote.tick readiness, checking artifact gates on their host."""
    rows, pending, seen = [], [], set()
    # A formerly blocked batch can become ready before its next writer tick.
    # Its inactive hold still locates the immutable jobs and current intents.
    for _, hold in admission_pause.prep_holds(active_only=False):
        if hold.get('host') != 'remote':
            continue
        if snapshot is None:
            raise ValueError('prep reservation requires remote resource evidence')
        root = Path(hold['state_dir']).resolve()
        if root in seen:
            continue
        seen.add(root)
        config = json.loads((root.parent / 'dispatcher.json').read_text())
        jobs = json.loads(Path(config['jobs']).read_text())
        validate(config, jobs)
        if config['host'] != 'remote' or Path(config['state_dir']).resolve() != root:
            raise ValueError('prep hold/config differ')
        state = json.loads((root / 'state.json').read_text())
        if state['queue_sha256'] != digest(config['jobs']):
            raise ValueError('prep immutable queue changed')
        states = state['jobs']
        if any(s['status'] == 'unknown' for s in states.values()):
            raise ValueError('unknown prep launch intent')
        for j in jobs:
            s = states.get(j['id'], {'status': 'pending'})
            status = s['status']
            if status in ('running', 'launching'):
                if s['container'] in snapshot.cotenants:
                    continue  # verified live quota already charged; never double count
                if status == 'running':
                    raise ValueError('running prep container missing from resource evidence')
            elif (status not in ('pending', 'retry') or (root / 'STOP').exists()
                  or not all(states.get(d, {}).get('status') == 'done' for d in j.get('deps', []))):
                continue
            row = dict(id=str(root) + ':' + j['id'], host='remote', gpu_profile='1g.24gb',
                       gpu=s['gpu'] if status == 'launching' else 'any', mem_gb=j['gpu_gib'],
                       ram_gb=j['ram_gib'], cpu_threads=j['cpus'], disk_gb=j.get('disk_gib', 32))
            if not all(math.isfinite(float(row[k])) and float(row[k]) > 0
                       for k in ('mem_gb', 'ram_gb', 'cpu_threads', 'disk_gb')):
                raise ValueError('invalid prep resource demand')
            if status != 'launching' and (j.get('requires') or j.get('external_states')):
                pending.append((row, j))
            else:
                rows.append(row)
    if pending:
        if client is None:
            from ops.remote.access import RemoteClient
            client = RemoteClient()
        gates = [dict(id=r['id'], **{k: j[k] for k in ('requires', 'external_states') if k in j})
                 for r, j in pending]
        ready = set(json.loads(client.checked_ssh(
            'python3 -I -c ' + shlex.quote(GATES) + ' ' + shlex.quote(json.dumps(gates)),
            idempotent=True).stdout))
        rows.extend(r for r, _ in pending if r['id'] in ready)
    return sorted(rows, key=lambda r: r['gpu'] == 'any')  # durable, pinned launch intents first


def reserve(policy, snapshot, client=None):
    """Reserve distinct MIGs plus full host demand before any campaign or pool row.

    Prefer empty devices. If there are more ready jobs than available devices,
    keep all remaining devices and the full host demand reserved. No live job
    is stopped or moved. These reservations outrank every campaign priority.
    """
    rows = ready_rows(snapshot, client)
    policy.reservations = {k: v for k, v in policy.reservations.items() if not k.startswith('prep:')}
    used, slots = set(), policy.slots()
    for row in rows:
        devices = policy.devices(row)
        if not devices:
            raise ValueError('ready prep has no compatible remote MIG')
        available = [g for g in devices if g not in used]
        gpu = min(available, key=lambda g: (slots.get(g, {}).get('jobs', 0) != 0,
                  slots.get(g, {}).get('mem_gb', 0), list(policy.limits).index(g))) if available else None
        used.add(gpu)
        policy.reservations['prep:' + row['id']] = dict(
            gpu=gpu, priority=float('-inf'), need={k: row[k] for k in ('ram_gb', 'cpu_threads', 'disk_gb')})
    return policy
