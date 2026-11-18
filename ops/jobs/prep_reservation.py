"""Read-only prep priority for pinned intents and ready rows that fit current MIG capacity.

Reserve full host demand through the first budget overflow, then defer later unpinned rows.
"""
import json
import math
import shlex
from pathlib import Path

from ops.jobs import admission_pause
from ops.jobs.priority import HOST_RESOURCES
from ops.jobs.rwprep_mig import digest, validate

GATES = '''
import json,pathlib,sys
out=[]
for j in json.loads(sys.argv[1]):
    if not all(pathlib.Path(p).is_file() for p in j.get('requires',[])):continue
    if not all(pathlib.Path(p).is_file() and all(
        json.loads(pathlib.Path(p).read_text())['jobs'].get(i,{}).get('status')=='done' for i in ids)
        for p,ids in j.get('external_states',{}).items()):continue
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
        live_names = None  # recheck after reading this batch's durable intents
        if any(s['status'] == 'unknown' for s in states.values()):
            raise ValueError('unknown prep launch intent')
        for j in jobs:
            s = states.get(j['id'], {'status': 'pending'})
            status = s['status']
            if status in ('running', 'launching'):
                if s['container'] in snapshot.cotenants:
                    continue  # verified live quota already charged; never double count
                if status == 'running':
                    # Reconciliation lags container exit. Recheck without tenant
                    # filters: a launch after the snapshot still needs its quota.
                    if live_names is None:
                        if client is None:
                            from ops.remote.access import RemoteClient
                            client = RemoteClient()
                        live_names = set(client.checked_ssh(
                            "docker ps --format '{{.Names}}'", idempotent=True).stdout.splitlines())
                    if s['container'] not in live_names:
                        continue  # finished/transient; leave reconciliation to prep
                    status = 'launching'  # live but uncharged: reserve its pinned demand
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
    """Reserve pinned demand and place ready prep against current device occupancy.

    Keep full host demand through the first budget overflow, skipping later unpinned rows.
    """
    rows = ready_rows(snapshot, client)
    policy.reservations = {k: v for k, v in policy.reservations.items() if not k.startswith('prep:')}
    live = policy.slots()
    slots = {g: dict(jobs=live.get(g, {}).get('jobs', 0),
                     mem_gb=live.get(g, {}).get('mem_gb', 0)) for g in policy.limits}
    free, held = policy.remaining(), dict.fromkeys(HOST_RESOURCES, 0.0)
    host_full = False
    for row in rows:
        devices = policy.devices(row)
        if not devices:
            raise ValueError('ready prep has no compatible remote MIG')
        if row['gpu'] != 'any':
            gpu = row['gpu']
        else:
            if host_full:
                continue
            available = [g for g in devices if slots[g]['jobs'] < int(policy.limits[g]['max_jobs'])
                         and slots[g]['mem_gb'] + float(row['mem_gb']) <= float(policy.limits[g]['max_mem_gb'])]
            if not available:
                continue
            gpu = min(available, key=lambda g: (slots[g]['jobs'] != 0, slots[g]['mem_gb'],
                                              list(policy.limits).index(g)))
        need = {k: row[k] for k in HOST_RESOURCES}
        policy.reservations['prep:' + row['id']] = dict(gpu=gpu, priority=float('-inf'), need=need)
        slots[gpu]['jobs'] += 1
        slots[gpu]['mem_gb'] += float(row['mem_gb'])
        for k in HOST_RESOURCES:
            held[k] += float(need[k])
        host_full = any(held[k] > free.get(k, 0) for k in HOST_RESOURCES)
    return policy
