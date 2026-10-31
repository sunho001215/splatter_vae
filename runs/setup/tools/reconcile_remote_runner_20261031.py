"""Hand control back from the remote-resident runner (remote_runner.py, local outage from 2026-10-31 23:00 KST) to the
local scheduler. Run this BEFORE restarting any local controller (scheduler daemon, remote/local drivers, guard).

    python -I reconcile_remote_runner_20261031.py --stop      # create ~/kaist/sunho/remote_runner/STOP, wait for exit
    python -I reconcile_remote_runner_20261031.py             # dry run: print the registry events it would write
    python -I reconcile_remote_runner_20261031.py --apply     # write them (under the registry lock)

For every `launched` event of the runner (events.jsonl, downloaded with checksum verification): rebuilds the launch
spec with the GPU the runner actually used (RemoteBackend.prepare, runs/remote/jobs/<id>-a<attempt>.json, exactly as
jobs.tick would have written it) and records the `launched` registry event jobs.tick records for a remote launch
(container id from the runner). Afterwards the scheduler daemon observes these containers like any remote job
(exit, result sync, verification). Events already present (same id and attempt launched) are skipped; a container
the registry already knows with a different attempt is reported and nothing is written for it.
"""
import argparse
import fcntl
import json
import sys
import time
from pathlib import Path

ROOT = Path('/home/ws/ws/splatter4d')
sys.path.insert(0, str(ROOT))
from scripts import jobs  # noqa: E402
from s4d.remote_access import RemoteClient  # noqa: E402
from s4d.remote_jobs import RemoteBackend  # noqa: E402

PLAN = ROOT / 'runs/remote/autonomy-20261031/plan.json'
IMAGE = 'sha256:59082834c2c2c0d688be50a0b3b362bc46bdd2b1923d8ab712a473a0bd51870a'
ap = argparse.ArgumentParser()
ap.add_argument('--stop', action='store_true')
ap.add_argument('--apply', action='store_true')
args = ap.parse_args()
client = RemoteClient()
base = '/home/compu/kaist/sunho/remote_runner'
if args.stop:
    client.checked_ssh(f'touch {base}/STOP', timeout=60)
    for _ in range(30):
        out = client.ssh(f'pgrep -f "python3 remote_runner.py" || true', timeout=60)
        if not out.stdout.strip():
            print('runner stopped')
            break
        time.sleep(10)
    else:
        raise SystemExit('runner still running after 5 minutes; investigate before continuing')
local = ROOT / 'runs/remote/autonomy-20261031/events-downloaded.jsonl'
client.rsync_download(f'{base}/events.jsonl', local)
plan = {item['id']: item for item in json.loads(PLAN.read_text())['items']}
launches = [e for e in map(json.loads, local.read_text().splitlines()) if e.get('event') == 'launched']
queue = jobs.load_queue()
by_id = {j['id']: j for j in queue['jobs']}
backend = RemoteBackend(ROOT, ROOT / 'runs')
manifests = {}
pending = []
with jobs.REGISTRY.with_suffix('.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    state = jobs.read_registry()
    for e in launches:
        job = {**by_id[e['id']], 'host': 'remote'}
        cur = state.get(e['id'], {})
        known = cur.get('attempt_containers', {}).get(str(e['attempt']))
        if known:
            print('already recorded', e['id'], e['attempt'])
            continue
        if cur.get('attempts', 0) >= e['attempt']:
            print('CONFLICT: registry attempt', cur.get('attempts'), 'vs runner attempt', e['attempt'], e['id'])
            continue
        commit = plan[e['id']]['commit']
        manifest = f'runs/remote/acceptance/{commit[:7]}-cc/manifest.json'
        host = {**queue['hosts']['remote'], 'enabled': True, 'commit': commit, 'image_digest': IMAGE,
                'evidence_manifest': manifest}
        if commit not in manifests:
            manifests[commit] = backend.ready(host)
        evidence = manifests[commit]
        path, spec = backend.prepare(job, e['gpu'], e['attempt'], host, evidence)
        assert spec['container'] == e['container'], (spec['container'], e['container'])
        need = float(job.get('ram_gb', 0))
        event = {'id': job['id'], 'host': 'remote', 'gpu': e['gpu'], 'attempt': e['attempt'],
                 'container': spec['container'], 'commit': spec['commit'], 'image_digest': spec['image_digest'],
                 'gpu_backend': spec['gpu_backend'], 'data_identity': evidence['data_identity'],
                 'evidence_sha256': evidence['manifest_sha256'], 'spec_path': str(path),
                 'required_results': spec['required_results'], 'script': job['script'],
                 'args': list(map(str, job.get('args', []))), 'job_sha256': jobs.job_fingerprint(job),
                 'expected_config_sha256': job.get('expected_config_sha256'), 'mem_gb': float(job['mem_gb']),
                 'ram_gb': need, 'disk_gb': float(job.get('disk_gb', 0)), 'container_id': e['container_id'],
                 'launched_by': 'remote_runner (local outage 2026-10-31)', 'runner_time': e['time']}
        pending.append({'event': 'launched', **event})
    print(json.dumps({'to_record': [(p['id'], p['attempt'], p['gpu'][4:8]) for p in pending]}, indent=1))
    if args.apply:
        for p in pending:
            jobs.record(p)
        print('recorded', len(pending))
