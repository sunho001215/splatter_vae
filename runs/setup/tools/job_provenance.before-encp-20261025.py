"""Per-job provenance ledger for the 2026-10-23 jobs (pilots, chamfer_lean, pool-placed rows): one record per launch
attempt with the actual code, image, configuration, data and seed (user instruction 2026-10-23: pin and record them for
every job regardless of host). Sources: the scheduler registry (launch events: commit, image, source and job
fingerprints, data identity for remote launches), the run directory's provenance.json / config.yaml (resolved
configuration hash) and the accepted data manifests runs/remote/data/*.json (per-file SHA-256 and size of every
dataset file; for local launches the files the job reads are matched to them and their sizes re-checked).
Writes runs/setup/job-provenance-20261023.json (rebuilt on each call; append-only history is the registry).
    python -I job_provenance.py
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path('/home/ws/ws/splatter4d')
sys.path.insert(0, str(ROOT))
from scripts import jobs  # noqa: E402

PREFIXES = ('chlrm-', 'pfs-', 'pft-', 't1stack-', 'chl-')
DATA = Path('/home/ws/data/metaworld/splatter4d_v1')
manifest_files = {}
for path in sorted((ROOT / 'runs/remote/data').glob('*.json')):
    record = json.loads(path.read_text())
    for f in record.get('files', []):
        manifest_files[f['source']] = {'sha256': f['sha256'], 'bytes': f['bytes'], 'manifest': path.stem}
queue = {j['id']: j for j in jobs.load_queue()['jobs']}
pool = json.loads((ROOT / 'runs/setup/pool-placement.json').read_text())['effective_jobs'] \
    if (ROOT / 'runs/setup/pool-placement.json').is_file() else {}


def task_of(job):
    args = job['args']
    if '--task' in args:
        return args[args.index('--task') + 1]
    cfg = [a for a in args if a.startswith('configs/metaworld/tasks/')]
    return Path(cfg[0]).stem if cfg else None


def seed_of(job):
    args = job['args']
    if '--seed' in args:
        return int(args[args.index('--seed') + 1])
    for a in args:
        if a.startswith('train.seed='):
            return int(a.split('=', 1)[1])
    return None


def run_dir(job):
    args = job['args']
    if '--run-dir' in args:
        return ROOT / args[args.index('--run-dir') + 1]
    if '--output-dir' in args:
        return ROOT / args[args.index('--output-dir') + 1]
    if '--name' in args:
        return ROOT / args[args.index('--output-root') + 1] / args[args.index('--name') + 1]
    if '--out' in args:
        return (ROOT / args[args.index('--out') + 1]).parent
    return ROOT / 'runs' / job['id']


records = []
for line in jobs.REGISTRY.read_text().splitlines():
    event = json.loads(line)
    if event.get('event') not in ('launched', 'launching') or not event.get('id', '').startswith(PREFIXES):
        continue
    if event.get('time', 0) < 1792677600:  # 2026-10-23 00:00 KST
        continue
    job = queue.get(event['id'])
    if job is None:
        continue
    task = task_of(job)
    data = None
    if job['script'] == 'scripts/train.py' and task:
        files = [DATA / f'{task}.hdf5', DATA / f'heldout_sets/{task}.hdf5', DATA / f'workspace_stats/{task}.json']
        data = {str(f): {**manifest_files.get(str(f), {}), 'bytes_now': f.stat().st_size if f.exists() else None}
                for f in files}
    elif job['script'] in ('scripts/train_rl.py', 'scripts/eval_rl.py'):
        data = 'Meta-World simulated live (no dataset file); export: ' + next(
            (a.split('=', 1)[1] for a in queue.get(event['id'].removesuffix('-eval'), job)['args']
             if a.startswith('vision.export_path=')), '?')
    rd = run_dir(job)
    prov = rd / 'provenance.json'
    config = rd / 'config.yaml'
    records.append({
        'id': event['id'], 'attempt': event.get('attempt'), 'host': event.get('host', 'local'),
        'commit': event.get('commit'), 'image_digest': event.get('image_digest', 'local host environment'),
        'source_sha256': hashlib.sha256(json.dumps(event['source_sha256'], sort_keys=True).encode()).hexdigest()
        if 'source_sha256' in event else None,
        'job_sha256': event.get('job_sha256'), 'seed': seed_of(job), 'task': task,
        'config_sha256': json.loads(prov.read_text()).get('config_sha256') if prov.is_file() else None,
        'config_file_sha256': hashlib.sha256(config.read_bytes()).hexdigest() if config.is_file() else None,
        'data_identity': event.get('data_identity'), 'data_files': data,
        'placement': pool.get(event['id'], {}).get('placement'),
    })
out = ROOT / 'runs/setup/job-provenance-20261023.json'
out.write_text(json.dumps({'records': records}, indent=1) + '\n')
print(len(records), out)
