"""Admit exactly one named, ready local row through the scheduler (jobs.tick on a one-row queue file, as the local
drivers do), while the local admission loops are stopped for the shutdown of 2026-10-31 23:00 KST. Used for the
self-render v4 bin-picking export (a seconds-long job) so its encoder can be uploaded for the remote FlashSAC rows.
Refuses unless: the row is local, pending, every dependency is done, the sources match docs/tests.json.

    python -I admit_one_local_20261031.py <job id> <MIG uuid>
"""
import copy
import fcntl
import json
import sys
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path('/home/ws/ws/splatter4d')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/setup/tools'))
from scripts import jobs  # noqa: E402
import pool_placement as pool  # noqa: E402

job_id, gpu = sys.argv[1], sys.argv[2]
with open(pool.ADMISSION_LOCK, 'w') as admission:
    fcntl.flock(admission, fcntl.LOCK_EX)
    queue = jobs.load_queue()
    by_id = {j['id']: j for j in queue['jobs']}
    state = jobs.read_registry()
    job = copy.deepcopy(by_id[job_id])
    assert job.get('host', 'local') == 'local', job.get('host')
    assert state.get(job_id, {}).get('status', jobs.PENDING) == jobs.PENDING, state.get(job_id)
    assert all(state.get(d, {}).get('status') == jobs.DONE for d in job.get('deps', [])), job.get('deps')
    source = jobs.source_identity()
    report = json.loads((ROOT / 'docs/tests.json').read_text())
    assert source['source_sha256'] == report['source_sha256'] and report['full_suite_passed'] is True
    assert gpu in queue['hosts']['local']['limits'], gpu
    job['gpu'] = gpu
    tag = datetime.now().strftime('%Y%m%d-%H%M%S')
    qpath = ROOT / 'runs/setup' / f'admit-one-local-{tag}.yaml'
    host = copy.deepcopy(queue['hosts']['local'])
    with qpath.open('x') as stream:
        yaml.safe_dump({'hosts': {'local': host}, 'jobs': [job]}, stream, sort_keys=False)
    for message in jobs.tick(queue_path=qpath):
        print(message)
