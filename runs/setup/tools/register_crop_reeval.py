"""Register the crop follow-up step 1 (user request 2026-10-15, report only): re-evaluate the existing 100k
checkpoints of the 2026-10-09 crop screen and its base reference (hammer, pick-place) with the current evaluator.
Checkpoints are only read; each evaluation writes to its own new run directory. Local; admitted job by job by the
Stage 2 local driver under the local full-charge cap. Idempotent.

    python -I register_crop_reeval.py
"""
import fcntl
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path('/home/ws/ws/splatter4d')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'analysis'))
from eval_config import prepare  # noqa: E402

from scripts import jobs  # noqa: E402

TAG = '20261015'
# source pretraining run -> evaluation job (M2D era, seed 0, base 200k schedule; the crop run stopped at 100k)
SOURCES = {f's1-pretrain-{t}-base': f'creval-base-{t}-100k' for t in ('hammer', 'pick-place')}
SOURCES.update({f's2-screen-crop-{t}': f'creval-crop-{t}-100k' for t in ('hammer', 'pick-place')})


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


new_jobs, checkpoints = [], {}
for source, job_id in SOURCES.items():
    checkpoint = f'runs/pretrain/{source}/checkpoints/step_0100000.pt'
    checkpoints[job_id] = {'source_run': source, 'checkpoint': checkpoint, 'sha256': sha(ROOT / checkpoint),
                           'source_config_sha256': sha(ROOT / f'runs/pretrain/{source}/config.yaml')}
    new_jobs.append({
        'id': job_id, 'script': 'scripts/evaluate.py',
        'args': ['--run', f'runs/{job_id}', '--checkpoint', checkpoint],
        'gpu': 'any', 'mem_gb': 5, 'ram_gb': 12, 'priority': 1, 'deps': [], 'max_restarts': 1, 'oom_score_adj': 300,
        'crop_followup': 'step1-reeval', 'report_only': True})

proof_path = ROOT / f'runs/setup/crop-reeval-registration-{TAG}.json'
queue_path = jobs.QUEUE
with jobs.REGISTRY.with_suffix('.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    queue = jobs.load_queue()
    old = {j['id']: j for j in queue['jobs']}
    existing = [j for j in new_jobs if j['id'] in old]
    if existing:
        assert len(existing) == len(new_jobs) and all(old[j['id']] == j for j in new_jobs) and proof_path.is_file()
        print('ALREADY_REGISTERED', len(existing))
        raise SystemExit(0)
    assert not proof_path.exists()
    assert not any((ROOT / 'runs' / j['id']).exists() for j in new_jobs)
    for job in new_jobs:  # evaluation run directory: the source config with the evaluation's run name
        prepare(ROOT / 'runs/pretrain' / checkpoints[job['id']]['source_run'], ROOT / 'runs' / job['id'])
    before = queue_path.read_bytes()
    candidate = ROOT / f'runs/setup/crop-reeval-queue-candidate-{TAG}.yaml'
    assert not candidate.exists()
    text = before.decode()
    jobs_node = next(value for key, value in yaml.compose(text).value if key.value == 'jobs')
    boundary = jobs_node.end_mark.index
    candidate.write_text(text[:boundary].rstrip() + '\n' + yaml.safe_dump(json.loads(json.dumps(new_jobs)),
                                                                          sort_keys=False) + '\n' + text[boundary:])
    combined = jobs.load_queue(candidate)
    assert combined['jobs'] == queue['jobs'] + new_jobs
    assert {k: v for k, v in combined.items() if k != 'jobs'} == {k: v for k, v in queue.items() if k != 'jobs'}
    snapshot = ROOT / f'runs/setup/crop-reeval-queue-before-{TAG}.yaml'
    assert not snapshot.exists()
    snapshot.write_bytes(before)
    temporary = queue_path.with_suffix('.crop-reeval.tmp')
    assert not temporary.exists()
    temporary.write_bytes(candidate.read_bytes())
    os.replace(temporary, queue_path)
    proof = {'registered_at': datetime.now().astimezone().isoformat(), 'request': 'crop follow-up step 1 (user, '
             '2026-10-15): report only; M2D-era single-seed results; no change to existing runs',
             'source_commit': jobs.source_identity()['commit'],
             'queue_sha256_before': hashlib.sha256(before).hexdigest(), 'queue_sha256_after': sha(queue_path),
             'checkpoints_read_only': checkpoints,
             'eval_configs': {j['id']: sha(ROOT / 'runs' / j['id'] / 'config.yaml') for j in new_jobs},
             'compatibility': 'all four checkpoints load strictly (every encoder and decoder key matched) into the '
                              'current model built from their own config, checked on CPU before registration',
             'jobs': new_jobs, 'admission': 'Stage 2 local driver, job by job under the local full-charge cap'}
    proof_path.write_text(json.dumps(proof, indent=2) + '\n')
    print('REGISTERED', len(new_jobs), proof_path, sha(proof_path))
