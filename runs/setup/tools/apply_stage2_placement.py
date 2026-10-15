"""Apply the user's 2026-10-15 host-neutral placement decision to existing, unstarted jobs.

Scientific registrations remain immutable. The placement manifest stores original and effective job definitions;
only host, runtime paths, explicit resource reservations and remote result requirements change. No new comparison,
seed, model or hyperparameter is introduced. Run once, after both drivers support the manifest.
"""
import copy
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
from scripts import jobs  # noqa: E402

SETUP = ROOT / 'runs/setup'
MANIFEST = SETUP / 'stage2-placement-20261015.json'
VARIANTS = ('nolambdadyn', 'nocoverage', 'misscaled-none', 'misscaled-scale')
TASKS = ('hammer', 'pick-place')
trainers = {f's2ab-{v}-seed0-{t}' for v in VARIANTS for t in TASKS}
evaluations = {f'fulleval-{i}-200k' for i in trainers}
rl = {f's2rl-{v}-{t}-s{s}{suffix}' for v in ('crop', 'cropnosynth') for t in TASKS
      for s in (1000, 1001, 1002) for suffix in ('', '-eval')}
wanted = trainers | evaluations | rl

with jobs.REGISTRY.with_suffix('.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    queue = jobs.load_queue()
    if MANIFEST.exists():
        existing = json.loads(MANIFEST.read_text())
        assert all(j == existing['effective_jobs'].get(j['id'], j) for j in queue['jobs'])
        print('ALREADY_PLACED', len(existing['effective_jobs']))
        raise SystemExit(0)
    state = jobs.read_registry()
    assert all(state.get(i, {}).get('status', jobs.PENDING) == jobs.PENDING for i in wanted)
    originals = {j['id']: copy.deepcopy(j) for j in queue['jobs'] if j['id'] in wanted}
    assert set(originals) == wanted
    effective = copy.deepcopy(originals)
    for jid, job in effective.items():
        job.update(host='remote', gpu='any', cpu_threads=8)
        args = job['args']
        if jid in trainers:
            args[args.index('--output-root') + 1] = 'runs'
            job.update(ram_gb=16, disk_gb=30,
                       required_results=['completion.json', 'metrics.jsonl', 'checkpoints/step_0200000.pt'])
        elif jid in evaluations:
            source = job['deps'][0]
            args[args.index('--checkpoint') + 1] = f'runs/{source}/checkpoints/step_0200000.pt'
            job.update(ram_gb=12, disk_gb=3,
                       required_results=['eval/step_0200000/summary.json'])
        elif jid.endswith('-eval'):
            args += ['--output-dir', f'runs/{jid}']
            job.update(disk_gb=3, required_results=['eval.jsonl'])
        else:
            job.update(disk_gb=5, required_results=['final.json'])
        assert job['script'] == originals[jid]['script'] and job['deps'] == originals[jid]['deps']
    before = jobs.QUEUE.read_bytes()
    nodes = next(v for k, v in yaml.compose(before.decode()).value if k.value == 'jobs')
    edits = []
    for node in nodes.value:
        jid = next(v.value for k, v in node.value if k.value == 'id')
        if jid in effective:
            # Replace only this mapping, retaining the queue's surrounding YAML and unrelated registrations.
            start = before.decode().rfind('\n', 0, node.start_mark.index) + 1
            end = node.end_mark.index
            text = yaml.safe_dump([effective[jid]], sort_keys=False)
            edits.append((start, end, text))
    text = before.decode()
    for start, end, replacement in sorted(edits, reverse=True):
        text = text[:start] + replacement + text[end:]
    candidate = SETUP / 'stage2-placement-queue-candidate-20261015.yaml'
    candidate.write_text(text)
    combined = jobs.load_queue(candidate)
    assert combined['jobs'] == [effective.get(j['id'], j) for j in queue['jobs']]
    assert {k: v for k, v in combined.items() if k != 'jobs'} == {k: v for k, v in queue.items() if k != 'jobs'}
    record = {'recorded_local': datetime.now().astimezone().isoformat(),
              'decision': 'user 2026-10-15: hosts equivalent; no same-host/equivalence/migration experiment gates',
              'purpose': 'finish registered Stage 2 representation attribution and crop RL during local downtime',
              'reporting': 'unchanged registered representation margins; success and return last5/final/AUC by camera set',
              'original_jobs': originals, 'effective_jobs': effective,
              'queue_sha256_before': hashlib.sha256(before).hexdigest(),
              'queue_sha256_after': hashlib.sha256(text.encode()).hexdigest(),
              'scientific_configs_unchanged': True}
    (SETUP / 'stage2-placement-queue-before-20261015.yaml').write_bytes(before)
    temporary = jobs.QUEUE.with_suffix('.placement.tmp')
    temporary.write_text(text)
    os.replace(temporary, jobs.QUEUE)
    MANIFEST.write_text(json.dumps(record, indent=2) + '\n')
    for jid in effective:
        jobs.record({'event': 'placement_registered', 'id': jid, 'host': 'remote',
                     'previous_host': 'local', 'reason': record['decision'],
                     'placement_manifest': str(MANIFEST.relative_to(ROOT))})
    print('PLACED', len(effective), 'jobs', MANIFEST, record['queue_sha256_after'])
