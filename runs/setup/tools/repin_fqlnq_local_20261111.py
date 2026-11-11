"""Re-pin never-launched rwnq/rwa100 FQL rows to the adopted LOCAL release before explicit local placement (2026-11-11).

Why: the rows were registered pinned to the remote role release 8299442. Main then adopted 377c4f5, which changed
s4d/rwlearn/inference.py. That file is inside scripts/train_realworld_fql.py's execution scope, although training
only imports the unchanged ``checkpoint_payload``. ops/jobs/realworld_local.py therefore refuses local placement
("local code differs from the release the row was placed with").

Rows chosen for LOCAL execution get the unified pool's local-release pin, as register_ours_realworld_20261111.py
--release-source local does: planned_commit = main HEAD (passing docs/tests.json for its exact sources),
evidence 'docs/tests.json (local evidence)', release_source 'local' (the remote driver never takes them) and
local_placement_pause. A ``repinned_from`` record keeps the old pin. Arguments, datasets and setting are unchanged.

Usage: repin_fqlnq_local_20261111.py ID [ID ...] [--execute]   (dry run by default)
Runs under stage2-driver.lock, then registry.lock. Refuses any id that is not rwnq-/rwa100-20261111-fql-*, has any
registry event, or is held by a pause other than its own holding pause.
"""
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO = Path('/home/ws/ws/splatter4d')
QUEUE = REPO / 'experiments/queue.yaml'
REGISTRY = REPO / 'experiments/registry.jsonl'
PAUSE = {'rwnq-20261111-fql-': 'fqlnq-await-placement-20261111', 'rwa100-20261111-fql-': 'fqla100-after-nq-20261111'}
CHANGED = {'planned_commit', 'evidence_manifest', 'release_source', 'local_placement_pause', 'repinned_from'}


def holding_pause(job_id):
    for prefix, name in PAUSE.items():
        if job_id.startswith(prefix):
            return name
    raise SystemExit(f'{job_id}: not an rwnq/rwa100 2026-11-11 FQL row')


def main(ids, execute):
    head = subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], check=True, capture_output=True,
                          text=True).stdout.strip()
    tests = json.loads((REPO / 'docs/tests.json').read_text())
    counts = tests.get('counts', {})
    assert tests.get('full_suite_passed') and counts.get('failed') == counts.get('error') == counts.get('skipped') == 0, \
        'docs/tests.json is not a passing full suite'
    with (REPO / 'runs/setup/stage2-driver.lock').open('a') as admission:
        fcntl.flock(admission, fcntl.LOCK_EX)
        with (REPO / 'experiments/registry.lock').open('a') as registry:
            fcntl.flock(registry, fcntl.LOCK_EX)
            seen = set()
            for line in REGISTRY.read_text().splitlines():
                try:
                    seen.add(json.loads(line).get('id'))
                except ValueError:
                    pass
            text = QUEUE.read_text()
            doc = yaml.safe_load(text)
            jobs = {j['id']: j for j in doc['jobs']}
            for job_id in ids:
                holding_pause(job_id)
                assert job_id in jobs, f'{job_id} not in the queue'
                assert job_id not in seen, f'{job_id} has registry events (launched/attempted): never re-pin'
                assert jobs[job_id].get('release_source') != 'local', f'{job_id} already local'
            before = {j['id']: dict(j) for j in doc['jobs']}
            stamp = time.strftime('%Y-%m-%dT%H:%M:%S%z')
            lines = text.splitlines(keepends=True)
            for job_id in ids:
                starts = [i for i, line in enumerate(lines) if line.rstrip('\n') in (f'- id: {job_id}', f'  id: {job_id}')]
                assert len(starts) == 1, (job_id, starts)
                start = starts[0]
                while not lines[start].startswith('- '):
                    start -= 1
                end = start + 1
                while end < len(lines) and not lines[end].startswith('- ') and not lines[end][:1].isalpha():
                    end += 1
                block = lines[start:end]
                old_commit = jobs[job_id].get('planned_commit')
                old_evidence = jobs[job_id].get('evidence_manifest')
                for k, line in enumerate(block):
                    if line.startswith('  planned_commit: '):
                        block[k] = f'  planned_commit: {head}\n'
                    elif line.startswith('  evidence_manifest: '):
                        block[k] = '  evidence_manifest: docs/tests.json (local evidence)\n'
                block += [f'  release_source: local\n', f'  local_placement_pause: {holding_pause(job_id)}\n',
                          '  repinned_from:\n', f'    planned_commit: {old_commit}\n',
                          f'    evidence_manifest: {old_evidence}\n', f"    at: '{stamp}'\n",
                          '    reason: local placement (unified pool); 377c4f5 changed s4d/rwlearn/inference.py inside '
                          'the FQL execution scope; training numerics unchanged\n']
                lines[start:end] = block
            out = ''.join(lines)
            new_jobs = {j['id']: j for j in yaml.safe_load(out)['jobs']}
            assert len(new_jobs) == len(before)
            for job_id, old in before.items():
                new = new_jobs[job_id]
                if job_id in ids:
                    diff = {k for k in set(old) | set(new) if old.get(k) != new.get(k)}
                    assert diff == CHANGED, (job_id, diff)
                    assert new['planned_commit'] == head and new['release_source'] == 'local'
                    assert new['repinned_from']['planned_commit'] == old.get('planned_commit')
                else:
                    assert old == new, job_id
            rest_old = {k: v for k, v in doc.items() if k != 'jobs'}
            rest_new = {k: v for k, v in yaml.safe_load(out).items() if k != 'jobs'}
            assert rest_old == rest_new
            print(json.dumps({'head': head, 'ids': ids, 'jobs': len(before), 'execute': execute}, indent=1))
            if not execute:
                return
            backup = QUEUE.with_name('queue.yaml.before-fqlnq-local-repin-' + time.strftime('%Y%m%d-%H%M%S'))
            shutil.copy2(QUEUE, backup)
            tmp = QUEUE.with_name(QUEUE.name + '.tmp-fqlnq-repin')
            tmp.write_text(out)
            with tmp.open('rb') as handle:
                os.fsync(handle.fileno())
            shutil.copymode(QUEUE, tmp)
            os.replace(tmp, QUEUE)
            print(f'written; backup {backup}; queue sha256 {hashlib.sha256(QUEUE.read_bytes()).hexdigest()}')


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if a != '--execute']
    main(args, '--execute' in sys.argv[1:])
