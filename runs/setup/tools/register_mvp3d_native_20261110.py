"""One-off native 3D-MVP registration, 2026-11-10. Default: stdout-only DRY RUN.

--execute publishes the queue and protective pause under stage2-driver.lock then
registry.lock, with the newtasks registrar's candidate/journal/proof convention.
No scheduler, campaign, launch API, network or registry writes are used. A pending
journal after an interruption requires inspection before retrying; the pause is
published BEFORE the queue, so an interrupted publication fails closed.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import re
import sys
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
import yaml  # noqa: E402

ROOT = Path('/home/ws/ws/splatter4d')
ROW_SPEC = Path('/home/ws/ws/realworld-ops/serving/evidence/mvpfix/native-rows.json')
TASKS = ('drawer-open', 'mug-hanging')
CONDITIONS = ('mvp3d_single', 'mvp3d_multiview')
STEM = 'rw-registration-mvp3d-native-20261110'
PAUSE_NAME = 'mvp3d-native-fql-feature-check-20261110'
PAUSE_REASON = "held until the cell's native fqldata passes the standardised A1 feature check"
PAUSE_PREFIXES = [f'rwcal-20261110-fql-{c}-{t}-' for c in CONDITIONS for t in TASKS]
QUEUE = 'experiments/queue.yaml'
REGISTRY = 'experiments/registry.jsonl'
PAUSES = 'runs/setup/admission-pauses.json'


class Refused(RuntimeError):
    pass


def require(ok, why):
    if not ok:
        raise Refused(why)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def json_sha(value):
    return sha(json.dumps(value, sort_keys=True).encode())


def json_bytes(value):
    return (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()


def now():
    return datetime.now(timezone.utc).isoformat()


def expected_graph():
    graph = {}
    for task in TASKS:
        stage = f'rwcal-20261110-nativedepth-mvp3d-{task}'
        pre = f'rwcal-20261110-mvp3d-{task}-pre-native'
        graph[stage], graph[pre] = [], [stage]
        for condition in CONDITIONS:
            data = f'rwcal-20261110-fqldata-{condition}-{task}-native'
            graph[data] = [pre]
            for seed in range(3):
                graph[f'rwcal-20261110-fql-{condition}-{task}-s{seed}'] = [data]
    return graph


def kind(row):
    return {'scripts/realworld_native_depth.py': 'native-depth',
            'scripts/train_mvp3d.py': 'pretraining',
            'scripts/realworld_fql_dataset.py': 'fqldata',
            'scripts/train_realworld_fql.py': 'FQL'}.get(row.get('script'))


def pinned_rows(raw, commit, image, evidence_manifest=None):
    require(re.fullmatch(r'[0-9a-f]{40}', commit or ''), '--commit needs a full 40-hex sha')
    require(re.fullmatch(r'sha256:[0-9a-f]{64}', image or ''), '--image needs sha256:<64 hex>')
    # register_realworld_newtasks_20261110.adapt_rows uses this exact convention.
    evidence = evidence_manifest or f'runs/remote/acceptance/{commit[:7]}-cc/manifest.json'
    path = Path(evidence)
    require(not path.is_absolute() and '..' not in path.parts and path.suffix == '.json'
            and '<' not in evidence and str(path) == evidence, 'invalid --evidence-manifest repo-relative JSON path')
    doc = json.loads(raw)
    require(isinstance(doc, dict), 'native-rows.json must be an object with the expected 20 ids/deps chain')
    rows = doc.get('rows')
    graph = expected_graph()
    require(isinstance(rows, list) and len(rows) == 20
            and all(isinstance(r, dict) and isinstance(r.get('id'), str) for r in rows),
            'native-rows.json needs exactly the expected 20 ids/deps chain')
    require(len({r['id'] for r in rows}) == 20 and {r['id'] for r in rows} == set(graph),
            'native-rows.json differs from the expected 20 ids/deps chain')
    rows = copy.deepcopy(rows)
    seen = set()
    for row in rows:
        job = row['id']
        require(row.get('deps') == graph[job] and set(graph[job]) <= seen,
                f'wrong deps chain or ordering: {job}')
        seen.add(job)
        task = next(t for t in TASKS if t in job)
        expected_kind = ('native-depth' if '-nativedepth-' in job else 'pretraining' if '-pre-native' in job
                         else 'fqldata' if '-fqldata-' in job else 'FQL')
        require(kind(row) == expected_kind and row.get('realworld_target') == task,
                f'wrong script/task: {job}')
        args = row.get('args')
        require(isinstance(args, list) and all(isinstance(a, str) for a in args), f'invalid args: {job}')
        if expected_kind == 'FQL':
            require('encoder.feature_norm=train_standardize' in args
                    and f"data.dataset=runs/{graph[job][0]}/dataset.h5" in args
                    and row.get('fql_seed') == int(job[-1]) and f'train.seed={job[-1]}' in args,
                    f'FQL must use native fqldata, the expected seed and train standardisation: {job}')
        row.update(planned_commit=commit, planned_image=image, evidence_manifest=evidence,
                   registration_tool='runs/setup/tools/' + Path(__file__).name)
    require('<release containing' not in json.dumps(rows), 'unreplaced release placeholder in row specs')
    require(Counter(kind(r) for r in rows) == {'native-depth': 2, 'pretraining': 2, 'fqldata': 4, 'FQL': 12},
            'wrong native row counts')
    return rows


def queue_document(raw):
    doc = yaml.safe_load(raw)
    require(isinstance(doc, dict) and isinstance(doc.get('jobs'), list), 'queue jobs must be a sequence')
    require(all(isinstance(r, dict) and isinstance(r.get('id'), str) for r in doc['jobs']), 'invalid queue row')
    ids = [r['id'] for r in doc['jobs']]
    require(len(ids) == len(set(ids)), 'duplicate job ids in queue')
    return doc


def queue_candidate(before, rows):
    """Same pure text append and parsed-equality proof as the newtasks registrar."""
    parsed = queue_document(before)
    node = next(value for key, value in yaml.compose(before.decode()).value if key.value == 'jobs')
    require(isinstance(node, yaml.SequenceNode), 'queue jobs must be a sequence')
    text = before.decode()
    if node.flow_style or not node.value:
        candidate = text[:node.start_mark.index] + '\n' + yaml.safe_dump(parsed['jobs'] + rows, sort_keys=False) \
            + text[node.end_mark.index:]
    else:
        candidate = text[:node.end_mark.index].rstrip() + '\n' + yaml.safe_dump(rows, sort_keys=False) + '\n' \
            + text[node.end_mark.index:]
    require(yaml.safe_load(candidate) == {**parsed, 'jobs': parsed['jobs'] + rows}, 'queue staging changed existing data')
    return candidate.encode()


def registry_ids(raw):
    ids = set()
    for line in raw.splitlines():
        event = json.loads(line)
        require(isinstance(event, dict) and isinstance(event.get('id'), str), 'invalid registry event')
        ids.add(event['id'])
    return ids


def pause_candidate(before, entry):
    doc = json.loads(before)
    require(isinstance(doc, dict) and isinstance(doc.get('doc'), str) and isinstance(doc.get('pauses'), list),
            'invalid admission pause document')
    require(all(isinstance(p, dict) and isinstance(p.get('name'), str) for p in doc['pauses']),
            'invalid admission pause entry')
    require(not any(p['name'] == PAUSE_NAME for p in doc['pauses']), f'pause already exists: {PAUSE_NAME}')
    return json_bytes({**doc, 'pauses': doc['pauses'] + [entry]})


def sync_directory(directory):
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write(path, raw):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    sync_directory(path.parent)


@contextmanager
def locks(repo):
    with (repo / 'runs/setup/stage2-driver.lock').open('a') as admission:
        fcntl.flock(admission, fcntl.LOCK_EX)
        with (repo / 'experiments/registry.lock').open('a') as registry:
            fcntl.flock(registry, fcntl.LOCK_EX)
            yield


def prepare(repo, rows_path, commit, image, evidence_manifest=None):
    spec = rows_path.read_bytes()
    rows = pinned_rows(spec, commit, image, evidence_manifest)
    setup = repo / 'runs/setup'
    require(not (setup / f'{STEM}.pending.json').exists() and not (setup / f'{STEM}.json').exists(),
            'pending transaction/proof exists; inspect before retrying')
    before = (repo / QUEUE).read_bytes()
    registry = (repo / REGISTRY).read_bytes() if (repo / REGISTRY).exists() else b''
    pauses = (repo / PAUSES).read_bytes()
    current = queue_document(before)
    ids = {r['id'] for r in current['jobs']} | registry_ids(registry)
    require(not ids.intersection(r['id'] for r in rows), 'a proposed ID already exists in queue or registry')
    require(not any((repo / 'runs' / r['id']).exists() for r in rows), 'a run directory already exists')
    entry = {'name': PAUSE_NAME, 'active': True, 'prefixes': list(PAUSE_PREFIXES), 'since': now(),
             'by': Path(__file__).stem, 'reason': PAUSE_REASON}
    paused = {r['id'] for r in rows if any(r['id'].startswith(p) for p in entry['prefixes'])}
    require(paused == {r['id'] for r in rows if kind(r) == 'FQL'}, 'pause must cover exactly the 12 FQL rows')
    candidate = queue_candidate(before, rows)
    pause_after = pause_candidate(pauses, entry)
    return {'schema': 'rw-registration/v1', 'status': 'dry-run', 'registered': False, 'prepared_at': entry['since'],
            'repo': str(repo.resolve()), 'release': {'commit': commit, 'image': image},
            'evidence_manifest': rows[0]['evidence_manifest'],
            'evidence_manifest_convention': 'newtasks registrar: runs/remote/acceptance/<commit[:7]>-cc/manifest.json',
            'jobs': rows, 'jobs_sha256': json_sha(rows), 'pause_entry': entry,
            'pause_document_after': json.loads(pause_after),
            'row_spec': str(rows_path.resolve()), 'row_spec_sha256': sha(spec),
            'tool_sha256': sha(Path(__file__).read_bytes()),
            'registry_exists_at_evaluation': (repo / REGISTRY).exists(),
            'registry_sha256_at_evaluation': sha(registry),
            'queue_sha256_before': sha(before), 'queue_sha256_after': sha(candidate),
            'pauses_sha256_before': sha(pauses), 'pauses_sha256_after': sha(pause_after),
            'queue_diff': {'before_rows': len(current['jobs']), 'after_rows': len(current['jobs']) + 20,
                           'added_rows': 20, 'modified_rows': 0, 'removed_rows': 0,
                           'counts': dict(Counter(kind(r) for r in rows)), 'added_ids': [r['id'] for r in rows],
                           'paused_FQL_rows': len(paused), 'pause_prefixes': len(PAUSE_PREFIXES)}}


def check_unchanged(repo, record):
    require(sha((repo / QUEUE).read_bytes()) == record['queue_sha256_before'], 'queue changed under locks')
    require(sha((repo / PAUSES).read_bytes()) == record['pauses_sha256_before'], 'pause file changed under locks')
    registry = repo / REGISTRY
    require(registry.exists() == record['registry_exists_at_evaluation']
            and sha(registry.read_bytes() if registry.exists() else b'') == record['registry_sha256_at_evaluation'],
            'registry changed under locks')
    require(sha(Path(record['row_spec']).read_bytes()) == record['row_spec_sha256'], 'row specs changed')
    require(sha(Path(__file__).read_bytes()) == record['tool_sha256'], 'registration tool changed')


def execute(repo, record):
    setup = repo / 'runs/setup'
    pending, proof = setup / f'{STEM}.pending.json', setup / f'{STEM}.json'
    candidate_path = setup / f'{STEM}.queue-candidate.yaml'
    with locks(repo):
        require(not pending.exists() and not proof.exists(), 'pending transaction/proof exists; inspect before retrying')
        require(record['repo'] == str(repo.resolve()) and json_sha(record['jobs']) == record['jobs_sha256'],
                'registration plan changed')
        check_unchanged(repo, record)
        before = (repo / QUEUE).read_bytes()
        candidate = queue_candidate(before, record['jobs'])
        pause_after = pause_candidate((repo / PAUSES).read_bytes(), record['pause_entry'])
        require(sha(candidate) == record['queue_sha256_after']
                and sha(pause_after) == record['pauses_sha256_after'], 'staged plan changed')
        atomic_write(candidate_path, candidate)
        record.update(status='pending', proof_file=proof.name)
        atomic_write(pending, json_bytes(record))
        check_unchanged(repo, record)
        # Two files cannot be renamed together. Publishing the hold first fails closed.
        atomic_write(repo / PAUSES, pause_after)
        require((repo / QUEUE).read_bytes() == before, 'queue changed under locks after publishing pause; inspect journal')
        os.replace(candidate_path, repo / QUEUE)
        sync_directory((repo / QUEUE).parent)
        record.update(status='published', registered=True, registered_at=now())
        atomic_write(proof, json_bytes(record))
        pending.unlink()
        sync_directory(setup)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=ROOT)
    parser.add_argument('--rows', type=Path, default=ROW_SPEC)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--evidence-manifest', help='optional explicit repo-relative JSON path; default follows newtasks')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    record = prepare(args.repo, args.rows, args.commit, args.image, args.evidence_manifest)
    if args.execute:
        record = execute(args.repo, record)
    print(json.dumps(record, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (Refused, OSError, ValueError, RuntimeError, KeyError, TypeError, yaml.YAMLError) as exc:
        print(f'REFUSED: {exc}', file=sys.stderr)
        raise SystemExit(2) from exc
