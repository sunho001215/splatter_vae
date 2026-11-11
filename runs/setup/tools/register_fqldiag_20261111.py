"""One-off diagnostic FQL registration, 2026-11-11. Default: stdout-only DRY RUN.

--execute publishes the queue under stage2-driver.lock then registry.lock, with
the newtasks registrar's candidate/journal/proof convention. No scheduler,
campaign, launch API, network or registry writes are used. A pending journal
after an interruption requires inspection before retrying.
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
ROW_SPEC = Path(__file__).with_name('rows.json')
CONDITIONS = ('reviwo', 'sincro')
VARIANTS = {'a10': (10.0, False), 'a30': (30.0, False),
            'a100': (100.0, False), 'a10nq': (10.0, True)}
DEPS = {'reviwo': 'rwng-20261107b-fqldata-reviwo-drawer-open',
        'sincro': 'rwcal-20261109-fqldata-sincro-drawer-open'}
ORIGINALS = {'reviwo': 'rwng-20261107b-fql-reviwo-drawer-open-s0',
             'sincro': 'rwcal-20261109-fql-sincro-drawer-open-s0'}
TESTED_SETTINGS = ('agent.alpha', 'train.offline_steps', 'agent.normalize_q_loss')
STEM = 'rw-registration-fqldiag-20261111'
QUEUE = 'experiments/queue.yaml'
REGISTRY = 'experiments/registry.jsonl'


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
    return {f'rwcal-20261111-fqldiag-{condition}-drawer-open-s0-{variant}': [DEPS[condition]]
            for condition in CONDITIONS for variant in VARIANTS}


def setting(args, key, value, job):
    require([a for a in args if a.partition('=')[0] == key] == [f'{key}={value}'],
            f'expected exactly one {key}={value}: {job}')


def other_args(row):
    """Preserve order and multiplicity of everything except the four allowed changes."""
    args, job = row.get('args'), row['id']
    require(isinstance(args, list) and all(isinstance(a, str) for a in args), f'invalid args: {job}')
    require(args.count('--name') == 1, f'expected exactly one --name: {job}')
    index = args.index('--name')
    require(index + 1 < len(args) and args[index + 1] == job, f'wrong --name: {job}')
    return [a for i, a in enumerate(args)
            if i not in (index, index + 1) and a.partition('=')[0] not in TESTED_SETTINGS]


def pinned_rows(raw, commit, image, evidence_manifest=None):
    require(re.fullmatch(r'[0-9a-f]{40}', commit or ''), '--commit needs a full 40-hex sha')
    require(re.fullmatch(r'sha256:[0-9a-f]{64}', image or ''), '--image needs sha256:<64 hex>')
    # register_realworld_newtasks_20261110.adapt_rows uses this exact convention.
    evidence = evidence_manifest or f'runs/remote/acceptance/{commit[:7]}-cc/manifest.json'
    path = Path(evidence)
    require(not path.is_absolute() and '..' not in path.parts and path.suffix == '.json'
            and '<' not in evidence and str(path) == evidence, 'invalid --evidence-manifest repo-relative JSON path')
    doc = json.loads(raw)
    require(isinstance(doc, dict), 'rows.json must be an object with the expected 8 ids/deps')
    rows = doc.get('rows')
    graph = expected_graph()
    require(isinstance(rows, list) and len(rows) == 8
            and all(isinstance(r, dict) and isinstance(r.get('id'), str) for r in rows),
            'rows.json needs exactly the expected 8 ids/deps')
    require(len({r['id'] for r in rows}) == 8 and {r['id'] for r in rows} == set(graph),
            'rows.json differs from the expected 8 ids/deps')
    rows = copy.deepcopy(rows)
    for row in rows:
        job = row['id']
        condition, variant = job.split('-')[3], job.rsplit('-', 1)[1]
        alpha, normalize = VARIANTS[variant]
        require(row.get('deps') == graph[job], f'wrong deps: {job}')
        require(row.get('script') == 'scripts/train_realworld_fql.py'
                and row.get('role') == 'realworld_fql' and row.get('host') == 'remote'
                and row.get('pin_host') == 'remote', f'wrong script/role/host/pin_host: {job}')
        require(row.get('realworld_target') == 'drawer-open' and row.get('fql_condition') == condition,
                f'wrong target/condition: {job}')
        require(type(row.get('fql_seed')) is int and row['fql_seed'] == 0, f'wrong fql_seed: {job}')
        require(row.get('derived_from') == ORIGINALS[condition], f'wrong or missing derived_from: {job}')
        require('supersedes' not in row, f'supersedes is forbidden: {job}')
        other_args(row)  # Also validates the args sequence and --name pair.
        args = row['args']
        for key, value in (('train.seed', '0'), ('agent.alpha', str(alpha)),
                           ('train.offline_steps', '200000'),
                           ('agent.normalize_q_loss', str(normalize).lower()),
                           ('data.dataset', f'runs/{DEPS[condition]}/dataset.h5')):
            setting(args, key, value, job)
        budget = row.get('fql_budget')
        require(isinstance(budget, dict) and type(budget.get('offline_steps')) is int
                and budget['offline_steps'] == 200000 and budget.get('alpha') == alpha
                and budget.get('normalize_q_loss') is normalize, f'wrong fql_budget: {job}')
        row.update(planned_commit=commit, planned_image=image, evidence_manifest=evidence,
                   registration_tool='runs/setup/tools/' + Path(__file__).name)
    require('<release containing' not in json.dumps(rows), 'unreplaced release placeholder in row specs')
    require(Counter(r['fql_condition'] for r in rows) == {'reviwo': 4, 'sincro': 4},
            'wrong diagnostic row counts')
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


def check_run_paths(repo, rows):
    require(not any((repo / 'runs' / r['id']).exists() or (repo / 'runs' / r['id']).is_symlink()
                    for r in rows), 'a run directory already exists')
    for dep in DEPS.values():
        require((repo / 'runs' / dep / 'dataset.h5').is_file(), f'missing dependency dataset: {dep}')


def check_originals(current, rows):
    originals = {r['id']: r for r in current['jobs']}
    for condition in CONDITIONS:
        require(DEPS[condition] in originals, f'dependency missing from queue: {DEPS[condition]}')
        require(ORIGINALS[condition] in originals, f'derived_from missing from queue: {ORIGINALS[condition]}')
    for row in rows:
        job = row['id']
        original = originals[row['derived_from']]
        require(other_args(row) == other_args(original), f'other args differ from derived_from: {job}')
        require(all(key in row and key in original and row[key] == original[key]
                    for key in ('mem_gb', 'ram_gb', 'disk_gb', 'gpu')),
                f'footprint differs from derived_from: {job}')
        budget = original.get('fql_budget')
        require(isinstance(budget, dict), f'invalid derived_from fql_budget: {job}')
        expected = {**budget, **{key: row['fql_budget'][key]
                                for key in ('alpha', 'offline_steps', 'normalize_q_loss')}}
        require(row['fql_budget'] == expected, f'other fql_budget values differ from derived_from: {job}')


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
    current = queue_document(before)
    ids = {r['id'] for r in current['jobs']} | registry_ids(registry)
    require(not ids.intersection(r['id'] for r in rows), 'a proposed ID already exists in queue or registry')
    check_run_paths(repo, rows)
    check_originals(current, rows)
    candidate = queue_candidate(before, rows)
    return {'schema': 'rw-registration/v1', 'status': 'dry-run', 'registered': False, 'prepared_at': now(),
            'repo': str(repo.resolve()), 'release': {'commit': commit, 'image': image},
            'evidence_manifest': rows[0]['evidence_manifest'],
            'evidence_manifest_convention': 'newtasks registrar: runs/remote/acceptance/<commit[:7]>-cc/manifest.json',
            'jobs': rows, 'jobs_sha256': json_sha(rows),
            'row_spec': str(rows_path.resolve()), 'row_spec_sha256': sha(spec),
            'tool_sha256': sha(Path(__file__).read_bytes()),
            'registry_exists_at_evaluation': (repo / REGISTRY).exists(),
            'registry_sha256_at_evaluation': sha(registry),
            'queue_sha256_before': sha(before), 'queue_sha256_after': sha(candidate),
            'queue_diff': {'before_rows': len(current['jobs']), 'after_rows': len(current['jobs']) + 8,
                           'added_rows': 8, 'modified_rows': 0, 'removed_rows': 0,
                           'counts': {'FQL': 8}, 'conditions': dict(Counter(r['fql_condition'] for r in rows)),
                           'added_ids': [r['id'] for r in rows]}}


def check_unchanged(repo, record):
    require(sha((repo / QUEUE).read_bytes()) == record['queue_sha256_before'], 'queue changed under locks')
    registry = repo / REGISTRY
    require(registry.exists() == record['registry_exists_at_evaluation']
            and sha(registry.read_bytes() if registry.exists() else b'') == record['registry_sha256_at_evaluation'],
            'registry changed under locks')
    require(sha(Path(record['row_spec']).read_bytes()) == record['row_spec_sha256'], 'row specs changed')
    require(sha(Path(__file__).read_bytes()) == record['tool_sha256'], 'registration tool changed')
    check_run_paths(repo, record['jobs'])


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
        require(sha(candidate) == record['queue_sha256_after'], 'staged plan changed')
        atomic_write(candidate_path, candidate)
        record.update(status='pending', proof_file=proof.name)
        atomic_write(pending, json_bytes(record))
        check_unchanged(repo, record)
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
