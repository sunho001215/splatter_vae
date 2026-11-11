"""NOAUG registration (user directive O, 2026-11-11 ~17:40): End-to-End CNN + FQL WITHOUT random cropping.

Six rows ``rwcal-20261111-fql-e2e_cnn_noaug-<task>-s<seed>`` (drawer-open, mug-hanging; seeds 0, 1, 2), each a copy
of its ORIGINAL e2e_cnn FQL row (``derived_from``) whose ONLY changes are: the id (+ its ``--name`` argument), one
appended override ``pixel.p_aug=0.0`` (the original inherits 0.5 from configs/realworld/fql/base.yaml), bookkeeping
fields (derived_from, plan_row_id, purpose, fql_variant) and the release pin set here from --commit / --image.
Every other row field (script, dataset, split, store, gate report, seed, alpha, steps, footprints, deps, role, host)
must equal the original's byte for byte (JSON equality); existing rows are never modified.

Default: stdout-only DRY RUN. --execute publishes the queue under stage2-driver.lock then registry.lock with the
fqldiag/newtasks registrar's candidate/journal/proof convention (modelled on
runs/setup/tools/register_fqldiag_20261111.py). No scheduler, campaign, launch API, network or registry writes. A
pending journal after an interruption requires inspection before retrying.
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
TASKS = {'drawer-open': 'rwng-20261107b', 'mug-hanging': 'rwcal-20261109'}  # original e2e_cnn row prefixes
SEEDS = (0, 1, 2)
P_AUG_ARG = 'pixel.p_aug=0.0'
BASE_CONFIG = 'configs/realworld/fql/base.yaml'
# Row fields allowed to differ from derived_from (everything else must be identical).
BOOKKEEPING = {'id', 'args', 'derived_from', 'plan_row_id', 'purpose', 'fql_variant',
               'planned_commit', 'planned_image', 'evidence_manifest', 'registration_tool'}
STEM = 'rw-registration-e2e-noaug-20261111'
QUEUE = 'experiments/queue.yaml'
REGISTRY = 'experiments/registry.jsonl'
N_ROWS = len(TASKS) * len(SEEDS)


class Refused(RuntimeError):
    pass


def require(ok, why):
    if not ok:
        raise Refused(why)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def json_sha(value):
    return sha(json.dumps(value, sort_keys=True).encode())


def canonical(value):
    """Type-sensitive comparison key (JSON text: false != 0, 1 != 1.0, nested too)."""
    return json.dumps(value, sort_keys=True, allow_nan=False)


def json_bytes(value):
    return (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()


def now():
    return datetime.now(timezone.utc).isoformat()


def dataset_dep(task):
    return f'{TASKS[task]}-fqldata-e2e_cnn-{task}'


def original_id(task, seed):
    return f'{TASKS[task]}-fql-e2e_cnn-{task}-s{seed}'


def expected_rows():
    """{new id: (task, seed, original id, dataset dependency)}."""
    return {f'rwcal-20261111-fql-e2e_cnn_noaug-{task}-s{seed}': (task, seed, original_id(task, seed), dataset_dep(task))
            for task in TASKS for seed in SEEDS}


def strip_name(row):
    """args without the single ``--name <id>`` pair (validated)."""
    args, job = row.get('args'), row['id']
    require(isinstance(args, list) and all(isinstance(a, str) for a in args), f'invalid args: {job}')
    require(args.count('--name') == 1, f'expected exactly one --name: {job}')
    index = args.index('--name')
    require(index + 1 < len(args) and args[index + 1] == job, f'wrong --name: {job}')
    return [a for i, a in enumerate(args) if i not in (index, index + 1)]


def p_aug_args(args):
    return [a for a in args if a.startswith('pixel.p_aug')]


def check_args(row, original):
    """New args == original args (minus --name pair) + exactly one trailing ``pixel.p_aug=0.0``."""
    job = row['id']
    new, old = strip_name(row), strip_name(original)
    require(p_aug_args(old) == [] and not any(a.startswith('pixel.') for a in old),
            f'derived_from already overrides pixel.*: {job}')
    require(p_aug_args(new) == [P_AUG_ARG] and new.count(P_AUG_ARG) == 1,
            f'expected exactly one {P_AUG_ARG}: {job}')
    require(new[-1] == P_AUG_ARG and '--set' in new[:-1] and new.index('--set') < len(new) - 1,
            f'{P_AUG_ARG} must be the last --set override: {job}')
    require(new[:-1] == old, f'other args differ from derived_from: {job}')
    expected = list(original['args'])  # full positional equality: --name value replaced IN PLACE, override appended
    expected[expected.index('--name') + 1] = job
    require(row['args'] == expected + [P_AUG_ARG], f'other args differ from derived_from (order/position): {job}')


def pinned_rows(raw, commit, image, evidence_manifest=None):
    require(re.fullmatch(r'[0-9a-f]{40}', commit or ''), '--commit needs a full 40-hex sha')
    require(re.fullmatch(r'sha256:[0-9a-f]{64}', image or ''), '--image needs sha256:<64 hex>')
    evidence = evidence_manifest or f'runs/remote/acceptance/{commit[:7]}-cc/manifest.json'
    path = Path(evidence)
    require(not path.is_absolute() and '..' not in path.parts and path.suffix == '.json'
            and '<' not in evidence and str(path) == evidence, 'invalid --evidence-manifest repo-relative JSON path')
    doc = json.loads(raw)
    require(isinstance(doc, dict), f'rows.json must be an object with the expected {N_ROWS} rows')
    rows = doc.get('rows')
    expected = expected_rows()
    require(isinstance(rows, list) and len(rows) == N_ROWS
            and all(isinstance(r, dict) and isinstance(r.get('id'), str) for r in rows),
            f'rows.json needs exactly the expected {N_ROWS} rows')
    require(len({r['id'] for r in rows}) == N_ROWS and {r['id'] for r in rows} == set(expected),
            f'rows.json differs from the expected {N_ROWS} ids')
    rows = copy.deepcopy(rows)
    for row in rows:
        job = row['id']
        task, seed, original, dep = expected[job]
        require(row.get('derived_from') == original, f'wrong or missing derived_from: {job}')
        require(row.get('deps') == [dep], f'wrong deps: {job}')
        require(row.get('script') == 'scripts/train_realworld_fql.py' and row.get('role') == 'realworld_fql'
                and row.get('host') == 'remote' and row.get('pin_host') == 'remote',
                f'wrong script/role/host/pin_host: {job}')
        require(row.get('realworld_target') == task and row.get('fql_condition') == 'e2e_cnn'
                and type(row.get('fql_seed')) is int and row['fql_seed'] == seed, f'wrong target/condition/seed: {job}')
        require(row.get('fql_variant') == 'noaug' and row.get('plan_row_id') == f'rw-fql-e2e_cnn_noaug-{task}-s{seed}',
                f'wrong fql_variant/plan_row_id: {job}')
        require(isinstance(row.get('purpose'), str) and 'p_aug=0.0' in row['purpose'], f'missing purpose: {job}')
        require('supersedes' not in row, f'supersedes is forbidden: {job}')
        require(not any(k in row for k in ('planned_commit', 'planned_image', 'evidence_manifest', 'registration_tool')),
                f'release pin fields belong to the registrar: {job}')
        strip_name(row)
        row.update(planned_commit=commit, planned_image=image, evidence_manifest=evidence,
                   registration_tool='runs/setup/tools/' + Path(__file__).name)
    require('<release containing' not in json.dumps(rows), 'unreplaced release placeholder in row specs')
    require(Counter(r['realworld_target'] for r in rows) == {t: len(SEEDS) for t in TASKS}, 'wrong row counts')
    return rows


def queue_document(raw):
    doc = yaml.safe_load(raw)
    require(isinstance(doc, dict) and isinstance(doc.get('jobs'), list), 'queue jobs must be a sequence')
    require(all(isinstance(r, dict) and isinstance(r.get('id'), str) for r in doc['jobs']), 'invalid queue row')
    ids = [r['id'] for r in doc['jobs']]
    require(len(ids) == len(set(ids)), 'duplicate job ids in queue')
    return doc


def queue_candidate(before, rows):
    """Same pure text append and parsed-equality proof as the fqldiag/newtasks registrars."""
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
    for task in TASKS:
        require((repo / 'runs' / dataset_dep(task) / 'dataset.h5').is_file(),
                f'missing dependency dataset: {dataset_dep(task)}')


def check_base_config(repo):
    """The originals inherit pixel.p_aug from the base config; it must still be the official 0.5 (crop on)."""
    cfg = yaml.safe_load((repo / BASE_CONFIG).read_bytes())
    pixel = cfg.get('pixel') if isinstance(cfg, dict) else None
    require(isinstance(pixel, dict) and pixel.get('p_aug') == 0.5 and pixel.get('encoder') == 'impala_small',
            f'{BASE_CONFIG}: expected pixel.p_aug 0.5 / impala_small (what the originals ran)')


def check_originals(current, rows):
    jobs = {r['id']: r for r in current['jobs']}
    for row in rows:
        job = row['id']
        original = jobs.get(row['derived_from'])
        require(original is not None, f'derived_from missing from queue: {row["derived_from"]}')
        require(row['deps'][0] in jobs, f'dependency missing from queue: {row["deps"][0]}')
        require(original.get('deps') == row['deps'], f'deps differ from derived_from: {job}')
        check_args(row, original)
        keys = (set(row) | set(original)) - BOOKKEEPING
        diff = sorted(k for k in keys if k not in row or k not in original or canonical(row[k]) != canonical(original[k]))
        require(not diff, f'fields differ from derived_from {diff}: {job}')
        require(all(key in row for key in ('mem_gb', 'ram_gb', 'disk_gb', 'gpu', 'fql_budget')),
                f'missing footprint/budget: {job}')


def check_evidence(repo, rows):
    require((repo / rows[0]['evidence_manifest']).is_file(), f'missing evidence manifest: {rows[0]["evidence_manifest"]}')


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
    check_base_config(repo)
    check_originals(current, rows)
    check_evidence(repo, rows)
    candidate = queue_candidate(before, rows)
    return {'schema': 'rw-registration/v1', 'status': 'dry-run', 'registered': False, 'prepared_at': now(),
            'repo': str(repo.resolve()), 'release': {'commit': commit, 'image': image},
            'evidence_manifest': rows[0]['evidence_manifest'],
            'evidence_manifest_convention': 'newtasks registrar: runs/remote/acceptance/<commit[:7]>-cc/manifest.json',
            'change_vs_derived_from': {'args': f'--name <new id>; appended {P_AUG_ARG}',
                                       'bookkeeping': sorted(BOOKKEEPING - {'args'})},
            'jobs': rows, 'jobs_sha256': json_sha(rows),
            'row_spec': str(rows_path.resolve()), 'row_spec_sha256': sha(spec),
            'tool_sha256': sha(Path(__file__).read_bytes()),
            'registry_exists_at_evaluation': (repo / REGISTRY).exists(),
            'registry_sha256_at_evaluation': sha(registry),
            'queue_sha256_before': sha(before), 'queue_sha256_after': sha(candidate),
            'queue_diff': {'before_rows': len(current['jobs']), 'after_rows': len(current['jobs']) + N_ROWS,
                           'added_rows': N_ROWS, 'modified_rows': 0, 'removed_rows': 0,
                           'counts': {'FQL': N_ROWS}, 'targets': dict(Counter(r['realworld_target'] for r in rows)),
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
    parser.add_argument('--rows', type=Path, required=True, help='row specs (realworld-ops/noaug/rows.json)')
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
