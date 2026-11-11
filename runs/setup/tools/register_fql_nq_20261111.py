"""FQLNQ registration (user decision 2026-11-11 ~23:10 KST): pretrained-encoder FQL with alpha 10 + normalize_q_loss.

The user adopted DIAG's recommendation (realworld-ops/diag/DIAG-REPORT.md section 11): every future FQL run on a
PRETRAINED (frozen) encoder uses ``agent.alpha=10.0`` + ``agent.normalize_q_loss=true``; ``agent.alpha=100.0`` with
``normalize_q_loss`` false is kept as a comparison baseline. configs/realworld/fql/base.yaml is NOT changed (it keeps
``normalize_q_loss: false``, so E2E CNN rows are untouched); the setting is carried by ``--set`` overrides.

This registrar adds 60 NEW rows and modifies none: for every frozen-encoder family whose latest valid alpha-10 1M-step
generation exists (reviwo, sincro, dynamo, mvp3d_single, mvp3d_multiview) x (drawer-open, mug-hanging) x seeds 0-2,

* ``rwnq-20261111-fql-<cond>-<task>-s<k>``  (fql_setting a10nq): source args with ``agent.normalize_q_loss=true``
  appended (alpha stays 10.0), source priority;
* ``rwa100-20261111-fql-<cond>-<task>-s<k>`` (fql_setting a100): ``agent.alpha=10.0`` replaced IN PLACE by
  ``agent.alpha=100.0`` and ``agent.normalize_q_loss=false`` appended explicitly (= base.yaml today, pinned against a
  future base change), priority source + 1 (the local scheduler/placement order rows by ascending ``priority``).

Each row is DERIVED from its source row in experiments/queue.yaml (``derived_from``): it equals the source in every
field except the id (+ the ``--name`` value, replaced in place), the alpha/normalize overrides above, ``fql_budget``
(alpha + normalize_q_loss recorded), the a100 priority, bookkeeping (derived_from, fql_setting, plan_row_id, purpose;
the source's ``supersedes`` is dropped because the new rows supersede nothing) and the release pin (planned_commit,
planned_image, evidence_manifest, registration_tool). Same script, dataset, split manifest, gate report, seed,
batch size, 1M offline steps, log/eval intervals, footprint, deps, role and host. The quarantined
rwcal-20261109-fql-mvp3d_* rows are never sources (the native-depth rwcal-20261110 generation is).

Release pin: the remote role release (runs/setup/role-release.json; the remote driver launches new role rows on it)
with its acceptance manifest; local placement (ops/jobs/realworld_local.py place) proves the same commit's FQL
execution scope against the local tree. Publication requires the coordinator's active admission pauses
(``PAUSES``) on every new row, re-checked under the locks: the remote driver orders role rows by (seed, queue index)
and ignores ``priority``, so the pauses are what keeps rows off the remote until local placement and keeps the a100
comparison behind the a10nq rows there.

Default: stdout-only DRY RUN (``--out`` may save the record outside runs/setup). ``--execute`` (main checkout copy
only) publishes the queue under stage2-driver.lock then registry.lock with the fqldiag/e2e-noaug registrars'
candidate/journal/proof convention. No scheduler, campaign driver, launch API, network or registry writes. A pending
journal after an interruption requires inspection before retrying.
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

MAIN = Path('/home/ws/ws/splatter4d')
TOOL = 'runs/setup/tools/' + Path(__file__).name
CONDITIONS = ('reviwo', 'sincro', 'dynamo', 'mvp3d_single', 'mvp3d_multiview')
TASKS = ('drawer-open', 'mug-hanging')
SEEDS = (0, 1, 2)
# Latest valid alpha-10, 1M-step generation per (condition, task) (brief fqlnq-20261111; all synced).
SOURCE_PREFIX = {('reviwo', 'drawer-open'): 'rwng-20261107b', ('reviwo', 'mug-hanging'): 'rwcal-20261109',
                 ('sincro', 'drawer-open'): 'rwcal-20261109', ('sincro', 'mug-hanging'): 'rwcal-20261109',
                 ('dynamo', 'drawer-open'): 'rwng-20261107b', ('dynamo', 'mug-hanging'): 'rwng-20261107b',
                 **{(c, t): 'rwcal-20261110' for c in ('mvp3d_single', 'mvp3d_multiview') for t in TASKS}}
QUARANTINED_PREFIXES = ('rwcal-20261109-fql-mvp3d_',)
SOURCE_ALPHA, SOURCE_STEPS = '10.0', '1000000'
# setting -> (id prefix, alpha, normalize_q_loss, priority offset vs the source)
SETTINGS = {'a10nq': ('rwnq-20261111', '10.0', True, 0), 'a100': ('rwa100-20261111', '100.0', False, 1)}
PAUSES = {'a10nq': 'fqlnq-await-placement-20261111', 'a100': 'fqla100-after-nq-20261111'}
DIRECTIVE = 'FQLNQ 2026-11-11 ~23:10 KST (user; memory fql-nq-setting-directive-20261111)'
PURPOSE = {
    'a10nq': f'{DIRECTIVE}: adopted pretrained-encoder FQL setting agent.alpha=10.0 + agent.normalize_q_loss=true '
             '(DIAG fqldiag section 11); identical to derived_from otherwise (dataset, split, gate report, seed, 1M '
             'steps, batch 256, intervals); from scratch',
    'a100': f'{DIRECTIVE}: comparison baseline agent.alpha=100.0 + agent.normalize_q_loss=false (explicit) for the '
            'adopted a10nq setting; identical to derived_from otherwise; lower priority than a10nq; never staged '
            'unless asked'}
# Row fields that may differ from derived_from; every other field must be equal (type-sensitive JSON equality).
CHANGED = {'id', 'args', 'priority', 'fql_budget', 'derived_from', 'fql_setting', 'plan_row_id', 'purpose',
           'supersedes', 'planned_commit', 'planned_image', 'evidence_manifest', 'registration_tool'}
STEM = 'rw-registration-fql-nq-20261111'
QUEUE, REGISTRY = 'experiments/queue.yaml', 'experiments/registry.jsonl'
N_ROWS = len(SETTINGS) * len(CONDITIONS) * len(TASKS) * len(SEEDS)
EXECUTION_REFUSAL = "execution is the coordinator's step from the main checkout copy of this tool"


class Refused(RuntimeError):
    pass


ERRORS = (Refused, OSError, ValueError, RuntimeError, KeyError, TypeError, AttributeError, IndexError, yaml.YAMLError)


def require(ok, why):
    if not ok:
        raise Refused(why)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def json_sha(value):
    return sha(json.dumps(value, sort_keys=True).encode())


def canonical(value):
    """Type-sensitive comparison key (JSON text: false != 0, 10 != 10.0, nested too)."""
    return json.dumps(value, sort_keys=True, allow_nan=False)


def json_bytes(value):
    return (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()


def now():
    return datetime.now(timezone.utc).isoformat()


def source_id(condition, task, seed):
    return f'{SOURCE_PREFIX[(condition, task)]}-fql-{condition}-{task}-s{seed}'


def new_id(setting, condition, task, seed):
    return f'{SETTINGS[setting][0]}-fql-{condition}-{task}-s{seed}'


def plan():
    """[(new id, setting, condition, task, seed, source id)] in publication order: all a10nq rows, then all a100."""
    return [(new_id(s, c, t, k), s, c, t, k, source_id(c, t, k))
            for s in SETTINGS for c in CONDITIONS for t in TASKS for k in SEEDS]


def indices(args, key):
    return [i for i, a in enumerate(args) if a.partition('=')[0] == key]


def check_source(row, condition, task, seed):
    """The source must be the expected alpha-10, 1M, normalize-unset frozen-encoder row."""
    job = row['id']
    require(not job.startswith(QUARANTINED_PREFIXES), f'quarantined source: {job}')
    args = row.get('args')
    require(isinstance(args, list) and all(isinstance(a, str) for a in args), f'invalid source args: {job}')
    require(row.get('script') == 'scripts/train_realworld_fql.py' and row.get('role') == 'realworld_fql'
            and row.get('host') == 'remote' and row.get('pin_host') == 'remote',
            f'source script/role/host/pin_host: {job}')
    require(row.get('realworld_target') == task and row.get('fql_condition') == condition
            and type(row.get('fql_seed')) is int and row['fql_seed'] == seed, f'source target/condition/seed: {job}')
    require(args.count('--name') == 1 and args[args.index('--name') + 1:][:1] == [job], f'source --name: {job}')
    require(args.count('--set') == 1, f'source needs exactly one --set: {job}')
    after = args[args.index('--set') + 1:]
    require(after and all('=' in a and not a.startswith('-') for a in after), f'source --set block: {job}')
    deps = row.get('deps')
    require(isinstance(deps, list) and len(deps) == 1 and isinstance(deps[0], str), f'source deps: {job}')
    for key, value in (('condition', condition), ('train.seed', str(seed)), ('agent.alpha', SOURCE_ALPHA),
                       ('train.offline_steps', SOURCE_STEPS), ('data.dataset', f'runs/{deps[0]}/dataset.h5')):
        found = indices(args, key)
        require(len(found) == 1 and args[found[0]] == f'{key}={value}', f'source expects one {key}={value}: {job}')
    require(not indices(args, 'agent.normalize_q_loss'), f'source already sets agent.normalize_q_loss: {job}')
    budget = row.get('fql_budget')
    require(isinstance(budget, dict) and budget.get('alpha') == 10.0 and budget.get('offline_steps') == 1000000
            and 'normalize_q_loss' not in budget, f'source fql_budget: {job}')
    require(type(row.get('priority')) is int, f'source priority: {job}')


def derive(source, job, setting, condition, task, seed):
    """The new row: a deep copy of the source with only the CHANGED fields set."""
    _, alpha, normalize, offset = SETTINGS[setting]
    row = copy.deepcopy(source)
    row.pop('supersedes', None)
    args = row['args']
    args[args.index('--name') + 1] = job
    alpha_index = indices(args, 'agent.alpha')[0]
    args[alpha_index] = f'agent.alpha={alpha}'
    args.append(f'agent.normalize_q_loss={str(normalize).lower()}')
    row.update(id=job, priority=source['priority'] + offset,
               fql_budget={**source['fql_budget'], 'alpha': float(alpha), 'normalize_q_loss': normalize},
               derived_from=source['id'], fql_setting=setting,
               plan_row_id=f'rw-fql-{condition}-{task}-s{seed}-{setting}', purpose=PURPOSE[setting])
    return row


def check_derived(row, source, setting):
    """Field-by-field: everything outside CHANGED equal; CHANGED fields exactly as specified."""
    job, (_, alpha, normalize, offset) = row['id'], SETTINGS[setting]
    keys = (set(row) | set(source)) - CHANGED
    diff = sorted(k for k in keys if k not in row or k not in source or canonical(row[k]) != canonical(source[k]))
    require(not diff, f'fields differ from derived_from {diff}: {job}')
    expected = list(source['args'])
    expected[expected.index('--name') + 1] = job
    expected[indices(expected, 'agent.alpha')[0]] = f'agent.alpha={alpha}'
    expected.append(f'agent.normalize_q_loss={str(normalize).lower()}')
    require(row['args'] == expected, f'args differ from derived_from + the setting: {job}')
    require(row['priority'] == source['priority'] + offset, f'priority: {job}')
    require(canonical(row['fql_budget']) == canonical(
        {**source['fql_budget'], 'alpha': float(alpha), 'normalize_q_loss': normalize}), f'fql_budget: {job}')
    require(row['derived_from'] == source['id'] and row['fql_setting'] == setting and 'supersedes' not in row,
            f'bookkeeping: {job}')


def pin(row, release):
    row.update(planned_commit=release['commit'], planned_image=release['image'],
               evidence_manifest=release['evidence_manifest'], registration_tool=TOOL)


def release_pin(repo, commit, image):
    """--commit/--image must name the current remote role release with an acceptance manifest."""
    require(re.fullmatch(r'[0-9a-f]{40}', commit or ''), '--commit needs a full 40-hex sha')
    require(re.fullmatch(r'sha256:[0-9a-f]{64}', image or ''), '--image needs sha256:<64 hex>')
    role = json.loads((repo / 'runs/setup/role-release.json').read_bytes())
    require(isinstance(role, dict) and role.get('commit') == commit and role.get('image') == image,
            'release pin differs from runs/setup/role-release.json (remote launches use the role release)')
    evidence = f'runs/remote/acceptance/{commit[:7]}-cc/manifest.json'
    require((repo / evidence).is_file(), f'missing acceptance manifest: {evidence}')
    return {'commit': commit, 'image': image, 'evidence_manifest': evidence}


def queue_document(raw):
    doc = yaml.safe_load(raw)
    require(isinstance(doc, dict) and isinstance(doc.get('jobs'), list), 'queue jobs must be a sequence')
    require(all(isinstance(r, dict) and isinstance(r.get('id'), str) for r in doc['jobs']), 'invalid queue row')
    ids = [r['id'] for r in doc['jobs']]
    require(len(ids) == len(set(ids)), 'duplicate job ids in queue')
    return doc


def build_rows(current, release):
    by_id = {r['id']: r for r in current['jobs']}
    rows = []
    for job, setting, condition, task, seed, source in plan():
        require(source in by_id, f'derived_from missing from queue: {source}')
        check_source(by_id[source], condition, task, seed)
        require(by_id[source]['deps'][0] in by_id, f'dependency missing from queue: {by_id[source]["deps"][0]}')
        row = derive(by_id[source], job, setting, condition, task, seed)
        check_derived(row, by_id[source], setting)
        pin(row, release)
        rows.append(row)
    require(len(rows) == N_ROWS and len({r['id'] for r in rows}) == N_ROWS, 'wrong row count')
    require(Counter(r['fql_setting'] for r in rows) == {s: N_ROWS // len(SETTINGS) for s in SETTINGS},
            'wrong per-setting counts')
    return rows


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
    for dep in sorted({r['deps'][0] for r in rows}):
        require((repo / 'runs' / dep / 'dataset.h5').is_file(), f'missing dependency dataset: {dep}')


def pause_status(rows, active_pauses):
    """{id: True} when the row's setting pause is active for it (ops.jobs.admission_pause.active_pauses, fresh)."""
    return {r['id']: PAUSES[r['fql_setting']] in active_pauses(r['id'], fresh=True) for r in rows}


def default_active_pauses(repo):
    sys.path.insert(0, str(repo))
    try:
        from ops.jobs.admission_pause import active_pauses  # noqa: PLC0415  (reads runs/setup/admission-pauses.json)
    finally:
        sys.path.remove(str(repo))
    return active_pauses


def prepare(repo, commit, image, active_pauses=None):
    release = release_pin(repo, commit, image)
    setup = repo / 'runs/setup'
    require(not (setup / f'{STEM}.pending.json').exists() and not (setup / f'{STEM}.json').exists(),
            'pending transaction/proof exists; inspect before retrying')
    before = (repo / QUEUE).read_bytes()
    registry = (repo / REGISTRY).read_bytes() if (repo / REGISTRY).exists() else b''
    current = queue_document(before)
    rows = build_rows(current, release)
    ids = {r['id'] for r in current['jobs']} | registry_ids(registry)
    require(not ids.intersection(r['id'] for r in rows), 'a proposed ID already exists in queue or registry')
    check_run_paths(repo, rows)
    candidate = queue_candidate(before, rows)
    paused = pause_status(rows, active_pauses or default_active_pauses(repo))
    return {'schema': 'rw-registration/v1', 'status': 'dry-run', 'registered': False, 'prepared_at': now(),
            'repo': str(repo.resolve()), 'release': release, 'evidence_manifest': release['evidence_manifest'],
            'change_vs_derived_from': {
                'a10nq': 'args: --name <new id>; appended agent.normalize_q_loss=true; fql_budget.normalize_q_loss',
                'a100': 'args: --name <new id>; agent.alpha=10.0 -> agent.alpha=100.0 in place; appended '
                        'agent.normalize_q_loss=false; fql_budget alpha/normalize_q_loss; priority +1',
                'bookkeeping': sorted(CHANGED - {'id', 'args', 'priority', 'fql_budget'})},
            'required_pauses': {s: {'name': PAUSES[s], 'prefixes': [SETTINGS[s][0] + '-fql-']} for s in SETTINGS},
            'pauses_active_now': {s: sum(paused[r['id']] for r in rows if r['fql_setting'] == s) for s in SETTINGS},
            'jobs': rows, 'jobs_sha256': json_sha(rows), 'tool_sha256': sha(Path(__file__).read_bytes()),
            'registry_exists_at_evaluation': (repo / REGISTRY).exists(), 'registry_sha256_at_evaluation': sha(registry),
            'queue_sha256_before': sha(before), 'queue_sha256_after': sha(candidate),
            'queue_diff': {'before_rows': len(current['jobs']), 'after_rows': len(current['jobs']) + N_ROWS,
                           'added_rows': N_ROWS, 'modified_rows': 0, 'removed_rows': 0,
                           'counts': dict(Counter(r['fql_setting'] for r in rows)),
                           'conditions': dict(Counter(r['fql_condition'] for r in rows)),
                           'targets': dict(Counter(r['realworld_target'] for r in rows)),
                           'derived_from': {r['id']: r['derived_from'] for r in rows}}}


def execution_allowed(repo):
    require(repo.resolve() == MAIN and Path(__file__).resolve() == MAIN / TOOL, EXECUTION_REFUSAL)


def check_publication(repo, record, active_pauses):
    execution_allowed(repo)  # also here and in execute(): an imported worktree copy cannot publish (Codex review 1)
    require(sha((repo / QUEUE).read_bytes()) == record['queue_sha256_before'], 'queue changed under locks')
    registry = repo / REGISTRY
    require(registry.exists() == record['registry_exists_at_evaluation']
            and sha(registry.read_bytes() if registry.exists() else b'') == record['registry_sha256_at_evaluation'],
            'registry changed under locks')
    require(sha(Path(__file__).read_bytes()) == record['tool_sha256'], 'registration tool changed')
    require(release_pin(repo, record['release']['commit'], record['release']['image']) == record['release'],
            'role release changed')
    check_run_paths(repo, record['jobs'])
    missing = sorted(i for i, ok in pause_status(record['jobs'], active_pauses).items() if not ok)
    require(not missing, f'{len(missing)} rows lack their active pause {PAUSES} (add them first), e.g. {missing[:2]}')


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


def execute(repo, record, active_pauses=None):
    execution_allowed(repo)
    active_pauses = active_pauses or default_active_pauses(repo)
    setup = repo / 'runs/setup'
    pending, proof = setup / f'{STEM}.pending.json', setup / f'{STEM}.json'
    candidate_path = setup / f'{STEM}.queue-candidate.yaml'
    with locks(repo):
        require(not pending.exists() and not proof.exists(), 'pending transaction/proof exists; inspect before retrying')
        require(record['repo'] == str(repo.resolve()) and json_sha(record['jobs']) == record['jobs_sha256'],
                'registration plan changed')
        check_publication(repo, record, active_pauses)
        before = (repo / QUEUE).read_bytes()
        rebuilt = build_rows(queue_document(before), record['release'])
        require(json_sha(rebuilt) == record['jobs_sha256'], 'rows re-derived under locks differ from the plan')
        candidate = queue_candidate(before, record['jobs'])
        require(sha(candidate) == record['queue_sha256_after'], 'staged plan changed')
        atomic_write(candidate_path, candidate)
        record.update(status='pending', proof_file=proof.name)
        atomic_write(pending, json_bytes(record))
        check_publication(repo, record, active_pauses)  # immediately before the queue replace
        os.replace(candidate_path, repo / QUEUE)
        sync_directory((repo / QUEUE).parent)
        record.update(status='published', registered=True, registered_at=now())
        atomic_write(proof, json_bytes(record))
        pending.unlink()
        sync_directory(setup)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--repo', type=Path, default=MAIN)
    parser.add_argument('--commit', required=True, help='remote role release commit (runs/setup/role-release.json)')
    parser.add_argument('--image', required=True, help='remote role release image digest')
    parser.add_argument('--out', type=Path, help='dry run only: also save the record here (never runs/setup)')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    require(not (args.out and args.execute), '--out is only for dry-run records')
    if args.execute:
        execution_allowed(args.repo)
    record = prepare(args.repo, args.commit, args.image)
    if args.execute:
        record = execute(args.repo, record)
    if args.out:
        path = args.out.parent.resolve() / args.out.name  # the name itself is never resolved (Codex review 2)
        require(not any(p.name == 'setup' and p.parent.name == 'runs' for p in path.parents)
                and not (path.parent.name == 'experiments' and path.name in ('queue.yaml', 'registry.jsonl')),
                '--out must not overwrite operational files')
        require(not path.exists() and not path.is_symlink(), '--out already exists')
        with path.open('xb') as stream:  # exclusive create: never overwrites or follows a planted file (Codex review 1)
            stream.write(json_bytes(record))
    print(json_bytes(record).decode(), end='')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except ERRORS as exc:
        print(f'REFUSED: {exc}', file=sys.stderr)
        raise SystemExit(2) from exc
