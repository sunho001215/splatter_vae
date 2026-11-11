"""FQLNQ follow-up registration (user decision 2026-11-11 ~23:10 KST): SNeRL and OURS frozen-encoder FQL rows.

Same setting rule as runs/setup/tools/register_fql_nq_20261111.py (whose helpers this tool reuses): every frozen-encoder
FQL row is ``rwnq-20261111-fql-<cond>-<task>-s<k>`` (``agent.alpha=10.0`` + ``agent.normalize_q_loss=true``) with a
``rwa100-20261111-fql-<cond>-<task>-s<k>`` comparison (``agent.alpha=100.0`` in place + ``agent.normalize_q_loss=false``,
priority + 1); configs/realworld/fql/base.yaml is unchanged (E2E untouched). Tasks drawer-open, mug-hanging; seeds 0-2.

``--group snerl`` (12 rows): DERIVED from the six never-launched alpha-10 rows ``rwcal-20261109-fql-snerl-<task>-s<k>``
(paused: superseded by these rows), exactly as the phase-1 rows (same derivation, same field-by-field check). They
depend on ``rwcal-20261109-fqldata-snerl-<task>`` (SNeRL pretraining still running) and are published only while the
coordinator's feature-check hold ``SNERL_HOLD`` holds every one of them (lifted per task when
realworld-ops/serving/evidence/fqlfeat-snerl-<task>.json passes, the rule of pause rwcal-fql-feature-check-20261109),
plus the a100 pause on the a100 rows.

``--group ours`` (2 + 12 rows): the FQL condition ``ours`` exists from the release that adds it to
s4d/rwlearn/observation.py (checked in the pinned commit). Rows:
* ``rwcal-20261111-fqldata-ours-<task>`` (scripts/realworld_fql_dataset.py, remote: inputs live there), derived from
  the SinCro dataset row of the same task (same store / gate report / outcomes / footprint), with ``--condition ours``,
  its own ``--out`` and ``--export runs/rwcal-20261111-ours-<task>-export/encoder.pt`` (dependency: that export row);
* nq + a100 FQL rows whose PROTOCOL TEMPLATE is the SinCro FQL row of the same task and seed (the other single-view
  sequence encoder: same split manifest, gate report, seed, batch, 1M steps, intervals, footprint), with
  ``condition=ours`` and the ours dataset; no alpha-10 ours row ever existed, so they record ``protocol_template``
  instead of ``derived_from``. The template's inputs must be the ours pretraining inputs (checked).
Published only while ``OURS_HOLD`` (pre-launch feature check of each ours dataset, same rule as SNeRL) holds every ours
FQL row and the a100 pause the a100 rows; the dataset rows are not held (they wait for their export).

Release pin: the remote role release (runs/setup/role-release.json + acceptance manifest), as phase 1. Default: stdout
DRY RUN; ``--execute`` only from the main checkout copy; candidate/journal/proof publication under
stage2-driver.lock then registry.lock (one proof per group).
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True
_SPEC = importlib.util.spec_from_file_location('fqlnq_phase1', Path(__file__).with_name('register_fql_nq_20261111.py'))
P1 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(P1)
yaml, Refused, require, sha, json_sha, canonical, json_bytes, now = (
    P1.yaml, P1.Refused, P1.require, P1.sha, P1.json_sha, P1.canonical, P1.json_bytes, P1.now)

MAIN = P1.MAIN
TOOL = 'runs/setup/tools/' + Path(__file__).name
TASKS, SEEDS, SETTINGS = P1.TASKS, P1.SEEDS, P1.SETTINGS
QUEUE, REGISTRY = P1.QUEUE, P1.REGISTRY
SNERL_HOLD = 'fqlnq-snerl-feature-check-20261111'
OURS_HOLD = 'fqlnq-ours-feature-check-20261112'
A100_PAUSE = P1.PAUSES['a100']
GROUPS = ('snerl', 'ours')
STEM = 'rw-registration-fql-nq-{group}-20261112'
OURS_PRE, OURS_EXPORT = 'rwcal-20261111-ours-{task}-pre', 'rwcal-20261111-ours-{task}-export'
OURS_DATA = 'rwcal-20261111-fqldata-ours-{task}'
SINCRO_DATA, SINCRO_FQL = 'rwcal-20261109-fqldata-sincro-{task}', 'rwcal-20261109-fql-sincro-{task}-s{seed}'
PURPOSE_OURS_DATA = (f'{P1.DIRECTIVE}: offline-FQL dataset of OUR real-world-pretrained Original export (single RGB '
                     'view, 3-frame joint state, s4d/rwlearn/observation.py "ours"); store / gate report / outcomes / '
                     'footprint as the SinCro dataset row of the task (protocol_template)')
PURPOSE_OURS_FQL = {
    'a10nq': f'{P1.DIRECTIVE}: OUR method, offline FQL with the adopted pretrained-encoder setting agent.alpha=10.0 + '
             'agent.normalize_q_loss=true on the ours dataset; protocol (split, gate report, seed, 1M steps, batch 256, '
             'intervals, footprint) as the SinCro FQL row of the task and seed (protocol_template); from scratch',
    'a100': f'{P1.DIRECTIVE}: OUR method, comparison baseline agent.alpha=100.0 + agent.normalize_q_loss=false '
            '(explicit); otherwise as the a10nq row; lower priority; never staged unless asked'}
DATA_CHANGED = {'id', 'args', 'deps', 'fql_condition', 'realworld_consumer', 'realworld_gate_sha256',
                'realworld_manifest_sha256', 'plan_row_id', 'purpose', 'protocol_template', 'planned_commit',
                'planned_image', 'evidence_manifest', 'registration_tool'}
OURS_FQL_CHANGED = P1.CHANGED - {'derived_from', 'supersedes'} | {
    'deps', 'fql_condition', 'realworld_consumer', 'realworld_gate_sha256', 'realworld_manifest_sha256',
    'protocol_template'}
EXECUTION_REFUSAL = P1.EXECUTION_REFUSAL


def indices(args, key):
    return P1.indices(args, key)


def set_arg(args, key, value, job):
    found = indices(args, key)
    require(len(found) == 1, f'expected exactly one {key}: {job}')
    args[found[0]] = f'{key}={value}'


def set_option(args, option, value, job):
    require(args.count(option) == 1 and args.index(option) + 1 < len(args), f'expected exactly one {option}: {job}')
    args[args.index(option) + 1] = value


# ------------------------------------------------------------------------------------------------ snerl
def snerl_rows(by_id, release):
    rows = []
    for setting in SETTINGS:
        for task in TASKS:
            for seed in SEEDS:
                source_id = f'rwcal-20261109-fql-snerl-{task}-s{seed}'
                require(source_id in by_id, f'derived_from missing from queue: {source_id}')
                source = by_id[source_id]
                P1.check_source(source, 'snerl', task, seed)
                require(source['deps'][0] in by_id, f'dependency missing from queue: {source["deps"][0]}')
                job = P1.new_id(setting, 'snerl', task, seed)
                row = P1.derive(source, job, setting, 'snerl', task, seed)
                P1.check_derived(row, source, setting)
                P1.pin(row, release)
                row['registration_tool'] = TOOL
                rows.append(row)
    return rows


# ------------------------------------------------------------------------------------------------ ours
def ours_data_row(by_id, task, release):
    template_id, export_id = SINCRO_DATA.format(task=task), OURS_EXPORT.format(task=task)
    require(template_id in by_id and export_id in by_id, f'template/export row missing: {template_id}, {export_id}')
    template, export = by_id[template_id], by_id[export_id]
    job = OURS_DATA.format(task=task)
    require(template.get('script') == 'scripts/realworld_fql_dataset.py' and template.get('fql_condition') == 'sincro'
            and template.get('realworld_target') == task and template.get('host') == 'remote', f'template: {job}')
    require(export.get('script') == 'scripts/export_encoder.py' and export.get('realworld_consumer') == 'ours'
            and export.get('realworld_target') == task and export.get('deps') == [OURS_PRE.format(task=task)],
            f'ours export row: {export_id}')
    inputs = export['realworld_inputs']
    require(template.get('realworld_inputs') == inputs, f'SinCro dataset row reads other inputs than ours: {job}')
    row = copy.deepcopy(template)
    args = row['args']
    set_option(args, '--condition', 'ours', job)
    set_option(args, '--out', f'runs/{job}/dataset.h5', job)
    set_option(args, '--export', f'runs/{export_id}/encoder.pt', job)
    for option, name in (('--store', 'store.yaml'), ('--gate-report', 'gate_report.json'),
                         ('--outcomes', 'outcomes.json')):
        require(args[args.index(option) + 1] == f'{inputs}/{name}', f'{option} is not the ours input: {job}')
    row.update(id=job, deps=[export_id], fql_condition='ours', realworld_consumer='fql_ours',
               realworld_gate_sha256=export['realworld_gate_sha256'],
               realworld_manifest_sha256=export['realworld_manifest_sha256'], plan_row_id=f'rw-fqldata-ours-{task}',
               purpose=PURPOSE_OURS_DATA, protocol_template=template_id)
    expected = list(template['args'])
    for option, value in (('--condition', 'ours'), ('--out', f'runs/{job}/dataset.h5'),
                          ('--export', f'runs/{export_id}/encoder.pt')):
        expected[expected.index(option) + 1] = value
    require(row['args'] == expected, f'args differ from the template beyond condition/out/export: {job}')
    check_other_fields(row, template, DATA_CHANGED)
    P1.pin(row, release)
    row['registration_tool'] = TOOL
    return row


def ours_fql_row(by_id, data_row, task, seed, setting, release):
    template_id = SINCRO_FQL.format(task=task, seed=seed)
    require(template_id in by_id, f'protocol template missing from queue: {template_id}')
    template = by_id[template_id]
    P1.check_source(template, 'sincro', task, seed)
    inputs = by_id[OURS_EXPORT.format(task=task)]['realworld_inputs']
    for key, name in (('data.split_manifest', 'split_seed0.json'), ('data.gate_report', 'gate_report.json')):
        found = [a for a in template['args'] if a.partition('=')[0] == key]
        require(found == [f'{key}={inputs}/{name}'], f'template {key} is not the ours input: {template_id}')
    job = P1.new_id(setting, 'ours', task, seed)
    row = P1.derive(template, job, setting, 'ours', task, seed)
    row.pop('derived_from', None)
    set_arg(row['args'], 'condition', 'ours', job)
    set_arg(row['args'], 'data.dataset', f'runs/{data_row["id"]}/dataset.h5', job)
    row.update(deps=[data_row['id']], fql_condition='ours', realworld_consumer='fql_ours',
               realworld_gate_sha256=data_row['realworld_gate_sha256'],
               realworld_manifest_sha256=data_row['realworld_manifest_sha256'], protocol_template=template_id,
               purpose=PURPOSE_OURS_FQL[setting])
    _, alpha, normalize, offset = SETTINGS[setting]
    expected = list(template['args'])
    expected[expected.index('--name') + 1] = job
    for key, value in (('condition', 'ours'), ('data.dataset', f'runs/{data_row["id"]}/dataset.h5'),
                       ('agent.alpha', alpha)):
        expected[indices(expected, key)[0]] = f'{key}={value}'
    expected.append(f'agent.normalize_q_loss={str(normalize).lower()}')
    require(row['args'] == expected, f'args differ from the template beyond the allowed changes: {job}')
    require(row['priority'] == template['priority'] + offset and row['fql_setting'] == setting
            and canonical(row['fql_budget']) == canonical({**template['fql_budget'], 'alpha': float(alpha),
                                                           'normalize_q_loss': normalize}), f'setting fields: {job}')
    check_other_fields(row, template, OURS_FQL_CHANGED)
    P1.pin(row, release)
    row['registration_tool'] = TOOL
    return row


def check_other_fields(row, template, changed):
    keys = (set(row) | set(template)) - changed
    diff = sorted(k for k in keys if k not in row or k not in template or canonical(row[k]) != canonical(template[k]))
    require(not diff, f'fields differ from the template {diff}: {row["id"]}')


def ours_rows(by_id, release):
    data = [ours_data_row(by_id, task, release) for task in TASKS]
    fql = [ours_fql_row(by_id, data[TASKS.index(task)], task, seed, setting, release)
           for setting in SETTINGS for task in TASKS for seed in SEEDS]
    return data + fql


def release_has_ours(repo, commit):
    """The pinned release must define the FQL condition ours (s4d/rwlearn/observation.py, gate.FQL_CONSUMER)."""
    def show(path):
        r = subprocess.run(['git', '-C', str(repo), 'show', f'{commit}:{path}'], capture_output=True, text=True)
        require(r.returncode == 0, f'cannot read {path} at {commit[:7]}')
        return r.stdout
    require('"ours": Condition("splatter4d"' in show('s4d/rwlearn/observation.py')
            and '"ours": "fql_ours"' in show('s4d/rwlearn/gate.py'),
            f'release {commit[:7]} has no FQL condition ours: register --group ours after its release')


# ------------------------------------------------------------------------------------------------ common
def required_pauses(row):
    """Pauses that must be active for a row at publication."""
    if row['script'] == 'scripts/realworld_fql_dataset.py':
        return []
    hold = SNERL_HOLD if row['fql_condition'] == 'snerl' else OURS_HOLD
    return [hold] + ([A100_PAUSE] if row['fql_setting'] == 'a100' else [])


def pause_status(rows, active_pauses):
    status = {}
    for row in rows:
        names = required_pauses(row)
        if names:
            active = active_pauses(row['id'], fresh=True)
            status[row['id']] = all(n in active for n in names)
    return status


def build(group, current, release, repo):
    by_id = {r['id']: r for r in current['jobs']}
    if group == 'ours':
        release_has_ours(repo, release['commit'])
        rows = ours_rows(by_id, release)
        require(len(rows) == 2 + 12, 'wrong ours row count')
    else:
        rows = snerl_rows(by_id, release)
        require(len(rows) == 12, 'wrong snerl row count')
    require(len({r['id'] for r in rows}) == len(rows), 'duplicate ids')
    return rows


def check_run_paths(repo, rows):
    require(not any((repo / 'runs' / r['id']).exists() or (repo / 'runs' / r['id']).is_symlink() for r in rows),
            'a run directory already exists')


def preview_pin(commit, image):
    """Dry-run-only pin of a candidate commit that is not the role release yet (``--preview``; never published)."""
    require(P1.re.fullmatch(r'[0-9a-f]{40}', commit or '') and P1.re.fullmatch(r'sha256:[0-9a-f]{64}', image or ''),
            '--commit needs 40 hex and --image sha256:<64 hex>')
    return {'commit': commit, 'image': image, 'evidence_manifest': f'runs/remote/acceptance/{commit[:7]}-cc/manifest.json',
            'preview': True}


def prepare(repo, group, commit, image, active_pauses=None, preview=False):
    require(group in GROUPS, f'--group must be one of {GROUPS}')
    release = preview_pin(commit, image) if preview else P1.release_pin(repo, commit, image)
    setup, stem = repo / 'runs/setup', STEM.format(group=group)
    require(not (setup / f'{stem}.pending.json').exists() and not (setup / f'{stem}.json').exists(),
            'pending transaction/proof exists; inspect before retrying')
    before = (repo / QUEUE).read_bytes()
    registry = (repo / REGISTRY).read_bytes() if (repo / REGISTRY).exists() else b''
    current = P1.queue_document(before)
    rows = build(group, current, release, repo)
    ids = {r['id'] for r in current['jobs']} | P1.registry_ids(registry)
    require(not ids.intersection(r['id'] for r in rows), 'a proposed ID already exists in queue or registry')
    check_run_paths(repo, rows)
    candidate = P1.queue_candidate(before, rows)
    paused = pause_status(rows, active_pauses or P1.default_active_pauses(repo))
    return {'schema': 'rw-registration/v1', 'status': 'dry-run', 'registered': False, 'prepared_at': now(),
            'group': group, 'repo': str(repo.resolve()), 'release': release,
            'evidence_manifest': release['evidence_manifest'],
            'required_pauses': {r['id']: required_pauses(r) for r in rows if required_pauses(r)},
            'pauses_active_now': f'{sum(paused.values())}/{len(paused)}',
            'jobs': rows, 'jobs_sha256': json_sha(rows), 'tool_sha256': sha(Path(__file__).read_bytes()),
            'helper_sha256': sha(Path(P1.__file__).read_bytes()),
            'registry_exists_at_evaluation': (repo / REGISTRY).exists(), 'registry_sha256_at_evaluation': sha(registry),
            'queue_sha256_before': sha(before), 'queue_sha256_after': sha(candidate),
            'queue_diff': {'before_rows': len(current['jobs']), 'after_rows': len(current['jobs']) + len(rows),
                           'added_rows': len(rows), 'modified_rows': 0, 'removed_rows': 0,
                           'counts': dict(Counter(r.get('fql_setting', 'dataset') for r in rows)),
                           'added_ids': [r['id'] for r in rows],
                           'derived_from': {r['id']: r.get('derived_from') or r.get('protocol_template')
                                            for r in rows}}}


def execution_allowed(repo):
    require(repo.resolve() == MAIN and Path(__file__).resolve() == MAIN / TOOL, EXECUTION_REFUSAL)


def check_publication(repo, record, active_pauses):
    execution_allowed(repo)
    require(not record['release'].get('preview'), 'a --preview record is never published')
    require(sha((repo / QUEUE).read_bytes()) == record['queue_sha256_before'], 'queue changed under locks')
    registry = repo / REGISTRY
    require(registry.exists() == record['registry_exists_at_evaluation']
            and sha(registry.read_bytes() if registry.exists() else b'') == record['registry_sha256_at_evaluation'],
            'registry changed under locks')
    require(sha(Path(__file__).read_bytes()) == record['tool_sha256']
            and sha(Path(P1.__file__).read_bytes()) == record['helper_sha256'], 'registration tool changed')
    require(P1.release_pin(repo, record['release']['commit'], record['release']['image']) == record['release'],
            'role release changed')
    check_run_paths(repo, record['jobs'])
    missing = sorted(i for i, ok in pause_status(record['jobs'], active_pauses).items() if not ok)
    require(not missing, f'{len(missing)} rows lack their active pauses (add them first), e.g. {missing[:2]}')


def execute(repo, record, active_pauses=None):
    execution_allowed(repo)
    active_pauses = active_pauses or P1.default_active_pauses(repo)
    setup, stem = repo / 'runs/setup', STEM.format(group=record['group'])
    pending, proof = setup / f'{stem}.pending.json', setup / f'{stem}.json'
    candidate_path = setup / f'{stem}.queue-candidate.yaml'
    with P1.locks(repo):
        require(not pending.exists() and not proof.exists(), 'pending transaction/proof exists; inspect before retrying')
        require(record['repo'] == str(repo.resolve()) and json_sha(record['jobs']) == record['jobs_sha256'],
                'registration plan changed')
        check_publication(repo, record, active_pauses)
        before = (repo / QUEUE).read_bytes()
        rebuilt = build(record['group'], P1.queue_document(before), record['release'], repo)
        require(json_sha(rebuilt) == record['jobs_sha256'], 'rows re-derived under locks differ from the plan')
        candidate = P1.queue_candidate(before, record['jobs'])
        require(sha(candidate) == record['queue_sha256_after'], 'staged plan changed')
        P1.atomic_write(candidate_path, candidate)
        record.update(status='pending', proof_file=proof.name)
        P1.atomic_write(pending, json_bytes(record))
        check_publication(repo, record, active_pauses)
        import os
        os.replace(candidate_path, repo / QUEUE)
        P1.sync_directory((repo / QUEUE).parent)
        record.update(status='published', registered=True, registered_at=now())
        P1.atomic_write(proof, json_bytes(record))
        pending.unlink()
        P1.sync_directory(setup)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--repo', type=Path, default=MAIN)
    parser.add_argument('--group', required=True, choices=GROUPS)
    parser.add_argument('--commit', required=True, help='remote role release commit (runs/setup/role-release.json)')
    parser.add_argument('--image', required=True, help='remote role release image digest')
    parser.add_argument('--out', type=Path, help='dry run only: also save the record here (never runs/setup)')
    parser.add_argument('--preview', action='store_true',
                        help='dry run only: pin --commit although it is not the role release yet (never published)')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    require(not (args.out and args.execute), '--out is only for dry-run records')
    require(not (args.preview and args.execute), '--preview is only for dry runs')
    if args.execute:
        execution_allowed(args.repo)
    record = prepare(args.repo, args.group, args.commit, args.image, preview=args.preview)
    if args.execute:
        record = execute(args.repo, record)
    if args.out:
        path = args.out.parent.resolve() / args.out.name
        require(not any(p.name == 'setup' and p.parent.name == 'runs' for p in path.parents)
                and not (path.parent.name == 'experiments' and path.name in ('queue.yaml', 'registry.jsonl')),
                '--out must not overwrite operational files')
        require(not path.exists() and not path.is_symlink(), '--out already exists')
        with path.open('xb') as stream:
            stream.write(json_bytes(record))
    print(json_bytes(record).decode(), end='')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except P1.ERRORS as exc:
        print(f'REFUSED: {exc}', file=sys.stderr)
        raise SystemExit(2) from exc
