"""PROPOSAL (Codex n01, completed by Claude): register ONE new real-world task (covering or pick-and-place).
Default: DRY RUN (stdout only; --out saves the proof); no network, no launch API, no registry events.

    nice -n 15 /home/ws/ws/splatter4d/.venv/bin/python -I -B \
        runs/setup/tools/register_realworld_newtasks_20261110.py --task covering --family image \
        --inputs-copy <local mirror> --remote-inputs <remote dir ending in /covering>

Proposed final location: runs/setup/tools/ of the main checkout (bookkeeping, not fingerprinted), installed after the
release that contains the covering / pick-and-place extension passed the coordinator's tests and the remote release
gate. Per task, a full pass is exactly the rows each existing task got: 14 image rows (ReViWo + DynaMo pretraining,
their FQL datasets, e2e_cnn dataset, 3 FQL seeds each) + 19 geometry rows (SNeRL, 3D-MVP, SinCro pretraining; SinCro,
SNeRL, 3D-MVP multiview + single datasets; 3 FQL seeds each), ids rwnt-20261110-<plan id without rw->, plan
priorities / GPU / RAM / restarts / role, pinned to runs/setup/role-release.json AS READ AT REGISTRATION TIME.
--family image|geometry admits the two consumer families independently, but never a partial family: SNeRL / 3D-MVP
stay refused by the planner until the task's scene geometry is measured (s4d/rwlearn/configs.py SCENE_PARAMETERS),
and geometry rows need the calibration hold; a family with any refused row is refused as a whole.

Learning inputs come from scripts/realworld_learning.py inputs --task T --ending-report R (store.yaml,
gate_report.json, split_seed0.json, outcomes.json), staged on the remote under --remote-inputs and mirrored locally in
--inputs-copy; all four are verified against the live gate (100 train_data outputs, bound raw identities, calibration
builds, QA families, seed-0 90/10 split, all-success labels carrying ending-check decisions) and hashed.
--publish is a FUTURE locked/journalled queue append; --recover repairs only its proof after an interrupted append.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True
ROOT = next((p for p in (*Path(__file__).resolve().parents, Path.cwd()) if (p / 's4d/rwlearn').is_dir()), Path.cwd())
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from ops.jobs import host_roles  # noqa: E402
from ops.resources.gpu_guard import APPROVED_HOST_GPUS  # noqa: E402  (the scheduler's own allowlist object)
from s4d.rwlearn import configs, gate, launch, outcomes  # noqa: E402
from s4d.rwlearn.store import split_episodes  # noqa: E402

PREFIX = 'rwnt-20261110-'
STEM = 'rw-registration-newtasks-20261110'
TASKS = tuple(t for t in configs.TARGETS if t in ('covering', 'pick-and-place'))
INPUT_NAMES = ('store.yaml', 'gate_report.json', 'split_seed0.json', 'outcomes.json')
OVERRIDE_KEYS = {'dataset.hdf5_path': 'store.yaml', 'data_interface.split_manifest': 'split_seed0.json',
                 'data_interface.gate_report': 'gate_report.json'}
DISK_GB = {'pretrain': 20, 'dataset': 20, 'fql': 10}  # as the two existing real-world registrations
# exactly what each existing task got (register_realworld_nongeometric_20261106 + register_realworld_geometric_20261109)
FAMILY_COUNTS = {'image': {'image-pretrain': 2, 'image-dataset': 3, 'image-fql': 9},
                 'geometry': {'geometry-pretrain': 3, 'geometry-dataset': 4, 'geometry-fql': 12}}


class Refused(RuntimeError):
    pass


def require(ok, why):
    if not ok:
        raise Refused(why)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def json_sha(value):
    return sha(json.dumps(value, sort_keys=True).encode())


def kind(row):
    return ('pretrain' if row['role'] == 'realworld_pretrain' else
            'dataset' if row['script'] == 'scripts/realworld_fql_dataset.py' else 'fql')


def release_pin(repo):
    release = json.loads((repo / 'runs/setup/role-release.json').read_bytes())
    require(isinstance(release, dict) and re.fullmatch(r'[0-9a-f]{40}', str(release.get('commit', ''))),
            'role-release.json needs a full commit')
    require(re.fullmatch(r'sha256:[0-9a-f]{64}', str(release.get('image', ''))), 'role release needs an image digest')
    return {k: release[k] for k in ('commit', 'image')}


def source_paths(task):
    """What the rows execute or were planned with (the scheduler-side role files are recorded by hash instead)."""
    return ('s4d/rwlearn/tasks.py', 's4d/rwlearn/configs.py', 's4d/rwlearn/gate.py', 's4d/rwlearn/outcomes.py',
            's4d/rwlearn/launch.py', 'ops/jobs/realworld_plan.py',
            's4d/rwlearn/identity.py', 's4d/rwlearn/store.py', 's4d/rwlearn/baseline_data.py',
            's4d/rwlearn/actions.py', 's4d/rwlearn/fql_data.py', 'configs/realworld/fql/base.yaml',
            'scripts/realworld_learning.py', 'scripts/train_realworld_fql.py', 'scripts/realworld_fql_dataset.py',
            *(f'scripts/train_{b}.py' for b in configs.BASELINES),
            *(f'configs/realworld/baselines/{b}/{task}.yaml' for b in configs.BASELINES))


def release_evidence(repo, task, release):
    acceptance = repo / f"runs/remote/acceptance/{release['commit'][:7]}-cc/manifest.json"
    raw = acceptance.read_bytes()
    rec = json.loads(raw)
    require(rec.get('commit') == release['commit'] and rec.get('image_digest') == release['image'],
            'acceptance manifest differs from active role release')
    evidence = [rec.get(k) for k in ('cross_host', 'data', 'full_suite')]
    gpu = rec.get('gpu_isolation') or {}
    require(set(gpu) == set(APPROVED_HOST_GPUS['remote']), 'acceptance GPU coverage differs from the remote allowlist')
    evidence += list(gpu.values())
    for item in evidence:
        require(isinstance(item, dict) and isinstance(item.get('path'), str), 'missing acceptance evidence')
        path = repo / item['path']
        require(path.resolve().is_relative_to(acceptance.parent.resolve()) and path.is_file()
                and sha(path.read_bytes()) == item.get('sha256'), 'missing/changed acceptance evidence')
    sources = {}
    for name in source_paths(task):
        current = (repo / name).read_bytes()
        pinned = subprocess.run(['git', '-C', str(repo), 'show', f"{release['commit']}:{name}"],
                                capture_output=True, check=True).stdout
        require(current == pinned, f'source differs from active release: {name}')
        sources[name] = sha(current)
    return {'acceptance_manifest': str(acceptance.relative_to(repo)), 'acceptance_sha256': sha(raw),
            'source_sha256': sources}


def verify_inputs(task, manifest, report, directory):
    """Pure local reads; compare the staged descriptor and bound report to live gate identities/builds."""
    files = {name: Path(directory) / name for name in INPUT_NAMES}
    require(all(p.is_file() for p in files.values()), 'learning-inputs copy needs all four files')
    store = yaml.safe_load(files['store.yaml'].read_bytes())
    live = report['tasks'][task]
    want = sorted(manifest['episodes'][ep]['preprocessing']['stages']['train_data']['outputs']
                  for ep in live['episodes'])
    require(len(want) == gate.TARGET and len(set(want)) == gate.TARGET, 'need 100 unique train_data outputs')
    require(store.get('kind') == 'compact' and sorted(store.get('paths') or []) == want,
            'store.yaml differs from the live 100 train_data outputs')
    saved = json.loads(files['gate_report.json'].read_bytes())
    rec = saved['tasks'][task]
    eps = rec.get('store_episodes') or []
    require(len(eps) == gate.TARGET and len(set(eps)) == gate.TARGET, 'gate report must bind 100 store episodes')
    identities = {v['store_episode']: v for v in rec['episode_identities'].values()}
    bound = gate.bind_store_episodes(report, {task: eps}, identities)
    # This task's sections must equal the live gate bound to the same store episodes. The manifest bytes may differ
    # (ingestion of other tasks keeps writing it); both hashes are recorded, as the existing registrars do.
    for section in ('tasks', 'calibration_hold', 'families'):
        require(saved[section][task] == bound[section][task], f'staged {section} differs from live gate')
    require(isinstance(saved.get('manifest', {}).get('sha256'), str), 'staged gate report names no manifest')
    split = json.loads(files['split_seed0.json'].read_bytes())
    train, val = split.get('train') or [], split.get('validation') or []
    require(len(train) == 90 and len(val) == 10 and len(set(train + val)) == 100
            and set(train + val) == set(eps), 'split must be a disjoint 90/10 partition of this task')
    identity_store = SimpleNamespace(episodes=lambda: eps, task=lambda ep: task,
                                     raw_recording=lambda ep: identities[ep]['raw_recording'],
                                     raw_sha256=lambda ep: identities[ep].get('raw_sha256'))
    require(split == split_episodes(identity_store), 'split differs from the established task-specific seed-0 split')
    labels = json.loads(files['outcomes.json'].read_bytes())
    require(labels.get('schema') == outcomes.OUTCOME_SCHEMA and labels.get('flags_change_labels') is False,
            'need recorded success labels with flags preserved')
    require(outcomes.load_outcomes(labels) == dict.fromkeys(eps, True), 'outcomes differ from the 100 successful demos')
    require(all(r.get('ending_decision') in outcomes.DECISIONS and isinstance(r.get('ending_flags'), list)
                for r in labels['episodes'].values()), 'new tasks need ending-check evidence (flags never exclude demos)')
    return {name: sha(path.read_bytes()) for name, path in files.items()}


def adapt_rows(templates, report, release, remote_inputs):
    """Same naming/artifact/resource adaptation as the existing registrars; no reused old-task dependencies."""
    mapping = {r['id']: PREFIX + r['id'][3:] for r in templates}
    paths = {}
    for r in templates:
        if kind(r) == 'pretrain':
            paths[f"runs/pretrain/{r['id']}/encoder.pt"] = f"runs/{mapping[r['id']]}/encoder.pt"
        elif kind(r) == 'dataset':
            paths[r['args'][r['args'].index('--out') + 1]] = f"runs/{mapping[r['id']]}/dataset.h5"
    rows = []
    for template in templates:
        row = copy.deepcopy(template)
        old_inputs = configs.run_inputs_dir(row['realworld_target'])
        row['id'] = mapping[template['id']]
        row['deps'] = [mapping[d] for d in row['deps']]
        args = row['args']
        for i, value in enumerate(args):
            if value in mapping:
                args[i] = mapping[value]
            elif value in paths:
                args[i] = paths[value]
            elif value.startswith('data.dataset='):
                args[i] = 'data.dataset=' + paths[value.split('=', 1)[1]]
            args[i] = args[i].replace(old_inputs + '/', remote_inputs + '/')
        if '--output-root' in args:
            args[args.index('--output-root') + 1] = 'runs'
        k = kind(row)
        if k == 'pretrain':
            args += ['--set', *(f'{key}={remote_inputs}/{name}' for key, name in OVERRIDE_KEYS.items())]
        elif k == 'fql':
            args += ['agent.alpha=10.0', 'agent.batch_size=256', 'train.offline_steps=1000000',
                     'train.log_interval=5000', 'train.eval_interval=100000']
        if row.get('inputs'):
            row['inputs'] = [p.replace(old_inputs + '/', remote_inputs + '/') for p in row['inputs']]
        if remote_inputs != old_inputs:
            require(not any(old_inputs + '/' in str(x) for x in args + row.get('inputs', [])),
                    f"old inputs left: {row['id']}")
        row.update(pin_host='remote', planned_commit=release['commit'], planned_image=release['image'],
                   disk_gb=DISK_GB[k], realworld_inputs=remote_inputs, plan_row_id=template['id'],
                   realworld_gate_sha256=json_sha(report), realworld_manifest_sha256=report['manifest']['sha256'],
                   registration_tool='runs/setup/tools/' + Path(__file__).name,
                   evidence_manifest=f"runs/remote/acceptance/{release['commit'][:7]}-cc/manifest.json",
                   required_results=(['encoder.pt', 'metrics.jsonl', 'completion.json'] if k == 'pretrain' else
                                     ['dataset.h5'] if k == 'dataset' else
                                     ['policy.pt', 'metrics.jsonl', 'completion.json']))
        require(all(row[key] == template[key] for key in ('mem_gb', 'ram_gb', 'gpu', 'priority', 'role', 'max_restarts')),
                'changed compute resources/role/priority')
        rows.append(row)
    return sorted(rows, key=lambda row: row['priority'])


def plan_task(task, report, release, remote_inputs, *, family='all'):
    require(task in TASKS, 'exactly one new real-world task is required')
    require(family in ('all', 'image', 'geometry'), 'unknown consumer family')
    require(report.get('target') == gate.TARGET and report['tasks'][task]['passed'], '100-episode gate refused')
    result = launch.plan(report, [task], release['commit'])
    templates = [r for r in result['rows'] if family == 'all' or
                 r['calibration_dependent'] == (family == 'geometry')]
    rows = adapt_rows(templates, report, release, remote_inputs)
    counts = dict(Counter(f"{'geometry' if r['calibration_dependent'] else 'image'}-{kind(r)}" for r in rows))
    want = {k: v for f, c in FAMILY_COUNTS.items() if family in ('all', f) for k, v in c.items()}
    # never a partial family: rows registered now would collide with the rest of the family later
    require(counts == want, f'{task}/{family}: incomplete consumer family {counts} != {want}; planner refused '
            + json.dumps(result['refused'], sort_keys=True)[:1500])
    return {'jobs': rows, 'plan_refused': result['refused'], 'counts': counts}


def prepare(repo, task, manifest_path, inputs_copy, remote_inputs, *, family='all'):
    release = release_pin(repo)
    report = gate.evaluate_file(manifest_path, [task])
    result = plan_task(task, report, release, remote_inputs, family=family)
    evidence = verify_inputs(task, json.loads(Path(manifest_path).read_bytes()), report, inputs_copy)
    require(sha(Path(manifest_path).read_bytes()) == report['manifest']['sha256'], 'manifest changed during evaluation')
    for row in result['jobs']:  # the planner already refused unmeasured scenes; the released configs must agree
        if row.get('baseline') in configs.SCENE_KEYS:
            cfg = yaml.safe_load((repo / f"configs/realworld/baselines/{row['baseline']}/{task}.yaml").read_bytes())
            section = cfg['data_interface'] if row['baseline'] == 'snerl' else cfg['render']
            require(all(section.get(k) is not None for k in configs.SCENE_KEYS[row['baseline']]),
                    f"{row['id']}: released config has unmeasured scene parameters")
    release_proof = release_evidence(repo, task, release)
    roles = host_roles.load_roles(repo / 'configs/hosts/roles.yaml')
    by_id = {r['id']: r for r in result['jobs']}
    for row in result['jobs']:
        require(gate.task_passed(report, task, row['realworld_consumer']), f"consumer gate refused: {row['id']}")
        require(host_roles.role_host(row, roles, by_id) == 'remote'
                and host_roles.refusal(row, 'remote', roles, {}, by_id) is None, f"remote role refused: {row['id']}")
        require(all(dep in by_id for dep in row['deps']), f"missing dependency: {row['id']}")
    return {'schema': 'rw-registration/v1', 'status': 'dry-run', 'task': task, 'family': family, 'release': release,
            'release_evidence': release_proof, 'roles_sha256': roles['sha256'], 'gate_evaluation': report,
            'learning_inputs': {'copy': str(inputs_copy), 'remote_dir': remote_inputs, 'sha256': evidence,
                                'staged_manifest_sha256': json.loads((Path(inputs_copy) / 'gate_report.json')
                                                                     .read_bytes())['manifest']['sha256']},
            **result, 'jobs_sha256': json_sha(result['jobs']), 'tool_sha256': sha(Path(__file__).read_bytes()),
            'registered': False,
            'not_registered': {'ours': 'no plan row: our own real-world pretraining stays deferred (user decision)'},
            'decisions': {'alpha': 10, 'gripper': 'continuous', 'encoders': 'task-specific', 'snerl_inputs': 3,
                          'single_view': 'random camera per sample, fixed through history + paired next obs',
                          'fql_seeds': list(launch.SEEDS), 'fql_budget': launch.FQL_BUDGET,
                          'reward': 'success(s\') - 1, success at the final frame; every demo labelled success'}}


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


def queue_candidate(before, rows):
    """Pure queue-text edit: preserve existing rows and comments, verify parsed equality."""
    parsed = yaml.safe_load(before)
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


@contextmanager
def locks(repo):
    with (repo / 'runs/setup/stage2-driver.lock').open('a') as admission:
        fcntl.flock(admission, fcntl.LOCK_EX)
        with (repo / 'experiments/registry.lock').open('a') as registry:
            fcntl.flock(registry, fcntl.LOCK_EX)
            yield


def check_unchanged(repo, record):
    require(release_pin(repo) == record['release'], 'active release changed')
    acceptance = record['release_evidence']
    require(sha((repo / acceptance['acceptance_manifest']).read_bytes()) == acceptance['acceptance_sha256'],
            'acceptance manifest changed')
    require(sha(Path(__file__).read_bytes()) == record['tool_sha256'], 'registration tool changed')
    report = record['gate_evaluation']
    require(sha(Path(report['manifest']['path']).read_bytes()) == report['manifest']['sha256'], 'manifest changed')
    for name, digest in record['release_evidence']['source_sha256'].items():
        require(sha((repo / name).read_bytes()) == digest, f'source changed: {name}')
    for name, digest in record['learning_inputs']['sha256'].items():
        require(sha((Path(record['learning_inputs']['copy']) / name).read_bytes()) == digest, f'inputs changed: {name}')
    require(host_roles.load_roles(repo / 'configs/hosts/roles.yaml')['sha256'] == record['roles_sha256'], 'roles changed')


def publish(repo, options):
    # Import only on the explicit publication path; pure tests/planning never load scheduler/runtime clients.
    from ops.jobs import scheduler

    setup = repo / 'runs/setup'
    pending, proof = setup / f'{STEM}.pending.json', setup / f'{STEM}-{options.task}-{options.family}.json'
    queue = repo / 'experiments/queue.yaml'
    with locks(repo):
        require(not pending.exists() and not proof.exists(), 'pending transaction/proof exists; inspect/recover first')
        record = prepare(repo, options.task, options.manifest, options.inputs_copy, options.remote_inputs,
                         family=options.family)
        before = queue.read_bytes()
        current = scheduler.load_queue(queue)
        known = scheduler.read_registry(repo / 'experiments/registry.jsonl')
        ids = {r['id'] for r in current['jobs']} | set(known)
        require(not ids.intersection(r['id'] for r in record['jobs']), 'a proposed ID already exists')
        require(not any((repo / 'runs' / r['id']).exists() for r in record['jobs']), 'a run directory already exists')
        candidate = queue_candidate(before, record['jobs'])
        candidate_path = setup / f'{STEM}.queue-candidate.yaml'
        atomic_write(candidate_path, candidate)
        scheduler.load_queue(candidate_path)
        record.update(status='pending', queue_sha256_before=sha(before), queue_sha256_after=sha(candidate),
                      proof_file=proof.name)
        atomic_write(pending, (json.dumps(record, indent=2) + '\n').encode())
        check_unchanged(repo, record)
        require(queue.read_bytes() == before, 'queue changed under locks')
        os.replace(candidate_path, queue)
        sync_directory(queue.parent)
        record.update(status='published', registered=True)
        atomic_write(proof, (json.dumps(record, indent=2) + '\n').encode())
        pending.unlink()
        sync_directory(setup)
    return record


def recover(repo):
    setup, queue = repo / 'runs/setup', repo / 'experiments/queue.yaml'
    pending = setup / f'{STEM}.pending.json'
    with locks(repo):
        record = json.loads(pending.read_bytes())
        require(record['jobs'] and json_sha(record['jobs']) == record['jobs_sha256'], 'journal rows changed')
        current = queue.read_bytes()
        if sha(current) == record['queue_sha256_before']:
            (setup / f'{STEM}.queue-candidate.yaml').unlink(missing_ok=True)
            pending.unlink()
            sync_directory(setup)
            return {'status': 'aborted staging; queue unchanged', 'registered': False}
        require(sha(current) == record['queue_sha256_after'], 'queue changed; manual recovery required')
        require(yaml.safe_load(current)['jobs'][-len(record['jobs']):] == record['jobs'], 'queue tail differs from journal')
        check_unchanged(repo, record)
        record.update(status='published', registered=True)
        require(Path(record['proof_file']).name == record['proof_file'], 'invalid proof path')
        atomic_write(setup / record['proof_file'], (json.dumps(record, indent=2) + '\n').encode())
        pending.unlink()
        sync_directory(setup)
        return record


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo', type=Path, default=ROOT)
    ap.add_argument('--task', choices=TASKS)
    ap.add_argument('--manifest', type=Path, default=gate.MANIFEST)
    ap.add_argument('--inputs-copy', type=Path)
    ap.add_argument('--remote-inputs', help='absolute staged remote directory for this task')
    ap.add_argument('--family', choices=('all', 'image', 'geometry'), default='all')
    ap.add_argument('--out', type=Path, help='dry-run proof only; queue and registry unchanged')
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument('--publish', action='store_true')
    mode.add_argument('--recover', action='store_true')
    args = ap.parse_args()
    require(not args.out or not (args.publish or args.recover), '--out is only for dry-run proofs')
    if args.recover:
        result = recover(args.repo)
    else:
        require(args.task and args.inputs_copy and args.remote_inputs, '--task, --inputs-copy and --remote-inputs required')
        remote = Path(args.remote_inputs)
        require(remote.is_absolute() and '..' not in remote.parts and remote.name == args.task,
                '--remote-inputs must be an absolute directory ending in the task name')
        result = (publish(args.repo, args) if args.publish else
                  prepare(args.repo, args.task, args.manifest, args.inputs_copy, str(remote), family=args.family))
    if args.out:
        # Explicit output only; a dry run otherwise writes nothing.
        atomic_write(args.out, (json.dumps(result, indent=2) + '\n').encode())
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    try:
        main()
    except (Refused, OSError, ValueError, RuntimeError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        print(f'REFUSED: {exc}', file=sys.stderr)
        raise SystemExit(2) from exc
