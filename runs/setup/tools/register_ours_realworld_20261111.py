"""OUR real-world pretraining registration. Default: stdout-only DRY RUN; no network.

Only the coordinator's main-checkout copy may publish, after the role release
matches --commit/--image. Publication uses candidate/journal/proof files and
stage2-driver.lock then registry.lock. Inspect a pending journal before retrying.
FQL specifications are held evidence and are never queue rows.
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
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True
SOURCE = next(p for p in Path(__file__).resolve().parents if (p / 's4d/rwlearn').is_dir())
sys.path.insert(0, str(SOURCE))
import yaml  # noqa: E402
from s4d.rwlearn import gate, ours  # noqa: E402
from s4d.rwlearn.store import split_episodes  # noqa: E402

MAIN = Path('/home/ws/ws/splatter4d')
TASKS = ('drawer-open', 'mug-hanging', 'covering', 'pick-and-place')
INPUT_NAMES = ('store.yaml', 'gate_report.json', 'split_seed0.json', 'outcomes.json')
DEFAULT_INPUTS = {t: f'/home/compu/kaist/sunho/realworld/learning_inputs/calrel-6c3b9ec/{t}' for t in TASKS[:2]}
# Measured on the local MIG smokes (realworld-ops/ours/DESIGN.md section 6): training peaks 2.27 GiB allocated
# (sim Original 3.76 -> its rows asked 10), PSS 4.3 GB with 6 + 3 loader workers on 3 episodes (per-worker episode LRU
# of 16 files bounds the full store); stats 6 s / one process on the sample; export as the simulation export rows.
FOOTPRINT = {'stats': {'mem_gb': 2, 'ram_gb': 8, 'disk_gb': 1},
             'pre': {'mem_gb': 8, 'ram_gb': 16, 'disk_gb': 20},
             'export': {'mem_gb': 2, 'ram_gb': 4, 'disk_gb': 1}}
TOOL = 'runs/setup/tools/' + Path(__file__).name
STEM = 'rw-registration-ours-realworld-20261111'
QUEUE, REGISTRY = 'experiments/queue.yaml', 'experiments/registry.jsonl'
EXECUTION_REFUSAL = "execution is the coordinator's step: copy this tool to the main checkout after the release"
HELD_BY = ("agreed ours FQL setting (after fqldiag analysis) + FQL condition 'ours' not implemented in "
           's4d/rwlearn/observation.py CONDITIONS / fql_data.py')


class Refused(RuntimeError):
    pass


ERRORS = (OSError, ValueError, RuntimeError, KeyError, TypeError, AttributeError, yaml.YAMLError)


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


def release_pin(repo):
    release = json.loads((repo / 'runs/setup/role-release.json').read_bytes())
    require(isinstance(release, dict) and re.fullmatch(r'[0-9a-f]{40}', str(release.get('commit', ''))),
            'role-release.json needs a full commit')
    require(re.fullmatch(r'sha256:[0-9a-f]{64}', str(release.get('image', ''))), 'role release needs an image digest')
    return {k: release[k] for k in ('commit', 'image')}


def execution_allowed(repo, release):
    require(repo.resolve() == MAIN and Path(__file__).resolve() == MAIN / TOOL, EXECUTION_REFUSAL)
    try:
        require(release_pin(repo) == release, EXECUTION_REFUSAL + ': role release differs')
    except ERRORS as exc:
        raise Refused(f'{EXECUTION_REFUSAL}: {exc}') from exc


def task_target(report, task):
    rec = report['tasks'][task]
    want = rec['amendment']['target'] if rec.get('amendment') else gate.TARGET
    require(rec.get('target') == want, f"{task}: gate target {rec.get('target')!r} != {want}")
    return want


def verify_inputs(task, manifest, report, directory):
    """Copied from newtasks: verify live outputs, bound sections and seed-0 split; only hash outcomes."""
    n = task_target(report, task)
    files = {name: Path(directory) / name for name in INPUT_NAMES}
    require(all(p.is_file() for p in files.values()), 'learning-inputs copy needs all four files')
    raw = {name: path.read_bytes() for name, path in files.items()}
    store = yaml.safe_load(raw['store.yaml'])
    live = report['tasks'][task]
    want = sorted(manifest['episodes'][ep]['preprocessing']['stages']['train_data']['outputs']
                  for ep in live['episodes'])
    require(len(want) == n and len(set(want)) == n, f'need {n} unique train_data outputs')
    require(store.get('kind') == 'compact' and sorted(store.get('paths') or []) == want,
            f'store.yaml differs from the live {n} train_data outputs')
    saved = json.loads(raw['gate_report.json'])
    rec = saved['tasks'][task]
    eps = rec.get('store_episodes') or []
    require(len(eps) == n and len(set(eps)) == n, f'gate report must bind {n} store episodes')
    identities = {v['store_episode']: v for v in rec['episode_identities'].values()}
    bound = gate.bind_store_episodes(report, {task: eps}, identities)
    for section in ('tasks', 'calibration_hold', 'families'):
        live_section = bound[section][task]
        if section == 'tasks':  # reports staged before the gate-amendment schema (2026-11-10) lack 'target' /
            # 'amendment'; absent there is equivalent only to the unamended default (100 episodes, no amendment)
            live_section = {k: v for k, v in live_section.items()
                            if k in saved[section][task] or not ((k == 'target' and v == gate.TARGET)
                                                                 or (k == 'amendment' and v is None))}
        require(saved[section][task] == live_section, f'staged {section} differs from live gate')
    require(isinstance(saved.get('manifest', {}).get('sha256'), str), 'staged gate report names no manifest')
    split = json.loads(raw['split_seed0.json'])
    train, val = split.get('train') or [], split.get('validation') or []
    require(len(set(train + val)) == len(train) + len(val) == n and set(train + val) == set(eps),
            'split must be a disjoint train/validation partition of this task')
    identity_store = SimpleNamespace(episodes=lambda: eps, task=lambda ep: task,
                                     raw_recording=lambda ep: identities[ep]['raw_recording'],
                                     raw_sha256=lambda ep: identities[ep].get('raw_sha256'))
    require(split == split_episodes(identity_store), 'split differs from the established task-specific seed-0 split')
    return {name: sha(value) for name, value in raw.items()}


def task_rows(task, inputs, report, release):
    ids = {k: f'rwcal-20261111-ours-{task}-{k}' for k in FOOTPRINT}
    stats, pre, export = (ids[k] for k in ('stats', 'pre', 'export'))
    common = dict(role='realworld_pretrain', host='remote', pin_host='remote', remote_host='192.168.10.22',
                  gpu='any', priority=1, oom_score_adj=0, realworld_target=task, realworld_consumer='ours',
                  calibration_dependent=True, gate_level='geometry (full-data gate + calibration_validated)',
                  pretraining='task-specific, simulation Original (Arm4-recommended) per configs/realworld/ours '
                              '(s4d/rwlearn/ours.py)',
                  planned_commit=release['commit'], planned_image=release['image'],
                  evidence_manifest=f"runs/remote/acceptance/{release['commit'][:7]}-cc/manifest.json",
                  realworld_inputs=inputs, inputs=[f'{inputs}/{f}' for f in INPUT_NAMES[:3]],
                  realworld_gate_sha256=json_sha(report), realworld_manifest_sha256=report['manifest']['sha256'],
                  registration_tool=TOOL)
    specs = [
        ('stats', 'scripts/compute_workspace_stats.py',
         ['--regime', 'realworld', '--path', f'{inputs}/store.yaml', '--split', f'{inputs}/split_seed0.json',
          '--out', f'runs/{stats}/workspace_stats.json'], [], ['workspace_stats.json']),
        ('pre', 'scripts/train.py',
         ['--config', f'configs/realworld/ours/{task}.yaml', '--name', pre, '--output-root', 'runs',
          '--resume', 'auto', '--set', f'data.store={inputs}/store.yaml',
          f'data.split_manifest={inputs}/split_seed0.json', f'data.gate_report={inputs}/gate_report.json',
          f'model.anchor_stats=runs/{stats}/workspace_stats.json'], [stats],
         ['completion.json', 'metrics.jsonl', 'checkpoints/step_0200000.pt', 'config.yaml']),
        ('export', 'scripts/export_encoder.py',
         ['--run', f'runs/{pre}', '--checkpoint', f'runs/{pre}/checkpoints/step_0200000.pt',
          '--out', f'runs/{export}/encoder.pt'], [pre], ['encoder.pt'])]
    return [dict(copy.deepcopy(common), **FOOTPRINT[k], id=ids[k], script=script, args=args, deps=deps,
                 required_results=results, max_restarts=2 if k == 'pre' else 1)
            for k, script, args, deps, results in specs]


def held_specs(task, inputs, current):
    dataset = f'rwcal-20261111-fqldata-ours-{task}'
    export = f'rwcal-20261111-ours-{task}-export'
    common = dict(registrable=False, held_by=HELD_BY, realworld_target=task, fql_condition='ours')
    specs = [dict(common, id=dataset, script='scripts/realworld_fql_dataset.py', deps=[export], args=[
        'build', '--store', f'{inputs}/store.yaml', '--condition', 'ours', '--out', f'runs/{dataset}/dataset.h5',
        '--gate-report', f'{inputs}/gate_report.json', '--outcomes', f'{inputs}/outcomes.json',
        '--export', f'runs/{export}/encoder.pt', '--device', 'cuda'])]
    by_id = {r['id']: r for r in current['jobs']}
    prefix = 'rwcal-20261109-fql-sincro-'
    source = task if any(f'{prefix}{task}-s{s}' in by_id for s in range(3)) else 'drawer-open'
    templates = [by_id[f'{prefix}{source}-s{s}'] for s in range(3) if f'{prefix}{source}-s{s}' in by_id]
    require(templates, 'missing SinCro FQL queue template')
    for seed in range(3):
        template = by_id.get(f'{prefix}{source}-s{seed}', templates[0])
        args = template.get('args')
        require(isinstance(args, list) and all(isinstance(a, str) for a in args), 'invalid FQL template args')
        require(template.get('script') == 'scripts/train_realworld_fql.py' and args.count('--name') == 1,
                'invalid FQL template script/name')
        job = f'rwcal-20261111-fql-ours-{task}-s{seed}'
        args = [a.replace(source, task) for a in args]
        index = args.index('--name') + 1
        require(index < len(args), 'missing FQL template name')
        args[index] = job
        changes = {'condition': 'ours', 'data.dataset': f'runs/{dataset}/dataset.h5', 'train.seed': str(seed),
                   'data.split_manifest': f'{inputs}/split_seed0.json', 'data.gate_report': f'{inputs}/gate_report.json'}
        for key, value in changes.items():
            indices = [i for i, a in enumerate(args) if a.partition('=')[0] == key]
            require(len(indices) == 1, f'FQL template needs exactly one {key}')
            args[indices[0]] = f'{key}={value}'
        specs.append(dict(common, id=job, script=template['script'], args=args, deps=[dataset],
                          fql_seed=seed, derived_from=template['id']))
    return specs


def queue_document(raw):
    doc = yaml.safe_load(raw)
    require(isinstance(doc, dict) and isinstance(doc.get('jobs'), list), 'queue jobs must be a sequence')
    require(all(isinstance(r, dict) and isinstance(r.get('id'), str) for r in doc['jobs']), 'invalid queue row')
    ids = [r['id'] for r in doc['jobs']]
    require(len(ids) == len(set(ids)), 'duplicate job ids in queue')
    return doc


def queue_candidate(before, rows):
    """fqldiag's pure text append and parsed-equality proof."""
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


def prepare(repo, tasks, manifest_path, inputs, copies, release, config_root=None):
    config_root = Path(config_root or repo)  # dry run before the release: the source checkout's generated configs
    require(re.fullmatch(r'[0-9a-f]{40}', release['commit']), '--commit needs a full 40-hex sha')
    require(re.fullmatch(r'sha256:[0-9a-f]{64}', release['image']), '--image needs sha256:<64 hex>')
    setup = repo / 'runs/setup'
    # one journal at a time; proofs are per published task set (proof_path), so covering / pick-and-place can be
    # registered later; duplicates are refused by the queue checks below (ids, one ours pretraining row per task)
    require(not (setup / f'{STEM}.pending.json').exists(),
            'pending transaction/proof exists; inspect before retrying')
    before = (repo / QUEUE).read_bytes()
    registry = (repo / REGISTRY).read_bytes() if (repo / REGISTRY).exists() else b''
    current = queue_document(before)
    known = {r['id'] for r in current['jobs']} | registry_ids(registry)
    admitted, refused, rows, held = {}, {}, [], []
    for task in tasks:
        try:
            remote, directory = inputs.get(task), copies.get(task)
            require(remote and directory, 'inputs not staged')
            path = Path(remote)
            require(path.is_absolute() and '..' not in path.parts and path.name == task,
                    'inputs must be an absolute remote directory ending in the task name')
            remote = str(path)
            raw = manifest_path.read_bytes()
            report = gate.evaluate_file(manifest_path, [task])
            if not gate.task_passed(report, task, 'ours'):
                refused[task] = gate.task_problems(report, task, 'ours') or ['gate refused consumer ours']
                continue
            require(report['manifest']['sha256'] == sha(raw), 'manifest changed during evaluation')
            hashes = verify_inputs(task, json.loads(raw), report, directory)
            config = config_root / f'configs/realworld/ours/{task}.yaml'
            cfg_raw = config.read_bytes()
            require(cfg_raw == ours.render(task).encode(), 'config differs from ours.render(task)')
            require(yaml.safe_load(cfg_raw)['data']['consumer'] == 'ours', 'data.consumer must be ours')
            proposed = task_rows(task, remote, report, release)
            for row in proposed:
                require(row['id'] not in known, f"proposed ID already exists in queue or registry: {row['id']}")
                run = repo / 'runs' / row['id']
                require(not run.exists() and not run.is_symlink(), f'run path already exists: {run}')
            require(not any(r.get('realworld_target') == task and r.get('realworld_consumer') == 'ours'
                            and r.get('role') == 'realworld_pretrain' for r in current['jobs']),
                    'ours realworld_pretrain row already exists for task')
            specs = held_specs(task, remote, current)
            admitted[task] = dict(reasons=[], target=task_target(report, task), gate_evaluation=report,
                                  config_sha256=sha(cfg_raw), learning_inputs={'remote': remote,
                                  'copy': str(Path(directory).resolve()), 'sha256': hashes},
                                  ids=[r['id'] for r in proposed])
            rows.extend(proposed)
            held.extend(specs)
        except ERRORS as exc:
            refused[task] = [str(exc)]
    candidate = queue_candidate(before, rows) if rows else before
    return dict(schema='rw-registration/v1', status='dry-run', registered=False, prepared_at=now(),
                repo=str(repo.resolve()), release=release, admitted=admitted, refused=refused, rows=rows,
                rows_sha256=json_sha(rows), held_spec=held, tool_sha256=sha(Path(__file__).read_bytes()),
                counts=dict(requested_tasks=len(tasks), admitted_tasks=len(admitted), refused_tasks=len(refused),
                            pretraining_rows=len(rows), held_fql_specs=len(held)),
                registry_exists_at_evaluation=(repo / REGISTRY).exists(), registry_sha256_at_evaluation=sha(registry),
                queue_sha256_before=sha(before), queue_sha256_after=sha(candidate),
                queue_diff=dict(before_rows=len(current['jobs']), after_rows=len(current['jobs']) + len(rows),
                                added_rows=len(rows), modified_rows=0, removed_rows=0, added_ids=[r['id'] for r in rows]))


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


def proof_path(setup, record):
    return setup / f"{STEM}-{'+'.join(sorted(record['admitted']))}.json"


def execute(repo, record):
    execution_allowed(repo, record['release'])
    require(record['rows'], 'no admitted tasks')
    setup = repo / 'runs/setup'
    pending, proof = setup / f'{STEM}.pending.json', proof_path(setup, record)
    candidate_path = setup / f'{STEM}.queue-candidate.yaml'
    with locks(repo):
        require(not pending.exists() and not proof.exists(), 'pending transaction/proof exists; inspect before retrying')
        require(record['repo'] == str(repo.resolve()) and json_sha(record['rows']) == record['rows_sha256'],
                'registration plan changed')
        check_publication(repo, record)
        candidate = queue_candidate((repo / QUEUE).read_bytes(), record['rows'])
        require(sha(candidate) == record['queue_sha256_after'], 'staged plan changed')
        atomic_write(candidate_path, candidate)
        journal = dict(record, status='pending', proof_file=proof.name)
        atomic_write(pending, json_bytes(journal))
        # Recheck evidence after the durable journal, immediately before replacing the queue.
        check_publication(repo, record)
        os.replace(candidate_path, repo / QUEUE)
        sync_directory((repo / QUEUE).parent)
        record.update(status='published', registered=True, registered_at=now(), proof_file=proof.name)
        atomic_write(proof, json_bytes(record))
        pending.unlink()
        sync_directory(setup)
    return record


def check_publication(repo, record):
    execution_allowed(repo, record['release'])
    require(sha((repo / QUEUE).read_bytes()) == record['queue_sha256_before'], 'queue changed under locks')
    registry = repo / REGISTRY
    require(registry.exists() == record['registry_exists_at_evaluation'] and
            sha(registry.read_bytes() if registry.exists() else b'') == record['registry_sha256_at_evaluation'],
            'registry changed under locks')
    require(sha(Path(__file__).read_bytes()) == record['tool_sha256'], 'tool changed')
    for task, evidence in record['admitted'].items():
        report = gate.evaluate_file(Path(evidence['gate_evaluation']['manifest']['path']), [task])
        stable = lambda r: {k: v for k, v in r.items() if k != 'checked_at_unix'}
        require(stable(report) == stable(evidence['gate_evaluation']), f'{task}: live gate changed')
        config = (repo / f'configs/realworld/ours/{task}.yaml').read_bytes()
        require(sha(config) == evidence['config_sha256'] and config == ours.render(task).encode(),
                f'{task}: config changed')
        for name, digest in evidence['learning_inputs']['sha256'].items():
            require(sha((Path(evidence['learning_inputs']['copy']) / name).read_bytes()) == digest, 'inputs changed')
        for job in evidence['ids']:
            run = repo / 'runs' / job
            require(not run.exists() and not run.is_symlink(), f'run path already exists: {run}')


def assignments(values):
    out = {}
    for value in values:
        task, sep, path = value.partition('=')
        require(sep and task in TASKS and path and task not in out, 'expected unique TASK=DIR assignments')
        out[task] = path
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=MAIN)
    parser.add_argument('--tasks', nargs='+', choices=TASKS, default=list(TASKS))
    parser.add_argument('--inputs', action='append', default=[], metavar='TASK=REMOTE_DIR')
    parser.add_argument('--inputs-copy', action='append', default=[], metavar='TASK=LOCAL_DIR')
    parser.add_argument('--manifest', type=Path, default=gate.MANIFEST)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--config-root', type=Path, default=None,
                        help='dry run only: checkout holding configs/realworld/ours (default --repo)')
    args = parser.parse_args(argv)
    release = dict(commit=args.commit, image=args.image)
    if args.execute:
        execution_allowed(args.repo, release)
    require(not (args.out and args.execute), '--out is only for dry-run proofs')
    require(not (args.config_root and args.execute), '--config-root is only for dry runs')
    require(len(set(args.tasks)) == len(args.tasks), 'duplicate requested tasks')
    inputs, copies = {**DEFAULT_INPUTS, **assignments(args.inputs)}, assignments(args.inputs_copy)
    record = prepare(args.repo, args.tasks, args.manifest, inputs, copies, release, args.config_root)
    if args.execute:
        record = execute(args.repo, record)
    if args.out:
        path = args.out.resolve()
        require(not any(p.name == 'setup' and p.parent.name == 'runs' for p in path.parents)
                and not (path.parent.name == 'experiments' and path.name in ('queue.yaml', 'registry.jsonl')),
                '--out must not overwrite operational files')
        require(not path.exists(), '--out already exists')
        atomic_write(path, json_bytes(record))
    print(json_bytes(record).decode(), end='')
    return 0 if record['admitted'] else 2


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except ERRORS as exc:
        print(f'REFUSED: {exc}', file=sys.stderr)
        raise SystemExit(2) from exc
