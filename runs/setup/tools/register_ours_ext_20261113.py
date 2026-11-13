"""Append the 15 Ours length-continuation jobs. Dry-run default; coordinator-only publication under both locks.

No driver, scheduler, launch API, network, or registry mutations. Copy to the released main checkout before --execute.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
SOURCE = next(p for p in Path(__file__).resolve().parents if (p / 's4d').is_dir())
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import yaml
from register_fql_nq_20261111 import (atomic_write, json_bytes, json_sha, locks, now, queue_candidate,
                                     queue_document, registry_ids, require, sha, sync_directory)
from s4d.train.continuation import file_sha
from s4d.config import load_config

MAIN = Path('/home/ws/ws/splatter4d')
TOOL = 'runs/setup/tools/register_ours_ext_20261113.py'
STEM = 'rw-registration-ours-ext-20261113'
OLD = 'rwcal-20261111-ours-drawer-open-pre'
STATS = 'rwcal-20261111-ours-drawer-open-stats'
BASE = 'configs/realworld/ours_ext/baseline200k.yaml'
OVERLAY = 'configs/realworld/ours_ext/continuation.yaml'
PREFIX = 'ourx-20261113-ours-drawer-open-'


def input_overrides(cfg):
    return [f'{section}.{key}={cfg[section][key]}' for section, key in
            [('data', 'store'), ('data', 'split_manifest'), ('data', 'gate_report'), ('model', 'anchor_stats')]]


def build_rows(current, release, checkpoint_sha, baseline_sha, cfg):
    by_id = {r['id']: r for r in current['jobs']}
    pre_priority = min(r.get('priority', 5) for r in current['jobs']) - 1
    fql_priority = min(r.get('priority', 5) for r in current['jobs']
                       if r.get('role') == 'realworld_fql' or 'fql' in r['id'].lower()
                       or any('fql' in str(a).lower() for a in r.get('args', []))) - 1
    pre_priority = min(pre_priority, fql_priority - 1)
    rows = []

    def derive(source_id, job, script=None, args=None, deps=None, priority=None, results=None):
        source = by_id[source_id]
        row = copy.deepcopy(source)
        for key in ('pin_host', 'placement', 'local_placement_pause', 'release_source', 'supersedes', 'plan_row_id',
                    'protocol_template', 'purpose'):
            row.pop(key, None)
        row.update(id=job, derived_from=source_id, host='remote', gpu='any', gpu_profile='1g.24gb',
                   cpu_threads=8, gpu_mem_gb=row['mem_gb'], planned_commit=release['commit'],
                   planned_image=release['image'], evidence_manifest=release['evidence_manifest'],
                   registration_tool=TOOL, priority=priority if priority is not None else fql_priority)
        if script is not None:
            row['script'] = script
        if args is not None:
            row['args'] = args
        if deps is not None:
            row['deps'] = deps
        if results is not None:
            row['required_results'] = results
        rows.append(row)
        return row

    for end, start in ((300000, 200000), (400000, 300000)):
        label = f'{end // 1000}k'
        pre, export = PREFIX + 'pre' + label, PREFIX + 'export' + label
        source = OLD if start == 200000 else PREFIX + 'pre300k'
        args = ['--name', pre, '--output-root', 'runs', '--source-checkpoint',
                f'runs/{source}/checkpoints/step_{start:07d}.pt', '--baseline-config', BASE,
                '--baseline-sha256', baseline_sha, '--start-step', str(start), '--stop-step', str(end)]
        if start == 200000:
            args += ['--source-sha256', checkpoint_sha]
        args += ['--set', *input_overrides(cfg)]
        derive(OLD, pre, 'scripts/train_ours_continuation.py', args, [source, STATS], pre_priority,
               ['completion.json', 'continuation.json', 'metrics.jsonl', 'resume_source.json', f'checkpoints/step_{end:07d}.pt'])
        derive('rwcal-20261111-ours-drawer-open-export', export, args=[
            '--run', f'runs/{pre}', '--checkpoint', f'runs/{pre}/checkpoints/step_{end:07d}.pt',
            '--out', f'runs/{export}/encoder.pt'], deps=[pre], results=['encoder.pt'])
        old_data = 'rwcal-20261111-fqldata-ours-drawer-open'
        dataset = f'ourx-20261113-fqldata{label}-ours-drawer-open'
        data_args = [x.replace(old_data, dataset).replace('rwcal-20261111-ours-drawer-open-export', export)
                     for x in by_id[old_data]['args']]
        derive(old_data, dataset, args=data_args, deps=[export])
        for seed in range(3):
            source_fql = f'rwnq-20261111-fql-ours-drawer-open-s{seed}'
            job = f'ourx-20261113-fql{label}-ours-drawer-open-s{seed}'
            old_args = by_id[source_fql]['args']
            required = {'agent.alpha=10.0', 'agent.normalize_q_loss=true', 'train.offline_steps=1000000',
                        f'train.seed={seed}'}
            require(required <= set(old_args), 'source FQL hyperparameters differ from a10nq / seed / 1M')
            args = [x.replace(source_fql, job).replace(old_data, dataset) for x in old_args]
            derive(source_fql, job, args=args, deps=[dataset])
    for step in (200000, 300000, 400000):
        source = OLD if step == 200000 else PREFIX + f'pre{step // 1000}k'
        job = PREFIX + f'eval{step // 1000}k'
        row = derive(OLD, job, 'scripts/evaluate_ours_length.py', [
            '--name', job, '--output-root', 'runs', '--checkpoint', f'runs/{source}/checkpoints/step_{step:07d}.pt',
            '--config', BASE, '--set', *input_overrides(cfg)], [source, STATS], fql_priority,
            ['evaluation.json', 'grasp_sites.json', 'completion.json'])
        row.update(max_restarts=1, disk_gb=2)
    return rows, {'pretraining': pre_priority, 'ours_fql': fql_priority,
                  'semantics': 'smaller numbers run first; recomputed from the publication queue'}


def prepare(repo, source_root, commit, image):
    require(re.fullmatch(r'[0-9a-f]{40}', commit or ''), 'full --commit required')
    require(re.fullmatch(r'sha256:[0-9a-f]{64}', image or ''), '--image must be a sha256 digest')
    release = dict(commit=commit, image=image, evidence_manifest=f'runs/remote/acceptance/{commit[:7]}-cc/manifest.json')
    before = (repo / 'experiments/queue.yaml').read_bytes()
    registry = (repo / 'experiments/registry.jsonl').read_bytes()
    current = queue_document(before)
    original = repo / 'runs' / OLD
    cfg = yaml.safe_load((original / 'config.yaml').read_bytes())
    baseline = yaml.safe_load((source_root / BASE).read_bytes())
    require({k: v for k, v in cfg.items() if k != 'run'} == baseline, 'baseline snapshot differs from the actual 200k run')
    require(cfg['train']['steps'] == 200000 and cfg['wandb']['project'] == 'splatter4d-realworld', 'wrong baseline')
    ckpt = original / 'checkpoints/step_0200000.pt'
    digest = file_sha(ckpt)
    rows, priorities = build_rows(current, release, digest, file_sha(source_root / BASE), baseline)
    known = {r['id'] for r in current['jobs']} | registry_ids(registry)
    require(not known.intersection(r['id'] for r in rows), 'proposed IDs already exist')
    require(all(not (repo / 'runs' / r['id']).exists() and not (repo / 'runs' / r['id']).is_symlink() for r in rows),
            'a proposed run directory already exists')
    require(not (repo / f'runs/setup/{STEM}.pending.json').exists(), 'pending journal requires inspection')
    fql_evidence = {}
    by_id = {r['id']: r for r in current['jobs']}
    for seed in range(3):
        job = f'rwnq-20261111-fql-ours-drawer-open-s{seed}'
        args = by_id[job]['args']
        config_path = source_root / args[args.index('--config') + 1]
        resolved = load_config([config_path], args[args.index('--set') + 1:])
        saved_path = repo / 'runs' / job / 'config.yaml'
        saved = yaml.safe_load(saved_path.read_bytes())
        saved.pop('run', None)
        require(resolved == saved, f'current FQL base/argv differs from saved science: {job}')
        fql_evidence[str(config_path)] = file_sha(config_path)
        fql_evidence[str(saved_path)] = file_sha(saved_path)
    candidate = queue_candidate(before, rows)
    inputs = {str(original / 'config.yaml'): file_sha(original / 'config.yaml'), str(ckpt): digest,
              **{str(source_root / path): file_sha(source_root / path) for path in (BASE, OVERLAY, TOOL)}}
    inputs.update(fql_evidence)
    return dict(schema='rw-registration/v1', status='dry-run', registered=False, prepared_at=now(),
                repo=str(repo.resolve()), source_root=str(source_root.resolve()), release=release, rows=rows,
                rows_sha256=json_sha(rows), priorities=priorities, input_sha256=inputs,
                queue_sha256_before=sha(before), queue_sha256_after=sha(candidate), registry_sha256=sha(registry),
                queue_diff=dict(added_rows=len(rows), modified_rows=0, removed_rows=0, added_ids=[r['id'] for r in rows]))


def execute(repo, record):
    require(repo.resolve() == MAIN and Path(__file__).resolve() == MAIN / TOOL,
            'only the coordinator may publish from the released main-checkout copy')
    require(Path(record['source_root']).resolve() == MAIN, 'execute must use main sources')
    setup = repo / 'runs/setup'
    pending, proof = setup / f'{STEM}.pending.json', setup / f'{STEM}.json'
    candidate_path = setup / f'{STEM}.queue-candidate.yaml'
    with locks(repo):
        require(not pending.exists() and not proof.exists(), 'existing journal/proof requires inspection')
        role = json.loads((setup / 'role-release.json').read_bytes())
        require(all(role.get(k) == record['release'][k] for k in ('commit', 'image')), 'accepted role release differs')
        require((repo / record['release']['evidence_manifest']).is_file(), 'missing accepted release evidence')
        def check():
            require(file_sha(repo / 'experiments/queue.yaml') == record['queue_sha256_before'], 'queue changed')
            require(file_sha(repo / 'experiments/registry.jsonl') == record['registry_sha256'], 'registry changed')
            require(json_sha(record['rows']) == record['rows_sha256'], 'plan changed')
            require(all(file_sha(p) == digest for p, digest in record['input_sha256'].items()), 'source/config changed')
            require(all(not (repo / 'runs' / r['id']).exists() and not (repo / 'runs' / r['id']).is_symlink()
                        for r in record['rows']), 'run directory appeared')
        from ops.jobs.realworld_local import placement_scope
        for row in record['rows']:
            placement_scope(row, root=repo)  # accepted release must contain the exact executable/import/config scope
        check()
        candidate = queue_candidate((repo / 'experiments/queue.yaml').read_bytes(), record['rows'])
        require(sha(candidate) == record['queue_sha256_after'], 'candidate changed')
        atomic_write(candidate_path, candidate)
        atomic_write(pending, json_bytes(dict(record, status='pending')))
        check()
        import os
        os.replace(candidate_path, repo / 'experiments/queue.yaml')
        sync_directory(repo / 'experiments')
        record.update(status='published', registered=True, registered_at=now())
        atomic_write(proof, json_bytes(record))
        pending.unlink()
        sync_directory(setup)
    return record


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=MAIN)
    p.add_argument('--source-root', type=Path, default=SOURCE)
    p.add_argument('--commit', required=True)
    p.add_argument('--image', required=True)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--summary', action='store_true')
    a = p.parse_args(argv)
    record = prepare(a.repo, a.source_root, a.commit, a.image)
    if a.execute:
        record = execute(a.repo, record)
    print(json.dumps({k: v for k, v in record.items() if k != 'rows'} if a.summary else record, indent=2))
    return 0


if __name__ == '__main__':
    main()
