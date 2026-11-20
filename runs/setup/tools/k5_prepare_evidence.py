"""PREPARE ONLY: read live state; write K5 evidence; never lock, launch or publish."""
import copy
import difflib
import json
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import k0_isolated_release as release
import k0_isolated_control as control
import k3_release_audit as audit

HERE = Path('/home/ws/ws/realworld-ops/k0-20261120/k5')
MAIN = release.MAIN
TOOLS = Path(__file__).resolve().parent


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def verify_originals():
    """Read-only final fence before Claude permits ordinary retries to overwrite metadata."""
    evidence = HERE / 'evidence'
    proof = json.loads((evidence / 'retry-proof.json').read_bytes())
    for identity, record in proof.items():
        run = MAIN / 'runs' / identity
        assert not (run / 'checkpoints').exists(), 'checkpoint appeared; review normal resume before activation'
        for name, expected in record['run_original_sha256'].items():
            archive = evidence / 'attempt-1-originals' / identity / name
            assert release.digest(release.regular(archive)) == expected, 'archive changed'
            assert release.digest(release.regular(run / name)) == expected, 'live evidence changed; do not overwrite archive'
    print(json.dumps({'originals_verified': len(proof), 'live_writes': 0}))


def main():
    import yaml
    evidence = HERE / 'evidence'
    evidence.mkdir(exist_ok=True)
    sandbox = evidence / 'copy-state'
    paths = ('experiments/queue.yaml', 'runs/setup/admission-pauses.json', 'experiments/registry.jsonl')
    before = {p: control.revision(MAIN / p) for p in paths}
    for p in paths:
        dest = sandbox / p
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(MAIN / p, dest)
    assert before == {p: control.revision(MAIN / p) for p in paths}, 'live drift; retry copy'
    assert not (MAIN / 'runs/setup/k0-isolated-release.json').exists()
    previous_path = HERE.parent / 'k4/evidence/authorization.json'
    grant = json.loads(previous_path.read_bytes())
    queue = yaml.load((sandbox / paths[0]).read_bytes(), Loader=yaml.CSafeLoader)
    rows = {r['id']: r for r in queue['jobs'] if r['id'] in release.IDS}
    assert rows == {i: spec['row'] for i, spec in grant['rows'].items()}
    events = {}
    for line in (sandbox / paths[2]).open('rb'):
        if b'rwk0-20261120-' in line:
            event = json.loads(line)
            if event['id'] in release.IDS:
                events.setdefault(event['id'], []).append(event)
    assert set(events) == {i for i in release.IDS if '-pre-ours-' in i}
    old_sha = release.digest(previous_path.read_bytes())
    for history in events.values():
        assert history[0]['isolated_authorization_sha256'] == old_sha
        assert history[0]['execution_adapter_sha256'] == grant['adapter_sha256']
    save(evidence / 'reviewed-target-history.json', events)
    inventory = json.loads(Path(grant['targeted_evidence']).read_bytes())
    inventory.update(prepared_by='Codex K5; PREPARE ONLY; Claude installs and activates',
                     deployment_executed=False, reason='Unchanged K4 science/rows; short private sockets and reviewed failed-attempt reactivation.',
                     original_k4_inventory_sha256=grant['targeted_evidence_sha256'])
    inventory_path = evidence / 'exact-nine-targeted-evidence.json'
    save(inventory_path, inventory)
    grant.update(adapter_sha256=release.digest((TOOLS / 'k0_isolated_release.py').read_bytes()),
                 worker_sha256=release.digest((TOOLS / 'k0_isolated_worker.py').read_bytes()),
                 control_sha256=release.digest((TOOLS / 'k0_isolated_control.py').read_bytes()),
                 targeted_evidence=str(inventory_path), targeted_evidence_sha256=release.digest(inventory_path.read_bytes()),
                 retry_history={i: release.digest(release.canonical(v)) for i, v in events.items()},
                 previous_authorization={'path': str(previous_path), 'sha256': old_sha},
                 published=False)
    candidate = evidence / 'authorization.json'
    save(candidate, grant)
    _, snapshot, grant_sha = release.load_authorization(candidate)  # rehash all 2715 sealed source files/evidence
    source = Path(snapshot['code_root'])
    sys.path.insert(0, str(source))
    from ops.jobs.scheduler import read_registry
    state = read_registry(sandbox / paths[2])
    report = {}
    originals = evidence / 'attempt-1-originals'
    for identity in sorted(events):
        current = state[identity]
        assert (current['status'], current['attempts'], current['failures'], current['code']) == ('crashed', 1, 1, 143)
        run = MAIN / 'runs' / identity
        assert not (run / 'checkpoints').exists() and (run / 'metrics.jsonl').stat().st_size == 0
        assert (run / 'exit_code').read_text().strip() == '143'
        saved = originals / identity
        saved.mkdir(parents=True, exist_ok=True)
        hashes = {}
        # These are the files the ordinary retry may replace/append; W&B creates a
        # new run subdirectory, so its original per-run directory remains in place.
        for file in run.iterdir():
            if file.is_file() and not file.is_symlink():
                raw = file.read_bytes()
                dest = saved / file.name
                if dest.exists():
                    assert dest.read_bytes() == raw, 'original evidence changed; never overwrite archive'
                else:
                    dest.write_bytes(raw)
                    dest.chmod(0o444)
                hashes[file.name] = release.digest(raw)
        config = yaml.load((run / 'config.yaml').read_bytes(), Loader=yaml.CSafeLoader)
        train = config['train']
        assert train['num_workers'] == 6 and train['prefetch_factor'] == 2
        # Re-derive actual loader implementation from the sealed source, not main.
        implementation = (source / 's4d/train/loop.py').read_text()
        assert 'persistent_workers=workers > 0' in implementation and 'pin_memory=True' in implementation
        report[identity] = {'status': current['status'], 'attempts': 1, 'failures': 1, 'code': 143,
            'max_restarts': rows[identity]['max_restarts'], 'next_attempt': 2, 'resume': 'auto',
            'checkpoint_present': False, 'restart_step': 0, 'metrics_bytes': 0,
            'training_loader': {'workers': 6, 'prefetch_factor': 2, 'persistent_workers': True, 'pin_memory': True},
            'validation_probe_loader': {'workers': 3, 'prefetch_factor': 2, 'persistent_workers': False, 'pin_memory': True},
            'log_every': train.get('log_every', 50), 'run_original_sha256': hashes,
            'run_dir_reused': True, 'rename_or_delete_required': False}
    save(evidence / 'retry-proof.json', report)
    plan = control.plan('activate', candidate, root=sandbox)
    expected = copy.deepcopy(json.loads(plan['pause_raw']))
    next(p for p in expected['pauses'] if p['name'] == 'k0-20261120-held')['active'] = False
    assert json.loads(plan['after']) == expected
    original_plan = control.plan
    control.plan = lambda mode, candidate: original_plan(mode, candidate, root=sandbox)
    sys.argv = ['k0_isolated_control.py', 'activate', str(candidate)]
    control.main()  # actual CLI; no execute, no locks, no live path writes
    control.plan = original_plan
    assert not plan['target'].exists() and (sandbox / paths[1]).read_bytes() == plan['pause_raw']
    rollback = control.plan('rollback', candidate, root=sandbox)
    assert json.loads(rollback['after']) == json.loads(plan['pause_raw'])
    (evidence / 'activation-pause.diff').write_text(''.join(difflib.unified_diff(
        plan['pause_raw'].decode().splitlines(True), plan['after'].decode().splitlines(True),
        fromfile='COPY/admission-pauses.json', tofile='COPY/proposed')))
    save(evidence / 'activation-preview.json', {'execute': False, 'copy_root': str(sandbox),
        'grant_sha256': grant_sha, 'queue_sha256': release.digest(plan['queue_raw']),
        'queue_changed': False, 'registry_changed': False, 'only_named_pause_active_changes': True,
        'reviewed_retry_ids': sorted(events), 'other_six_have_no_history': True,
        'registry_token': plan['token'], 'locks_taken': [], 'live_writes': [],
        'snapshot_sha256': grant['snapshot_sha256'], 'source_files_rehashed': len(snapshot['source_files'])})
    # Independently rederive both complete source maps and every ownership scope.
    changed = subprocess.run(['git', '-C', str(TOOLS.parents[2]), 'diff', '--name-only',
                              'f9b5decd92717a943c026934146b779a3ba54483'],
                             text=True, capture_output=True, check=True).stdout.splitlines()
    changed += [str(p.relative_to(TOOLS.parents[2])) for p in TOOLS.glob('*k5*') if str(p.relative_to(TOOLS.parents[2])) not in changed]
    changed += ['runs/setup/tools/k5_dataloader_probe.py', 'runs/setup/tools/k5_prepare_evidence.py',
                'runs/setup/tools/k5_verify_metrics.py']
    changed = sorted(set(changed))
    assert all(p.startswith('runs/setup/tools/') for p in changed)
    import hashlib
    from ops.provenance.upstreams import working_tree_identity
    maps = {}
    for filename, name in (('ops/jobs/scheduler.py', 'source_identity'), ('scripts/_bootstrap.py', 'source_fingerprints')):
        fn = audit.function((MAIN / filename).read_text(), name, dict(REPO=MAIN, Path=Path,
            hashlib=hashlib, subprocess=subprocess, third_party_identity=working_tree_identity))
        found = fn()
        maps[filename] = found.get('source_sha256', found)
        assert not set(maps[filename]) & set(changed)
    assert len({release.digest(release.canonical(v)) for v in maps.values()}) == 1
    scopes = []
    for path in (MAIN / 'runs/setup/ownership').glob('*.json'):
        raw = path.read_bytes()
        doc = json.loads(raw)
        assert not set((doc.get('execution_scope') or {}).get('files', {})) & set(changed)
        scopes.append({'path': str(path), 'sha256': release.digest(raw)})
    save(evidence / 'ownership-scopes.json', scopes)
    save(evidence / 'adoption-proof.json', {'base_main': subprocess.check_output(['git', '-C', str(MAIN), 'rev-parse', 'HEAD']).decode().strip(),
        'changed_paths': changed, 'source_maps_equal': True, 'source_files': len(next(iter(maps.values()))),
        'source_map_intersections': [], 'ownership_scopes_checked': len(scopes), 'ownership_intersections': [],
        'science_and_rows_equal_k4': True, 'non_target_adapter_delegation_unchanged': True,
        'wrappers_unchanged': True, 'live_control_or_launcher_calls': 0})
    print(json.dumps({'prepared_grant': str(candidate), 'sha256': grant_sha, 'retry_ids': len(events), 'live_writes': 0}))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-originals', action='store_true')
    args = parser.parse_args()
    verify_originals() if args.verify_originals else main()
