"""Evidence must bind the complete release transition; no filename-only or runtime target waivers."""
import ast
import hashlib
import json
from pathlib import Path

import pytest

from ops.jobs import release_repin as rr
from ops.jobs import run_identity as ri

C1, C2, IMG = '1' * 40, '2' * 40, 'sha256:' + 'a' * 64
SCRIPTS = ['scripts/train.py', 'scripts/remote/entry.py']


def digest(value):
    return hashlib.sha256(value).hexdigest()


def fixture_scope(tmp_path):
    records = {name: {'before_sha256': digest(before), 'after_sha256': digest(after)}
               for name, before, after in [('s4d/model.py', b'old', b'new'), ('ops/jobs/scheduler.py', b'base', b'mig')]}
    scope = {'identical': False, 'differences': list(records), 'operational_differences': [],
             'difference_records': records, 'files': {k: v['after_sha256'] for k, v in records.items()}}
    evidence = {'from_commit': C1, 'to_commit': C2, 'scripts': SCRIPTS,
                'differences': {'s4d/model.py': {'change': dict(records['s4d/model.py']), 'justification': 'uncalled helper',
                                                'reviewer_verdict': '/review.md#B'}}}
    path = tmp_path / 'evidence.json'
    path.write_text(json.dumps(evidence))
    return scope, path, evidence


def test_covered_scope_records_full_evidence_and_hash(tmp_path):
    scope, path, _ = fixture_scope(tmp_path)
    covered = rr.reviewed_scope(scope, path, C1, C2, SCRIPTS,
                                lambda n: b'base' if n == 'ops/jobs/scheduler.py' else b'new',
                                lambda n: b'mig' if n == 'ops/jobs/scheduler.py' else b'new')
    event = rr.repin_event('j', {'commit': C1, 'image_digest': IMG}, C2, IMG, covered, SCRIPTS, 'resume')
    assert event['differences'] == scope['difference_records']
    assert event['allowed_differences'] == {'path': str(path), 'sha256': digest(path.read_bytes())}
    assert event['difference_coverage'] == {'s4d/model.py': 'reviewer_evidence',
                                             'ops/jobs/scheduler.py': 'approved_mig_patch'}


@pytest.mark.parametrize('change', ['missing', 'hash', 'verdict', 'target', 'scripts'])
def test_uncovered_or_stale_evidence_refuses(tmp_path, change):
    scope, path, evidence = fixture_scope(tmp_path)
    if change == 'missing':
        evidence['differences'] = {}
    elif change == 'hash':
        evidence['differences']['s4d/model.py']['change']['after_sha256'] = 'f' * 64
    elif change == 'verdict':
        evidence['differences']['s4d/model.py']['reviewer_verdict'] = ' '
    elif change == 'target':
        evidence['to_commit'] = '3' * 40
    else:
        evidence['scripts'] = ['scripts/other.py']
    path.write_text(json.dumps(evidence))
    with pytest.raises(rr.RepinError):
        rr.reviewed_scope(scope, path, C1, C2, SCRIPTS, lambda n: b'base', lambda n: b'mig')


@pytest.mark.parametrize('extra', [{'status': 'running'}, {'unknown': True}, {'status': 'unreachable'},
                                  {'repinned_from': {'commit': C1}}])
def test_stopped_only_and_single_transition(extra):
    state = {'host': 'remote', 'status': 'crashed', 'commit': C1, 'image_digest': IMG, **extra}
    with pytest.raises(rr.RepinError):
        rr.check_state('j', state, C2, IMG)


def test_runtime_only_accepts_recorded_destination_and_same_identity():
    previous = {'host': 'remote', 'git_commit': C1, 'image_digest': IMG, 'data_id': 'd' * 64, 'config_sha256': 'e' * 64}
    identity = {**previous, 'git_commit': C2, 'repinned_from': {'git_commit': C1, 'image_digest': IMG,
                                                            'to_git_commit': C2, 'to_image_digest': IMG}}
    ri._check_previous(previous, identity)
    assert ri._release_history(previous, identity)[0]['to_commit'] == C2
    for bad in ({**identity, 'git_commit': '3' * 40}, {**identity, 'image_digest': 'sha256:' + 'b' * 64},
                {**identity, 'data_id': 'f' * 64}, {**identity, 'config_sha256': 'f' * 64}):
        with pytest.raises(ri.RunIdentityError):
            ri._check_previous(previous, bad)


def test_driver_uses_exact_registry_target_without_importing_entrypoint():
    source = (Path(__file__).parents[1] / 'ops/campaign/remote_campaign_driver.py').read_text()
    fn = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'job_release')
    state = {'j': {'commit': C2, 'image_digest': IMG, 'repinned_from': {'commit': C1},
                   'repinned_target': {'commit': C2, 'image_digest': IMG}}}
    ns = {'state': state, 'SNERL_NT_IDS': (), 'IMAGE': IMG}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), '<job_release>', 'exec'), ns)
    assert ns['job_release']({'id': 'j'}) == (C2, IMG)
    state['j']['commit'] = '3' * 40
    with pytest.raises(ValueError, match='recorded re-pin target'):
        ns['job_release']({'id': 'j'})


def test_full_execution_patch_needs_its_own_approval_binding(tmp_path, monkeypatch):
    from ops.jobs import mig_variant
    monkeypatch.setattr(mig_variant, 'mig_patch', lambda *a: b'container patch')
    monkeypatch.setattr(mig_variant, 'git', lambda *a: b'full patch')
    path = tmp_path / 'approved.json'
    record = {'mig_base': C1, 'mig_head': C2, 'patch_sha256': digest(b'container patch'), 'approved_by': 'reviewer'}
    path.write_text(json.dumps(record))
    assert not rr.approved_patch(tmp_path, C1, C2, path, full_scope=True)['approved']
    record['execution_patch_sha256'] = digest(b'full patch')
    path.write_text(json.dumps(record))
    assert rr.approved_patch(tmp_path, C1, C2, path, full_scope=True)['approved']


def test_cli_plan_is_read_only_and_apply_requires_acceptance_and_approval(tmp_path, monkeypatch, capsys):
    from ops.jobs import mig_variant, remote_backend
    from ops.jobs import scheduler as jobs
    from ops.provenance import upstreams

    job = {'id': 'j', 'script': 'scripts/train.py', 'args': []}
    fingerprint = jobs.job_fingerprint(job)
    registry = tmp_path / 'registry.jsonl'
    events = [{'event': 'launched', 'id': 'j', 'host': 'remote', 'commit': C1, 'image_digest': IMG,
               'job_sha256': fingerprint, 'data_identity': 'd' * 64, 'attempt': 1, 'time': 1},
              {'event': 'exited', 'id': 'j', 'host': 'remote', 'attempt': 1, 'code': 137, 'time': 2}]
    registry.write_text(''.join(json.dumps(e) + '\n' for e in events))
    original = registry.read_bytes()
    monkeypatch.setattr(jobs, 'load_queue', lambda *a: {'jobs': [job]})
    monkeypatch.setattr(mig_variant, 'git', lambda root, *a: a[-1].removesuffix('^{commit}').encode())
    monkeypatch.setattr(rr, 'commit_reader', lambda *a: (lambda n: None, []))
    monkeypatch.setattr(upstreams, 'commit_identity', lambda *a: {})
    approved = {'approved': False}
    monkeypatch.setattr(rr, 'approved_patch', lambda *a, **kw: dict(approved))
    monkeypatch.setattr(remote_backend, 'acceptance_manifest', lambda *a: {'manifest_sha256': 'f' * 64,
                                                                        'data_identity': 'd' * 64})
    evidence = tmp_path / 'evidence.json'
    evidence.write_text(json.dumps({'from_commit': C1, 'to_commit': C2, 'scripts': SCRIPTS, 'differences': {}}))
    args = ['--to-commit', C2, '--to-image', IMG, '--mig-base', C1, '--mig-head', C2,
            '--registry', str(registry), '--allowed-differences', str(evidence), 'j']
    assert rr.main(['plan', *args]) == 3
    plan = json.loads(capsys.readouterr().out)
    assert plan['scope_covered'] and len(plan['apply_blockers']) == 2
    assert registry.read_bytes() == original and not registry.with_suffix('.lock').exists()
    with pytest.raises(rr.RepinError, match='approval pending'):
        rr.main(['apply', '--authorization', 'resume', *args])
    approved['approved'] = True
    with pytest.raises(rr.RepinError, match='acceptance pending'):
        rr.main(['apply', '--authorization', 'resume', *args])
    assert registry.read_bytes() == original
    assert rr.main(['apply', '--authorization', 'resume', '--acceptance-manifest', 'manifest.json', *args]) == 0
    state = jobs.read_registry(registry)['j']
    assert state['commit'] == C2 and state['job_sha256'] == fingerprint and state['data_identity'] == 'd' * 64
    assert state['repinned_target'] == {'commit': C2, 'image_digest': IMG}
    with pytest.raises(rr.RepinError, match='already re-pinned once'):
        rr.main(['apply', '--authorization', 'resume', '--acceptance-manifest', 'manifest.json', *args])
