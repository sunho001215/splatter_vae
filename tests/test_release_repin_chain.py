"""CPU-only chained re-pin continuity, evidence and lock regression tests."""

import copy
import hashlib
import json
import multiprocessing

import pytest

from ops.jobs import release_repin as rr
from ops.jobs import run_identity as ri
from ops.jobs import scheduler as jobs

C0, C1, C2, C3 = (str(n) * 40 for n in range(1, 5))
IMG = 'sha256:' + 'a' * 64
DATA = 'd' * 64
SCRIPTS = ['scripts/train.py', 'scripts/remote/entry.py']


def transition(source, target, attempt=1):
    return {'event': 'release_repinned', 'id': 'j', 'host': 'remote', 'attempt': attempt,
            'from_commit': source, 'from_image_digest': IMG, 'commit': target, 'image_digest': IMG,
            'sources_identical': True, 'authorization': 'user: resume same id', 'time': attempt + 2}


def write_events(path, events):
    path.write_text(''.join(json.dumps(event) + '\n' for event in events))


@pytest.fixture
def chain(tmp_path, monkeypatch):
    from ops.jobs import mig_variant, remote_backend
    from ops.provenance import upstreams

    job = {'id': 'j', 'script': SCRIPTS[0], 'args': []}
    events = [
        {'event': 'launched', 'id': 'j', 'host': 'remote', 'attempt': 1, 'time': 1,
         'commit': C0, 'image_digest': IMG, 'data_identity': DATA, 'job_sha256': jobs.job_fingerprint(job)},
        {'event': 'exited', 'id': 'j', 'attempt': 1, 'code': 143, 'time': 2},
        transition(C0, C1),
        {'event': 'launched', 'id': 'j', 'host': 'remote', 'attempt': 2, 'time': 4,
         'commit': C1, 'image_digest': IMG},
        {'event': 'checkpoint_paused', 'id': 'j', 'host': 'remote', 'attempt': 2, 'time': 5,
         'intentional_stop': True, 'checkpoint_step': 1440000, 'checkpoint': 'runs/j/checkpoints/latest.pt'},
        {'event': 'exited', 'id': 'j', 'attempt': 2, 'code': 143, 'time': 6},
    ]
    registry = tmp_path / 'registry.jsonl'
    write_events(registry, events)
    monkeypatch.setattr(jobs, 'load_queue', lambda *a: {'jobs': [job]})
    monkeypatch.setattr(mig_variant, 'git', lambda root, *a: a[-1].removesuffix('^{commit}').encode())
    trees = {commit: {'scripts/train.py': b'import s4d.model\n', 'scripts/remote/entry.py': b'',
                      's4d/__init__.py': b'', 's4d/model.py': content}
             for commit, content in [(C0, b'original'), (C1, b'old'), (C2, b'new'), (C3, b'other')]}
    monkeypatch.setattr(rr, 'commit_reader', lambda root, commit: (trees[commit].get, list(trees[commit])))
    monkeypatch.setattr(upstreams, 'commit_identity', lambda *a: {})
    approved = {'approved': True}
    monkeypatch.setattr(rr, 'approved_patch', lambda root, base, head, path, **kw:
                        {**approved, 'base': base, 'head': head, 'full_scope': kw['full_scope']})
    acceptance = {'manifest_sha256': 'f' * 64, 'data_identity': DATA}
    monkeypatch.setattr(remote_backend, 'acceptance_manifest', lambda *a: acceptance)

    def args(target=C2):
        evidence = tmp_path / f'{target[0]}-evidence.json'
        change = {f'{side}_sha256': hashlib.sha256(trees[commit]['s4d/model.py']).hexdigest()
                  for side, commit in [('before', C1), ('after', target)]}
        evidence.write_text(json.dumps({'from_commit': C1, 'to_commit': target, 'scripts': SCRIPTS,
                                       'differences': {'s4d/model.py': {'change': change,
                                                       'justification': 'reviewed uncalled helper',
                                                       'reviewer_verdict': 'ACCEPT'}}}))
        return ['--to-commit', target, '--to-image', IMG, '--mig-base', C1, '--mig-head', target,
                '--registry', str(registry), '--allowed-differences', str(evidence),
                '--acceptance-manifest', 'manifest.json', '--authorization', 'user: resume same id', 'j']

    return registry, events, args, approved, acceptance


def test_second_repin_preserves_history_stop_and_identity(chain, capsys):
    registry, events, args, _, _ = chain
    command = args()
    before = registry.read_bytes()
    assert rr.main(['plan', *command]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan['from_commit'] == C1 and plan['to_commit'] == C2
    assert plan['coverage'] == {'s4d/model.py': 'reviewer_evidence'}
    assert registry.read_bytes() == before and not registry.with_suffix('.lock').exists()
    assert rr.main(['apply', *command]) == 0
    state = jobs.read_registry(registry)['j']
    assert [e['commit'] for e in state['release_history']] == [C1, C2]
    assert state['release_history'][0]['authorization'] == events[2]['authorization']
    assert state['release_history'][-1]['mig_patch']['full_scope'] is True
    assert state['repinned_from'] == {'commit': C1, 'image_digest': IMG, 'attempt': 2}
    assert state['repinned_target'] == {'commit': C2, 'image_digest': IMG}
    assert state['data_identity'] == DATA and state['job_sha256'] == events[0]['job_sha256']
    assert state['intentional_stops'][0]['checkpoint_step'] == 1440000
    with pytest.raises(rr.RepinError, match='already on'):
        rr.main(['apply', *command])


@pytest.mark.parametrize('bad', [
    {'from_commit': C0}, {'commit': C0}, {'commit': C1}, {'attempt': 1},
    {'authorization': ' '}, {'image_digest': 'sha256:' + 'b' * 64}, {'host': 'local'},
])
def test_reducer_refuses_chain_gaps_duplicates_and_stale_attempts(chain, bad):
    registry, events, *_ = chain
    write_events(registry, [*events, {**transition(C1, C2, 2), 'repin_chain': [events[2]], **bad}])
    with pytest.raises(ValueError, match='re-pin'):
        jobs.read_registry(registry)


def test_reducer_uses_last_of_three_transitions_and_rejects_competitor(chain):
    registry, events, *_ = chain
    second = {**transition(C1, C2, 2), 'repin_chain': [events[2]]}
    third = {**transition(C2, C3, 2), 'repin_chain': [events[2], transition(C1, C2, 2)]}
    write_events(registry, [*events, second, third])
    state = jobs.read_registry(registry)['j']
    assert state['commit'] == C3 and len(state['release_history']) == 3
    assert state['repinned_from']['commit'] == C2
    write_events(registry, [*events, second, transition(C1, C3, 2)])
    with pytest.raises(ValueError, match='re-pin'):
        jobs.read_registry(registry)


@pytest.mark.parametrize('bad', ['legacy_range', 'no_evidence', 'missing_file', 'stale_hash', 'authorization',
                                 'approval', 'acceptance', 'data', 'fingerprint', 'running', 'unknown', 'image'])
def test_second_repin_keeps_all_evidence_requirements(chain, bad):
    registry, events, args, approved, acceptance = chain
    command = args()
    evidence = command.index('--allowed-differences')
    if bad == 'legacy_range':
        command[command.index('--mig-base') + 1] = C0
    elif bad == 'no_evidence':
        del command[evidence:evidence + 2]
    elif bad in {'missing_file', 'stale_hash'}:
        from pathlib import Path
        path = Path(command[evidence + 1])
        payload = json.loads(path.read_text())
        if bad == 'missing_file':
            payload['differences'] = {}
        else:
            payload['differences']['s4d/model.py']['change']['after_sha256'] = '0' * 64
        path.write_text(json.dumps(payload))
    elif bad == 'authorization':
        command[command.index('--authorization') + 1] = ' '
    elif bad == 'approval':
        approved['approved'] = False
    elif bad == 'acceptance':
        index = command.index('--acceptance-manifest')
        del command[index:index + 2]
    elif bad == 'data':
        acceptance['data_identity'] = '0' * 64
    elif bad == 'image':
        command[command.index('--to-image') + 1] = 'sha256:' + 'b' * 64
    else:
        if bad == 'fingerprint':
            events[0]['job_sha256'] = '0' * 64
        elif bad == 'running':
            events.pop()
        else:
            events.append({'event': 'unknown', 'id': 'j', 'reason': 'unconfirmed', 'time': 7})
        write_events(registry, events)
    original = registry.read_bytes()
    with pytest.raises(rr.RepinError):
        rr.main(['apply', *command])
    assert registry.read_bytes() == original


def test_competing_applies_serialize_and_only_one_appends(chain):
    registry, _, args, *_ = chain
    commands = [args(C2), args(C3)]
    context = multiprocessing.get_context('fork')
    results = context.Queue()
    start = context.Event()

    def apply(command):
        start.wait(10)
        try:
            results.put(rr.main(['apply', *command]))
        except rr.RepinError as exc:
            results.put(str(exc))

    workers = [context.Process(target=apply, args=(command,)) for command in commands]
    try:
        for worker in workers:
            worker.start()
        start.set()
        for worker in workers:
            worker.join(15)
            assert worker.exitcode == 0
        outcomes = [results.get(timeout=5) for _ in workers]
        assert outcomes.count(0) == 1
        assert any(isinstance(value, str) and 'range' in value for value in outcomes)
        state = jobs.read_registry(registry)['j']
        assert len(state['release_history']) == 2 and state['commit'] in {C2, C3}
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(5)


def runtime_identity(source, target, attempt):
    return {'host': 'remote', 'campaign_eligible': True, 'git_commit': target, 'image_digest': IMG,
            'data_id': DATA, 'config_sha256': 'e' * 64, 'job_id': 'j', 'attempt': attempt,
            'repinned_from': {'git_commit': source, 'image_digest': IMG,
                              'to_git_commit': target, 'to_image_digest': IMG}}


def runtime_events(count):
    return [{**transition(a, b, i), 'job_sha256': 'f' * 64}
            for i, (a, b) in enumerate([(C0, C1), (C1, C2), (C2, C3)][:count], 1)]


def test_immutable_runtime_accepts_prepared_chain_and_preserves_guard(tmp_path):
    from ops.containers.repin_provenance import prepare

    path = tmp_path / 'runs/j/provenance.json'
    path.parent.mkdir(parents=True)
    original = runtime_identity(C0, C0, 1)
    original.pop('repinned_from')
    path.write_text(json.dumps(original))
    # First transition still uses the unmodified runtime's one-repin path.
    ri.start_remote_run(tmp_path, runtime_identity(C0, C1, 2), SCRIPTS[0], ['--resume', 'auto'])
    previous = json.loads(path.read_text())
    assert len(previous['release_history']) == 1
    with pytest.raises(ri.RunIdentityError):
        ri._check_previous(previous, runtime_identity(C1, C2, 3))
    for count, source, target in [(2, C1, C2), (3, C2, C3)]:
        before = path.read_bytes()
        prepare(path, runtime_events(count), DATA, 'f' * 64)
        receipts = list((path.parent / '.release-repins').glob('*.json'))
        assert any(json.loads(p.read_text())['before'].encode() == before for p in receipts)
        identity = runtime_identity(source, target, count + 1)
        ri.start_remote_run(tmp_path, identity, SCRIPTS[0], ['--resume', 'auto'])
        saved = json.loads(path.read_text())
        assert saved['git_commit'] == target and len(saved['release_history']) == count
        prepare(path, runtime_events(count), DATA, 'f' * 64)
        ri.start_remote_run(tmp_path, identity, SCRIPTS[0], ['--resume', 'auto'])
        assert json.loads(path.read_text())['release_history'] == saved['release_history']
    for key in ('host', 'campaign_eligible', 'data_id', 'config_sha256'):
        assert saved[key] == original[key]


@pytest.mark.parametrize('bad', ['gap', 'tail', 'image', 'data', 'job', 'history', 'symlink', 'fingerprint'])
def test_preparation_refuses_invalid_provenance(tmp_path, bad):
    from ops.containers.repin_provenance import prepare

    previous = runtime_identity(C0, C0, 1)
    first = runtime_identity(C0, C1, 2)
    previous = {**first, 'release_history': ri._release_history(previous, first)}
    path = tmp_path / 'provenance.json'
    if bad == 'gap':
        previous['release_history'][0]['from_commit'] = C3
    elif bad == 'tail':
        previous['git_commit'] = C3
    elif bad == 'history':
        previous['release_history'] = {}
    elif bad in {'image', 'data', 'job'}:
        previous[{'image': 'image_digest', 'data': 'data_id', 'job': 'job_id'}[bad]] = 'wrong'
    path.write_text(json.dumps(previous))
    if bad == 'symlink':
        other = tmp_path / 'linked.json'
        other.symlink_to(path)
        path = other
    before = path.read_bytes()
    with pytest.raises(ValueError):
        prepare(path, runtime_events(2), DATA, 'wrong' if bad == 'fingerprint' else 'f' * 64)
    assert path.read_bytes() == before


def test_receipt_recovers_interrupted_replace_and_runtime_still_checks_config(tmp_path, monkeypatch):
    from ops.containers import repin_provenance as prep

    path = tmp_path / 'provenance.json'
    previous = runtime_identity(C0, C0, 1)
    previous.pop('repinned_from')
    path.write_text(json.dumps(previous))
    replace = prep.os.replace
    def interrupted(*a):
        raise OSError('interrupted')
    monkeypatch.setattr(prep.os, 'replace', interrupted)
    with pytest.raises(OSError, match='interrupted'):
        prep.prepare(path, runtime_events(2), DATA, 'f' * 64)
    assert json.loads(path.read_text()) == previous
    monkeypatch.setattr(prep.os, 'replace', replace)
    prep.prepare(path, runtime_events(2), DATA, 'f' * 64)
    saved = json.loads(path.read_text())
    identity = runtime_identity(C1, C2, 3)
    ri._check_previous(saved, identity)
    for key in ('config_sha256', 'data_id', 'image_digest'):
        with pytest.raises(ri.RunIdentityError):
            ri._check_previous(saved, {**identity, key: 'wrong'})


def test_launch_accepts_only_the_latest_contiguous_record(tmp_path):
    from ops.containers.launch import recorded_repin

    registry = tmp_path / 'experiments/registry.jsonl'
    registry.parent.mkdir()
    first = transition(C0, C1)
    second = {**transition(C1, C2, 2), 'repin_chain': [first]}
    spec = {'id': 'j', 'commit': C2, 'image_digest': IMG,
            'repinned_from': {'commit': C1, 'image_digest': IMG}}
    write_events(registry, [first, second])
    assert recorded_repin(tmp_path, spec)
    assert not recorded_repin(tmp_path, {**spec, 'commit': C1,
                                       'repinned_from': {'commit': C0, 'image_digest': IMG}})
    for invalid in [{**second, 'from_commit': C0}, first, {**second, 'authorization': ' '}]:
        write_events(registry, [first, invalid])
        assert not recorded_repin(tmp_path, spec)


def test_state_refuses_corrupt_or_rollback_history(chain):
    registry, *_ = chain
    state = {**jobs.read_registry(registry)['j'], 'release_history': [chain[1][2]]}
    rr.check_state('j', state, C2, IMG)
    with pytest.raises(rr.RepinError, match='duplicate'):
        rr.check_state('j', state, C0, IMG)
    for bad in [None, {}, [state['release_history'][0]] * 2]:
        with pytest.raises(rr.RepinError):
            rr.check_state('j', {**state, 'release_history': copy.deepcopy(bad)}, C2, IMG)


def test_chain_preparation_is_after_admission_and_before_docker_only():
    import shlex
    import subprocess

    from ops.containers.launch import build_remote_command
    from tests.test_remote_launch import job_spec

    spec = job_spec()
    spec.update(commit=C2, image_digest=IMG, max_mem_gb=21.5,
                repinned_from={'commit': C1, 'image_digest': IMG})
    spec['labels'].update({'s4d.commit': C2, 's4d.image_digest': IMG})
    events = [{**event, 'id': spec['id']} for event in runtime_events(2)]
    command = build_remote_command(spec, DATA, C2, 'a' * 64, {}, [], repin_history=events)
    assert subprocess.run(['bash', '-n'], input=command, text=True, capture_output=True).returncode == 0
    preparation = command[command.index('\npython3 -I -c ') + 1:command.index('\ncid=$(docker run')]
    assert command.rindex('existing') < command.index(preparation) < command.index('cid=$(docker run')
    assert command.index('guard_code=$?') < command.index(preparation)
    # The helper is shipped from the coordinator; immutable target source is never invoked or edited.
    source, payload = shlex.split(preparation)[3:]
    assert 'def prepare(' in source
    path, recorded, data, fingerprint, invocation = json.loads(payload)
    assert recorded == events and data == DATA and fingerprint == 'f' * 64
    assert invocation == {'script': spec['script'], 'arguments': spec['args']}
    assert path.endswith('/runs/' + spec['id'] + '/provenance.json')
    legacy = build_remote_command(spec, DATA, C2, 'a' * 64, {}, [])
    assert '.release-repins' not in legacy


def test_legacy_registry_state_has_no_new_fields(chain):
    registry, events, *_ = chain
    state = jobs.read_registry(registry)['j']
    assert 'release_history' not in state
    # A second legacy-shaped event remains invalid; only the explicit chain record opts in.
    write_events(registry, [*events, transition(C1, C2, 2)])
    with pytest.raises(ValueError, match='re-pin'):
        jobs.read_registry(registry)
