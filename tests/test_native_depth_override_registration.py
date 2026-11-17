"""Coordinator-only amendment, untouched placed rows, and strict input alias checks."""
import copy
import fcntl
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from ops.jobs import host_roles, realworld_local as L
from ops.jobs.native_depth_override_registration import OVERRIDE_SCRIPT, TARGET
from s4d.rwlearn.dsreset_entrypoints import ENTRYPOINTS, legacy_row
from tests.test_realworld_local import mirrored, pinned_scope, native_row, localize  # noqa: F401

pytestmark = pytest.mark.local_host_data
ROOT = Path(__file__).resolve().parents[1]
T = None


@pytest.fixture(scope='module', autouse=True)
def load_coordinator_tool():
    # The remote bundle does not ship coordinator tools. Defer this read until
    # a selected local-host-data test runs; remote collection must still work.
    global T
    spec = importlib.util.spec_from_file_location('native_depth_amend_test',
        ROOT/'runs/setup/tools/register_native_depth_override_20261117.py')
    T = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(T)


@pytest.fixture
def publication(tmp_path, monkeypatch):
    root = tmp_path/'repo'
    for name in ('experiments', 'runs/setup/ownership'):
        (root/name).mkdir(parents=True)
    target = dict(id=TARGET, script=T.OLD_SCRIPT, args=['build'], deps=['qa'], required_results=['store-native.yaml'],
        host='remote', pin_host='remote', realworld_target='drawer-open-v2', planned_commit='a'*40,
        planned_image='sha256:'+'b'*64, registration_tool='runs/setup/tools/register_dsreset_20261113.py')
    rows = [target] + [dict(id=f'other-{n}', script='scripts/untouched.py', host='local') for n in range(38)]
    # Include a multibyte comment to exercise YAML character rather than byte offsets.
    (root/'experiments/queue.yaml').write_text('# 기존 identity\n'+yaml.safe_dump({'jobs': rows},sort_keys=False))
    (root/'experiments/registry.jsonl').write_text('{"id":"other-0","event":"launched"}\n')
    (root/'runs/setup/pool-placement.json').write_text(json.dumps({'effective_jobs': {'other-0': rows[1]}}))
    (root/'runs/setup/admission-pauses.json').write_text(json.dumps({'pauses':[
        {'name': T.PAUSE, 'active': True, 'ids': [TARGET]}]}))
    proof = root/'runs/setup/rwv2-20261113-S-test.registration.json'
    proof.write_bytes(T.encoded({'status':'published','registered':True,'variants':['S'],'rows':rows}))
    monkeypatch.setattr(T, 'verify_pin', lambda root, row: None)
    return root, tmp_path/'evidence', proof


def snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}


def test_dry_run_is_read_only_and_amendment_changes_only_script(publication):
    root, evidence, proof = publication
    before = snapshot(root)
    dry = T.repair(root, evidence)
    assert dry['status'] == 'dry-run' and snapshot(root) == before and not evidence.exists()
    result = T.repair(root, evidence, True)
    assert result['status'] == 'applied'
    old = yaml.safe_load(before['experiments/queue.yaml'])
    new = yaml.safe_load((root/'experiments/queue.yaml').read_bytes())
    expected = copy.deepcopy(old)
    expected['jobs'][0]['script'] = OVERRIDE_SCRIPT
    assert new == expected
    assert (root/'experiments/registry.jsonl').read_bytes() == before['experiments/registry.jsonl']
    assert (root/'runs/setup/pool-placement.json').read_bytes() == before['runs/setup/pool-placement.json']
    for backup, digest in result['backups'].items():
        assert T.sha(Path(backup).read_bytes()) == digest
    amended = json.loads(proof.read_bytes())
    assert amended['rows'][0]['script'] == OVERRIDE_SCRIPT
    assert amended['rows_sha256'] == T.sha(json.dumps(amended['rows'],sort_keys=True).encode())
    assert amended['native_depth_entrypoint_amendment']['changed_fields'] == ['script']
    after = snapshot(root)
    assert T.repair(root, evidence, True)['status'] == 'already-applied'
    assert snapshot(root) == after


@pytest.mark.parametrize('block', ['ownership', 'registry', 'output', 'pool', 'hold', 'pin', 'pending'])
def test_refuses_placed_started_or_drifted_target_without_writes(publication, block):
    root, evidence, proof = publication
    if block == 'ownership':
        (root/'runs/setup/ownership'/f'{TARGET}.json').write_text('{"state":"placed"}')
    elif block == 'registry':
        with (root/'experiments/registry.jsonl').open('a') as f:
            f.write(json.dumps({'id': TARGET, 'event': 'launched'})+'\n')
    elif block == 'output':
        (root/'runs'/TARGET).mkdir()
    elif block == 'pool':
        (root/'runs/setup/pool-placement.json').write_text(json.dumps({'effective_jobs': {TARGET: {}}}))
    elif block == 'hold':
        (root/'runs/setup/admission-pauses.json').write_text('{"pauses":[]}')
    elif block == 'pin':
        doc = json.loads(proof.read_bytes());doc['rows'][0]['planned_commit'] = 'c'*40
        proof.write_text(json.dumps(doc))
    else:
        proof.with_name(proof.name.replace('.registration.json','.pending.json')).write_text('{}')
    before = snapshot(root)
    with pytest.raises(ValueError):
        T.repair(root, evidence, True)
    # Locks may be created, but no queue/registry/config/evidence byte changes.
    assert all((root/p).read_bytes() == b for p,b in before.items()) and not evidence.exists()


def test_busy_driver_lock_never_changes_queue(publication):
    root, evidence, _ = publication
    with (root/'runs/setup/stage2-driver.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        before = snapshot(root)
        with pytest.raises(BlockingIOError):
            T.repair(root, evidence, True)
        assert snapshot(root) == before and not evidence.exists()


def test_interrupted_queue_publish_recovers_proof(publication, monkeypatch):
    root, evidence, proof = publication
    original = T.atomic
    def fail_proof(path, raw):
        if path == proof:
            raise OSError('injected proof write interruption')
        original(path, raw)
    monkeypatch.setattr(T,'atomic',fail_proof)
    with pytest.raises(OSError, match='injected'):
        T.repair(root,evidence,True)
    assert (evidence/'pending.json').exists()
    monkeypatch.setattr(T,'atomic',original)
    assert T.repair(root,evidence,True)['status'] == 'applied'
    assert T.repair(root,evidence)['status'] == 'already-applied'


def test_alias_preserves_strict_native_localization(mirrored, monkeypatch, tmp_path):
    f = mirrored
    monkeypatch.setattr(L,'LOCAL_RAW_ROOT',tmp_path)
    job = native_row(f)
    plain_args, plain_record = localize(f,job)
    override = {**job, 'script': OVERRIDE_SCRIPT}
    assert L.has_resolver(override)
    assert L.needed_inputs(override) == L.needed_inputs(job)
    args, record = localize(f,override)
    assert args == plain_args and record == plain_record
    with pytest.raises(L.LocalInputError):
        L.needed_inputs({**override,'args': job['args']+['--raw-root','unverified=path']})
    assert legacy_row(override)['script'] == 'scripts/realworld_native_depth.py'
    assert legacy_row(job) == job
    assert OVERRIDE_SCRIPT not in ENTRYPOINTS


def test_unregistered_row_waits_without_writes(tmp_path):
    (tmp_path/'experiments').mkdir()
    (tmp_path/'experiments/queue.yaml').write_text('jobs: []\n')
    assert T.repair(tmp_path,tmp_path/'evidence')['status'] == 'awaiting-registration'
