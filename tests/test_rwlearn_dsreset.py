"""Paired episode membership and labels are dataset invariants, independent of models."""
import json
from pathlib import Path

import pytest

from s4d.rwlearn.dsreset import paired_manifests, prep_bundle, write_bundle


def fixture_roots(tmp_path):
    roots = {}
    for label in ('success', 'failure'):
        root = tmp_path / label
        root.mkdir()
        for n in range(10):
            (root / f'episode_{n:06d}.h5').write_bytes(f'{label}:{n}'.encode())
        (root / 'episode_999999.h5.partial').write_bytes(b'incomplete')
        (root / 'discarded').mkdir()
        (root / 'discarded/episode_888888.h5').write_bytes(b'discarded')
        Path(str(root) + '.VERIFIED').touch()
        roots[label] = root
    return roots


def test_paired_stratified_deterministic_split(tmp_path):
    roots = fixture_roots(tmp_path)
    m = paired_manifests('covering', roots, inspect=lambda p: {})
    assert m == paired_manifests('covering', roots, inspect=lambda p: {})
    s, sf = m['S'], m['SF']
    for split in ('train', 'validation'):
        assert set(s['split'][split]) == {r['id'] for r in sf['episodes']
                                         if r['split'] == split and r['label'] == 'success'}
    assert not set(s['split']['validation']) & set(sf['split']['train'])
    assert len(sf['split']['validation']) == 2
    assert len(sf['episodes']) == 20
    assert len({r['id'] for r in sf['episodes']}) == 20
    assert sum(sf['outcomes'].values()) == 10
    assert all('discarded' not in r['path'] and r['path'].endswith('.h5') for r in sf['episodes'])


def test_incomplete_transfer_and_duplicate_content_fail_closed(tmp_path):
    roots = fixture_roots(tmp_path)
    with pytest.raises(ValueError, match='expected'):
        paired_manifests('covering', roots, expected={'success': 100, 'failure': 50}, inspect=lambda p: {})
    Path(str(roots['failure']) + '.VERIFIED').unlink()
    with pytest.raises(ValueError, match='verification'):
        paired_manifests('covering', roots, inspect=lambda p: {})
    Path(str(roots['failure']) + '.VERIFIED').touch()
    (roots['failure'] / 'episode_000000.h5').write_bytes(b'success:0')
    with pytest.raises(ValueError, match='duplicate'):
        paired_manifests('covering', roots, inspect=lambda p: {})


def test_legacy_split_reuse_and_prep_never_cross_labels(tmp_path):
    roots = fixture_roots(tmp_path)
    names = [f'episode_{i:06d}.h5' for i in range(10)]
    legacy = {'train': names[2:], 'validation': names[:2]}
    m = paired_manifests('pick-and-place', roots, inspect=lambda p: {}, success_split=legacy)
    assert [r['filename'] for r in m['S']['episodes'] if r['split'] == 'validation'] == names[:2]
    first = m['S']['episodes'][0]
    template = {'adapter': {'config': {}}, 'stages': {'calibrate': {
        'rig_calibration': '/rig/pick-and-place-session.json'}, 'train_data': {},
        'foundation_stereo_depth': {'valid_iters': 32}, 'track_everything': {'window': 16}}}
    reuse = {first['id']: {'raw_sha256': first['sha256'], 'train_data': '/existing/validated'}}
    plan = prep_bundle(m, template, tmp_path / 'processed', reuse=reuse)
    assert plan[0]['reuse'] == '/existing/validated'
    assert all(row['config']['stages']['train_data']['task'] == 'pick-and-place' for row in plan[1:])
    assert len({row['config']['output_root'] for row in plan[1:]}) == 19
    write_bundle(tmp_path / 'bundle', m, plan)
    assert json.loads((tmp_path / 'bundle/SF/outcomes.json').read_text()) == m['SF']['outcomes']
    with pytest.raises(FileExistsError):
        write_bundle(tmp_path / 'bundle', m, plan)
    reuse[first['id']]['raw_sha256'] = 'bad'
    with pytest.raises(ValueError, match='reuse'):
        prep_bundle(m, template, tmp_path, reuse=reuse)


def test_wrong_task_calibration_refused(tmp_path):
    m = paired_manifests('drawer-open-v2', fixture_roots(tmp_path), inspect=lambda p: {})
    template = {'adapter': {'config': {}}, 'stages': {'calibrate': {
        'rig_calibration': '/rig/drawer-open-v1.json'}, 'train_data': {}}}
    with pytest.raises(ValueError, match='rig path'):
        prep_bundle(m, template, tmp_path / 'processed')


def test_outcome_labels_preserve_success_arrays_and_bootstrap_failure(tmp_path):
    import copy
    import h5py
    import numpy as np
    from s4d.rwlearn import actions as A, fql_data as FD
    from s4d.rwlearn.store import MemoryStore
    from tests.test_rwlearn_fql_pipeline import store_on_fixture, FIXTURE
    base = store_on_fixture(n=8, hw=8)
    template = next(iter(base._eps.values()))
    eps = {}
    for label in ('success', 'failure'):
        raw = A.raw_from_arrays(np.load(FIXTURE))
        for c in raw.cameras.values():
            c['t'], c['segment'] = c['t'][:8], c['segment'][:8]
        raw.gripper_command = np.clip(raw.gripper_command + (.01 if label == 'failure' else 0), 0, 1)
        path = tmp_path / f'{label}.npz'
        np.savez(path, **A.raw_to_arrays(raw))
        eps[label] = copy.deepcopy(template)
        eps[label]['raw'] = str(path)
    store = MemoryStore(eps, (8, 8))
    success_only = MemoryStore({'success': eps['success']}, (8, 8))
    old, new, sf = [tmp_path / f'{s}.h5' for s in ('old', 'new', 'sf')]
    FD.build_fql_dataset(success_only, 'e2e_cnn', None, old, episodes=['success'], assume_final_success=True)
    FD.build_fql_dataset(success_only, 'e2e_cnn', None, new, episodes=['success'], outcomes={'success': True})
    FD.build_fql_dataset(store, 'e2e_cnn', None, sf, episodes=['success', 'failure'],
                         outcomes={'success': True, 'failure': False})
    with h5py.File(old) as a, h5py.File(new) as b, h5py.File(sf) as f:
        for cam in a['episodes/success']:
            for key in a[f'episodes/success/{cam}']:
                # Explicit S labels preserve every existing training tensor bit for bit.
                np.testing.assert_array_equal(a[f'episodes/success/{cam}/{key}'][:],
                                              b[f'episodes/success/{cam}/{key}'][:])
            g, h = f[f'episodes/success/{cam}'], f[f'episodes/failure/{cam}']
            for data, successful in ((g, True), (h, False)):
                ok = data['transition_ok'][:]
                final = np.flatnonzero(ok)[-1]
                assert final == 6 and not ok[-1]
                assert data['mask'][final] == (0 if successful else 1)
                assert data['reward'][final] == (0 if successful else -1)
                assert data['terminal'][final]  # legacy chain boundary, not an absorbing mask
                assert bool(data['success'][:].any()) == successful
                assert data['reward'][:][ok][:-1].tolist() == [-1.] * (ok.sum()-1)


def test_tiny_cpu_fql_prefers_success_actions_at_matched_states():
    import torch
    from s4d.rwlearn.fql import FQLAgent
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        agent = FQLAgent.create(0, 2, 1, {'actor_hidden_dims': (16, 16), 'value_hidden_dims': (16, 16),
                                         'batch_size': 32, 'alpha': 10., 'normalize_q_loss': True})
        # Two 2-transition trajectories: same states, different actions/outcomes.
        states = torch.tensor([[0., 0.], [0., 1.], [0., 0.], [0., 1.]])
        actions = torch.tensor([[.7], [.7], [-.7], [-.7]])
        batch = {'observations': states.repeat(8, 1), 'actions': actions.repeat(8, 1),
                 'next_observations': torch.tensor([[0., 1.], [0., 2.], [0., 1.], [0., 2.]]).repeat(8, 1),
                 'rewards': torch.tensor([-1., 0., -1., -1.]).repeat(8),
                 'masks': torch.tensor([1., 0., 1., 1.]).repeat(8)}
        for _ in range(500):
            agent.update(batch)
        with torch.no_grad():
            q = agent.critic(states, actions).mean(0)
        assert torch.all(q[:2] > q[2:] + .02), q
    finally:
        torch.set_num_threads(old_threads)


def test_provisional_prep_is_not_a_training_manifest(tmp_path):
    from s4d.rwlearn.dsreset import available_preprocessing, validate_manifest
    roots = fixture_roots(tmp_path)
    Path(str(roots['failure'])+'.VERIFIED').unlink()
    def qa(path):
        if path.name == 'episode_000001.h5':
            raise ValueError('nonmonotonic camera')
        return {}
    m, rejected = available_preprocessing('drawer-open-v2', roots, inspect=qa)
    assert len(m['SF']['episodes']) == 18 and len(rejected) == 2
    assert not any('partial' in row['path'] or 'discarded' in row['path'] for row in m['SF']['episodes'])
    with pytest.raises(ValueError, match='manifest'):
        validate_manifest(m['SF'])


def test_label_aware_raw_identity_supports_overlapping_counters():
    from s4d.rwlearn.identity import match_recordings, raw_identity
    a = raw_identity('covering','/raw/covering/episode_000001.h5','a'*64)
    b = raw_identity('covering','/raw/covering-failure/episode_000001.h5','b'*64)
    expected = {'s':a,'f':b}
    assert match_recordings(expected,{'success-copy':a,'failure-copy':b}) == {'success-copy':'s','failure-copy':'f'}
    with pytest.raises(RuntimeError,match='duplicate'):
        match_recordings({'s':a,'again':a},{'s':a})
    with pytest.raises(RuntimeError,match='sha256 differs'):
        match_recordings(expected,{'s':dict(a,raw_sha256='c'*64),'f':b})


def test_manifest_gate_keeps_calibration_and_family_checks(tmp_path):
    import copy
    from s4d.rwlearn.dsreset import manifest_gate, validate_manifest
    from tests.test_rwlearn_gate import entry
    m = paired_manifests('covering', fixture_roots(tmp_path), inspect=lambda p: {})['SF']
    ingest = {'episodes':{r['id']:entry('covering',int(r['filename'][8:14]),sha=r['sha256'],validated=True)
                          for r in m['episodes']}}
    report = manifest_gate(m,ingest)
    assert report['tasks']['covering']['target'] == 20
    assert report['calibration_hold']['covering']['passed']
    tampered = copy.deepcopy(m)
    tampered['episodes'][0]['split'] = 'validation' if tampered['episodes'][0]['split']=='train' else 'train'
    with pytest.raises(ValueError,match='cached split'):
        validate_manifest(tampered)
    tampered = copy.deepcopy(m)
    tampered['episodes'][0]['label'] = 'failure'
    with pytest.raises(ValueError,match='directory label'):
        validate_manifest(tampered)
    # Family failure must remain a hard block for the corresponding consumer.
    first = next(iter(ingest['episodes'].values()))
    first['preprocessing']['stages']['qa']['metrics']['families']['geometry']['passed'] = False
    report = manifest_gate(m,ingest)
    from s4d.rwlearn import gate
    assert not gate.task_passed(report,'covering','ours')
    assert gate.task_passed(report,'covering','dynamo')


def test_raw_qa_reports_camera_only_recordings_instead_of_aborting_inventory(tmp_path):
    import h5py
    from s4d.rwlearn.dsreset import inspect_raw
    path=tmp_path/'episode_000077.h5'
    with h5py.File(path,'w') as f:
        f.attrs['status']='complete'
        f.attrs['format']='panda_teleop_episode'
        f.attrs['format_version']=2
        f.create_group('cameras')
        f.create_group('events')
    with pytest.raises(ValueError,match='required action/state schema'):
        inspect_raw(path)


def test_existing_task_amendment_preserves_hash_bound_exclusion_and_paired_split(tmp_path):
    from s4d.rwlearn.dsreset import sha256,validate_manifest
    roots=fixture_roots(tmp_path)
    included={f'covering/{p.name}':sha256(p) for p in roots['success'].glob('*.h5') if p.name!='episode_000001.h5'}
    omitted=roots['success']/'episode_000001.h5'
    # The real amendment's base count remains 100; extend the fixture accordingly.
    for i in range(10,100):
        p=roots['success']/f'episode_{i:06d}.h5';p.write_bytes(f'success:{i}'.encode())
        included[f'covering/{p.name}']=sha256(p)
    amendment={'format':'rw-gate-amendment/1','task':'covering','base_target':100,'target':99,
               'decided_by':'fixture user','decision':'exclude invalid episode, no timestamp repair',
               'episodes':included,'excluded':{'covering/'+omitted.name:{'raw_sha256':sha256(omitted),'reason':'invalid'}}}
    def qa(path):
        assert path != omitted
        return {}
    m=paired_manifests('covering',roots,inspect=qa,amendment=amendment)
    for v in m.values():
        validate_manifest(v)
        assert len(v['excluded'])==1 and v['excluded'][0]['split']=='excluded'
    assert m['S']['episodes']==[ep for ep in m['SF']['episodes'] if ep['label']=='success']
    omitted.write_bytes(b'changed')
    with pytest.raises(ValueError,match='amendment raw sha256'):
        paired_manifests('covering',roots,inspect=qa,amendment=amendment)
