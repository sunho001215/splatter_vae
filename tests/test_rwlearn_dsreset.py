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
