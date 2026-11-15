"""Scientific isolation, S-only graph, immutable local input closure and dry-run safety."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest
from ops.jobs import realworld_local as L
from s4d.rwlearn import rgb_inputs as rgb
from tests.test_dsreset_registration import canonical_manifests
from tests.test_realworld_train_store import built

ROOT = Path(__file__).resolve().parents[1]


def registrar():
    spec = importlib.util.spec_from_file_location('drawer_rgb_reg',ROOT/rgb.TOOL)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_success_graph_preserves_science_and_local_input_closure(tmp_path):
    reg = registrar()
    manifest = canonical_manifests(rgb.TASK)['S']
    rows = reg.build_rows(tmp_path/'S',manifest,dict(commit='a'*40,image='sha256:'+'b'*64,source='local'))
    assert len(rows) == 32 and len({r['id'] for r in rows}) == 32
    for row in rows:
        L.needed_inputs(row)
        assert L.has_resolver(row)
        assert row['realworld_variant'] == 'S' and '-SF' not in row['id']
        assert row['priority'] == (-3 if row['id'].split('-')[2] in ('pre','export') else -1)
        assert not any('-qa-ours-' in d or '-calib-' in d for d in row['deps'])
        if row['id'].split('-')[2] in ('pre','fqldata'):
            assert set(rgb.hybrid_ids(manifest)) <= set(row['deps'])
        if '-pre-' in row['id']:
            L._baseline_fields(L.legacy_row(row),ROOT,L.REMOTE_PREFIX)
        if '-fql-' in row['id']:
            args = row['args']
            assert ('pixel.p_aug=0.0' in args) == (row['rgb_method']=='e2e_cnn_noaug')
            if row['rgb_method'] in ('reviwo','dynamo'):
                assert 'agent.alpha=10.0' in args and 'agent.normalize_q_loss=true' in args
    bad = copy.deepcopy(rows[0])
    bad['registration_tool'] = 'other'
    with pytest.raises(ValueError,match='invalid RGB'):
        L.needed_inputs(bad)
    bad = copy.deepcopy(rows[0])
    bad['script'] = 'scripts/dsreset_train_sincro.py'
    with pytest.raises(ValueError,match='not allowed'):
        L.needed_inputs(bad)
    for variant in ('SF',):
        with pytest.raises(ValueError,match='SUCCESS-only'):
            reg.build_rows(tmp_path/'S',canonical_manifests(rgb.TASK)[variant],{})


def test_preview_needs_no_bundle_and_writes_nothing(tmp_path,capsys):
    reg = registrar()
    path = tmp_path/'manifest.json'
    path.write_text(json.dumps(canonical_manifests(rgb.TASK)['S']))
    before = sorted(tmp_path.rglob('*'))
    assert reg.main(['--manifest',str(path),'--bundle',str(tmp_path/'S'),
                     '--commit','a'*40,'--image','sha256:'+'b'*64]) == 0
    record = json.loads(capsys.readouterr().out)
    assert not record['ready'] and not record['registered'] and len(record['rows']) == 32
    assert sorted(tmp_path.rglob('*')) == before


def test_publication_digest_and_inventory_fail_closed(tmp_path):
    from ops.jobs.local_phase_a import inventory, sha
    bundle = tmp_path/'S'; bundle.mkdir()
    manifest = canonical_manifests(rgb.TASK)['S']
    (bundle/'manifest.json').write_text(json.dumps(manifest))
    # Malformed/failed report is refused even with a consistent inventory.
    (bundle/'gate_report.json').write_text('{}')
    proof = dict(schema='drawer-rgb-inputs/1',prepared=True,out=str(bundle),files=inventory(bundle))
    (bundle/'publication.json').write_text(json.dumps(proof))
    checksum = sha(bundle/'publication.json')
    with pytest.raises(ValueError,match='RGB gate'):
        rgb.verify(bundle,checksum)
    (bundle/'manifest.json').write_text('{}')
    with pytest.raises(ValueError,match='inventory drift'):
        rgb.verify(bundle,checksum)
    with pytest.raises(ValueError,match='SHA drift'):
        rgb.verify(bundle,'0'*64)


def test_rgb_only_matches_consumed_bytes_and_refuses_geometry(built, tmp_path, monkeypatch):
    """Same samples with all phase-B/SAM records absent; never read later geometry."""
    import shutil
    import numpy as np
    import torch
    from s4d.realworld import train_store as TS
    from s4d.realworld.rgb_train_store import write_episode
    from tests.test_realworld_train_store import _optional_store
    from s4d.realworld import pipeline
    from s4d.rwlearn.baseline_data import RealworldReViWoStates, RealworldDynaMoWindows
    from s4d.rwlearn.store import sync_index
    source = tmp_path/'input'
    ep = built['ep']
    shutil.copytree(built['root']/ 'episodes'/ep/'normalized',source/'episodes'/ep/'normalized')
    full = _optional_store(built,tmp_path,'reference',('rig0','rig1','rig2'))
    monkeypatch.setattr(pipeline,'accepted_calibration',lambda *a,**k: pytest.fail('RGB read calibration'))
    dest = tmp_path/'rgb'
    write_episode(source,ep,dest,task='synthetic',depth_stage='sensor_depth',rgb_only=True,semantic=False)
    rgb = TS.CompactStore([dest])
    key = rgb.episodes()[0]
    for cam in rgb.cameras(key):
        indices = np.arange(len(rgb.timestamps(key,cam)))
        for name in ('timestamps','segments','intrinsics'):
            np.testing.assert_array_equal(getattr(rgb,name)(key,cam),getattr(full,name)(key,cam))
        np.testing.assert_array_equal(rgb.rgb(key,cam,indices),full.rgb(key,cam,indices))
        for name in ('ok','probe_state'):
            np.testing.assert_array_equal(rgb.episode(key).robot(cam)[name],full.episode(key).robot(cam)[name])
        with pytest.raises((RuntimeError,ValueError),match='geometry|calibration'):
            rgb.w2c(key,cam)
    np.testing.assert_array_equal(sync_index(rgb,key,geometry=False).frames,sync_index(full,key,geometry=False).frames)
    np.testing.assert_array_equal(rgb.sync_sets(key,geometry=False)['surface_disp_m'],full.sync_sets(key,geometry=False)['surface_disp_m'])
    a,b = [RealworldReViWoStates(s,[key],num_views=2) for s in (rgb,full)]
    assert torch.equal(a[0]['images'],b[0]['images'])
    a,b = [RealworldDynaMoWindows(s,[key],window=3) for s in (rgb,full)]
    assert torch.equal(a[0],b[0])
    with pytest.raises(ValueError,match='semantic'):
        write_episode(source,ep,tmp_path/'bad',rgb_only=True,semantic=True)
    rgb.close(); full.close()


def test_local_gate_binds_exact_raw_identities_and_never_releases_geometry():
    from s4d.rwlearn import dsreset_gate as gate
    m = canonical_manifests(rgb.TASK)['S']
    class Store:
        rows = {r['id']:r for r in m['episodes']}
        def episodes(self): return list(self.rows)
        def task(self,e): return rgb.TASK
        def raw_recording(self,e): return self.rows[e]['path']
        def raw_sha256(self,e): return self.rows[e]['sha256']
    evidence = {r['id']:dict(singleview=dict(passed=True),m5=dict(accepted=1),raw_validation=dict(checked=True))
                for r in m['episodes']}
    report = rgb.local_report(m,Store(),evidence,dict(verification='local'))
    assert 'local' in report['verification_source'] and not report['geometry_passed']
    for consumer in ('dynamo','reviwo','e2e_cnn','fql_reviwo'):
        assert gate.task_passed(report,rgb.TASK,consumer)
    for consumer in ('ours','sincro','snerl','mvp3d','mvp3d_single','fql_ours','fql_sincro',None):
        assert not gate.task_passed(report,rgb.TASK,consumer)
    evidence[m['episodes'][0]['id']]['singleview']['passed'] = False
    with pytest.raises(ValueError,match='gate failed'):
        rgb.local_report(m,Store(),evidence,{})
