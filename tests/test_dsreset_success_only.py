"""Independent S publication must never read an SF file or failure root."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from s4d.rwlearn import dsreset as D
from tests.test_dsreset_registration import registrar, canonical_manifests
from tests.test_rwlearn_gate import entry


def test_s_inventory_never_opens_failure_root(tmp_path):
    success=tmp_path/'success';success.mkdir()
    for i in range(3):(success/f'episode_{i:06d}.h5').write_text(str(i))
    m=D.paired_manifests('drawer-open-v2',{'success':success,'failure':tmp_path/'missing'},
        variants=('S',),require_verified=False,inspect=lambda p:{})
    assert set(m)=={'S'} and all(e['label']=='success' for e in m['S']['episodes'])
    template={'adapter':{'config':{}},'stages':{'train_data':{},'calibrate':{'rig_calibration':'/rig/drawer-open-v2.json'}}}
    assert len(D.prep_bundle(m,template,tmp_path/'processed'))==3
    with pytest.raises(ValueError,match='variants'):D.selected_variants(['SF'])


def test_s_inputs_keep_canonical_qa_split_and_calibration_gates_without_sf(tmp_path):
    r=registrar();m=canonical_manifests('covering')['S']
    ingest={'episodes':{e['id']:entry('covering',i,sha=e['sha256'],validated=True) for i,e in enumerate(m['episodes'])}}
    for e in m['excluded']:
        ingest['episodes'][e['id']]={'task':'covering','filename':e['filename'],'source':{'path':e['path']},
            'sha256':None,'validation':{'status':'invalid'}}
    class Store:
        def episodes(self):return [e['id'] for e in m['episodes']]
        def task(self,ep):return 'covering'
        def raw_recording(self,ep):return next(e['path'] for e in m['episodes'] if e['id']==ep)
        def raw_sha256(self,ep):return next(e['sha256'] for e in m['episodes'] if e['id']==ep)
    docs=D.learning_inputs(m,ingest,Store())
    paths=[]
    for e in m['episodes']:
        path='/home/compu/kaist/sunho/realworld/processed/'+e['id'];paths.append(path)
        ingest['episodes'][e['id']]['preprocessing']['stages']['train_data']['outputs']=path
    directory=tmp_path/'S';directory.mkdir()
    for name,doc in docs.items():(directory/name).write_text(json.dumps(doc))
    (directory/'store.yaml').write_text(yaml.safe_dump({'kind':'compact','paths':paths}))
    assert len(r.validate_inputs('covering',tmp_path,ingest,('S',)))==5
    mutated=copy.deepcopy(ingest)
    next(iter(mutated['episodes'].values()))['preprocessing']['stages']['calibration_validated']['metrics']['decision']='rejected'
    with pytest.raises((ValueError,r.R.Refused),match='calibration|QA'):r.validate_inputs('covering',tmp_path,mutated,('S',))
    wrong=copy.deepcopy(m);wrong['amendment']['target']=100
    with pytest.raises(r.R.Refused,match='amendment'):r.validate_membership('covering',{'S':wrong},('S',))
    split=json.loads((directory/'split_seed0.json').read_text());split['train'].append(split['validation'][0])
    (directory/'split_seed0.json').write_text(json.dumps(split))
    with pytest.raises(r.R.Refused,match='split'):r.validate_inputs('covering',tmp_path,ingest,('S',))


def test_s_graph_has_all_baselines_and_no_sf_or_failure_argv(tmp_path):
    r=registrar();(tmp_path/'experiments').mkdir()
    (tmp_path/'experiments/queue.yaml').write_text('jobs: []\n');(tmp_path/'experiments/registry.jsonl').write_text('')
    cfg={'variants':['S'],'ours_input_barriers_only':True,'tasks':{'covering':{
        'inputs':'/home/compu/kaist/sunho/realworld/learning_inputs/dsreset/covering',
        'rig':'/home/compu/kaist/sunho/realworld/rig/covering.json'}}}
    path=tmp_path/'spec.json';path.write_text(json.dumps(cfg))
    record=r.prepare(tmp_path,cfg,path,tmp_path/'absent',{'commit':'a'*40,'image':'sha256:'+'b'*64},preview=True)
    assert record['variants']==['S']
    assert '-SF' not in json.dumps(record['rows']) and '/SF/' not in json.dumps(record['rows'])
    ours=[j for j in record['rows'] if '-ours-' in j['id']]
    assert len(ours)==4 and all(j['id'].split('-')[2] in ('prep','calib','qa') for j in ours)
    assert len([j for j in record['rows'] if '-fql-' in j['id']])==24
    assert len([j for j in record['rows'] if '-eval-' in j['id']])==24


def test_s_measurement_uses_train_only_and_rejects_sf_report(monkeypatch):
    from s4d.rwlearn import dsreset_measure as M
    m=canonical_manifests('covering')['S'];train=set(m['split']['train'])
    class Store:
        has_semantic=True
        def episodes(self):return [e['id'] for e in m['episodes']]
        def task(self,ep):return 'covering'
        def raw_recording(self,ep):return next(e['path'] for e in m['episodes'] if e['id']==ep)
        def raw_sha256(self,ep):return next(e['sha256'] for e in m['episodes'] if e['id']==ep)
        def cameras(self,ep):assert ep in train;return ['cam']
        def timestamps(self,ep,cam):return np.arange(12)
        def depth(self,ep,cam,idx):
            assert ep in train and len(idx)==8
            return np.ones((8,2,3))
        def semantic(self,ep,cam,idx):return np.tile(np.array([[1,2,3],[1,2,3]]),(8,1,1))
        def intrinsics(self,ep,cam):return np.eye(3)
        def w2c(self,ep,cam):return np.eye(4)
        def depth_provenance(self,eps):assert set(eps)==train;return {'verified':True}
    monkeypatch.setattr(M,'scene_statistics',lambda store,eps,**kw: {'lookat_m':[0,0,0],
        'snerl_pose_scale_for_distance_4':4,'episodes':len(eps)})
    report=M.measure(m,Store(),m['split'])
    assert report['variant']=='S' and set(report['episodes'])==train
    with pytest.raises(ValueError,match='hash-bound'):M.validate_report(report,canonical_manifests('covering')['SF'])
    bad=copy.deepcopy(report);bad['containment']['robot']=float('nan')
    with pytest.raises(ValueError):M.validate_report(bad,m)


def test_sampled_native_measurement_depth_matches_established_overlay_builder(tmp_path):
    from tests.test_rwlearn_native_depth import make_store,build
    from s4d.rwlearn.store import open_store
    from s4d.rwlearn.dsreset_measure import SampledNativeStore
    fixture=make_store(tmp_path/'native')
    build(fixture)
    base=open_store(fixture.locator);native=open_store(fixture.out)
    sample=SampledNativeStore(base,fixture.rig)
    ep=base.episodes()[0]
    for cam in base.cameras(ep):
        np.testing.assert_array_equal(sample.depth(ep,cam,[0,2,3]),native.depth(ep,cam,[0,2,3]))
    assert sample.depth_provenance([ep])['rig_document_sha256']==native.rig_document_sha256
    base.close();native.close()


def test_exact_s_pause_is_installed_before_atomic_queue_publication(tmp_path,monkeypatch):
    r=registrar();(tmp_path/'experiments').mkdir();(tmp_path/'runs/setup').mkdir(parents=True)
    (tmp_path/'experiments/queue.yaml').write_text('jobs: []\n');(tmp_path/'experiments/registry.jsonl').write_text('')
    pauses=tmp_path/'runs/setup/admission-pauses.json';pauses.write_text('{"pauses": []}')
    ingest=tmp_path/'ingest.json';ingest.write_text('{}')
    cfg={'variants':['S'],'methods':['e2e_cnn'],'ours_input_barriers_only':True,'tasks':{'covering':{
        'inputs':'/home/compu/kaist/sunho/realworld/learning_inputs/test/covering','inputs_copy':'/unused',
        'rig':'/home/compu/kaist/sunho/realworld/rig/covering.json'}}}
    path=tmp_path/'spec.json';path.write_text(json.dumps(cfg))
    monkeypatch.setattr(r,'validate_inputs',lambda *a:{})
    release={'commit':'a'*40,'image':'sha256:'+'b'*64,'source':'local'}
    record=r.prepare(tmp_path,cfg,path,ingest,release)
    tool=tmp_path/r.TOOL;tool.parent.mkdir(parents=True,exist_ok=True);tool.write_bytes(Path(r.__file__).read_bytes())
    monkeypatch.setattr(r,'MAIN',tmp_path);monkeypatch.setattr(r,'__file__',str(tool))
    monkeypatch.setattr(r.R,'current_pin',lambda *a:release)
    from ops.jobs import admission_pause
    monkeypatch.setattr(admission_pause,'FILE',pauses)
    original=r.R.atomic_write
    def write(target,data):
        if target==tmp_path/'experiments/queue.yaml':
            assert all(r.PAUSE in admission_pause.active_pauses(j['id']) for j in record['rows'])
            assert all('-SF' not in i for p in json.loads(pauses.read_bytes())['pauses'] for i in p['ids'])
        original(target,data)
    monkeypatch.setattr(r.R,'atomic_write',write)
    published=r.execute(tmp_path,record)
    assert published['registered'] and published['pause_sha256_before']!=published['pause_sha256_after']
    assert not list((tmp_path/'runs/setup').glob('*.pending.json'))
    with pytest.raises(r.R.Refused,match='journal|proof'):r.execute(tmp_path,record)


def test_offline_s_input_publication_is_append_only_and_verifies_compact_closure(tmp_path,monkeypatch):
    import importlib.util
    from ops.jobs import realworld_local as L
    path=Path(__file__).resolve().parents[1]/'runs/setup/tools/stage_dsreset_s_inputs_20261115.py'
    spec=importlib.util.spec_from_file_location('stage_s_test',path)
    stage=importlib.util.module_from_spec(spec);spec.loader.exec_module(stage)
    r=registrar();monkeypatch.setattr(r,'validate_inputs',lambda *a:{})
    monkeypatch.setattr(stage,'registrar',lambda:r)
    mirror=tmp_path/'mirror';mirror.mkdir()
    records={}
    for name in ('episode.h5','_COMPLETE.json'):
        target=mirror/'processed/episode'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(name)
        records['processed/episode/'+name]=dict(kind='copy',sha256=L._hash(target),remote_sha256=L._hash(target),
            bytes=target.stat().st_size,mtime_ns=target.stat().st_mtime_ns,requires=[])
    manifest=dict(version=1,remote_root=L.REMOTE_PREFIX,local_root='runs/realworld-inputs',files=records)
    L._json(mirror/'MANIFEST.json',manifest)
    source=tmp_path/'inputs/S';source.mkdir(parents=True)
    for name in ('manifest.json','gate_report.json','split_seed0.json','outcomes.json'):(source/name).write_text('{}')
    (source/'store.yaml').write_text(yaml.safe_dump({'kind':'compact','paths':[L.REMOTE_PREFIX+'/processed/episode']}))
    cfg={'variants':['S'],'tasks':{'covering':{'inputs':L.REMOTE_PREFIX+'/learning_inputs/test/covering',
        'inputs_copy':str(source.parent)}}}
    before=(mirror/'MANIFEST.json').read_bytes()
    assert stage.publish(cfg,{},mirror)['status']=='dry-run'
    assert (mirror/'MANIFEST.json').read_bytes()==before
    stage.publish(cfg,{},mirror,execute=True)
    files=L._manifest(mirror/'MANIFEST.json')['files']
    assert L._check('learning_inputs/test/covering/S/store.yaml',files,mirror,set())
    (source/'outcomes.json').write_text('{"changed":true}')
    with pytest.raises(ValueError,match='conflict'):stage.publish(cfg,{},mirror,execute=True)
