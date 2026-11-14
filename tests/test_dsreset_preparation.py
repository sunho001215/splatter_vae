"""Compile real worker argv, preserve label identity, and reject leaked measurements."""
import copy
import importlib.util
import json
from pathlib import Path
import pytest
import yaml
from ops.jobs.rwprep_mig import validate,ready
from s4d.rwlearn.dsreset_measure import validate_report

REPO=Path(__file__).resolve().parents[1]


def script(name):
    spec=importlib.util.spec_from_file_location(name,REPO/'scripts'/f'{name}.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


def test_compiled_queue_preserves_colliding_labels_and_emits_phaseB_input_plan(tmp_path):
    rows=[]
    for label in ('success','failure'):
        ep=f'drawer-open-v2-{label}-episode_000001'
        cfg=tmp_path/f'{label}.yaml'
        cfg.write_text(yaml.safe_dump({'output_root':'/old','adapter':{'config':{'episodes':{
            ep:f'/home/ws/data/real_world/drawer-open-v2/{label}/episode_000001.h5'}}},
            'stages':{'calibrate':{'rig_calibration':'/pending'}}}))
        rows.append({'episode':ep,'config':str(cfg),'raw_sha256':label[0]*64})
    plan=tmp_path/'plan.json';plan.write_text(json.dumps(rows))
    out=tmp_path/'queue'
    jobs=script('realworld_prep_register').compile_plans([plan],out,tmp_path/'processed',tmp_path/'raw')
    assert len({j['id'] for j in jobs})==2
    configs=[yaml.safe_load(Path(r['config']).read_text()) for r in json.loads((out/'prep-plan.json').read_text())]
    assert len({c['output_root'] for c in configs})==2
    assert all('--stages' in j['steps'][0]['argv'] and j['gpu_gib']==16 for j in jobs)
    assert all('/old' not in c['output_root'] for c in configs)


def test_calibration_graph_has_real_stages_and_one_bounded_matcher_per_MIG(tmp_path):
    source=[]
    for n in range(3):
        ep=f'drawer-open-v2-success-episode_{n:06d}';c=tmp_path/f'{n}.yaml'
        c.write_text(yaml.safe_dump({'output_root':str(tmp_path/f'p{n}')}))
        source.append({'id':f'rwv2-20261113-prep-ours-{ep}','episode':ep,'inputs':{str(c):'a'*64}})
    jobs=tmp_path/'source.json';jobs.write_text(json.dumps(source))
    cfg=dict(host='local',image='sha256:'+'a'*64,commit='b'*40,jobs=str(jobs),state_dir=str(tmp_path/'phase-state'),priority_file=str(tmp_path/'holds/phaseA.json'))
    config=tmp_path/'config.json';config.write_text(json.dumps(cfg))
    out=tmp_path/'calib'
    count=script('realworld_dsreset_calibration').plan(config,'drawer-open-v2',out,tmp_path/'work',tmp_path/'upstream')
    rows=json.loads((out/'jobs.json').read_text());validate(cfg,rows)
    assert count==len(rows) and not ready(rows,{})
    byid={j['id']:j for j in rows}
    assert all(set(j['deps'])<=set(byid) for j in rows)
    modules=[s['argv'][1] for j in rows for s in j['steps'] if s['argv'][0]=='-m']
    assert 's4d.realworld.calib.criteria' in modules
    assert 's4d.realworld.rig_dw_fixedness' in modules
    loma=[j for j in rows if '-loma' in j['id']]
    assert loma and all(len(j['steps'])==1 and j['steps'][0]['cuda_cap_gib']==20 for j in loma)
    assert all('--device' in j['steps'][0]['argv'] and 'cpu' in j['steps'][0]['argv'] for j in loma)
    assert len([j for j in rows if '-solve-loco-' in j['id']])==3
    assert all('/' not in j['id'] for j in rows)


def test_native_scene_report_binds_membership_and_containment():
    import hashlib
    manifest={'task':'drawer-open-v2','split':{'train':['success-1','failure-1']}}
    report={'schema':'dsreset-scene-measurement/1','accepted':True,'task':'drawer-open-v2',
        'depth_source':'native_realsense','manifest_sha256':hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest(),
        'episodes':['success-1','failure-1'],'bounds':[-1,-1,-1,1,1,1],'lookat':[0,0,0],'pose_scale':4,
        'containment':{'robot':.995,'object':1,'fixture':.99},'native_provenance':{'verified':True},'scene_statistics':{'measured':True}}
    assert validate_report(report,manifest)['pose_scale']==4
    for change in ({'episodes':['success-1','validation-1']},{'depth_source':'foundation_stereo_depth'},
                   {'pose_scale':float('nan')},{'containment':{'robot':.98,'object':1,'fixture':1}}):
        with pytest.raises(ValueError):validate_report({**report,**change},manifest)


def test_automatic_publication_uses_the_established_frozen_verdict_contract(tmp_path):
    from tests.test_realworld_calibration_release import _fake_results
    from s4d.realworld.calib.loma.publish import load_results,release_verdict,check_against_frozen
    _,fixed,_=_fake_results(tmp_path/'results')
    verdict=release_verdict(load_results(tmp_path/'results'),fixed)
    frozen=script('realworld_dsreset_calibration').freeze_release1(verdict)
    check_against_frozen(verdict,frozen)
    next(iter(frozen['samfree'].values()))['corr_challenge_all_rejected']=False
    with pytest.raises(SystemExit,match='SAM-free challenge'):
        check_against_frozen(verdict,frozen)


def test_scene_runtime_settings_are_identical_across_variants_and_stage_priority_dominates():
    from ops.jobs.dsreset_plan import build_rows
    from tests.test_dsreset_registration import registrar
    r=registrar();inputs='/home/compu/kaist/sunho/realworld/learning_inputs/dsreset/drawer-open-v2'
    rows=build_rows('drawer-open-v2',inputs,{'commit':'a'*40,'image':'sha256:'+'b'*64},
        stage_rows=r.default_stages('drawer-open-v2',inputs),rig='/rig/drawer-open-v2.json',
        scene_parameters={'lookat':[.3,0,.4],'pose_scale':3.7})
    snerl=[j for j in rows if j['script']=='scripts/dsreset_train_snerl.py']
    assert len(snerl)==2 and all('data_interface.lookat=[0.3,0,0.4]' in j['args'] for j in snerl)
    for method in ('ours','snerl','dynamo','sincro','reviwo','mvp3d_multiview'):
        pre = [j['priority'] for j in rows if f'-pre-{method}-' in j['id'] or f'-pre400k-{method}-' in j['id']]
        fql = [j['priority'] for j in rows if f'-fql-{method}-' in j['id']]
        assert pre and fql and max(pre)<min(fql)



def test_snapshot_reuses_all_five_unhashed_exclusions_and_preserves_keys(tmp_path, monkeypatch):
    import copy
    import hashlib
    import json
    import sys
    import types
    from scripts import realworld_dsreset_snapshot as snapshot
    from s4d.rwlearn.dsreset import manifest_gate, paired_manifests
    from tests.test_rwlearn_gate import entry
    source = {'episodes': {}}
    args = ['snapshot', '--source', str(tmp_path/'source.json'), '--remote-hashes', str(tmp_path/'hashes.json'),
            '--prep-config', str(tmp_path/'dispatcher.json'), '--out-dir', str(tmp_path/'out')]
    manifests = []
    keys = []
    for task, excluded_indices in [('covering',(57,61)), ('pick-and-place',(14,80,101))]:
        roots = {}
        indices = sorted(set(range(100)) | set(excluded_indices))
        while len(indices)>100:
            indices.remove(next(n for n in reversed(indices) if n not in excluded_indices))
        names = [f'episode_{n:06d}.h5' for n in indices]  # exactly 100 success files, with interleaved counters
        for label in ('success','failure'):
            directory = tmp_path/task/label
            directory.mkdir(parents=True)
            roots[label] = directory
            Path(str(directory)+'.VERIFIED').touch()
            for name in names if label=='success' else names[:50]:
                (directory/name).write_text(f'{task}:{label}:{name}')
        omitted = {f'{task}/episode_{i:06d}.h5': {'reason':'invalid raw schema',
                   'raw_sha256':hashlib.sha256((roots['success']/f'episode_{i:06d}.h5').read_bytes()).hexdigest()}
                   for i in excluded_indices}
        hashes = {f'{task}/{path.name}':hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in roots['success'].iterdir() if f'{task}/{path.name}' not in omitted}
        amendment = dict(format='rw-gate-amendment/1', task=task, base_target=100,target=len(hashes),
                         episodes=hashes,excluded=omitted,decided_by='user',decision='exclude invalid')
        m = paired_manifests(task,roots,inspect=lambda p:{},amendment=amendment)['SF']
        manifests.append(m)
        for row in m['episodes']:
            source['episodes'][row['id']] = dict(entry(task,int(row['filename'][8:14]),sha=row['sha256'],validated=True),
                source={'path':row['path']},filename=row['filename'],episode_label=row['label'])
        for row in m['excluded']:
            key = f"original/{task}/{row['filename']}"
            keys.append(key)
            source['episodes'][key] = dict(task=task, filename=row['filename'], source={'path':row['path']},
                                           sha256=None,validation={'status':'invalid'}, key=key)
        mp = tmp_path/f'{task}.json'
        mp.write_text(json.dumps(m))
        args += ['--manifest', str(mp)]
    (tmp_path/'source.json').write_text(json.dumps(source))
    (tmp_path/'hashes.json').write_text('{}')
    (tmp_path/'jobs.json').write_text('[]')
    (tmp_path/'dispatcher.json').write_text(json.dumps({'jobs':str(tmp_path/'jobs.json')}))
    monkeypatch.setitem(sys.modules,'episode_validator',types.SimpleNamespace(validate=lambda *a,**kw:{'status':'invalid'}))
    monkeypatch.setattr(sys,'argv',args)
    snapshot.main()
    result = json.loads((tmp_path/'out/manifest.json').read_text())
    assert set(result['episodes']) == set(source['episodes'])
    assert len(keys) == 5
    for key in keys:
        assert result['episodes'][key]['key'] == key and result['episodes'][key]['sha256']
        assert result['episodes'][key]['validation']['status'] == 'invalid'
    assert json.loads((tmp_path/'source.json').read_text()) == source
    for m in manifests: assert manifest_gate(m,result)['passed']
    # Already-duplicated identities must fail instead of silently choosing a key.
    duplicate = copy.deepcopy(source)
    duplicate['episodes']['duplicate'] = dict(duplicate['episodes'][keys[0]])
    (tmp_path/'source.json').write_text(json.dumps(duplicate))
    with pytest.raises(ValueError,match='duplicate ingest raw identity'): snapshot.main()
