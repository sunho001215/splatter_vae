"""The reset graph must be paired, complete, and portable before publication."""
import importlib.util
import json
from pathlib import Path

import pytest
from ops.jobs import realworld_local as L
from ops.jobs.dsreset_plan import METHODS, build_rows, row_id

REPO = Path(__file__).resolve().parents[1]


def registrar():
    spec=importlib.util.spec_from_file_location('dsreset_registrar_test',REPO/'runs/setup/tools/register_dsreset_20261113.py')
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_complete_paired_graph_supports_both_hosts_and_no_retired_tasks(tmp_path):
    r=registrar()
    release={'commit':'a'*40,'image':'sha256:'+'b'*64,'source':'local'}
    total=[]
    for task in ('drawer-open-v2','covering','pick-and-place'):
        inputs=f'/home/compu/kaist/sunho/realworld/learning_inputs/dsreset/{task}'
        rows=build_rows(task,inputs,release,stage_rows=r.default_stages(task,inputs),rig=f'/home/compu/kaist/sunho/realworld/rig/{task}.json')
        byid={row['id']:row for row in rows}
        assert len(byid)==len(rows)
        for row in rows:
            assert all(d in byid for d in row['deps'])
            L.needed_inputs(row)  # exact local argv validation, including outputs and declared exports
            assert L.has_resolver(row)
            stage = row['id'].split('-')[2]
            ours = '-ours-' in row['id']
            expected = -4 if stage in ('prep','calib','qa') else (-3 if ours else 0) if stage in ('pre','pre400k') else (-1 if ours else 1)
            assert row['priority'] == expected
        for variant in ('S','SF'):
            for method in METHODS:
                for seed in (0,1,2):
                    q=byid[row_id('fql',method,task,variant,seed)]
                    assert q['deps']==[row_id('fqldata',method,task,variant)]
                    a=q['args']
                    if method.startswith('e2e_cnn'):
                        assert 'agent.normalize_q_loss=true' not in a
                        assert ('pixel.p_aug=0.0' in a)==(method=='e2e_cnn_noaug')
                    else:
                        assert 'agent.alpha=10.0' in a and 'agent.normalize_q_loss=true' in a
        assert not any('mug-hanging' in row['id'] or '-drawer-open-S' in row['id'] for row in rows)
        total+=rows
    assert len({row['id'] for row in total})==len(total)


def test_pretrain_science_differs_only_in_manifest_derived_inputs():
    r=registrar()
    release={'commit':'a'*40,'image':'sha256:'+'b'*64}
    inputs='/home/compu/kaist/sunho/realworld/learning_inputs/dsreset/covering'
    rows=build_rows('covering',inputs,release,stage_rows=r.default_stages('covering',inputs),rig='/home/compu/kaist/sunho/realworld/rig/covering.json')
    byid={row['id']:row for row in rows}
    for method in ('ours','sincro','reviwo','dynamo','snerl','mvp3d_multiview'):
        a,b=[byid[row_id('pre',method,'covering',v)] for v in ('S','SF')]
        normalize=lambda args:[arg.replace('/SF/','/VAR/').replace('/S/','/VAR/').replace('-SF','-VAR').replace('-S','-VAR')
                               for arg in args]
        assert a['script']==b['script'] and normalize(a['args'])==normalize(b['args'])


def test_preview_is_read_only_and_cannot_publish(tmp_path):
    r=registrar()
    (tmp_path/'experiments').mkdir()
    queue=tmp_path/'experiments/queue.yaml'
    registry=tmp_path/'experiments/registry.jsonl'
    queue.write_text('jobs: []\n')
    registry.write_text('')
    spec=tmp_path/'spec.json'
    cfg={'tasks':{'covering':{'inputs':'/home/compu/kaist/sunho/realworld/learning_inputs/dsreset/covering',
                              'inputs_copy':'/missing', 'rig':'/home/compu/kaist/sunho/realworld/rig/covering.json'}}}
    spec.write_text(json.dumps(cfg))
    before=(queue.read_bytes(),registry.read_bytes())
    record=r.prepare(tmp_path,cfg,spec,tmp_path/'not-arrived.json',
                     {'commit':'a'*40,'image':'sha256:'+'b'*64,'source':'local'},preview=True)
    assert record['status']=='preview' and not record['registered']
    assert (queue.read_bytes(),registry.read_bytes())==before
    with pytest.raises(r.R.Refused,match='coordinator'):
        r.execute(tmp_path,record)


def test_bound_inputs_reject_label_split_and_live_calibration_mutations(tmp_path):
    import copy
    import yaml
    from s4d.rwlearn.dsreset import learning_inputs, paired_manifests
    from tests.test_rwlearn_dsreset import fixture_roots
    from tests.test_rwlearn_gate import entry
    r = registrar()
    manifests = canonical_manifests('covering')
    ingest = {'episodes':{ep['id']:entry('covering',int(ep['filename'][8:14]),sha=ep['sha256'],validated=True)
                          for ep in manifests['SF']['episodes']}}
    for ep in manifests['SF']['excluded']:
        ingest['episodes'][ep['id']] = {'task': 'covering', 'filename': ep['filename'],
            'source': {'path': ep['path']}, 'sha256': None, 'validation': {'status': 'invalid'}}
    class Identities:
        def __init__(self,rows): self.rows={ep['id']:ep for ep in rows}
        def episodes(self): return list(self.rows)
        def task(self,ep): return self.rows[ep]['task']
        def raw_recording(self,ep): return self.rows[ep]['path']
        def raw_sha256(self,ep): return self.rows[ep]['sha256']
    inputs = tmp_path/'inputs'
    for variant,m in manifests.items():
        store = Identities(m['episodes'])
        docs = learning_inputs(m,ingest,store)
        paths=[]
        for ep in m['episodes']:
            path=f"/home/compu/kaist/sunho/realworld/processed/{ep['id']}"
            ingest['episodes'][ep['id']]['preprocessing']['stages']['train_data']['outputs']=path
            paths.append(path)
        directory=inputs/variant
        directory.mkdir(parents=True)
        for name,doc in docs.items(): (directory/name).write_text(json.dumps(doc))
        (directory/'store.yaml').write_text(yaml.safe_dump({'kind':'compact','paths':paths}))
    assert r.validate_inputs('covering',inputs,ingest)
    target=inputs/'SF/outcomes.json'
    saved=target.read_bytes()
    outcomes=json.loads(saved)
    outcomes[next(ep for ep,v in outcomes.items() if not v)]=True
    target.write_text(json.dumps(outcomes))
    with pytest.raises(r.R.Refused,match='outcomes'):
        r.validate_inputs('covering',inputs,ingest)
    target.write_bytes(saved)
    split=inputs/'SF/split_seed0.json'
    saved=split.read_bytes()
    doc=json.loads(saved)
    doc['train'].append(doc['validation'][0])
    split.write_text(json.dumps(doc))
    with pytest.raises(r.R.Refused,match='split'):
        r.validate_inputs('covering',inputs,ingest)
    split.write_bytes(saved)
    mutated=copy.deepcopy(ingest)
    first=next(iter(mutated['episodes'].values()))
    first['preprocessing']['stages']['calibration_validated']['metrics']['decision']='rejected'
    with pytest.raises(r.R.Refused,match='calibration'):
        r.validate_inputs('covering',inputs,mutated)


def test_ours_exports_only_final_single_cycle_400k():
    import yaml
    from s4d.config import load_config
    from s4d.train.schedule import lr_function
    r=registrar()
    task='covering'
    inputs=f'/home/compu/kaist/sunho/realworld/learning_inputs/dsreset/{task}'
    rows=build_rows(task,inputs,{'commit':'a'*40,'image':'sha256:'+'b'*64},
                    stage_rows=r.default_stages(task,inputs),rig=inputs+'/rig-covering.json')
    byid={row['id']:row for row in rows}
    for variant in ('S','SF'):
        final=byid[row_id('pre400k','ours',task,variant)]
        export=byid[row_id('export','ours',task,variant)]
        assert export['deps']==[final['id']]
        assert export['args'][export['args'].index('--checkpoint')+1].endswith('/step_0400000.pt')
        args=final['args']
        base=args[args.index('--baseline-config')+1]
        overlay=args[args.index('--schedule-config')+1]
        cfg=load_config([REPO/base,REPO/overlay])
        assert cfg['train']['restarts']==[{'start':200000,'end':400000,'warmup':2500,'peak_lr':.00025}]
        lr=lr_function(cfg)
        assert lr(200000)*cfg['train']['lr']==pytest.approx(1e-5)
        assert lr(202500)*cfg['train']['lr']==pytest.approx(2.5e-4)
        assert lr(400000)*cfg['train']['lr']==pytest.approx(1e-5)
        assert cfg['data']['task']==task


def test_existing_ours_S_chain_reuse_requires_identical_hashes_split_and_science(tmp_path):
    import copy
    from ops.jobs.dsreset_reuse import aliases,prove_reuse,reuse_rows
    from s4d.rwlearn.dsreset import paired_manifests,manifest_gate,sha256
    from tests.test_rwlearn_dsreset import fixture_roots
    from tests.test_rwlearn_gate import entry
    task='covering'
    m=paired_manifests(task,fixture_roots(tmp_path),inspect=lambda p:{})['S']
    ingest={'episodes':{r['id']:entry(task,i,sha=r['sha256'],validated=True) for i,r in enumerate(m['episodes'])}}
    report=manifest_gate(m,ingest)
    for key,rec in report['tasks'][task]['episode_identities'].items():rec['store_episode']=key
    inputs='/home/compu/kaist/sunho/realworld/learning_inputs/dsreset/covering'
    r=registrar()
    rows=build_rows(task,inputs,{'commit':'a'*40,'image':'sha256:'+'b'*64},methods=('ours',),
                    stage_rows=r.default_stages(task,inputs),rig=inputs+'/rig-covering.json')
    mapping=aliases(task)
    current={}
    for row in rows:
        if row['id'] not in mapping:continue
        old=copy.deepcopy(row)
        old['id']=mapping[row['id']]
        old['deps']=[mapping.get(d,d) for d in old['deps']]
        for a,b in sorted(mapping.items(),key=lambda pair:-len(pair[0])):
            old['args']=[x.replace(a,b) for x in old['args']]
        current[old['id']]=old
    pre=current[row_id('pre400k','ours',task,'S')]
    base='configs/realworld/ours_ext/baseline200k-covering-S.yaml'
    pre['args'][pre['args'].index('--baseline-config')+1]=base
    pre['args'][pre['args'].index('--baseline-sha256')+1]=sha256(REPO/base)
    descriptor={'kind':'compact','paths':['/exact/same/compact/store']}
    proof=prove_reuse(task,m,report,m['split'],m['outcomes'],descriptor,descriptor,rows,current,REPO,new_gate=report)
    assert proof['aliases']==mapping
    changed_gate=copy.deepcopy(report)
    next(iter(changed_gate['calibration_hold'][task]['episodes'].values()))['calibrate_build_id']='changed'
    with pytest.raises(ValueError,match='calibration builds'):
        prove_reuse(task,m,report,m['split'],m['outcomes'],descriptor,descriptor,rows,current,REPO,new_gate=changed_gate)
    changed=copy.deepcopy(m['split'])
    changed['train'][0],changed['validation'][0]=changed['validation'][0],changed['train'][0]
    with pytest.raises(ValueError,match='splits differ'):
        prove_reuse(task,m,report,changed,m['outcomes'],descriptor,descriptor,rows,current,REPO,new_gate=report)
    rewritten=reuse_rows(rows,task,current)
    assert row_id('pre400k','ours',task,'S') not in {r['id'] for r in rewritten}
    ev=next(r for r in rewritten if r['id']==row_id('eval','ours',task,'S',0))
    assert ev['deps']==[row_id('fql400k','ours',task,'S',0)]
    assert f"runs/{ev['deps'][0]}/policy.pt" in ev['args']


def canonical_manifests(task):
    """Canonical success identities plus synthetic, distinct failure hashes; no raw I/O."""
    import copy
    import hashlib
    from s4d.rwlearn import gate
    amendment = gate.load_amendments()[task]['record']
    def row(name, sha, label, index):
        return dict(id=f'{task}-{label}-{Path(name).stem}',task=task,label=label,filename=Path(name).name,
                    path=f'/raw/{task}/{label}/{Path(name).name}',sha256=sha,qa={},
                    split='validation' if index % 10 == 0 else 'train')
    success = [row(name, sha, 'success', i) for i,(name,sha) in enumerate(amendment['episodes'].items())]
    failure = [row(f'episode_{i:06d}.h5',hashlib.sha256(f'{task}:failure:{i}'.encode()).hexdigest(), 'failure',i)
               for i in range(50)]
    excluded = [dict(row(name,rec['raw_sha256'],'success',i), split='excluded',
                     qa={'status':'excluded','reason':rec['reason']})
                for i,(name,rec) in enumerate(amendment['excluded'].items())]
    result = {}
    for variant,episodes in [('S',success),('SF',success+failure)]:
        result[variant] = dict(schema='dsreset-episodes/v1',task=task,variant=variant,seed=0,validation_fraction=.1,
            episodes=copy.deepcopy(episodes),amendment=copy.deepcopy(amendment),excluded=copy.deepcopy(excluded),
            split={s:[r['id'] for r in episodes if r['split']==s] for s in ('train','validation')},
            outcomes={r['id']:r['label']=='success' for r in episodes})
    return result


@pytest.mark.parametrize('task', ['covering','pick-and-place','drawer-open-v2'])
def test_canonical_registration_membership_and_amendment_evidence(task):
    import copy
    from s4d.rwlearn.dsreset import validate_manifest
    r = registrar()
    docs = canonical_manifests(task)
    for manifest in docs.values(): validate_manifest(manifest)
    r.validate_membership(task, docs)
    def reject(mutate, message):
        broken = copy.deepcopy(docs)
        mutate(broken)
        with pytest.raises(r.R.Refused, match=message): r.validate_membership(task, broken)
    reject(lambda d: d['S'].update(episodes=d['S']['episodes'][:10]), 'success set')
    reject(lambda d: d['SF'].update(episodes=d['SF']['episodes'][:-1]), '50 failures')
    reject(lambda d: d['SF'].update(episodes=[e for e in d['SF']['episodes'] if e['label']=='success']), '50 failures')
    reject(lambda d: d['SF']['episodes'].append(dict(d['SF']['episodes'][-1])), '50 failures')
    reject(lambda d: d['S']['amendment'].update(decision='unapproved'), 'amendment evidence')
    reject(lambda d: d['S']['excluded'][0]['qa'].update(reason='not the documented reason'), 'exclusion evidence')
    reject(lambda d: d['S']['excluded'][0].update(path='/another/success/'+d['S']['excluded'][0]['filename']),
           'S/SF amendment/exclusion evidence')
    reject(lambda d: d['S']['episodes'][0].update(split='train' if d['S']['episodes'][0]['split']=='validation' else 'validation'), 'success subset')
    # A count-correct replacement is still not the approved success set.
    reject(lambda d: d['S']['episodes'][0].update(sha256='f'*64), 'success set')


def test_registrar_rejects_small_self_consistent_pair_before_qa(tmp_path):
    from s4d.rwlearn.dsreset import paired_manifests
    from tests.test_rwlearn_dsreset import fixture_roots
    r = registrar()
    docs = paired_manifests('covering',fixture_roots(tmp_path),inspect=lambda p:{})
    # Includes the independent review's zero-failure counterexample.
    docs['SF'] = dict(docs['S'], variant='SF')
    for variant, manifest in docs.items():
        directory = tmp_path/'inputs'/variant
        directory.mkdir(parents=True)
        (directory/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(r.R.Refused, match='amendment evidence'):
        r.validate_inputs('covering',tmp_path/'inputs',{'episodes':{}})


def test_paired_runtime_is_isolated_and_used_by_all_new_entrypoints(tmp_path):
    import os
    import subprocess
    import sys
    from s4d.rwlearn.dsreset_entrypoints import ENTRYPOINTS
    from ops.jobs.migration import import_closure, tree_reader
    read, _ = tree_reader(REPO)
    for paired, legacy in ENTRYPOINTS.items():
        closure = import_closure([paired], read)
        assert legacy in closure
        assert 's4d/rwlearn/dsreset_runtime.py' in closure
        assert 's4d/rwlearn/dsreset_identity.py' in closure
    # Process isolation is part of the contract: installing cannot contaminate
    # test/scheduler processes which have already imported legacy trainers.
    code = '''
from s4d.rwlearn import gate, identity, tasks, configs
from s4d.rwlearn import dsreset_gate, dsreset_identity, dsreset_configs
assert tasks.TARGETS == ("drawer-open", "mug-hanging", "covering", "pick-and-place")
assert "drawer-open-v2" not in configs.SCENE_PARAMETERS
assert gate.match_recordings is identity.match_recordings
assert dsreset_gate.match_recordings is dsreset_identity.match_recordings
from s4d.rwlearn.dsreset_runtime import install
install()
from s4d.rwlearn import gate as paired, identity as ids, tasks as ts, configs as cs
from s4d.rwlearn import baseline_data, store, fql_data
assert paired is not gate and ids is not identity
assert "drawer-open-v2" in ts.TARGETS and "drawer-open-v2" in cs.TARGET_TASKS
assert cs.pending_scene_parameters("snerl", "drawer-open-v2") == ["lookat", "pose_scale"]
from pathlib import Path
for baseline in cs.BASELINES:
    assert cs.render(baseline, 'drawer-open-v2') == (Path.cwd()/'configs/realworld/baselines'/baseline/'drawer-open-v2.yaml').read_text()
a=ids.raw_identity("covering", "/raw/success/episode_000001.h5", "a"*64)
b=ids.raw_identity("covering", "/raw/failure/episode_000001.h5", "b"*64)
assert paired.match_recordings({"s":a,"f":b},{"s":a,"f":b}) == {"s":"s","f":"f"}
assert gate.match_recordings is identity.match_recordings
try: identity.match_recordings({"s":a,"f":b},{"s":a,"f":b})
except RuntimeError: pass
else: raise AssertionError("legacy behavior mutated")
'''
    code += r"""
import copy, hashlib, json
from pathlib import Path
import sys
import h5py
import numpy as np
from s4d.rwlearn import actions as A
from tests.test_rwlearn_fql_pipeline import store_on_fixture, FIXTURE
from tests.test_rwlearn_gate import entry
out = Path(sys.argv[1])
eps = {}
for label in ('success', 'failure'):
    raw = A.raw_from_arrays(np.load(FIXTURE))
    for camera in raw.cameras.values():
        camera['t'], camera['segment'] = camera['t'][:8], camera['segment'][:8]
    raw.gripper_command = np.clip(raw.gripper_command + (.01 if label=='failure' else 0),0,1)
    directory = out/label
    directory.mkdir()
    path = directory/'episode_000001.npz'
    np.savez(path, **A.raw_to_arrays(raw))
    item = copy.deepcopy(next(iter(store_on_fixture(n=8,hw=8)._eps.values())))
    item.update(task='drawer-open-v2',raw=str(path),raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    eps[label] = item
memory = store.MemoryStore(eps,(8,8))
live = {}
for label, item in eps.items():
    live[label] = dict(entry('drawer-open-v2',1,sha=item['raw_sha256'],validated=True),
                       filename=item['raw'],source={'path':item['raw']},episode_label=label)
report = paired.evaluate({'schema':paired.SCHEMA,'schema_version':paired.SCHEMA_VERSION,'episodes':live,
                          'counts':{'drawer-open-v2':{'target':2,'complete':True}}}, ['drawer-open-v2'],target=2,amendments={})
assert report['passed'], report
split = {'train':['success'],'validation':['failure']}
checked = baseline_data.check_gate(memory,report,steps=200000,split=split,target='drawer-open-v2',consumer='e2e_cnn')
bound = paired.bind_store_episodes(report,{'drawer-open-v2':memory.episodes()},ids.store_identities(memory))
fql_data.build_fql_dataset(memory,'e2e_cnn',None,out/'dataset.h5',episodes=memory.episodes(),outcomes={'success':True,'failure':False})
paired.check_dataset_gate(out/'dataset.h5',bound,steps=1000000,split=split)
with h5py.File(out/'dataset.h5') as f:
    for label in eps:
        for cam in f['episodes'][label].values():
            final = np.flatnonzero(cam['transition_ok'][:])[-1]
            assert cam['mask'][final] == (label=='failure')
            assert cam['reward'][final] == (-1 if label=='failure' else 0)
"""
    result = subprocess.run([sys.executable,'-c',code,str(tmp_path)],cwd=REPO,capture_output=True,text=True,env=os.environ)
    assert result.returncode == 0, result.stderr


@pytest.fixture(scope='module')
def placement_scope_audit():
    from ops.jobs.dsreset_scope import audit
    return audit(REPO/'tests/fixtures/dsreset-live-scopes.yaml',REPO)


def test_every_persisted_placement_scope_has_zero_candidate_changes(placement_scope_audit):
    report = placement_scope_audit
    assert report['rows'], 'the committed ownership snapshot must include placed rows'
    assert report['zero_release_changes'], [(r['job_id'],r['release_changes']) for r in report['rows'] if r['release_changes']]
    covering = next(r for r in report['rows'] if r['job_id']=='rwv2-20261113-pre400k-ours-covering-S')
    assert covering['scope_persisted'] and not covering['differences'], covering
    running = [r for r in report['rows'] if r['running']]
    assert len(running) == 8  # registry snapshot includes all eight running rows
    assert all(r['scope_persisted'] and not r['release_changes'] for r in running), running


def test_every_live_or_placed_scope_has_same_differences_as_main(placement_scope_audit):
    report = placement_scope_audit
    assert report['baseline'] == '9c901bf7b0acabe399420c9b5e9a1c2b6bc42479'
    assert len(report['rows']) == 89
    differences = {r['job_id']:(r['differences'],r['baseline_differences'])
                   for r in report['rows'] if r['differences'] != r['baseline_differences']}
    assert report['same_scope_differences_as_main'] and not differences, differences
    # Four legacy placements have no persisted scope; record rather than invent it.
    assert sum(not r['scope_persisted'] for r in report['rows']) == 4
    assert sum(bool(r['baseline_differences']) for r in report['rows']) == 78


@pytest.mark.parametrize('baseline_content,candidate_content,same_differences,unchanged',[
    ('main','main',True,True),       # preserve pre-existing placement drift
    ('main','third',True,False),     # equal mismatch sets cannot hide a new digest
    ('main','row',False,False),      # even repairing a pinned mismatch changes scope
    ('row','third',False,False),     # introducing a new placement mismatch
])
def test_scope_audit_compares_baseline_and_exact_candidate_bytes(
        tmp_path,baseline_content,candidate_content,same_differences,unchanged):
    import hashlib
    import subprocess
    from ops.jobs.dsreset_scope import audit
    candidate = tmp_path/'candidate'
    candidate.mkdir()
    subprocess.run(['git','init','-q',str(candidate)],check=True)
    source = candidate/'shared.py'
    source.write_text(baseline_content)
    subprocess.run(['git','-C',str(candidate),'add','shared.py'],check=True)
    subprocess.run(['git','-C',str(candidate),'-c','user.name=Scope Test',
                    '-c','user.email=scope@example.invalid','commit','-qm','baseline'],check=True)
    baseline = subprocess.check_output(['git','-C',str(candidate),'rev-parse','HEAD'],text=True).strip()
    source.write_text(candidate_content)
    ownership = tmp_path/'ownership'
    ownership.mkdir()
    (ownership/'row.json').write_text(json.dumps(dict(job_id='row',state='placed',
        execution_scope=dict(source_commit=baseline,files={'shared.py':hashlib.sha256(b'row').hexdigest()}))))
    report = audit(ownership,candidate,baseline)
    assert report['same_scope_differences_as_main'] is same_differences
    assert report['zero_release_changes'] is unchanged
