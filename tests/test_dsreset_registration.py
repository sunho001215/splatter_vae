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
            assert row['priority'] in (-3,-2,-1,0,1)
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
    manifests = paired_manifests('covering',fixture_roots(tmp_path),inspect=lambda p:{})
    ingest = {'episodes':{ep['id']:entry('covering',int(ep['filename'][8:14]),sha=ep['sha256'],validated=True)
                          for ep in manifests['SF']['episodes']}}
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
