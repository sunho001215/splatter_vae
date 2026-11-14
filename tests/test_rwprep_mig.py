"""Offline admission, priority and crash recovery for the dedicated prep route."""
import copy
import json
from pathlib import Path
import pytest
from ops.jobs import rwprep_mig as P


def snap():
    return dict(charge=10*P.GIB,growth=0,prep_ram=0,available=200*P.GIB,cpu_used=0,cpus=64,
                load=1,idle=.9,psi_cpu=0,disk_free=500*P.GIB,disk_total=1000*P.GIB,prep_disk=0,busy=set(),
                devices={'MIG-a':dict(processes=[],used_mib=0,total_mib=24192)})


def job():
    return dict(id='rwv2-20261113-prep-ours-test',gpu_profile='1g.24gb',gpu_gib=16,ram_gib=48,cpus=4,
                inputs={'/raw':'a'*64},steps=[{'kind':'rwprep','argv':[]}])


@pytest.mark.parametrize('field,value,reason',[
    ('charge',130*P.GIB,'RAM'),('load',9000,'CPU'),('idle',.1,'idle'),
    ('available',50*P.GIB,'reserve'),('disk_free',100*P.GIB,'disk'),('growth',130*P.GIB,'growth')])
def test_admission_rejects_without_bypassing_other_users(field,value,reason):
    s=snap();s[field]=value
    gpu,reasons=P.admission(job(),s,{'host':'local'})
    assert gpu is None and any(reason in r for r in reasons)


def test_mig_exclusive_and_cumulative_resource_admission():
    s=snap()
    assert P.admission(job(),s,{'host':'local'})[0]=='MIG-a'
    s['devices']['MIG-a']['processes']=[(42,1)]
    assert P.admission(job(),s,{'host':'local'})[0] is None
    s['devices']['MIG-a']['processes']=[]
    s['busy']={'MIG-a'}
    assert P.admission(job(),s,{'host':'local'})[0] is None
    s['busy']=set();s['prep_ram']=100*P.GIB
    assert P.admission(job(),s,{'host':'local'})[0] is None


def test_stop_dependencies_and_failed_jobs_do_not_reserve(tmp_path):
    a=job();b={**job(),'id':'b','deps':[a['id']],'requires':[str(tmp_path/'rig')]}
    assert P.ready([a,b],{})==[a]
    assert P.ready([a,b],{},True)==[]
    state={a['id']:{'status':'done'}}
    assert P.ready([a,b],state)==[]
    (tmp_path/'rig').touch()
    assert P.ready([a,b],state)==[b]
    state['b']={'status':'failed'}
    assert P.ready([a,b],state)==[]


def test_priority_is_live_and_fail_closed(tmp_path,monkeypatch):
    from ops.jobs import admission_pause as A
    monkeypatch.setattr(A,'FILE',tmp_path/'pauses')
    monkeypatch.setattr(A,'PREP_PRIORITY',tmp_path/'prep')
    monkeypatch.setattr(A,'PREP_PRIORITY_DIR',tmp_path/'holds')
    P.atomic(A.PREP_PRIORITY,{'active':True})
    assert A.paused('any-pretrain') and A.paused('any-fql')
    P.atomic(A.PREP_PRIORITY,{'active':False})
    assert not A.paused('any-pretrain')
    P.atomic(A.PREP_PRIORITY_DIR/'calibration.json',{'active':True})
    assert A.paused('any-pretrain')  # a drained phase A cannot clear calibration's hold
    P.atomic(A.PREP_PRIORITY_DIR/'calibration.json',{'active':False})
    A.PREP_PRIORITY.write_text('{broken')
    with pytest.raises(ValueError):A.paused('any-pretrain')


def test_launch_intent_recovers_container_without_duplicate_run(tmp_path,monkeypatch):
    j=job();c=dict(host='local',image='sha256:'+'b'*64,commit='c'*40,jobs=str(tmp_path/'jobs'),
        state_dir=str(tmp_path/'state'),admission_lock=str(tmp_path/'lock'),priority_file=str(tmp_path/'priority'))
    P.atomic(c['jobs'],[j]);sha=P.digest(c['jobs'])
    P.atomic(Path(c['state_dir'])/'state.json',{'queue_sha256':sha,'jobs':{j['id']:{
        'status':'launching','attempt':1,'gpu':'MIG-a','container':'named-intent'}}})
    calls=[]
    def fake(argv):
        calls.append(argv)
        if argv[1]=='ps':return 'container-id'
        return json.dumps([{'Config':{'Labels':{'s4d.queue_sha256':sha,'s4d.rwprep.job':j['id'],
            's4d.role':P.ROLE,'s4d.commit':c['commit']}},'State':{'Running':True}}])
    monkeypatch.setattr(P,'run',fake)
    result=P.tick(c,True)
    assert result['counts']['running']==1 and not result['launched']
    assert not any('run' in argv for argv in calls)


def test_oversize_trackeverything_refused():
    c=dict(host='local',image='sha256:'+'a'*64,commit='b'*40)
    j=job();j['gpu_gib']=40
    with pytest.raises(ValueError,match='invalid MIG'):P.validate(c,[j])


@pytest.mark.parametrize('status,now,expected',[
    ('launching',100,'launching'),('launching',1001,'retry'),('running',100,'unknown')])
def test_remote_lost_launch_reply_keeps_intent_until_deadline(status,now,expected):
    from ops.jobs import rwprep_remote as R
    state={'j':dict(status=status,container='named',attempt=1,deadline=600)}
    R.reconcile(state,{'named':None},{},'q',now)
    assert state['j']['status']==expected


def test_remote_identity_and_hash_bound_completion():
    from ops.jobs import rwprep_remote as R
    state={'j':dict(status='launching',container='named',attempt=1,deadline=600,gpu='MIG-x')}
    config={'commit':'abc'}
    labels={'s4d.commit':'abc','s4d.rwprep.job':'j','s4d.gpu':'MIG-x',
            's4d.role':'rwprep','s4d.owner':'kaist-sunho'}
    item=dict(labels=labels,state={'Running':False,'ExitCode':0},proof={'status':'ok','queue_sha256':'q'})
    R.reconcile(state,{'named':item},config,'q',0)
    assert state['j']['status']=='done'
    state['j']['status']='launching';item['proof']['queue_sha256']='wrong'
    R.reconcile(state,{'named':item},config,'q',0)
    assert state['j']['status']=='retry'
    state['j']['status']='running';labels['s4d.gpu']='GPU-whole'
    with pytest.raises(ValueError,match='identity'):R.reconcile(state,{'named':item},config,'q',0)


@pytest.mark.parametrize('problem',['quota','ram','data_disk','duplicate','input','none'])
def test_remote_guard_preserves_capacity_and_rewrites_worker(tmp_path,monkeypatch,problem):
    """Execute the actual embedded wrapper, with fake remote evidence and no Docker."""
    from ops.jobs import rwprep_remote as R
    import shutil
    import sys
    import time
    from types import SimpleNamespace
    p=tmp_path/'raw';p.write_bytes(b'raw')
    spec=dict(deadline=time.time()+600,code='/code',commit='abc',worker_job='/configs/job.json',
              inputs={str(p):R.digest(p)},slot_guard=dict(uuid='MIG-x',ram_bytes=48*R.GIB,reserve_bytes=120*R.GIB))
    if problem=='input':p.write_bytes(b'changed')
    job=dict(name='named',episode='ep',gpu='MIG-x',docker_args=['docker','run','WANDB_MODE=disabled','old_worker','old_job'])
    calls=[];guards=[]
    def run(*args):
        calls.append(args)
        if args[:2]==('docker','ps'):
            return 'named' if problem=='duplicate' and '--format' in args else ''
        if args[:2]==('git','-C'):return 'abc' if 'rev-parse' in args else ''
        if args[:2]==('docker','run'):return 'cid'
        raise AssertionError(args)
    def decide(data):
        guards.append(dict(data))
        assert data['reserve_bytes']==200*R.GIB and data['disk_floor_fraction']==.155
        return {'admit':problem not in ('quota','ram'),'reasons':[problem] if problem in ('quota','ram') else []}
    monkeypatch.setattr(shutil,'disk_usage',lambda _:SimpleNamespace(free=1 if problem=='data_disk' else 100,total=100))
    monkeypatch.setattr(sys,'argv',['guard',json.dumps(job)])
    env=dict(ROUTE_SPEC=spec,run=run,decide=decide,json=json,sys=sys)
    if problem in ('quota','ram','data_disk','duplicate'):
        with pytest.raises(SystemExit) as exc:exec(R.GUARD_WRAPPER,env)
        assert exc.value.code==20
    elif problem=='input':
        with pytest.raises(ValueError,match='input changed'):exec(R.GUARD_WRAPPER,env)
    else:
        exec(R.GUARD_WRAPPER,env)
        assert len(guards)==2  # revalidate fresh MIG/resource evidence after raw SHA checks
        command=next(c for c in calls if c[:2]==('docker','run'))
        assert command[-2:]==('/opt/src/splatter4d/scripts/realworld_prep_worker.py','/configs/job.json')
        assert 'WANDB_MODE=offline' in command
    if problem!='none':assert not any(c[:2]==('docker','run') for c in calls)


def test_remote_rewrite_preserves_label_inclusive_identity():
    from ops.jobs import rwprep_remote as R
    doc={'/local/configs/drawer-open-v2-failure-000001.yaml':['/local/state/server.yaml','drawer-open-v2-success-000001']}
    assert R.rewrite(doc,'/local','/remote')=={
        '/remote/configs/drawer-open-v2-failure-000001.yaml':['/remote/state/server.yaml','drawer-open-v2-success-000001']}
