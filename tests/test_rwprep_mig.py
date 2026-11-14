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


@pytest.mark.parametrize('problem',['quota','ram','data_disk','duplicate','input','stop','stop_after_verify','none'])
def test_remote_guard_preserves_capacity_and_rewrites_worker(tmp_path,monkeypatch,problem):
    """Execute the actual embedded wrapper, with fake remote evidence and no Docker."""
    from ops.jobs import rwprep_remote as R
    import shutil
    import sys
    import time
    from types import SimpleNamespace
    p=tmp_path/'raw';p.write_bytes(b'raw')
    spec=dict(deadline=time.time()+600,code='/code',commit='abc',worker_job='/configs/job.json',
              stop_path=str(tmp_path/'STOP'),storage_root=str(tmp_path/'rw'),
              inputs={str(p):R.digest(p)},slot_guard=dict(uuid='MIG-x',ram_bytes=48*R.GIB,reserve_bytes=120*R.GIB))
    if problem=='input':p.write_bytes(b'changed')
    if problem=='stop':Path(spec['stop_path']).touch()
    job=dict(name='named',episode='ep',gpu='MIG-x',docker_args=['docker','run','WANDB_MODE=disabled','old_worker','old_job'])
    calls=[];guards=[]
    def run(*args):
        calls.append(args)
        if args[:2]==('docker','ps'):
            return 'named' if problem=='duplicate' and '--format' in args else ''
        if args[:2]==('git','-C'):
            if problem=='stop_after_verify':Path(spec['stop_path']).touch()
            return 'abc' if 'rev-parse' in args else ''
        if args[:2]==('docker','run'):return 'cid'
        raise AssertionError(args)
    def decide(data):
        guards.append(dict(data))
        assert data['reserve_bytes']==200*R.GIB and data['disk_floor_fraction']==.155
        return {'admit':problem not in ('quota','ram'),'reasons':[problem] if problem in ('quota','ram') else []}
    monkeypatch.setattr(shutil,'disk_usage',lambda _:SimpleNamespace(free=1 if problem=='data_disk' else 100,total=100))
    monkeypatch.setattr(sys,'argv',['guard',json.dumps(job)])
    env=dict(ROUTE_SPEC=spec,run=run,decide=decide,json=json,sys=sys,Path=Path,
             protect_processed=R.protect_processed)
    if problem in ('quota','ram','data_disk','duplicate','stop','stop_after_verify'):
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


@pytest.mark.parametrize('stop_during', ['launch', 'snapshot', 'decision'])
def test_remote_stop_drains_current_batch_offline(tmp_path, monkeypatch, stop_during):
    from types import SimpleNamespace
    from ops.jobs import rwprep_remote as R
    from ops.jobs import release_reservation
    from ops.resources.gpu_guard import REMOTE_MIG_DEVICES
    jobs = [dict(job(), id=job()['id']+str(n)) for n in range(2)]
    config = dict(host='remote', image='sha256:'+'a'*64, commit='b'*40,
                  jobs=str(tmp_path/'jobs.json'), state_dir=str(tmp_path/'state'),
                  priority_file=str(tmp_path/'priority'), remote='/remote')
    P.atomic(config['jobs'], jobs)
    stop = tmp_path/'state/STOP'
    calls = []
    def snapshot(_):
        if stop_during == 'snapshot': stop.touch()
        return {}
    def decide(*a, **kw):
        if stop_during == 'decision': stop.touch()
        return {'candidates':list(REMOTE_MIG_DEVICES), 'host':[]}
    parallel = SimpleNamespace(load=lambda *a: ({}, ''), snapshot=snapshot, decide=decide)
    monkeypatch.setattr(R, 'established', lambda: (None, parallel))
    monkeypatch.setattr(R, 'guard_payload', lambda *a, **kw: {'input':{'cpu_thread_budget':320}})
    monkeypatch.setattr(release_reservation, 'reserved_gpus', lambda: set())
    def launch(*a):
        calls.append(a[2]['id'])
        if stop_during == 'launch': stop.touch()
    monkeypatch.setattr(R, 'launch_job', launch)
    client = SimpleNamespace(checked_ssh=lambda *a, **kw: SimpleNamespace(stdout=json.dumps([j['id'] for j in jobs])))
    result = R.tick(config, client, apply=True)
    assert calls == ([jobs[0]['id']] if stop_during == 'launch' else [])
    assert result['counts']['pending'] == (1 if calls else 2)


def test_processed_mount_guard_confines_cleanup_and_rejects_aliases(tmp_path):
    from ops.jobs.rwprep_remote import protect_processed
    root = tmp_path/'rw'
    old = root/'processed/previous-release'
    old.mkdir(parents=True)
    (old/'sentinel').write_text('keep')
    outside = tmp_path/'outside'
    outside.mkdir()
    broad = f'type=bind,src={root}/processed,dst={root}/processed'
    args = protect_processed(['docker','run','--mount',broad,'image'],str(root))
    mounts = [args[n+1] for n,x in enumerate(args) if x=='--mount']
    assert broad+',readonly' in mounts
    target = root/'processed/dsreset-20261113'
    assert mounts.count(f'type=bind,src={target},dst={target}') == 1
    # The only writable processed mount is the fresh release, including when
    # inherited from the planner's wider realworld mount. Every sibling stays RO.
    for path in (old, outside, root/'processed/dsreset-202611130'):
        assert not path.is_relative_to(target)
    for destination in (old, outside):
        target.rmdir()
        target.symlink_to(destination, target_is_directory=True)
        with pytest.raises(ValueError, match='alias'):
            protect_processed(['docker','run','image'],str(root))
        target.unlink()
        target.mkdir()
    assert (old/'sentinel').read_text() == 'keep'


@pytest.mark.parametrize('admit', [True, False])
def test_planner_uses_locked_shared_admission_and_owned_accounting(tmp_path, monkeypatch, admit):
    import shlex
    import sys
    import subprocess
    from types import SimpleNamespace
    from ops.jobs import rwprep_remote as R
    from ops.remote import launch_hold
    queue = tmp_path/'queue.yaml'
    queue.write_text('hosts:\n  remote:\n    cpu_thread_budget: 42\n    host_ram_reserve_gb: 220\n')
    monkeypatch.setattr(R, 'LIVE_QUEUE', queue)
    monkeypatch.setattr(R, 'RW', str(tmp_path/'rw'))
    monkeypatch.setattr(launch_hold, 'check_shell', lambda: ':')
    commands = []
    monkeypatch.setattr(launch_hold, 'locked', lambda body: commands.append(body) or body)
    launched = []
    def checked(command, **kw):
        assert command == commands[-1] and 'python3 -I -c' in command
        source = shlex.split(command)[-1]
        library, body = source.split('\ndata=',1)
        ns = {}
        exec(library, ns)
        def decide(data):
            assert data['host_only'] and data['cpu_threads']==2 and data['cpu_thread_budget']==42
            assert data['ram_bytes']==8*R.GIB and data['reserve_bytes']==220*R.GIB
            return {'admit':admit, 'reasons':[] if admit else ['host CPU thread budget']}
        ns['decide'] = decide
        exec('\ndata='+body, ns)
        return SimpleNamespace(stdout='planned')
    monkeypatch.setattr(subprocess, 'run', lambda args, **kw: launched.append(args))
    client = SimpleNamespace(checked_ssh=checked)
    config = dict(code='/code', image='sha256:'+'a'*64, commit='b'*40)
    if admit:
        assert R.planner(client,config,['-m','planner']) == 'planned'
        args = launched[0]
        labels = dict(args[n+1].split('=',1) for n,x in enumerate(args) if x=='--label')
        assert labels['s4d.project']=='splatter4d' and labels['s4d.owner']=='kaist-sunho'
        assert labels['s4d.managed']=='false' and labels['s4d.role']=='rwprep'
        assert labels['s4d.disk_bytes']==str(32*R.GIB) and labels['s4d.ram_bytes']==str(8*R.GIB)
    else:
        with pytest.raises(SystemExit) as exc:R.planner(client,config,['-m','planner'])
        assert exc.value.code==20 and not launched


def test_local_stop_after_slow_worker_upload_prevents_launch(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from ops.jobs import rwprep_remote as R
    state = tmp_path/'state'
    state.mkdir()
    jobs = tmp_path/'jobs.json'
    jobs.write_text('[]')
    calls = []
    def push(*a):
        (state/'STOP').touch()
    launcher = SimpleNamespace(push_bytes=push, launch=lambda *a:calls.append(a))
    monkeypatch.setattr(R, 'established', lambda:(launcher,None))
    config = dict(jobs=str(jobs),state_dir=str(state),remote='/remote')
    with pytest.raises(RuntimeError,match='STOP'):
        R.launch_job(None,config,job(),{'attempt':1})
    assert not calls
