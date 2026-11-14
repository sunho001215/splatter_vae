"""Container-side bridge to the established remote rwprep launcher; never local Docker.

Immutable candidate job bundles, durable local intents, shared scheduler MIG guard,
remote boot hold and both remote launch locks. No campaign driver imports.
"""
from __future__ import annotations
import argparse
import fcntl
import importlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import tempfile
import time

from ops.jobs.rwprep_mig import atomic, digest, validate, priority_path
from ops.containers.slot_guard import SOURCE
from ops.containers.rwprep_capacity import guard_payload
from ops.remote.access import RemoteClient

LIVE_QUEUE = Path('/home/ws/ws/splatter4d/experiments/queue.yaml')

REPO = Path(__file__).resolve().parents[2]
RW = '/home/compu/kaist/sunho/realworld'
RAW = '/home/compu/kaist/sunho/data/real_world'
TOOLS = Path('/home/ws/ws/realworld-ops/pre/tools')
IMAGE = 's4d-rwprep-f7e075f51c79:20261106'
GIB = 2**30


def established():
    sys.path.insert(0, str(TOOLS))
    return importlib.import_module('rwprep_launch'), importlib.import_module('rwprep_parallel')


def rewrite(value, old, new):
    if isinstance(value, str):
        return value.replace(str(old), str(new))
    if isinstance(value, list):
        return [rewrite(x, old, new) for x in value]
    if isinstance(value, dict):
        return {rewrite(k, old, new): rewrite(v, old, new) for k, v in value.items()}
    return value


def prepare(plans, out, client, commit=None):
    """Compile without canonical local aliases, stage in two batched uploads, launch nothing."""
    import yaml
    from scripts.realworld_prep_register import compile_plans
    from ops.resources.gpu_guard import REMOTE_MIG_DEVICES
    out = Path(out).resolve()
    commit = commit or subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
    remote = RW + '/configs/' + out.name
    if not out.name.startswith('dsreset-'):
        raise ValueError('bundle name must start with dsreset-')
    image = client.checked_ssh('docker image inspect --format '+shlex.quote('{{.Id}}')+' '+IMAGE,
                               idempotent=True).stdout.strip()
    jobs = compile_plans(plans, out, RW+'/processed/dsreset-20261113', RAW)
    for job in jobs:
        job['priority'] = -4
        path = next(Path(x) for x in job['inputs'] if x.endswith('.yaml'))
        # The server path emitted by the compiler is replaced along with the config paths.
        job.update(rewrite(job, out, remote))
        job['inputs'][str(path).replace(str(out),remote)] = digest(path)
    (out/'state').mkdir()
    server = yaml.safe_load((TOOLS/'rwprep/configs/rwprep_server.yaml').read_text())
    server.update(gpu_allowlist=list(REMOTE_MIG_DEVICES),
                  mig_parents={u:v[0] for u,v in REMOTE_MIG_DEVICES.items()})
    (out/'state/server.yaml').write_text(yaml.safe_dump(server))
    launcher, _ = established()
    tools_id = launcher.push_tools(client)
    code = RW+'/code/dsreset-'+commit
    config = dict(host='remote', image=image, commit=commit, jobs=str(out/'jobs.json'),
                  state_dir=str(out/'state'), remote=remote, code=code, tools_id=tools_id,
                  priority_file=priority_path('/home/ws/ws/splatter4d/runs/setup/rwprep-mig-priority.d',out/'state'))
    validate(config,jobs)
    atomic(out/'jobs.json',jobs)
    # Keep a local plan for inspection; remote copy uses its own canonical config paths.
    atomic(out/'prep-plan.json',rewrite(json.loads((out/'prep-plan.json').read_text()),out,remote))
    atomic(out/'dispatcher.json',config)
    remote_config={**config,'jobs':remote+'/jobs.json','state_dir':remote+'/state',
        'server':remote+'/state/server.yaml','bundle':remote,'processed':RW+'/processed/dsreset-20261113',
        'raw':RAW,'weights':RW+'/weights','tools':RW+'/tools/'+tools_id,
        'assets':RW+'/assets/validation-20261031','extra_readonly':[]}
    atomic(out/'remote-dispatcher.json',remote_config)
    # A shallow ordinary .git is required by LocalServer.verify_source, without worktree pointers.
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        clone = Path(tmp)/'code'
        subprocess.run(['git','clone','--quiet','--depth','1','--no-local',
                        '--branch',subprocess.check_output(['git','-C',str(REPO),'branch','--show-current'],text=True).strip(),
                        REPO.as_uri(),str(clone)],check=True)
        if subprocess.check_output(['git','-C',str(clone),'rev-parse','HEAD'],text=True).strip()!=commit:
            raise ValueError('source branch moved while staging')
        # Install already-local pinned submodules without network, retaining ordinary git metadata.
        for name in ('dino','snerl','dynamo','rvt'):
            source = REPO/'third_party'/name
            if (source/'.git').exists():
                dest = clone/'third_party'/name
                subprocess.run(['git','clone','--quiet','--no-hardlinks',str(source),str(dest)],check=True)
                pin = subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD:third_party/'+name],text=True).strip()
                subprocess.run(['git','-C',str(dest),'checkout','--quiet','--detach',pin],check=True)
        archive=out/'code.tar'
        with tarfile.open(archive,'w') as tar:
            tar.add(clone,arcname='.')
        publish_archive(client,archive,code,commit)
    archive=out/'bundle.tar'
    with tarfile.open(archive,'w') as tar:
        for name in ('configs','state/server.yaml','jobs.json','prep-plan.json'):
            tar.add(out/name,arcname=name)
        tar.add(out/'remote-dispatcher.json',arcname='dispatcher.json')
    publish_archive(client,archive,remote)
    return {'registered':len(jobs),'launched':False,'config':str(out/'dispatcher.json')}


def publish_archive(client, archive, destination, expected_commit=''):
    """Fresh immutable publication. A failed/stale target is never overwritten."""
    client.ensure_remote_path(destination,create_parent=True)
    token=digest(archive)
    incoming=RW+'/incoming/'+token+'.tar'
    client.checked_ssh('mkdir -p '+shlex.quote(RW+'/incoming'),idempotent=True)
    client.rsync_upload(archive,incoming)
    script="""import hashlib, pathlib, subprocess, sys
src,dest,sha,commit=sys.argv[1:]
p=pathlib.Path(dest)
assert hashlib.sha256(pathlib.Path(src).read_bytes()).hexdigest()==sha
p.parent.mkdir(parents=True,exist_ok=True)
if p.exists():
    if commit:
        assert subprocess.check_output(['git','-C',dest,'rev-parse','HEAD'],text=True).strip()==commit
        assert not subprocess.check_output(['git','-C',dest,'status','--porcelain','--untracked-files=no'],text=True).strip()
    else:
        assert (p/'.route-archive.sha256').read_text()==sha
    sys.exit(0)
p.mkdir(exist_ok=False)
subprocess.run(['tar','-xf',src,'-C',dest],check=True)
(p/'.route-archive.sha256').write_text(sha)
"""
    client.checked_ssh('python3 -I -c '+shlex.quote(script)+' '+shlex.join([incoming,destination,token,expected_commit]))


# Reuse exactly the scheduler's tenant validation and resource arithmetic. The
# appended wrapper is executed under admission.lock AND realworld/state/launch.lock.
GUARD_LIBRARY = SOURCE.rsplit('\ntry:\n',1)[0]
GUARD_WRAPPER = r'''
import hashlib
import shutil
import time
job=json.loads(sys.argv[1])
spec=ROUTE_SPEC
problems=[]
if time.time()>spec['deadline']:
    problems.append('expired launch intent')
if job['name'] in run('docker','ps','-a','--format','{{.Names}}').split():
    problems.append('container name already exists')
if run('docker','ps','-q','--filter','label=s4d.role=rwprep','--filter',
       'label=s4d.rwprep.episode='+job['episode']).split():
    problems.append('episode already has a live rwprep writer')
data=spec['slot_guard']
assert data['uuid']==job['gpu'] and data['uuid'].startswith('MIG-')
data.update(reserve_bytes=max(data['reserve_bytes'],200*2**30),disk_floor_fraction=.155)
verdict=decide(data)
problems+=verdict.get('reasons',[])
# Existing campaign RAM cap and secondary-disk floor, independent of global load/PSI.
ids=run('docker','ps','-q','--filter','label=s4d.project=splatter4d',
        '--filter','label=s4d.owner=kaist-sunho').split()
charged=0
for cid in ids:
    item=json.loads(run('docker','inspect',cid))[0]
    cap=item['HostConfig']['Memory']
    if cap<=0: raise ValueError('unbounded campaign RAM co-tenant')
    charged+=cap
if charged+data['ram_bytes']>1000*2**30:
    problems.append('campaign RAM cap')
disk=shutil.disk_usage('/data_8T')
if disk.free/disk.total<.09: problems.append('/data_8T free floor')
if problems:
    print(json.dumps({'refused':problems})); sys.exit(20)
for path,sha in spec['inputs'].items():
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    if h.hexdigest()!=sha: raise ValueError('input changed: '+path)
if run('git','-C',spec['code'],'rev-parse','HEAD').strip()!=spec['commit']:
    raise ValueError('source commit changed')
if run('git','-C',spec['code'],'status','--porcelain','--untracked-files=no').strip():
    raise ValueError('dirty source')
# Input verification can take time; recheck the deadline immediately before launch.
if time.time()>spec['deadline']:
    print(json.dumps({'refused':['expired launch intent']}));sys.exit(20)
args=job['docker_args']
for path in spec.get('extra_readonly',[]):
    assert path.startswith('/home/compu/kaist/sunho/realworld/')
    args[2:2]=['--mount','type=bind,src='+path+',dst='+path+',readonly']
args[-2:]=['/opt/src/splatter4d/scripts/realworld_prep_worker.py',spec['worker_job']]
# The established launcher currently sets disabled: this campaign requires offline W&B.
args=[x.replace('WANDB_MODE=disabled','WANDB_MODE=offline') for x in args]
verdict=decide(data)  # fresh after potentially slow SHA verification
if not verdict['admit']:
    print(json.dumps({'refused':verdict.get('reasons',[])}));sys.exit(20)
cid=run(*args).strip()
print(json.dumps({'container_id':cid,'decision':verdict}))
'''


def launch_job(client, config, job, intent):
    launcher,_=established()
    qsha=digest(config['jobs'])
    worker={**job,'attempt':intent['attempt'],'queue_sha256':qsha,
            'result_dir':RW+'/jobs/'+job['id'],'server':config['remote']+'/state/server.yaml'}
    worker_path=config['remote']+'/configs/'+job['id']+'-a'+str(intent['attempt'])+'.json'
    launcher.push_bytes(client,json.dumps(worker).encode(),worker_path)
    spec=dict(inputs=job['inputs'],code=config['code'],commit=config['commit'],
              worker_job=worker_path,deadline=intent['deadline'],extra_readonly=config.get('extra_readonly',[]),
              slot_guard=guard_payload(dict(gpu=intent['gpu'],gpu_need_bytes=int(job['gpu_gib']*GIB),
                  ram_bytes=int(job['ram_gib']*GIB),cpus=job['cpus'],disk_bytes=int(job.get('disk_gib',32)*GIB)),
                  queue_path=LIVE_QUEUE)['input'])
    # Use a private guard file, never modify the stopped legacy dispatcher's tools.
    with tempfile.TemporaryDirectory() as temp:
        guard=Path(temp)/'guard.py'
        guard.write_text(GUARD_LIBRARY+'\nROUTE_SPEC='+repr(spec)+'\n'+GUARD_WRAPPER)
        old=launcher.GUARD_SOURCE
        launcher.GUARD_SOURCE=guard
        try:
            return launcher.launch(client,dict(job_id=job['id'],attempt=intent['attempt'],gpu=intent['gpu'],
                image=config['image'],commit=config['commit'],code_dir=config['code'],tools=config['tools_id'],
                episode_key=job.get('episode',job['id']),gpu_need_bytes=int(job['gpu_gib']*GIB),
                ram_bytes=int(job['ram_gib']*GIB),cpus=job['cpus'],threads=job['cpus'],
                disk_bytes=int(job.get('disk_gib',32)*GIB),tmpfs='16g',steps=[],
                env={'S4D_TE_CUDA_MEMORY_CAP_GIB':'5.5'}))
        finally:
            launcher.GUARD_SOURCE=old


def remote_observe(client, names):
    script='''
import json, pathlib, subprocess, sys
out={}
for name,path in json.loads(sys.argv[1]).items():
    found=subprocess.run(['docker','ps','-aq','--filter','name=^/'+name+'$'],check=True,capture_output=True,text=True).stdout.split()
    if not found:out[name]=None;continue
    item=json.loads(subprocess.run(['docker','inspect',found[0]],check=True,capture_output=True,text=True).stdout)[0]
    p=pathlib.Path(path)
    out[name]={'labels':item['Config']['Labels'],'state':item['State'],
               'proof':json.loads(p.read_text()) if p.is_file() else {}}
print(json.dumps(out))
'''
    return json.loads(client.checked_ssh('python3 -I -c '+shlex.quote(script)+' '+shlex.quote(json.dumps(names)),
                                        idempotent=True).stdout)


def reconcile(states, observed, config, qsha, now):
    for jid,s in states.items():
        if s['status'] not in ('launching','running'):continue
        item=observed.get(s['container'])
        if item is None:
            # A lost SSH reply may still be executing. Wait through the remote
            # launch deadline + clock skew before allowing a retry.
            if s['status']=='running':s.update(status='unknown',reason='running container vanished')
            elif now>s['deadline']+300:s['status']='retry' if s['attempt']<3 else 'failed'
            continue
        labels=item['labels']
        if any(labels.get(k)!=v for k,v in {'s4d.commit':config['commit'],'s4d.rwprep.job':jid,
                's4d.gpu':s['gpu'],'s4d.role':'rwprep','s4d.owner':'kaist-sunho'}.items()):
            raise ValueError('container identity differs from durable intent')
        running=item['state']
        if running['Running'] or running.get('Paused') or running.get('Restarting'):
            s['status']='running'
        else:
            ok=running['ExitCode']==0 and item['proof'].get('status')=='ok' and item['proof'].get('queue_sha256')==qsha
            s['status']='done' if ok else ('retry' if s['attempt']<3 else 'failed')
            s['exit_code']=running['ExitCode']


def tick(config, client, apply=False, max_launch=16):
    from ops.jobs import release_reservation
    from ops.resources.gpu_guard import REMOTE_MIG_DEVICES
    root=Path(config['state_dir']);root.mkdir(exist_ok=True,parents=True)
    with (root/'dispatcher.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        jobs=json.loads(Path(config['jobs']).read_text());validate(config,jobs)
        sha=digest(config['jobs']);path=root/'state.json'
        state=json.loads(path.read_text()) if path.exists() else {'queue_sha256':sha,'jobs':{}}
        if state['queue_sha256']!=sha:raise ValueError('immutable queue changed')
        states=state['jobs']
        for j in jobs:states.setdefault(j['id'],{'status':'pending','attempt':0})
        names={s['container']:RW+'/jobs/'+jid+'/a'+str(s['attempt'])+'/result.json'
               for jid,s in states.items() if s['status'] in ('launching','running')}
        if names:reconcile(states,remote_observe(client,names),config,sha,time.time())
        pending=[j for j in sorted(jobs,key=lambda x:x.get('priority',-4))
                 if states[j['id']]['status'] in ('pending','retry')
                 and all(states[d]['status']=='done' for d in j.get('deps',[]))]
        # Artifact gates reside remotely. Query once; no canonical local mounts required.
        if pending:
            script='''
import json,pathlib,sys
out=[]
for j in json.loads(sys.argv[1]):
    if not all(pathlib.Path(p).is_file() for p in j.get('requires',[])):continue
    if not all(pathlib.Path(p).is_file() and all(json.loads(pathlib.Path(p).read_text())['jobs'].get(i,{}).get('status')=='done' for i in ids) for p,ids in j.get('external_states',{}).items()):continue
    out.append(j['id'])
print(json.dumps(out))
'''
            ready=set(json.loads(client.checked_ssh('python3 -I -c '+shlex.quote(script)+' '+shlex.quote(json.dumps([
                {k:j[k] for k in ('id','requires','external_states') if k in j} for j in pending])),idempotent=True).stdout))
            pending=[j for j in pending if j['id'] in ready]
        stopped=(root/'STOP').exists()
        uncertain=any(s['status']=='unknown' for s in states.values())
        active=bool((pending and not stopped) or any(s['status'] in ('running','launching','unknown') for s in states.values()))
        if apply:atomic(config['priority_file'],dict(active=active,host='remote',priority=-4,state_dir=str(root)))
        launched=[];blocked={}
        if apply:atomic(path,state)
        if not stopped and not uncertain:
            _,parallel=established()
            policy,why=parallel.load(TOOLS.parent/'prep_parallel.json','calrel-6c3b9ec-prep')
            if policy is None:raise ValueError('established resource policy unavailable: '+str(why))
            snap=parallel.snapshot(client) if pending else None
            for j in pending:
                if len(launched)>=max_launch:break
                # No global load/idle/PSI admission: unrelated users cannot consume
                # our quota budget. Preserve every RAM and disk check in decide().
                decision=parallel.decide({**snap,'psi_cpu_some_avg60':0},policy,list(REMOTE_MIG_DEVICES),
                    int(j['gpu_gib']*GIB),int(j['ram_gib']*GIB),j['cpus'],int(j.get('disk_gib',32)*GIB),
                    cpu_rule='quota',cpu_budget_threads=guard_payload(dict(gpu=next(iter(REMOTE_MIG_DEVICES)),
                        gpu_need_bytes=1,ram_bytes=1,cpus=1),queue_path=LIVE_QUEUE)['input']['cpu_thread_budget'])
                candidates=[g for g in decision['candidates'] if g not in release_reservation.reserved_gpus()]
                if not candidates:
                    blocked[j['id']]=decision['host'] or 'no fitting scheduler-accounted MIG';break
                if not apply:
                    launched.append({'would_launch':j['id'],'candidates':candidates});continue
                s=states[j['id']];attempt=s['attempt']+1
                s.update(status='launching',attempt=attempt,gpu=candidates[0],deadline=time.time()+600,
                         container='s4d-rwprep-'+j['id']+'-a'+str(attempt))
                atomic(path,state)
                try:
                    launch_job(client,config,j,s)
                except (Exception,SystemExit) as exc:
                    # Includes confirmed refusal: next tick reconciles exact name;
                    # no blind retry on ambiguous remote transport failures.
                    s['launch_error']=type(exc).__name__;blocked[j['id']]=str(exc)[:300]
                    atomic(path,state);break
                s['status']='running';launched.append(j['id']);atomic(path,state)
                snap=parallel.snapshot(client)  # charge newly enforced quotas before next launch
        result={'launched':launched,'blocked':blocked,'counts':{status:sum(s['status']==status for s in states.values())
                for status in ('pending','launching','running','done','retry','failed','unknown')}}
        if apply:
            state['last_tick']=result;atomic(path,state)
            # Calibration's external dependency gates read this mirror. A failed
            # publication delays them, never marks unfinished work complete.
            script='import pathlib,sys; p=pathlib.Path(sys.argv[1]); t=p.with_suffix(".tmp"); t.write_text(sys.argv[2]); t.replace(p)'
            client.checked_ssh('python3 -I -c '+shlex.quote(script)+' '+shlex.quote(config['remote']+'/state/state.json')+' '+shlex.quote(json.dumps(state)),idempotent=True)
        return result


def attach(remote, out, client, base):
    """Attach an already compiled remote candidate bundle; launch nothing."""
    if not remote.startswith(RW+'/configs/dsreset-'):
        raise ValueError('expected isolated dsreset bundle')
    script="import json,pathlib,sys; p=pathlib.Path(sys.argv[1]); print(json.dumps({n:json.loads((p/n).read_text()) for n in ('dispatcher.json','jobs.json')}))"
    docs=json.loads(client.checked_ssh('python3 -I -c '+shlex.quote(script)+' '+shlex.quote(remote),idempotent=True).stdout)
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False);(out/'state').mkdir()
    jobs=docs['jobs.json']
    for j in jobs:j['priority']=-4
    if docs['dispatcher.json']['commit']!=base['commit']:raise ValueError('source pin mismatch')
    config={**docs['dispatcher.json'],'code':base['code'],'tools_id':base['tools_id'],'image':base['image'],'host':'remote','remote':remote,'jobs':str(out/'jobs.json'),
            'state_dir':str(out/'state'),
            'priority_file':priority_path('/home/ws/ws/splatter4d/runs/setup/rwprep-mig-priority.d',out/'state')}
    validate(config,jobs);atomic(out/'jobs.json',jobs);atomic(out/'dispatcher.json',config)
    return {'registered':len(jobs),'launched':False,'config':str(out/'dispatcher.json')}


def planner(client,config,argv):
    """Only explicit coordinator planning commands; no GPU, bounded remote container."""
    from ops.remote import launch_hold
    args=['docker','run','--rm','--read-only','--network','none','--user','1000:1000',
          '--memory','8g','--memory-swap','8g','--cpus','2','--pids-limit','512',
          '--tmpfs','/tmp:rw,size=2g,mode=1777','--env','NVIDIA_VISIBLE_DEVICES=void',
          '--env','PYTHONDONTWRITEBYTECODE=1','--env','WANDB_MODE=offline',
          '--env','HOME=/tmp','--env','PYTHONPATH=/opt/src/splatter4d',
          '--mount','type=bind,src='+config['code']+',dst=/opt/src/splatter4d,readonly',
          '--mount','type=bind,src='+RW+',dst='+RW,
          '--mount','type=bind,src='+RAW+',dst='+RAW+',readonly',
          '--workdir','/opt/src/splatter4d','--entrypoint','/opt/s4d/.venv/bin/python',
          config['image'],*argv]
    # No restart policy, service or host changes. Explicit planning is not a GPU job.
    command=launch_hold.locked('set -eu; '+launch_hold.check_shell()+'; '+shlex.join(args))
    return client.checked_ssh(command,timeout=300).stdout


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--plan',type=Path,action='append',required=True);prep.add_argument('--out',type=Path,required=True)
    run=sub.add_parser('tick');run.add_argument('--config',type=Path,required=True);run.add_argument('--apply',action='store_true');run.add_argument('--loop',action='store_true');run.add_argument('--max-launch',type=int,default=16)
    attach_p=sub.add_parser('attach');attach_p.add_argument('--remote',required=True);attach_p.add_argument('--out',type=Path,required=True);attach_p.add_argument('--base-config',type=Path,required=True)
    plan_p=sub.add_parser('plan');plan_p.add_argument('--config',type=Path,required=True);plan_p.add_argument('argv',nargs=argparse.REMAINDER)
    a=p.parse_args();client=RemoteClient()
    if a.command=='prepare':print(json.dumps(prepare(a.plan,a.out,client)));return
    if a.command=='attach':print(json.dumps(attach(a.remote,a.out,client,json.loads(a.base_config.read_text()))));return
    if a.command=='plan':
        argv=a.argv[1:] if a.argv[:1]==['--'] else a.argv
        print(planner(client,json.loads(a.config.read_text()),argv));return
    while True:
        try:print(json.dumps(tick(json.loads(a.config.read_text()),client,a.apply,a.max_launch)),flush=True)
        except Exception as exc:
            if not a.loop:raise
            print(json.dumps({'deferred':type(exc).__name__,'reason':str(exc)[:300]}),flush=True)
        if not a.loop:break
        time.sleep(15)


if __name__=='__main__':main()
