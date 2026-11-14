"""Host-local, non-preemptive MIG preprocessing queue. No SSH or training imports.

Adapted from realworld-ops/pre/tools/rwprep_dispatch.py: durable intents, named
containers, bounded retries, ordinary fingerprinted stages, STOP and locked ticks.
Run on the executing host; a tick defaults to inspection, --apply enables launch.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
GIB = 2**30
ROLE = 'rwprep'  # established scheduler co-tenant class; never scheduler-managed


def run(argv):
    return subprocess.run(list(map(str, argv)), check=True, capture_output=True, text=True, timeout=180).stdout


def atomic(path, doc):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with tmp.open('w') as stream:
        json.dump(doc, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def priority_path(directory,state_dir):
    return str(Path(directory)/(hashlib.sha256(str(state_dir).encode()).hexdigest()[:24]+'.json'))


def validate(config, jobs):
    if config['host'] not in ('local', 'remote') or not re.fullmatch('sha256:[0-9a-f]{64}', config['image']):
        raise ValueError('host and immutable local Docker image ID required')
    if not re.fullmatch('[0-9a-f]{40}', config['commit']):
        raise ValueError('full source commit required')
    if len({j['id'] for j in jobs}) != len(jobs):
        raise ValueError('duplicate queue identity')
    for job in jobs:
        if (not re.fullmatch(r'rwv2-20261113-[A-Za-z0-9_.-]+', job['id']) or len(job['id']) > 180
                or not 0 < job['gpu_gib'] <= 21.5 or not 0 < job['ram_gib'] <= 96
                or not 0 < job['cpus'] <= 16 or job.get('gpu_profile') != '1g.24gb'
                or not job['steps'] or not job.get('inputs')):
            raise ValueError('invalid MIG job or missing hash-bound inputs')
        if set(job.get('deps', [])) - {j['id'] for j in jobs}:
            raise ValueError('unknown job dependency')
        for path, sha in job['inputs'].items():
            if not Path(path).is_absolute() or not re.fullmatch('[0-9a-f]{64}', sha):
                raise ValueError('inputs must use absolute paths and sha256')


def ready(jobs, state, stopped=False):
    return [] if stopped else [j for j in jobs if
        state.get(j['id'], {}).get('status', 'pending') in ('pending', 'retry')
        and all(state.get(d, {}).get('status') == 'done' for d in j.get('deps', []))
        and all(Path(p).is_file() for p in j.get('requires', []))
        and all(Path(p).is_file() and all(json.loads(Path(p).read_text())['jobs'].get(i,{}).get('status')=='done'
                for i in ids) for p,ids in j.get('external_states',{}).items())]


def admission(job, snap, config):
    """Conservative full-cap accounting also covers launch-time growth and tmpfs."""
    ram, cpu, disk = job['ram_gib']*GIB, job['cpus'], job.get('disk_gib', 32)*GIB
    reasons = []
    cap = min(config.get('ram_cap_bytes', 138_000_000_000),
              138_000_000_000 if config['host'] == 'local' else 1000*GIB)
    if snap['charge'] + snap['growth'] + snap['prep_ram'] + ram > cap:
        reasons.append('campaign RAM cap/growth')
    if snap['available'] < config.get('host_reserve_bytes', 60*GIB) + snap['prep_ram'] + ram:
        reasons.append('host RAM reserve')
    if snap['cpu_used'] + cpu > snap['cpus'] or snap['load'] + cpu > .85*snap['cpus']:
        reasons.append('CPU quota/load')
    if snap['idle'] < .2 or snap['psi_cpu'] > 75:
        reasons.append('CPU idle/pressure')
    if snap['disk_free'] - snap['prep_disk'] - disk < max(32*GIB, .15*snap['disk_total']):
        reasons.append('disk floor/transient growth')
    devices = [u for u, d in snap['devices'].items() if u not in snap['busy']
               and not d['processes'] and d['used_mib'] <= 256
               and d['total_mib'] >= 23000 and (d['total_mib']-d['used_mib'])*1048576 >= (job['gpu_gib']+1)*GIB]
    if not devices:
        reasons.append('no idle fitting 1g.24gb MIG')
    return (None if reasons else devices[0]), reasons


def snapshot(config, live):
    from ops.resources.gpu_guard import MIG_DEVICES, REMOTE_MIG_DEVICES
    from ops.resources.nvidia_query import mig_query
    from ops.resources import memory_budget as mb
    from ops.jobs import scheduler
    from ops.jobs.footprints import ram_gb
    inventory = MIG_DEVICES if config['host'] == 'local' else REMOTE_MIG_DEVICES
    from ops.resources.mig_inventory import parse_list
    listed={m['uuid']:(p['uuid'],m['device'],m['profile']) for p in parse_list(run(['nvidia-smi','-L'])) for m in p['mig']}
    if any(listed.get(u)!=(v[0],v[2],'1g.24gb') for u,v in inventory.items()):
        raise ValueError('live MIG UUID/profile/parent differs from accepted inventory')
    devices = mig_query(inventory, state_dir=Path(config['state_dir']), timeout=60)
    memory = mb.snapshot()
    queue = scheduler.load_queue(Path(config['queue']))
    registry = scheduler.read_registry(Path(config['registry']))
    rows = {j['id']: j for j in queue['jobs']}
    pss = mb.session_nonfile_pss()
    growth = sum(max(0, ram_gb(rows.get(i, s))*GIB-pss.get(int(s.get('pid', -1)), 0))
                 for i, s in registry.items() if s.get('host', 'local') == config['host']
                 and s['status'] in scheduler.RESERVING)
    cpu = sum(float(rows.get(i, s).get('cpu_threads', 8)) for i, s in registry.items()
              if s.get('host', 'local') == config['host'] and s['status'] in scheduler.RESERVING)
    cpu1 = list(map(int, Path('/proc/stat').read_text().splitlines()[0].split()[1:]))
    time.sleep(.1)
    cpu2 = list(map(int, Path('/proc/stat').read_text().splitlines()[0].split()[1:]))
    delta = [b-a for a, b in zip(cpu1[:8], cpu2[:8])]
    idle = (delta[3]+delta[4])/max(1, sum(delta))
    psi = mb._pressure(Path('/proc/pressure/cpu'))['some']['avg60']
    disk = shutil.disk_usage(config['processed'])
    # Full container limits are deliberately counted even when their current
    # resident memory is already in memory.current: safe conservative admission.
    other_ram=other_cpu=0
    own_names={f"s4d-{j['id']}-a" for j in live}
    cids=run(['docker','ps','-q','--filter','label=s4d.project=splatter4d','--filter','label=s4d.owner=kaist-sunho']).split()
    for cid in cids:
        item=json.loads(run(['docker','inspect',cid]))[0]
        if any(item['Name'].lstrip('/').startswith(n) for n in own_names):
            continue
        cap=item['HostConfig']['Memory'];quota=item['HostConfig'].get('NanoCpus',0)/1e9
        if cap<=0 or quota<=0:
            raise ValueError('unbounded campaign co-tenant; no safe growth admission')
        other_ram+=cap;other_cpu+=quota
    return dict(devices=devices, charge=memory['budget_usage'], growth=growth,
                available=memory['host_mem_available'], cpu_used=cpu+other_cpu+sum(x['cpus'] for x in live),
                prep_ram=other_ram+sum(x['ram_gib']*GIB for x in live), prep_disk=sum(x.get('disk_gib',32)*GIB for x in live),
                busy={x['gpu'] for x in live}, load=os.getloadavg()[0], idle=idle, psi_cpu=psi,
                cpus=int(queue['hosts'][config['host']].get('cpu_thread_budget', os.cpu_count() or 1)),
                disk_free=disk.free, disk_total=disk.total)


def container_command(config, job, attempt, gpu, jobfile):
    name = f"s4d-{job['id']}-a{attempt}"
    argv = ['docker','run','-d','--name',name,'--read-only','--init','--pid=host',
            '--network','none','--security-opt','no-new-privileges','--cap-drop','ALL',
            '--user',str(config.get('uid',1000))+':'+str(config.get('gid',1000)),
            '--memory',str(int(job['ram_gib']*GIB)),'--memory-swap',str(int(job['ram_gib']*GIB)),
            '--cpus',str(job['cpus']),'--pids-limit','4096','--shm-size','2g',
            '--tmpfs','/ram:rw,exec,size=16g,mode=1777','--tmpfs','/tmp:rw,exec,size=2g,mode=1777',
            '--device',f'nvidia.com/gpu={gpu}']
    labels = {'s4d.project':'splatter4d','s4d.owner':'kaist-sunho','s4d.role':ROLE,
              's4d.managed':'false','s4d.gpu':gpu,'s4d.rwprep.job':job['id'],
              's4d.commit':config['commit'],'s4d.queue_sha256':config['queue_sha256'],
              's4d.dispatcher':'rwprep-mig','s4d.mem_bytes':str(int(job['gpu_gib']*GIB)),
              's4d.ram_bytes':str(int(job['ram_gib']*GIB)),'s4d.disk_bytes':str(int(job.get('disk_gib',32)*GIB))}
    for k,v in labels.items():
        argv += ['--label',f'{k}={v}']
    mounts = [(config['code'],'/opt/src/splatter4d',True),(config['tools'],'/opt/rwprep',True),
              (config['weights'],'/ram/weights',True),(config['assets'],'/home/ws/ws/validation-20261031',True),
              (config['raw'],config['raw'],True),(config['bundle'],config['bundle'],True),
              (config['state_dir'],config['state_dir'],False),(config['processed'],config['processed'],False)]
    mounts += [(p,p,True) for p in config.get('extra_readonly',[])]
    for src,dst,ro in mounts:
        argv += ['--mount',f'type=bind,src={src},dst={dst}'+(',readonly' if ro else '')]
    env = dict(HOME='/ram/home',TMPDIR='/ram/tmp',S4D_CACHE_ROOT='/ram/s4d-cache',
               TORCH_HOME='/ram/torch',HF_HOME='/ram/hf',HF_HUB_OFFLINE='1',WANDB_MODE='offline',
               XDG_CACHE_HOME='/ram/cache',TORCH_EXTENSIONS_DIR='/ram/torch-ext',TRITON_CACHE_DIR='/ram/triton',
               PYTHONDONTWRITEBYTECODE='1',NVIDIA_VISIBLE_DEVICES='void',CUDA_VISIBLE_DEVICES=gpu,
               S4D_RWPREP_GPU=gpu,S4D_RWPREP_CONTAINER=name,S4D_RWPREP_IMAGE=config['image'],
               S4D_HOST_CONFIG='/opt/src/splatter4d/configs/hosts/'+config['host']+'.yaml',
               CC='/opt/envs/sam3/bin/cc-zig',S4D_TE_CUDA_MEMORY_CAP_GIB='5.5',
               OMP_NUM_THREADS=str(job['cpus']),MKL_NUM_THREADS=str(job['cpus']),OPENBLAS_NUM_THREADS=str(job['cpus']))
    for k,v in env.items():
        argv += ['--env',f'{k}={v}']
    return argv+['--entrypoint','/opt/s4d/.venv/bin/python',config['image'],'-I','-B',
                 '/opt/src/splatter4d/scripts/realworld_prep_worker.py',str(jobfile)]


def tick(config, apply=False):
    root = Path(config['state_dir'])
    root.mkdir(parents=True, exist_ok=True)
    locks = [root/'dispatcher.lock',Path(config['admission_lock'])]
    handles = []
    try:
        for path in locks:
            handle = path.open('a')
            handles.append(handle)
            fcntl.flock(handle,fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        for handle in handles:
            handle.close()
        return {'held':'admission lock busy'}
    try:
        jobs = json.loads(Path(config['jobs']).read_text())
        validate(config,jobs)
        sha = digest(config['jobs'])
        config = {**config,'queue_sha256':sha}
        statefile = root/'state.json'
        state = json.loads(statefile.read_text()) if statefile.exists() else {'queue_sha256':sha,'jobs':{}}
        if state['queue_sha256'] != sha:
            raise ValueError('immutable queue changed; use a new state directory')
        states, live = state['jobs'], []
        for job in jobs:
            s = states.setdefault(job['id'],{'status':'pending','attempt':0})
            if s['status'] not in ('launching','running'):
                continue
            # Inspect exact names, including stopped containers. A daemon crash
            # between Docker start and state publication never duplicates a job.
            found = run(['docker','ps','-aq','--no-trunc','--filter','name=^/'+s['container']+'$']).split()
            if not found:
                if s['status'] == 'running':
                    s.update(status='unknown',reason='running container vanished; explicit diagnosis required')
                else:
                    s['status'] = 'retry' if s['attempt'] < 3 else 'failed'
                continue
            item = json.loads(run(['docker','inspect',found[0]]))[0]
            labels = item['Config']['Labels']
            if (labels.get('s4d.queue_sha256') != sha or labels.get('s4d.rwprep.job') != job['id']
                    or labels.get('s4d.role') != ROLE or labels.get('s4d.commit') != config['commit']):
                raise ValueError('container identity differs from launch intent')
            if item['State']['Running'] or item['State'].get('Restarting') or item['State'].get('Paused'):
                s['status']='running'
                live.append({**job,'gpu':s['gpu']})
            else:
                result = root/job['id']/f"a{s['attempt']}"/'result.json'
                proof = json.loads(result.read_text()) if result.exists() else {}
                ok = item['State']['ExitCode']==0 and proof.get('status')=='ok' and proof.get('queue_sha256')==sha
                s['status']='done' if ok else ('retry' if s['attempt']<3 else 'failed')
                s['exit_code']=item['State']['ExitCode']
        pending = ready(jobs,states,(root/'STOP').exists())
        # Unknown attempts retain reservations and priority until diagnosed.
        uncertain = any(s['status']=='unknown' for s in states.values())
        active = bool(pending or live or uncertain)
        if apply:
            atomic(config['priority_file'],{'active':active,'host':config['host'],'state_dir':str(root),
                                          'updated_unix':time.time(),'reason':'ready/running preprocessing precedes training'})
        launched, blocked = [], {}
        for job in pending:
            if uncertain:
                break
            snap = snapshot(config,live)
            gpu, reasons = admission(job,snap,config)
            if reasons:
                blocked[job['id']]=reasons
                continue
            if not apply:
                launched.append({'would_launch':job['id'],'gpu':gpu})
                live.append({**job,'gpu':gpu})
                continue
            for path, sha256 in job['inputs'].items():
                if digest(path) != sha256:
                    raise ValueError(f'input changed: {path}')
            if run(['git','-C',config['code'],'rev-parse','HEAD']).strip()!=config['commit']:
                raise ValueError('source release moved')
            if run(['git','-C',config['code'],'status','--porcelain','--untracked-files=no']).strip():
                raise ValueError('source release is dirty')
            s = states[job['id']]
            attempt = s['attempt']+1
            s.update(status='launching',attempt=attempt,gpu=gpu,container=f"s4d-{job['id']}-a{attempt}")
            jobdir = root/job['id']
            jobdir.mkdir(exist_ok=True)
            jobfile = jobdir/f'job-a{attempt}.json'
            atomic(jobfile,{**job,'attempt':attempt,'result_dir':str(jobdir),'queue_sha256':sha,
                            'server':config['server']})
            atomic(statefile,state)  # intent is durable BEFORE docker run
            try:
                run(container_command(config,job,attempt,gpu,jobfile))
            except (subprocess.SubprocessError,OSError) as exc:
                s['launch_error']=type(exc).__name__  # next tick reconciles; never blind retry
                atomic(statefile,state)
                break
            s['status']='running'
            live.append({**job,'gpu':gpu})
            launched.append(job['id'])
            atomic(statefile,state)
        state['last_tick']={'time':time.time(),'launched':launched,'blocked':blocked}
        if apply:
            atomic(statefile,state)
        return {**state['last_tick'],'counts':{x:sum(s['status']==x for s in states.values())
                for x in ('pending','running','done','retry','failed','unknown')}}
    finally:
        for handle in reversed(handles):
            handle.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--apply',action='store_true')
    p.add_argument('--loop',action='store_true')
    a=p.parse_args()
    while True:
        config=json.loads(a.config.read_text())
        print(json.dumps(tick(config,a.apply)),flush=True)
        if not a.loop:
            break
        time.sleep(15)  # STOP drains: reconciliation continues; no signals to workers


if __name__ == '__main__':
    main()
