#!/usr/bin/env python3
"""Coordinator-only staging; no network/mutations occur on import or --help.

Run with adopted main .venv/bin/python. Every subcommand is explicit; the
implementer only syntax-checks this file. Operational transfers use RemoteClient.
"""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import os
import shlex
import shutil
import sys
import tarfile

P=Path(__file__).resolve().parent
MAIN=Path('/home/ws/ws/splatter4d')
A=MAIN/'runs/rwv2-prep-assets'
R=Path('/home/compu/kaist/sunho/realworld')
X=R/'processed/dsreset-20261113'
TASKS=('drawer-open-v2','covering','pick-and-place')
sys.path.insert(0,str(MAIN))


def read(p):return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def write(p,data,new=False):
    p=Path(p)
    if new and p.exists():raise ValueError(f'already exists: {p}')
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    with tmp.open('w') as f:
        json.dump(data,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    tmp.replace(p)


def manifest(task,variant='SF'):
    bundle='drawer' if task=='drawer-open-v2' else task
    return read(P/f'{bundle}-canonical-manifests/{variant}/manifest.json')


def remote_json(client,code):
    return json.loads(client.checked_ssh('python3 -c '+shlex.quote(code),idempotent=True,timeout=3600).stdout)


def client():
    from ops.remote.access import RemoteClient
    return RemoteClient(repo=MAIN)


def space(path,size):
    d=shutil.disk_usage(path)
    if d.free-size<max(32*2**30,.15*d.total):raise ValueError('staging disk floor')


def reused():
    from ops.jobs.realworld_local import _remote_batch
    dirs=[]
    for task in ('covering','pick-and-place'):
        for row in read(P/f'{task}-canonical-manifests/prep-plan.json'):
            if 'reuse' in row:dirs.append(str(Path(row['reuse']).relative_to(R)))
    assert len(dirs)==195 and len(set(dirs))==195
    c=client(); found,_=_remote_batch([],sorted(dirs),c,str(R))
    for d in dirs:
        assert {d+'/episode.h5',d+'/_COMPLETE.json'}<=set(found),d
    target=A/'canonical-realworld';target.mkdir(parents=True,exist_ok=True)
    wanted={r:v['sha256'] for r,v in found.items()}
    for r,h in wanted.items():
        dst=target/r
        if dst.is_symlink() or any(p.is_symlink() for p in dst.parents):raise ValueError('symlink transfer target')
        if dst.exists() and sha(dst)!=h:raise ValueError(f'conflicting immutable mirror: {dst}')
    space(target,sum(v['bytes'] for r,v in found.items() if not (target/r).exists()))
    c.download_batch(str(R),target,wanted,'dsreset-reused-success',priority='dependency')
    assert all(sha(target/r)==h for r,h in wanted.items())
    write(P/'reused-compact-mirror.json',found)
    print('195 exact reused compact directories verified')


def raw_proof():
    rows=[r for t in TASKS for r in manifest(t)['episodes']]
    expected={str(Path('/home/compu/kaist/sunho/data/real_world')/r['path'].split('/real_world/',1)[1]):r['sha256'] for r in rows}
    assert len(expected)==443
    code='''import hashlib,json,pathlib
expected=EXPECTED
out={}
for name,want in expected.items():
 p=pathlib.Path(name)
 if not p.is_file() or p.is_symlink():raise ValueError(name)
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 have=h.hexdigest()
 if have!=want:raise ValueError('raw SHA mismatch: '+name)
 out[want]=have
print(json.dumps(out))
'''.replace('EXPECTED',repr(expected))
    result=remote_json(client(),code)
    assert len(result)==443 and all(k==v for k,v in result.items())
    write(P/'remote-raw-hashes.json',result,new=True)
    print('443 independent remote raw SHA proofs recorded')


def upstream():
    c=client();stage=A/'calib-upstream';stage.mkdir(parents=True,exist_ok=True)
    roots={'LoMa':str(R/'calib-loma/upstream/LoMa'),
           'romav2':str(R/'calib-dw/upstream/romav2-95c9968'),
           'torch-loma':str(R/'calib-loma/torch-home'),
           'torch-roma':str(R/'calib-dw/torch-home')}
    code='''import fcntl,json,os,pathlib,subprocess,tarfile
roots=ROOTS
for name,prefix in [('LoMa','415caed'),('romav2','95c9968')]:
 got=subprocess.check_output(['git','-C',roots[name],'rev-parse','HEAD'],text=True).strip()
 if not got.startswith(prefix):raise ValueError(name+' pin mismatch')
out={}
for name,path in roots.items():
 archive=pathlib.Path('/home/compu/kaist/sunho/realworld/dsreset-'+name+'-20261114.tar')
 with archive.with_suffix('.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  if not archive.exists():
   tmp=archive.with_suffix('.tmp')
   with tarfile.open(tmp,'w') as f:f.add(path,arcname='torch-home' if name.startswith('torch-') else name)
   os.replace(tmp,archive)
 out[name]=str(archive)
print(json.dumps(out))
'''.replace('ROOTS',repr(roots))
    archives=remote_json(c,code);provenance={}
    for name,remote in archives.items():
        local=A/Path(remote).name;c.rsync_download(remote,local);provenance[remote]=sha(local)
        # Extract/merge regular files only; reject links and path escape. Byte
        # conflicts in the two torch caches are errors, never last-writer-wins.
        with tarfile.open(local) as archive:
            for member in archive:
                dest=(stage/member.name).resolve()
                if not dest.is_relative_to(stage.resolve()):raise ValueError('archive path escape')
                if member.isdir():dest.mkdir(parents=True,exist_ok=True);continue
                if not member.isfile():raise ValueError(f'nonregular upstream member: {member.name}')
                src=archive.extractfile(member);data=src.read()
                if dest.exists():
                    if dest.read_bytes()!=data:raise ValueError(f'conflicting upstream: {dest}')
                else:
                    space(stage,len(data));dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
    write(P/'calibration-upstream-provenance.json',provenance,new=True)
    print('Pinned LoMa/RoMa source and conflict-checked torch caches staged')


def mount_reuse():
    for task,root in [('covering',R/'processed/calrel-6c3b9ec/covering'),
                      ('pick-and-place',R/'processed/calv2-4368f22/pick-and-place')]:
        file=P/f'local-measure-{task}/dispatcher.json';d=read(file)
        assert not (Path(d['state_dir'])/'state.json').exists(),'already started; use a new immutable bundle'
        assert root.is_dir(),root
        d['extra_readonly']=list(dict.fromkeys([*d.get('extra_readonly',[]),str(root)]))
        write(file,d)
    print('Read-only reused compact roots added to unstarted measurement bundles')


def publish():
    c=client();pairs=[]
    for task in TASKS:
        config=read(P/f'local-phaseB-{task}/dispatcher.json')
        states=read(Path(config['state_dir'])/'state.json')['jobs']
        jobs=read(config['jobs'])
        assert all(states.get(j['id'],{}).get('status')=='done' for j in jobs),'phase B not complete'
        for j in jobs:
            cp=next(p for p in j['inputs'] if p.endswith('.yaml'))
            import yaml
            root=Path(yaml.safe_load(Path(cp).read_text())['output_root'])/'train_data'/j['episode']
            assert root.is_relative_to(X) and (root/'_COMPLETE.json').is_file() and (root/'episode.h5').is_file()
            pairs.extend((f,f) for f in root.iterdir() if f.is_file() and not f.is_symlink())
        rig=X/'calibration'/task/f'rig-{task}-accepted.json'
        assert read(X/'measurements'/task/'scene.json')['accepted'] is True
        pairs += [(rig,rig),(rig.with_suffix('.reference.npz'),rig.with_suffix('.reference.npz'))]
        for v in ('S','SF'):
            pairs += [(P/'learning-inputs'/task/v/n,R/'learning_inputs/dsreset'/task/v/n)
                      for n in ('store.yaml','manifest.json','split_seed0.json','outcomes.json','gate_report.json')]
    inventory={str(remote):sha(local) for local,remote in pairs}
    # Inventory all destinations before the first write. Publication itself uses
    # hard-link creation, so a race cannot overwrite an existing different file.
    code='''import hashlib,json,pathlib
expected=EXPECTED
for name,want in expected.items():
 p=pathlib.Path(name)
 if p.is_symlink():raise ValueError(name)
 if p.exists():
  h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
  if h.hexdigest()!=want:raise ValueError('immutable remote conflict: '+name)
print(json.dumps(True))
'''.replace('EXPECTED',repr(inventory))
    # Keep each SSH command below Linux's per-argument size limit.
    entries=list(inventory.items())
    for start in range(0,len(entries),64):
        batch=dict(entries[start:start+64])
        assert remote_json(c,code.replace(repr(inventory),repr(batch)))
    for local,remote in pairs:
        h=inventory[str(remote)];staged=A/'publish'/remote.relative_to(R)
        staged.parent.mkdir(parents=True,exist_ok=True)
        if not staged.exists():space(A,local.stat().st_size);shutil.copyfile(local,staged)
        assert sha(staged)==h
        temp=str(remote)+'.publish-'+h
        c.rsync_upload(staged,temp)
        linkcode='''import hashlib,os,pathlib
p=pathlib.Path(DEST); src=SOURCE;want=WANT
try:os.link(src,p)
except FileExistsError:pass
if p.is_symlink():raise ValueError('destination symlink')
h=hashlib.sha256()
with p.open('rb') as f:
 for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
if h.hexdigest()!=want:raise ValueError('publication conflict')
'''.replace('DEST',repr(str(remote))).replace('SOURCE',repr(temp)).replace('WANT',repr(h))
        c.checked_ssh('python3 -c '+shlex.quote(linkcode),idempotent=True)
    write(P/'published-inputs.json',inventory,new=True)
    print('New compact stores, rig documents and paired inputs published without overwrite')


def spec():
    d=read(P/'registration-spec.example.json')
    for task in TASKS:d['tasks'][task]['rig']=str(X/'calibration'/task/f'rig-{task}-accepted.json')
    write(P/'registration-spec.json',d,new=True)


def pause(active):
    handles=[]
    try:
        for p in (MAIN/'runs/setup/remote-driver.lock',MAIN/'runs/setup/stage2-driver.lock'):
            f=p.open('a');fcntl.flock(f,fcntl.LOCK_EX);handles.append(f)
        file=MAIN/'runs/setup/admission-pauses.json';d=read(file)
        name='dsreset-await-placement-20261113'
        rows=[r for r in d['pauses'] if r.get('name')==name]
        assert len(rows)<=1
        if not active:
            holds=list((MAIN/'runs/setup/rwprep-mig-priority.d').glob('*.json'))
            legacy=MAIN/'runs/setup/rwprep-mig-priority.json'
            if legacy.exists():holds.append(legacy)
            assert all(read(p)['active'] is False for p in holds),'preprocessing has not drained'
            for task in TASKS:assert read(X/'measurements'/task/'scene.json')['accepted'] is True
        if rows:
            assert rows[0]['prefixes']==['rwv2-20261113-'];rows[0]['active']=active
        else:
            assert active,'cannot remove a missing pause'
            d['pauses'].append({'name':name,'active':True,'prefixes':['rwv2-20261113-'],
                               'reason':'S/SF registration waits for verified placement and release pins'})
        write(file,d)
    finally:
        for f in reversed(handles):f.close()
    print(name,'active='+str(active))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['reused','raw-proof','upstream','mount-reuse','publish','spec','pause','unpause'])
    a=p.parse_args()
    if a.command in ('pause','unpause'):pause(a.command=='pause')
    else:globals()[a.command.replace('-','_')]()


if __name__=='__main__':main()
