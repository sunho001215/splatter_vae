"""Coordinator-only uncapped BH raw upload. Never invoked by the agent.

No persistent rate limit is changed: only these rsync processes omit --bwlimit. Three streams, resumable parts,
local+remote SHA verification on EVERY file (including existing canonical copies), no replacement/deletion.
"""
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from ops.remote.access import RemoteClient


class NiceClient(RemoteClient):
    def checked_ssh(self, command, **kwargs):
        return super().checked_ssh('nice -n 19 ionice -c 3 bash -c '+shlex.quote(command), **kwargs)


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def verified_sums(root):
    expected={f'episode_{i:06d}.h5' for i in range(102) if i not in (25,54)}
    assert root.with_name('success.VERIFIED').is_file(), 'wait for coordinator transfer verification'
    def read(suffix):
        out={}
        for line in Path(str(root)+suffix).read_text().splitlines():
            sha,name=line.split();name=Path(name.lstrip('*')).name
            assert name not in out and len(sha)==64 and all(c in '0123456789abcdef' for c in sha)
            out[name]=sha
        return out
    local,robot=read('.local.sha256'),read('.robotpc.sha256')
    assert local==robot and set(local)==expected, 'exact 100 verified recordings required'
    assert {p.name for p in root.glob('episode_*.h5')}==expected
    assert not any(p.name.startswith('.') for p in root.iterdir()), 'in-flight transfer entries present'
    return local


def upload(root, out, client=None):
    sums=verified_sums(root)
    out.mkdir(parents=True,exist_ok=True)
    limits=[Path('/home/ws/ws/realworld-ops/ingest/upload-budget.json'),
            Path('/home/ws/ws/splatter4d/runs/remote/access/download-budget.json')]
    original={str(p):digest(p) for p in limits if p.is_file()}
    (out/'limits-original.json').write_text(json.dumps(original,indent=2))
    c=client or NiceClient()
    remote=c.remote_path('data/real_world')+'/ball-hiding/success'
    c.checked_ssh('mkdir -p -- '+shlex.quote(remote+'/.incoming'),timeout=300)
    def one(name):
        path=root/name; want=sums[name]
        assert not path.is_symlink() and digest(path)==want, f'{name}: local SHA differs'
        target=f'{remote}/{name}'; partial=f'{remote}/.incoming/{name}.part'
        have=c._remote_checksum(target,allow_missing=True)
        if have is not None:
            assert have==want, f'{name}: existing canonical file differs; HOLD'
            return name,want
        args=['nice','-n','19','ionice','-c','3','rsync','-t','--partial','--append-verify','--protect-args',
              '--chmod=F0644','--timeout=300','--rsync-path=nice -n 19 ionice -c 3 rsync',
              '-e',shlex.join(c.ssh_args()),str(path),f'{c.target}:{partial}']
        p=subprocess.run(args,env=c._environment(),capture_output=True,text=True)
        if p.returncode:raise RuntimeError(f'{name}: rsync failed; resumable part retained: '+c._redact(p.stderr)[-300:])
        assert c._remote_checksum(partial)==want and digest(path)==want, f'{name}: SHA mismatch; HOLD, retain part'
        c.checked_ssh('mv -n -- '+shlex.quote(partial)+' '+shlex.quote(target),timeout=300)
        assert c._remote_checksum(target)==want, f'{name}: canonical verification failed'
        return name,want
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:proof=dict(pool.map(one,sorted(sums)))
        assert proof==sums
        (out/'remote-sha256.json').write_text(json.dumps(proof,indent=2))
        # Marker is local operational evidence, never a dataset mutation.
        (out/'ball-hiding__success.REMOTE_VERIFIED').write_text(json.dumps({'verified':100,'files':proof},indent=2)+'\n')
        print(json.dumps({'verified':100,'marker':str(out/'ball-hiding__success.REMOTE_VERIFIED')}))
    finally:
        current={str(p):digest(p) for p in limits if p.is_file()}
        (out/'limits-unchanged.json').write_text(json.dumps({'unchanged':current==original,'before':original,'after':current},indent=2))
        assert current==original, 'persistent budgets changed outside this upload; inspect, never overwrite operator changes'


if __name__=='__main__':
    upload(Path('/home/ws/data/real_world/ball-hiding/success'),
           Path('/home/ws/ws/realworld-ops/ballhiding-20261120/upload'))
