"""Publish one completed episode atomically at the canonical remote root; dry-run default.

Archive and every payload are SHA256 checked. Existing destinations are accepted
only if byte-identical. No overwrite, remote job launch, or remote hold changes.
"""
from __future__ import annotations
import argparse
import contextlib
import json
from pathlib import Path
import shlex
import sys
import tarfile

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 's4d').is_dir())
sys.path.insert(0, str(ROOT))
from ops.jobs import local_phase_a as a  # noqa: E402

REMOTE_INSTALL = r'''
import fcntl,hashlib,json,os,pathlib,shutil,sys,tarfile,tempfile
archive,target,expected=sys.argv[1:]
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
 return h.hexdigest()
assert sha(archive)==expected
p=pathlib.Path(target)
assert str(p).startswith('/home/compu/kaist/sunho/realworld/processed/dsreset-20261113/')
assert not any(x.is_symlink() for x in (p,*p.parents))
p.parent.mkdir(parents=True,exist_ok=True)
with open(p.parent/('.'+p.name+'.publish.lock'),'a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 with tempfile.TemporaryDirectory(dir=p.parent,prefix='.'+p.name+'.incoming-') as tmp:
  tmp=pathlib.Path(tmp)
  with tarfile.open(archive,'r') as t:
   for m in t.getmembers():
    q=pathlib.PurePosixPath(m.name)
    assert not q.is_absolute() and '..' not in q.parts and m.isfile(),m.name
   t.extractall(tmp,filter='data')
  manifest=json.loads((tmp/'.publication.json').read_text())
  assert all(sha(tmp/n)==h for n,h in manifest['files'].items())
  assert {str(f.relative_to(tmp)) for f in tmp.rglob('*') if f.is_file()}==set(manifest['files'])|{'.publication.json'}
  if p.exists():
   actual={str(f.relative_to(p)):sha(f) for f in p.rglob('*') if f.is_file() and f.name!='.publication.json'}
   assert actual==manifest['files'],'existing canonical output differs; refusing overwrite'
  else:
   os.rename(tmp,p)
   fd=os.open(p.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
   tmp.mkdir() # TemporaryDirectory cleanup only the empty temporary name
print(json.dumps({'published':str(p),'archive_sha256':expected,'files':len(manifest['files'])}))
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episode', required=True)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    parts = a.identity(args.episode)
    source = a.PROCESSED.joinpath(*parts)
    target = a.REMOTE + '/' + '/'.join(parts)
    a.require(source.is_dir(), 'local episode root does not exist')
    with a.output_lock(source) if args.execute else contextlib.nullcontext():
        proof = json.loads((source / 'local-phase-a.json').read_text())
        a.require(proof['status'] == 'ok' and proof['integrity_after'] is True and proof['stages'] == list(a.STAGES), 'episode is incomplete')
        for stage in a.STAGES:
            tail = 'normalized' if stage == 'ingest' else 'derived/' + stage
            a.require((source/'episodes'/args.episode/tail/'_COMPLETE.json').is_file(), 'missing stage completion')
        files = a.inventory(source)
        files.pop('.local-phase-a.lock', None)
        manifest = {'schema': 'local-phase-a-publication/1', 'episode': args.episode, 'files': files}
        if not args.execute:
            print(json.dumps({'source': str(source), 'target': target, 'files': len(files), 'published': False}, indent=2))
            return 0
        import io
        import tempfile
        import hashlib
        incoming = ROOT / 'runs/local-phase-a-upload'
        incoming.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=incoming) as tmp:
            archive = Path(tmp) / 'episode.tar'
            with tarfile.open(archive, 'w') as t:
                for name in files:
                    t.add(source / name, arcname=name, recursive=False)
                data = json.dumps(manifest, sort_keys=True).encode()
                info = tarfile.TarInfo('.publication.json'); info.size = len(data)
                t.addfile(info, io.BytesIO(data))
            # Recheck originals after packing; concurrent writers/drift fail before network publication.
            a.require(all(a.sha(source/n) == h for n,h in files.items()), 'source changed during archive')
            token = a.sha(archive)
            from ops.remote.access import RemoteClient
            client = RemoteClient()
            remote_archive = '/home/compu/kaist/sunho/realworld/incoming/local-phase-a-' + token + '.tar'
            client.ensure_remote_path(remote_archive, create_parent=True)
            client.rsync_upload(archive, remote_archive)
            command = 'python3 -I -c ' + shlex.quote(REMOTE_INSTALL) + ' ' + shlex.join([remote_archive, target, token])
            result = client.checked_ssh(command, idempotent=True, wait=False, timeout=3600)
            print(result.stdout)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
