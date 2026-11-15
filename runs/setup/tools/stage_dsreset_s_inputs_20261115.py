"""Offline, append-only S input publication into the existing local mirror.

Default prints a dry-run. Only the adopted coordinator copy can --execute.
The remote-root paths are scheduler namespace keys; this tool performs no remote
publication and makes no assertion that these new files exist remotely.
"""
from __future__ import annotations
import argparse
import fcntl
import importlib.util
import json
import os
import tempfile
from pathlib import Path
import sys

SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(SOURCE))
from ops.jobs import realworld_local as L


def registrar():
    spec=importlib.util.spec_from_file_location('stage_dsreset_registrar',SOURCE/'runs/setup/tools/register_dsreset_20261113.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def inventory(config, ingest, mirror):
    """Validate S independently, then bind generated inputs to verified compact closure."""
    r=registrar()
    r.R.require(config.get('variants') == ['S'],'only variants: [S] may be published')
    records={}
    files=L._manifest(mirror/'MANIFEST.json')['files']
    checked=set()
    for task,cfg in config['tasks'].items():
        r.validate_inputs(task,cfg['inputs_copy'],ingest,('S',))
        prefix=L._mapped(cfg['inputs'],L.REMOTE_PREFIX)
        r.R.require(prefix is not None,'inputs outside scheduler namespace')
        for name in ('manifest.json','store.yaml','gate_report.json','split_seed0.json','outcomes.json'):
            source=Path(cfg['inputs_copy'])/'S'/name
            L._regular(source)
            records[prefix+'/S/'+name]=source.read_bytes()
    for remote,local in config.get('local_publication_files',{}).items():
        rel=L._mapped(remote,L.REMOTE_PREFIX)
        r.R.require(rel is not None and not rel.startswith('learning_inputs/'),'invalid auxiliary publication path')
        path=Path(local);L._regular(path);records[rel]=path.read_bytes()
    entries={}
    for rel,data in records.items():
        if Path(rel).name=='store.yaml':
            localized,dirs=L._descriptor(data,mirror=mirror)
            requires=[]
            for directory in dirs:
                for name in ('_COMPLETE.json','episode.h5'):
                    item=directory+'/'+name
                    L._check(item,files,mirror,checked)
                    requires.append(item)
            entries['_remote/'+rel]=(data,'copy',[],L._sha(data),len(data))
            entries[rel]=(localized,'localized',sorted(requires),L._sha(data),len(data))
        else:
            entries[rel]=(data,'copy',[],L._sha(data),len(data))
    for rel,(data,*_) in entries.items():
        target=mirror/rel
        if target.exists():
            L._regular(target)
            if target.read_bytes()!=data:raise ValueError(f'append-only publication conflict: {rel}')
        if rel in files and files[rel]['sha256']!=L._sha(data):
            raise ValueError(f'append-only manifest conflict: {rel}')
    return entries,checked


def publish(config, ingest, mirror, *, execute=False):
    entries,checked=inventory(config,ingest,mirror)
    if execute:
        # The caller holds the existing mirror fetch lock through revalidation and publication.
        spec=L._manifest(mirror/'MANIFEST.json')
        for rel,(data,kind,requires,digest,size) in entries.items():
            target=mirror/rel
            if target.is_symlink() or any(p.is_symlink() for p in target.parents):raise ValueError('symlink publication target')
            target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():
                with tempfile.NamedTemporaryFile(dir=target.parent,delete=False) as stream:
                    tmp=Path(stream.name);stream.write(data);stream.flush();os.fsync(stream.fileno())
                try:
                    try:os.link(tmp,target)
                    except FileExistsError:pass
                finally:tmp.unlink()
            L._regular(target)
            if target.read_bytes()!=data:raise ValueError(f'publication race/conflict: {rel}')
            os.chmod(target,0o444)
            stat=target.stat()
            spec['files'][rel]=dict(kind=kind,sha256=L._sha(data),bytes=len(data),mtime_ns=stat.st_mtime_ns,
                remote_sha256=digest,remote_bytes=size,requires=requires,
                origin='locally-generated S publication; no remote write',fetched_at=L.host_roles.now())
        L._json(mirror/'MANIFEST.json',spec)
    return dict(status='published' if execute else 'dry-run',network=False,variants=['S'],
        files={rel:L._sha(item[0]) for rel,item in entries.items()},verified_compact_files=len(checked))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,default=Path('/home/ws/ws/splatter4d'))
    p.add_argument('--spec',type=Path,required=True)
    p.add_argument('--ingest-manifest',type=Path,required=True)
    p.add_argument('--execute',action='store_true')
    a=p.parse_args();config=json.loads(a.spec.read_bytes());ingest=json.loads(a.ingest_manifest.read_bytes())
    mirror=a.repo/'runs/realworld-inputs'
    if a.execute:
        r=registrar()
        r.R.require(SOURCE==r.MAIN and a.repo.resolve()==r.MAIN,'only adopted coordinator copy may execute')
        r.R.local_release_pin(a.repo)
        with (mirror/'.fetch.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            result=publish(config,ingest,mirror,execute=True)
    else:result=publish(config,ingest,mirror)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
