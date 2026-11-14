"""Build a NEW ingest snapshot from verified raw + ordinary prep completion records.

Never rewrite the legacy ingest worker's live task inventory. The snapshot is
then updated only through established calibration_release/ingest_worker.set_stage.
"""
import argparse
import copy
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ops.jobs.rwprep_mig import atomic,digest


def stages(root,ep,sha,commit,config_sha,host):
    root=Path(root);base=root/'episodes'/ep
    paths={'normalized':base/'normalized','train_data':root/'train_data'/ep}
    if (base/'derived').is_dir():paths.update({p.name:p for p in (base/'derived').iterdir() if p.is_dir() and not p.name.startswith('.')})
    records={}
    common=dict(input_sha256=sha,code_commit=commit,config_sha256=config_sha,host=host,attempt=1)
    for name,p in paths.items():
        if not (p/'_COMPLETE.json').is_file():continue
        r=json.loads((p/'_COMPLETE.json').read_text());summary=r.get('summary') or {}
        store={k:summary.get(k) for k in ('format','schema_version','writer_version','code_commit','file_bytes','file_sha256','cameras','frames','semantic')} if name=='train_data' else None
        fmt=f"s4d.realworld.stage/{r['stage']}@{r['version']}"
        if store:fmt=store['format'] if str(store['format']).endswith(str(store['schema_version'])) else f"{store['format']}/{store['schema_version']}"
        records[name]={**common,'status':'done','code_version':f"{r['stage']}@{r['version']}",
            'output_format':fmt,'outputs':str(p if name=='train_data' else base),
            'metrics':{'build_id':r['build_id'],'fingerprint':r['fingerprint'],**({'train_store':store} if store else {})}}
    qa=root/'qa.json'
    if qa.is_file():
        q=json.loads(qa.read_text())
        if q.get('schema')!='rwprep-qa/3':raise ValueError('per-family QA v3 required')
        records['qa']={**common,'status':'done' if q.get('passed') else 'failed','code_version':'rwprep-qa@3',
            'output_format':'rwprep-qa/3','outputs':str(qa),'metrics':{'families':q.get('families'),'flags':q.get('flags')},
            **({} if q.get('passed') else {'failure':{'message':'per-episode QA failed'}})}
    return records


def main():
    import yaml
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--manifest',type=Path,action='append',required=True,help='each task SF manifest')
    p.add_argument('--prep-config',type=Path,action='append',required=True,help='phase-A or B dispatcher configs')
    p.add_argument('--remote-hashes',type=Path,required=True,help='JSON {raw sha256: same verified remote sha256}')
    p.add_argument('--out-dir',type=Path,required=True)
    a=p.parse_args()
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runs/setup/tools/realworld_ingest_20261106'))
    import episode_validator as V
    source=json.loads(a.source.read_text());doc=copy.deepcopy(source)
    proofs=json.loads(a.remote_hashes.read_text());configs={}
    for dispatcher in a.prep_config:
        d=json.loads(dispatcher.read_text())
        for job in json.loads(Path(d['jobs']).read_text()):
            ep=job.get('episode')
            if ep:configs[ep]=(d,job,next(p for p in job['inputs'] if p.endswith('.yaml')))
    for mp in a.manifest:
        manifest=json.loads(mp.read_text())
        if manifest['variant']!='SF':raise ValueError('one SF manifest per task')
        for r in manifest['episodes']+manifest.get('excluded',[]):
            matches=[(k,e) for k,e in doc['episodes'].items() if e.get('task')==manifest['task'] and e.get('sha256')==r['sha256']]
            if len(matches)>1:raise ValueError('duplicate ingest raw identity')
            if matches and r['id'] not in configs:continue  # explicit existing success reuse
            if digest(r['path'])!=r['sha256']:raise ValueError('raw changed since immutable manifest')
            excluded=r.get('split')=='excluded'
            if not excluded and proofs.get(r['sha256'])!=r['sha256']:raise ValueError('missing exact remote raw verification proof')
            v=V.validate(r['path'],grace_s=0,task_names=[manifest['task']])
            if not excluded and v['status']!='valid':raise ValueError(f"raw rejected: {r['id']}: {v}")
            if excluded and v['status']=='valid':raise ValueError('amended exclusion is no longer invalid')
            key=matches[0][0] if matches else r['id']
            e=copy.deepcopy(matches[0][1]) if matches else {}
            e.update(key=key,task=manifest['task'],filename=r['filename'],sha256=r['sha256'],
                source={'path':r['path'],'present':True},episode_id=r['id'],episode_label=r['label'],
                episode_index=int(r['filename'][8:14]),file_index=int(r['filename'][8:14]),flags={},
                validation={**v,'status':'invalid' if excluded else 'valid'},
                transfer={'status':'verified' if not excluded else 'excluded','remote_sha256':r['sha256'] if not excluded else None})
            if r['id'] in configs:
                d,j,cp=configs[r['id']];cfg=yaml.safe_load(Path(cp).read_text())
                recs=stages(cfg['output_root'],r['id'],r['sha256'],d['commit'],digest(cp),d['host'])
                e['preprocessing']={'schema':'rw-preprocess','schema_version':1,'status':
                    'failed' if any(x['status']=='failed' for x in recs.values()) else 'done','stages':recs}
            else:e.setdefault('preprocessing',{'schema':'rw-preprocess','schema_version':1,'status':'pending','stages':{}})
            doc['episodes'][key]=e
    a.out_dir.mkdir(parents=True,exist_ok=False)
    atomic(a.out_dir/'manifest.json',doc)
    atomic(a.out_dir/'provenance.json',{'source':{str(a.source):digest(a.source)},
        'manifests':{str(p):digest(p) for p in a.manifest},'remote_hashes_sha256':digest(a.remote_hashes)})


if __name__=='__main__':main()
