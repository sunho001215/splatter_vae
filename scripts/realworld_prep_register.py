"""Compile immutable prep plans into a host-local MIG queue; never launch jobs."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import yaml
from ops.campaign.rwprep_mig import atomic, digest, validate


def compile_plans(plans, out, processed, raw, phase='A', rig_map=None):
    jobs=[]
    out=Path(out).resolve()
    out.mkdir(parents=True,exist_ok=False)
    (out/'configs').mkdir()
    for plan in plans:
        for row in json.loads(Path(plan).read_text()):
            if 'config' not in row:  # explicit hash-bound success reuse
                continue
            ep=row['episode']
            cfg=yaml.safe_load(Path(row['config']).read_text())
            rawfile=Path(cfg['adapter']['config']['episodes'][ep])
            # Resolve from the known raw-root suffix, never by episode number.
            tail=str(rawfile).split('/real_world/',1)[1]
            rawfile=Path(raw)/tail
            cfg['adapter']['config']['episodes']={ep:str(rawfile)}
            if phase=='A':
                task,label,number=ep.rsplit('-',2)
                cfg['output_root']=str(Path(processed)/task/label/number)
            else:
                rig=(rig_map or {}).get(ep)
                if not rig or not Path(rig).is_file():
                    raise ValueError(f'{ep}: phase B needs a measured accepted session rig')
                cfg['stages']['calibrate']['rig_calibration']=str(Path(rig).resolve())
            path=out/'configs'/f'{ep}.yaml'
            path.write_text(yaml.safe_dump(cfg,sort_keys=False))
            server=str(out/'state/server.yaml')
            inputs={str(path):digest(path),str(rawfile):row['raw_sha256']}
            if phase=='B':
                inputs[str(Path(rig).resolve())]=digest(rig)
            stages='ingest,sensor_depth,foundation_stereo_depth,sam3_masks' if phase=='A' else 'calibrate,track_everything,track_motion,train_data'
            steps=[{'kind':'rwprep','argv':['episode','--pipeline',str(path),'--server',server,
                    '--episode',ep,'--stages',stages]}]
            if phase=='B':
                steps += [{'kind':'rwprep','argv':['contracts','--pipeline',cfg['output_root']+'/pipeline.yaml',
                           '--server',server,'--episode',ep]},
                          {'kind':'rwprep','argv':['qa','--roots',cfg['output_root'],'--episode',ep,
                           '--out',cfg['output_root']+'/qa.json','--families']}]
            jobs.append(dict(id=f'rwv2-20261113-prep-ours-{ep}-phase{phase}',episode=ep,phase=phase,
                priority=-3,gpu_profile='1g.24gb',gpu_gib=16 if phase=='A' else 7,
                ram_gib=48 if phase=='A' else 16,cpus=4,disk_gib=32,inputs=inputs,deps=[],steps=steps))
    atomic(out/'jobs.json',jobs)
    return jobs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,action='append',required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--processed',type=Path,required=True)
    p.add_argument('--raw',type=Path,default=Path('/home/ws/data/real_world'))
    p.add_argument('--phase',choices=['A','B'],default='A')
    p.add_argument('--rig-map',type=Path)
    p.add_argument('--host',choices=['local','remote'],default='local')
    p.add_argument('--image',required=True,help='installed prep image immutable sha256 ID')
    p.add_argument('--code',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--tools',type=Path,default=Path('/home/ws/ws/realworld-ops/pre/tools/rwprep'))
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--assets',type=Path,default=Path('/home/ws/ws/validation-20261031'))
    p.add_argument('--extra-readonly',type=Path,action='append',default=[])
    a=p.parse_args()
    from ops.resources.gpu_guard import MIG_DEVICES,REMOTE_MIG_DEVICES
    from ops.campaign.rwprep_mig import run
    rigmap=json.loads(a.rig_map.read_text()) if a.rig_map else None
    a.out=a.out.resolve()
    jobs=compile_plans(a.plan,a.out,a.processed,a.raw,a.phase,rigmap)
    (a.out/'state').mkdir()
    a.processed.mkdir(parents=True,exist_ok=True)
    inventory=MIG_DEVICES if a.host=='local' else REMOTE_MIG_DEVICES
    server=yaml.safe_load((a.tools/'configs/rwprep_server.yaml').read_text())
    server.update(gpu_allowlist=list(inventory),mig_parents={u:v[0] for u,v in inventory.items()})
    (a.out/'state/server.yaml').write_text(yaml.safe_dump(server))
    config=dict(host=a.host,image=a.image,commit=run(['git','-C',a.code,'rev-parse','HEAD']).strip(),
        code=str(a.code.resolve()),tools=str(a.tools.resolve()),weights=str(a.weights.resolve()),
        assets=str(a.assets.resolve()),raw=str(a.raw.resolve()),processed=str(a.processed.resolve()),
        bundle=str(a.out),state_dir=str(a.out/'state'),jobs=str(a.out/'jobs.json'),server=str(a.out/'state/server.yaml'),
        queue=str(a.code/'experiments/queue.yaml'),registry=str(a.code/'experiments/registry.jsonl'),
        admission_lock=str(a.code/'runs/setup/stage2-driver.lock'),
        priority_file=str(a.code/'runs/setup/rwprep-mig-priority.json'),
        ram_cap_bytes=138_000_000_000 if a.host=='local' else 1000*2**30,
        extra_readonly=list(map(str,a.extra_readonly)))
    validate(config,jobs)
    atomic(a.out/'dispatcher.json',config)
    print(json.dumps({'registered':len(jobs),'launched':False,'config':str(a.out/'dispatcher.json')}))


if __name__=='__main__':
    main()
