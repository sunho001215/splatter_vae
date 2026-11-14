"""Register CPU calibration validation or native scene measurements on prep MIGs."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ops.campaign.rwprep_mig import atomic,digest,validate,priority_path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('kind',choices=['records','measure'])
    p.add_argument('--base-config',type=Path,required=True)
    p.add_argument('--task',required=True)
    p.add_argument('--rig',type=Path,required=True)
    p.add_argument('--inputs',type=Path,help='SF learning inputs for measure')
    p.add_argument('--manifest',type=Path,help='ingest snapshot for records')
    p.add_argument('--out',type=Path,required=True,help='new queue bundle')
    p.add_argument('--result',type=Path,required=True,help='output under the processed mount')
    a=p.parse_args();cfg=json.loads(a.base_config.read_text())
    if not a.result.resolve().is_relative_to(Path(cfg['processed']).resolve()):raise ValueError('result must be under processed mount')
    a.out.mkdir(parents=True,exist_ok=False);a.result.mkdir(parents=True,exist_ok=True)
    files=[a.rig,a.rig.with_suffix('.reference.npz')]
    if a.kind=='records':
        files.append(a.manifest)
        argv=['-m','s4d.realworld.calib.release','plan','--manifest',str(a.manifest),'--rig',a.task+'='+str(a.rig),'--out',str(a.result/'plan.json')]
    else:
        files += [a.inputs/n for n in ('store.yaml','manifest.json','split_seed0.json','outcomes.json','gate_report.json')]
        argv=['scripts/realworld_dsreset_measure.py','--inputs',str(a.inputs),'--rig',str(a.rig),'--out',str(a.result)]
    job=dict(id=f'rwv2-20261113-qa-ours-{a.task}-{a.kind}',gpu_profile='1g.24gb',priority=-3,
             gpu_gib=1,ram_gib=16,cpus=4,disk_gib=32,inputs={str(p.resolve()):digest(p) for p in files},
             deps=[],steps=[{'kind':'python','argv':argv}])
    cfg.update(bundle=str(a.out.resolve()),state_dir=str(a.out.resolve()/'state'),jobs=str(a.out.resolve()/'jobs.json'),
               priority_file=priority_path(Path(cfg['priority_file']).parent,a.out.resolve()/'state'),
               extra_readonly=list(dict.fromkeys([*cfg.get('extra_readonly',[]),*[str(p.resolve().parent) for p in files]])))
    validate(cfg,[job]);Path(cfg['state_dir']).mkdir()
    atomic(a.out/'jobs.json',[job]);atomic(a.out/'dispatcher.json',cfg)
    print(json.dumps({'registered':1,'launched':False,'config':str(a.out/'dispatcher.json')}))


if __name__=='__main__':main()
