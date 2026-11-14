"""Compile/run task-local LoMa-G jobs on the admitted preprocessing route."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ops.campaign.rwprep_mig import atomic,digest

CAMS=['135122070988/color','135122073201/color','135122076054/color']
TE='/opt/envs/trackeverything/bin/python'


def module(name,*args,python=None,env=None,cap=None):
    return {'kind':'python','argv':['-m','s4d.realworld.'+name,*map(str,args)],
            **({'python':python} if python else {}),**({'env':env} if env else {}),
            **({'cuda_cap_gib':cap} if cap else {})}


def plan(phase_config,task,out,work,upstream):
    """One-process GPU matching, independent CPU solves; no historical driver import."""
    import yaml
    cfg=json.loads(Path(phase_config).read_text());source=json.loads(Path(cfg['jobs']).read_text())
    selected=[j for j in source if j['episode'].startswith(task+'-')]
    if not selected:raise ValueError('no phase-A task jobs')
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    work=Path(work).resolve();work.mkdir(parents=True,exist_ok=True)
    aliases={};roots={}
    for n,j in enumerate(sorted(selected,key=lambda j:j['episode'])):
        alias=f'{task}-{n:06d}';ep=j['episode']
        path=next(p for p in j['inputs'] if p.endswith('.yaml'))
        roots[alias]={'episode':ep,'root':yaml.safe_load(Path(path).read_text())['output_root']}
        aliases[alias]=ep
    spec={'task':task,'roots':roots,'aliases':aliases,'work':str(work),'commit':cfg['commit']}
    atomic(out/'spec.json',spec)
    freeze=out/'FREEZE.json'
    atomic(freeze,{'protocol':'rig_loma-release/2','source_commit':cfg['commit'],
        'membership':aliases,'spec_sha256':digest(out/'spec.json'),'depth':'foundation_stereo_depth',
        'sam':'validation only; SAM-free static flags and LoMa-G fit',
        'variants':['full','all','half:0..4','group:0..1','block:0..1','loco:each camera','start:1..8'],
        'selection':'established automatic G1-G6; no task exceptions; held cameras do not publish accepted rig'})
    inputs={str(out/'spec.json'):digest(out/'spec.json'),str(freeze):digest(freeze)}
    prefix=f'rwv2-20261113-calib-ours-{task}'
    jobs=[]
    def add(name,steps,deps=(),gpu=1,ram=16,**extra):
        jid=prefix+'-'+name
        jobs.append(dict(id=jid,gpu_profile='1g.24gb',priority=-3,gpu_gib=gpu,ram_gib=ram,cpus=4,
                         disk_gib=16,inputs=inputs,deps=[prefix+'-'+d for d in deps],steps=steps,**extra))
        return name
    def helper(cmd):
        return {'kind':'python','argv':['scripts/realworld_dsreset_calibration.py',cmd,'--spec',str(out/'spec.json')]}
    view=work/'view';matches=work/'matches';frames=work/'frames';flags=work/'flags';corr0=work/'corr-r0';corr2=work/'corr-r2'
    prep=add('view',[helper('view')],external_states={str(Path(cfg['state_dir'])/'state.json'):[j['id'] for j in selected]})
    env={'PYTHONPATH':str(Path(upstream)/'romav2/src')+':'+str(Path(upstream)/'LoMa/src')+':/opt/src/splatter4d',
         'TORCH_HOME':str(Path(upstream)/'torch-home')}
    matchjobs=[]
    for alias in aliases:
        matchjobs.append(add('roma-'+alias.rsplit('-',1)[1],[module('workers.romav2_match_worker',
            '--task-root',view,'--out',matches,'--times',6,'--num',5000,'--setting','precise','--seed',0,
            '--cameras',*CAMS,'--episodes',alias,python=TE,env=env,cap=20)],['view'],gpu=21,ram=32))
    add('frames',[module('rig_loma_data','--task-root',view,'--out',frames,'--matches',matches),
                  module('workers.static_flags_worker','--task-root',view,'--matches',matches,'--out',flags,python=TE),
                  helper('init')],matchjobs,ram=48)
    r0=[]
    fit=[a for n,a in enumerate(aliases) if n%5!=2][:25]
    for alias in fit:
        r0.append(add('loma0-'+alias.rsplit('-',1)[1],[module('workers.loma_render_match_worker',
            '--task-root',view,'--frames',frames,'--mesh',work/'mesh.npz','--state',work/'init.json',
            '--out',corr0,'--max-frames-per-camera',6,'--device','cpu','--episodes',alias,
            python=TE,env=env,cap=20)],['frames'],gpu=21,ram=48))
    base=['--task',task,'--matches',matches,'--flags',flags,'--processes',4,'--ee-sigma-logd',.03,'--mount']
    add('r0',[module('rig_loma_run',*base,'--corr',corr0,'--init',work/'init.json',
                    '--variant','all','--no-heldout','--out',work/'r0.json')],r0,ram=32)
    r2=[]
    for alias in aliases:
        r2.append(add('loma2-'+alias.rsplit('-',1)[1],[module('workers.loma_render_match_worker',
            '--task-root',view,'--frames',frames,'--mesh',work/'mesh.npz','--state',work/'r0.json',
            '--out',corr2,'--max-frames-per-camera',16,'--device','cpu','--episodes',alias,
            python=TE,env=env,cap=20)],['r0'],gpu=21,ram=48))
    add('evidence',[module('rig_dw_data','--task-root',view,'--out',work/'evidence'),
                    module('rig_dw_fixedness','--task-root',view,'--segments',1,'--out',work/'fixedness'),
                    module('rig_loma_depth','--stage','foundation_stereo_depth','--root',view,
                           '--corr',corr2,'--out-corr',work/'corrdepth-fs'),
                    module('rig_loma_depth','--stage','sensor_depth','--root',view,
                           '--corr',corr2,'--out-corr',work/'corrdepth-sensor')],r2,ram=32)
    base += ['--corr',corr2,'--init',work/'r0.json','--split','final']
    variants=['full','all']+[f'half:{k}' for k in range(5)]+[f'group:{k}' for k in range(2)]+[f'block:{k}' for k in range(2)]+[f'loco:{c.split("/")[0]}' for c in CAMS]+[f'start:{k}' for k in range(1,9)]
    solves=[]
    for variant in variants:
        name=variant.replace(':','-').replace('/','_')
        args=[*base,'--corr-depth-dir',work/'corrdepth-fs','--variant',variant,'--out',work/'results'/f'{name}.json']
        if variant=='full':args+=['--evidence',work/'evidence','--per-episode']
        elif variant.startswith('loco:'):args+=['--evidence',work/'evidence','--no-sam-validation']
        elif variant.startswith('start:'):args+=['--ref-result',work/'results/full.json']
        solves.append(add('solve-'+name,[module('rig_loma_run',*args)],['evidence']+(['solve-full'] if variant.startswith('start:') else []),ram=32))
    # Independent native-depth scale solve; no S2M2 backend or depth substitution.
    solves.append(add('sensor',[module('rig_loma_run',*base,'--corr-depth-dir',work/'corrdepth-sensor',
        '--depth-stage','sensor_depth','--variant','all','--out',work/'sensor.json')],['evidence'],ram=32))
    add('supplement',[module('workers.romav2_match_worker','--task-root',view,'--out',work/'matches-pairwise',
         '--cameras',*CAMS,'--pairing','pairwise','--supplement-of',matches,'--times',6,'--num',5000,
         '--setting','precise','--seed',0,python=TE,env=env,cap=20)],matchjobs,gpu=21,ram=32)
    add('static',[module('calib.criteria','static-eval','--full',work/'results/full.json','--corr',corr2,
        '--evidence',work/'evidence','--matches',matches,'--matches',work/'matches-pairwise','--task-root',view,
        '--check-frozen','--out',work/'static.json')],['solve-full','supplement'],ram=32)
    add('publish',[helper('publish')],[*solves,'static'],ram=16)
    atomic(out/'jobs.json',jobs)
    config={**cfg,'bundle':str(out),'jobs':str(out/'jobs.json'),'state_dir':str(out/'state'),
        'extra_readonly':list(dict.fromkeys([*cfg.get('extra_readonly',[]),str(Path(upstream).resolve()),str(Path(cfg['state_dir']))]))}
    Path(config['state_dir']).mkdir()
    atomic(out/'dispatcher.json',config)
    return len(jobs)


def view(spec):
    from s4d.realworld import io
    work=Path(spec['work'])
    for alias,r in spec['roots'].items():
        src=Path(r['root'])/'episodes'/r['episode']
        for name in ('normalized','derived/sensor_depth','derived/foundation_stereo_depth','derived/sam3_masks'):
            if not (src/name/io.RECORD).is_file():raise ValueError(f'phase A incomplete: {src/name}')
        dst=work/'view'/alias/'episodes'/alias
        dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists():dst.symlink_to(src,target_is_directory=True)
        elif dst.resolve()!=src.resolve():raise ValueError('calibration alias target changed')


def init(spec):
    import numpy as np
    from s4d.realworld.calib.loma.data import mask_free_init
    from s4d.realworld.pipeline import RobotGeometry,load_meta
    from s4d.realworld.calib.loma.render import build_robot_mesh
    work=Path(spec['work']);aliases=list(spec['aliases'])
    build_robot_mesh(RobotGeometry(load_meta(work/'view'/aliases[0],aliases[0])).model).save(work/'mesh.npz')
    for n,alias in enumerate(aliases):
        if n%5==2:continue  # established held-out split never enters initialisation
        result=mask_free_init(work/'view'/alias,alias)
        if all(c in result and result[c]['T'] is not None and np.isfinite(result[c]['T']).all() for c in CAMS):
            atomic(work/'init.json',{'T_cam_base':{c:result[c]['T'] for c in CAMS},
                'source_episode':spec['aliases'][alias],'method':'task-only mask_free_init; no SAM or other task rig'})
            return
    raise ValueError('task-local mask-free initialisation could not produce all camera poses')


def publish(spec,specpath):
    import numpy as np
    from s4d.realworld.calib import criteria as C
    from s4d.realworld.calib.loma.publish import load_results,release_verdict,build_document
    from s4d.realworld.calib.rig import publish_rig,rig_camera_status
    w=Path(spec['work']);R=load_results(w/'results');fixed=json.loads((w/'fixedness.json').read_text())
    static=json.loads((w/'static.json').read_text());C.check_static_eval(static,w/'results/full.json')
    verdict=release_verdict(R,fixed)
    atomic(w/'verdict-release1.json',verdict)
    ref=np.load(w/'fixedness.reference.npz')
    sources={k:{'path':str(p),'sha256':digest(p)} for k,p in {
        'freeze_record':Path(specpath).parent/'FREEZE.json','alias_map':Path(specpath),
        'verdict':w/'verdict-release1.json','static_eval':w/'static.json','final':w/'results/all.json',
        'full':w/'results/full.json','fixedness':w/'fixedness.json','reference_source':w/'fixedness.reference.npz'}.items()}
    doc=build_document(task=spec['task'],R=R,final=R['all'],fixed=fixed,frozen=verdict,
        ref_files={k:ref[k] for k in ref.files},depth_results={'sensor_depth':json.loads((w/'sensor.json').read_text())},
        eligible=[],commit=spec['commit'],sources=sources,
        user_decision='2026-11-14: automatic established criteria only; no new fixedness exception',
        criteria_v2={'static_eval':static,'fixreg':None})
    # Numeric aliases exist only inside the legacy calibration tools. Publish the
    # original label-inclusive identities so rig-mode G5b covers real episodes.
    mapping=spec['aliases']
    doc['episodes']=[mapping[e] for e in doc['episodes']]
    doc['reference_episode']=mapping[doc['reference_episode']]
    for c,rows in doc['fixedness_evidence'].items():
        doc['fixedness_evidence'][c]={mapping[e]:r for e,r in rows.items()}
    verdict2={c:rig_camera_status(doc,c)[0] for c in doc['cameras']}
    atomic(w/'automatic-decision.json',{'cameras':verdict2,'accepted':all(v=='validated' for v in verdict2.values())})
    if any(v!='validated' for v in verdict2.values()):raise ValueError('rig held by established automatic criteria; see automatic-decision.json')
    out=w/f"rig-{spec['task']}-accepted.json"
    if not out.exists():publish_rig(out,doc,{k:ref[k] for k in ref.files})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['plan','view','init','publish'])
    p.add_argument('--spec',type=Path);p.add_argument('--phase-config',type=Path);p.add_argument('--task')
    p.add_argument('--out',type=Path);p.add_argument('--work',type=Path);p.add_argument('--upstream',type=Path)
    a=p.parse_args()
    if a.command=='plan':print(json.dumps({'jobs':plan(a.phase_config,a.task,a.out,a.work,a.upstream),'launched':False}));return
    spec=json.loads(a.spec.read_text())
    if a.command=='publish':publish(spec,a.spec)
    else:globals()[a.command](spec)


if __name__=='__main__':main()
