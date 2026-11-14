"""Deterministic native-depth scene measurements on the exact SF training set."""
from __future__ import annotations
import hashlib
import json
import numpy as np
from s4d.rwlearn.dsreset import validate_manifest
from s4d.rwlearn.store import scene_statistics


def validate_report(report, manifest):
    digest=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    b=np.asarray(report.get('bounds',[]),float)
    look=np.asarray(report.get('lookat',[]),float)
    scale=report.get('pose_scale',0)
    if (report.get('schema')!='dsreset-scene-measurement/1' or report.get('accepted') is not True
        or report.get('task')!=manifest['task'] or report.get('depth_source')!='native_realsense'
        or report.get('manifest_sha256')!=digest or sorted(report.get('episodes',[]))!=sorted(manifest['split']['train'])
        or b.shape!=(6,) or not np.isfinite(b).all() or np.any(b[:3]>=b[3:])
        or look.shape!=(3,) or not np.isfinite(look).all() or not isinstance(scale,(float,int))
        or not np.isfinite(scale) or scale<=0
        or any(report.get('containment',{}).get(c,0)<.99 for c in ('robot','object','fixture'))
        or not report.get('native_provenance') or not report.get('scene_statistics')):
        raise ValueError('need accepted hash-bound train-only native scene measurement')
    return {'lookat':look.tolist(),'pose_scale':float(scale),'bounds':b.tolist()}


def measure(manifest,store,split):
    validate_manifest(manifest)
    if manifest['variant']!='SF':raise ValueError('measure the shared SF training partition')
    if not hasattr(store,'depth_provenance'):raise ValueError('verified native RealSense overlays required')
    from s4d.rwlearn.identity import store_identities
    identities=store_identities(store)
    by_sha={r['raw_sha256']:ep for ep,r in identities.items()}
    if len(by_sha)!=len(manifest['episodes']):raise ValueError('store membership differs')
    mapping={r['id']:by_sha.get(r['sha256']) for r in manifest['episodes']}
    if None in mapping.values():raise ValueError('store raw hashes differ')
    expected={s:sorted(mapping[e] for e in manifest['split'][s]) for s in ('train','validation')}
    if any(sorted(split[s])!=expected[s] for s in expected):raise ValueError('split differs from manifest')
    classes={1:[],2:[],3:[]}
    for ep in expected['train']:
        for cam in store.cameras(ep):
            count=len(store.timestamps(ep,cam));idx=np.linspace(0,count-1,min(8,count)).round().astype(int)
            dep=store.depth(ep,cam,idx);sem=store.semantic(ep,cam,idx)
            K=store.intrinsics(ep,cam);T=np.linalg.inv(store.w2c(ep,cam))
            for d,labels in zip(dep,sem):
                ys,xs=np.nonzero(np.isfinite(d)&(d>0))
                z=d[ys,xs];xyz=np.stack([(xs+.5-K[0,2])/K[0,0]*z,(ys+.5-K[1,2])/K[1,1]*z,z,np.ones_like(z)],1)@T.T
                for c in classes:
                    p=xyz[labels[ys,xs]==c,:3]
                    if len(p):classes[c].append(p)
    clouds={c:np.concatenate(p) for c,p in classes.items() if p}
    if set(clouds)!=set(classes):raise ValueError('need native robot/object/fixture points')
    lo=np.min([np.quantile(p,.001,axis=0) for p in clouds.values()],axis=0)-.01
    hi=np.max([np.quantile(p,.999,axis=0) for p in clouds.values()],axis=0)+.01
    bounds=np.r_[lo,hi].tolist()
    names={1:'robot',2:'object',3:'fixture'}
    containment={names[c]:float(((p>=lo)&(p<=hi)).all(1).mean()) for c,p in clouds.items()}
    stats=scene_statistics(store,expected['train'],bounds=bounds)
    report={'schema':'dsreset-scene-measurement/1','task':manifest['task'],'depth_source':'native_realsense',
        'manifest_sha256':hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest(),
        'episodes':manifest['split']['train'],'store_episodes':expected['train'],'bounds':bounds,
        'lookat':stats['lookat_m'],'pose_scale':stats['snerl_pose_scale_for_distance_4'],
        'containment':containment,'point_counts':{names[c]:len(p) for c,p in clouds.items()},
        'sampling':'8 evenly spaced frames per camera per SF TRAIN episode; all native valid pixels',
        'estimator':'union per-class [0.001,0.999] coordinate quantiles + 1 cm; require >=99% of every class',
        'native_provenance':store.depth_provenance(expected['train']),'scene_statistics':stats,
        'accepted':all(v>=.99 for v in containment.values())}
    validate_report(report,manifest)
    return report
