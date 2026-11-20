"""Read-only BH schema/timing audit; invoke with nice19/ionice3 and <=4 BLAS threads."""
import argparse, hashlib, io, json, time
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import h5py
import numpy as np
from PIL import Image, ImageDraw


def plain(x):
    if isinstance(x, bytes): return x.decode(errors='replace')
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    return x


def schema(f):
    out = {}
    def visit(n, obj):
        rec = {'kind': 'dataset' if isinstance(obj, h5py.Dataset) else 'group',
               'attrs': {k: plain(v) for k,v in obj.attrs.items()}}
        if isinstance(obj, h5py.Dataset):
            rec.update(dtype=str(obj.dtype), tail_shape=list(obj.shape[1:]), rows=obj.shape[0] if obj.shape else None)
        out[n] = rec
    visit('/', f)
    f.visititems(visit)
    return out


def audit(path, reference):
    before = path.stat()
    rec = {'file': str(path), 'bytes': before.st_size, 'mtime_ns':before.st_mtime_ns, 'anomalies': [], 'timestamps': {}, 'sync': {}}
    with h5py.File(path, 'r') as f:
        s = schema(f)
        rec['schema'] = s
        rec['schema_diff'] = {'missing': sorted(set(reference)-set(s)), 'added': sorted(set(s)-set(reference)),
             'types': {k: {'reference': reference[k], 'actual': s[k]} for k in set(s)&set(reference)
                       if any(s[k].get(a)!=reference[k].get(a) for a in ('kind','dtype','tail_shape'))},
             'attribute_changes': {k: {'reference': reference[k]['attrs'], 'actual':s[k]['attrs']}
                                   for k in set(s)&set(reference) if s[k]['attrs']!=reference[k]['attrs']}}
        if rec['schema_diff']['missing'] or rec['schema_diff']['types']: rec['anomalies'].append('schema mismatch')
        rec['cameras'] = {}
        for cam,g in f['cameras'].items():
            info = json.loads(g.attrs['info_json'])
            rec['cameras'][cam] = {'serial':info['serial_number'], 'info':info,
               'frames':len(g['color']), 'stereo_frames':len(g['stereo/t']), 'depth_present':'depth' in g['stereo']}
        def times(n,o):
            if not isinstance(o,h5py.Dataset) or o.ndim!=1 or o.dtype.kind not in 'fi': return
            if n.rsplit('/',1)[-1] not in ('t','t_active','t_recv','hw_timestamp_ms','frame_timestamp_us','sensor_timestamp_us'): return
            a=np.asarray(o[:],dtype=float); d=np.diff(a); unit=1e-3 if n.endswith('_ms') else 1e-6 if n.endswith('_us') else 1
            d=d*unit; pos=d[d>0]; med=float(np.median(pos)) if len(pos) else None
            out={'n':len(a),'finite':bool(np.isfinite(a).all()),'strict_monotonic':bool((d>0).all()),
                'nonpositive':int((d<=0).sum()),'duration_s':float((a[-1]-a[0])*unit) if len(a)>1 else 0,
                'median_rate_hz':1/med if med else None,'max_gap_s':float(d.max()) if len(d) else None,
                'gaps_over_1_5x_median':int((d>1.5*med).sum()) if med else None}
            group=f[n.rsplit('/',1)[0]]
            segment=group.get('segment')
            if segment is None and n.startswith('robot/'):
                segment=f.get('action/segment')
            if segment is not None and len(segment)==len(a):
                same=np.diff(segment[:])==0
                out['within_segment_max_gap_s']=float(d[same].max()) if same.any() else None
                out['between_segment_gaps']=int((~same).sum())
            rec['timestamps'][n]=out
            if not out['finite'] or out['nonpositive']: rec['anomalies'].append(n+': invalid/nonmonotonic')
            if med and (d>1.5*med).any(): rec['anomalies'].append(n+': timing gaps')
        f.visititems(times)
        cams=list(f['cameras'])
        for i,c in enumerate(cams):
            a=f['cameras'][c]['t'][:]
            for bcam in cams[i+1:]:
                b=f['cameras'][bcam]['t'][:]; idx=np.searchsorted(b,a); l=np.clip(idx-1,0,len(b)-1); r=np.clip(idx,0,len(b)-1)
                diff=np.minimum(abs(a-b[l]),abs(a-b[r]))*1000
                if (diff>20).any():rec['anomalies'].append(c+'__'+bcam+': nearest-camera offset >20 ms; advisory, M5 motion gate applies')
                rec['sync'][c+'__'+bcam]={'nearest_ms_p50':float(np.median(diff)),'nearest_ms_p95':float(np.quantile(diff,.95)),
                    'nearest_ms_max':float(diff.max()),'over_20ms':int((diff>20).sum())}
        rec['sample_images']={}
        for cam,g in f['cameras'].items():
            for stream in ('color','stereo/left','stereo/right','stereo/depth'):
                ds=g[stream]; samples=[]
                for i in sorted({0,len(ds)//2,len(ds)-1}):
                    im=np.asarray(Image.open(io.BytesIO(ds[i].tobytes())))
                    samples.append({'frame':i,'shape':list(im.shape),'dtype':str(im.dtype),
                                    'min':int(im.min()),'max':int(im.max()),'nonzero_fraction':float((im!=0).mean())})
                rec['sample_images'][cam+'/'+stream]=samples
            for flag in ('depth_ok','pair_ok','t_valid'):
                values=g['stereo/'+flag][:]
                if not (values==1).all():rec['anomalies'].append(cam+'/stereo/'+flag+': invalid samples')
        rec['numeric_fields']={}
        for group in ('robot','action','gripper'):
            for k,v in f[group].items():
                a=v[:]
                rec['numeric_fields'][group+'/'+k]={'shape':list(a.shape),'finite':bool(np.isfinite(a).all()),
                                      'min':float(a.min()),'max':float(a.max())}
                if not np.isfinite(a).all():rec['anomalies'].append(group+'/'+k+': nonfinite')
        if f.attrs.get('status')!='complete': rec['anomalies'].append('recording incomplete')
    from s4d.rwlearn.dsreset import inspect_raw
    try:rec['canonical_validation']={'passed':True, 'report':inspect_raw(path)}
    except (ValueError,KeyError,OSError) as e:
        rec['canonical_validation']={'passed':False,'error':str(e)};rec['anomalies'].append('canonical validation failed: '+str(e))
    after=path.stat()
    rec['stable_during_read']=(before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    if not rec['stable_during_read']: rec['anomalies'].append('file changed during audit')
    return rec


def contact(files,out):
    rows=[]
    for p in files:
        with h5py.File(p,'r') as f:
            for c,g in f['cameras'].items():
                ims=[]
                for idx in (0,len(g['color'])//2,len(g['color'])-1):
                    im=Image.open(io.BytesIO(g['color'][idx].tobytes())).convert('RGB'); im.thumbnail((320,240))
                    tile=Image.new('RGB',(320,262),'white');tile.paste(im,(0,22));ImageDraw.Draw(tile).text((2,3),f'{p.stem} {c} #{idx}',fill='black');ims.append(tile)
                rows.append(ims)
    canvas=Image.new('RGB',(960,262*len(rows)), 'white')
    for r,ims in enumerate(rows):
        for c,im in enumerate(ims):canvas.paste(im,(c*320,r*262))
    canvas.save(out,quality=85)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/home/ws/data/real_world/ball-hiding/success'))
    p.add_argument('--out',type=Path,default=Path('runs/bh-data-evidence'));a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    refs=sorted(Path('/home/ws/data/real_world/pick-and-place-v2/success').glob('episode_*.h5'))
    with h5py.File(refs[0],'r') as f: reference=schema(f)
    (a.out/'pnp-reference-schema.json').write_text(json.dumps(reference,indent=2))
    ref_shapes={}
    for file in refs:
        with h5py.File(file,'r') as f:
            sch=schema(f)
            ref_shapes[file.name]={'fields':{n:{k:v for k,v in info.items() if k not in ('rows','attrs')} for n,info in sch.items()},
                                  'root_attrs':sch['/']['attrs'],
                                  'attribute_names':{n:sorted(info['attrs']) for n,info in sch.items()}}
    (a.out/'pnp-all-reference-schemas.json').write_text(json.dumps(ref_shapes,indent=2))
    files=sorted(a.root.glob('episode_*.h5')); results=[]
    for file in files:
        dest=a.out/(file.stem+'.json')
        if dest.exists() and 'within_segment_max_gap_s' in json.loads(dest.read_text()).get('timestamps',{}).get('robot/t',{}) and 'canonical_validation' in json.loads(dest.read_text()) and json.loads(dest.read_text()).get('mtime_ns')==file.stat().st_mtime_ns: rec=json.loads(dest.read_text())
        else:
            try:rec=audit(file,reference)
            except (OSError,ValueError,KeyError) as e:rec={'file':str(file),'anomalies':[str(e)],'stable_during_read':False}
            if rec['stable_during_read']:dest.write_text(json.dumps(rec,indent=2))
        results.append(rec)
    stable=[Path(r['file']) for r in results if r['stable_during_read']]
    if stable:
        selected=[stable[i] for i in sorted({0,len(stable)//2,len(stable)-1})]
        contact(selected,a.out/'contact-sheet.jpg')
        (a.out/'contact-sheet.json').write_text(json.dumps({'episodes':[str(p) for p in selected], 'frames':'first/middle/last', 'cameras':'all'},indent=2))
    expected={f'episode_{i:06d}.h5' for i in range(102) if i not in (25,54)}
    summary={'at_unix':time.time(),'episodes_found':len(files),'episodes_audited':len(stable),'expected':100,
       'marker_present':a.root.with_name('success.VERIFIED').is_file(), 'missing':sorted(expected-{p.name for p in files}),
       'unexpected':sorted({p.name for p in files}-expected),
       'anomalies':{Path(r['file']).name:r['anomalies'] for r in results if r['anomalies']},
       'verdict':'PARTIAL; transfer in progress' if len(stable)!=100 else 'AUDITED; require canonical check-data and transfer marker',
       'canonical_passed':sum(r.get('canonical_validation',{}).get('passed') is True for r in results),
       'schema_mismatches':{Path(r['file']).name:r['schema_diff'] for r in results if r.get('schema_diff') and
             any(r['schema_diff'][k] for k in ('missing','added','types'))},
       'writes_to_data':False,'robot_pc_io':False}
    (a.out/'data-summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
