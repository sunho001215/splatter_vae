"""BH3 membership and immutable re-staging regressions; CPU fakes only."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from tests.test_ball_hiding_prep import isolated
from s4d.rwlearn import ball_hiding_membership as M


def test_committed_membership_uses_original_split_rule():
    result = isolated('''from s4d.rwlearn.ball_hiding import bootstrap;bootstrap()
import json, numpy as np
from pathlib import Path
from s4d.rwlearn.ball_hiding_membership import amendment,SPLIT,validate_membership
from s4d.rwlearn.store import split_episodes
from tests.test_pnpv2_geometry_route import FakeStore
s=json.loads(SPLIT.read_text());a=amendment()
rows=[dict(id='ball-hiding-success-'+Path(n).stem,path='/raw/ball-hiding/success/'+n,sha256=h) for n,h in a['episodes'].items()]
r=split_episodes(FakeStore({'episodes':rows}),seed=0,validation_fraction=.1)
assert {k:r[k] for k in ('train','validation')}=={k:s[k] for k in ('train','validation')}
print(json.dumps({'train':len(r['train']),'validation':len(r['validation']),'changed':len(s['changes_from_100'])}))''')
    assert result == dict(train=89, validation=10, changed=14)
    rec = M.amendment()['excluded']['ball-hiding/episode_000008.h5']
    assert (rec['camera'], rec['frame_index'], rec['flag'], rec['value'], rec['frames']) == (
        'cam_135122073201', 121, 'stereo/depth_ok', False, 150)
    assert rec['raw_sha256'] == '364c3eff063cd9ab495a2455a9bd7392dba8a8a3583de199e74b29858833c322'


def test_recorder_evidence_requires_exact_bytes_and_flags(tmp_path, monkeypatch):
    import h5py
    import numpy as np
    path = tmp_path/'episode_000008.h5'
    def write(flagged):
        with h5py.File(path, 'w') as f:
            for cam in ('cam_135122070988','cam_135122073201','cam_135122076054'):
                for key in ('t_valid','stereo/t_valid','stereo/depth_ok'):
                    a=np.ones(150,dtype='i8')
                    if flagged and (cam,key)==('cam_135122073201','stereo/depth_ok'):a[121]=0
                    f.create_dataset('cameras/'+cam+'/'+key,data=a)
    write(True)
    a=M.amendment();a['excluded']['ball-hiding/'+path.name]['raw_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(M,'amendment',lambda:a)
    assert M.verify_rejection(path)['status']=='invalid'
    write(False)
    with pytest.raises(ValueError,match='sha256'):M.verify_rejection(path)
    a['excluded']['ball-hiding/'+path.name]['raw_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='evidence'):M.verify_rejection(path)


@pytest.mark.parametrize('bundle,stage',[('normalize','prepN'),('phaseA','prepA')])
def test_advancer_ignores_only_exact_exclusion_and_requires_all_99(tmp_path,monkeypatch,bundle,stage):
    from tools.ball_hiding import advance as A
    monkeypatch.setattr(A,'WORK',tmp_path)
    root=tmp_path/bundle/'state';root.mkdir(parents=True)
    states={f'bh-20261120-{stage}-ball-hiding-success-'+Path(n).stem:{'status':'done'}
            for n in M.amendment()['episodes']}
    bad=f'bh-20261120-{stage}-ball-hiding-success-episode_000008'
    states[bad]={'status':'failed'}
    def save(): (root/'state.json').write_text(json.dumps({'jobs':states}))
    save();assert A.done(bundle)
    active=next(i for i in states if i!=bad)
    states[active]['status']='failed';save()
    with pytest.raises(RuntimeError,match='failed/unknown'):A.done(bundle)
    states[active]['status']='pending';save();assert not A.done(bundle)
    del states[active];save();assert not A.done(bundle)
    states[active]={'status':'done'};states['unexpected-episode_000008']={'status':'failed'};save()
    with pytest.raises(RuntimeError,match='unexpected jobs'):A.done(bundle)
    del states['unexpected-episode_000008'];save();(root/'STOP').touch()
    with pytest.raises(RuntimeError,match='STOP'):A.done(bundle)


def test_99_rgb_roots_and_geometry_keep_original_queue_hash(tmp_path):
    result=isolated(f'''from s4d.rwlearn.ball_hiding import bootstrap;bootstrap()
import json,pytest
from pathlib import Path
from tests import test_pnpv2_geometry_route as T
from ops.jobs import pnpv2_rgb as RGB
root=Path({str(tmp_path)!r})
with pytest.MonkeyPatch.context() as mp:
 c=T.campaign.__wrapped__(root,mp)
 bundle=root/'A';processed=Path(c.p['remote_processed'])
 cfg,jobs=T.phase(bundle,c.m,processed=processed)
 bad='ball-hiding-success-episode_000008'
 jobs.append(dict(id='bh-20261120-prepA-'+bad,episode=bad,phase='A',inputs={{'/do-not-read.yaml':'x'}}))
 T.write(bundle/'jobs.json',jobs)
 state={{'queue_sha256':T.G.digest(bundle/'jobs.json'),'jobs':{{j['id']:{{'status':'done'}} for j in jobs}}}}
 state['jobs'][jobs[-1]['id']]['status']='failed';T.write(bundle/'state/state.json',state)
 T.G.completed_phase(bundle,c.m,'A',processed)
 assert T.reg.dispatcher_evidence(bundle,{{r['id'] for r in c.m['episodes']}},'bh-20261120-prepA')['status']=='done'
 # Remote normalized roots use a separate 100-job fixture with N ids and local empty completion markers.
 for j in jobs:
  j['id']=j['id'].replace('-prepA-','-prepN-')
 T.write(bundle/'jobs.json',jobs)
 state['jobs']={{j['id']:{{'status':'done' if j['episode']!=bad else 'failed'}} for j in jobs}}
 T.write(bundle/'state/state.json',state)
 import yaml
 for j in jobs[:-1]:
  p=next(iter(j['inputs']));cfg=yaml.safe_load(Path(p).read_text());cfg['output_root']=str(root/'outputs'/j['episode']);Path(p).write_text(yaml.safe_dump(cfg))
  target=Path(cfg['output_root'])/'episodes'/j['episode']/RGB.STAGE_COMPLETE;target.parent.mkdir(parents=True);target.touch()
 roots=RGB.phase_a_roots(bundle,[r['id'] for r in c.m['episodes']]);assert len(roots)==99 and bad not in roots
 print(json.dumps({{'roots':len(roots),'original_jobs':len(jobs)}}))''')
    assert result==dict(roots=99,original_jobs=100)


def test_new_rgb_bundle_requires_99_and_current_source(tmp_path):
    result=isolated(f'''from s4d.rwlearn.ball_hiding import bootstrap;bootstrap()
import contextlib,io,json,pytest
from pathlib import Path
from types import SimpleNamespace
from tests import test_pnpv2_geometry_route as T
from scripts import pick_and_place_v2 as P
from ops.jobs import rwprep_remote as R
root=Path({str(tmp_path)!r})
with pytest.MonkeyPatch.context() as mp:
 c=T.campaign.__wrapped__(root,mp)
 norm=Path(c.p['rgb_source']);norm.mkdir(parents=True)
 jobs=[{{'episode':r['id'],'id':'bh-20261120-prepN-'+r['id']}} for r in c.m['episodes']+c.m['excluded']]
 T.write(norm/'jobs.json',jobs);T.write(norm/'dispatcher.json',{{'jobs':str(norm/'jobs.json'),'commit':'old','image':'image','remote':'/remote/normalize'}})
 mp.setattr(P,'ATTACH_ROOT',root/'runs');mp.setattr(P,'_check_data',lambda *a:{{'passed':True}});mp.setattr(R,'RemoteClient',lambda:None)
 calls=[]
 def stage(jobs,out,client,**kw):
  assert kw['commit'] is None and kw['name']=='bh-20261120-rgb2'
  assert len(jobs[0]['external_states']['/remote/normalize/state/state.json'])==99
  assert not any('episode_000008' in i for i in jobs[0]['external_states']['/remote/normalize/state/state.json'])
  calls.append(kw);return {{'launched':False}}
 mp.setattr(R,'stage',stage)
 with contextlib.redirect_stdout(io.StringIO()):P.rgb_remote(c.spec,SimpleNamespace(),[])
 assert (Path(c.p['work'])/'rgb2-remote').is_symlink()
 print(json.dumps({{'bundles':len(calls)}}))''')
    assert result==dict(bundles=1)


def test_calibration_and_phase_b_plan_99_without_rewriting_phase_a(tmp_path):
    result=isolated(f'''from s4d.rwlearn.ball_hiding import bootstrap;bootstrap()
import contextlib,io,json,pytest,sys
from pathlib import Path
from tests import test_pnpv2_geometry_route as T
from scripts import pick_and_place_v2 as P
from s4d.rwlearn import pnpv2_offline as offline
root=Path({str(tmp_path)!r})
with pytest.MonkeyPatch.context() as mp:
 c=T.campaign.__wrapped__(root,mp)
 bundle=root/'A';cfg,jobs=T.phase(bundle,c.m)
 cfg.update(state_dir=str(bundle/'state'),priority_file=str(root/'priority/bundle.json'),code='/old',commit='old')
 jobs.append(dict(id='bh-20261120-prepA-ball-hiding-success-episode_000008',episode='ball-hiding-success-episode_000008',inputs={{'/never-read-excluded.yaml':'x'}}))
 T.write(bundle/'jobs.json',jobs);T.write(bundle/'dispatcher.json',cfg)
 rows=[{{'episode':j['episode'],'config':next(iter(j['inputs'])),'raw_sha256':'a'*64}} for j in jobs]
 T.write(bundle/'prep-plan.json',rows)
 before={{p:p.read_bytes() for p in bundle.rglob('*') if p.is_file()}}
 out=root/'calib2'
 mp.setattr(sys,'argv',['plan',str(bundle/'dispatcher.json'),'ball-hiding',str(out),str(root/'calibration-work'),str(root/'upstream'),'bh-20261120-calib','[]','fresh','/fresh-code'])
 with contextlib.redirect_stdout(io.StringIO()):exec(P._CALIB_PLAN,{{}})
 planned=T.G.load(out/'jobs.json');spec=T.G.load(out/'spec.json');dispatcher=T.G.load(out/'dispatcher.json')
 assert len(planned)==253 and len(spec['aliases'])==99
 assert len(planned[0]['external_states'][str(bundle/'state/state.json')])==99
 assert dispatcher['commit']=='fresh' and dispatcher['code']=='/fresh-code'
 assert len(spec['membership_amendment']['episodes'])==99
 assert before=={{p:p.read_bytes() for p in bundle.rglob('*') if p.is_file()}}
 # The Phase-B subprocess must receive a NEW 99-row plan, not the original 100-row plan.
 outb=root/'B';captured=[]
 mp.setattr(offline,'phase_b_compiler',lambda *a:'compiler.py')
 def compile(args,**kw):
  selected=T.G.load(args[args.index('--plan')+1]);captured.extend(selected)
  assert len(selected)==99 and all('episode_000008' not in r['episode'] for r in selected)
  T.write(outb/'jobs.json',[{{'id':r['episode'],'disk_gib':32}} for r in selected])
 import subprocess
 mp.setattr(subprocess,'run',compile)
 cfg.update(processed='/processed',raw='/raw',weights='/weights',image='image',tools='/tools',assets='/assets')
 T.write(bundle/'dispatcher.json',cfg)
 before={{p:p.read_bytes() for p in bundle.rglob('*') if p.is_file()}}
 mp.setattr(sys,'argv',['plan',str(bundle),str(outb),str(root/'rig.json'),'bh-20261120-prepB','/calibration','4','automatic'])
 with contextlib.redirect_stdout(io.StringIO()):exec(P._PHASE_B_PLAN,{{}})
 assert len(captured)==99 and len(T.G.load(outb/'jobs.json'))==99
 assert all(j['disk_gib']==4 for j in T.G.load(outb/'jobs.json'))
 assert before=={{p:p.read_bytes() for p in bundle.rglob('*') if p.is_file()}}
 print(json.dumps({{'calibration':len(planned),'phaseB':len(captured)}}))''')
    assert result==dict(calibration=253,phaseB=99)


def test_shared_amendment_loader_accepts_evidence_bound_record():
    from s4d.rwlearn.gate import load_amendments
    record=load_amendments()['ball-hiding']['record']
    assert record==M.amendment()
    doc=M.ROOT/record['document']['path']
    assert hashlib.sha256(doc.read_bytes()).hexdigest()==record['document']['sha256']


def test_manifest_rejects_old_split_missing_episode_or_changed_hash(tmp_path):
    assert isolated(f'''from s4d.rwlearn.ball_hiding import bootstrap;bootstrap()
import copy,json,pytest
from pathlib import Path
from tests import test_pnpv2_geometry_route as T
from s4d.rwlearn.pnpv2_publication import manifest_check
with pytest.MonkeyPatch.context() as mp:
 c=T.campaign.__wrapped__(Path({str(tmp_path)!r}),mp)
 manifest_check(c.m,c.spec)
 for mutation in ('split','missing','hash','amendment'):
  m=copy.deepcopy(c.m)
  if mutation=='split':
   x=m['split']['train'].pop();y=m['split']['validation'].pop();m['split']['train'].append(y);m['split']['validation'].append(x)
   for r in m['episodes']:
    if r['id']==x:r['split']='validation'
    elif r['id']==y:r['split']='train'
  elif mutation=='missing':m['episodes'].pop()
  elif mutation=='hash':m['episodes'][0]['sha256']='0'*64
  else:m['amendment']['decision']='changed'
  with pytest.raises(ValueError):manifest_check(m,c.spec)
 print('true')''') is True


def test_rgb_fetch_accepts_99_and_uses_new_bundle(tmp_path):
    assert isolated(f'''from s4d.rwlearn.ball_hiding import bootstrap;bootstrap()
import contextlib,io,json,pytest,yaml
from pathlib import Path
from types import SimpleNamespace
from tests import test_pnpv2_geometry_route as T
from scripts import pick_and_place_v2 as P
from ops.jobs import pnpv2_rgb as RGB
from ops.remote import access
with pytest.MonkeyPatch.context() as mp:
 c=T.campaign.__wrapped__(Path({str(tmp_path)!r}),mp)
 T.write(Path(c.p['work'])/'rgb2-remote/state/state.json',{{'jobs':{{RGB.RGB_JOB:{{'status':'done'}}}}}})
 remote=RGB.remote_out(c.spec)
 docs={{name:json.dumps({{}}) for name in RGB.DOCS}}
 docs['manifest.json']=json.dumps(c.m)
 docs['store.yaml']=yaml.safe_dump({{'kind':'compact','paths':[remote+'/episodes/'+r['id'] for r in c.m['episodes']]}})
 class Client:
  def checked_ssh(self,*a,**kw):return SimpleNamespace(stdout=json.dumps(docs))
 mp.setattr(access,'RemoteClient',Client)
 with contextlib.redirect_stdout(io.StringIO()):P.rgb_fetch(c.spec,None,[])
 assert len(yaml.safe_load(Path(c.p['rgb_store_local']).read_text())['paths'])==99
 print('true')''') is True
