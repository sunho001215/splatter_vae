"""BH PREP namespace and fine-grained barriers; CPU fakes, no network/GPU/launches."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def isolated(source):
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'}
    p = subprocess.run([sys.executable, '-I', '-c', f'import sys;sys.path.insert(0,{str(ROOT)!r});'+source],
                       text=True, capture_output=True, cwd=ROOT, env=env)
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout)


def test_dedicated_context_has_no_pnp_identities():
    result = isolated('''from s4d.rwlearn.ball_hiding import bootstrap;bootstrap()
from s4d.rwlearn.pnpv2_spec import *
from ops.jobs import pnpv2_rgb as rgb, pnpv2_geometry as geo
s=load_spec(SPEC);p=values(s)
print(__import__('json').dumps(dict(task=TASK,stem=STEM,paths=p,rgb=rgb.RGB_JOB,geo=geo.JOB,
    rgb_source=rgb.SOURCE_STEM,split=s['collection'])))''')
    assert result['task'] == 'ball-hiding'
    assert result['stem'] == 'bh-20261120'
    assert result['rgb_source'] == 'bh-20261120-prepN-'
    assert 'pnpv2-20261117' not in json.dumps(result)
    assert result['split'] == dict(expected_success=100, validation_fraction=.1, split_seed=0, failures_enabled=False)


def test_context_refuses_switch_after_consumers_imported():
    import ops.jobs.pnpv2_rgb  # noqa: F401
    from s4d.rwlearn.pnpv2_spec import select_task, TASK
    with pytest.raises(ValueError, match='before importing'):
        select_task('ball-hiding')
    assert TASK == 'pick-and-place-v2'


def test_split_preserves_pipeline_and_per_episode_dependency():
    from s4d.rwlearn.ball_hiding import split_jobs
    jobs = [dict(id=f'bh-20261120-prepA-e{i}', episode=f'e{i}', phase='A', inputs={'config':'sha'}, deps=[],
                 steps=[dict(kind='rwprep', argv=['episode','--pipeline','config','--stages',
                                                  'ingest,sensor_depth,foundation_stereo_depth,sam3_masks'])]) for i in range(2)]
    before = copy.deepcopy(jobs)
    norm, phase = split_jobs(jobs, '/remote/normalize')
    assert jobs == before
    for n,a,orig in zip(norm,phase,before):
        assert n['inputs'] == a['inputs'] == orig['inputs']
        assert n['steps'][0]['argv'][-1] == 'ingest'
        assert a['steps'] == orig['steps']
        assert a['external_states'] == {'/remote/normalize/state/state.json':[n['id']]}
        assert n['id'] == orig['id'].replace('-prepA-', '-prepN-')


def test_rgb_wave_requires_full_normalization_but_not_depth():
    result=isolated('''from s4d.rwlearn.ball_hiding import bootstrap;bootstrap()
from ops.jobs.pnpv2_registration import ready_waves
from ops.jobs.pnpv2_plan import STAGES
ev={k:{'status':'pending'} for k in STAGES}
for k in ('arrival','check-data','ingest','normalized','rgb-publication'):ev[k]={'status':'done'}
import json
print(json.dumps(dict(ready=ready_waves(ev),dependencies=STAGES['wave:rgb'][1])))''')
    assert result['ready'] == ['rgb']
    assert 'phaseA' not in result['dependencies']


def test_bh_remote_workers_select_context(tmp_path):
    manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    result=isolated(f'''from s4d.rwlearn.ball_hiding import bootstrap;bootstrap()
from s4d.rwlearn.pnpv2_spec import load_spec,SPEC
from ops.jobs.pnpv2_rgb import remote_job
import json
print(json.dumps(remote_job(load_spec(SPEC),{str(tmp_path)!r},'/normalize',['n1'])))''')
    assert result['steps'][0]['argv'][:3] == ['-m','ops.jobs.ball_hiding_worker','rgb']
    assert result['external_states'] == {'/normalize/state/state.json':['n1']}


def test_bh_override_is_refused():
    result=isolated('''from s4d.rwlearn.ball_hiding import bootstrap;bootstrap()
from s4d.rwlearn.pnpv2_offline import no_override
try:no_override({'acceptance_override':{'task':'ball-hiding'}})
except ValueError:print('true')
else:print('false')''')
    assert result is True


def test_registration_not_exposed():
    p=subprocess.run([sys.executable,'-I',str(ROOT/'scripts/ball_hiding.py'),'register'],cwd=ROOT,
                     capture_output=True,text=True)
    assert p.returncode != 0 and 'PREP-only' in p.stderr


def test_bh_release_mounts_only_its_own_output(tmp_path):
    from ops.jobs import rwprep_remote as R
    args=R.protect_processed(['docker','run','image'],str(tmp_path),'bh-20261120')
    assert f'type=bind,src={tmp_path}/processed/bh-20261120,dst={tmp_path}/processed/bh-20261120' in args
    assert f'type=bind,src={tmp_path}/processed,dst={tmp_path}/processed,readonly' in args
    assert R.RELEASE_STEMS['bh-20261120']=='bh-20261120-'
    # The remote guard embeds this standalone function; it must accept the new release too.
    import inspect
    ns={'Path':Path};exec(inspect.getsource(R.protect_processed),ns)
    assert ns['protect_processed'](['docker','run','image'],str(tmp_path),'bh-20261120')==args


def test_bh_producer_refuses_partial_data_on_cpu(tmp_path):
    p=subprocess.run([sys.executable,'-I',str(ROOT/'scripts/ball_hiding.py'),'--entry','manifest',
                      '--raw-root',str(tmp_path)],cwd=ROOT,capture_output=True,text=True,
                     env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
    assert p.returncode != 0
    assert 'arrival.json' in p.stderr or 'dataset not complete' in p.stderr
    assert 'gpu_guard' not in p.stderr


def test_two_bundles_use_hash_bound_copies_and_per_episode_deps(tmp_path):
    result=isolated(f'''from s4d.rwlearn.ball_hiding import bootstrap;bootstrap()
import contextlib,io,json
from pathlib import Path
from types import SimpleNamespace
from scripts import pick_and_place_v2 as P
from ops.jobs import rwprep_remote as R
from s4d.rwlearn.pnpv2_spec import load_spec,SPEC,values
from s4d.rwlearn.ball_hiding import phase_a_remote
import yaml
root=Path({str(tmp_path)!r});s=load_spec(SPEC);s['roots']['work']=str(root/'work');p=values(s)
Path(p['prep_plan']).parent.mkdir(parents=True)
cfg=root/'pipeline.yaml';ep='ball-hiding-success-episode_000000'
cfg.write_text(yaml.safe_dump({{'output_root':'/unused','adapter':{{'config':{{'episodes':{{ep:'/home/ws/data/real_world/ball-hiding/success/episode_000000.h5'}}}}}},'stages':{{}}}}))
Path(p['prep_plan']).write_text(json.dumps([{{'episode':ep,'config':str(cfg),'raw_sha256':'a'*64}}]))
P.ATTACH_ROOT=root/'runs';P._check_data=lambda *a:{{}};P._remote_raw_verified=lambda *a:{{}}
R.RemoteClient=lambda:None
staged=[]
def stage(jobs,out,client,**kw):
    for j in jobs:
        path=next(k for k in j['inputs'] if k.endswith('.yaml'))
        assert Path(path).is_relative_to(out)
        assert R.digest(path)==j['inputs'][path]
        assert j['disk_gib']==4
    R.atomic(out/'dispatcher.json',{{'image':'sha256:'+'b'*64}})
    staged.append({{'name':kw['name'],'jobs':jobs}})
    return {{'registered':len(jobs),'launched':False}}
R.stage=stage
with contextlib.redirect_stdout(io.StringIO()):phase_a_remote(s,SimpleNamespace(),[])
print(json.dumps(staged))''')
    norm,phase=result
    assert norm['name']=='bh-20261120-normalize'
    assert phase['name']=='bh-20261120-phaseA'
    assert phase['jobs'][0]['external_states']=={
        '/home/compu/kaist/sunho/realworld/configs/bh-20261120-normalize/state/state.json':[norm['jobs'][0]['id']]}
    assert norm['jobs'][0]['steps'][0]['argv'][-1]=='ingest'
