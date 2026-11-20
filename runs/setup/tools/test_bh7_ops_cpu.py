"""BH7 prepared tooling checks; fake registry/remote, no live writes or native acceptance claim."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT/'runs/setup/tools')]
import bh_register as reg
import bh_remote_pin as remote


def history():
    identity=reg.BH7_REFUSED_ID
    return [dict(event='launching',id=identity,attempt=1,host='remote',commit=reg.BH6_COMMIT,
                 script='runs/setup/tools/bh_realworld_fql_dataset.py',gpu='fake-gpu',container='s4d-'+identity+'-a1'),
            dict(event='unknown',id=identity,reason='container missing from confirmed snapshot'),
            dict(event='reset',id=identity,reconciliation=dict(job_id=identity,attempt=1,gpu='fake-gpu',
                container='s4d-'+identity+'-a1',launch_attempt_age_seconds=601,
                conclusion='launch refused before docker run; deterministic container never existed',
                checks=[dict(containers_with_name=0,remote_output='absent',accepted_release_dir='exists')]*2))]


def test_only_reconciled_exact_never_executed_history_is_allowed():
    events=history();reg.require_unlaunched(events,{reg.BH7_REFUSED_ID})
    reg.require_unlaunched([],{reg.BH7_REFUSED_ID})
    for field,value in [('event','launched'),('attempt',2),('commit','a'*40)]:
        bad=copy.deepcopy(events);bad[0][field]=value
        with pytest.raises(ValueError):reg.require_unlaunched(bad,{reg.BH7_REFUSED_ID})
    for bad in [events[:-1],events+[dict(event='launched',id=reg.BH7_REFUSED_ID)],
                [{**e,'id':'foreign'} for e in events]]:
        with pytest.raises(ValueError):reg.require_unlaunched(bad,{bad[0]['id']})
    bad=copy.deepcopy(events);bad[-1]['reconciliation']['checks'][0]['remote_output']='exists'
    with pytest.raises(ValueError):reg.require_unlaunched(bad,{reg.BH7_REFUSED_ID})


def test_host_release_mixed_pins_require_explicit_map():
    p=dict(release=dict(commit='a'*40,image='image'),rows=[dict(host='local',planned_commit='a'*40,planned_image='image'),
        dict(host='remote',planned_commit='b'*40,planned_image='image')])
    assert reg.host_release(p,'local')['commit']=='a'*40
    with pytest.raises(ValueError):reg.host_release(p,'remote')
    p['release_by_host']={'remote':dict(commit='b'*40,image='image')}
    assert reg.host_release(p,'remote')['commit']=='b'*40
    p['rows'][1]['planned_image']='drift'
    with pytest.raises(ValueError):reg.host_release(p,'remote')


def test_shipped_aliases_are_exact_id_only():
    from ops.jobs import realworld_local,remote_offload
    from s4d.rwlearn import dsreset_entrypoints
    modules=(realworld_local,remote_offload,dsreset_entrypoints)
    previous=[m.legacy_row for m in modules]
    try:
        remote.install_remote_aliases({reg.BH7_REFUSED_ID})
        row=dict(id=reg.BH7_REFUSED_ID,script='scripts/bh_realworld_fql_dataset.py')
        for module in modules:
            assert module.legacy_row(row)=={**row,'script':'scripts/realworld_fql_dataset.py'}
            foreign={**row,'id':'foreign'}
            assert module.legacy_row(foreign)==foreign
    finally:
        for module,fn in zip(modules,previous):module.legacy_row=fn


def test_established_reconciler_checks_dataset_directory_twice(tmp_path,monkeypatch):
    from ops.jobs import scheduler
    from ops.remote import access
    identity=reg.BH7_REFUSED_ID
    current={**history()[0], 'status':'launching','unknown':True,'reason':'container missing from confirmed snapshot',
             'attempt_since':time.time()-1000,'args':['build','--out',f'runs/{identity}/dataset.h5']}
    (tmp_path/'runs/setup').mkdir(parents=True)
    calls=[];events=[]
    class Client:
        def remote_path(self,name):return '/remote/'+name
        def checked_ssh(self,command,**kwargs):
            calls.append(command)
            assert 'docker ps -a' in command and f'/remote/runs/{identity}' in command
            assert 'docker run' not in command and 'rm ' not in command
            return type('Result',(),dict(stdout='0 absent exists 2026-11-21T00:00:00Z'))()
    monkeypatch.setattr(access,'RemoteClient',Client)
    monkeypatch.setattr(scheduler,'REGISTRY',tmp_path/'registry.jsonl')
    monkeypatch.setattr(scheduler,'read_registry',lambda:{identity:current})
    monkeypatch.setattr(scheduler,'record',lambda event:events.append(event))
    monkeypatch.setattr(time,'sleep',lambda _:None)
    monkeypatch.setattr(sys,'argv',['reconcile',identity])
    source=(ROOT/'ops/campaign/reconcile_unlaunched_remote.py').read_text()
    tree=ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node,ast.Constant) and node.value=='/home/ws/ws/splatter4d':node.value=str(tmp_path)
    exec(compile(tree,'reconcile-test','exec'),{})
    assert len(calls)==2 and len(events)==1 and events[0]['event']=='reset'
    reg.require_unlaunched([*history()[:-1],events[0]],{identity})


def test_remote_prepare_uses_seal_and_preserves_original_backend(tmp_path,monkeypatch):
    from ops.jobs import remote_backend,remote_offload
    from ops.jobs import realworld_local
    from s4d.rwlearn import dsreset_entrypoints
    from ops.resources.gpu_guard import APPROVED_HOST_GPUS
    from ops.remote.access import REMOTE_ROOT
    identity=reg.BH7_REFUSED_ID;commit='a'*40;image='sha256:'+'b'*64
    code=tmp_path/'source';(code/'scripts').mkdir(parents=True)
    script='scripts/bh_realworld_fql_dataset.py';path=code/script;path.write_text('# guard_gpus\n')
    path.chmod(0o444);code.chmod(0o555)
    snapshot=tmp_path/'snapshot.json'
    snapshot.write_text(json.dumps(dict(commit=commit,code_root=str(code),source_files={script:dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest())})))
    row=dict(id=identity,script=script,gpu='any',args=['build','--out',f'runs/{identity}/dataset.h5'],host='remote',
             required_results=['dataset.h5'],role='realworld_fql',ram_gb=8,mem_gb=1,disk_gb=2)
    grant=dict(commit=commit,image=image,rows={identity:dict(row=row)},snapshot=str(snapshot),snapshot_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest())
    directory=tmp_path/'grants';directory.mkdir();proof=directory/'remote-rgb.json';proof.write_text('{}')
    monkeypatch.setattr(remote,'load',lambda p:(grant,hashlib.sha256(proof.read_bytes()).hexdigest()))
    import bh_isolated_release
    monkeypatch.setattr(bh_isolated_release,'install_aliases',lambda:None)
    # The hook uses the fully qualified module in the live wrapper.
    from runs.setup.tools import bh_isolated_release as qualified
    monkeypatch.setattr(qualified,'install_aliases',lambda:None)
    old_prepare=remote_backend.RemoteBackend.prepare;old_pin=remote_offload.admitted_pin
    modules=(realworld_local,remote_offload,dsreset_entrypoints);aliases=[m.legacy_row for m in modules]
    gpu=APPROVED_HOST_GPUS['remote'][0]
    host=dict(commit=commit,image_digest=image,limits={gpu:dict(max_jobs=4,max_mem_gb=21.5)},host_ram_reserve_gb=120)
    backend=remote_backend.RemoteBackend(tmp_path/'accepted-main',tmp_path/'runs',client=type('Client',(),dict(root=REMOTE_ROOT))())
    try:
        assert remote.install(directory)
        specpath,spec=backend.prepare({**row,'gpu':gpu},gpu,2,host,{})
        assert specpath==tmp_path/'runs/remote/jobs'/f'{identity}-a2.json'
        assert spec['script']==script and backend.repo==tmp_path/'accepted-main'
        with pytest.raises(ValueError,match='pin/row differs'):
            backend.prepare({**row,'gpu':'other'},gpu,3,host,{})
        with pytest.raises(ValueError,match='pin/row differs'):
            backend.prepare({**row,'args':['drift']},gpu,3,host,{})
        foreign={**row,'id':'foreign'}
        with pytest.raises(ValueError,match='repository-relative'):backend.prepare(foreign,gpu,1,host,{})
    finally:
        remote_backend.RemoteBackend.prepare=old_prepare;remote_offload.admitted_pin=old_pin
        for module,fn in zip(modules,aliases):module.legacy_row=fn
        code.chmod(0o755);path.chmod(0o644)


def test_advancer_preserves_installed_local_rgb_and_uses_new_remote_seal(tmp_path,monkeypatch):
    import bh_advance as advance
    image='sha256:'+'b'*64;old='a'*40;new='c'*40
    main=tmp_path/'main';setup=main/'runs/setup';setup.mkdir(parents=True)
    installed=setup/'bh-isolated-release.d/local-rgb.json';installed.parent.mkdir();installed.write_text('installed local grant')
    release=dict(commit=old,image=image)
    proof=dict(release=release,route='hybrid',release_by_host={'local':release,'remote':dict(commit=new,image=image)},
        rows=[dict(id='local',host='local',planned_commit=old,planned_image=image),
              dict(id='remote',host='remote',planned_commit=new,planned_image=image)])
    (setup/'bh-20261120-rgb.registration.json').write_text(json.dumps(proof))
    candidate=tmp_path/'new.json';candidate.write_text(json.dumps(dict(commit=new)))
    local=tmp_path/'old.json';local.write_text(json.dumps(dict(commit=old)))
    out=tmp_path/'out';calls=[]
    monkeypatch.setattr(advance,'MAIN',main)
    def run(argv,**kw):
        calls.append(argv)
        return type('Result',(),dict(returncode=2 if '--wave' in argv else 0))()
    monkeypatch.setattr(advance.subprocess,'run',run)
    monkeypatch.setattr(sys,'argv',['advance','--snapshot',str(candidate),'--binding','new-binding',
        '--rgb-local-snapshot',str(local),'--rgb-local-binding','old-binding','--image',image,
        '--accepted-by','test','--out',str(out),'--once','--apply'])
    assert advance.main()==0
    assert installed.read_text()=='installed local grant'
    prepares=[a for a in calls if any(x.endswith('/bh_prepare_release.py') for x in a)]
    assert len(prepares)==1 and 'remote' in prepares[0] and prepares[0][prepares[0].index('--snapshot')+1]==str(candidate)
    assert not any('local' in a or any(x.endswith('/bh_fetch_inputs.py') for x in a) for a in calls)
    proof['rows'][1]['planned_commit']=old
    (setup/'bh-20261120-rgb.registration.json').write_text(json.dumps(proof))
    with pytest.raises(ValueError,match='host rows differ'):advance.main()
