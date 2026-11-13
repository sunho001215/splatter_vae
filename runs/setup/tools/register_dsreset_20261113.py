"""Paired S/SF registration. Dry-run by default; only the coordinator runs --execute.

Bulk prep remains in the established dispatcher, using realworld_dsreset.py's
per-episode configs. The prep/calib/qa queue rows audit those completed artifacts.
No import of campaign drivers, no network, no launch or placement calls.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE))
import yaml
from ops.jobs.dsreset_plan import METHODS, PAUSE, STEM, build_rows, row_id
from s4d.rwlearn import gate
from s4d.rwlearn.dsreset import TASKS, manifest_gate, sha256, validate_manifest

MAIN = Path('/home/ws/ws/splatter4d')
TOOL = 'runs/setup/tools/register_dsreset_20261113.py'
spec = importlib.util.spec_from_file_location('dsreset_registration_common', SOURCE/'runs/setup/tools/register_ours_realworld_20261111.py')
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)


def validate_inputs(task, path, ingest):
    docs = {}
    hashes = {}
    for variant in ('S','SF'):
        directory = Path(path)/variant
        blobs = {n:(directory/n).read_bytes() for n in ('manifest.json','store.yaml','split_seed0.json','outcomes.json','gate_report.json')}
        hashes.update({str(directory/n):R.sha(b) for n,b in blobs.items()})
        m = validate_manifest(json.loads(blobs['manifest.json']))
        R.require(m['task']==task and m['variant']==variant, 'task/variant differs from directory')
        report = json.loads(blobs['gate_report.json'])
        live = manifest_gate(m, ingest)
        saved = report['tasks'][task]
        eps = saved['store_episodes']
        ids = {r['store_episode']:r for r in saved['episode_identities'].values()}
        R.require(len(ids)==len(m['episodes']) and set(ids)==set(eps), 'invalid bound gate identities')
        bound = gate.bind_store_episodes(live,{task:eps},ids)
        for section in ('tasks','families','calibration_hold'):
            R.require(bound[section]==report[section], 'live QA/calibration evidence changed; regenerate inputs')
        store = yaml.safe_load(blobs['store.yaml'])
        want_paths = []
        for r in m['episodes']:
            matches = [e for e in ingest['episodes'].values() if e.get('task')==task and e.get('sha256')==r['sha256']]
            want_paths.append(matches[0]['preprocessing']['stages']['train_data']['outputs'])
        R.require(store.get('kind')=='compact' and sorted(store.get('paths',[]))==sorted(want_paths),
                  'store paths differ from live train_data outputs')
        by_sha = {r['sha256']:r for r in m['episodes']}
        split, outcomes = json.loads(blobs['split_seed0.json']),json.loads(blobs['outcomes.json'])
        wanted_split = {s:sorted(ep for ep,r in ids.items() if by_sha[r['raw_sha256']]['split']==s)
                        for s in ('train','validation')}
        R.require(all(sorted(split[s])==wanted_split[s] and len(split[s])==len(set(split[s])) for s in wanted_split),
                  'split differs from manifest')
        R.require(set(outcomes)==set(ids) and all(type(v) is bool for v in outcomes.values()) and
                  all(outcomes[ep]==(by_sha[r['raw_sha256']]['label']=='success') for ep,r in ids.items()),
                  'outcomes differ from directory labels')
        for c in ('ours','sincro','reviwo','dynamo','snerl','mvp3d','e2e_cnn'):
            R.require(gate.task_passed(report,task,c),f'{task}/{variant}: QA not passed for {c}')
        docs[variant] = m
    success = [r for r in docs['SF']['episodes'] if r['label']=='success']
    R.require(docs['S']['episodes']==success, 'S must be exactly the success subset of SF with identical splits')
    return hashes


def default_stages(task, directory):
    inputs = directory+'/SF'
    common = ['--store',inputs+'/store.yaml','--manifest',inputs+'/manifest.json',
              '--split',inputs+'/split_seed0.json','--outcomes',inputs+'/outcomes.json',
              '--gate-report',inputs+'/gate_report.json']
    return {stage:{'script':'scripts/realworld_dsreset_check.py',
                   'args':['--stage',stage,*common,'--out',f"runs/{row_id(stage,'ours',task)}/readiness.json"],
                   'required_results':['readiness.json'],'mem_gb':1,'ram_gb':4}
            for stage in ('prep','calib','qa')}


def prepare(repo, config, config_path, ingest_path, release, *, preview=False):
    R.require(re.fullmatch('[0-9a-f]{40}',release['commit'] or ''), 'full commit required')
    R.require(re.fullmatch('sha256:[0-9a-f]{64}',release['image'] or ''), 'image digest required')
    methods = tuple(config.get('methods',METHODS))
    R.require(len(set(methods))==len(methods) and set(methods)<=set(METHODS), 'invalid methods')
    R.require(config.get('tasks') and set(config['tasks'])<=set(TASKS), 'only the three reset tasks are allowed')
    ingest = {} if preview else json.loads(ingest_path.read_bytes())
    inputs = {str(config_path):sha256(config_path)}
    if not preview:
        inputs[str(ingest_path)] = sha256(ingest_path)
    before = (repo/'experiments/queue.yaml').read_bytes()
    registry = (repo/'experiments/registry.jsonl').read_bytes()
    current = {r['id']:r for r in R.queue_document(before)['jobs']}
    rows, reused = [], []
    from s4d.rwlearn.configs import pending_scene_parameters
    for task, task_cfg in config['tasks'].items():
        directory = task_cfg['inputs'].rstrip('/')
        R.require(directory.startswith('/home/compu/kaist/sunho/realworld/'), 'inputs must be under verified remote root')
        R.require(task in task_cfg['rig'], 'per-task rig document required')
        if not preview:
            inputs.update(validate_inputs(task,task_cfg['inputs_copy'],ingest))
            if any(method.startswith('mvp3d_') for method in methods):
                measurement = Path(task_cfg['native_bounds_report'])
                b = json.loads(measurement.read_text())
                sf = json.loads((Path(task_cfg['inputs_copy'])/'SF/manifest.json').read_text())
                import hashlib
                digest = hashlib.sha256(json.dumps(sf,sort_keys=True).encode()).hexdigest()
                bounds = b.get('bounds',[])
                R.require(b.get('task')==task and b.get('depth_source')=='native_realsense'
                          and b.get('manifest_sha256')==digest and b.get('accepted') is True
                          and len(bounds)==6 and all(type(v) in (int,float) and math.isfinite(v) for v in bounds)
                          and all(bounds[i]<bounds[i+3] for i in range(3))
                          and sorted(b.get('episodes',[]))==sorted(sf['split']['train'])
                          and bool(b.get('scene_statistics')), 'native-depth bounds need accepted train-only SF measurements')
                task_cfg = dict(task_cfg,native_bounds=bounds)
                inputs[str(measurement)] = sha256(measurement)
            for method in methods:
                b = 'mvp3d' if method.startswith('mvp3d') else method
                R.require(not pending_scene_parameters(b,task),f'{task}/{b}: measure scene parameters and release first')
        task_rows = build_rows(task,directory,release,stage_rows=default_stages(task,directory),rig=task_cfg['rig'],methods=methods,
                           native_bounds=task_cfg.get('native_bounds'))
        if task_cfg.get('reuse_ours_S'):
            from ops.jobs.dsreset_reuse import aliases,prove_reuse,reuse_rows
            R.require('ours' in methods,'reuse_ours_S requires ours')
            if not preview:
                pre=current[row_id('pre400k','ours',task,'S')]
                cfgargs=pre['args']
                base=SOURCE/cfgargs[cfgargs.index('--baseline-config')+1]
                inputs[str(base)]=sha256(base)
                old_directory=task_cfg['reuse_ours_S_inputs_copy']
                names=('gate_report.json','split_seed0.json','outcomes.json','store.yaml')
                blobs={n:(Path(old_directory)/n).read_bytes() for n in names}
                inputs.update({str(Path(old_directory)/n):R.sha(b) for n,b in blobs.items()})
                new=Path(task_cfg['inputs_copy'])/'S'
                proof=prove_reuse(task,json.loads((new/'manifest.json').read_text()),
                      json.loads(blobs['gate_report.json']),json.loads(blobs['split_seed0.json']),
                      json.loads(blobs['outcomes.json']),yaml.safe_load(blobs['store.yaml']),
                      yaml.safe_load((new/'store.yaml').read_text()),task_rows,current,SOURCE,
                      new_gate=json.loads((new/'gate_report.json').read_text()))
                reused.append(proof)
            else:
                reused.append({'task':task,'variant':'S','method':'ours','aliases':aliases(task),'unchecked_preview':True})
            task_rows=reuse_rows(task_rows,task,current)
        rows += task_rows
    R.require(len({r['id'] for r in rows})==len(rows),'duplicate row IDs')
    # Validate every argv against the same resolver used at placement; no hidden inputs.
    from ops.jobs.realworld_local import needed_inputs
    from ops.jobs.host_roles import load_roles, refusal
    by_id = {**current, **{r['id']:r for r in rows}}
    for row in rows:
        needed_inputs(row)
        R.require(refusal(row,'local',load_roles(),{},by_id) is None,'row is not eligible for local placement')
        R.require(refusal(row,'remote',load_roles(),{},by_id) is None,'row is not eligible for remote placement')
        R.require((SOURCE/row['script']).is_file(),'missing executable script')
        for a in row['args']:
            if a.startswith('configs/') and a.endswith('.yaml'):
                inputs[str(SOURCE/a)] = sha256(SOURCE/a)
    known = {r['id'] for r in R.queue_document(before)['jobs']} | R.registry_ids(registry)
    R.require(not known.intersection(r['id'] for r in rows),'row IDs already registered')
    R.require(all(not (repo/'runs'/rid).exists() for rid in (r['id'] for r in rows)),'run output already exists')
    candidate = R.queue_candidate(before,rows)
    return {'schema':'dsreset-registration/v1','registered':False,'status':'preview' if preview else 'dry-run',
            'repo':str(repo.resolve()),'release':release,'rows':rows,'rows_sha256':R.json_sha(rows),
            'methods':list(methods),'input_sha256':inputs,'tool_sha256':sha256(Path(__file__)),
            'queue_sha256_before':R.sha(before),'queue_sha256_after':R.sha(candidate),
            'registry_sha256':R.sha(registry),'required_pause':PAUSE,
            'prep_policy':'reuse established dispatcher outputs; queue prep/calib/qa rows are read-only readiness audits',
            'reuse_existing_S':reused,
            'reuse_reason':'existing chains require exact raw/split/store and science proof; otherwise new S rows'}


def execute(repo, record):
    R.require(repo.resolve()==MAIN and Path(__file__).resolve()==MAIN/TOOL,'only coordinator released main copy may execute')
    R.require(record['status']=='dry-run','preview cannot be executed')
    setup = repo/'runs/setup'
    pending, proof = setup/f'{STEM}.pending.json', setup/f'{STEM}.registration.json'
    def check():
        R.require(R.current_pin(repo,record['release'])==record['release'],'release differs from adopted sources')
        if record['release'].get('source')!='local':
            R.require((repo/f"runs/remote/acceptance/{record['release']['commit'][:7]}-cc/manifest.json").is_file(),
                      'remote acceptance manifest missing')
        R.require(sha256(Path(__file__))==record['tool_sha256'],'registrar changed')
        R.require(R.sha((repo/'experiments/queue.yaml').read_bytes())==record['queue_sha256_before'],'queue changed')
        R.require(R.sha((repo/'experiments/registry.jsonl').read_bytes())==record['registry_sha256'],'registry changed')
        R.require(all(sha256(Path(p))==v for p,v in record['input_sha256'].items()),'inputs changed')
        R.require(all(not (repo/'runs'/r['id']).exists() and not (repo/'runs'/r['id']).is_symlink() for r in record['rows']),
                  'run output appeared')
        from ops.jobs.admission_pause import active_pauses
        R.require(all(PAUSE in active_pauses(r['id']) for r in record['rows']),f'publish only under active {PAUSE} pause')
        R.require(R.json_sha(record['rows'])==record['rows_sha256'],'rows changed')
    with R.locks(repo):
        R.require(not pending.exists() and not proof.exists(),'existing journal/proof requires inspection')
        check()
        candidate = R.queue_candidate((repo/'experiments/queue.yaml').read_bytes(),record['rows'])
        R.require(R.sha(candidate)==record['queue_sha256_after'],'candidate changed')
        R.atomic_write(pending,R.json_bytes(record))
        check()
        R.atomic_write(repo/'experiments/queue.yaml',candidate)
        record = dict(record,status='published',registered=True,registered_at=R.now())
        R.atomic_write(proof,R.json_bytes(record))
        pending.unlink()
        R.sync_directory(setup)
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,default=MAIN)
    p.add_argument('--spec',type=Path,required=True)
    p.add_argument('--ingest-manifest',type=Path,default=gate.MANIFEST)
    p.add_argument('--commit',required=True)
    p.add_argument('--image',required=True)
    p.add_argument('--release-source',choices=('local','role'),default='local')
    p.add_argument('--preview',action='store_true',help='graph/argv only, never executable')
    p.add_argument('--execute',action='store_true')
    p.add_argument('--out',type=Path)
    a = p.parse_args()
    R.require(not (a.preview and a.execute),'preview cannot execute')
    R.require(not (a.out and a.execute),'--out is dry-run only')
    release = {'commit':a.commit,'image':a.image}
    if a.release_source=='local':
        release['source']='local'
    record = prepare(a.repo,json.loads(a.spec.read_text()),a.spec,a.ingest_manifest,release,preview=a.preview)
    if a.execute:
        record = execute(a.repo,record)
    if a.out:
        out=a.out.resolve()
        R.require(not out.exists() and not any(p.name in ('experiments','setup') for p in out.parents),
                  'output must be a new non-operational proof path')
        out.write_bytes(R.json_bytes(record))
    print(R.json_bytes(record).decode(),end='')


if __name__=='__main__':
    main()
