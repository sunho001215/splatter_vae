"""New 12-run scratch matrix. Stdout-only by default; released-main publication only.

Existing continuation rows are immutable. Missing PNP evaluation is appended.
Unpublished variants wait for dsreset QA/publication and train-only stats IDs.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path

SOURCE = next(p for p in Path(__file__).resolve().parents if (p / 's4d').is_dir())
sys.path[:0] = [str(SOURCE), str(Path(__file__).resolve().parent)]
import yaml
import register_ours400k_20261113 as original

from s4d.config import load_config
from s4d.rwlearn.dsreset_entrypoints import ENTRYPOINTS
from s4d.train.continuation import file_sha

R = original.reviewed
MAIN = R.MAIN
TOOL = 'runs/setup/tools/register_ours_matrix_20261115.py'
STEM = 'rw-registration-ours-matrix-20261115'
LOCAL_PAUSE = 'ours-matrix-await-local-placement-20261115'
TASKS = ('drawer-open-v2', 'covering', 'pick-and-place')
ARMS = ('orig', 'ch01')
CONFIG = 'configs/realworld/ours_matrix'


def row_id(stage, task, variant, arm, protocol='scratch400k', seed=None):
    R.require(task in TASKS and variant in ('S', 'SF') and arm in ARMS, 'invalid matrix identity')
    return f'rwv3-20261115-{stage}-ours-{task}-{variant}-{arm}-{protocol}' + (f'-s{seed}' if seed is not None else '')


def manifest(task, variant, root=SOURCE):
    """Existing S inputs are reused; deferred variants name canonical publication outputs."""
    if task != 'drawer-open-v2':
        m = json.loads((root/f'configs/realworld/ours_ext/{task}-S.json').read_bytes())['data_manifest']
    else:
        cfg = load_config([root/'configs/realworld/dsreset/ours/drawer-open-v2-S.yaml'])
        m = dict(store=cfg['data']['store'], split_manifest=cfg['data']['split_manifest'],
                 gate_report=cfg['data']['gate_report'], anchor_stats=cfg['model']['anchor_stats'],
                 stats_id='rwv2-20261113-prep-ours-drawer-open-v2-S',
                 recipe_source='rwcal-20261111-ours-drawer-open-pre')
    m = copy.deepcopy(m)
    deferred = variant == 'SF' or task == 'drawer-open-v2'
    if deferred:
        directory = f'/home/compu/kaist/sunho/realworld/learning_inputs/dsreset/{task}/{variant}'
        m.update(store=directory+'/store.yaml', split_manifest=directory+'/split_seed0.json',
                 gate_report=directory+'/gate_report.json', outcomes=directory+'/outcomes.json',
                 stats_id=f'rwv2-20261113-prep-ours-{task}-{variant}',
                 anchor_stats=f'runs/rwv2-20261113-prep-ours-{task}-{variant}/workspace_stats.json',
                 deps=[f'rwv2-20261113-qa-ours-{task}'])
    m['include_eval'] = True
    return m, deferred


def validate_recipes(root):
    """Reject drift outside the approved horizon, coefficient and variant input paths."""
    paths=[]
    for task in TASKS:
        reference=(f'configs/realworld/ours_ext/baseline200k-{task}-S.yaml' if task!='drawer-open-v2'
                   else 'configs/realworld/dsreset/ours/drawer-open-v2-S.yaml')
        paths.append(root/reference)
        if task!='drawer-open-v2':
            paths.append(root/f'configs/realworld/ours_ext/{task}-S.json')
        for variant in ('S','SF'):
            m,_=manifest(task,variant,root)
            expected=load_config([root/reference])
            expected['train']['steps']=400000
            for section,key,name in (('data','store','store'),('data','split_manifest','split_manifest'),
                                     ('data','gate_report','gate_report'),('model','anchor_stats','anchor_stats')):
                expected[section][key]=m[name]
            for arm in ARMS:
                cfg=copy.deepcopy(expected)
                if arm=='ch01':
                    cfg['loss']['chamfer']=0.1
                path=root/f'{CONFIG}/{task}-{variant}-{arm}-scratch400k.yaml'
                R.require(load_config([path])==cfg,f'unapproved recipe drift: {path.name}')
                paths.append(path)
    return paths


def build_rows(current, release, root=SOURCE):
    """Reuse the established 400k downstream argv; replace the two-phase pre with one scratch row."""
    rows = []
    paired = {old:new for new,old in ENTRYPOINTS.items()}
    for task in TASKS:
        for variant in ('S', 'SF'):
            m, deferred = manifest(task, variant, root)
            for arm in ARMS:
                config = f'{CONFIG}/{task}-{variant}-{arm}-scratch400k.yaml'
                # The old pure builder checks this historical name; all output argv below use the new pinned config.
                adapter = dict(m, baseline_config=f'configs/realworld/ours_ext/baseline200k-{task}-{variant}.yaml',
                               baseline_sha256='0'*64, gate_sha256=m.get('gate_sha256','0'*64),
                               manifest_sha256=m.get('manifest_sha256','0'*64))
                generated = original.build_rows(task, variant, adapter, current=current, release=release)
                mapping = {r['id']:row_id({'pre400k':'pre','export400k':'export','fqldata400k':'fqldata',
                                          'fql400k':'fql','eval400k':'eval'}.get(r['id'].split('-')[2], 'unused'),
                                         task,variant,arm,
                                         seed=int(r['id'][-1]) if re.search(r'-s[012]$',r['id']) else None)
                           for r in generated}
                old_pre = f'rwv2-20261113-pre-ours-{task}-{variant}'
                for row in generated:
                    if row['id'] == old_pre:
                        continue
                    stage = row['id'].split('-')[2]
                    row['id'] = mapping[row['id']]
                    row['args'] = [mapping.get(a,a) for a in row['args']]
                    # IDs also occur inside run output/checkpoint paths and key=value overrides.
                    for old,new in mapping.items():
                        row['args'] = [a.replace(old,new) for a in row['args']]
                    row['args'] = [config if a == adapter['baseline_config'] else a for a in row['args']]
                    row['deps'] = [mapping.get(d,d) for d in row['deps'] if d != old_pre]
                    if stage == 'pre400k':
                        row['script'] = 'scripts/train.py'
                        row['args'] = ['--config',config,'--name',row['id'],'--output-root','runs',
                                       '--resume','auto','--set',*R.input_overrides(load_config([root/config]))]
                        row['required_results'] = ['completion.json','metrics.jsonl','checkpoints/step_0400000.pt']
                    row.update(registration_tool=TOOL, local_placement_pause=LOCAL_PAUSE,
                               realworld_arm='Original' if arm=='orig' else 'Chamfer0.1',
                               realworld_protocol='scratch', realworld_checkpoint=400000,
                               realworld_variant=variant)
                    if stage == 'eval400k':
                        row['deps'] += [row_id('fql',task,variant,arm,seed=s) for s in range(3)]
                    if deferred:
                        row.pop('realworld_gate_sha256',None)
                        row.pop('realworld_manifest_sha256',None)
                        row['learning_input_publication'] = f'rwv2-20261113-qa-ours-{task}'
                        row['script'] = paired.get(row['script'], row['script'])
                    if stage == 'fql400k':
                        row['deployment_hook'] = dict(schema='realworld-policy-ready/v1',
                            task=task, dataset=variant, arm=row['realworld_arm'], protocol='scratch',
                            checkpoint=400000, seed=int(row['id'][-1]), policy=f"runs/{row['id']}/policy.pt",
                            evaluation_id=row_id('eval',task,variant,arm), owner='robot-pc-pipeline')
                    rows.append(row)
    return rows


def continuation_rows(current, release, root=SOURCE):
    """Audit both existing chains, append only a missing task-aware PNP eval."""
    by_id = {r['id']:r for r in current['jobs']}
    added, audit = [], {}
    for task in ('covering','pick-and-place'):
        ids = [f'rwv2-20261113-{stage}-ours-{task}-S' for stage in ('pre400k','export400k','fqldata400k')]
        ids += [f'rwv2-20261113-fql400k-ours-{task}-S-s{s}' for s in range(3)]
        R.require(all(i in by_id for i in ids), f'incomplete continuation downstream: {task}')
        pre = ids[0]
        evals = [r['id'] for r in current['jobs'] if r.get('script')=='scripts/evaluate_ours_task_length.py'
                 and f'runs/{pre}/checkpoints/step_0400000.pt' in r.get('args',[])]
        if not evals:
            m,_ = manifest(task,'S',root)
            candidate = next(r for r in original.build_rows(task,'S',m|dict(
                baseline_sha256=file_sha(root/m['baseline_config']),include_eval=True),
                current=current,release=release) if 'eval400k' in r['id'])
            candidate.update(id=row_id('eval',task,'S','orig','continuation400k'), registration_tool=TOOL,
                             local_placement_pause=LOCAL_PAUSE, realworld_arm='Original',
                             realworld_protocol='continuation', realworld_checkpoint=400000)
            candidate['args'][candidate['args'].index('--name')+1] = candidate['id']
            candidate['deps'] += ids[3:]
            added.append(candidate)
            evals=[candidate['id']]
        audit[task] = dict(existing_ids=ids, evaluation_ids=evals, protocol='continuation',
                          policy_hooks=[dict(task=task,dataset='S',arm='Original',protocol='continuation',
                              checkpoint=400000,seed=s,policy=f'runs/{ids[3+s]}/policy.pt',
                              evaluation_id=evals[-1],owner='robot-pc-pipeline') for s in range(3)])
    return added,audit


def ready_local_inputs(rows, repo, root):
    """Read-only proof that the four immediate S pretrains have their full local closure."""
    from ops.jobs import scheduler, realworld_local as rw
    state=scheduler.read_registry(repo/'experiments/registry.jsonl')
    immediate=[r for r in rows if '-pre-' in r['id'] and not r.get('learning_input_publication')]
    R.require(len(immediate)==4,'expected exactly four immediately ready pretrains')
    ready={r['id'] for r in scheduler.eligible(immediate,state)}
    R.require(ready=={r['id'] for r in immediate},'existing S statistics dependencies are not done')
    proof=[]
    for row in immediate:
        args,verified=rw._inputs(row,state,repo/'runs/realworld-inputs',
            repo/'runs/realworld-inputs/MANIFEST.json',repo/'runs',root=root,
            finished_dir=repo/'runs/setup/realworld-local-finished')
        proof.append(dict(id=row['id'],deps=row['deps'],args=args,verified=verified))
    return proof


def prepare(repo, root, commit, image):
    R.require(re.fullmatch('[0-9a-f]{40}',commit or ''), 'full commit required')
    R.require(re.fullmatch('sha256:[0-9a-f]{64}',image or ''), 'image digest required')
    release=dict(commit=commit,image=image,source='local',evidence_manifest=R.LOCAL_EVIDENCE)
    if repo.resolve()==root.resolve():
        release=R.local_release_pin(repo,commit,image)
    before=(repo/'experiments/queue.yaml').read_bytes()
    registry=(repo/'experiments/registry.jsonl').read_bytes()
    current=R.queue_document(before)
    validate_recipes(root)
    rows=build_rows(current,release,root)
    extra,audit=continuation_rows(current,release,root)
    rows+=extra
    known={j['id'] for j in current['jobs']}|R.registry_ids(registry)
    R.require(len(rows)==len({r['id'] for r in rows}) and not known.intersection(r['id'] for r in rows),
              'duplicate or existing matrix IDs')
    R.require(all(not (repo/'runs'/r['id']).exists() and not (repo/'runs'/r['id']).is_symlink() for r in rows),
              'run path already exists')
    R.require(not (repo/f'runs/setup/{STEM}.pending.json').exists(), 'pending journal requires reconciliation')
    inputs={}
    for path in [*validate_recipes(root), root/TOOL,root/original.TOOL,root/R.TOOL]:
        inputs[str(path.resolve())]=file_sha(path)
    for task in ('covering','pick-and-place'):
        reference=root/f'configs/realworld/ours_ext/baseline200k-{task}-S.yaml'
        saved=repo/f'runs/rwcal-20261111-ours-{task}-pre/config.yaml'
        if saved.is_file():
            actual=load_config([saved]);actual.pop('run',None)
            R.require(actual==load_config([reference]),f'actual Original recipe changed: {task}')
            inputs[str(saved)]=file_sha(saved)
        else:
            source=next(r for r in current['jobs'] if r['id']==f'rwcal-20261111-ours-{task}-pre')
            args=source['args'];path=root/args[args.index('--config')+1]
            actual=load_config([path],args[args.index('--set')+1:])
            R.require(actual==load_config([reference]),f'queued Original recipe changed: {task}')
            inputs[str(path)]=file_sha(path)
    # Bind actual FQL seed science, as in the original 400k registrar.
    by_id={j['id']:j for j in current['jobs']}
    for seed in range(3):
        identity=f'rwnq-20261111-fql-ours-drawer-open-s{seed}'
        args=by_id[identity]['args']; cfg=root/args[args.index('--config')+1]
        saved=repo/'runs'/identity/'config.yaml'
        science=yaml.safe_load(saved.read_bytes());science.pop('run',None)
        R.require(load_config([cfg],args[args.index('--set')+1:])==science,'FQL template science changed')
        inputs.update({str(p):file_sha(p) for p in (cfg,saved)})
    from ops.jobs.realworld_local import needed_inputs
    for row in rows:
        needed_inputs(row)
    if repo.resolve()==root.resolve():
        inputs[str(repo/'docs/tests.json')]=file_sha(repo/'docs/tests.json')
    local_inputs=ready_local_inputs(rows,repo,root)
    mirror=repo/'runs/realworld-inputs'
    inputs[str(mirror/'MANIFEST.json')]=file_sha(mirror/'MANIFEST.json')
    for proof in local_inputs:
        inputs.update({str(repo/name):digest for name,digest in proof['verified']['dependency_inputs'].items()})
    candidate=R.queue_candidate(before,rows)
    return dict(schema='rw-registration/v1',status='dry-run',registered=False,prepared_at=R.now(),
        repo=str(repo.resolve()),source_root=str(root.resolve()),release=release,rows=rows,
        rows_sha256=R.json_sha(rows),input_sha256=inputs,local_pause=LOCAL_PAUSE,journal_stem=STEM,
        holds=[R.local_hold(rows,LOCAL_PAUSE)],queue_sha256_before=R.sha(before),
        queue_sha256_after=R.sha(candidate),registry_sha256=R.sha(registry),
        local_input_evidence=local_inputs,continuation_audit=audit,priorities=dict(preprocessing=-4,pretraining=-3,downstream=-1),
        deferred_dependencies=sorted({d for r in rows for d in r.get('deps',[]) if d not in by_id
                                      and d not in {x['id'] for x in rows}}),
        queue_diff=dict(added_rows=len(rows),modified_rows=0,removed_rows=0,added_ids=[r['id'] for r in rows]))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,default=MAIN)
    p.add_argument('--source-root',type=Path,default=SOURCE)
    p.add_argument('--commit',required=True)
    p.add_argument('--image',required=True)
    p.add_argument('--execute',action='store_true')
    a=p.parse_args(argv)
    record=prepare(a.repo,a.source_root,a.commit,a.image)
    if a.execute:
        record=R.execute(a.repo,record)
    print(json.dumps(record,indent=2))


if __name__=='__main__':
    main()
