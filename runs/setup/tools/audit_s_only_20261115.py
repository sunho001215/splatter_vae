"""Read-only live S/SF inventory; writes evidence and filtered plans only to --output-dir.

No network, locks, registry append, placement, registration or launch. Rerun after
hybrid registration: the same 150 failure identities automatically move from the
planned inventory into registered_cancelled. Snapshot hashes/time bound findings.
"""
from __future__ import annotations
import argparse
import collections
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import sys
import tempfile

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(SOURCE), str(Path(__file__).resolve().parent)]
import yaml
from ops.jobs import admission_pause, scheduler
from ops.jobs.dsreset_plan import build_rows, METHODS
from ops.jobs.local_devices import identical_placed_source
from s4d.rwlearn.dsreset import reuse_from_ingest
import register_dsreset_20261113 as reset
import register_ours_matrix_20261115 as matrix

MAIN = Path('/home/ws/ws/splatter4d')
PIPELINE = Path('/home/ws/ws/realworld-ops/dsreset-20261113/pipeline')
TASKS = ('covering', 'pick-and-place', 'drawer-open-v2')
PATTERN = re.compile(r'(?i)(?:^|[^a-z0-9])(?:sf|failure|failures|mixed|success[+_-]failure|s\+f)(?:$|[^a-z0-9])')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def cancelled(row):
    # Deliberately exclude runtime status/reasons: a failed S run is still S.
    fields = {k:row[k] for k in ('id','script','args','inputs','realworld_variant','realworld_inputs',
                                'dataset','variant','outcome','labels','setting','data_variant') if k in row}
    return bool(PATTERN.search(json.dumps(fields)))


def infer_task(row):
    return row.get('realworld_target') or next((t for t in TASKS if t in row.get('id','')), None)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=MAIN)
    parser.add_argument('--pipeline', type=Path, default=PIPELINE)
    parser.add_argument('--output-dir', type=Path, default=SOURCE/'docs/s-only-20261115')
    a = parser.parse_args(argv)
    out = a.output_dir.resolve()
    if SOURCE == a.root.resolve() or SOURCE not in out.parents:
        parser.error('run from the evidence worktree; output must be inside that worktree')
    out.mkdir(parents=True, exist_ok=True)
    observed = datetime.now(timezone.utc).isoformat()
    evidence = {}
    def read(path):
        data = Path(path).read_bytes()
        evidence[str(path)] = {'sha256':sha(data), 'bytes':len(data)}
        return data
    queuebytes = read(a.root/'experiments/queue.yaml')
    registrybytes = read(a.root/'experiments/registry.jsonl')
    queue = yaml.safe_load(queuebytes)
    by_id = {r['id']:r for r in queue['jobs']}
    if len(by_id) != len(queue['jobs']):
        raise ValueError('duplicate queue IDs')
    with tempfile.TemporaryDirectory(prefix='s-only-audit-') as tmp:
        path = Path(tmp)/'registry.jsonl'
        path.write_bytes(registrybytes)
        state = scheduler.read_registry(path)
    pausesdoc = json.loads(read(a.root/'runs/setup/admission-pauses.json'))
    pauses = [p for p in pausesdoc['pauses'] if p.get('active') is True]
    events = collections.defaultdict(list)
    for line in registrybytes.splitlines():
        event = json.loads(line)
        events[event['id']].append(event)
    bad = {r['id'] for r in queue['jobs'] if cancelled(r)}
    direct = set(bad)
    while True:
        more = {r['id'] for r in queue['jobs'] if set(r.get('deps',[])) & bad}
        if more <= bad:
            break
        bad |= more
    py = str(a.root/'.venv/bin/python')
    def cmd(*args):
        return shlex.join([py, '-I', '-B', *map(str,args)])
    def rowinfo(row, planned=False):
        identity = row['id']; current = state.get(identity, {})
        status = current.get('status','never-launched') if identity in by_id else 'unregistered'
        deps = [{'id':dep,'registered':dep in by_id,'status':state.get(dep,{}).get('status','never-launched' if dep in by_id else 'unregistered'),
                 'ready':scheduler.dependency_ready(row,state.get(dep,{}))} for dep in row.get('deps',[])]
        blocked = [d for d in deps if not d['ready']]
        active = admission_pause.active_pauses(identity, pauses=pauses)
        fixes = []
        if current.get('unknown'):
            fixes += [{'command':cmd(a.root/'scripts/jobs.py','status'), 'condition':'diagnosis only; after remote recovery the existing driver must reconcile this attempt; never reset unknown/running work'}]
        if identity in bad or cancelled(row):
            fix = 'cancelled by user; preserve history and pause admissions'
        elif status == 'done':
            fix = 'complete; reuse results'
        elif identity.endswith('eval400k-ours-covering-S'):
            fix = 'historical wrong-task evaluator is superseded by '+identity+'-v2 (done); do not reset original'
        elif status in ('running','launching','sync_pending'):
            fix = 'wait for existing attempt and verified closeout; never duplicate'
        elif row.get('registration_tool') == matrix.TOOL:
            fixes += [{'command':cmd(a.root/'runs/setup/tools/place_ours_matrix_20261115.py',identity),
                       'condition':'read-only placement diagnosis; after cancelled IDs are separately paused, add --execute for this S ID only when deps/input closeouts are ready'}]
            fix = 'verified local placement/capacity; retain all other admission holds'
        elif row.get('registration_tool') == 'runs/setup/tools/register_ours400k_20261113.py' or '-v2' in identity:
            fixes += [{'command':cmd(a.root/'runs/setup/tools/place_ours400k_20261113.py',identity),
                       'condition':'read-only first; add --execute only after producer closeouts/inputs are ready'}]
            fix = 'place exact S continuation descendant after producer closeout'
        else:
            fixes += [{'command':cmd(a.root/'ops/jobs/realworld_local.py','needs',identity), 'condition':'read-only missing-input diagnosis'},
                      {'command':cmd(a.root/'ops/jobs/realworld_local.py','place','--dry-run',identity),
                       'condition':'read-only; feature-check/placement holds remain binding'}]
            fix = 'normal verified input/placement route; gate owners must resolve named feature-check holds'
        if blocked:
            fix += '; wait for listed dependencies'
        return dict(id=identity, task=infer_task(row), status=status, host=row.get('host','local'),
                    unknown=current.get('unknown',False), attempts=current.get('attempts',0),
                    planned_commit=row.get('planned_commit'), script=row.get('script'),
                    variant=row.get('realworld_variant'), deps=deps, blocked_dependencies=blocked,
                    pauses=active, reason=current.get('reason'), fix=fix, commands=fixes,
                    registered=identity in by_id, deployment_hook=row.get('deployment_hook'))
    registered = [rowinfo(by_id[i]) | {'classification':'explicit' if i in direct else 'dependency-descendant'} for i in sorted(bad)]
    success = [rowinfo(r) for r in queue['jobs'] if infer_task(r) in TASKS and r['id'] not in bad]
    plans, preprocessing, sfprep = [], [], []
    ingestpath = Path('/home/ws/ws/realworld-ops/ingest/manifest.json')
    ingest = json.loads(read(ingestpath))
    for path in sorted(a.pipeline.glob('*-canonical-manifests/prep-plan.json')):
        rows = json.loads(read(path))
        srows = [r for r in rows if '-success-episode_' in r['episode']]
        (out/('success-plan-'+path.parent.name+'.json')).write_text(json.dumps(srows,indent=2)+'\n')
        task = srows[0]['episode'].split('-success-')[0]
        manifestpath = path.parent/'S/manifest.json'
        manifest = json.loads(read(manifestpath))
        accepted_reuse = reuse_from_ingest(manifest, ingest)
        preprocessing.append(dict(task=task,successes=len(srows),explicit_reused=sum('reuse' in r for r in srows),
                                  live_ingest_hash_bound_reuse=len(accepted_reuse),
                                  new_local_rows=sum('config' in r for r in srows),
                                  plan=str(path),filtered_success_plan=str(out/('success-plan-'+path.parent.name+'.json'))))
        for row in rows:
            if '-failure-' not in row['episode']:
                continue
            identity = 'rwv2-20261113-prep-ours-'+row['episode']+'-hybridA'
            item = dict(id=identity,episode=row['episode'],raw_sha256=row['raw_sha256'],source=str(path),
                        config=row.get('config'),registered=identity in by_id,
                        status=state.get(identity,{}).get('status','never-launched' if identity in by_id else 'unregistered'))
            sfprep.append(item)
    # Reconstruct every planned dsreset row from its pure builder, rather than
    # trusting old preview counts or requiring unavailable publication inputs.
    configpath = a.pipeline/'registration-spec.example.json'
    config = json.loads(read(configpath))
    release = dict(commit='4efd6e2f7b45a3c51a810427f0813a5ab0367e8f',image='sha256:'+'0'*64,source='local',evidence_manifest='docs/tests.json (local evidence)')
    dsrows = []
    for task,cfg in config['tasks'].items():
        rows = build_rows(task,cfg['inputs'],release,stage_rows=reset.default_stages(task,cfg['inputs']),
                          rig=cfg['rig'],methods=METHODS)
        for row in rows:
            dsrows.append(row)
            if cancelled(row) and row['id'] not in by_id:
                plans.append(dict(source='ops/jobs/dsreset_plan.py + '+str(configpath),**rowinfo(row)))
    mxrows = matrix.build_rows(queue,release,SOURCE)
    for row in mxrows:
        if cancelled(row) and row['id'] not in by_id:
            plans.append(dict(source=matrix.TOOL,**rowinfo(row)))
    continuation_templates = []
    for task in TASKS:
        for variant in ('S', 'SF'):
            m, _ = matrix.manifest(task, variant, SOURCE)
            adapter = dict(m, baseline_config=f'configs/realworld/ours_ext/baseline200k-{task}-{variant}.yaml',
                           baseline_sha256='0'*64, gate_sha256=m.get('gate_sha256','0'*64),
                           manifest_sha256=m.get('manifest_sha256','0'*64))
            for row in matrix.original.build_rows(task, variant, adapter, current=queue, release=release):
                item = dict(source=matrix.original.TOOL, template_only=True, **rowinfo(row))
                continuation_templates.append(item)
                if cancelled(row) and row['id'] not in by_id:
                    plans.append(item)
    planned_s = [rowinfo(r) for r in dsrows if not cancelled(r) and r['id'] not in by_id]
    # Both new scratch arms replace legacy NEW Ours chains; retain barriers/stats.
    publish_s = [r for r in planned_s if '-ours-' not in r['id'] or r['id'].split('-')[2] in ('prep','calib','qa')]
    # The future S-only graph must repair shared barriers currently hard-coded to
    # SF. It must also retain the existing covering/PNP S stats reuse, not create
    # redundant producers. These are proposed source changes, not accepted rows.
    publish_s = [r for r in publish_s if r['id'] not in (
        'rwv2-20261113-prep-ours-covering-S', 'rwv2-20261113-prep-ours-pick-and-place-S')]
    for row in dsrows:
        if row['script'] == 'scripts/realworld_dsreset_check.py' and row['id'] not in by_id:
            repaired = dict(row, args=[v.replace('/SF/', '/S/') for v in row['args']])
            publish_s.append(dict(rowinfo(repaired), prospective_source_edit='S-only shared readiness inputs; not yet implemented'))
    for task in ('covering','pick-and-place'):
        read(a.pipeline/f'reuse-proof-{task}-S.json')
    oldpin = []
    from ops.provenance.upstreams import working_tree_identity
    upstreams = working_tree_identity(a.root)
    for row in queue['jobs']:
        if row.get('host','local')=='local' and row.get('planned_commit'):
            current = state.get(row['id'], {'status':'pending'})
            if current.get('status') in ('pending','running','crashed','failed'):
                ownerpath = a.root/'runs/setup/ownership'/(row['id']+'.json')
                owned = json.loads(read(ownerpath)) if ownerpath.exists() else {}
                files = owned.get('execution_scope',{}).get('files',{})
                differences = []
                for name,digest in files.items():
                    path = a.root/name
                    actual = upstreams.get(name) if '@' in name else sha(path.read_bytes()) if path.is_file() else None
                    if actual != digest:
                        differences.append(name)
                oldpin.append(dict(id=row['id'],status=current['status'],pauses=admission_pause.active_pauses(row['id'],pauses=pauses),
                                   identical_placed_source=identical_placed_source(row,a.root),planned_commit=row['planned_commit'],
                                   source_differences=differences,ownership_record=str(ownerpath) if ownerpath.exists() else None))
    matrixproof = a.root/'runs/setup/rw-registration-ours-matrix-20261115.json'
    proof = json.loads(read(matrixproof)) if matrixproof.exists() else {}
    hooks = {r['id']:r.get('deployment_hook') for r in success if r.get('deployment_hook')}
    for task,audit in proof.get('continuation_audit',{}).items():
        for hook in audit.get('policy_hooks',[]):
            identity = Path(hook['policy']).parent.name
            hooks[identity] = hook
    deployment = []
    for row in queue['jobs']:
        if (infer_task(row) not in TASKS or row['id'] in bad
                or not row.get('script','').endswith('train_realworld_fql.py')):
            continue
        identity = row['id'];hook = hooks.get(identity)
        evaluation = hook.get('evaluation_id') if hook else None
        deployment.append(dict(id=identity,task=infer_task(row),registered=True,
            policy=f'runs/{identity}/policy.pt',policy_status=state.get(identity,{}).get('status','never-launched'),
            deployment_hook=hook,evaluation_id=evaluation,
            evaluation_status=state.get(evaluation,{}).get('status','unregistered' if evaluation not in by_id else 'never-launched') if evaluation else 'no linked legacy queue evaluation',
            target_installation='not verified offline; no deployment queue row',
            contract_check=cmd(a.root/'scripts/check_realworld_deploy.py','--out',f'/tmp/{identity}-contracts.json',
                               'contracts','--runs-root',a.root/'runs','--out-dir',f'/tmp/{identity}-contracts',identity),
            condition='after policy verified closeout; recorded-observation equality and Robot PC owner/catalog checks still required'))
    generators = []
    for path in sorted((SOURCE/'runs/setup/tools').glob('register*.py')):
        lines = path.read_text().splitlines()
        hits = [{'line':i+1,'text':line.strip()} for i,line in enumerate(lines)
                if re.search(r"['\"]SF['\"]|failure|mixed",line)]
        generators.append(dict(path=str(path.relative_to(SOURCE)),sha256=sha(path.read_bytes()),failure_variant_hits=hits))
    # Pause command uses the exact helper's never-launched contract. Running or
    # attempted rows are still listed above, but cannot be fed to add_pause.
    pause_ids = [r['id'] for r in registered if r['status'] in ('never-launched','pending') and not r['attempts']]
    pause_name = 's-only-cancelled-20261115-'+sha(json.dumps(pause_ids).encode())[:8]
    pause_command = cmd(a.root/'runs/setup/tools/add_pause_20261112.py',pause_name,
                        'user 2026-11-15 cancelled all SF/failure/mixed dataset work',*pause_ids)
    (out/'pause-registered-sf.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\n'+pause_command+' "$@"\n')
    live_locks = []
    locklines = Path('/proc/locks').read_text().splitlines()
    for relative in ('runs/setup/stage2-driver.lock','runs/setup/remote-driver.lock','experiments/registry.lock'):
        lockpath = a.root/relative
        if not lockpath.exists():
            continue
        stat = lockpath.stat()
        key = f'{os.major(stat.st_dev):02x}:{os.minor(stat.st_dev):02x}:{stat.st_ino}'
        for line in locklines:
            if key not in line:
                continue
            fields = line.replace('->','').split()
            proc = Path('/proc')/fields[4]
            try:
                argv = (proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')[:1600]
                wchan = (proc/'wchan').read_text()
            except FileNotFoundError:
                argv,wchan = None,None
            live_locks.append(dict(path=str(lockpath),pid=int(fields[4]),waiting='->' in line,argv=argv,wchan=wchan))
    report = dict(schema='s-only-audit/20261115',observed_utc=observed,operational_lock_blockers=live_locks,
                  snapshot_policy='unlocked read-only snapshots; no claim of atomic multi-file live state or remote reachability',
                  queue_rows=len(by_id),evidence=evidence,registered_cancelled=registered,
                  pause_helper_ids=pause_ids,hybrid_failure_plan=sfprep,preprocessing=preprocessing,
                  planned_unregistered_cancelled=plans,registered_success=success,
                  planned_unregistered_success_legacy=planned_s,planned_success_matrix_publication=publish_s,
                  continuation_templates=continuation_templates,
                  old_pin_local_scope_check=oldpin,deployment_inventory=deployment,registration_tool_scan=generators,
                  counts=dict(registered_cancelled=len(registered),planned_failure_hybrid=len(sfprep),
                              registered_failure_hybrid=sum(x['registered'] for x in sfprep),
                              planned_unregistered_cancelled=len(plans),planned_unregistered_cancelled_unique_ids=len({r['id'] for r in plans}),registered_success=len(success),
                              planned_success_matrix_publication=len(publish_s)))
    (out/'AUDIT.json').write_text(json.dumps(report,indent=2)+'\n')
    # Full per-job tables are separate to keep the analysis readable.
    with (out/'ROWS.md').open('w') as handle:
        handle.write('# Snapshot row inventory\n\nSee AUDIT.json for complete dependencies, source hashes and commands.\n\n')
        for name,rows in [('Registered cancelled',registered),('Registered success',success),('Missing S publication rows',publish_s)]:
            handle.write('## '+name+'\n\n| ID | Status | Blocking dependencies | Admission pauses |\n|---|---|---|---|\n')
            for row in rows:
                handle.write('| '+row['id']+' | '+row['status']+' | '+', '.join(d['id']+' ('+d['status']+')' for d in row['blocked_dependencies'])+' | '+', '.join(row['pauses'])+' |\n')
            handle.write('\n')
    print(json.dumps(report['counts'],indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
