"""Prove and reuse existing Ours success chains without mutating their queue rows."""
from __future__ import annotations
import copy
from pathlib import Path
from ops.jobs.dsreset_plan import row_id


def aliases(task):
    mapping={row_id(stage,'ours',task,'S'):row_id(stage+'400k','ours',task,'S')
             for stage in ('export','fqldata')}
    mapping[row_id('pre400k','ours',task,'S')]=row_id('pre400k','ours',task,'S')
    mapping.update({row_id('fql','ours',task,'S',seed):row_id('fql400k','ours',task,'S',seed)
                    for seed in (0,1,2)})
    return mapping


def reuse_rows(rows,task,current):
    """Call only after prove_reuse (or graph-only preview, which cannot publish)."""
    mapping=aliases(task)
    if not set(mapping.values()) <= set(current):
        raise ValueError(f'{task}: existing Ours success chain is incomplete')
    omit=set(mapping)|{row_id('prep','ours',task,'S'),row_id('pre','ours',task,'S')}
    rewritten=[]
    for original in rows:
        if original['id'] in omit:
            continue
        row=copy.deepcopy(original)
        row['deps']=[mapping.get(dep,dep) for dep in row['deps']]
        # Longest first keeps prefixes from partially rewriting a longer identifier.
        for old,new in sorted(mapping.items(),key=lambda pair:-len(pair[0])):
            row['args']=[arg.replace(f'runs/{old}/',f'runs/{new}/') for arg in row['args']]
        rewritten.append(row)
    return rewritten


def _arg(row,flag):
    args=row['args']
    return args[args.index(flag)+1]


def _config(row,root):
    from s4d.config import load_config
    sets=row['args'][row['args'].index('--set')+1:] if '--set' in row['args'] else []
    return load_config([root/_arg(row,'--config')],sets)


def prove_reuse(task,manifest,old_gate,old_split,old_outcomes,old_store,new_store,rows,current,root,*,new_gate):
    """Exact raw hashes, train/val membership, store paths, Original recipe and FQL science."""
    from s4d.config import load_config
    from s4d.train.continuation import science_config
    from s4d.rwlearn.dsreset import sha256,validate_manifest
    from s4d.rwlearn.gate import task_passed
    from s4d.rwlearn.outcomes import load_outcomes
    validate_manifest(manifest)
    if manifest['task']!=task or manifest['variant']!='S' or not task_passed(old_gate,task,'ours'):
        raise ValueError('existing Ours S task/gate is not equivalent')
    def calibration(report):
        identities=report['tasks'][task]['episode_identities']
        bykey={(r.get('store_episode') or ep):r['raw_sha256'] for ep,r in identities.items()}
        return {bykey[ep]:{'calibrate_build_id':r['calibrate_build_id'],'cameras':sorted(r['cameras'])}
                for ep,r in report['calibration_hold'][task]['episodes'].items()}
    if not task_passed(new_gate,task,'ours') or calibration(old_gate)!=calibration(new_gate):
        raise ValueError('existing Ours S calibration builds/cameras differ')
    identities=old_gate['tasks'][task]['episode_identities']
    by_ep={r['store_episode']:r['raw_sha256'] for r in identities.values()}
    labels=load_outcomes(old_outcomes)
    if set(labels)!=set(by_ep) or not all(labels.values()):
        raise ValueError('existing Ours S outcomes are not all success')
    if old_store.get('kind')!='compact' or new_store.get('kind')!='compact' or sorted(old_store['paths'])!=sorted(new_store['paths']):
        raise ValueError('existing Ours S compact stores differ')
    wanted={s:{r['sha256'] for r in manifest['episodes'] if r['split']==s} for s in ('train','validation')}
    if (set(old_split['train'])&set(old_split['validation']) or
            set(old_split['train'])|set(old_split['validation']) != set(by_ep) or
            any({by_ep[ep] for ep in old_split[s]}!=wanted[s] for s in wanted)):
        raise ValueError('existing Ours S raw hashes/splits differ; cannot reuse')
    pre=current[row_id('pre400k','ours',task,'S')]
    if (_arg(pre,'--start-step')!='200000' or _arg(pre,'--stop-step')!='400000'
            or _arg(pre,'--schedule-config')!='configs/realworld/ours_ext/continuation400k.yaml'
            or sha256(root/_arg(pre,'--baseline-config'))!=_arg(pre,'--baseline-sha256')):
        raise ValueError('existing Ours continuation is not the pinned single-cycle recipe')
    old=science_config(load_config([root/_arg(pre,'--baseline-config')]))
    new=science_config(load_config([root/f'configs/realworld/dsreset/ours/{task}-S.yaml']))
    for cfg in (old,new):
        for k in ('store','split_manifest','gate_report'):cfg['data'].pop(k,None)
        cfg['model'].pop('anchor_stats',None)
    if old!=new:
        raise ValueError('existing Ours Original science differs')
    byid={r['id']:r for r in rows}
    for seed in (0,1,2):
        old=_config(current[row_id('fql400k','ours',task,'S',seed)],root)
        new=_config(byid[row_id('fql','ours',task,'S',seed)],root)
        for cfg in (old,new):
            for k in ('dataset','split_manifest','gate_report','store'):cfg['data'].pop(k,None)
        if old!=new:
            raise ValueError('existing Ours FQL science differs')
    mapping=aliases(task)
    export=current[mapping[row_id('export','ours',task,'S')]]
    fqldata=current[mapping[row_id('fqldata','ours',task,'S')]]
    if (_arg(export,'--checkpoint')!=f"runs/{pre['id']}/checkpoints/step_0400000.pt"
            or pre['id'] not in export['deps'] or export['id'] not in fqldata['deps']
            or _arg(fqldata,'--condition')!='ours'
            or _arg(fqldata,'--export')!=f"runs/{export['id']}/encoder.pt"):
        raise ValueError('existing Ours export/fqldata do not bind final 400k')
    for seed in (0,1,2):
        if current[row_id('fql400k','ours',task,'S',seed)]['deps']!=[fqldata['id']]:
            raise ValueError('existing Ours FQL dependencies differ')
    return {'task':task,'variant':'S','method':'ours','aliases':mapping,
            'reason':'same raw hashes, split, compact outputs, Original recipe, single-cycle 400k and FQL science'}
