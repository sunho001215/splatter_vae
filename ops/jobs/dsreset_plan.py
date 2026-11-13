"""Pure queue-row construction for paired real-world datasets. No scheduling calls."""
from __future__ import annotations

from s4d.rwlearn.dsreset import TASKS, sha256
from pathlib import Path

METHODS = ('ours', 'sincro', 'reviwo', 'dynamo', 'snerl', 'mvp3d_single', 'mvp3d_multiview',
           'e2e_cnn', 'e2e_cnn_noaug')
PRETRAIN = ('ours', 'sincro', 'reviwo', 'dynamo', 'snerl', 'mvp3d_multiview')
STEM = 'rwv2-20261113'
PAUSE = 'dsreset-await-placement-20261113'


def row_id(stage, method, task, variant=None, seed=None):
    return '-'.join([STEM, stage, method, task] + ([variant] if variant else [])
                    + ([f's{seed}'] if seed is not None else []))


def build_rows(task, inputs, release, *, stage_rows, rig, methods=METHODS, native_bounds=None):
    """Build dependencies after bulk prep publishes immutable learning inputs.

    Shared prep/calib/qa rows recheck dispatcher evidence at consumption time.
    Bulk LoMa-G/FS/TE preparation remains in the established dispatcher and is
    runnable independently from realworld_dsreset.py's per-episode configs.
    """
    import copy
    if task not in TASKS or not set(methods) <= set(METHODS) or not methods:
        raise ValueError('invalid task/method selection')
    if set(stage_rows) != {'prep', 'calib', 'qa'}:
        raise ValueError('supply prep/calib/qa evidence barriers')
    common = dict(host='remote', pin_host='remote', remote_host='192.168.10.22', gpu='any',
                  max_restarts=1, oom_score_adj=0, cpu_threads=4, disk_gb=2,
                  realworld_target=task, planned_commit=release['commit'], planned_image=release['image'],
                  registration_tool='runs/setup/tools/register_dsreset_20261113.py',
                  release_source=release.get('source', 'role'), local_placement_pause=PAUSE,
                  evidence_manifest='docs/tests.json (local evidence)' if release.get('source') == 'local' else
                  f"runs/remote/acceptance/{release['commit'][:7]}-cc/manifest.json")
    rows = []
    def add(stage, method, variant, script, args, deps, results, *, seed=None, mem=2, ram=8, **extra):
        rid = row_id(stage, method, task, variant, seed)
        priority = -3 if stage in ('prep', 'calib', 'qa') else (-2 if method == 'ours' else 0) \
            if stage in ('pre', 'pre400k', 'export') else (-1 if method == 'ours' else 1)
        role = 'realworld_pretrain' if stage in ('pre', 'pre400k', 'export') or script == 'scripts/compute_workspace_stats.py' \
            else 'realworld_fql'
        rows.append(dict(copy.deepcopy(common), id=rid, script=script, args=args, deps=deps,
                         required_results=results, role=role, priority=priority, mem_gb=mem, ram_gb=ram, **extra))
        return rid
    previous = []
    for stage in ('prep', 'calib', 'qa'):
        row = copy.deepcopy(stage_rows[stage])
        if not row.get('script', '').startswith('scripts/') or not row.get('args') or not row.get('required_results'):
            raise ValueError(f'{stage}: script, args, required_results required')
        rid = row_id(stage, 'ours', task)
        rows.append({**common, 'role':'realworld_preprocess', 'priority':-3, 'mem_gb':8, 'ram_gb':16,
                         **{k:v for k,v in row.items() if k not in common and k not in ('id','deps','priority','role')},
                         'id':rid, 'deps':previous})
        previous = [rid]
    qa = previous[0]
    for variant in ('S', 'SF'):
        directory = f'{inputs}/{variant}'
        store, split, gate = [f'{directory}/{name}' for name in ('store.yaml','split_seed0.json','gate_report.json')]
        native = None
        if any(m.startswith('mvp3d_') for m in methods):
            native = row_id('prep', 'mvp3d_multiview', task, variant)
            add('prep', 'mvp3d_multiview', variant, 'scripts/realworld_native_depth.py',
                ['build', '--store', store, '--rig', rig, '--out-dir', f'runs/{native}/overlays',
                 '--out-store', f'runs/{native}/store-native.yaml'], [qa], ['store-native.yaml'], mem=1)
        exports = {}
        for method in PRETRAIN:
            if method not in methods and not (method == 'mvp3d_multiview' and 'mvp3d_single' in methods):
                continue
            pre = row_id('pre', method, task, variant)
            exp = row_id('export', method, task, variant)
            if method == 'ours':
                stats = row_id('prep', 'ours', task, variant)
                add('prep', 'ours', variant, 'scripts/compute_workspace_stats.py',
                    ['--regime','realworld','--path',store,'--split',split,'--out',f'runs/{stats}/workspace_stats.json'],
                    [qa], ['workspace_stats.json'])
                sets = [f'data.store={store}',f'data.split_manifest={split}',f'data.gate_report={gate}',
                        f'model.anchor_stats=runs/{stats}/workspace_stats.json', 'train.steps=200000']
                baseline = f'configs/realworld/dsreset/ours/{task}-{variant}.yaml'
                add('pre', method, variant, 'scripts/train.py',
                    ['--config',baseline,'--name',pre,'--output-root','runs',
                     '--resume','auto','--set',*sets], [qa,stats],
                    ['completion.json','metrics.jsonl','checkpoints/step_0200000.pt'],mem=8,ram=16)
                final = row_id('pre400k', method, task, variant)
                add('pre400k', method, variant, 'scripts/train_ours_continuation.py',
                    ['--name',final,'--output-root','runs','--source-checkpoint',
                     f'runs/{pre}/checkpoints/step_0200000.pt','--baseline-config',baseline,
                     '--baseline-sha256',sha256(Path(__file__).resolve().parents[2]/baseline),
                     '--schedule-config','configs/realworld/ours_ext/continuation400k.yaml',
                     '--start-step','200000','--stop-step','400000','--set',*sets[:-1]],
                    [pre,stats],['completion.json','continuation.json','resume_source.json','metrics.jsonl',
                                 'checkpoints/step_0400000.pt'],mem=8,ram=16)
                add('export', method, variant, 'scripts/export_encoder.py',
                    ['--run',f'runs/{final}','--checkpoint',f'runs/{final}/checkpoints/step_0400000.pt',
                     '--out',f'runs/{exp}/encoder.pt'], [final], ['encoder.pt'])
            else:
                b = 'mvp3d' if method.startswith('mvp3d') else method
                source = f'runs/{native}/store-native.yaml' if b == 'mvp3d' else store
                sets = [f'dataset.hdf5_path={source}',f'data_interface.split_manifest={split}',
                        f'data_interface.gate_report={gate}']
                if b == 'mvp3d' and native_bounds is not None:
                    import json
                    sets += ['render.bounds='+json.dumps(native_bounds,separators=(',',':'))]
                deps = [qa] + ([native] if b == 'mvp3d' else [])
                resume = ['--resume','auto'] if b in ('mvp3d','snerl','dynamo') else []
                mem, ram = {'sincro':(12,24),'reviwo':(4,16),'dynamo':(12,32),'snerl':(16,20),'mvp3d':(12,24)}[b]
                add('pre', method, variant, f'scripts/train_{b}.py',
                    ['--config',f'configs/realworld/baselines/{b}/{task}.yaml','--name',pre,'--output-root','runs',
                     *resume,'--set',*sets], deps, ['encoder.pt','metrics.jsonl'],mem=mem,ram=ram)
                add('export', method, variant, 'scripts/realworld_dsreset_artifact.py',
                    ['--source',f'runs/{pre}/encoder.pt','--kind','encoder','--out',f'runs/{exp}/encoder.pt'],
                    [pre], ['encoder.pt'])
            exports[method] = exp
        for method in methods:
            encoder = 'mvp3d_multiview' if method.startswith('mvp3d') else method
            pixel = method.startswith('e2e_cnn')
            condition = 'e2e_cnn' if pixel else method
            data = row_id('fqldata', method, task, variant)
            dep = None if pixel else exports[encoder]
            source = f'runs/{native}/store-native.yaml' if method.startswith('mvp3d') else store
            deps = [qa] + ([dep] if dep else []) + ([native] if method.startswith('mvp3d') else [])
            args = ['build','--store',source,'--condition',condition,'--out',f'runs/{data}/dataset.h5',
                    '--gate-report',gate,'--outcomes',f'{directory}/outcomes.json','--device','cpu' if pixel else 'cuda']
            if dep:
                args += ['--export',f'runs/{dep}/encoder.pt']
            add('fqldata',method,variant,'scripts/realworld_fql_dataset.py',args,deps,['dataset.h5'],
                mem=1 if pixel else 8,ram=8 if pixel else 16)
            for seed in (0,1,2):
                fql = row_id('fql',method,task,variant,seed)
                sets = [f'condition={condition}',f'data.dataset=runs/{data}/dataset.h5',
                        f'data.split_manifest={split}',f'data.gate_report={gate}',f'train.seed={seed}']
                if pixel:
                    sets += [f'data.store={store}']
                    if method == 'e2e_cnn_noaug':
                        sets += ['pixel.p_aug=0.0']
                else:
                    sets += ['agent.alpha=10.0','agent.normalize_q_loss=true']
                if method.startswith('mvp3d'):
                    sets += ['encoder.feature_norm=train_standardize']
                add('fql',method,variant,'scripts/train_realworld_fql.py',
                    ['--config','configs/realworld/fql/base.yaml','--name',fql,'--output-root','runs','--set',*sets],
                    [data],['policy.pt','metrics.jsonl'],seed=seed,mem=12 if pixel else 6,ram=24 if pixel else 12)
                ev = row_id('eval',method,task,variant,seed)
                add('eval',method,variant,'scripts/realworld_dsreset_artifact.py',
                    ['--source',f'runs/{fql}/policy.pt','--kind','policy','--out',f'runs/{ev}/evaluation.json'],
                    [fql],['evaluation.json'],seed=seed)
    return rows
