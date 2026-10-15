"""Register Stage 2 frozen-method attribution (local): 10 variants x 2 development tasks at 200k, seed 0, full-split
evaluations, guarded exports for the RL arms, and 1M-step DrM RL (seeds 1000-1002) for the full method and four
ablations. Idempotent; asserts every variant config differs from the frozen config only in its intended keys."""
import fcntl
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path('/home/ws/ws/splatter4d')
sys.path.insert(0, str(ROOT))
from scripts import jobs  # noqa: E402
from s4d.config import load_config  # noqa: E402

MARGINS_SHA = '5fadd04fd5050a440f13d5adad540e424a0f78257ed3f82eacec36df31db5677'
ABL = 'configs/metaworld/ablations/'
TASKS = ('hammer', 'pick-place')
# name: (extra config files, extra overrides, gets RL, expected flattened difference from the frozen config)
VARIANTS = {
    't1': ([ABL + 'single_frame.yaml'], [], True,
           {'model.single_frame': True, 'model.encoder.num_frames': 1, 'loss.motion': 0.0}),
    'nomotion': ([ABL + 'no_motion.yaml'], [], True, {'loss.motion': 0.0}),
    'noinfonce': ([], ['loss.invariance=0'], True, {'loss.invariance': 0}),
    'singlegroup': ([ABL + 'single_group.yaml'], [], True, {'model.decoder.single_group': True}),
    'nosynth': ([], ['aug.synth_views=0', 'aug.synth_render=false'], False,
                {'aug.synth_views': 0, 'aug.synth_render': False}),
    'nocons': ([], ['loss.consistency=0'], False, {'loss.consistency': 0}),
    'nolambdadyn': ([], ['loss.lambda_dyn=0'], False, {'loss.lambda_dyn': 0}),
    'nocoverage': ([ABL + 'no_coverage.yaml'], [], False, {'loss.coverage': 0.0}),
    'misscaled-none': ([ABL + 'misscaled_none.yaml'], [], False, {'loss.teacher_scale_jitter': [0.7, 1.4]}),
    'misscaled-scale': ([ABL + 'misscaled_scale.yaml'], [], False,
                        {'loss.teacher_scale_jitter': [0.7, 1.4], 'loss.depth_align': 'scale'}),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args):
    return subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True, text=True, check=True).stdout


def flat(tree, prefix=''):
    out = {}
    for key, value in tree.items():
        if isinstance(value, dict):
            out.update(flat(value, f'{prefix}{key}.'))
        else:
            out[prefix + key] = value
    return out


freeze_path = ROOT / 'runs/setup/method-frozen-v1.json'
margins_path = ROOT / 'runs/setup/stage2-reporting-margins-20261013.json'
freeze = json.loads(freeze_path.read_text())
freeze_sha = sha(freeze_path)
assert sha(margins_path) == MARGINS_SHA
assert json.loads(margins_path.read_text())['stage2_registered_jobs_before'] == 0
commit = freeze['source_commit']
assert freeze['pretraining_steps'] == 200000 and freeze['ablation_rl_seeds'] == [1000, 1001, 1002]
assert git('rev-parse', 'method-frozen-v1^{commit}').strip() == commit
assert json.loads(git('for-each-ref', '--format=%(contents)', 'refs/tags/method-frozen-v1')) == freeze
assert not git('diff', commit, 'HEAD', '--', 's4d', 'scripts', 'tests', 'configs', 'pyproject.toml', 'uv.lock')
meta = {'stage': 2, 'frozen_method': 'method-frozen-v1', 'freeze_manifest_sha256': freeze_sha,
        'stage2_margins_sha256': MARGINS_SHA}

new_jobs, configs, exports = [], {}, {}
for variant, (files, sets, rl, expected) in VARIANTS.items():
    for task in TASKS:
        cfg = load_config([ROOT / 'configs/metaworld/base.yaml', ROOT / f'configs/metaworld/tasks/{task}.yaml',
                           *[ROOT / f for f in files]], [*freeze['overrides'], *sets])
        frozen = flat(freeze['task_configs'][task])
        got = flat(cfg)
        diff = {k: got.get(k) for k in set(frozen) | set(got) if got.get(k, KeyError) != frozen.get(k, KeyError)}
        assert diff == expected, (variant, task, diff)
        trainer = f's2ab-{variant}-seed0-{task}'
        configs[trainer] = cfg
        checkpoint = f'runs/pretrain/{trainer}/checkpoints/step_0200000.pt'
        new_jobs.append({
            'id': trainer, 'script': 'scripts/train.py',
            'args': ['--config', 'configs/metaworld/base.yaml', f'configs/metaworld/tasks/{task}.yaml', *files,
                     '--name', trainer, '--output-root', 'runs/pretrain', '--resume', 'auto',
                     '--set', *freeze['overrides'], *sets],
            'gpu': 'any', 'mem_gb': 10, 'ram_gb': 13, 'priority': 0 if rl else 2, 'deps': [], 'max_restarts': 6,
            'oom_score_adj': 0, 'stage2_variant': variant, **meta})
        evaluation = f'fulleval-{trainer}-200k'
        new_jobs.append({
            'id': evaluation, 'script': 'scripts/evaluate.py',
            'args': ['--run', f'runs/{evaluation}', '--checkpoint', checkpoint],
            'gpu': 'any', 'mem_gb': 5, 'ram_gb': 12, 'priority': 1, 'deps': [trainer], 'max_restarts': 1,
            'oom_score_adj': 300, 'stage2_variant': variant, **meta})
        if rl:
            exporter = f'export-{trainer}-200k'
            out = f'runs/exports/{trainer}-200k.pt'
            new_jobs.append({
                'id': exporter, 'script': 'scripts/export_encoder.py',
                'args': ['--run', f'runs/pretrain/{trainer}', '--checkpoint', checkpoint, '--out', out, '--wait'],
                'gpu': 'any', 'mem_gb': 1, 'ram_gb': 4, 'priority': 0, 'deps': [trainer], 'max_restarts': 1,
                'oom_score_adj': 300, 'stage2_variant': variant, **meta})
            exports[(variant, task)] = (out, exporter)
for task in TASKS:
    exports[('full', task)] = (freeze['development_encoder_reuse'][task], f'export-ls-a200-{task}-200k')

arms = ['full'] + [v for v, spec in VARIANTS.items() if spec[2]]
for seed in freeze['ablation_rl_seeds']:
    for task in TASKS:
        for arm in arms:
            export, dependency = exports[(arm, task)]
            trainer = f's2rl-{arm}-{task}-s{seed}'
            common = {'gpu': 'any', 'priority': 1, 'deps': [dependency], 'max_restarts': 1, 'stage2_variant': arm,
                      **meta}
            new_jobs.append({**common, 'id': trainer, 'script': 'scripts/train_rl.py',
                             'args': ['--task', task, '--encoder', 'splatter4d', '--seed', str(seed),
                                      '--run-dir', f'runs/{trainer}', '--set', 'train.num_train_steps=1000000',
                                      f'vision.export_path={export}'],
                             'mem_gb': 4, 'ram_gb': 6, 'oom_score_adj': 300})
            new_jobs.append({**common, 'id': trainer + '-eval', 'script': 'scripts/eval_rl.py',
                             'args': ['--run-dir', f'runs/{trainer}', '--train-job', trainer,
                                      '--workers', '3', '--envs-per-worker', '8'],
                             'mem_gb': 3, 'ram_gb': 6, 'oom_score_adj': 400})

counts = {s: sum(j['script'] == s for j in new_jobs) for s in
          ('scripts/train.py', 'scripts/evaluate.py', 'scripts/export_encoder.py', 'scripts/train_rl.py',
           'scripts/eval_rl.py')}
assert counts == {'scripts/train.py': 20, 'scripts/evaluate.py': 20, 'scripts/export_encoder.py': 8,
                  'scripts/train_rl.py': 30, 'scripts/eval_rl.py': 30}, counts
assert len({j['id'] for j in new_jobs}) == len(new_jobs) == 108
proof_path = ROOT / 'runs/setup/stage2-registration-20261013.json'
queue_path = jobs.QUEUE
with jobs.REGISTRY.with_suffix('.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    queue = jobs.load_queue()
    old = {j['id']: j for j in queue['jobs']}
    state = jobs.read_registry()
    for task in TASKS:
        export, dependency = exports[('full', task)]
        assert state[dependency]['status'] == jobs.DONE
        assert old[dependency]['args'][old[dependency]['args'].index('--out') + 1] == export
        marker = ROOT / 'runs/remote/exports' / (Path(export).name + '.json')
        assert json.loads(marker.read_text())['sha256'] == sha(ROOT / export)
    existing = [j for j in new_jobs if j['id'] in old]
    if existing:
        assert len(existing) == len(new_jobs) and all(old[j['id']] == j for j in new_jobs) and proof_path.is_file()
        print('ALREADY_REGISTERED', len(existing))
        raise SystemExit(0)
    assert not proof_path.exists() and not any(j.get('stage') == 2 for j in queue['jobs'])
    assert not any((ROOT / 'runs/pretrain' / j['id']).exists() or (ROOT / 'runs' / j['id']).exists() for j in new_jobs)
    before = queue_path.read_bytes()
    # The first candidate (stage2-registered-20261013.yaml) failed to parse: shared deps lists were dumped as
    # &id001 anchors that collide with an anchor already in the queue. It is kept; JSON round-trip removes sharing.
    candidate = ROOT / 'runs/setup/stage2-registered-20261013-v2.yaml'
    assert not candidate.exists()
    text = before.decode()
    jobs_node = next(value for key, value in yaml.compose(text).value if key.value == 'jobs')
    boundary = jobs_node.end_mark.index
    unshared = json.loads(json.dumps(new_jobs))
    candidate.write_text(text[:boundary].rstrip() + '\n' + yaml.safe_dump(unshared, sort_keys=False)
                         + '\n' + text[boundary:])
    combined = jobs.load_queue(candidate)
    assert combined['jobs'] == queue['jobs'] + new_jobs
    assert {k: v for k, v in combined.items() if k != 'jobs'} == {k: v for k, v in queue.items() if k != 'jobs'}
    ids = {q['id'] for q in combined['jobs']}
    assert all(set(j['deps']) <= ids for j in new_jobs)
    snapshot = ROOT / 'runs/setup/stage2-queue-before-20261013.yaml'
    assert not snapshot.exists()
    snapshot.write_bytes(before)
    temporary = queue_path.with_suffix('.stage2.tmp')
    assert not temporary.exists()
    temporary.write_bytes(candidate.read_bytes())
    os.replace(temporary, queue_path)
    proof = {'registered_at': datetime.now().astimezone().isoformat(), 'source_commit': commit,
             'method_tag': 'method-frozen-v1', 'freeze_manifest_sha256': freeze_sha,
             'reporting_margins': str(margins_path.relative_to(ROOT)), 'reporting_margins_sha256': MARGINS_SHA,
             'queue_sha256_before': hashlib.sha256(before).hexdigest(), 'queue_sha256_after': sha(queue_path),
             'host': 'local (same host as the A200 representation references)', 'pretraining_steps': 200000,
             'pretraining_seed': 0, 'rl_seeds': freeze['ablation_rl_seeds'], 'rl_agent_steps': 1000000,
             'variants': {v: {'files': f, 'overrides': s, 'rl': rl, 'expected_difference': e}
                          for v, (f, s, rl, e) in VARIANTS.items()},
             'rl_arms': arms, 'encoders': {f'{a}/{t}': {'export': e, 'dependency': d} for (a, t), (e, d) in exports.items()},
             'counts': counts, 'configs': configs, 'jobs': new_jobs,
             'purpose': 'attribution only; the frozen method and its hyperparameters do not change',
             'admission': 'job by job through scripts/jobs.py under the local full-charge cap'}
    proof_path.write_text(json.dumps(proof, indent=2) + '\n')
    print('REGISTERED', len(new_jobs), proof_path, sha(proof_path))
