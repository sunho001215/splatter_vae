import copy
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
from scripts import jobs
from s4d.config import load_config

freeze_path = ROOT / 'runs/setup/method-frozen-v1.json'
freeze = json.loads(freeze_path.read_text())
freeze_sha = hashlib.sha256(freeze_path.read_bytes()).hexdigest()
commit = freeze['source_commit']
assert freeze['pretraining_steps'] == 200000
assert subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'method-frozen-v1^{commit}'],
                      capture_output=True, text=True, check=True).stdout.strip() == commit
annotation = subprocess.run(['git', '-C', str(ROOT), 'for-each-ref', '--format=%(contents)',
                             'refs/tags/method-frozen-v1'], capture_output=True, text=True, check=True).stdout
assert json.loads(annotation) == freeze
assert not subprocess.run(['git', '-C', str(ROOT), 'diff', commit, 'HEAD', '--',
                          's4d', 'scripts', 'tests', 'configs', 'pyproject.toml', 'uv.lock'],
                         capture_output=True, check=True).stdout

new_jobs = []
encoders = {}
for task, expected in freeze['task_configs'].items():
    cfg = load_config([ROOT / 'configs/metaworld/base.yaml', ROOT / f'configs/metaworld/tasks/{task}.yaml'],
                      freeze['overrides'])
    assert cfg == expected, task
    if task in freeze['development_encoder_reuse']:
        export = freeze['development_encoder_reuse'][task]
        dependency = f'export-ls-a200-{task}-200k'
        encoders[task] = {'export': export, 'dependency': dependency, 'reused': True,
                          'sha256': hashlib.sha256((ROOT / export).read_bytes()).hexdigest()}
        continue
    trainer = f's3-m3d-seed0-{task}-200k'
    exporter = f's3-export-m3d-seed0-{task}-200k'
    new_jobs.append({
        'id': trainer, 'host': 'remote', 'script': 'scripts/train.py',
        'args': ['--config', 'configs/metaworld/base.yaml', f'configs/metaworld/tasks/{task}.yaml',
                 '--name', trainer, '--output-root', 'runs', '--resume', 'auto', '--set', *freeze['overrides']],
        'gpu': 'any', 'mem_gb': 10, 'ram_gb': 16, 'disk_gb': 30, 'priority': 1,
        'deps': [], 'max_restarts': 2, 'stage': 3,
        'required_results': ['completion.json', 'metrics.jsonl', 'checkpoints/step_0200000.pt'],
        'frozen_method': 'method-frozen-v1', 'freeze_manifest_sha256': freeze_sha,
    })
    export = f'runs/{exporter}/encoder.pt'
    new_jobs.append({
        'id': exporter, 'host': 'remote', 'script': 'scripts/export_encoder.py',
        'args': ['--run', f'runs/{trainer}', '--checkpoint', f'runs/{trainer}/checkpoints/step_0200000.pt',
                 '--out', export],
        'gpu': 'any', 'mem_gb': 2, 'ram_gb': 4, 'disk_gb': 1, 'priority': 0,
        'deps': [trainer], 'max_restarts': 1, 'stage': 3, 'required_results': ['encoder.pt'],
        'frozen_method': 'method-frozen-v1', 'freeze_manifest_sha256': freeze_sha,
    })
    encoders[task] = {'export': export, 'dependency': exporter, 'reused': False, 'trainer': trainer}

for seed in freeze['rl_seeds']:
    for task, encoder in encoders.items():
        trainer = f's3-drm-splatter4d-{task}-s{seed}'
        evaluator = trainer + '-eval'
        common = {'host': 'remote', 'gpu': 'any', 'priority': 2, 'deps': [encoder['dependency']],
                  'max_restarts': 1, 'stage': 3, 'frozen_method': 'method-frozen-v1',
                  'freeze_manifest_sha256': freeze_sha}
        new_jobs.append({**common, 'id': trainer, 'script': 'scripts/train_rl.py',
                         'args': ['--task', task, '--encoder', 'splatter4d', '--seed', str(seed),
                                  '--run-dir', f'runs/{trainer}', '--set', 'train.num_train_steps=1000000',
                                  f"vision.export_path={encoder['export']}"],
                         'mem_gb': 4, 'ram_gb': 8, 'disk_gb': 5, 'oom_score_adj': 300,
                         'required_results': ['final.json', 'train.jsonl']})
        new_jobs.append({**common, 'id': evaluator, 'script': 'scripts/eval_rl.py',
                         'args': ['--run-dir', f'runs/{trainer}', '--train-job', trainer,
                                  '--workers', '3', '--envs-per-worker', '8', '--output-dir', f'runs/{evaluator}'],
                         'mem_gb': 3, 'ram_gb': 8, 'disk_gb': 5, 'oom_score_adj': 400,
                         'required_results': ['eval.jsonl']})
assert len(new_jobs) == 60
assert len({j['id'] for j in new_jobs}) == 60
proof_path = ROOT / 'runs/setup/stage3-final-registration-20261013.json'
queue_path = ROOT / 'experiments/queue.yaml'
with jobs.REGISTRY.with_suffix('.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    queue = jobs.load_queue()
    old = {j['id']: j for j in queue['jobs']}
    state = jobs.read_registry()
    for task, encoder in encoders.items():
        if encoder['reused']:
            assert state[encoder['dependency']]['status'] == jobs.DONE
            previous = old[encoder['dependency']]
            assert previous['args'][previous['args'].index('--out') + 1] == encoder['export']
            marker = ROOT / 'runs/remote/exports' / (Path(encoder['export']).name + '.json')
            assert json.loads(marker.read_text())['sha256'] == encoder['sha256']
    existing = [j for j in new_jobs if j['id'] in old]
    if existing:
        assert len(existing) == len(new_jobs) and all(old[j['id']] == j for j in new_jobs)
        assert proof_path.is_file()
        print('ALREADY_REGISTERED', len(existing))
    else:
        assert not proof_path.exists()
        before = queue_path.read_bytes()
        candidate = ROOT / 'runs/setup/stage3-final-registered-20261013-v2.yaml'
        assert not candidate.exists()
        text = before.decode()
        document = yaml.compose(text)
        jobs_node = next(value for key, value in document.value if key.value == 'jobs')
        boundary = jobs_node.end_mark.index
        candidate.write_text(text[:boundary].rstrip() + '\n' + yaml.safe_dump(new_jobs, sort_keys=False)
                             + '\n' + text[boundary:])
        combined = jobs.load_queue(candidate)
        assert combined['jobs'] == queue['jobs'] + new_jobs
        assert all(set(j.get('deps', [])) <= {q['id'] for q in combined['jobs']} for j in new_jobs)
        snapshot = ROOT / 'runs/setup/stage3-final-queue-before-20261013.yaml'
        assert not snapshot.exists()
        snapshot.write_bytes(before)
        temporary = queue_path.with_suffix('.stage3.tmp')
        assert not temporary.exists()
        temporary.write_bytes(candidate.read_bytes())
        os.replace(temporary, queue_path)
        proof = {'registered_at': datetime.now().astimezone().isoformat(), 'source_commit': commit,
                 'freeze_manifest_sha256': freeze_sha, 'method_tag': 'method-frozen-v1',
                 'queue_sha256_before': hashlib.sha256(before).hexdigest(),
                 'queue_sha256_after': hashlib.sha256(queue_path.read_bytes()).hexdigest(),
                 'job_count': len(new_jobs), 'pretraining_jobs': 6, 'export_jobs': 6,
                 'rl_trainers': 24, 'rl_evaluators': 24, 'rl_agent_steps': 1000000,
                 'encoders': encoders, 'jobs': new_jobs,
                 'speculative_300k_adopted': False, 'admission': 'job by job through scripts/jobs.py'}
        proof_path.write_text(json.dumps(proof, indent=2) + '\n')
        print('REGISTERED', len(new_jobs), proof_path)
