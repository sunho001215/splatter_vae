"""Local smoke run of one encoder-pilot arm (user directive 2026-10-25: smoke runs per arm on both hosts' images, recording
s/it and peak memory). Runs scripts/train.py of ``--source`` (the release checkout) on hammer for 600 steps (train.steps
600: during the 10k-step warm-up the learning rate equals that of the 200k schedule), no start evaluation, no W&B, the
end evaluation and checkpoint included. Samples the run's session PSS (non-file: anonymous + shmem) every 2 s.
Output: <out>/<arm>/smoke.json with the median logged s/it (steps > 100), peak GPU memory (max_memory_allocated) and peak
session PSS.

    python -I enc_smoke_local.py --source <root> --arm hn|cs|h2|w4|w8|chlrm --gpu <MIG uuid> --out <dir>
"""
import argparse
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--source', required=True)
ap.add_argument('--arm', required=True)
ap.add_argument('--gpu', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--steps', type=int, default=600)
args = ap.parse_args()
root = Path(args.source)
out = Path(args.out) / args.arm
out.mkdir(parents=True, exist_ok=True)
configs = ['configs/metaworld/base.yaml', 'configs/metaworld/tasks/hammer.yaml', 'configs/metaworld/variants/chamfer_lean.yaml',
           'configs/metaworld/variants/chamfer_lean_randmask.yaml']
if args.arm != 'chlrm':
    configs.append(f'configs/metaworld/variants/enc_{args.arm}.yaml')
name = f'encsmoke-{args.arm}-hammer'
cmd = ['/home/ws/ws/splatter4d/.venv/bin/python', '-I', 'scripts/train.py', '--config', *configs, '--name', name,
       '--output-root', str(out.relative_to(root)) if out.is_relative_to(root) else str(out),
       '--set', f'train.steps={args.steps}', 'train.seed=0', 'train.eval_at_start=false', 'wandb.enabled=false']
env = {**os.environ, 'CUDA_VISIBLE_DEVICES': args.gpu, 'MUJOCO_GL': 'egl', 'PYOPENGL_PLATFORM': 'egl'}
log = open(out / 'console.log', 'w')
start = time.time()
proc = subprocess.Popen(['setsid', *cmd], cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)


def session_pss(sid):
    total = 0
    for pid in os.listdir('/proc'):
        if not pid.isdigit():
            continue
        try:
            if int(Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[3]) != sid:
                continue
            fields = {line.split(':')[0]: int(line.split()[1]) for line in Path(f'/proc/{pid}/smaps_rollup').read_text().splitlines()[1:]}
            total += (fields.get('Pss_Anon', 0) + fields.get('Pss_Shmem', 0)) * 1024
        except (OSError, ValueError, IndexError):
            continue
    return total


peak = 0
while proc.poll() is None:
    peak = max(peak, session_pss(proc.pid))
    time.sleep(2)
wall = time.time() - start
rows = [json.loads(line) for line in (out / name / 'metrics.jsonl').read_text().splitlines() if line.strip()] \
    if (out / name / 'metrics.jsonl').is_file() else []
train = [r for r in rows if 'train/step_time' in r and r.get('step', 0) > 100]
record = {'arm': args.arm, 'source': str(root), 'gpu': args.gpu, 'host': 'local', 'exit_code': proc.returncode,
          'steps': args.steps, 'wall_seconds': wall,
          'median_step_time_s': statistics.median(r['train/step_time'] for r in train) if train else None,
          'median_gpu_time_s': statistics.median(r['train/gpu_time'] for r in train if 'train/gpu_time' in r) if train else None,
          'peak_gpu_mem_gb': max((r['train/gpu_mem_gb'] for r in rows if 'train/gpu_mem_gb' in r), default=None),
          'peak_session_pss_bytes': peak,
          'last_logged': {k: v for k, v in (train[-1] if train else {}).items()
                          if k.startswith(('loss/', 'metric/hardneg', 'metric/psnr'))},
          'finite': all(isinstance(v, (int, float)) and v == v for r in train for k, v in r.items() if k.startswith('loss/'))}
(out / 'smoke.json').write_text(json.dumps(record, indent=1) + '\n')
print(json.dumps({k: record[k] for k in ('arm', 'exit_code', 'median_step_time_s', 'peak_gpu_mem_gb', 'peak_session_pss_bytes', 'finite')}))
