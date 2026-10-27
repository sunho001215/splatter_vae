"""Local smoke check of the T=1 spatial ablation arm (t1spatial; user directive 2026-10-27, paper-definition-20261027.json)
on a chosen encoder architecture: perframe_spatial.yaml on top of chamfer_lean + chamfer_lean_randmask + the arm's
variant file(s); DrM with vision.frame_features=stack. Adapted from encx_smoke_local.py, whose text follows. (enc_cs.yaml (patch-local conv
stem) stacked with enc_w4.yaml (width 256, depth 4, 4 heads) on top of chamfer_lean + chamfer_lean_randmask, run from the
encoder-pilot release checkout (``--source``, d281258) on one idle local MIG slice. Not an experiment row.

Steps (each must exit 0):
1. pretraining, hammer, 600 steps (train.steps=600: inside the 10k warm-up the learning rate equals the 200k schedule's),
   no start evaluation, no W&B, end evaluation and checkpoint (as enc_smoke_local.py);
2. export of that checkpoint (scripts/export_encoder.py) and a reload check: load_encoder, stem type, width / depth / heads,
   parameter count, policy_state dimension 256 on a 3-frame 128 x 128 window;
3. DrM, hammer, 6000 agent steps (2000 seed steps, then updates), proprioception as registered, snapshots every 5000,
   2 evaluation episodes per camera, no W&B;
4. the evaluator (scripts/eval_rl.py) on that run's snapshots.
Output: <out>/csw4/smoke.json (s/it, peak GPU memory, peak session PSS, finiteness, export check, RL and evaluator exit
codes and logged losses).

    python -I encx_smoke_local.py --source <release root> --gpu <MIG uuid> --out <dir>
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
ap.add_argument('--arch', required=True, choices=('chlrm', 'cs', 'w4', 'csw4'))
ap.add_argument('--gpu', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--steps', type=int, default=600)
ap.add_argument('--rl-steps', type=int, default=6000)
ap.add_argument('--reuse-pretrain', action='store_true', help='keep a finished smoke pretraining (completion.json) and its record')
args = ap.parse_args()
root = Path(args.source)
out = Path(args.out) / f'{args.arch}-t1spatial'
out.mkdir(parents=True, exist_ok=True)
PY = '/home/ws/ws/splatter4d/.venv/bin/python'
V = 'configs/metaworld/variants/'
ARCH_FILES = {'chlrm': [], 'cs': [V + 'enc_cs.yaml'], 'w4': [V + 'enc_w4.yaml'], 'csw4': [V + 'enc_cs.yaml', V + 'enc_w4.yaml']}
configs = ['configs/metaworld/base.yaml', 'configs/metaworld/tasks/hammer.yaml', V + 'chamfer_lean.yaml',
           V + 'chamfer_lean_randmask.yaml', *ARCH_FILES[args.arch], V + 'perframe_spatial.yaml']
name = f'ablsmoke-t1spatial-{args.arch}-hammer'
env = {**os.environ, 'CUDA_VISIBLE_DEVICES': args.gpu, 'MUJOCO_GL': 'egl', 'PYOPENGL_PLATFORM': 'egl', 'WANDB_MODE': 'disabled'}


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


def run(cmd, logname):
    log = open(out / logname, 'w')
    start = time.time()
    proc = subprocess.Popen(['setsid', 'nice', '-n', '5', *cmd], cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
    peak = 0
    while proc.poll() is None:
        peak = max(peak, session_pss(proc.pid))
        time.sleep(2)
    return {'exit_code': proc.returncode, 'wall_seconds': time.time() - start, 'peak_session_pss_bytes': peak}


record = {'arm': 't1spatial', 'arch': args.arch, 'source': str(root), 'gpu': args.gpu, 'host': 'local', 'configs': configs}
pre_dir = out / name
rel = str(out.relative_to(root)) if out.is_relative_to(root) else str(out)
reuse = args.reuse_pretrain and (pre_dir / 'completion.json').is_file() and (out / 'smoke.json').is_file()
if reuse:
    record['pretrain'] = json.loads((out / 'smoke.json').read_text())['pretrain']
    record['pretrain']['reused'] = True
else:
  record['pretrain'] = run([PY, '-I', 'scripts/train.py', '--config', *configs, '--name', name, '--output-root', rel,
                          '--set', f'train.steps={args.steps}', 'train.seed=0', 'train.eval_at_start=false',
                          'wandb.enabled=false'], 'pretrain.log')
rows = [json.loads(line) for line in (pre_dir / 'metrics.jsonl').read_text().splitlines() if line.strip()] \
    if (pre_dir / 'metrics.jsonl').is_file() else []
train = [r for r in rows if 'train/step_time' in r and r.get('step', 0) > 100]
record['pretrain'].update(
    steps=args.steps,
    median_step_time_s=statistics.median(r['train/step_time'] for r in train) if train else None,
    peak_gpu_mem_gb=max((r['train/gpu_mem_gb'] for r in rows if 'train/gpu_mem_gb' in r), default=None),
    first_logged={k: v for k, v in (rows[0] if rows else {}).items() if k in ('loss/total', 'loss/infonce', 'metric/psnr')},
    last_logged={k: v for k, v in (train[-1] if train else {}).items() if k.startswith(('loss/', 'metric/psnr'))},
    finite=bool(train) and all(isinstance(v, (int, float)) and v == v for r in train for k, v in r.items()
                               if k.startswith('loss/')))
ckpts = sorted((pre_dir / 'checkpoints').glob('step_*.pt'))
export = out / 'export/encoder.pt'
export.parent.mkdir(parents=True, exist_ok=True)
if record['pretrain']['exit_code'] == 0 and ckpts:
    record['export'] = run([PY, '-I', 'scripts/export_encoder.py', '--run', str(pre_dir), '--checkpoint', str(ckpts[-1]),
                            '--out', str(export)], 'export.log')
    check = subprocess.run([PY, '-I', '-c', f'''
import json, sys, torch
sys.path.insert(0, {str(root)!r})
from torch.utils.flop_counter import FlopCounterMode
from s4d.model.encoder import load_encoder, PatchCNNStem
payload = torch.load({str(export)!r}, map_location="cpu", weights_only=True)
enc = load_encoder({str(export)!r}).eval()
c = enc.cfg
x = torch.rand(1, 1, 3, 128, 128)  # one frame: the stack adapter encodes each of the three frames like this
with torch.no_grad(), FlopCounterMode(display=False) as fc:
    s = enc.policy_state(x)
print(json.dumps({{"stem": c.stem, "stem_class": type(enc.patch_embed).__name__,
                   "stem_proj_out": enc.patch_embed.proj.out_channels if isinstance(enc.patch_embed, PatchCNNStem) else None,
                   "width": c.width, "depth": c.depth, "heads": c.heads, "patch_size": c.patch_size, "slot_dim": c.slot_dim,
                   "mask_mode": c.mask_mode, "mask_ratio": c.mask_ratio,
                   "parameters": sum(p.numel() for p in enc.parameters()), "gflops_per_frame": fc.get_total_flops() / 1e9,
                   "num_frames": c.num_frames, "policy_frames": payload.get("policy_frames"),
                   "policy_state_shape": list(s.shape), "finite": bool(torch.isfinite(s).all())}}))
'''], cwd=root, env={**env, 'CUDA_VISIBLE_DEVICES': ''}, capture_output=True, text=True)
    record['export']['check'] = json.loads(check.stdout.strip().splitlines()[-1]) if check.returncode == 0 else check.stderr[-2000:]
if record.get('export', {}).get('exit_code') == 0:
    rl_dir = out / 'drm'
    record['rl'] = run([PY, '-I', 'scripts/train_rl.py', '--task', 'hammer', '--encoder', 'splatter4d', '--seed', '2000',
                        '--run-dir', str(rl_dir), '--set', f'train.num_train_steps={args.rl_steps}',
                        f'vision.export_path={export}', 'eval.every_steps=5000', 'eval.episodes_per_train_camera=2',
                        'eval.episodes_per_heldout_camera=2', 'eval.episodes_per_trajectory=2',
                        'train.checkpoint_every_steps=5000', 'wandb.enabled=false', 'vision.frame_features=stack'], 'rl.log')
    tj = [json.loads(line) for line in (rl_dir / 'train.jsonl').read_text().splitlines() if line.strip()] \
        if (rl_dir / 'train.jsonl').is_file() else []
    import yaml
    rl_cfg = yaml.safe_load((rl_dir / 'config.yaml').read_text()) if (rl_dir / 'config.yaml').is_file() else {}
    record['rl'].update(frame_features=(rl_cfg.get('vision') or {}).get('frame_features'))
    record['rl'].update(final_json=(rl_dir / 'final.json').is_file(), train_records=len(tj),
                        last_train_record=tj[-1] if tj else None,
                        snapshots=sorted(p.name for p in (rl_dir / 'snapshots').glob('*.pt')))
    if record['rl']['exit_code'] == 0:
        record['eval'] = run([PY, '-I', 'scripts/eval_rl.py', '--run-dir', str(rl_dir), '--train-job', f'ablsmoke-t1spatial-{args.arch}-drm',
                              '--workers', '2', '--envs-per-worker', '4', '--poll-seconds', '5',
                              '--output-dir', str(out / 'drm-eval')], 'eval.log')
        ev = [json.loads(line) for line in (out / 'drm-eval/eval.jsonl').read_text().splitlines() if line.strip()] \
            if (out / 'drm-eval/eval.jsonl').is_file() else []
        record['eval'].update(records=len(ev), steps=[e.get('step') for e in ev])
(out / 'smoke.json').write_text(json.dumps(record, indent=1, default=str) + '\n')
print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk in
                                                         ('exit_code', 'median_step_time_s', 'peak_gpu_mem_gb', 'finite',
                                                          'check', 'records', 'final_json', 'train_records',
                                                          'frame_features')})
                  for k, v in record.items() if k in ('pretrain', 'export', 'rl', 'eval')}, default=str))
