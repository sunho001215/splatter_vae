"""Four-arm campaign (arm4, user directive 2026-11-06): local functional smoke of each arm's code path on ONE authorized
local MIG instance. Not an experiment row; outputs under <out>/<arm>/ (outside runs/, never a campaign run directory).

Per arm (A ocsw4, B ochm, C c50dg, D c50vis), hammer, sequentially:
1. pretraining through scripts/train.py with the arm's REGISTERED configuration (config list and --set overrides exactly as
   the registration rows) plus smoke-only keys: train.stop_step=<steps> (the 200k schedule, stopped early, so the learning
   rate is the real run's), train.save_every=<steps>, train.log_every=50, wandb.enabled=false. The step-0 evaluation is
   kept (eval_at_start): with the same seed and architecture it must reproduce the matched-initialisation reference's
   step-0 evaluation (B: the Original hammer run i2-m3d-seed0-hammer; A, C, D: cst50-seed0-hammer-200k), within GPU
   nondeterminism; the predicted config_sha256 (ops/jobs/run_identity.numeric_config) must equal the run's own.
2. export of the final checkpoint (scripts/export_encoder.py) and a reload check (stem, width / depth / heads, policy state
   dimension 256, finite);
3. DrM, hammer, seed 2000, <rl-steps> agent steps, snapshots every 5000, 2 episodes per camera, no W&B;
4. the evaluator (scripts/eval_rl.py) on those snapshots.
Records: s/it (median after step 100), peak GPU memory, peak session non-file PSS per stage (footprints), exit codes,
finiteness, the step-0 comparison, loss terms present. Gate before every stage: memory_watch.py fits <bytes>.
Runs from the main checkout (its docs/tests.json evidence matches its sources; s4d / scripts / configs identical to
the arm4 branch).

    nice -n 15 .venv/bin/python -I runs/setup/tools/arm4_smoke_local_20261106.py --gpu <MIG uuid> --out <dir> [--arms A B C D]
"""
import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/home/ws/ws/splatter4d')
sys.path.insert(0, str(ROOT / 'runs/setup/tools'))
import arm4_common as C  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--gpu', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--arms', nargs='+', default=['A', 'B', 'C', 'D'])
ap.add_argument('--steps', type=int, default=400)
ap.add_argument('--rl-steps', type=int, default=6000)
ap.add_argument('--task', default='hammer')
args = ap.parse_args()
PY = str(ROOT / '.venv/bin/python')
env = {**os.environ, 'CUDA_VISIBLE_DEVICES': args.gpu, 'MUJOCO_GL': 'egl', 'PYOPENGL_PLATFORM': 'egl',
       'WANDB_MODE': 'disabled', 'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8'}
FOOT = {'pretrain': 14e9, 'export': 4e9, 'rl': 6e9, 'eval': 8e9}


def fits(nbytes):
    r = subprocess.run([PY, '-I', str(ROOT / 'runs/setup/tools/memory_watch.py'), 'fits', str(int(nbytes))],
                       capture_output=True, text=True, cwd=ROOT)
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return {'fits': False, 'error': r.stdout[-500:] + r.stderr[-500:]}


def session_pss(sid):
    total = 0
    for pid in os.listdir('/proc'):
        if not pid.isdigit():
            continue
        try:
            if int(Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[3]) != sid:
                continue
            fields = {line.split(':')[0]: int(line.split()[1])
                      for line in Path(f'/proc/{pid}/smaps_rollup').read_text().splitlines()[1:]}
            total += (fields.get('Pss_Anon', 0) + fields.get('Pss_Shmem', 0)) * 1024
        except (OSError, ValueError, IndexError):
            continue
    return total


def run(cmd, log, stage):
    gate = fits(FOOT[stage])
    if not gate.get('fits'):
        return {'exit_code': None, 'refused_by_memory_gate': gate}
    start = time.time()
    with open(log, 'w') as stream:
        proc = subprocess.Popen(['setsid', 'nice', '-n', '15', *cmd], cwd=ROOT, env=env, stdout=stream,
                                stderr=subprocess.STDOUT)
        peak = 0
        while proc.poll() is None:
            peak = max(peak, session_pss(proc.pid))
            time.sleep(2)
    return {'exit_code': proc.returncode, 'wall_seconds': round(time.time() - start, 1), 'peak_session_pss_bytes': peak,
            'memory_gate': gate}


def rows(path):
    if not path.is_file():
        return []
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


summary = {}
for arm in args.arms:
    spec = C.ARMS[arm]
    out = Path(args.out) / spec['prefix']
    out.mkdir(parents=True, exist_ok=True)
    record = {'arm': arm, 'prefix': spec['prefix'], 'gpu': args.gpu, 'task': args.task, 'source': str(ROOT)}
    files, sets = C.pretrain_stack(arm, args.task)
    smoke = [f'train.stop_step={args.steps}', f'train.save_every={args.steps}', 'train.log_every=50',
             'wandb.enabled=false']
    name = f'arm4smoke-{spec["prefix"]}-{args.task}'
    record['config'] = {'files': files, 'set': sets, 'smoke_only': smoke}
    record['pretrain'] = run([PY, '-I', 'scripts/train.py', '--config', *files, '--name', name, '--output-root', str(out),
                              '--no-wandb', '--set', *sets, *smoke], out / 'pretrain.log', 'pretrain')
    run_dir = out / name
    met = rows(run_dir / 'metrics.jsonl')
    train = [r for r in met if 'train/step_time' in r and r.get('step', 0) > 100]
    step0 = next((r for r in met if r.get('step') == 0 and 'val/psnr@s2' in r), {})
    ref_dir = C.matched_init_reference(arm, args.task)
    ref0 = next((r for r in rows(ref_dir / 'metrics.jsonl') if r.get('step') == 0 and 'val/psnr@s2' in r), {})
    keys = sorted(k for k in step0 if k.startswith('val/') and k in ref0)
    cfg_yaml = run_dir / 'config.yaml'
    predicted = C.config_sha256(C.load(files, sets + smoke))
    import yaml
    recorded = yaml.safe_load(cfg_yaml.read_text())['run'].get('config_sha256') if cfg_yaml.is_file() else None
    record['pretrain'].update(
        median_step_time_s=statistics.median(r['train/step_time'] for r in train) if train else None,
        median_gpu_time_s=statistics.median(r['train/gpu_time'] for r in train if 'train/gpu_time' in r) if train else None,
        peak_gpu_mem_gb=max((r['train/gpu_mem_gb'] for r in met if 'train/gpu_mem_gb' in r), default=None),
        loss_terms=sorted({k.split('/', 1)[1].rsplit('_t', 1)[0] for r in train for k in r if k.startswith('loss/')}),
        finite=bool(train) and all(isinstance(v, (int, float)) and v == v for r in train for k, v in r.items()
                                   if k.startswith('loss/')),
        last_logged={k: v for k, v in (train[-1] if train else {}).items() if k in ('step', 'loss/total', 'metric/psnr')},
        config_sha256={'predicted': predicted, 'recorded': recorded, 'equal': predicted == recorded},
        step0_vs_matched_init_reference={
            'reference_run': str(ref_dir.relative_to(ROOT)), 'keys_compared': len(keys),
            'max_abs_diff': max((abs(step0[k] - ref0[k]) for k in keys), default=None),
            'max_rel_diff': max((abs(step0[k] - ref0[k]) / max(abs(ref0[k]), 1e-12) for k in keys), default=None),
            'bitwise_equal_keys': sum(step0[k] == ref0[k] for k in keys),
            'psnr_s2': [step0.get('val/psnr@s2'), ref0.get('val/psnr@s2')]})
    ckpts = sorted((run_dir / 'checkpoints').glob('step_*.pt'))
    export = out / 'export/encoder.pt'
    export.parent.mkdir(parents=True, exist_ok=True)
    if record['pretrain'].get('exit_code') == 0 and ckpts:
        record['export'] = run([PY, '-I', 'scripts/export_encoder.py', '--run', str(run_dir), '--checkpoint',
                                str(ckpts[-1]), '--out', str(export)], out / 'export.log', 'export')
        check = subprocess.run([PY, '-I', '-c', f'''
import json, sys, torch
sys.path.insert(0, {str(ROOT)!r})
from s4d.model.encoder import load_encoder, PatchCNNStem
enc = load_encoder({str(export)!r}).eval()
c = enc.cfg
x = torch.rand(1, 3, 3, 128, 128)
with torch.no_grad():
    s = enc.policy_state(x)
print(json.dumps({{"stem": c.stem, "stem_class": type(enc.patch_embed).__name__, "width": c.width, "depth": c.depth,
                   "heads": c.heads, "patch_size": c.patch_size, "slot_dim": c.slot_dim, "mask_mode": c.mask_mode,
                   "mask_ratio": c.mask_ratio, "parameters": sum(p.numel() for p in enc.parameters()),
                   "policy_state_shape": list(s.shape), "finite": bool(torch.isfinite(s).all())}}))
'''], cwd=ROOT, env={**env, 'CUDA_VISIBLE_DEVICES': ''}, capture_output=True, text=True)
        record['export']['check'] = json.loads(check.stdout.strip().splitlines()[-1]) if check.returncode == 0 \
            else check.stderr[-2000:]
    if record.get('export', {}).get('exit_code') == 0:
        rl_dir = out / 'drm'
        record['rl'] = run([PY, '-I', 'scripts/train_rl.py', '--task', args.task, '--encoder', 'splatter4d', '--seed',
                            '2000', '--run-dir', str(rl_dir), '--set', f'train.num_train_steps={args.rl_steps}',
                            f'vision.export_path={export}', 'eval.every_steps=5000', 'eval.episodes_per_train_camera=2',
                            'eval.episodes_per_heldout_camera=2', 'eval.episodes_per_trajectory=2',
                            'train.checkpoint_every_steps=5000', 'wandb.enabled=false'], out / 'rl.log', 'rl')
        tj = rows(rl_dir / 'train.jsonl')
        record['rl'].update(final_json=(rl_dir / 'final.json').is_file(), train_records=len(tj),
                            last_train_record=tj[-1] if tj else None,
                            snapshots=sorted(p.name for p in (rl_dir / 'snapshots').glob('*.pt')))
        if record['rl'].get('exit_code') == 0:
            record['eval'] = run([PY, '-I', 'scripts/eval_rl.py', '--run-dir', str(rl_dir), '--train-job',
                                  f'arm4smoke-{spec["prefix"]}-drm', '--workers', '2', '--envs-per-worker', '4',
                                  '--poll-seconds', '5', '--output-dir', str(out / 'drm-eval')], out / 'eval.log', 'eval')
            ev = rows(out / 'drm-eval/eval.jsonl')
            record['eval'].update(records=len(ev), steps=[e.get('step') for e in ev],
                                  keys=sorted(k for k in (ev[-1] if ev else {}) if k.endswith('_success'))[:20])
    (out / 'smoke.json').write_text(json.dumps(record, indent=1, default=str) + '\n')
    summary[arm] = {k: ({kk: vv for kk, vv in v.items() if kk in ('exit_code', 'median_step_time_s', 'peak_gpu_mem_gb',
                                                                   'peak_session_pss_bytes', 'finite', 'check', 'records',
                                                                   'final_json', 'config_sha256',
                                                                   'step0_vs_matched_init_reference', 'loss_terms',
                                                                   'wall_seconds')}
                        if isinstance(v, dict) else v) for k, v in record.items() if k in ('pretrain', 'export', 'rl', 'eval')}
    print(json.dumps({arm: summary[arm]}, default=str), flush=True)
(Path(args.out) / 'summary.json').write_text(json.dumps(summary, indent=1, default=str) + '\n')
