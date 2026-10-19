"""Local cleanup (user request 2026-10-19, second request): completed-training replay buffers, obsolete RL policy
snapshots, and unneeded intermediate pretraining checkpoints under /home/ws/ws/splatter4d/runs. No free-space stopping
rule; W&B files, datasets, source code and remote files untouched.

Reference set: every `runs/...{.pt,.npz}` path named in experiments/queue.yaml, experiments/registry.jsonl,
runs/setup/** (registrations, manifests, proofs, reports) and docs/** is kept.

RL run R (runs/R with snapshots/ or replay/) qualifies only if: registry status `done` (remote: verified final sync);
final.json completed with steps == config train.num_train_steps (total); evaluator R-eval (when registered) `done`
(remote: verified final sync); eval history ends at total; Stage 2 runs have their closeout proof; no queued job that is
not done depends on R or R-eval. Then
  - replay: every regular file under runs/R/replay/ is deleted;
  - snapshots: snapshots/step_N.pt with N < max snapshot step, N recorded in the eval history, not referenced, deleted;
    the final (max-step) snapshot is the final RL policy and is kept; checkpoints/latest.pt (final full state) is kept.
Pretraining run D (runs/D or runs/pretrain/D with checkpoints/step_N.pt, no snapshots/) qualifies only if registry status
`done`, completion.json says completed with steps == max checkpoint step == config train.steps. Then
checkpoints/step_N.pt is deleted when N is not the final step, not a multiple of 100,000 (reported endpoints: 100k
screening, 200k, 300k), not referenced, and eval/step_N/summary.json exists. latest.pt and non-step files are kept.
Everything else (failed/crashed/unregistered runs, other file names) is preserved. Each file is rechecked (state,
regular file inside runs/, not open by any process) immediately before deletion.

    python -I disk_cleanup_local_models_20261019.py [--execute]
"""
import gzip
import json
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path('/home/ws/ws/splatter4d')
RUNS = ROOT / 'runs'
sys.path.insert(0, str(ROOT))
from scripts import jobs  # noqa: E402

OUT = ROOT / 'runs/setup/disk-cleanup-20261019'
EXECUTE = '--execute' in sys.argv
REF = re.compile(r'runs/[A-Za-z0-9_.\-/]+?\.(?:pt|npz)\b')
STEP = re.compile(r'^step_(\d{7})\.pt$')


def references():
    refs = set()
    sources = [ROOT / 'experiments/queue.yaml', ROOT / 'experiments/registry.jsonl']
    for base in (ROOT / 'runs/setup', ROOT / 'docs'):
        sources += [p for p in base.rglob('*') if p.is_file() and p.suffix in ('.json', '.jsonl', '.yaml', '.yml', '.md',
                                                                             '.txt', '.log', '.py')
                    and p.stat().st_size < 100e6]
    for path in sources:
        try:
            text = path.read_text(errors='ignore')
        except OSError:
            continue
        refs.update(m.group(0).lstrip('./') for m in REF.finditer(text))
    return refs


def alloc(path):
    return path.lstat().st_blocks * 512


def eval_steps(run):
    for path in (RUNS / f'{run}-eval' / 'eval.jsonl', RUNS / run / 'eval.jsonl'):
        if path.is_file():
            return [json.loads(line)['step'] for line in path.read_text().splitlines() if line.strip()]
    return None


def synced(s):
    return s.get('host', 'local') != 'remote' or (s.get('sync', {}).get('final') and s.get('sync', {}).get('verified'))


def rl_check(run, state, queue):
    s = state.get(run)
    if s is None:
        return 'not in registry'
    if s['status'] != 'done' or s.get('unknown') or not synced(s):
        return f"status {s['status']}"
    final = json.loads((RUNS / run / 'final.json').read_text())
    total = yaml.safe_load((RUNS / run / 'config.yaml').read_text())['train']['num_train_steps']
    if final.get('completed') is not True or final.get('steps') != total:
        return 'budget not verified'
    ev = f'{run}-eval'
    if ev in queue or ev in state:
        es = state.get(ev, {})
        if es.get('status') != 'done' or not synced(es):
            return 'evaluator not done'
    steps = eval_steps(run)
    if not steps or max(steps) != total:
        return 'evaluation history incomplete'
    if run.startswith('s2rl-') and not (ROOT / 'runs/setup' / f's2-{run}-completion.json').is_file():
        return 'closeout proof missing'
    for job in queue.values():
        if (run in job.get('deps', []) or ev in job.get('deps', [])) and state.get(job['id'], {}).get('status') != 'done':
            return f"pending dependent {job['id']}"
    return None


def pretrain_check(run_dir, state):
    run = run_dir.name
    s = state.get(run)
    if s is None:
        return 'not in registry', None
    if s['status'] != 'done' or s.get('unknown') or not synced(s):
        return f"status {s['status']}", None
    comp = run_dir / 'completion.json'
    if not comp.is_file():
        return 'no completion.json', None
    c = json.loads(comp.read_text())
    steps = sorted(int(m.group(1)) for f in os.listdir(run_dir / 'checkpoints') if (m := STEP.match(f)))
    cfg_steps = yaml.safe_load((run_dir / 'config.yaml').read_text()).get('train', {}).get('steps')
    if c.get('status') != 'completed' or c.get('steps') != steps[-1] or cfg_steps != steps[-1]:
        return f"completion not verified ({c.get('steps')}, max {steps[-1]}, config {cfg_steps})", None
    return None, steps[-1]


state = jobs.read_registry()
queue = {j['id']: j for j in jobs.load_queue()['jobs']}
refs = references()
cands, kept = [], {}


def keep(reason, path):
    kept.setdefault(reason, {'files': 0, 'bytes': 0, 'examples': []})
    kept[reason]['files'] += 1
    kept[reason]['bytes'] += alloc(path)
    if len(kept[reason]['examples']) < 4:
        kept[reason]['examples'].append(str(path.relative_to(ROOT)))


# RL runs
for run_dir in sorted(p for p in RUNS.iterdir() if p.is_dir() and not p.is_symlink()
                      and ((p / 'snapshots').is_dir() or (p / 'replay').is_dir())):
    run = run_dir.name
    files = [Path(dp) / f for sub in ('replay', 'snapshots') if (run_dir / sub).is_dir()
             for dp, _, fs in os.walk(run_dir / sub, followlinks=False) for f in fs]
    try:
        verdict = rl_check(run, state, queue) if (run_dir / 'final.json').is_file() else 'no final.json'
    except (OSError, KeyError, ValueError, TypeError) as exc:
        verdict = f'check error {type(exc).__name__}'
    if verdict:
        for f in files:
            keep(f'RL run not eligible: {verdict.split(" (")[0]}', f)
        continue
    steps = set(eval_steps(run))
    snaps = {int(m.group(1)): run_dir / 'snapshots' / f for f in os.listdir(run_dir / 'snapshots')
             if (m := STEP.match(f))} if (run_dir / 'snapshots').is_dir() else {}
    last = max(snaps) if snaps else None
    for f in files:
        rel = str(f.relative_to(ROOT))
        if f.is_symlink() or not f.is_file():
            keep('not a regular file', f)
        elif rel in refs:
            keep('referenced in campaign records', f)
        elif f.parent.name == 'replay':
            cands.append({'path': rel, 'run': run, 'category': 'replay', 'bytes': alloc(f),
                          'reason': 'replay of completed, evaluated, closed-out training'})
        else:
            m = STEP.match(f.name)
            if not m:
                keep('snapshot with non-step name', f)
            elif int(m.group(1)) == last:
                keep('final RL policy snapshot', f)
            elif int(m.group(1)) not in steps:
                keep('snapshot step not in eval history', f)
            else:
                cands.append({'path': rel, 'run': run, 'category': 'policy_snapshot', 'bytes': alloc(f),
                              'reason': f'intermediate snapshot, evaluated at step {int(m.group(1))}; final {last} kept'})

# pretraining runs
for base in (RUNS, RUNS / 'pretrain'):
    for run_dir in sorted(p for p in base.iterdir() if p.is_dir() and not p.is_symlink()
                          and (p / 'checkpoints').is_dir() and not (p / 'snapshots').is_dir()):
        files = [run_dir / 'checkpoints' / f for f in os.listdir(run_dir / 'checkpoints')]
        if not any(STEP.match(f.name) for f in files):
            continue
        try:
            verdict, final = pretrain_check(run_dir, state)
        except (OSError, KeyError, ValueError, TypeError) as exc:
            verdict, final = f'check error {type(exc).__name__}', None
        for f in files:
            rel = str(f.relative_to(ROOT))
            m = STEP.match(f.name)
            if verdict:
                keep(f'pretraining run not eligible: {verdict.split(" (")[0]}', f)
            elif f.is_symlink() or not f.is_file():
                keep('not a regular file', f)
            elif not m:
                keep('pretraining resume state / non-step file', f)
            elif int(m.group(1)) == final:
                keep('final pretrained model', f)
            elif int(m.group(1)) % 100000 == 0:
                keep('100k-multiple milestone (reported endpoints)', f)
            elif rel in refs:
                keep('referenced in campaign records', f)
            elif not (run_dir / 'eval' / f'step_{int(m.group(1)):07d}' / 'summary.json').is_file():
                keep('no recorded evaluation for this step', f)
            else:
                cands.append({'path': rel, 'run': run_dir.name, 'category': 'checkpoint', 'bytes': alloc(f),
                              'reason': f'intermediate pretraining checkpoint, evaluation recorded; final {final} kept'})

by_cat = {}
for c in cands:
    by_cat.setdefault(c['category'], [0, 0])
    by_cat[c['category']][0] += 1
    by_cat[c['category']][1] += c['bytes']
manifest = {'recorded_local': datetime.now().astimezone().isoformat(), 'host': 'local', 'request': 'second local request',
            'free_bytes_before': shutil.disk_usage('/home/ws').free, 'reference_paths': len(refs),
            'candidates_by_category': {k: {'files': v[0], 'bytes': v[1]} for k, v in by_cat.items()},
            'kept_by_reason': kept, 'candidates': cands}
for k, (n, b) in by_cat.items():
    print(f'candidate {k:16s} {n:6d} files {b / 1e9:7.1f} GB')
for k, v in sorted(kept.items(), key=lambda x: -x[1]['bytes']):
    print(f"kept {k:55s} {v['files']:6d} files {v['bytes'] / 1e9:7.1f} GB")
if not EXECUTE:
    with gzip.open(OUT / 'local-models-manifest.json.gz', 'wt') as fh:
        json.dump(manifest, fh, indent=1)
    raise SystemExit(0)

# recheck immediately before deletion, category by category
held = set()
for pid in os.listdir('/proc'):
    if pid.isdigit():
        try:
            for fd in os.listdir(f'/proc/{pid}/fd'):
                try:
                    held.add(os.readlink(f'/proc/{pid}/fd/{fd}'))
                except OSError:
                    pass
        except OSError:
            pass
state = jobs.read_registry()
results = {}
for category in ('replay', 'policy_snapshot', 'checkpoint'):
    free0 = shutil.disk_usage('/home/ws').free
    done_bytes, done_files, skipped = 0, 0, []
    for c in [c for c in cands if c['category'] == category]:
        path = ROOT / c['path']
        real = os.path.realpath(path)
        if (not path.is_file() or path.is_symlink() or not real.startswith(str(RUNS) + '/') or real in held
                or state.get(c['run'], {}).get('status') != 'done' or state.get(c['run'], {}).get('unknown')):
            skipped.append(c['path'])
            continue
        path.unlink()
        c['deleted'] = True
        done_bytes += c['bytes']
        done_files += 1
    results[category] = {'files': done_files, 'allocated_bytes': done_bytes, 'skipped': skipped,
                         'free_before': free0, 'free_after': shutil.disk_usage('/home/ws').free}
    print(f"{category}: deleted {done_files} files, {done_bytes / 1e9:.1f} GB allocated; df "
          f"{free0 / 1e9:.1f} -> {results[category]['free_after'] / 1e9:.1f} GB; skipped {len(skipped)}")
manifest.update(executed_local=datetime.now().astimezone().isoformat(), results=results,
                free_bytes_after=shutil.disk_usage('/home/ws').free)
with gzip.open(OUT / 'local-models-manifest.json.gz', 'wt') as fh:
    json.dump(manifest, fh, indent=1)
