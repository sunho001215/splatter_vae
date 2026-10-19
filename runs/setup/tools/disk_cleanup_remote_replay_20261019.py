"""Remote stage 1 (user request 2026-10-19): delete replay buffers (runs/<id>/replay/*, e.g. ram_snapshot.npz) of
completed remote training runs under the authorized root /home/compu/kaist/sunho/runs. Remote checkpoints, snapshots,
final models, results, logs and W&B stay. A run's replay is deleted only if all of:
  1. no running managed container carries its run id (docker ps, label s4d.run_id);
  2. registry status `done` on host remote with a verified final result sync;
  3. the remote final.json has the sha256 recorded in that verified sync, and reports completed with steps equal to
     the configured total (train_rl.py writes final.json only when the loop reaches the configured total);
  4. the evaluator (<id>-eval, when registered) is `done` with a verified final sync, and the local eval history ends
     at the final step with one row per evaluation interval; Stage 2 runs also need their closeout proof;
  5. the final policy snapshot (snapshots/step_<final>.pt) and checkpoints/latest.pt exist remotely (kept);
  6. no queued job that is not done depends on the run or its evaluator.
Each remaining replay file is rechecked (container, registry) immediately before deletion.

    python -I disk_cleanup_remote_replay_20261019.py            # manifest only
    python -I disk_cleanup_remote_replay_20261019.py --execute
"""
import hashlib
import json
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path('/home/ws/ws/splatter4d')
sys.path.insert(0, str(ROOT))
from scripts import jobs  # noqa: E402

REMOTE = '/home/compu/kaist/sunho'
OUT = ROOT / 'runs/setup/disk-cleanup-20261019'
EXECUTE = '--execute' in sys.argv


def ssh(command, timeout=900):
    result = subprocess.run([sys.executable, '-I', str(ROOT / 's4d/remote_access.py'), 'ssh', '--idempotent', '--',
                             command], capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stderr[-500:])
    return result.stdout


def remote_state():
    out = ssh(f"cd {REMOTE}/runs && df -B1 {REMOTE} | tail -1 && echo __RUNNING__ && "
              "docker ps --filter label=s4d.managed=true --format '{{.Label \"s4d.run_id\"}}' && echo __REPLAY__ && "
              "find . -mindepth 3 -maxdepth 3 -path './*/replay/*' -type f -printf '%k %s %T@ %p\\n'")
    head, rest = out.split('__RUNNING__\n', 1)
    running, replay = rest.split('__REPLAY__\n', 1)
    files = []
    for line in replay.splitlines():
        kb, size, mtime, path = line.split(' ', 3)
        files.append({'path': f'runs/{path[2:]}', 'run': path[2:].split('/')[0], 'bytes': int(kb) * 1024,
                      'apparent_bytes': int(size), 'mtime': float(mtime)})
    return int(head.split()[3]), set(running.split()), files


def eval_rows(run):
    for path in (ROOT / 'runs' / f'{run}-eval' / 'eval.jsonl', ROOT / 'runs' / run / 'eval.jsonl'):
        if path.is_file():
            return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return None


def check(run, state, queue, running, remote_facts):
    s = state.get(run)
    if run in running:
        return 'container running'
    if s is None:
        return 'not in registry'
    if s['status'] != 'done' or s.get('unknown') or s.get('host') != 'remote':
        return f"registry status {s['status']} host {s.get('host')}"
    sync = s.get('sync', {})
    if not (sync.get('final') and sync.get('verified')):
        return 'final sync not verified'
    facts = remote_facts.get(run, {})
    if facts.get('final_sha') != sync.get('files', {}).get('final.json'):
        return 'remote final.json differs from verified sync'
    final = json.loads((ROOT / 'runs' / run / 'final.json').read_text())
    cfg = yaml.safe_load((ROOT / 'runs' / run / 'config.yaml').read_text())
    total = int(cfg['train']['num_train_steps']) if 'num_train_steps' in cfg.get('train', {}) else None
    if final.get('completed') is not True or total is None or final.get('steps') != total:
        return f'budget not verified (final {final.get("steps")}, configured {total})'
    every = int(cfg['eval']['every_steps'])
    ev = f'{run}-eval'
    if ev in queue:
        es = state.get(ev, {})
        if es.get('status') != 'done' or not (es.get('sync', {}).get('final') and es.get('sync', {}).get('verified')):
            return 'evaluator not closed out'
    rows = eval_rows(run)
    if not rows or rows[-1]['step'] != total or len(rows) != total // every:
        return 'evaluation history incomplete'
    if run.startswith('s2rl-') and not (ROOT / 'runs/setup' / f's2-{run}-completion.json').is_file():
        return 'Stage 2 closeout proof missing'
    if not facts.get('final_snapshot') or not facts.get('latest_checkpoint'):
        return 'final snapshot or last checkpoint missing remotely'
    for job in queue.values():
        if (run in job.get('deps', []) or ev in job.get('deps', [])) and state.get(job['id'], {}).get('status') != 'done':
            return f"pending dependent {job['id']}"
    return None


free_before, running, files = remote_state()
state = jobs.read_registry()
queue = {j['id']: j for j in jobs.load_queue()['jobs']}
runs = sorted({f['run'] for f in files})
facts = {}
probe = []
for run in runs:
    cfg_path = ROOT / 'runs' / run / 'config.yaml'
    total = None
    if cfg_path.is_file():
        cfg = yaml.safe_load(cfg_path.read_text())
        total = cfg.get('train', {}).get('num_train_steps')
    q = shlex.quote(run)
    probe.append(f"echo __RUN__ {q}; sha256sum {q}/final.json 2>/dev/null | cut -c1-64; "
                 f"test -f {q}/snapshots/step_{int(total or 0):07d}.pt && echo SNAP; "
                 f"test -f {q}/checkpoints/latest.pt && echo LATEST")
out = ssh(f"cd {REMOTE}/runs && " + ' ; '.join(probe)) if probe else ''
for block in out.split('__RUN__ ')[1:]:
    lines = block.splitlines()
    run = lines[0].strip()
    facts[run] = {'final_sha': next((ln for ln in lines[1:] if len(ln) == 64), None),
                  'final_snapshot': 'SNAP' in lines, 'latest_checkpoint': 'LATEST' in lines}
verdicts = {}
for run in runs:
    try:
        verdicts[run] = check(run, state, queue, running, facts)
    except (OSError, KeyError, ValueError, TypeError) as exc:
        verdicts[run] = f'check error {type(exc).__name__}'
candidates = [{**f, 'host': 'remote', 'category': 'replay', 'reason': 'replay buffer of completed, closed-out training',
               'evidence': {'registry': 'done, final sync verified', 'final_json_sha256': facts[f['run']]['final_sha'],
                            'final_snapshot_kept': True, 'latest_checkpoint_kept': True}}
              for f in files if verdicts[f['run']] is None]
retained = {run: v for run, v in verdicts.items() if v is not None}
manifest = {'recorded_local': datetime.now().astimezone().isoformat(), 'stage': 1, 'host': 'remote',
            'root': REMOTE, 'free_bytes_before': free_before, 'running_containers': sorted(running),
            'candidates': candidates, 'retained_runs': retained}
print(f'replay files {len(files)} in {len(runs)} runs; candidates {len(candidates)} '
      f'({sum(c["bytes"] for c in candidates) / 1e9:.1f} GB); retained runs {len(retained)}')
reasons = {}
for v in retained.values():
    key = v.split(' (')[0]
    reasons[key] = reasons.get(key, 0) + 1
print('retained by reason:', reasons)
if not EXECUTE:
    (OUT / 'remote-stage1-manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    raise SystemExit(0)

# recheck immediately before deletion
_, running_now, _ = remote_state()
state = jobs.read_registry()
todo = [c for c in candidates if c['run'] not in running_now and state.get(c['run'], {}).get('status') == 'done'
        and not state.get(c['run'], {}).get('unknown')]
skipped = [c['path'] for c in candidates if c not in todo]
listing = '\n'.join(c['path'] for c in todo)
digest = hashlib.sha256(listing.encode()).hexdigest()
if todo:
    script = (f"cd {REMOTE} && while IFS= read -r p; do case \"$p\" in runs/*/replay/*) ;; *) echo BAD \"$p\"; continue;; esac; "
              f"if [ -f \"$p\" ] && [ ! -L \"$p\" ]; then rm -f -- \"$p\" && echo DEL \"$p\"; fi; done <<'S4DLIST'\n"
              f"{listing}\nS4DLIST")
    out = ssh(script, timeout=1800)
    gone = {line[4:] for line in out.splitlines() if line.startswith('DEL ')}
else:
    gone = set()
free_after, _, _ = remote_state()
manifest.update(executed_local=datetime.now().astimezone().isoformat(), list_sha256=digest,
                deleted=sorted(gone), deleted_bytes=sum(c['bytes'] for c in todo if c['path'] in gone),
                skipped_at_recheck=skipped, free_bytes_after=free_after)
(OUT / 'remote-stage1-manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
print(f"deleted {len(gone)} files ({manifest['deleted_bytes'] / 1e9:.1f} GB allocated); remote free "
      f"{free_before / 1e9:.1f} -> {free_after / 1e9:.1f} GB")
