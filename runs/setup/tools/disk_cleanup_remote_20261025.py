"""Remote cleanup (user request 2026-10-25) under the authorized root /home/compu/kaist/sunho, with the rules of
disk_cleanup_remote_replay_20261019.py (replay) and disk_cleanup_local_models_20261025.py (snapshots, checkpoints).

Categories (exact files, runs/<id>/<sub>/<file> only, no symlinks, realpath inside the root):
  replay           every file under runs/R/replay/ of a completed, closed-out RL run R;
  policy_snapshot  runs/R/snapshots/step_N.pt of such a run with N < the final step, N in the (synced) eval history and
                   the path not referenced in campaign records; the final snapshot and checkpoints/latest.pt are kept;
  checkpoint       runs/D/checkpoints/step_N.pt of a completed remote pretraining run D with N not the final step, not a
                   100k multiple, not referenced, and eval/step_N/summary.json synced locally; latest.pt is kept;
  tmp              entries of runtime/cache/tmp (the containers' TMPDIR) whose newest file is older than 24 h and older
                   than the start of the oldest running campaign container (so no running job created them).
An RL run qualifies only if: no running container carries its run id; registry status `done` on host remote with a
verified final sync; the remote final.json equals that sync; final.json completed with steps == configured total; its
evaluator (when registered) `done` with a verified final sync and the eval history ends at the total with one row per
interval; Stage 2 closeout proof when applicable; final snapshot and latest.pt exist remotely; no queued job that is
not done depends on it. A pretraining run qualifies only if: no running container; registry `done`, host remote,
verified final sync; synced completion.json completed with steps == max remote step == configured train.steps; no
queued job that is not done depends on it. Checkpoints, snapshots, results, logs, W&B, releases, verification evidence
and datasets are otherwise untouched. Every candidate is rechecked (running containers, registry) right before deletion.

    python -I disk_cleanup_remote_20261025.py            # manifest only
    python -I disk_cleanup_remote_20261025.py --execute
"""
import hashlib
import json
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path('/home/ws/ws/splatter4d')
sys.path.insert(0, str(ROOT))
from scripts import jobs  # noqa: E402

REMOTE = '/home/compu/kaist/sunho'
OUT = ROOT / 'runs/setup/disk-cleanup-20261025'
EXECUTE = '--execute' in sys.argv
REF = re.compile(r'runs/[A-Za-z0-9_.\-/{}:]+?\.(?:pt|npz)\b')
STEP = re.compile(r'^step_(\d{7})\.pt$')
TMP_MIN_AGE = 24 * 3600


def ssh(command, timeout=900):
    result = subprocess.run([sys.executable, '-I', str(ROOT / 's4d/remote_access.py'), 'ssh', '--idempotent', '--',
                             command], capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stderr[-500:])
    return result.stdout


def references():
    refs = set()
    sources = [ROOT / 'experiments/queue.yaml', ROOT / 'experiments/registry.jsonl']
    for base in (ROOT / 'runs/setup', ROOT / 'docs', ROOT / 'reports/results-artifact/src', ROOT / 'analysis',
                 ROOT / 'scripts', ROOT / 's4d'):
        sources += [p for p in base.rglob('*') if p.is_file() and p.suffix in ('.json', '.jsonl', '.yaml', '.yml', '.md',
                                                                             '.txt', '.log', '.py')
                    and p.stat().st_size < 100e6
                    # cleanup manifests list candidates; they are not references (pass 1 kept every snapshot this way)
                    and not any(part.startswith('disk-cleanup-') for part in p.relative_to(ROOT).parts)]
    for path in sources:
        try:
            text = path.read_text(errors='ignore')
        except OSError:
            continue
        refs.update(m.group(0).lstrip('./') for m in REF.finditer(text))
    templates = [re.compile('[^/]+'.join(re.escape(p) for p in re.split(r'\{[^}]*\}', r))) for r in refs if '{' in r]
    return refs, templates


REMOTE_STATE = (
    f"set -eu; cd {REMOTE}/runs; df -B1 {REMOTE} | tail -1; echo __RUNNING__; "
    "docker ps --filter label=s4d.managed=true --format '{{.Label \"s4d.run_id\"}}'; echo __OLDEST__; "
    "ids=$(docker ps -q --filter label=s4d.managed=true); "
    "if [ -n \"$ids\" ]; then docker inspect --format '{{.State.StartedAt}}' $ids | sort | head -1; fi; echo __FILES__; "
    "find . -mindepth 3 -maxdepth 3 \\( -path './*/replay/*' -o -path './*/snapshots/step_*.pt' "
    "-o -path './*/checkpoints/step_*.pt' \\) -type f -printf '%k %s %T@ %p\\n'; echo __TMP__; "
    f"cd {REMOTE}/runtime/cache/tmp; for e in * .[!.]*; do [ -e \"$e\" ] || continue; [ -L \"$e\" ] && continue; "
    "printf '%s %s %s\\n' \"$(find \"$e\" -printf '%T@\\n' | sort -n | tail -1)\" \"$(du -sk -- \"$e\" | cut -f1)\" \"$e\"; done")


def remote_state():
    out = ssh(REMOTE_STATE)
    head, rest = out.split('__RUNNING__\n', 1)
    running, rest = rest.split('__OLDEST__\n', 1)
    oldest, rest = rest.split('__FILES__\n', 1)
    listing, tmp = rest.split('__TMP__\n', 1)
    files = []
    for line in listing.splitlines():
        kb, size, mtime, path = line.split(' ', 3)
        rel = path[2:]
        files.append({'path': f'runs/{rel}', 'run': rel.split('/')[0], 'sub': rel.split('/')[1],
                      'name': rel.split('/')[2], 'bytes': int(kb) * 1024, 'apparent_bytes': int(size),
                      'mtime': float(mtime)})
    entries = []
    for line in tmp.splitlines():
        newest, kb, name = line.split(' ', 2)
        entries.append({'path': f'runtime/cache/tmp/{name}', 'newest_mtime': float(newest), 'bytes': int(kb) * 1024})
    oldest = oldest.strip()
    started = datetime.fromisoformat(oldest[:26].rstrip('Z') + '+00:00').timestamp() if oldest else time.time()
    return int(head.split()[3]), set(running.split()), files, entries, started


def eval_rows(run):
    for path in (ROOT / 'runs' / f'{run}-eval' / 'eval.jsonl', ROOT / 'runs' / run / 'eval.jsonl'):
        if path.is_file():
            return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return None


def verified(s):
    return bool(s.get('sync', {}).get('final') and s.get('sync', {}).get('verified'))


def dependents(run, queue, state):
    names = {run, f'{run}-eval'}
    return [j['id'] for j in queue.values() if names & set(j.get('deps', []))
            and state.get(j['id'], {}).get('status') != 'done']


def rl_check(run, state, queue, running, facts):
    s = state.get(run)
    if run in running or f'{run}-eval' in running:
        return 'container running'
    if s is None:
        return 'not in registry'
    if s['status'] != 'done' or s.get('unknown') or s.get('host') != 'remote':
        return f"registry status {s['status']} host {s.get('host')}"
    sync = s.get('sync', {})
    if not verified(s):
        return 'final sync not verified'
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
        if es.get('status') != 'done' or not verified(es):
            return 'evaluator not closed out'
    rows = eval_rows(run)
    if not rows or rows[-1]['step'] != total or len(rows) != total // every:
        return 'evaluation history incomplete'
    if run.startswith('s2rl-') and not (ROOT / 'runs/setup' / f's2-{run}-completion.json').is_file():
        return 'Stage 2 closeout proof missing'
    if not facts.get('final_snapshot') or not facts.get('latest_checkpoint'):
        return 'final snapshot or last checkpoint missing remotely'
    pending = dependents(run, queue, state)
    if pending:
        return f'pending dependent {pending[0]}'
    return None


def pretrain_check(run, steps, state, queue, running):
    s = state.get(run)
    if run in running:
        return 'container running', None
    if s is None:
        return 'not in registry', None
    if s['status'] != 'done' or s.get('unknown') or s.get('host') != 'remote':
        return f"registry status {s['status']} host {s.get('host')}", None
    if not verified(s):
        return 'final sync not verified', None
    comp, cfg = ROOT / 'runs' / run / 'completion.json', ROOT / 'runs' / run / 'config.yaml'
    if not comp.is_file() or not cfg.is_file():
        return 'no synced completion.json / config.yaml', None
    c = json.loads(comp.read_text())
    cfg_steps = yaml.safe_load(cfg.read_text()).get('train', {}).get('steps')
    if c.get('status') != 'completed' or c.get('steps') != max(steps) or cfg_steps != max(steps):
        return f"completion not verified ({c.get('steps')}, max {max(steps)}, config {cfg_steps})", None
    pending = dependents(run, queue, state)
    if pending:
        return f'pending dependent {pending[0]}', None
    return None, max(steps)


free_before, running, files, tmp_entries, oldest_start = remote_state()
state = jobs.read_registry()
queue = {j['id']: j for j in jobs.load_queue()['jobs']}
refs, templates = references()


def referenced(rel):
    return rel in refs or any(t.fullmatch(rel) for t in templates)


by_run = {}
for f in files:
    by_run.setdefault(f['run'], []).append(f)
rl_runs = sorted(r for r, fs in by_run.items() if any(f['sub'] in ('replay', 'snapshots') for f in fs))
pre_runs = sorted(r for r in by_run if r not in rl_runs)
facts = {}
probe = []
for run in rl_runs:
    cfg_path = ROOT / 'runs' / run / 'config.yaml'
    total = yaml.safe_load(cfg_path.read_text()).get('train', {}).get('num_train_steps') if cfg_path.is_file() else 0
    q = shlex.quote(run)
    probe.append(f"echo __RUN__ {q}; sha256sum {q}/final.json 2>/dev/null | cut -c1-64; "
                 f"test -f {q}/snapshots/step_{int(total or 0):07d}.pt && echo SNAP; "
                 f"test -f {q}/checkpoints/latest.pt && echo LATEST; true")
for i in range(0, len(probe), 100):
    out = ssh(f"cd {REMOTE}/runs && " + ' ; '.join(probe[i:i + 100]))
    for block in out.split('__RUN__ ')[1:]:
        lines = block.splitlines()
        facts[lines[0].strip()] = {'final_sha': next((ln for ln in lines[1:] if len(ln) == 64), None),
                                   'final_snapshot': 'SNAP' in lines, 'latest_checkpoint': 'LATEST' in lines}
candidates, kept, verdicts = [], {}, {}


def keep(reason, f):
    k = kept.setdefault(reason, {'files': 0, 'bytes': 0, 'examples': []})
    k['files'] += 1
    k['bytes'] += f['bytes']
    if len(k['examples']) < 4:
        k['examples'].append(f['path'])


for run in rl_runs:
    try:
        verdicts[run] = rl_check(run, state, queue, running, facts.get(run, {}))
    except (OSError, KeyError, ValueError, TypeError) as exc:
        verdicts[run] = f'check error {type(exc).__name__}'
    if verdicts[run]:
        for f in by_run[run]:
            keep(f"RL run not eligible: {verdicts[run].split(' (')[0]}", f)
        continue
    total = yaml.safe_load((ROOT / 'runs' / run / 'config.yaml').read_text())['train']['num_train_steps']
    steps = {r['step'] for r in eval_rows(run)}
    for f in by_run[run]:
        m = STEP.match(f['name'])
        if referenced(f['path']):
            keep('referenced in campaign records', f)
        elif f['sub'] == 'replay':
            candidates.append({**f, 'category': 'replay', 'reason': 'replay of completed, evaluated, closed-out training'})
        elif f['sub'] == 'snapshots' and m and int(m.group(1)) < int(total) and int(m.group(1)) in steps:
            candidates.append({**f, 'category': 'policy_snapshot',
                               'reason': f'intermediate snapshot evaluated at step {int(m.group(1))}; final {total} kept'})
        elif f['sub'] == 'snapshots' and m and int(m.group(1)) == int(total):
            keep('final RL policy snapshot', f)
        else:
            keep('other file of an eligible RL run', f)
for run in pre_runs:
    steps = [int(STEP.match(f['name']).group(1)) for f in by_run[run]]
    try:
        verdict, final = pretrain_check(run, steps, state, queue, running)
    except (OSError, KeyError, ValueError, TypeError) as exc:
        verdict, final = f'check error {type(exc).__name__}', None
    verdicts[run] = verdict
    for f in by_run[run]:
        n = int(STEP.match(f['name']).group(1))
        if verdict:
            keep(f"pretraining run not eligible: {verdict.split(' (')[0]}", f)
        elif n == final:
            keep('final pretrained model', f)
        elif n % 100000 == 0:
            keep('100k-multiple milestone (reported endpoints)', f)
        elif referenced(f['path']):
            keep('referenced in campaign records', f)
        elif not (ROOT / 'runs' / run / 'eval' / f'step_{n:07d}' / 'summary.json').is_file():
            keep('no synced evaluation for this step', f)
        else:
            candidates.append({**f, 'category': 'checkpoint',
                               'reason': f'intermediate pretraining checkpoint, evaluation synced; final {final} kept'})
now = time.time()
for e in tmp_entries:
    if now - e['newest_mtime'] >= TMP_MIN_AGE and e['newest_mtime'] < oldest_start - 3600:
        candidates.append({**e, 'run': None, 'category': 'tmp',
                           'reason': 'abandoned container TMPDIR entry older than every running campaign container'})
by_cat = {}
for c in candidates:
    by_cat.setdefault(c['category'], [0, 0])
    by_cat[c['category']][0] += 1
    by_cat[c['category']][1] += c['bytes']
manifest = {'recorded_local': datetime.now().astimezone().isoformat(), 'host': 'remote', 'root': REMOTE,
            'request': 'user request 2026-10-25 (cleanup on both hosts)', 'free_bytes_before': free_before,
            'running_containers': sorted(running), 'oldest_running_container_start': oldest_start,
            'candidates_by_category': {k: {'files': v[0], 'bytes': v[1]} for k, v in by_cat.items()},
            'kept_by_reason': kept, 'retained_runs': {r: v for r, v in verdicts.items() if v},
            'candidates': candidates}
for k, (n, b) in by_cat.items():
    print(f'candidate {k:16s} {n:6d} files {b / 1e9:8.1f} GB')
for k, v in sorted(kept.items(), key=lambda x: -x[1]['bytes']):
    print(f"kept {k:55s} {v['files']:6d} files {v['bytes'] / 1e9:8.1f} GB")
if not EXECUTE:
    (OUT / 'remote-manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    raise SystemExit(0)

# recheck immediately before deletion
_, running_now, _, _, oldest_now = remote_state()
state = jobs.read_registry()
todo, skipped = [], []
for c in candidates:
    s = state.get(c['run'], {}) if c['run'] else {}
    if c['category'] == 'tmp':
        ok = c['newest_mtime'] < oldest_now - 3600
    else:
        ok = (c['run'] not in running_now and f"{c['run']}-eval" not in running_now and s.get('status') == 'done'
              and not s.get('unknown'))
    (todo if ok else skipped).append(c)
results = {}
for category in ('replay', 'policy_snapshot', 'checkpoint', 'tmp'):
    batch = [c for c in todo if c['category'] == category]
    gone = set()
    # small batches (one SSH command argument must stay below 128 KiB; the wrapper times out after 120 s and retries,
    # losing the first attempt's output), paced so the shared host's sshd is not flooded (2026-10-25 16:40 outage);
    # the authoritative result is the existence check below, not the DEL lines
    size = 400 if category != 'replay' else 5
    for i in range(0, len(batch), size):
        time.sleep(1)
        listing = '\n'.join(c['path'] for c in batch[i:i + size])
        if category == 'tmp':
            body = ("case \"$p\" in runtime/cache/tmp/*/*|*..*) echo BAD \"$p\"; continue;; runtime/cache/tmp/?*) ;; "
                    "*) echo BAD \"$p\"; continue;; esac; "
                    "if [ -e \"$p\" ] && [ ! -L \"$p\" ]; then rm -rf -- \"$p\" && echo DEL \"$p\"; fi")
        else:
            body = ("case \"$p\" in *..*) echo BAD \"$p\"; continue;; runs/*/replay/*|runs/*/snapshots/step_*.pt|"
                    "runs/*/checkpoints/step_*.pt) ;; *) echo BAD \"$p\"; continue;; esac; "
                    f"r=$(realpath -e -- \"$p\" 2>/dev/null) || continue; case \"$r\" in {REMOTE}/runs/*) ;; *) "
                    "echo BAD \"$p\"; continue;; esac; "
                    "if [ -f \"$p\" ] && [ ! -L \"$p\" ]; then rm -f -- \"$p\" && echo DEL \"$p\"; fi")
        script = f"cd {REMOTE} && while IFS= read -r p; do {body}; done <<'S4DLIST'\n{listing}\nS4DLIST"
        out = ssh(script, timeout=3600)
        gone |= {line[4:] for line in out.splitlines() if line.startswith('DEL ')}
        bad = [line for line in out.splitlines() if line.startswith('BAD ')]
        assert not bad, bad[:3]
    present = set()
    for i in range(0, len(batch), 400):
        time.sleep(1)
        listing = '\n'.join(c['path'] for c in batch[i:i + 400])
        out = ssh(f"cd {REMOTE} && while IFS= read -r p; do if [ -e \"$p\" ] || [ -L \"$p\" ]; then echo PRESENT \"$p\"; "
                  f"fi; done <<'S4DLIST'\n{listing}\nS4DLIST\necho __END__")
        assert '__END__' in out
        present |= {line[8:] for line in out.splitlines() if line.startswith('PRESENT ')}
    gone = {c['path'] for c in batch} - present
    for c in batch:
        c['deleted'] = c['path'] in gone
    results[category] = {'files': len(gone), 'bytes': sum(c['bytes'] for c in batch if c['path'] in gone),
                         'not_deleted': [c['path'] for c in batch if c['path'] not in gone][:50]}
    print(f"{category}: deleted {len(gone)} of {len(batch)} ({results[category]['bytes'] / 1e9:.1f} GB allocated)")
free_after, _, _, _, _ = remote_state()
manifest.update(executed_local=datetime.now().astimezone().isoformat(), results=results,
                skipped_at_recheck=[c['path'] for c in skipped],
                list_sha256=hashlib.sha256('\n'.join(c['path'] for c in todo).encode()).hexdigest(),
                free_bytes_after=free_after)
(OUT / 'remote-manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
print(f"skipped at recheck {len(skipped)}; remote free {free_before / 1e9:.1f} -> {free_after / 1e9:.1f} GB")
