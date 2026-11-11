"""Local cleanup (user directive 2026-11-11 ~17:30): dated copy of disk_cleanup_local_models_20261025.py (rules of the
2026-10-19/10-25 reviewed tools, tightened as in the 2026-11-06 CLEAN pass and disk_cleanup_common_20261111.py).
Obsolete RL policy snapshots and intermediate pretraining checkpoints under /home/ws/ws/splatter4d/runs. Replay is NOT
handled here (scripts/cleanup_replay.py runs as a background loop and owns that scope). W&B files, datasets, source
code, results, evaluations, exports, logs and configurations are untouched.

RL run R (runs/R/snapshots/step_N.pt) qualifies only if: not protected (common.PROTECT_PREFIXES/SUBSTRINGS, named under
realworld-ops, active admission pause); registry status `done`, exit 0, not unknown, executed on this host (local copies of remote runs
are kept); final.json completed with steps == configured budget (queue override or config.yaml); the final snapshot
step_<budget>.pt exists; R-eval (when queued or registered) done (remote: verified sync); the eval history ends at the
budget; Stage 2 closeout proof when applicable; every other queue row that references runs/R (evaluators, camera
re-evaluations, continuations) or depends on R / R-eval is done. Then snapshots/step_N.pt is a candidate when
N < budget, N is not a multiple of 100,000 (milestones kept as on 2026-11-06), N is in the eval history and neither the path (exact / templated) nor the (run, N) pair is referenced.
Final snapshot and checkpoints/latest.pt are kept.
Pretraining run D (runs/D or runs/pretrain/D with checkpoints/step_N.pt, no snapshots/) qualifies only if not
protected; registry `done`, exit 0, not unknown (remote: verified sync); completion.json completed with steps == max
checkpoint step == config train.steps; no non-done queue row references or depends on D. Then checkpoints/step_N.pt is
a candidate when N is not the final step, not a multiple of 100,000, unreferenced (path, template, (run, N) pair) and
eval/step_N/summary.json exists. latest.pt, exports and non-step files are kept.
Every file is rechecked immediately before deletion (registry state unchanged, regular non-symlink file whose realpath
is itself inside runs/, same size and mtime as in the manifest, untouched >= 1 h, not open by any process, no live
process referencing the run directory, kept final file present).

    nice -n 19 ionice -c3 .venv/bin/python -I runs/setup/tools/disk_cleanup_local_models_20261111.py [--execute]
"""
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import disk_cleanup_common_20261111 as C  # noqa: E402

ROOT, RUNS = C.ROOT, C.RUNS
EXECUTE = '--execute' in sys.argv
STEP = re.compile(r'^step_(\d{7})\.pt$')
MIN_AGE = 3600
MANIFEST = C.OUT / 'local-models-manifest.json'


def alloc(path):
    return path.lstat().st_blocks * 512


def eval_steps(run, run_dir):
    for path in (RUNS / f'{run}-eval' / 'eval.jsonl', run_dir / 'eval.jsonl'):
        if path.is_file():
            return [json.loads(line)['step'] for line in path.read_text().splitlines() if line.strip()]
    return None


def synced(s):
    # tightened as on 2026-11-06: only runs that executed on this host; the local copies of remote runs' checkpoints
    # may be the only copies left (remote intermediates were deleted on 10-25 / 11-06), so they are not touched here
    return s.get('host', 'local') == 'local'


def common_check(run, state, rows, byid, pauses, rw_runs):
    why = C.protected_prefix(run)
    if why:
        return f'protected: {why}'
    if run in rw_runs:
        return f'protected: named in realworld-ops ({rw_runs[run]})'
    p = C.paused(run, pauses)
    if p:
        return f'admission pause active: {p[0]}'
    s = state.get(run)
    if s is None:
        return 'not in registry'
    if s['status'] != 'done' or s.get('unknown') or s.get('code') not in (0, None):
        return f"registry status {s['status']} code {s.get('code')}"
    if not synced(s):
        return f"executed on host {s.get('host')} (local copy kept; may be the only copy)"
    deps = [r['id'] for r in rows if (run in r.get('deps', []) or f'{run}-eval' in r.get('deps', []))
            and state.get(r['id'], {}).get('status') != 'done']
    if deps:
        return f'non-done dependent {deps[0]}'
    refs = [i for i in C.referencing_rows(run, rows) if state.get(i, {}).get('status') != 'done']
    if refs:
        return f'non-done referencing row {refs[0]}'
    return None


def rl_check(run, run_dir, state, rows, byid, pauses, rw_runs):
    why = common_check(run, state, rows, byid, pauses, rw_runs)
    if why:
        return why, None
    final = json.loads((run_dir / 'final.json').read_text())
    args = [str(a) for a in (byid.get(run) or {}).get('args', [])]
    queued = next((int(a.split('=', 1)[1]) for a in args if a.startswith('train.num_train_steps=')), None)
    configured = yaml.safe_load((run_dir / 'config.yaml').read_text())['train']['num_train_steps']
    total = queued if queued is not None else configured
    if queued is not None and configured != queued:
        return 'config budget differs from queue override', None
    if final.get('completed') is not True or final.get('steps') != total:
        return 'budget not verified', None
    if not (run_dir / 'snapshots' / f'step_{int(total):07d}.pt').is_file():
        return 'final snapshot missing', None
    ev = f'{run}-eval'
    if ev in byid or ev in state:
        es = state.get(ev, {})
        if es.get('status') != 'done' or not synced(es):
            return 'evaluator not done', None
    steps = eval_steps(run, run_dir)
    if not steps or max(steps) != total:
        return 'evaluation history incomplete', None
    if run.startswith('s2rl-') and not (ROOT / 'runs/setup' / f's2-{run}-completion.json').is_file():
        return 'closeout proof missing', None
    return None, (int(total), set(steps))


def pretrain_check(run_dir, state, rows, byid, pauses, rw_runs):
    run = run_dir.name
    why = common_check(run, state, rows, byid, pauses, rw_runs)
    if why:
        return why, None
    comp = run_dir / 'completion.json'
    if not comp.is_file():
        return 'no completion.json', None
    c = json.loads(comp.read_text())
    steps = sorted(int(m.group(1)) for f in os.listdir(run_dir / 'checkpoints') if (m := STEP.match(f)))
    cfg_steps = yaml.safe_load((run_dir / 'config.yaml').read_text()).get('train', {}).get('steps')
    if c.get('status') != 'completed' or c.get('steps') != steps[-1] or cfg_steps != steps[-1]:
        return 'completion not verified', None
    return None, steps[-1]


def plan():
    state = C.read_registry()
    rows = C.load_queue_rows()
    byid = {r['id']: r for r in rows}
    pauses = C.active_pauses()
    rw_runs = C.realworld_ops_runs()
    rl_dirs = sorted(p for p in RUNS.iterdir() if p.is_dir() and not p.is_symlink() and (p / 'snapshots').is_dir())
    pre_dirs = sorted(p for base in (RUNS, RUNS / 'pretrain') for p in base.iterdir()
                      if p.is_dir() and not p.is_symlink() and (p / 'checkpoints').is_dir() and not (p / 'snapshots').is_dir()
                      and any(STEP.match(f) for f in os.listdir(p / 'checkpoints')))
    verdicts, prelim = {}, {}
    for d in rl_dirs:
        try:
            verdicts[d], prelim[d] = rl_check(d.name, d, state, rows, byid, pauses, rw_runs) if (d / 'final.json').is_file() \
                else ('no final.json', None)
        except (OSError, KeyError, ValueError, TypeError) as exc:
            verdicts[d], prelim[d] = f'check error {type(exc).__name__}', None
    for d in pre_dirs:
        try:
            verdicts[d], prelim[d] = pretrain_check(d, state, rows, byid, pauses, rw_runs)
        except (OSError, KeyError, ValueError, TypeError, IndexError) as exc:
            verdicts[d], prelim[d] = f'check error {type(exc).__name__}', None
    eligible = {d.name for d, v in verdicts.items() if v is None}
    refs = C.reference_scan(eligible)
    cands, kept = [], {}

    def keep(reason, path):
        k = kept.setdefault(reason, {'files': 0, 'bytes': 0, 'examples': []})
        k['files'] += 1
        k['bytes'] += alloc(path)
        if len(k['examples']) < 4:
            k['examples'].append(str(path))

    now = time.time()
    for d in rl_dirs:
        run, verdict = d.name, verdicts[d]
        files = [d / 'snapshots' / f for f in os.listdir(d / 'snapshots')]
        if verdict:
            for f in files:
                keep(f'RL run not eligible: {verdict.split(" (")[0][:90]}', f)
            continue
        total, steps = prelim[d]
        for f in files:
            m = STEP.match(f.name)
            rel = str(f.relative_to(ROOT))
            n = int(m.group(1)) if m else None
            if f.is_symlink() or not f.is_file():
                keep('not a regular file', f)
            elif not m:
                keep('snapshot with non-step name', f)
            elif n >= total:
                keep('final RL policy snapshot', f)
            elif n % 100000 == 0:
                keep('100k-multiple milestone snapshot (kept as on 2026-11-06)', f)
            elif n not in steps:
                keep('snapshot step not in eval history', f)
            elif (r := C.referenced(rel, run, n, refs)):
                keep(f'referenced in campaign records ({r})', f)
            elif now - f.lstat().st_mtime < MIN_AGE:
                keep('modified within 1 h', f)
            else:
                st = f.lstat()
                cands.append({'path': str(f), 'bytes': alloc(f), 'owner_uid': st.st_uid, 'category': 'rl_intermediate_snapshot',
                              'run': run, 'reason': f'intermediate RL policy snapshot of a completed, evaluated run '
                                                    f'(step {n} evaluated; final step_{total:07d}.pt and checkpoints/latest.pt kept)',
                              'evidence': {'registry': state[run]['status'], 'code': state[run].get('code'),
                                           'host': state[run].get('host', 'local'), 'budget': total,
                                           'evaluator': state.get(f'{run}-eval', {}).get('status'),
                                           'eval_history_max': max(steps), 'step_in_eval_history': True,
                                           'referenced': False, 'pauses': [], 'size': st.st_size, 'mtime': st.st_mtime},
                              'kept_final': str(d / 'snapshots' / f'step_{total:07d}.pt')})
    for d in pre_dirs:
        run, verdict = d.name, verdicts[d]
        files = [d / 'checkpoints' / f for f in os.listdir(d / 'checkpoints')]
        for f in files:
            m = STEP.match(f.name)
            rel = str(f.relative_to(ROOT))
            n = int(m.group(1)) if m else None
            if verdict:
                keep(f'pretraining run not eligible: {verdict.split(" (")[0][:90]}', f)
            elif f.is_symlink() or not f.is_file():
                keep('not a regular file', f)
            elif not m:
                keep('pretraining resume state / non-step file', f)
            elif n == prelim[d]:
                keep('final pretrained model', f)
            elif n % 100000 == 0:
                keep('100k-multiple milestone (reported endpoints)', f)
            elif (r := C.referenced(rel, run, n, refs)):
                keep(f'referenced in campaign records ({r})', f)
            elif not (d / 'eval' / f'step_{n:07d}' / 'summary.json').is_file():
                keep('no recorded evaluation for this step', f)
            elif now - f.lstat().st_mtime < MIN_AGE:
                keep('modified within 1 h', f)
            else:
                st = f.lstat()
                cands.append({'path': str(f), 'bytes': alloc(f), 'owner_uid': st.st_uid,
                              'category': 'pretrain_intermediate_checkpoint', 'run': run,
                              'reason': f'intermediate pretraining checkpoint of a completed run (step {n}: not final, '
                                        f'not a 100k multiple, evaluation recorded; final step_{prelim[d]:07d}.pt kept)',
                              'evidence': {'registry': state[run]['status'], 'code': state[run].get('code'),
                                           'host': state[run].get('host', 'local'), 'completion_steps': prelim[d],
                                           'eval_summary': True, 'referenced': False, 'size': st.st_size,
                                           'mtime': st.st_mtime},
                              'kept_final': str(d / 'checkpoints' / f'step_{prelim[d]:07d}.pt')})
    by_cat = {}
    for c in cands:
        by_cat.setdefault(c['category'], {'files': 0, 'bytes': 0, 'runs': set()})
        by_cat[c['category']]['files'] += 1
        by_cat[c['category']]['bytes'] += c['bytes']
        by_cat[c['category']]['runs'].add(c['run'])
    return {'recorded_local': C.now_iso(), 'host': 'local', 'tool': __file__,
            'request': 'user directive 2026-11-11 ~17:30 (disk cleanup, both hosts)',
            'free_bytes_before': shutil.disk_usage('/home/ws').free,
            'reference_scan': {'files_scanned': refs['files_scanned'], 'paths': len(refs['paths']),
                               'pairs': len(refs['pairs']), 'templates': sorted(t.pattern for t in refs['templates'])},
            'candidates_by_category': {k: {'files': v['files'], 'bytes': v['bytes'], 'runs': sorted(v['runs'])}
                                       for k, v in by_cat.items()},
            'kept_by_reason': kept, 'runs_not_eligible': {str(d.relative_to(ROOT)): v for d, v in verdicts.items() if v},
            'candidates': cands}


def execute(manifest):
    state = C.read_registry()
    rows = C.load_queue_rows()
    byid = {r['id']: r for r in rows}
    pauses = C.active_pauses()
    rw_runs = C.realworld_ops_runs()
    # the reviewed manifest intersected with a fresh plan incl. a fresh reference scan (Codex review 2026-11-11 finding 5)
    fresh = {c['path'] for c in plan()['candidates']} if manifest['candidates'] else set()
    by_run = {}
    for c in manifest['candidates']:
        if c['path'] not in fresh:
            c['skipped'] = 'not selected by the fresh plan'
            continue
        by_run.setdefault(c['run'], []).append(c)
    results, t0 = {}, time.time()
    log = (C.OUT / 'executed-actions.jsonl').open('a')
    for run, items in sorted(by_run.items()):
        if time.time() - t0 > 120:  # registry / queue re-read every 2 min
            state, rows, pauses = C.read_registry(), C.load_queue_rows(), C.active_pauses()
            byid = {r['id']: r for r in rows}
            t0 = time.time()
        run_dir = Path(items[0]['path']).parent.parent
        why = common_check(run, state, rows, byid, pauses, rw_runs)
        if not why and items[0]['category'] == 'rl_intermediate_snapshot':
            ev = state.get(f'{run}-eval')
            if (f'{run}-eval' in byid or ev) and (ev or {}).get('status') != 'done':
                why = 'evaluator no longer done'
        if not why and not Path(items[0]['kept_final']).is_file():
            why = 'kept final file missing'
        if not why:
            cmd, held, cwds = C.proc_scan()
            absdir = str(run_dir)
            pat = re.compile(r'runs/(?:pretrain/)?' + re.escape(run) + r'(?![A-Za-z0-9_.-])')
            users = [p for p, c in cmd.items() if pat.search(c) or absdir in c]
            holders = [p for t, ps in held.items() if t.startswith(absdir + '/') for p in ps]
            inside = [c for c in cwds if c == absdir or c.startswith(absdir + '/')]
            if users or holders or inside:
                why = f'live process references/holds the run dir: {sorted(set(users + holders))[:5]} {inside[:2]}'
        for it in items:
            cat = results.setdefault(it['category'], {'files': 0, 'bytes': 0, 'skipped': []})
            p = Path(it['path'])
            reason = why
            if not reason:
                try:
                    st = p.lstat()
                except OSError:
                    reason = 'gone'
                else:
                    if (p.is_symlink() or not p.is_file() or os.path.realpath(p) != str(p)
                            or not str(p).startswith(str(RUNS) + '/') or not STEP.match(p.name)
                            or p.parent.name not in ('snapshots', 'checkpoints') or p.parent.parent != run_dir):
                        reason = 'not the expected regular file'
                    elif st.st_size != it['evidence']['size'] or st.st_mtime != it['evidence']['mtime']:
                        reason = 'size or mtime changed since the manifest'
                    elif time.time() - st.st_mtime < MIN_AGE:
                        reason = 'modified within 1 h'
            if reason:
                cat['skipped'].append({'path': it['path'], 'reason': reason})
                log.write(json.dumps({'time': time.time(), 'host': 'local', 'action': 'skip', 'category': it['category'],
                                      'path': it['path'], 'reason': reason}) + '\n')
                continue
            p.unlink()
            it['deleted'] = True
            cat['files'] += 1
            cat['bytes'] += it['bytes']
            log.write(json.dumps({'time': time.time(), 'host': 'local', 'action': 'delete', 'category': it['category'],
                                  'path': it['path'], 'allocated_bytes': it['bytes'], 'owner_uid': it['owner_uid'],
                                  'reason': it['reason'], 'evidence': {**it['evidence'],
                                  'recheck': 'registry done, no pause, referencing rows done, kept final present, '
                                             'same size/mtime, no live process'}}) + '\n')
        log.flush()
    log.close()
    return results


if __name__ == '__main__':
    C.OUT.mkdir(parents=True, exist_ok=True)
    if not EXECUTE:
        m = plan()
        for k, v in m['candidates_by_category'].items():
            print(f"candidate {k:34s} {v['files']:6d} files {v['bytes'] / 1e9:8.2f} GB {len(v['runs'])} runs")
        for k, v in sorted(m['kept_by_reason'].items(), key=lambda x: -x[1]['bytes'])[:30]:
            print(f"kept {k[:80]:80s} {v['files']:6d} files {v['bytes'] / 1e9:8.2f} GB")
        MANIFEST.write_text(json.dumps(m, indent=1) + '\n')
        raise SystemExit(0)
    m = json.loads(MANIFEST.read_text())
    free0 = shutil.disk_usage('/home/ws').free
    res = execute(m)
    m.update(executed_local=C.now_iso(), results=res, free_bytes_before_execute=free0,
             free_bytes_after=shutil.disk_usage('/home/ws').free)
    MANIFEST.write_text(json.dumps(m, indent=1) + '\n')
    for k, v in res.items():
        print(f"{k}: deleted {v['files']} files {v['bytes'] / 1e9:.2f} GB; skipped {len(v['skipped'])}")
    print(f"free {free0 / 1e9:.1f} -> {m['free_bytes_after'] / 1e9:.1f} GB")
