"""Four-arm local Meta-World campaign (arm4; user directive 2026-11-06 ~12:00 KST): registration of the whole eight-task
campaign, before any result of these arms. Definition: runs/setup/arm4-definition-20261106.json; arms and naming:
runs/setup/tools/arm4_common.py.

Rows (304): per (task, arm) chain, interleaved task-major (hammer A B C D, pick-place A B C D, ...): pretraining,
export, then three (DrM trainer, evaluator) pairs (seeds 2000-2002) = 32 + 32 + 96 + 96; then 48 render-analysis rows
ana-arm4-render-<model>-<task> (the four arms after their pretraining; orig and cst50 re-rendered under the same
pipeline, no dependencies). Every campaign row is a copy of the cst50 HAMMER row of the same kind AS IN THE QUEUE (the
local reference template: host local, local output roots, resources, max_restarts, stage, required results) that
differs ONLY in:
- names: cst50- -> <prefix>- and hammer -> <task> (run name, run / output dirs, export paths, train-job, dependencies);
- the pretraining configuration: the arm's config list and --set overrides (arm4_common.pretrain_stack; for A / B the
  Original row form base + task + the four method-frozen-v1 overrides, for C / D the cst50 row form);
- bookkeeping: variant, variant_definition; pin_host 'local' (already the template's).
Priorities are the template's (pretraining / DrM / evaluator 1, export 0); render rows 1 and registered after every
chain row, so they never take the head-of-line reservation ahead of a training row (Codex arm4-b #3); required result
done.json.
Self-checks (all asserted): resolved pretraining configs differ from the reference runs' STORED configs by exactly the
intended keys (and, for hammer / pick-place Original, train.stop_step); every DrM / evaluator row equals the cst50 one with
names substituted; the DrM rows' resolved RL configs equal the stored reference DrM configs with the export path
substituted (arm4-config-audit-20261106.json, re-checked here for the trainer args); no id or run directory exists.

    python -I register_arm4_eight_task_20261106.py [--publish | --recover]

Publication as register_cst_eight_task_20261104.py: under the local admission lock and the registry lock the queue is
replaced atomically (staged + os.replace), then the proof runs/setup/arm4-registration-eight-task-20261106.json is
written atomically (tmp + os.replace); --recover writes the proof for exactly these rows if a crash left them in the
queue without it. The local driver (ops/campaign/baseline_local_driver.py REGISTRATIONS) admits the rows only once its
release lists the proof file, so publication before the release is inert.
"""
import argparse
import copy
import fcntl
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path('/home/ws/ws/splatter4d')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/setup/tools'))
from scripts import jobs  # noqa: E402
import arm4_common as C  # noqa: E402
import pool_placement as pool  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--publish', action='store_true')
ap.add_argument('--recover', action='store_true')
args = ap.parse_args()
DEFINITION = 'runs/setup/arm4-definition-20261106.json'
PROOF = ROOT / 'runs/setup/arm4-registration-eight-task-20261106.json'
RENDER_SCRIPT = 'reports/results-artifact/src/run_arm4_render_20261106.py'
assert (ROOT / DEFINITION).is_file(), 'definition record missing'
definition = json.loads((ROOT / DEFINITION).read_text())
queue = jobs.load_queue()
by_queue = {j['id']: j for j in queue['jobs']}
T = 'hammer'
# evaluator layout of each arm = its reference's invocation (Codex arm4-a #9): Original 3 workers x 8 envs, cst50 6 x 8.
# Measured (runs/setup/arm4-evalcheck-20261106.json): repeated evaluations of one snapshot are not bit-reproducible even
# with the same layout (renderer / GPU nondeterminism), and 3 x 8 vs 6 x 8 differ by the same order; matching the
# reference's invocation removes the batching difference from the main contrasts at no cost.
EVAL_LAYOUT = {'original': ('3', '8'), 'cst50': ('6', '8')}
TEMPLATE = {'pretrain': by_queue[f'cst50-seed0-{T}-200k'], 'export': by_queue[f'cst50-export-seed0-{T}-200k'],
            'drm': [by_queue[f'cst50-drm-{T}-s{s}'] for s in C.SEEDS],
            'eval': [by_queue[f'cst50-drm-{T}-s{s}-eval'] for s in C.SEEDS]}
assert all(r.get('host') == 'local' and r.get('pin_host') == 'local' for r in
           [TEMPLATE['pretrain'], TEMPLATE['export'], *TEMPLATE['drm'], *TEMPLATE['eval']])


def renamed(row: dict, prefix: str, task: str) -> dict:
    text = json.dumps(row).replace('cst50-', f'{prefix}-').replace(f'-{T}-', f'-{task}-').replace(
        f'"{T}"', f'"{task}"').replace(f'/{T}.yaml', f'/{task}.yaml')
    return json.loads(text)


new_jobs, hashes = [], {}
for task in C.TASKS:
    for arm, spec in C.ARMS.items():
        assert definition['arms'][arm]['prefix'] == spec['prefix'], arm
        files, sets = C.pretrain_stack(arm, task)
        assert definition['arms'][arm]['overrides'] == [s for s in sets], (arm, definition['arms'][arm]['overrides'])
        ids = C.ids(arm, task)
        pre = renamed(TEMPLATE['pretrain'], spec['prefix'], task)
        a = pre['args']
        pre['args'] = ['--config', *files] + a[a.index('--name'):a.index('--set') + 1] + sets
        assert pre['id'] == ids['pretrain'] and a[a.index('--name') + 1] == ids['pretrain'], pre['id']
        rows = [pre, renamed(TEMPLATE['export'], spec['prefix'], task)]
        for d, e in zip(TEMPLATE['drm'], TEMPLATE['eval']):
            ev = renamed(e, spec['prefix'], task)
            ev['args'][ev['args'].index('--workers') + 1] = EVAL_LAYOUT[spec['base']][0]
            ev['args'][ev['args'].index('--envs-per-worker') + 1] = EVAL_LAYOUT[spec['base']][1]
            rows += [renamed(d, spec['prefix'], task), ev]
        assert [r['id'] for r in rows] == [ids['pretrain'], ids['export']] + \
            [x for pair in zip(ids['drm'], ids['eval']) for x in pair], [r['id'] for r in rows]
        for row in rows:
            row.update(variant=spec['variant'], variant_definition=DEFINITION)
            assert row['host'] == 'local' and row['pin_host'] == 'local'
            assert 'cst50' not in json.dumps(row) and (task == T or T not in json.dumps(row)), row['id']
            # everything but names and the pretraining configuration equals the template (key sets and values)
            template = {'scripts/train.py': TEMPLATE['pretrain'], 'scripts/export_encoder.py': TEMPLATE['export']}.get(
                row['script'])
            if template is None:
                template = (TEMPLATE['drm'] if row['script'] == 'scripts/train_rl.py' else TEMPLATE['eval'])[
                    C.SEEDS.index(int(row['id'].split('-s')[-1].split('-')[0]))]
            back = renamed(template, spec['prefix'], task)
            back.update(variant=spec['variant'], variant_definition=DEFINITION)
            diff = {k for k in set(back) | set(row) if back.get(k) != row.get(k)}
            allowed = {'args'} if row['script'] == 'scripts/train.py' or (
                row['script'] == 'scripts/eval_rl.py' and spec['base'] == 'original') else set()
            assert diff <= allowed, (row['id'], diff)
            if row['script'] == 'scripts/train.py':  # only the configuration part of the arguments differs
                r, b = row['args'], back['args']
                assert r[r.index('--name'):r.index('--set') + 1] == b[b.index('--name'):b.index('--set') + 1], row['id']
        # independent expected names / paths / seeds (Codex arm4-a #11, arm4-b #8): not derived from renamed()
        p, x = spec['prefix'], f'runs/{spec["prefix"]}-export-seed0-{task}-200k/encoder.pt'
        assert rows[0]['args'][rows[0]['args'].index('--output-root') + 1] == 'runs/pretrain'
        assert rows[1]['args'] == ['--run', f'runs/pretrain/{p}-seed0-{task}-200k', '--checkpoint',
                                   f'runs/pretrain/{p}-seed0-{task}-200k/checkpoints/step_0200000.pt', '--out', x]
        assert rows[1]['deps'] == [ids['pretrain']] and rows[0]['deps'] == []
        for seed, d, e in zip(C.SEEDS, rows[2::2], rows[3::2]):
            assert d['args'] == ['--task', task, '--encoder', 'splatter4d', '--seed', str(seed), '--run-dir',
                                 f'runs/{p}-drm-{task}-s{seed}', '--set', 'train.num_train_steps=1000000',
                                 f'vision.export_path={x}'], d['args']
            assert e['args'] == ['--run-dir', f'runs/{p}-drm-{task}-s{seed}', '--train-job', f'{p}-drm-{task}-s{seed}',
                                 '--workers', EVAL_LAYOUT[spec['base']][0], '--envs-per-worker',
                                 EVAL_LAYOUT[spec['base']][1], '--output-dir', f'runs/{p}-drm-{task}-s{seed}-eval'], e['args']
            assert d['deps'] == e['deps'] == [ids['export']]
        hashes[ids['pretrain']] = C.config_sha256(C.load(files, sets))
        new_jobs += rows
for task in C.TASKS:
    for model, arm in C.RENDER_MODELS.items():
        rid = C.render_row_id(model, task)
        deps = [C.ids(arm, task)['pretrain']] if arm in C.ARMS else []
        new_jobs.append({'id': rid, 'host': 'local', 'pin_host': 'local', 'script': RENDER_SCRIPT,
                         'args': ['--task', task, '--model', model, '--row', rid], 'gpu': 'any', 'mem_gb': 6, 'ram_gb': 12,
                         'disk_gb': 2, 'priority': 1, 'deps': deps, 'max_restarts': 1, 'stage': 5,
                         'required_results': ['done.json'], 'variant': 'arm4_analysis', 'variant_definition': DEFINITION})
ids = [j['id'] for j in new_jobs]
assert len(new_jobs) == 304 and len(set(ids)) == 304
assert all(j['id'].startswith(C.PREFIXES + ('ana-arm4-render-',)) for j in new_jobs)
for j in new_jobs:
    assert all(d in ids and ids.index(d) < ids.index(j['id']) for d in j['deps']), (j['id'], j['deps'])  # campaign rows, earlier

# ---- self-check: resolved pretraining configurations vs the reference runs' stored configurations ------------------
for task in C.TASKS:
    for arm, spec in C.ARMS.items():
        row = next(j for j in new_jobs if j['id'] == C.ids(arm, task)['pretrain'])
        a = row['args']
        cfg = C.load(a[a.index('--config') + 1:a.index('--name')], a[a.index('--set') + 1:])
        ref = yaml.safe_load((C.reference_pretrain_dir(spec['base'], task) / 'config.yaml').read_text())
        expected = dict(spec['diff'])
        if spec['base'] == 'original' and task in ('hammer', 'pick-place'):
            expected['train.stop_step'] = (200000, C.ABSENT)
        assert C.keydiff(ref, cfg) == expected, (arm, task, C.keydiff(ref, cfg))
        assert definition['config_sha256'][row['id']] == hashes[row['id']], row['id']
# ---- self-check: DrM rows resolve to the reference DrM configuration (export path substituted) --------------------
from s4d.rl.protocol import resolve_config  # noqa: E402

for task in C.TASKS:
    for arm, spec in C.ARMS.items():
        trainer = next(j for j in new_jobs if j['id'] == C.ids(arm, task)['drm'][0])
        a = trainer['args']
        sets = a[a.index('--set') + 1:]
        export = C.original_export(task) if spec['base'] == 'original' else C.cst50_export(task)
        sets = [s if not s.startswith('vision.export_path=') else f'vision.export_path={export}' for s in sets]
        cfg = resolve_config(task, 'splatter4d', int(a[a.index('--seed') + 1]), sets)
        ref = yaml.safe_load((ROOT / 'runs' / C.reference_drm(spec['base'], task, 2000) / 'config.yaml').read_text())
        assert not C.keydiff(ref, cfg, ignore=('run.', 'vision.export_path')), (arm, task)
assert args.recover or not set(ids) & set(by_queue), 'already registered'
assert args.recover or not any((ROOT / 'runs' / i).exists() or (ROOT / 'runs/pretrain' / i).exists() for i in ids), \
    'run dir exists'
counts = {}
for j in new_jobs:
    key = j['script'].split('/')[-1]
    counts[key] = counts.get(key, 0) + 1
priorities = {}
for j in new_jobs:
    priorities.setdefault(j['script'].split('/')[-1], set()).add(j['priority'])
record = {'prepared_at': datetime.now().astimezone().isoformat(), 'purpose': __doc__.split('\n\n')[0],
          'tasks': list(C.TASKS), 'arms': {a: s['prefix'] for a, s in C.ARMS.items()}, 'definition': DEFINITION,
          'definition_sha256': hashlib.sha256((ROOT / DEFINITION).read_bytes()).hexdigest(),
          'code_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in
                          (RENDER_SCRIPT, 'runs/setup/tools/arm4_common.py',
                           'runs/setup/tools/register_arm4_eight_task_20261106.py')},
          'config_sha256': hashes, 'priorities': {k: sorted(v) for k, v in priorities.items()}, 'counts': counts,
          'jobs': new_jobs}


def write_proof(rec):
    tmp = PROOF.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(rec, indent=2) + '\n')
    json.loads(tmp.read_text())
    os.replace(tmp, PROOF)


if args.recover:  # both locks; the rows are pinned local and never moved, so the tail must match EXACTLY (Codex arm4-b #4)
    with open(pool.ADMISSION_LOCK, 'w') as admission:
        fcntl.flock(admission, fcntl.LOCK_EX)
        with jobs.REGISTRY.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            assert not PROOF.exists(), 'proof exists: nothing to recover'
            snapshot = jobs.QUEUE.read_bytes()
            current = jobs.load_queue()['jobs']
            assert hashlib.sha256(jobs.QUEUE.read_bytes()).digest() == hashlib.sha256(snapshot).digest()
            tail = current[-len(new_jobs):]
            assert tail == new_jobs, 'queue tail is not exactly this registration'
            assert all(t.get('host') == t.get('pin_host') == 'local' for t in tail)
            record.update(registered_at='recovered ' + datetime.now().astimezone().isoformat(),
                          queue_sha256_after=hashlib.sha256(snapshot).hexdigest())
            write_proof({**record, 'jobs': tail})
            print('RECOVERED proof for', len(tail), 'rows')
    raise SystemExit(0)
if not args.publish:
    deps = {}
    for j in new_jobs:
        kind = j['script'].split('/')[-1]
        dkinds = sorted({next(x['script'].split('/')[-1] for x in new_jobs if x['id'] == d) for d in j['deps']})
        deps.setdefault(kind, set()).add(','.join(dkinds) or '-')
    print(json.dumps({'n': len(new_jobs), 'counts': counts, 'priorities': record['priorities'],
                      'dependency_kinds': {k: sorted(v) for k, v in deps.items()},
                      'hosts': sorted({(j['host'], j['pin_host']) for j in new_jobs}),
                      'first_ids': ids[:10], 'render_ids': ids[256:262]}, indent=1, default=str))
    for rid in (C.ids('A', 'hammer')['pretrain'], C.ids('B', 'stick-push')['pretrain'], C.ids('C', 'door-open')['drm'][1],
                C.ids('D', 'coffee-push')['eval'][2], C.render_row_id('orig', 'pick-place')):
        print(json.dumps(next(j for j in new_jobs if j['id'] == rid)))
    raise SystemExit(0)
with open(pool.ADMISSION_LOCK, 'w') as admission:
    fcntl.flock(admission, fcntl.LOCK_EX)
    with jobs.REGISTRY.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert not PROOF.exists()
        # preconditions re-checked inside the critical section (Codex arm4-d): no id collision, no run directory
        assert not set(ids) & {j['id'] for j in jobs.load_queue()['jobs']}, 'already registered'
        assert not set(ids) & set(jobs.read_registry()), 'ids already in the registry'
        assert not any((ROOT / 'runs' / i).exists() or (ROOT / 'runs/pretrain' / i).exists() for i in ids), 'run dir exists'
        before = jobs.QUEUE.read_bytes()
        text = before.decode()
        node = next(value for key, value in yaml.compose(text).value if key.value == 'jobs')
        staged = PROOF.with_suffix('.queue-candidate.yaml')
        staged.write_text(text[:node.end_mark.index].rstrip() + '\n'
                          + yaml.safe_dump(json.loads(json.dumps(new_jobs)), sort_keys=False) + '\n'
                          + text[node.end_mark.index:])
        assert jobs.load_queue(staged)['jobs'] == jobs.load_queue()['jobs'] + new_jobs
        assert jobs.load_queue(staged)['hosts'] == queue['hosts']
        tmp = jobs.QUEUE.with_suffix('.arm4.tmp')
        tmp.write_bytes(staged.read_bytes())
        os.replace(tmp, jobs.QUEUE)
        record.update(registered_at=datetime.now().astimezone().isoformat(),
                      queue_sha256_before=hashlib.sha256(before).hexdigest(),
                      queue_sha256_after=hashlib.sha256(jobs.QUEUE.read_bytes()).hexdigest())
        write_proof(record)
        staged.unlink()
print('REGISTERED', len(new_jobs), 'rows; proof', PROOF)
