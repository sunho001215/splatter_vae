"""Camera suite v2 re-evaluation of completed policies without retraining (user directive 2026-10-29, item 3;
s4d/rl/camera_suites.py, scripts/eval_rl.py --suites v2; definitions logged in docs/EXPERIMENT_LOG.md before any
policy evaluation).

One evaluator row ds-cv2-<run>-eval per completed run: scripts/eval_rl.py --suites v2 --snapshots final, i.e. the final
(1M) policy of every run on vis0..3 (task exceptions applied), the stronger moving levels traj_{lateral,circular}_
{2x,3x}, and the matched references ref_train0..5 / ref_traj_{lateral,circular} re-run in the same pass on the same
paired episode seeds (20 episodes per group). Results go to runs/<row>/eval_cameras_v2.jsonl (never eval.jsonl).
Historical seeds and counts are kept: every completed run of a family is re-evaluated, with its own run seed.

Why the final snapshot only: 193 of the 216 completed DrM runs (2026-10-29 inventory) retain only their final
policy snapshot (earlier snapshots were not kept), so last-5 re-evaluation is impossible for most runs; every method is
re-evaluated identically on its final policy.

A row runs on the host that holds the run's final snapshot (no policy transfer; remote rows are not movable: their
partner id is not a row). csw4 and CNN first (priority 1), the other families after (priority 2).

    python -I register_cameras_v2_reeval_20261029.py [--families ...] [--algorithms drm sac] [--publish]
"""
import argparse
import fcntl
import hashlib
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path('/home/ws/ws/splatter4d')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/setup/tools'))
from scripts import jobs  # noqa: E402
import pool_placement as pool  # noqa: E402
from s4d.remote_access import RemoteClient  # noqa: E402

TASKS = ('hammer', 'bin-picking', 'door-open', 'peg-insert-side', 'peg-unplug-side', 'pick-place', 'stick-push',
         'coffee-push', 'drawer-open', 'window-open', 'soccer', 'sweep-into')
# family -> run id pattern per algorithm ({t} task, {s} seed); the CNN bin-picking s2000 DrM run of record is -r4
DRM = {'csw4': 'enc-csw4-drm-{t}-s{s}', 'cnn': 's4-drm-cnn-{t}-s{s}', 'sincro': 's4-drm-sincro-{t}-s{s}',
       'reviwo': 's4-drm-reviwo-{t}-s{s}', 'dynamo': 's4-drm-dynamo-{t}-s{s}', 'snerl': 's4-drm-snerl-{t}-s{s}',
       'mvp3d-multiview': 's4-drm-mvp3d-multiview-{t}-s{s}', 'mvp3d-single': 's4-drm-mvp3d-single-{t}-s{s}',
       'cnnnoshift': 's4-drm-cnnnoshift-{t}-s{s}'}
RERUN = {'s4-drm-cnn-bin-picking-s2000': 's4-drm-cnn-bin-picking-s2000-r4'}
SAC = {f: f'ds-sac-{f}-{{t}}-s{{s}}' for f in DRM if f != 'cnnnoshift'}
FIRST = ('csw4', 'cnn')
RES = {'mem_gb': 3, 'ram_gb': 8, 'disk_gb': 1}
STAGE = 7
DEFINITION = 'runs/setup/camera-suite-v2-definition-20261029.json'
STAMP = datetime.now().astimezone().isoformat(timespec='seconds')

ap = argparse.ArgumentParser()
ap.add_argument('--families', nargs='+', default=list(DRM))
ap.add_argument('--algorithms', nargs='+', default=['drm'], choices=('drm', 'sac'))
ap.add_argument('--seeds', nargs='+', type=int, default=[2000, 2001, 2002])
ap.add_argument('--publish', action='store_true')
args = ap.parse_args()
queue = jobs.load_queue()
in_queue = {j['id']: j for j in queue['jobs']}
state = jobs.read_registry()

runs = []  # (family, algorithm, run id, training row)
for algorithm in args.algorithms:
    patterns = DRM if algorithm == 'drm' else SAC
    for fam in sorted(args.families, key=lambda f: (f not in FIRST, list(DRM).index(f))):
        if fam not in patterns:
            continue
        for seed in args.seeds:
            for task in TASKS:
                rid = patterns[fam].format(t=task, s=seed)
                rid = RERUN.get(rid, rid)
                row = in_queue.get(rid)
                if row is None or state.get(rid, {}).get('status') != jobs.DONE:
                    continue
                runs.append((fam, algorithm, rid, row))

final_name = 'snapshots/step_{:07d}.pt'
remote_rows = [r for r in runs if r[3].get('host') == 'remote']
remote_has = set()
if remote_rows:
    client = RemoteClient()
    names = ' '.join(f"runs/{rid}/{final_name.format(int(_cfg_steps))}" for _, _, rid, row in remote_rows
                     for _cfg_steps in [1_000_000])
    probe = client.ssh(f'cd {client.root} && for f in {names}; do test -s "$f" && echo "$f"; done', idempotent=True,
                       timeout=300)
    remote_has = set(probe.stdout.split())

new_jobs, skipped = [], []
for fam, algorithm, rid, row in runs:
    host = row.get('host') or 'local'
    final = f'runs/{rid}/' + final_name.format(1_000_000)
    present = (ROOT / final).is_file() if host == 'local' else final in remote_has
    if not present:
        skipped.append((rid, host, 'final snapshot missing'))
        continue
    jid = f'ds-cv2-{rid}-eval'
    new_jobs.append({
        'host': host, 'gpu': 'any', 'deps': [], 'max_restarts': 1, 'id': jid, 'script': 'scripts/eval_rl.py',
        'args': ['--run-dir', f'runs/{rid}', '--suites', 'v2', '--snapshots', 'final', '--workers', '6',
                 '--envs-per-worker', '8', '--output-dir', f'runs/{jid}'],
        **RES, 'required_results': ['eval_cameras_v2.jsonl'], 'stage': STAGE,
        'variant': f'cameras_v2_{algorithm}_{fam}', 'variant_definition': DEFINITION,
        'placement': f'pool {STAMP} (re-evaluation on the host holding the final snapshot; not movable)',
        'priority': 1 if fam in FIRST else 2, 'reevaluates': rid})
ids = [j['id'] for j in new_jobs]
assert len(ids) == len(set(ids)) and not set(ids) & set(in_queue), 'duplicate or already registered'
assert not any((ROOT / 'runs' / i).exists() for i in ids), 'run dir exists'
assert not any(pool.movable(j, state, {**in_queue, **{k['id']: k for k in new_jobs}}) for j in new_jobs)
record = {'prepared_at': datetime.now().astimezone().isoformat(), 'purpose': __doc__.split('\n\n')[0],
          'definition': DEFINITION, 'algorithms': args.algorithms, 'families': args.families, 'seeds': args.seeds,
          'counts': {h: sum(j['host'] == h for j in new_jobs) for h in ('local', 'remote')},
          'skipped': skipped, 'jobs': new_jobs}
tag = '-'.join(args.algorithms)
proof = ROOT / f'runs/setup/cameras-v2-reeval-registration-{tag}-20261029.json'
if not args.publish:
    print(json.dumps(record['counts']), len(new_jobs), 'skipped', skipped[:5], proof.name)
    print(json.dumps(new_jobs[0]) if new_jobs else 'no rows')
    raise SystemExit(0)
with open(pool.ADMISSION_LOCK, 'w') as admission:
    fcntl.flock(admission, fcntl.LOCK_EX)
    with jobs.REGISTRY.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert not proof.exists()
        before = jobs.QUEUE.read_bytes()
        text = before.decode()
        node = next(value for key, value in yaml.compose(text).value if key.value == 'jobs')
        staged = proof.with_suffix('.queue-candidate.yaml')
        staged.write_text(text[:node.end_mark.index].rstrip() + '\n'
                          + yaml.safe_dump(json.loads(json.dumps(new_jobs)), sort_keys=False) + '\n'
                          + text[node.end_mark.index:])
        assert jobs.load_queue(staged)['jobs'] == jobs.load_queue()['jobs'] + new_jobs
        tmp = jobs.QUEUE.with_suffix('.cv2.tmp')
        tmp.write_bytes(staged.read_bytes())
        os.replace(tmp, jobs.QUEUE)
        record.update(registered_at=datetime.now().astimezone().isoformat(),
                      queue_sha256_before=hashlib.sha256(before).hexdigest(),
                      queue_sha256_after=hashlib.sha256(jobs.QUEUE.read_bytes()).hexdigest())
        proof.write_text(json.dumps(record, indent=2) + '\n')
        print('REGISTERED', len(new_jobs), proof)
