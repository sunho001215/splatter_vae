"""One pass of the local Stage 2 driver: verify finished pretraining/evaluations/exports/RL, prepare evaluation configs,
admit eligible jobs one by one under the local full-charge cap through scripts/jobs.py. Never deletes anything.

    python -I stage2_local_driver.py [--dry-run]
"""
import copy
import fcntl
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path('/home/ws/ws/splatter4d')
sys.path.insert(0, str(ROOT))
from scripts import jobs  # noqa: E402

sys.path.insert(0, str(ROOT / 'runs/setup/tools'))
import memory_budget as mb  # noqa: E402

# Local budget (user instruction 2026-10-21): cgroup working set max(0, memory.current - inactive_file) vs 150 GB,
# minus the transient-overhead margin (runs/setup/tools/memory_budget.py)
CAP = mb.CEILING - mb.TRANSIENT_OVERHEAD
TRAINER_PSS = 10.77e9  # measured trainer peak PSS (user rule)
# Measured splatter4d DrM peaks (2026-10-14): trainer RSS <= 2.65 GB at 1M (runs/s3-drm-splatter4d-*/final.json),
# local session PSS 2.80 GB at 667k; evaluator session PSS 3.5-3.8 GB. ~25% margin. Other jobs: declared ram_gb.
# Full-split evaluation (2026-10-15): session PSS peak 8.00 GB, sampled every 5 s over the whole run of
# fulleval-s2ab-singlegroup-seed0-pick-place-200k; reserve +25 %, as for RL (declared 12 GiB until then).
MEASURED = {'scripts/train.py': TRAINER_PSS, 'scripts/train_rl.py': 3.5e9, 'scripts/eval_rl.py': 4.5e9,
            'scripts/evaluate.py': 10.0e9}
DATA = Path('/home/ws/data/metaworld/splatter4d_v1')
DATA_FILES = [DATA / f'{t}.hdf5' for t in ('hammer', 'pick-place')] + \
    [DATA / f'heldout_sets/{t}.hdf5' for t in ('hammer', 'pick-place')]
WINDOWS = {'hammer': {'2': 2058, '6': 1978}, 'pick-place': {'2': 1996, '6': 1916}}
RL_STEPS = 1_000_000
DRY = '--dry-run' in sys.argv
stamp = datetime.now().astimezone()
tag = stamp.strftime('%Y%m%d-%H%M%S')
SETUP = ROOT / 'runs/setup'
_lock = open(SETUP / 'stage2-driver.lock', 'w')
try:
    fcntl.flock(_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    print('STAGE2_DRIVER another pass is running; skipped')
    raise SystemExit(0)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def proof_of(job_id):
    return SETUP / f's2-{job_id}-completion.json'


def write_proof(job_id, record):
    if not DRY:
        with proof_of(job_id).open('x') as stream:
            json.dump({'job_id': job_id, 'verified_local': stamp.isoformat(), **record}, stream, indent=2)
    messages.append(f'VERIFIED {job_id}')


registration = json.loads((SETUP / 'stage2-registration-20261013.json').read_text())
original = {j['id'] for j in registration['jobs']}  # the 108 Stage 2 jobs registered 2026-10-13
# Later Stage 2 registrations with the same job structure and closeouts: the crop variants (user approval
# 2026-10-15), admitted at the "middle" rank (see admissible/rank below).
# ... and the RL arm of the completed no-synth pretrains (registered 2026-10-16 before any of its RL results).
# ... and the 2026-10-17 follow-ups (crop on six more tasks, pretraining shift, their missing matched controls;
# registered before any result; remote placement inside the registration itself).
REUSED = {}
for name in ['stage2-crop-registration-20261015.json', 'stage2-nosynth-rl-registration-20261016.json',
             'stage2-extra-rl-registration-20261016.json', 'followup-registration-20261017.json']:
    if (SETUP / name).is_file():
        later = json.loads((SETUP / name).read_text())
        assert not set(later['configs']) & set(registration['configs'])
        assert not {j['id'] for j in later['jobs']} & original
        later_jobs = [j for j in later['jobs'] if j.get('stage') == 2]  # the follow-up's CNN rows are remote-driver work
        registration = {**registration, 'jobs': registration['jobs'] + later_jobs,
                        'configs': {**registration['configs'], **later['configs']}}
        REUSED.update(later.get('reused_pretraining', {}))
queue = jobs.load_queue()
ids = [j['id'] for j in registration['jobs']]
stage2 = [j for j in queue['jobs'] if j['id'] in set(ids)]
# Host-neutral placement is an execution overlay, not a rewrite of scientific registrations.
placement_path = SETUP / 'stage2-placement-20261015.json'
placement = json.loads(placement_path.read_text()) if placement_path.exists() else {'effective_jobs': {}}
# 2026-10-18 host-neutral overlay for unstarted follow-up RL pairs (followup-placement-20261018.json)
FU_PLACEMENT = SETUP / 'followup-placement-20261018.json'
fu_effective = json.loads(FU_PLACEMENT.read_text())['effective_jobs'] if FU_PLACEMENT.exists() else {}
expected = [fu_effective.get(j['id'], placement['effective_jobs'].get(j['id'], j)) for j in registration['jobs']]
assert stage2 == expected, 'registered Stage 2 jobs or their explicit placement overlay changed in the queue'
assert all(placement.get('original_jobs', {}).get(j['id'], j) == j for j in registration['jobs'])
# Report-only evaluations registered later (crop follow-up step 1, user 2026-10-15): same admission path and cap,
# own closeout below; they never touch the Stage 2 registration or its proofs.
EXTRA_REGISTRATIONS = ['crop-reeval-registration-20261015.json']
extra = []
for name in EXTRA_REGISTRATIONS:
    if (SETUP / name).is_file():
        registered = json.loads((SETUP / name).read_text())
        wanted = {j['id'] for j in registered['jobs']}
        found = [j for j in queue['jobs'] if j['id'] in wanted]
        assert found == registered['jobs'], f'{name}: registered jobs changed in the queue'
        extra += [{**j, '_registration': registered} for j in found]
order = ids + [j['id'] for j in extra]
by_id = {j['id']: j for j in stage2 + extra}
state = jobs.read_registry()
messages = []


def status(job_id):
    current = state.get(job_id, {})
    if (current.get('status') == 'crashed'
            and current['failures'] <= int(by_id.get(job_id, {}).get('max_restarts', 1))):
        return jobs.PENDING  # restartable (e.g. checkpoint-paused), exactly as scripts/jobs.py tick treats it
    return current.get('status', jobs.PENDING)


def task_of(job):
    args = job['args']
    if '--task' in args:
        return args[args.index('--task') + 1]
    config = next((a for a in args if a.startswith('configs/metaworld/tasks/')), None)
    if config:
        return config.rsplit('/', 1)[1].removesuffix('.yaml')
    if job['deps'] and job['deps'][0] in by_id:
        return task_of(by_id[job['deps'][0]])
    if job['deps'] and job['deps'][0] in REUSED:
        return REUSED[job['deps'][0]]['config']['data']['task']
    return 'pick-place' if 'pick-place' in job['id'] else 'hammer'


def pretrain_run(job_id):
    if job_id in REUSED:
        return ROOT / 'runs' / job_id
    return ROOT / 'runs' / job_id if by_id[job_id].get('host') == 'remote' else ROOT / 'runs/pretrain' / job_id


# 0. reused pretraining runs (the full method's remote final encoders as follow-up controls): proof once the synced
# 200k checkpoint still has the registered checksum and the run's config is the frozen task configuration
for jid, reused in REUSED.items():
    if proof_of(jid).exists() or state.get(jid, {}).get('status') != jobs.DONE:
        continue
    run = pretrain_run(jid)
    cfg = yaml.safe_load((run / 'config.yaml').read_text())
    cfg.pop('run')
    assert cfg == reused['config'], jid
    assert sha(ROOT / reused['checkpoint']) == reused['sha256'], jid
    write_proof(jid, {'reused_pretraining': True, 'config_equals_registration': True,
                      'checkpoint_sha256': reused['sha256']})


# 1. pretraining closeouts, then evaluation configs (copied only from a verified 200k pretraining run)
for job in stage2:
    jid = job['id']
    if job['script'] != 'scripts/train.py' or status(jid) != jobs.DONE or proof_of(jid).exists():
        continue
    run = pretrain_run(jid)
    completion = json.loads((run / 'completion.json').read_text())
    assert completion == {'status': 'completed', 'steps': 200000}, (jid, completion)
    cfg = yaml.safe_load((run / 'config.yaml').read_text())
    cfg.pop('run')
    assert cfg == registration['configs'][jid], jid
    write_proof(jid, {'completion': completion, 'config_equals_registration': True,
                      'checkpoint_sha256': sha(run / 'checkpoints/step_0200000.pt')})
for job in stage2:
    if job['script'] != 'scripts/evaluate.py':
        continue
    trainer, target = job['deps'][0], ROOT / 'runs' / job['id']
    if proof_of(trainer).exists() and not (target / 'config.yaml').exists() and not DRY:
        subprocess.run([sys.executable, '-I', str(ROOT / 'analysis/eval_config.py'),
                        str(pretrain_run(trainer)), str(target)], check=True)
        messages.append(f"EVAL_CONFIG {job['id']}")

# 2. full-split evaluations, exports and RL pairs
for job in stage2:
    jid = job['id']
    if status(jid) != jobs.DONE or proof_of(jid).exists():
        continue
    if job['script'] == 'scripts/evaluate.py':
        run = ROOT / 'runs' / jid
        summary_path = run / 'eval/step_0200000/summary.json'
        summary = json.loads(summary_path.read_text())
        task = task_of(job)
        if task in WINDOWS:
            windows = WINDOWS[task]
            assert summary['step'] == 200000 and summary['num_batches'] == {k: math.ceil(v / 16) for k, v in windows.items()}
        else:  # follow-up tasks: full split (both strides, non-empty); equality across arms is checked at reporting
            assert summary['step'] == 200000 and set(summary['num_batches']) == {'2', '6'}
            assert all(int(n) > 0 for n in summary['num_batches'].values())
        text = (run / 'console.log').read_text()
        assert 'Traceback' not in text[text.rfind('=== attempt'):]
        console, _ = json.JSONDecoder().raw_decode(text[text.rfind('\n{\n') + 1:])
        rest = {k: v for k, v in summary.items() if k not in {'step', 'num_batches'}}
        assert json.dumps(console, sort_keys=True) == json.dumps(rest, sort_keys=True), jid
        checkpoint = ROOT / job['args'][job['args'].index('--checkpoint') + 1]
        assert sha(checkpoint) == json.loads(proof_of(job['deps'][0]).read_text())['checkpoint_sha256']
        write_proof(jid, {'summary_sha256': sha(summary_path), 'num_batches': summary['num_batches'],
                          'console_equals_summary': True, 'checkpoint_sha256': sha(checkpoint),
                          'variant_comparison_performed': False})
    elif job['script'] == 'scripts/export_encoder.py':
        import torch

        source = job['deps'][0]
        out = ROOT / job['args'][job['args'].index('--out') + 1]
        payload = torch.load(out, map_location='cpu', weights_only=True)
        cfg = registration['configs'][source]
        assert payload['format'] == 'splatter4d-encoder-v1' and payload['step'] == 200000
        assert payload['run_name'] == source and payload['frame_strides'] == cfg['data']['strides']
        assert payload['source_checkpoint'] == job['args'][job['args'].index('--checkpoint') + 1]
        assert all(payload['encoder_config'][k] == v for k, v in cfg['model']['encoder'].items())
        checkpoint = json.loads(proof_of(source).read_text())['checkpoint_sha256']
        assert sha(ROOT / payload['source_checkpoint']) == checkpoint
        write_proof(jid, {'encoder_sha256': sha(out), 'step': payload['step'], 'source_job': source,
                          'source_checkpoint_sha256': checkpoint})
    elif job['script'] == 'scripts/train_rl.py':
        if status(jid + '-eval') != jobs.DONE:
            continue
        run = ROOT / 'runs' / jid
        final = json.loads((run / 'final.json').read_text())
        eval_path = ROOT / 'runs' / (jid + '-eval') / 'eval.jsonl'  # --output-dir companions (remote and follow-ups)
        if not eval_path.is_file():
            eval_path = run / 'eval.jsonl'
        evals = [json.loads(line) for line in eval_path.read_text().splitlines() if line.strip()]
        steps = [r['step'] for r in evals]
        assert final['completed'] is True and final['steps'] == RL_STEPS, (jid, final)
        assert steps == list(range(10000, RL_STEPS + 1, 10000)), (jid, steps[-3:])
        free = subprocess.run(['df', '-h', '--output=avail', '/home/ws'], capture_output=True, text=True,
                              check=True).stdout.split()[-1]
        if job.get('host') == 'remote':
            from s4d.remote_access import RemoteClient

            remote_free = RemoteClient().checked_ssh(
                'df -B1 --output=avail /home/compu/kaist/sunho | tail -1', wait=False,
                idempotent=True, timeout=600).stdout.strip()
            free = f'remote {int(remote_free) / 1e12:.2f}T'
        write_proof(jid, {'final_steps': final['steps'], 'eval_points': len(steps),
                          'last5_train_success': sum(r['train_cameras_success'] for r in evals[-5:]) / 5,
                          'eval_jsonl_sha256': sha(eval_path), 'replay': 'retained',
                          'free_home_ws': free, 'progress_recorded': False})

# 2b. report-only re-evaluations: summary at the checkpoint's own step, full batch counts, console equals summary,
# source checkpoint unchanged since registration (it is only read)
for job in extra:
    jid = job['id']
    if job['script'] != 'scripts/evaluate.py' or status(jid) != jobs.DONE or proof_of(jid).exists():
        continue
    source = job['_registration']['checkpoints_read_only'][jid]
    step = int(Path(source['checkpoint']).stem.removeprefix('step_'))
    run = ROOT / 'runs' / jid
    summary_path = run / f'eval/step_{step:07d}/summary.json'
    summary = json.loads(summary_path.read_text())
    windows = WINDOWS[task_of(job)]
    assert summary['step'] == step and summary['num_batches'] == {k: math.ceil(v / 16) for k, v in windows.items()}
    text = (run / 'console.log').read_text()
    assert 'Traceback' not in text[text.rfind('=== attempt'):]
    console, _ = json.JSONDecoder().raw_decode(text[text.rfind('\n{\n') + 1:])
    rest = {k: v for k, v in summary.items() if k not in {'step', 'num_batches'}}
    assert json.dumps(console, sort_keys=True) == json.dumps(rest, sort_keys=True), jid
    assert sha(ROOT / source['checkpoint']) == source['sha256'], jid
    write_proof(jid, {'summary_sha256': sha(summary_path), 'num_batches': summary['num_batches'], 'step': step,
                      'console_equals_summary': True, 'checkpoint_sha256': source['sha256'],
                      'checkpoint_unchanged': True, 'report_only': True})

# 3. disk rows for finished RL runs (docs/PROGRESS.md is shared with the remote driver: write under its lock)
pending_rows = [j['id'] for j in stage2 if j['script'] == 'scripts/train_rl.py' and proof_of(j['id']).exists()
                and not json.loads(proof_of(j['id']).read_text())['progress_recorded']]
if pending_rows and not DRY:
    with open(SETUP / 'remote-driver.lock', 'w') as shared:
        try:
            fcntl.flock(shared, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            messages.append('PROGRESS rows deferred: remote driver pass holds the lock')
        else:
            progress = ROOT / 'docs/PROGRESS.md'
            text = progress.read_text()
            anchor = text.index('\n## Next actions')
            rows = []
            for jid in pending_rows:
                record = json.loads(proof_of(jid).read_text())
                rows.append(f"| {stamp.strftime('%Y-%m-%d %H:%M')} | {record['free_home_ws']}B ({jid} completed with "
                            f"final eval at {record['final_steps'] // 1000}k; replay retained) |\n")
            progress.write_text(text[:anchor].rstrip('\n') + '\n' + ''.join(rows) + text[anchor:])
            for jid in pending_rows:
                record = json.loads(proof_of(jid).read_text())
                record['progress_recorded'] = True
                proof_of(jid).write_text(json.dumps(record, indent=2))
            messages.append(f'PROGRESS rows {pending_rows}')

# 4. admission, job by job (no group requirement); a large job never blocks a later fitting one
state = jobs.read_registry()


MIDDLE_WORK = ('scripts/train.py', 'scripts/train_rl.py', 'scripts/eval_rl.py')
original_rl_pending = any(status(j['id']) == jobs.PENDING for j in stage2
                          if j['id'] in original and j['script'] in ('scripts/train_rl.py', 'scripts/eval_rl.py'))
middle_pretrain_pending = any(status(j['id']) == jobs.PENDING for j in stage2
                              if j.get('stage2_admission_rank') == 'middle' and j['script'] == 'scripts/train.py')


def admissible(job):
    script = job['script']
    middle = job.get('stage2_admission_rank') == 'middle'
    if middle and script in MIDDLE_WORK and original_rl_pending:
        return False  # user 2026-10-15: crop pretrains/RL only after every original Stage 2 RL job is admitted
    if script == 'scripts/train.py' and not middle and job['priority'] != 0 and middle_pretrain_pending:
        return False  # ... and the pretrain-only variants only after the crop pretrains
    if script == 'scripts/evaluate.py':
        return (ROOT / 'runs' / job['id'] / 'config.yaml').is_file()
    if script in ('scripts/train_rl.py', 'scripts/eval_rl.py') and job['deps'][0] in by_id:
        return proof_of(job['deps'][0]).exists()  # ablation encoders only after export verification
    return True


def footprint(job):
    return int(MEASURED.get(job['script'], float(job['ram_gb']) * 2**30))


def rank(job):
    script = job['script']
    if script in ('scripts/export_encoder.py', 'scripts/evaluate.py'):
        return 0
    if job.get('stage2_admission_rank') == 'middle':
        return 2.5  # after the original Stage 2 RL pairs (2), before the pretrain-only variants (3)
    if script == 'scripts/train.py':
        return 1 if job['priority'] == 0 else 3
    return 2


def key(job):
    is_eval = job['script'] == 'scripts/eval_rl.py'
    trainer = by_id[job['id'].removesuffix('-eval')] if is_eval else job
    companion = is_eval and status(trainer['id']) in jobs.ACTIVE | {jobs.DONE}
    seed = int(trainer['args'][trainer['args'].index('--seed') + 1]) if '--seed' in trainer['args'] else 0
    return rank(job), not companion, seed, order.index(job['id'])


eligible_view = {i: {**s, 'status': status(i)} for i, s in state.items()}
plain_extra = [{k: v for k, v in j.items() if k != '_registration'} for j in extra]
candidates = sorted((j for j in jobs.eligible(stage2 + plain_extra, eligible_view)
                     if j.get('host', 'local') == 'local' and admissible(j)), key=key)
from admission_pause import paused  # noqa: E402  (intentional pauses, user 2026-10-28)
candidates = [j for j in candidates if not paused(j['id'])]
# Planned local shutdown (user, 2026-10-15): while this file exists, admit only the listed jobs and nothing after
# its cutoff, so no local job is started that cannot finish before the stop; closeouts continue unchanged.
HOLD = SETUP / 'stage2-admission-hold.json'
if HOLD.is_file():
    hold = json.loads(HOLD.read_text())
    cutoff = datetime.fromisoformat(hold['admit_until'])
    withheld = [j['id'] for j in candidates if j['id'] not in set(hold['allow']) or stamp >= cutoff]
    candidates = [j for j in candidates if j['id'] not in withheld]
    messages.append(f"ADMISSION_HOLD {hold['reason']}: {len(withheld)} eligible withheld, admit_until {hold['admit_until']}")


def session_pss():
    """Non-file PSS (Pss_Anon + Pss_Shmem, bytes) per session id: mapped file cache stays out of the budget."""
    return mb.session_nonfile_pss()


memory = mb.snapshot()
# budget (user decision 2026-10-21, option (b); memory_budget.py): every byte of the cgroup except verified clean,
# disk-backed file cache, active or inactive; clean dataset cache is excluded, so no dataset-cache allowance
charge = memory['budget_usage']
cache_reserve = 0
pss = session_pss()
# remaining growth of every active local job: its measured/declared footprint minus what it already uses
queue_scripts = {j['id']: j['script'] for j in queue['jobs']}
# other drivers' local DrM jobs: the measured DrM footprints too (their resident replay is already in the charge)
growth = {i: max(0, (footprint(by_id[i]) if i in by_id else int(MEASURED.get(queue_scripts.get(i), float(s.get('ram_gb', 0)) * 2**30)))
                 - pss.get(int(s.get('pid', -1)), 0))
          for i, s in state.items() if s.get('host', 'local') == 'local' and s['status'] in jobs.ACTIVE}
ramp = sum(growth.values())
total = charge + ramp + cache_reserve
admitted, taken = [], set()
for job in candidates:
    if job['script'] == 'scripts/eval_rl.py':
        trainer = job['id'].removesuffix('-eval')
        if status(trainer) not in jobs.ACTIVE | {jobs.DONE} and trainer not in taken:
            continue
    if total + footprint(job) >= CAP:
        continue
    admitted.append(job)
    taken.add(job['id'])
    total += footprint(job)
counts, crop_counts = {}, {}
for job in stage2:
    table = counts if job['id'] in original else crop_counts
    table.setdefault(job['script'].split('/')[-1], {}).setdefault(status(job['id']), 0)
    table[job['script'].split('/')[-1]][status(job['id'])] += 1
if admitted and not DRY:
    source = jobs.source_identity()
    report = json.loads((ROOT / 'docs/tests.json').read_text())
    assert source['source_sha256'] == report['source_sha256'] and report['full_suite_passed'] is True
    qpath = SETUP / f's2-local-{tag}.yaml'
    with qpath.open('x') as stream:
        yaml.safe_dump({'hosts': {'local': copy.deepcopy(queue['hosts']['local'])},
                        'jobs': json.loads(json.dumps(admitted))}, stream, sort_keys=False)
    with (SETUP / f's2-local-admission-{tag}.json').open('x') as stream:
        json.dump({'recorded_local': stamp.isoformat(), 'charge_bytes': charge, 'budget_metric': 'budget_usage',
                   'memory_snapshot': memory, 'remaining_growth_bytes': ramp, 'remaining_growth_by_job': growth,
                   'active_cache_allowance_bytes': cache_reserve, 'accounted_plus_admitted_bytes': total,
                   'admission_ceiling_bytes': CAP, 'commit': source['commit'], 'admission': 'job by job',
                   'admitted': {j['id']: footprint(j) for j in admitted}}, stream, indent=2)
    messages.append(f"ADMITTING {[j['id'] for j in admitted]} charge {charge} growth {ramp} reserve {cache_reserve}")
    messages.extend(jobs.tick(queue_path=qpath))
elif admitted:
    messages.append(f"DRY_RUN would admit {[j['id'] for j in admitted]} total {total} charge {charge} "
                    f"growth {ramp} reserve {cache_reserve}")
print('\n'.join(messages))
print('STAGE2', json.dumps(counts, sort_keys=True))
if crop_counts:
    print('STAGE2_CROP', json.dumps(crop_counts, sort_keys=True), 'gates', json.dumps(
        {'original_rl_pending': original_rl_pending, 'crop_pretrain_pending': middle_pretrain_pending}))
if extra:
    print('EXTRA', json.dumps({j['id']: status(j['id']) for j in extra}, sort_keys=True))
