"""One pass of the accepted remote campaign dispatcher, serialized by flock."""
import copy
import fcntl
import hashlib
import json
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

REPO = Path('/home/ws/ws/splatter4d')
sys.path.insert(0, str(REPO))
from scripts import jobs
from s4d.remote_access import RemoteClient
from s4d.remote_jobs import RemoteBackend

IMAGE = 'sha256:c3ffea7f7c114dd12755f9d537f760a6d5b31c96c923233185b775e0324799b7'
FROZEN_COMMIT = '226d8ae89b6f65097d372e597e0a7734d2eee3e0'  # method-frozen-v1 source (freeze manifest identity)
# Admission release (2026-10-16): cffa071 adds the new baselines and typed verification; model/loss/train/data code
# is identical to the frozen source and the single-camera RL path is unchanged (runs/remote/acceptance/cffa071).
COMMIT = '7c0d0d211774ceec0729a2d67ada318f1e872d39'  # 7c0d0d2: e4206f1 + aug.shift / agent.random_shift + evaluator fixes
MANIFEST = f'runs/remote/acceptance/{COMMIT[:7]}/manifest.json'
# Pool release (2026-10-23, user instructions: local and remote are one experiment pool; fix the remote environment):
# code fed4c17 = main 7aacd74 sources (chamfer_lean, future_k2, masking / per-frame pilots) + a probed Chamfer backend
# (exact fallback only where Triton cannot build); image = the accepted runtime image + gcc / libc6-dev
# (runs/setup/image-cc-20261023/: Dockerfile, build log, kernel check), so the fused Chamfer kernel builds on the remote.
# Rows placed on the remote by runs/setup/tools/pool_placement.py run on this (commit, image); every other kind keeps
# (COMMIT, IMAGE) for new admissions, and a retry always keeps the commit and image of its first launch.
# 2026-10-23 21:xx: pool release f3538b1 = fed4c17 + vision-only support (zero-width proprio replay fields, campaign W&B
# tags; tests/test_vision_only.py); jobs launched on fed4c17 keep it for retries.
POOL_COMMIT = 'f3538b1890c358a01ecbbaaa660c1822f8d89a44'
POOL_IMAGE = 'sha256:59082834c2c2c0d688be50a0b3b362bc46bdd2b1923d8ab712a473a0bd51870a'
RELEASES = {(COMMIT, IMAGE): MANIFEST,
            ('fed4c177949d367fa0dbefb1da7bc4690aa38d02', POOL_IMAGE): 'runs/remote/acceptance/fed4c17-cc/manifest.json',
            (POOL_COMMIT, POOL_IMAGE): f'runs/remote/acceptance/{POOL_COMMIT[:7]}-cc/manifest.json'}
POOL_READY = (REPO / RELEASES[(POOL_COMMIT, POOL_IMAGE)]).is_file()
# Encoder-pilot release (user directive 2026-10-25): d281258 = main with the encoder-pilot features (every new flag off by
# default, bit-for-bit identity tests), derived image. ONLY enc-* rows (and their encsmoke-* smoke rows) run on it: earlier
# code lacks the features (an enc row on it would fail on its missing variant file), and every other pool row keeps
# POOL_COMMIT; retries keep the release of their first launch. Enc rows rank after every other row (lowest priority);
# admission is capacity-based (user 2026-10-25 16:xx, superseding the ENC_HOLD / no-backfill rule): every ready row
# that fits the remaining capacity starts, in rank order.
ENC_COMMIT = 'd281258ad1480e49a155882628e87765c60a5d9a'
ENC_PREFIXES = ('enc-', 'encsmoke-')
RELEASES[(ENC_COMMIT, POOL_IMAGE)] = f'runs/remote/acceptance/{ENC_COMMIT[:7]}-cc/manifest.json'
ENC_READY = (REPO / RELEASES[(ENC_COMMIT, POOL_IMAGE)]).is_file()
ROOT = '/home/compu/kaist/sunho'
CAP = 1_000_000_000_000
GPU_LOAD_LIMIT = float((REPO / 'runs/setup/remote-gpu-load-limit.txt').read_text()) \
    if (REPO / 'runs/setup/remote-gpu-load-limit.txt').exists() else 2.0
MAX_REVIWO = 8
GPUS = ('GPU-4391fcee-537f-d408-d138-10d8a3866eea', 'GPU-6122c7a9-eaa6-b539-6005-f20e339da4ea',
        'GPU-9e9f1e97-b2ca-04e0-eb2a-8035398daa79', 'GPU-ca65e1b7-c0ac-6757-7d3f-816c3119a4c1')
FINISHED = REPO / 'runs/remote/finished'
DATA_LOGS = ['stage4-baseline-data-sync-20261012.log', 'remote-data-sync-w1-20261012.log',
             'remote-data-sync-w2-20261012.log', 'remote-data-sync-w3-20261012.log', 'remote-data-sync-w4-20261012.log']
INITIAL_MANIFEST = '56bd8ce9c59e81077061f577888f708541c2c14d605248f0cbdf81521e54bae1'
DRY = '--dry-run' in sys.argv
stamp = datetime.now().astimezone()
tag = stamp.strftime('%Y%m%d-%H%M%S')
FINISHED.mkdir(parents=True, exist_ok=True)
_lock = open(REPO / 'runs/setup/remote-driver.lock', 'w')
try:
    fcntl.flock(_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    print('REMOTE_DRIVER another pass is running; skipped')
    raise SystemExit(0)


PLACEMENT_PATH = REPO / 'runs/setup/stage2-placement-20261015.json'
placement = json.loads(PLACEMENT_PATH.read_text()) if PLACEMENT_PATH.exists() else {'effective_jobs': {}}
S2_REMOTE = dict(placement['effective_jobs'])
# 2026-10-17 follow-ups: Stage 2-structured jobs whose remote placement is part of their own registration
FOLLOWUP_PATH = REPO / 'runs/setup/followup-registration-20261017.json'
if FOLLOWUP_PATH.exists():
    S2_REMOTE.update({j['id']: j for j in json.loads(FOLLOWUP_PATH.read_text())['jobs']
                      if j.get('stage') == 2 and j.get('host') == 'remote'})
S2_KINDS = {'scripts/train.py': 's2_pretrain', 'scripts/evaluate.py': 's2_eval',
            'scripts/train_rl.py': 's2_rl', 'scripts/eval_rl.py': 's2_rl'}


def kind(job_id):
    row = globals().get('by_id', {}).get(job_id)
    if row is not None and str(row.get('placement', '')).startswith('pool'):
        return 'pool_pretrain' if row['script'] == 'scripts/train.py' else 'pool_rl'
    if job_id in S2_REMOTE:
        return S2_KINDS[S2_REMOTE[job_id]['script']]
    if job_id.startswith('ls-proxy-'):
        return 'proxy'
    # New baselines (registration runs/setup/baselines-sanity-remote-registration-20261016.json)
    if job_id.startswith(('s4-dynamo-', 's4-snerl-', 's4-mvp3d-')) and job_id.endswith(('-rem', '-rem-b')):
        return 'nbaseline'
    if job_id.startswith('bxs-') and job_id.endswith(('-rem', '-rem-eval', '-rem-b', '-rem-b-eval')):
        return 'nbrl'
    if job_id.startswith('s4-drm-') and not job_id.startswith('s4-drm-cnn-'):
        return 'brl'
    if job_id.startswith(('s4-sincro-', 's4-reviwo-')):
        return 'baseline'
    if job_id.startswith('s3spec-'):
        return 'stage3'
    if job_id.startswith('s3-export-m3d-'):
        return 'final_export'
    if job_id.startswith('s3-drm-splatter4d-'):
        return 'orl'
    if job_id.startswith('s3-m3d-'):
        return 'final_pretrain'
    return None


def load(job):
    jid = job['id']
    if kind(jid) == 'final_export':
        return 0.1
    if kind(jid) == 's2_eval':
        return 0.6
    if jid.endswith('-eval'):
        return 0.3
    if kind(jid) in ('proxy', 'brl', 'orl', 's2_rl', 'nbrl', 'pool_rl'):
        return 0.5
    if jid.startswith('s4-reviwo-'):
        return 0.4
    return 1.0


def pretrain_target(job):
    steps = [int(a.split('=', 1)[1]) for a in job['args'] if a.startswith('train.steps=')]
    assert len(steps) == 1, job['id']
    return steps[0]


queue = jobs.load_queue()
state = jobs.read_registry()
by_id = {j['id']: j for j in queue['jobs']}
remote = [j for j in queue['jobs'] if j.get('host') == 'remote' and kind(j['id'])]
freeze_path = REPO / 'runs/setup/method-frozen-v1.json'
freeze = json.loads(freeze_path.read_text()) if freeze_path.exists() else None
if freeze:
    assert freeze['source_commit'] == FROZEN_COMMIT
    freeze_sha = hashlib.sha256(freeze_path.read_bytes()).hexdigest()
    for job in remote:
        if kind(job['id']) in ('final_pretrain', 'final_export', 'orl'):
            assert job['frozen_method'] == freeze['tag'] and job['freeze_manifest_sha256'] == freeze_sha
            if kind(job['id']) == 'final_pretrain':
                assert job['args'][job['args'].index('--set') + 1:] == freeze['overrides']
client = RemoteClient()
messages = []

for job in remote:
    jid = job['id']
    if kind(jid).startswith('s2_'):
        # Scientific Stage 2 closeouts are verified by the Stage 2 driver, including remote result checksums.
        proof = REPO / 'runs/setup' / f's2-{jid}-completion.json'
        if (not jid.endswith('-eval') and state.get(jid, {}).get('status') == jobs.DONE and proof.exists()
                and not (FINISHED / f'{jid}.json').exists()):
            ids = [jid, f'{jid}-eval'] if kind(jid) == 's2_rl' else [jid]
            # W&B never blocks numerical work (user rule): offline runs are uploaded by the separate
            # sync_wandb_remote.py batch, not inline before admission (2026-10-16; status runs/remote/wandb/status.json).
            record = {'job_id': jid, 'finished_local': stamp.isoformat(), 'stage2_closeout': str(proof),
                      'wandb': 'deferred to the sync_wandb_remote.py batch', 'wandb_runs': ids}
            if not DRY:
                (FINISHED / f'{jid}.json').write_text(json.dumps(record, indent=2) + '\n')
            messages.append(f'STAGE2_FINISHED {jid}')
        continue
    if jid.endswith('-eval') or state.get(jid, {}).get('status') != jobs.DONE or (FINISHED / f'{jid}.json').exists():
        continue
    run = REPO / 'runs' / jid
    record = {'job_id': jid, 'finished_local': stamp.isoformat()}
    jobs._verify_synced_exit(jid, state[jid], state[jid]['sync'], REPO / 'runs')
    if kind(jid) == 'final_export':
        import torch

        export = run / 'encoder.pt'
        payload = torch.load(export, map_location='cpu', weights_only=True)
        source = job['deps'][0]
        task = by_id[source]['args'][by_id[source]['args'].index('--config') + 2].rsplit('/', 1)[1].removesuffix('.yaml')
        assert payload['format'] == 'splatter4d-encoder-v1' and payload['step'] == freeze['pretraining_steps']
        assert payload['run_name'] == source and payload['frame_strides'] == freeze['task_configs'][task]['data']['strides']
        assert payload['source_checkpoint'] == job['args'][job['args'].index('--checkpoint') + 1]
        assert all(payload['encoder_config'][key] == value
                   for key, value in freeze['task_configs'][task]['model']['encoder'].items())
        checkpoint = REPO / payload['source_checkpoint']
        expected = state[source]['sync']['files']['checkpoints/step_0200000.pt']
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == expected
        record.update(encoder_sha256=hashlib.sha256(export.read_bytes()).hexdigest(),
                      step=payload['step'], source_checkpoint_sha256=expected, source_job=source)
        (FINISHED / f'{jid}.json').write_text(json.dumps(record, indent=2) + '\n')
        messages.append(f'EXPORT_VERIFIED {jid}')
        continue
    if kind(jid) in ('proxy', 'brl', 'orl', 'nbrl', 'pool_rl'):
        if state.get(f'{jid}-eval', {}).get('status') != jobs.DONE:
            continue
        jobs._verify_synced_exit(f'{jid}-eval', state[f'{jid}-eval'], state[f'{jid}-eval']['sync'], REPO / 'runs')
        final = json.loads((run / 'final.json').read_text())
        evals = [json.loads(line) for line in (REPO / 'runs' / f'{jid}-eval' / 'eval.jsonl').read_text().splitlines()]
        steps = [r['step'] for r in evals]
        target = next(int(a.split('=')[1]) for a in job['args'] if a.startswith('train.num_train_steps=')) \
            if any(a.startswith('train.num_train_steps=') for a in job['args']) else 1000000
        assert final['completed'] is True and final['steps'] == target, (jid, final)
        assert steps == list(range(10000, target + 1, 10000)), (jid, steps)
        record.update(final_steps=final['steps'], eval_points=len(steps),
                      last5_train_success=sum(r['train_cameras_success'] for r in evals[-5:]) / 5)
        ids = [jid, f'{jid}-eval']
        out = client.checked_ssh(
            f'df -B1 --output=avail {shlex.quote(ROOT)} | tail -1',
            wait=False, timeout=600, idempotent=True).stdout.strip()
        record.update(remote_replay='retained', remote_free_bytes=int(out))
    else:
        # NUL runs left by an unclean remote shutdown (maintenance reboot) precede the record written after resume
        rows = [json.loads(line.strip('\x00')) for line in (run / 'metrics.jsonl').read_text().splitlines()
                if line.strip('\x00').strip()]
        last = max(r.get('step', 0) for r in rows)
        if kind(jid) == 'nbaseline':
            completion = json.loads((run / 'completion.json').read_text())
            assert completion.get('status') == 'completed', (jid, completion)
            assert (run / 'encoder.pt').is_file() and (run / 'encoder_init.pt').is_file(), jid
            target = last + 1000  # the recipe's own length (epochs or steps) is recorded in completion.json
            record['completion'] = completion
        elif kind(jid) == 'pool_pretrain':
            completion = json.loads((run / 'completion.json').read_text())
            target = pretrain_target(job)
            assert completion['status'] == 'completed' and completion['steps'] == target, (jid, completion)
            record['completion'] = completion
        elif kind(jid) in ('stage3', 'final_pretrain'):
            completion = json.loads((run / 'completion.json').read_text())
            target = pretrain_target(job)
            assert completion['status'] == 'completed' and completion['steps'] == target, (jid, completion)
            if kind(jid) == 'final_pretrain':
                cfg = yaml.safe_load((run / 'config.yaml').read_text())
                cfg.pop('run')
                assert cfg == freeze['task_configs'][cfg['data']['task']], jid
                record['final_checkpoint_sha256'] = state[jid]['sync']['files']['checkpoints/step_0200000.pt']
        else:
            target = 300000 if '-sincro-' in jid else 100001
            assert (run / 'encoder.pt').is_file(), jid
        assert last >= target - 1000, (jid, last)
        record.update(last_metrics_step=last, configured_steps=target)
        ids = [jid]
    # W&B never blocks numerical closeout and must not saturate the link during dispatch (2026-10-16): offline runs
    # are pulled/uploaded by the separate sync_wandb_remote.py batch (status runs/remote/wandb/status.json).
    record['wandb'] = 'deferred to the sync_wandb_remote.py batch'
    record['wandb_runs'] = ids
    (FINISHED / f'{jid}.json').write_text(json.dumps(record, indent=2) + '\n')
    if 'remote_free_bytes' in record and kind(jid) != 'pool_rl':  # pool RL closeouts: no per-run PROGRESS rows
        row = (f"| {stamp.strftime('%Y-%m-%d %H:%M')} | remote {record['remote_free_bytes'] / 1e12:.2f} TB ({jid} "
               f"completed with final eval at {record['final_steps'] // 1000}k; remote replay retained) |\n")
        progress = REPO / 'docs/PROGRESS.md'
        text = progress.read_text()
        anchor = text.index('\n## Next actions')
        progress.write_text(text[:anchor].rstrip('\n') + '\n' + row + text[anchor:])
    messages.append(f'FINISHED {jid}')

uploaded = set()
for job in queue['jobs']:
    if not job['id'].startswith(('export-ls-', 'export-s2ab-crop', 'export-s2ab-shift')):
        continue
    if job['id'].startswith(('export-s2ab-crop', 'export-s2ab-shift')) and not (
            REPO / 'runs/setup' / f"s2-{job['id']}-completion.json").exists():
        continue
    out = job['args'][job['args'].index('--out') + 1]
    marker = REPO / 'runs/remote/exports' / (Path(out).name + '.json')
    if marker.exists():
        assert json.loads(marker.read_text())['sha256'] == hashlib.sha256((REPO / out).read_bytes()).hexdigest()
        uploaded.add(out)
        continue
    if state.get(job['id'], {}).get('status') != jobs.DONE:
        continue
    local = REPO / out
    digest = hashlib.sha256(local.read_bytes()).hexdigest()
    client.rsync_upload(local, f'{ROOT}/{out}')
    assert hashlib.sha256(local.read_bytes()).hexdigest() == digest
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({'export': out, 'sha256': digest, 'uploaded': stamp.isoformat()}, indent=2) + '\n')
    uploaded.add(out)
    messages.append(f'EXPORT_UPLOADED {out}')

targets = set()
for name in DATA_LOGS:
    path = REPO / 'runs/setup' / name
    if path.exists():
        for line in path.read_text().splitlines():
            if line.startswith('MANIFEST '):
                manifest = json.loads((REPO / 'runs/remote/data' / f'{line.split()[1]}.json').read_text())
                targets |= {f['target'].split('/splatter4d_v1/', 1)[1] for f in manifest['files']}
manifest = json.loads((REPO / 'runs/remote/data' / f'{INITIAL_MANIFEST}.json').read_text())
targets |= {f['target'].split('/splatter4d_v1/', 1)[1] for f in manifest['files']}


def task_of(job):
    args = job['args']
    if '--task' in args:
        return args[args.index('--task') + 1]
    config = next(a for a in args if a.startswith('configs/'))
    return config.rsplit('/', 1)[1].removesuffix('.yaml') if 'baselines' in config else \
        next(a for a in args if a.startswith('configs/metaworld/tasks/')).rsplit('/', 1)[1].removesuffix('.yaml')


# Source features of the accepted remote release (COMMIT). A job naming a feature the release lacks is never launched:
# its configuration keys would be silently ignored by older code. 'aug-shift': aug.shift, agent.random_shift.
RELEASE_FEATURES = {'aug-shift'}  # 7c0d0d2 (accepted 2026-10-17 19:34, manifest 5dd4dee0...)


def ready(job):
    jid = job['id']
    k = kind(jid)
    if job.get('requires_feature') and job['requires_feature'] not in RELEASE_FEATURES:
        return False
    if k == 's2_eval':
        proof = REPO / 'runs/setup' / f"s2-{job['deps'][0]}-completion.json"
        config = REPO / 'runs' / jid / 'config.yaml'
        if not proof.is_file() or not config.is_file():
            return False
        marker = REPO / 'runs/setup' / f'{jid}-remote-config.json'
        digest = hashlib.sha256(config.read_bytes()).hexdigest()
        if marker.exists():
            assert json.loads(marker.read_text())['sha256'] == digest
        else:
            client.rsync_upload(config, f'{ROOT}/runs/{jid}/config.yaml')
            marker.write_text(json.dumps({'sha256': digest, 'uploaded': stamp.isoformat()}) + '\n')
        return True
    if k == 'pool_rl':  # the export (a local file) is uploaded once, checksum-verified, before the pair starts
        trainer = by_id[jid.removesuffix('-eval')]
        export = next((a.split('=', 1)[1] for a in trainer['args'] if a.startswith('vision.export_path=')), None)
        if export is None:  # end-to-end CNN trainer (2026-10-24): no frozen export to upload
            return True
        local = REPO / export
        if not local.is_file():
            return False
        marker = REPO / 'runs/remote/exports' / (export.replace('/', '__') + '.json')
        digest = hashlib.sha256(local.read_bytes()).hexdigest()
        if marker.exists() and json.loads(marker.read_text())['sha256'] == digest:
            return True
        client.rsync_upload(local, f'{ROOT}/{export}')
        remote = client.checked_ssh(f'sha256sum {shlex.quote(ROOT + "/" + export)}', wait=False, idempotent=True,
                                    timeout=600).stdout.split()[0]
        assert remote == digest, export
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({'export': export, 'sha256': digest, 'uploaded': stamp.isoformat()}) + '\n')
        return True
    if k == 'nbrl':
        trainer = by_id[jid.removesuffix('-eval')]
        export = trainer['baseline_export']
        if not export.startswith('runs/pretrain/'):
            return True  # a remote pretraining output, released by the dependency on that run
        marker = REPO / 'runs/remote/exports' / (export.replace('/', '__') + '.json')
        digest = hashlib.sha256((REPO / export).read_bytes()).hexdigest()
        if marker.exists():
            return json.loads(marker.read_text())['sha256'] == digest
        client.rsync_upload(REPO / export, f'{ROOT}/{export}')
        remote = client.checked_ssh(f'sha256sum {shlex.quote(ROOT + "/" + export)}', wait=False, idempotent=True,
                                    timeout=600).stdout.split()[0]
        assert remote == digest, export
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({'export': export, 'sha256': digest, 'uploaded': stamp.isoformat()}) + '\n')
        return True
    if k in ('proxy', 'orl', 's2_rl'):
        trainer = by_id[jid.removesuffix('-eval')]
        export = next(a.split('=', 1)[1] for a in trainer['args'] if a.startswith('vision.export_path='))
        if export in uploaded:
            return True
        if k == 'orl':
            proof = FINISHED / f"{trainer['deps'][0]}.json"
            return proof.is_file() and hashlib.sha256((REPO / export).read_bytes()).hexdigest() == \
                json.loads(proof.read_text())['encoder_sha256']
        return False
    if k == 'brl':
        return True
    if k == 'final_export':
        return (FINISHED / f"{job['deps'][0]}.json").is_file()
    if k == 'stage3' and freeze and freeze['pretraining_steps'] != 300000:
        return False
    task = task_of(job)
    need = {f'{task}.hdf5', f'splits/{task}_seed0.json'}
    if k in ('stage3', 'final_pretrain', 's2_pretrain', 'pool_pretrain'):
        need |= {f'heldout_sets/{task}.hdf5', f'workspace_stats/{task}.json'}
    return need <= targets


RANK = {'proxy': 0, 'final_export': 0, 's2_eval': 0, 's2_rl': 0.5, 'baseline': 1,
        'final_pretrain': 1, 'nbaseline': 1.5, 'pool_pretrain': 1.6, 'brl': 2, 'orl': 2, 'pool_rl': 2, 'nbrl': 2.2,
        's2_pretrain': 2.5, 'stage3': 3}


def dispatch_key(job):
    trainer = by_id[job['id'].removesuffix('-eval')]
    is_evaluator = job['id'].endswith('-eval')
    trainer_status = state.get(trainer['id'], {}).get('status', jobs.PENDING)
    companion = is_evaluator and trainer_status in jobs.ACTIVE | {jobs.DONE}
    seed = int(trainer['args'][trainer['args'].index('--seed') + 1]) if '--seed' in trainer['args'] else 0
    rank = RANK[kind(job['id'])]
    if kind(job['id']) == 's2_pretrain' and job.get('stage2_extension') == 'followup-20261017':
        rank = 1.6  # the follow-up pretrains gate 24 RL pairs: after the baseline pretrains, before baseline RL
    if job['id'].startswith(ENC_PREFIXES):
        rank += 100  # encoder pilots: after every other remote row
    return rank, not companion, seed, [j['id'] for j in remote].index(job['id'])


# Crashed jobs with restarts left are retried, exactly as scripts/jobs.py _tick treats them (status -> pending when
# failures <= max_restarts); the scheduler resumes them from checkpoints/latest.pt (--resume auto) under a new attempt.
# (2026-10-21: after the 08:38-08:48 remote outage the raw registry view never offered the seven crashed SNeRL
# pretrains for retry.)
retry_view = {i: ({**s, 'status': jobs.PENDING} if s.get('status') == 'crashed' and i in by_id
                  and s.get('failures', 0) <= int(by_id[i].get('max_restarts', 1)) else s) for i, s in state.items()}
candidates = [j for j in jobs.eligible(remote, retry_view) if ready(j)]
if (REPO / 'runs/setup/remote-hold-admissions').exists():
    messages.append('ADMISSIONS_HELD ' + (REPO / 'runs/setup/remote-hold-admissions').read_text().strip())
    candidates = []
# Capacity-based scheduling (user 2026-10-25 16:xx): encoder pilots need only their accepted release; they are offered
# after every other ready row (dispatch_key, pool unit order) and take whatever capacity remains (enc_gate.py retired).
ENC_ALLOWED = ENC_READY
if not ENC_ALLOWED:
    candidates = [j for j in candidates if not j['id'].startswith(ENC_PREFIXES)]
candidates.sort(key=dispatch_key)
active = {i: s for i, s in state.items() if s.get('host') == 'remote' and s['status'] in jobs.ACTIVE}
reserved = sum(int(float(by_id.get(i, {}).get('ram_gb', 0)) * 2**30) for i in active)
gpu_load = {g: 0.0 for g in GPUS}
gpu_mem = {g: 0.0 for g in GPUS}
for i, s in active.items():
    if s.get('gpu') in gpu_load and i in by_id:
        gpu_load[s['gpu']] += load(by_id[i])
        gpu_mem[s['gpu']] += float(by_id[i]['mem_gb'])
reviwo = sum(1 for i in active if i.startswith('s4-reviwo-'))
chosen, total, taken = [], reserved, set()


def place(job):
    fitting = [g for g in GPUS
               if gpu_load[g] + load(job) <= GPU_LOAD_LIMIT + (2.0 if kind(job['id']) == 'proxy' else 0.0) + 1e-9
               and gpu_mem[g] + float(job['mem_gb']) <= 90]
    return min(fitting, key=lambda g: (gpu_load[g], gpu_mem[g])) if fitting else None


for job in candidates:
    if job['id'].endswith('-eval'):
        trainer = job['id'].removesuffix('-eval')
        if state.get(trainer, {}).get('status') not in jobs.ACTIVE | {jobs.DONE} and trainer not in taken:
            continue
    if job['id'].startswith('s4-reviwo-') and reviwo >= MAX_REVIWO:
        continue
    need = int(float(job['ram_gb']) * 2**30)
    if total + need > CAP:
        continue
    gpu = place(job)
    if gpu is None:
        continue
    gpu_load[gpu] += load(job)
    gpu_mem[gpu] += float(job['mem_gb'])
    chosen.append({**copy.deepcopy(job), 'gpu': gpu})
    taken.add(job['id'])
    total += need
    reviwo += job['id'].startswith('s4-reviwo-')
# ---- host-neutral pool (user instruction 2026-10-23): eligible, never-launched local rows that the remote can start
# now move here (runs/setup/tools/pool_placement.py: host-specific fields only) and are admitted in this pass. Pretraining
# first (it gains most from the full GPUs), then RL trainer + evaluator pairs, each in priority / registration order.
held = (REPO / 'runs/setup/remote-hold-admissions').exists()
if POOL_READY and not DRY and not held:
    sys.path.insert(0, str(REPO / 'runs/setup/tools'))
    import pool_placement  # noqa: E402

    units = []
    for job in jobs.eligible(queue['jobs'], retry_view):
        if job.get('host', 'local') != 'local' or job['script'] not in ('scripts/train.py', 'scripts/train_rl.py'):
            continue
        if not pool_placement.movable(job, state, by_id):
            continue
        members = [job] + ([by_id[job['id'] + '-eval']] if job['script'] == 'scripts/train_rl.py' else [])
        units.append(members)
    if not ENC_ALLOWED:
        units = [m for m in units if not m[0]['id'].startswith(ENC_PREFIXES)]
    # encoder-pilot units after every other unit (stable sort keeps the established order otherwise)
    units.sort(key=lambda m: (m[0]['id'].startswith(ENC_PREFIXES), 0 if m[0]['script'] == 'scripts/train.py' else 1))
    plan = []
    for members in units:
        tentative = []
        for u in members:
            row_load = 1.0 if u['script'] == 'scripts/train.py' else (0.3 if u['id'].endswith('-eval') else 0.5)
            need = int(float(u['ram_gb']) * 2**30)
            fitting = [g for g in GPUS if gpu_load[g] + row_load <= GPU_LOAD_LIMIT + 1e-9
                       and gpu_mem[g] + float(u['mem_gb']) <= 90]
            if total + need > CAP or not fitting:
                break
            g = min(fitting, key=lambda g: (gpu_load[g], gpu_mem[g]))
            gpu_load[g] += row_load
            gpu_mem[g] += float(u['mem_gb'])
            total += need
            tentative.append((u['id'], g, row_load, float(u['mem_gb']), need))
        if len(tentative) != len(members):
            for _, g, row_load, mem, need in tentative:
                gpu_load[g] -= row_load
                gpu_mem[g] -= mem
                total -= need
            continue
        plan.append(tentative)
    wanted = [t[0] for tentative in plan for t in tentative]
    moved = set(pool_placement.move(wanted, 'remote', 'free remote capacity (remote driver pass)')) if wanted else set()
    if wanted and not moved:
        messages.append(f'POOL move skipped (local admission in progress or rows already admitted): {wanted[:4]}')
    if moved:
        queue = jobs.load_queue()
        by_id = {j['id']: j for j in queue['jobs']}
        remote = [j for j in queue['jobs'] if j.get('host') == 'remote' and kind(j['id'])]
        messages.append(f'POOL moved to remote: {sorted(moved)}')
        for tentative in plan:
            ok = all(i in moved for i, *_ in tentative) and all(ready(by_id[i]) for i, *_ in tentative)
            for i, g, row_load, mem, need in tentative:
                if ok:
                    chosen.append({**copy.deepcopy(by_id[i]), 'gpu': g})
                    taken.add(i)
                else:  # placed but not startable yet (e.g. export upload pending): admitted by a later pass
                    gpu_load[g] -= row_load
                    gpu_mem[g] -= mem
                    total -= need


def job_release(job):
    """(commit, image) a job runs on: a retry keeps its first launch's; pool rows the pool release; others COMMIT/IMAGE."""
    previous = state.get(job['id'], {})
    if previous.get('commit'):
        return previous['commit'], previous.get('image_digest', IMAGE)
    if job['id'].startswith(ENC_PREFIXES):
        return ENC_COMMIT, POOL_IMAGE
    return (POOL_COMMIT, POOL_IMAGE) if kind(job['id']) in ('pool_pretrain', 'pool_rl') else (COMMIT, IMAGE)


if chosen and DRY:
    messages.append(f"DRY_RUN would admit {[(j['id'], j['gpu'][4:8], job_release(j)[0][:7]) for j in chosen]}, reserved {total} bytes")
if chosen and not DRY:
    groups = {}
    for job in chosen:
        groups.setdefault(job_release(job), []).append(job)
    for (commit, image), group in groups.items():
        if (commit, image) not in RELEASES or not (REPO / RELEASES[(commit, image)]).is_file():
            messages.append(f'HELD {[j["id"] for j in group]}: release {commit[:7]}/{image[7:15]} has no accepted manifest')
            continue
        host = copy.deepcopy(queue['hosts']['remote'])
        host.update(enabled=True, commit=commit, image_digest=image, evidence_manifest=RELEASES[(commit, image)])
        qtag = f'{tag}-{commit[:7]}' if len(groups) > 1 else tag
        qpath = REPO / 'runs/setup' / f'remote-dispatch-{qtag}.yaml'
        with qpath.open('x') as stream:
            yaml.safe_dump({'hosts': {'local': {'enabled': False}, 'remote': host}, 'jobs': group}, stream, sort_keys=False)
        with (REPO / 'runs/setup' / f'remote-dispatch-{qtag}.json').open('x') as stream:
            json.dump({'recorded_local': stamp.isoformat(), 'commit': commit, 'image_digest': image,
                       'gpu_load_limit': GPU_LOAD_LIMIT, 'max_reviwo': MAX_REVIWO, 'cap_bytes': CAP,
                       'reserved_before_bytes': reserved, 'reserved_after_bytes': total,
                       'admission': 'job by job', 'freeze_manifest_sha256': freeze_sha if freeze else None,
                       'admitted': {j['id']: j['gpu'] for j in group}, 'gpu_load_after': gpu_load, 'queue': str(qpath)},
                      stream, indent=2)
        messages.append(f"ADMITTING {[(j['id'], j['gpu'][4:8]) for j in group]} on {commit[:7]}")
        backend = RemoteBackend(REPO, REPO / 'runs')
        messages += jobs.tick(queue_path=qpath, remote_backend=backend)
state = jobs.read_registry()
counts = {}
for job in remote:
    if job['id'].endswith('-eval'):
        continue
    k = kind(job['id'])
    s = state.get(job['id'], {}).get('status', jobs.PENDING)
    c = counts.setdefault(k, {'finished': 0, 'active': 0, 'pending': 0, 'other': 0})
    if (FINISHED / f"{job['id']}.json").exists():
        c['finished'] += 1
    elif s in jobs.ACTIVE:
        c['active'] += 1
    elif s == jobs.PENDING:
        c['pending'] += 1
    else:
        c['other'] += 1
print('\n'.join(messages))
print('REMOTE', json.dumps(counts), 'gpu_load', {g[4:8]: round(v, 1) for g, v in gpu_load.items()})
