"""Remote autonomy plan for the local outage of 2026-10-31 23:00 KST (user authorization 2026-10-31).

Builds, for every job the remote should run while the local server (and with it the remote campaign driver, the
scheduler daemon and Claude) is offline, the exact launch command the local driver would issue
(s4d.remote_launch.build_remote_command for the job's accepted release; the launch-side checks for foreign GPU
processes, RAM reserve, disk floor, per-GPU job and memory caps are part of each command), built once per authorized
GPU (the acceptance binding names that GPU's isolation evidence); the remote runner picks the GPU at launch time. Nothing is launched here.

Order: the self-render v4 pilot chain first (user 2026-10-30/31: raised priority), then every ready remote row in the
remote driver's dispatch order (rank, companion, seed, queue order). Units: an RL trainer and its pending companion
evaluator. Retries of crashed rows keep the release of their first launch and take the next attempt number.
Specs go to a scratch copy (runs/remote/autonomy-20261031/specs), not runs/remote/jobs, so the local scheduler's own
launch records stay untouched.

    python -I gen_remote_autonomy_plan_20261031.py [--srv4-migrated] [--limit N]
"""
import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path('/home/ws/ws/splatter4d')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/setup/tools'))
from scripts import jobs  # noqa: E402
from s4d.remote_access import RemoteClient  # noqa: E402
from s4d.remote_jobs import RemoteBackend, acceptance_manifest  # noqa: E402
from s4d import remote_launch as RL  # noqa: E402
from admission_pause import paused  # noqa: E402

OUT = ROOT / 'runs/remote/autonomy-20261031'
GPUS = ('GPU-4391fcee-537f-d408-d138-10d8a3866eea', 'GPU-6122c7a9-eaa6-b539-6005-f20e339da4ea',
        'GPU-9e9f1e97-b2ca-04e0-eb2a-8035398daa79', 'GPU-ca65e1b7-c0ac-6757-7d3f-816c3119a4c1')
PLACEHOLDER = GPUS[0]
IMAGE = 'sha256:59082834c2c2c0d688be50a0b3b362bc46bdd2b1923d8ab712a473a0bd51870a'
# releases (remote_campaign_driver.py as of 2026-10-31)
POOL_COMMIT = 'f3538b1890c358a01ecbbaaa660c1822f8d89a44'
ENC_COMMIT = 'd281258ad1480e49a155882628e87765c60a5d9a'
DS_COMMIT = 'b9196a8f292c5a6b1782a20ef2af0ba653cdd078'
SRV4_COMMIT = 'e8ad5ac17de1434d85256527d5a42e7e10128b70'
SNERL_NT_COMMIT = 'dd86d78321963d10d84645cbeb6ed3938983df06'
SNERL_NT_IDS = ('ds-pre-snerl-drawer-open-r2', 'ds-pre-snerl-window-open', 'ds-pre-snerl-soccer', 'ds-pre-snerl-sweep-into')
GPU_HEAVY = ('ds-sac-cnn-', 'ds-sac-snerl-', 'ds-ppo-cnn-', 'ds-dreamer-', 'ds-flashsac-cnn-', 'ds-pre-')
RANK = {'pool_export': 0, 'pool_pretrain': 1.6, 'pool_rl': 2}

ap = argparse.ArgumentParser()
ap.add_argument('--srv4-migrated', action='store_true', help='include the migrated srv4 pretraining resumes')
ap.add_argument('--limit', type=int, default=10**9)
args = ap.parse_args()

queue = jobs.load_queue()
state = jobs.read_registry()
by_id = {j['id']: j for j in queue['jobs']}
order = {j['id']: i for i, j in enumerate(queue['jobs'])}


def kind(job):
    if str(job.get('placement', '')).startswith('pool'):
        if job['script'] == 'scripts/export_encoder.py':
            return 'pool_export'
        return 'pool_pretrain' if job['script'] == 'scripts/train.py' else 'pool_rl'
    return 'pool_rl'


def load(job):
    jid = job['id']
    if jid.startswith(GPU_HEAVY) and not jid.endswith('-eval'):
        return 1.0
    if kind(job) == 'pool_export':
        return 0.1
    if jid.endswith('-eval'):
        return 0.3
    if kind(job) == 'pool_pretrain':  # the driver charges pool pretrainings a full unit
        return 1.0
    return 0.5


def release(job):
    previous = state.get(job['id'], {})
    if job['id'] in SNERL_NT_IDS and not previous.get('commit'):
        return SNERL_NT_COMMIT
    if previous.get('commit') and previous.get('host') == 'remote':
        return previous['commit']
    if job['id'].startswith(('enc-', 'encsmoke-')):
        return ENC_COMMIT
    if job['id'].startswith('srv4-'):
        return SRV4_COMMIT
    if job['id'].startswith(('ds-', 'sr-')):
        return DS_COMMIT
    return POOL_COMMIT


def status(jid):
    return state.get(jid, {}).get('status', jobs.PENDING)


def next_attempt(jid):
    return int(state.get(jid, {}).get('attempts', 0)) + 1


def retryable(jid):
    s = state.get(jid, {})
    return s.get('status') == 'crashed' and s.get('failures', 0) <= int(by_id[jid].get('max_restarts', 1))


def dep_done(dep):
    s = state.get(dep, {})
    return s.get('status') == jobs.DONE or (s.get('host') == 'remote' and s.get('status') == 'sync_pending'
                                           and (s.get('remote_verified') or {}).get('attempt') == s.get('attempts'))


# ---------------------------------------------------------------- selection
plan, chosen = [], set()


def add(job, deps=(), dep_results=None, note='', attempts=1, require_files=()):
    plan.append({'id': job['id'], 'job': job, 'deps': list(deps), 'dep_results': dep_results or {}, 'note': note,
                 'fallback_attempts': attempts, 'require_files': list(require_files)})
    chosen.add(job['id'])


# 1. self-render v4 pilot chain
TASKS = ('hammer', 'pick-place', 'bin-picking')
for t in TASKS:
    pre, exp = f'srv4-csw4-seed0-{t}-200k', f'srv4-csw4-export-seed0-{t}-200k'
    pre_remote = state.get(pre, {}).get('host') == 'remote'  # recorded `migrated` event (migrate_srv4_to_remote)
    if pre_remote:
        add({**by_id[pre], 'host': 'remote'}, note='migrated resume from the local checkpoint pause', attempts=2)
        add({**by_id[exp], 'host': 'remote'}, deps=[pre],
            dep_results={pre: ['completion.json', 'checkpoints/step_0200000.pt']}, attempts=2)
    for s in ('2000', '2001', '2002'):
        tr, ev = f'ds-flashsac-srv4-csw4-{t}-s{s}', f'ds-flashsac-srv4-csw4-{t}-s{s}-eval'
        req = ()
        if t == 'bin-picking':  # export run locally before the shutdown and uploaded (see the briefing)
            deps, dres, req = [], {}, (f'runs/{exp}/encoder.pt',)
        elif pre_remote:
            deps, dres = [exp], {exp: ['encoder.pt']}
        else:
            continue
        add({**by_id[tr], 'host': 'remote'}, deps=deps, dep_results=dres, note='self-render v4 FlashSAC', attempts=2,
            require_files=req)
        add({**by_id[ev], 'host': 'remote'}, deps=deps, dep_results=dres, note='companion evaluator', attempts=2,
            require_files=req)

# 2. ready remote rows in the remote driver's dispatch order
ready = []
for j in queue['jobs']:
    if j.get('host') != 'remote' or j['id'] in chosen or paused(j['id']):
        continue
    if j['id'].startswith('s3spec-'):  # remote driver kind 'stage3': held (frozen method is 200k, not 300k)
        continue
    if j.get('requires_feature') and j['requires_feature'] != 'aug-shift':  # RELEASE_FEATURES of the driver
        continue
    if not (status(j['id']) == jobs.PENDING or retryable(j['id'])):
        continue
    if not all(dep_done(d) for d in j.get('deps', [])):
        continue
    if j.get('reevaluates') and status(j['reevaluates']) != jobs.DONE:
        continue
    ready.append(j)


def dispatch_key(j):
    trainer = by_id.get(j.get('reevaluates') or j['id'].removesuffix('-eval'), j)
    companion = j['id'].endswith('-eval') and status(trainer['id']) in jobs.ACTIVE | {jobs.DONE}
    seed = int(trainer['args'][trainer['args'].index('--seed') + 1]) if '--seed' in trainer['args'] else 0
    rank = RANK.get(kind(j), 2) + (100 if j['id'].startswith(('enc-', 'encsmoke-')) else 0)
    return rank, not companion, seed, order[j['id']]


ready.sort(key=dispatch_key)
for j in ready[: args.limit]:
    add(j)

# ---------------------------------------------------------------- launch commands
import shutil  # noqa: E402
for sub in ('specs-runs', 'commands'):  # regenerated from scratch on every run
    shutil.rmtree(OUT / sub, ignore_errors=True)
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'commands').mkdir(exist_ok=True)
backend = RemoteBackend(ROOT, OUT / 'specs-runs')
client = RemoteClient()
client.check_tunnel(wait=False)
host_base = dict(queue['hosts']['remote'])
manifests = {POOL_COMMIT: f'runs/remote/acceptance/{POOL_COMMIT[:7]}-cc/manifest.json',
             ENC_COMMIT: f'runs/remote/acceptance/{ENC_COMMIT[:7]}-cc/manifest.json',
             DS_COMMIT: f'runs/remote/acceptance/{DS_COMMIT[:7]}-cc/manifest.json',
             SRV4_COMMIT: f'runs/remote/acceptance/{SRV4_COMMIT[:7]}-cc/manifest.json',
             SNERL_NT_COMMIT: f'runs/remote/acceptance/{SNERL_NT_COMMIT[:7]}-cc/manifest.json'}
prepared = {}  # commit -> evidence
launch_cache = {}  # (commit, gpu) -> launch provenance
items = []
for item in plan:
    job = item['job']
    commit = release(job)
    if commit not in manifests or not (ROOT / manifests[commit]).is_file():
        print('SKIP (no accepted release)', job['id'], commit[:7])
        continue
    host = {**host_base, 'enabled': True, 'commit': commit, 'image_digest': IMAGE,
            'evidence_manifest': manifests[commit]}
    if commit not in prepared:
        evidence = backend.ready(host)
        prepared[commit] = {'evidence': evidence}
    evidence = prepared[commit]['evidence']
    first = next_attempt(job['id'])
    commands = []
    for attempt in range(first, first + int(item['fallback_attempts'])):
        per_gpu = {}
        for gpu in GPUS:  # one command per GPU: the acceptance binding includes that GPU's isolation evidence
            gbackend = RemoteBackend(ROOT, OUT / 'specs-runs' / gpu[4:12])
            _, spec = gbackend.prepare(job, gpu, attempt, host, evidence)
            spec = RL.validate_spec(json.loads(json.dumps(spec)))
            key = (commit, gpu)
            if key not in launch_cache:
                artifacts, data_identity, payloads = RL.evidence_artifacts(spec, client.main_repo)
                proof_path = f"{RL.REMOTE_ROOT}/releases/{commit}/provenance.json"
                local_proof = client.state_dir / 'proofs' / (commit + '.json')
                client.rsync_download(proof_path, local_proof, wait=False)
                origin_commit = RL.validate_release_proof(json.loads(local_proof.read_text()), commit)
                acceptance = f"{RL.REMOTE_ROOT}/runtime/acceptance/{commit}-{IMAGE[7:]}"
                checksums = {}
                for path, name, digest in artifacts:
                    client.rsync_upload(path, f"{acceptance}/binding-{name}", wait=False)
                    checksums[f"binding-{name}"] = digest
                    extracted = client.state_dir / 'acceptance' / f"{commit}-{IMAGE[7:]}" / name
                    extracted.parent.mkdir(parents=True, exist_ok=True)
                    extracted.write_text(json.dumps(payloads[name], indent=2, sort_keys=True) + "\n")
                    client.rsync_upload(extracted, f"{acceptance}/{name}", wait=False)
                    checksums[name] = RL.sha256(extracted)
                data_files = [(f['target'], f['sha256']) for f in payloads['data.json']['files']]
                launch_cache[key] = (data_identity, origin_commit, RL.sha256(local_proof), checksums, data_files)
            data_identity, origin_commit, proof_sha, checksums, data_files = launch_cache[key]
            command = RL.build_remote_command(spec, data_identity, origin_commit, proof_sha, checksums, data_files)
            name = f"{job['id']}-a{attempt}-{gpu[4:12]}.sh"
            (OUT / 'commands' / name).write_text(command + "\n")
            per_gpu[gpu] = f'commands/{name}'
        commands.append({'attempt': attempt, 'container': spec['container'], 'commands': per_gpu})
    items.append({'id': job['id'], 'commit': commit, 'load': load(job), 'ram_gb': float(job.get('ram_gb', 0)),
                  'mem_gb': float(job['mem_gb']), 'disk_gb': float(job.get('disk_gb', 0)),
                  'deps': item['deps'], 'dep_results': item['dep_results'], 'note': item['note'],
                  'require_files': item['require_files'],
                  'required_results': job['required_results'],
                  'run_dir': f"runs/{job['id']}", 'attempts': commands,
                  'companion_of': job['id'].removesuffix('-eval') if job['id'].endswith('-eval') else None})
loads = {}
for j in queue['jobs']:
    if j.get('host') == 'remote' or j['id'] in chosen:
        loads[j['id']] = load(j)
doc = {'version': 1, 'generated': datetime.now().astimezone().isoformat(timespec='seconds'),
       'purpose': __doc__.split('\n\n')[0], 'placeholder_gpu': PLACEHOLDER, 'gpus': list(GPUS),
       'items': items, 'loads': loads}
(OUT / 'plan.json').write_text(json.dumps(doc, indent=1) + "\n")
print(json.dumps({'items': len(items), 'first': [i['id'] for i in items[:8]],
                  'releases': sorted({i['commit'][:7] for i in items})}))
