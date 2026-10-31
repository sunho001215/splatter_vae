"""Migrate the checkpoint-paused self-render v4 pretrainings srv4-csw4-seed0-{hammer,pick-place}-200k to the remote for
the local outage of 2026-10-31 23:00 KST (user authorization 2026-10-31: migrate a local job when code, data,
dependencies and resume state can be made available and verified before the deadline; preserve the raised priority of
the self-render pilot).

--upload : upload checkpoints/latest.pt (its step must equal the paused step) and the paused step file (sha256 must
           equal the recorded intentional-stop checkpoint), config.yaml,
           provenance.json, console.log, log.txt and metrics.jsonl to ~/kaist/sunho/runs/<id>/ and verify each
           remote sha256; record runs/setup/srv4-migration-upload-20261031.json. W&B offline and eval/ folders stay
           local (not needed to resume: scripts/train.py --resume auto reads only checkpoints/latest.pt).
--migrate: under the admission and registry locks, for every uploaded run: a `migrated` registry event (host remote,
           release e8ad5ac = the sources the run was launched with: the local launch commit 02db420 changed only
           docs/tools relative to e8ad5ac) and pool placement of the pretraining, its export and its six FlashSAC
           rows to the remote (host fields only; pool_placement.placed_row).
"""
import argparse
import fcntl
import hashlib
import json
import os
import subprocess
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

SRV4_COMMIT = 'e8ad5ac17de1434d85256527d5a42e7e10128b70'
IMAGE = 'sha256:59082834c2c2c0d688be50a0b3b362bc46bdd2b1923d8ab712a473a0bd51870a'
TASKS = ('hammer', 'pick-place')
FILES = ('checkpoints/latest.pt', 'config.yaml', 'console.log', 'log.txt', 'metrics.jsonl')
# provenance.json goes up as provenance.local-20261031.json: the remote entry (s4d.run_identity.start_remote_run)
# refuses a run directory whose provenance.json names another host/commit; the remote continuation writes its own.
RECORD = ROOT / 'runs/setup/srv4-migration-upload-20261031.json'
ap = argparse.ArgumentParser()
ap.add_argument('--upload', action='store_true')
ap.add_argument('--migrate', action='store_true')
ap.add_argument('--only', nargs='*', default=None)
args = ap.parse_args()


def ckpt_step(path):
    import itertools, pickletools, zipfile
    with zipfile.ZipFile(path) as archive:
        name = next(n for n in archive.namelist() if n.endswith('/data.pkl'))
        ops = list(itertools.islice(pickletools.genops(archive.read(name)), 7))
    assert ops[-3][1] == 'step'
    return ops[-1][1]


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


ids = [f'srv4-csw4-seed0-{t}-200k' for t in TASKS if args.only is None or t in args.only]
record = json.loads(RECORD.read_text()) if RECORD.is_file() else {}
if args.upload:
    client = RemoteClient()
    state = jobs.read_registry()
    for job_id in ids:
        cur = state[job_id]
        stops = [s for s in cur.get('intentional_stops', []) if s.get('attempt') == cur.get('attempt')]
        assert stops and cur['status'] not in jobs.ACTIVE, (job_id, cur['status'])
        stop = stops[-1]
        run = ROOT / 'runs' / job_id
        step_file = f"checkpoints/step_{int(stop['checkpoint_step']):07d}.pt"
        assert sha(run / step_file) == stop['checkpoint_sha256'], job_id  # the pause record names the step file
        assert ckpt_step(run / 'checkpoints/latest.pt') == stop['checkpoint_step'], job_id  # same step, own archive name
        files = {}
        for rel in FILES + (step_file,):
            client.rsync_upload(run / rel, f'~/kaist/sunho/runs/{job_id}/{rel}')
            files[rel] = sha(run / rel)
        client.rsync_upload(run / 'provenance.json', f'~/kaist/sunho/runs/{job_id}/provenance.local-20261031.json')
        files['provenance.local-20261031.json'] = sha(run / 'provenance.json')
        listing = ' '.join(f'runs/{job_id}/{rel}' for rel in FILES + (step_file, 'provenance.local-20261031.json'))
        out = client.checked_ssh(f'cd /home/compu/kaist/sunho && sha256sum {listing}', timeout=300)
        remote = {line.split()[1].split('/', 2)[2]: line.split()[0] for line in out.stdout.splitlines()}
        assert remote == files, (job_id, remote, files)
        record[job_id] = {'complete': True, 'checkpoint_step': stop['checkpoint_step'],
                          'checkpoint_sha256': stop['checkpoint_sha256'], 'files': files,
                          'uploaded': datetime.now().astimezone().isoformat(timespec='seconds')}
        RECORD.write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps({'uploaded': job_id, 'step': stop['checkpoint_step']}), flush=True)
if args.migrate:
    with open(pool.ADMISSION_LOCK, 'w') as admission:
        fcntl.flock(admission, fcntl.LOCK_EX)
        with jobs.REGISTRY.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            queue = jobs.load_queue()
            by_id = {j['id']: j for j in queue['jobs']}
            state = jobs.read_registry()
            stamp = datetime.now().astimezone().isoformat(timespec='seconds')
            effective, events = {}, []
            for job_id in ids:
                cur = state[job_id]
                if cur.get('host') == 'remote':
                    continue
                stop = [s for s in cur.get('intentional_stops', []) if s.get('attempt') == cur['attempt']][-1]
                up = record.get(job_id, {})
                assert up.get('complete') and up['checkpoint_step'] == stop['checkpoint_step'], job_id
                assert up['files'][f"checkpoints/step_{int(stop['checkpoint_step']):07d}.pt"] == stop['checkpoint_sha256'], job_id
                task = job_id.removeprefix('srv4-csw4-seed0-').removesuffix('-200k')
                related = [job_id, f'srv4-csw4-export-seed0-{task}-200k'] + [
                    f'ds-flashsac-srv4-csw4-{task}-s{s}{e}' for s in (2000, 2001, 2002) for e in ('', '-eval')]
                for rid in related:
                    row = by_id[rid]
                    if row.get('host', 'local') != 'remote':
                        effective[rid] = pool.placed_row(row, 'remote', stamp)
                        assert jobs.job_fingerprint(effective[rid]) == jobs.job_fingerprint(row), rid
                launch = [json.loads(l) for l in jobs.REGISTRY.read_text().splitlines() if f'"{job_id}"' in l]
                launch = [e for e in launch if e['id'] == job_id and e['event'] == 'launched'][-1]
                events.append({'event': 'migrated', 'id': job_id, 'from_host': 'local', 'host': 'remote',
                               'checkpoint_step': stop['checkpoint_step'], 'checkpoint_sha256': stop['checkpoint_sha256'],
                               'from_commit': launch['commit'], 'commit': SRV4_COMMIT, 'image_digest': IMAGE,
                               'sources_identical': True,
                               'source_comparison': 'git diff e8ad5ac..02db420 touches no s4d/, scripts/, tests/, configs/ '
                                                    'or third_party/ file (02db420 = docs/tools only)',
                               'upload': str(RECORD.relative_to(ROOT)),
                               'reason': 'local shutdown 2026-10-31 23:00 KST (user authorization 2026-10-31): the '
                                         'self-render v4 pilot continues on the remote; launched there by '
                                         '~/kaist/sunho/remote_runner while the local server is offline'})
            if not events:
                print(json.dumps({'migrated': []}))
                raise SystemExit(0)
            before = jobs.QUEUE.read_bytes()
            text = before.decode()
            node = next(value for key, value in yaml.compose(text).value if key.value == 'jobs')
            new_jobs = [effective.get(j['id'], j) for j in queue['jobs']]
            staged = ROOT / 'runs/setup/srv4-migration-queue-candidate-20261031.yaml'
            staged.write_text(text[:node.start_mark.index] + yaml.safe_dump(json.loads(json.dumps(new_jobs)), sort_keys=False)
                              + text[node.end_mark.index:])
            assert jobs.load_queue(staged)['jobs'] == new_jobs and jobs.load_queue(staged)['hosts'] == queue['hosts']
            tmp = jobs.QUEUE.with_suffix('.srv4migrate.tmp')
            tmp.write_bytes(staged.read_bytes())
            os.replace(tmp, jobs.QUEUE)
            for event in events:
                jobs.record(event)
            after = jobs.read_registry()
            assert all(after[e['id']]['host'] == 'remote' and after[e['id']]['status'] == jobs.PENDING for e in events)
            manifest = json.loads(pool.MANIFEST.read_text())
            manifest['moves'].append({'time': stamp, 'to': 'remote', 'reason': 'local shutdown 2026-10-31: srv4 '
                                      'pretraining migration (checkpoint pause, upload, migrated event) with its export '
                                      'and FlashSAC rows', 'ids': sorted(effective), 'rewritten_exports': [],
                                      'original': {i: by_id[i] for i in effective},
                                      'queue_sha256_before': hashlib.sha256(before).hexdigest(),
                                      'queue_sha256_after': hashlib.sha256(jobs.QUEUE.read_bytes()).hexdigest()})
            manifest['effective_jobs'].update(effective)
            tmp = pool.MANIFEST.with_suffix('.tmp')
            tmp.write_text(json.dumps(manifest, indent=1) + '\n')
            os.replace(tmp, pool.MANIFEST)
            print(json.dumps({'migrated': [e['id'] for e in events], 'rows': sorted(effective)}))
