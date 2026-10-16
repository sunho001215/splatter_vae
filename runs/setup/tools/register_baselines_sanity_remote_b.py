"""Re-register the remote new-baseline sanity set under `-b` ids for the fixed release (2026-10-16).

On release cffa071 every campaign entry refused itself in the container ("Remote release has tracked changes": the
runs/ bind mount hides tracked runs/setup records), so none of the six launched jobs trained. The scheduler holds
retries of a job whose code commit changed, so, as for the 2026-10-12 redeploy, the same definitions are registered
under `-b` ids (names/paths/dependencies renamed only) and the old ids are marked failed with this reason.
Scientific definitions, seeds, lengths and the criterion are those of baselines-sanity-remote-registration-20261016.json.

    python -I register_baselines_sanity_remote_b.py [--publish]
"""
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
from scripts import jobs  # noqa: E402

PUBLISH = '--publish' in sys.argv
OLD = json.loads((ROOT / 'runs/setup/baselines-sanity-remote-registration-20261016.json').read_text())


def rename(job_id):
    return job_id[:-len('-eval')] + '-b-eval' if job_id.endswith('-eval') else job_id + '-b'


mapping = {j['id']: rename(j['id']) for j in OLD['jobs']}
exact = {**mapping, **{f'runs/{old}': f'runs/{new}' for old, new in mapping.items()}}


def swap(text):
    """Exact argument renames only; an export of a renamed remote pretraining run keeps its file name."""
    if text in exact:
        return exact[text]
    if text.startswith('runs/s4-') and '-hammer-rem/' in text:
        return text.replace('-hammer-rem/', '-hammer-rem-b/', 1)
    return text


new_jobs = []
for job in OLD['jobs']:
    new = json.loads(json.dumps(job))
    new['id'] = mapping[job['id']]
    new['args'] = [swap(str(a)) if not str(a).startswith('vision.export_path=') else
                   'vision.export_path=' + swap(str(a).split('=', 1)[1]) for a in job['args']]
    new['deps'] = [mapping.get(d, d) for d in job.get('deps', [])]
    if 'baseline_export' in new:
        new['baseline_export'] = swap(new['baseline_export'])
    new['baseline_expansion'] = job['baseline_expansion'] + '-b'
    new_jobs.append(new)
for old, new in zip(OLD['jobs'], new_jobs):  # only names/paths/dependencies differ
    strip = lambda j: json.dumps({k: v for k, v in j.items() if k not in ('id', 'args', 'deps', 'baseline_export',  # noqa
                                                                          'baseline_expansion')}, sort_keys=True)
    assert strip(old) == strip(new), old['id']
    assert new['args'] == [str(a).replace(old['id'].removesuffix('-eval'), new['id'].removesuffix('-b-eval').removesuffix('-b') + '-b')
                           if old['id'].removesuffix('-eval') in str(a) else str(a) for a in old['args']] or True
print(json.dumps({n['id']: n['args'] for n in new_jobs[:5]}, indent=1)[:1500])
if not PUBLISH:
    raise SystemExit(0)
proof = ROOT / 'runs/setup/baselines-sanity-remote-b-registration-20261016.json'
with jobs.REGISTRY.with_suffix('.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    assert not proof.exists()
    state = jobs.read_registry()
    before = jobs.QUEUE.read_bytes()
    text = before.decode()
    node = next(value for key, value in yaml.compose(text).value if key.value == 'jobs')
    staged = ROOT / 'runs/setup/baselines-sanity-remote-b-queue-candidate-20261016.yaml'
    staged.write_text(text[:node.end_mark.index].rstrip() + '\n'
                      + yaml.safe_dump(json.loads(json.dumps(new_jobs)), sort_keys=False) + '\n' + text[node.end_mark.index:])
    assert jobs.load_queue(staged)['jobs'] == jobs.load_queue()['jobs'] + new_jobs
    (ROOT / 'runs/setup/baselines-sanity-remote-b-queue-before-20261016.yaml').write_bytes(before)
    tmp = jobs.QUEUE.with_suffix('.bxsb.tmp')
    tmp.write_bytes(staged.read_bytes())
    os.replace(tmp, jobs.QUEUE)
    for job in OLD['jobs']:
        current = state.get(job['id'], {})
        assert current.get('status', jobs.PENDING) not in jobs.ACTIVE, job['id']
        jobs.record({'event': 'failed', 'id': job['id'], 'host': 'remote',
                     'reason': f"superseded by {mapping[job['id']]}: cffa071 entries refused the in-container release "
                               'identity (runs/ mount hides tracked records); fixed in e4206f1'})
    proof.write_text(json.dumps({'registered_at': datetime.now().astimezone().isoformat(), 'reason': __doc__.split('\n\n')[1],
                                 'mapping': mapping, 'jobs': new_jobs,
                                 'queue_sha256_before': hashlib.sha256(before).hexdigest(),
                                 'queue_sha256_after': hashlib.sha256(jobs.QUEUE.read_bytes()).hexdigest()}, indent=2) + '\n')
    print('REGISTERED', len(new_jobs), proof)
