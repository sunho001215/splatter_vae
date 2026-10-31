"""Assemble the campaign acceptance manifest for a release from its synced typed-verification evidence.

Inputs (all checksum-verified closeouts of the scheduler): runs/verify-native-<short>/evidence/tests.json and
runs/verify-isolation-<i>-<short>/evidence/gpu_isolation.json for the four authorized remote GPUs; the accepted
development data identity; the cross-host record carried over unchanged from 226d8ae (the 2026-10-15 20:09 user
decision removed cross-host equivalence as a placement/acceptance criterion; the record is labelled as such and is
not a new measurement). The result is validated with s4d.remote_jobs.acceptance_manifest before it is written.

    python -I assemble_remote_acceptance.py <commit>
"""
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path('/home/ws/ws/splatter4d')
sys.path.insert(0, str(ROOT))
from ops.resources.gpu_guard import APPROVED_HOST_GPUS  # noqa: E402
from ops.jobs.remote_backend import acceptance_manifest  # noqa: E402
from ops.jobs import scheduler as jobs  # noqa: E402

IMAGE = 'sha256:c3ffea7f7c114dd12755f9d537f760a6d5b31c96c923233185b775e0324799b7'
commit = sys.argv[1]
if '--image' in sys.argv:  # 2026-10-23: a release on another image (job ids / directory carry --tag)
    IMAGE = sys.argv[sys.argv.index('--image') + 1]
    assert len(IMAGE) == 71 and IMAGE.startswith('sha256:')
TAG = sys.argv[sys.argv.index('--tag') + 1] if '--tag' in sys.argv else ''
short = commit[:7] + (f'-{TAG}' if TAG else '')
out = ROOT / 'runs/remote/acceptance' / short
out.mkdir(parents=True, exist_ok=True)
state = jobs.read_registry()


def verified(job_id, name):
    current = state[job_id]
    assert current['status'] == jobs.DONE and current['commit'] == commit and current['image_digest'] == IMAGE, job_id
    assert current['sync']['final'] is True and current['sync']['verified'] is True, job_id
    path = ROOT / 'runs' / job_id / 'evidence' / name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == current['sync']['files'][f'evidence/{name}'], job_id
    return json.loads(path.read_text())


def write(name, payload):
    path = out / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    return {'path': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


suite = verified(f'verify-native-{short}', 'tests.json')
manifest = {'commit': commit, 'image_digest': IMAGE,
            'full_suite': write('tests.json', {'commit': commit, 'image_digest': IMAGE, 'report': suite,
                                               'job': f'verify-native-{short}'})}
isolation = {}
for index, gpu in enumerate(APPROVED_HOST_GPUS['remote']):
    report = verified(f'verify-isolation-{index}-{short}', 'gpu_isolation.json')
    isolation[gpu] = write(f'isolation-{gpu}.json', {'commit': commit, 'image_digest': IMAGE, 'report': report,
                                                     'job': f'verify-isolation-{index}-{short}'})
manifest['gpu_isolation'] = isolation
data = json.loads((ROOT / 'runs/remote/acceptance/226d8ae/data.json').read_text())
manifest['data'] = write('data.json', {**data, 'commit': commit, 'image_digest': IMAGE,
                                       'carried_over_from': '226d8ae (same files, same checksums, same remote paths)'})
cross = json.loads((ROOT / 'runs/remote/acceptance/226d8ae/cross_host.json').read_text())
manifest['cross_host'] = write('cross_host.json', {
    **cross, 'commit': commit, 'image_digest': IMAGE,
    'carried_over_again': {'from': '226d8ae', 'recorded_at': datetime.now().astimezone().isoformat(),
                           'basis': 'user decision 2026-10-15 20:09: hosts are equivalent; cross-host equivalence is '
                                    'no longer a placement or acceptance criterion. Historical e267ba8 record kept '
                                    'unchanged for the validator; not a new measurement.'}})
path = out / 'manifest.json'
path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
checked = acceptance_manifest({'commit': commit, 'image_digest': IMAGE,
                               'evidence_manifest': str(path.relative_to(ROOT))}, ROOT)
print('ACCEPTED', path, checked['manifest_sha256'], checked['data_identity'][:12])
