"""Prepare typed remote verification (s4d/remote_verification.py) for a pushed release commit on the accepted image.

Writes, under runs/remote/acceptance/<short>/:
  binding-data.json             the accepted eight-file development data identity, bound to this commit/image
  verification-native_suite.json
  verification-gpu_isolation.json (only with --suite <synced tests.json>: binds the passing remote suite report)
and a scoped queue under runs/remote/verification/ holding the verification job(s). The scheduler admits them only
through RemoteBackend.ready_verification; campaign acceptance is never replaced by these files.

    python -I prepare_remote_verification.py <commit> [--suite runs/<verify job>/evidence/tests.json]
        [--image <digest> --tag <suffix>]   (2026-10-23: a release on another image, e.g. the derived image with a C
                                             compiler; job ids and the acceptance directory carry the suffix)
"""
import hashlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path('/home/ws/ws/splatter4d')
sys.path.insert(0, str(ROOT))
from ops.resources.gpu_guard import APPROVED_HOST_GPUS  # noqa: E402
from ops.jobs.remote_backend import commit_source_sha256  # noqa: E402
from ops.containers.verification import POLICIES, validate_development_data, validate_report  # noqa: E402

IMAGE = 'sha256:c3ffea7f7c114dd12755f9d537f760a6d5b31c96c923233185b775e0324799b7'
if '--image' in sys.argv:
    IMAGE = sys.argv[sys.argv.index('--image') + 1]
    assert len(IMAGE) == 71 and IMAGE.startswith('sha256:')
TAG = sys.argv[sys.argv.index('--tag') + 1] if '--tag' in sys.argv else ''
DATA = ROOT / 'runs/remote/acceptance/226d8ae/data.json'
commit = sys.argv[1]
assert len(commit) == 40
suite_path = Path(sys.argv[sys.argv.index('--suite') + 1]) if '--suite' in sys.argv else None
short = commit[:7] + (f'-{TAG}' if TAG else '')
out = ROOT / 'runs/remote/acceptance' / short
out.mkdir(parents=True, exist_ok=True)
source = commit_source_sha256(ROOT, commit)


def write(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    return {'path': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


report = json.loads(DATA.read_text())
validate_development_data(report)
data_ref = write(out / 'binding-data.json', {'commit': commit, 'image_digest': IMAGE, 'report': report})
base = {'schema_version': 1, 'campaign_eligible': False, 'commit': commit, 'image_digest': IMAGE,
        'source_sha256': source, 'data': data_ref}
write(out / 'verification-native_suite.json', {**base, 'verification_purpose': 'native_suite'})
manifests = {'native_suite': str((out / 'verification-native_suite.json').relative_to(ROOT))}
gpus = list(APPROVED_HOST_GPUS['remote'])
jobs_out = []
if suite_path is None:
    jobs_out.append({'id': f'verify-native-{short}', 'host': 'remote', 'gpu': gpus[0], 'script': POLICIES['native_suite'][0],
                     'args': [], 'verification_purpose': 'native_suite', 'campaign_eligible': False,
                     'ram_gb': 32, 'mem_gb': 12, 'disk_gb': 1, 'priority': 0,
                     'required_results': ['evidence/' + n for n in POLICIES['native_suite'][4]]})
else:
    suite = json.loads(suite_path.read_text())
    validate_report('native_suite', suite, source, commit, IMAGE, '')
    suite_ref = write(out / 'binding-tests.json', {'commit': commit, 'image_digest': IMAGE, 'report': suite})
    write(out / 'verification-gpu_isolation.json', {**base, 'verification_purpose': 'gpu_isolation',
                                                    'full_suite': suite_ref})
    manifests['gpu_isolation'] = str((out / 'verification-gpu_isolation.json').relative_to(ROOT))
    for index, gpu in enumerate(gpus):
        jobs_out.append({'id': f'verify-isolation-{index}-{short}', 'host': 'remote', 'gpu': gpu,
                         'script': POLICIES['gpu_isolation'][0], 'args': [], 'verification_purpose': 'gpu_isolation',
                         'campaign_eligible': False, 'ram_gb': 8, 'mem_gb': 4, 'disk_gb': 1, 'priority': 0,
                         'required_results': ['evidence/' + n for n in POLICIES['gpu_isolation'][4]]})
queue = yaml.safe_load((ROOT / 'experiments/queue.yaml').read_text())
host = dict(queue['hosts']['remote'])
host.update(enabled=True, commit=commit, image_digest=IMAGE, verification_manifests=manifests)
host.pop('evidence_manifest', None)
qdir = ROOT / 'runs/remote/verification'
qpath = qdir / f"{'isolation' if suite_path else 'native'}-{short}-queue.yaml"
qpath.write_text(yaml.safe_dump({'hosts': {'local': {'enabled': False}, 'remote': host}, 'jobs': jobs_out},
                                sort_keys=False))
print(qpath)
