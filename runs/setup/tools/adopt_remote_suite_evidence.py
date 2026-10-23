"""Adopt a passing REMOTE full native suite as the repository's test evidence (docs/tests.json) for identical sources
(user instruction 2026-10-23: use remote RAM for the remaining validation; transparently revise local-only validation
requirements; preserve accurate test provenance).

The launch gate (scripts/_bootstrap.py require_passed_tests) requires a complete passing suite for the exact sources. The
local host cannot reserve the suite's ~40 GB headroom while its pretraining runs, so the same full suite run on the
remote host (typed verification job in an accepted image, with the job's checksum-verified result sync) is adopted when:
- the verification job is done, its final sync verified and the report's checksum equals the synced one;
- the report passed in full (no failure, error or skip) and its source fingerprints equal this checkout's
  (``source_fingerprints()``: s4d, scripts, tests, configs, mujoco_mig_setup.py, third-party pins).
The written evidence keeps the report's own fields and adds an ``adopted_evidence`` block: host, job, commit, image,
report checksum, the local runs given with --local (e.g. the changed test module), and this policy. The previous
docs/tests.json is kept as docs/tests.before-adoption-<time>.json.
    python -I adopt_remote_suite_evidence.py <verify job id> [--local "<test module>: <result>"] [--dry-run]
"""
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path('/home/ws/ws/splatter4d')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from scripts import jobs  # noqa: E402
from _bootstrap import source_fingerprints  # noqa: E402

job_id = sys.argv[1]
local = [sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == '--local']
state = jobs.read_registry()[job_id]
assert state['status'] == jobs.DONE and state['host'] == 'remote', state['status']
assert state['sync']['final'] is True and state['sync']['verified'] is True
path = ROOT / 'runs' / job_id / 'evidence' / 'tests.json'
digest = hashlib.sha256(path.read_bytes()).hexdigest()
assert digest == state['sync']['files']['evidence/tests.json']
report = json.loads(path.read_text())
counts = report.get('counts', {})
assert report.get('full_suite_passed') is True and report.get('exit_code') == 0 and counts.get('passed', 0) > 0
assert not any(counts.get(k, 1) for k in ('failed', 'error', 'skipped')), counts
here = source_fingerprints()
if report.get('source_sha256') != here:
    differ = sorted(k for k in set(here) | set(report.get('source_sha256', {})) if here.get(k) != report['source_sha256'].get(k))
    raise SystemExit(f'source fingerprints differ from this checkout: {differ[:10]}')
evidence = {**report, 'adopted_evidence': {
    'host': 'remote', 'job': job_id, 'commit': state['commit'], 'image_digest': state['image_digest'],
    'report': str(path.relative_to(ROOT)), 'report_sha256': digest, 'gpu': state.get('gpu'),
    'adopted_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'local_runs': local,
    'policy': 'user instruction 2026-10-23: remote RAM for the remaining validation; the full native suite of the exact '
              'sources on the remote host is accepted as the launch-gate evidence when the local host lacks the suite '
              'headroom; provenance kept here (not a local run)'}}
if '--dry-run' in sys.argv:
    print(json.dumps({'ok': True, 'counts': counts, 'job': job_id}))
    raise SystemExit(0)
docs = ROOT / 'docs'
if (docs / 'tests.json').is_file():
    shutil.copy2(docs / 'tests.json', docs / f"tests.before-adoption-{time.strftime('%Y%m%d-%H%M%S')}.json")
for name in ('tests.log', 'tests.junit.xml'):
    src = ROOT / 'runs' / job_id / 'evidence' / name
    if src.is_file():
        shutil.copy2(src, docs / name)
tmp = docs / 'tests.json.tmp'
tmp.write_text(json.dumps(evidence, indent=2) + '\n')
tmp.replace(docs / 'tests.json')
print('ADOPTED', job_id, digest[:12], counts)
