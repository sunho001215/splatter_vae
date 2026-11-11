"""Fast-forward main to a pushed candidate and adopt LOCAL full-suite evidence (user directive 2026-11-11 ~20:15:
unified pool; the remote is held for the MIG reboot, so its typed native suite + isolation jobs cannot run).

Same test-evidence semantics as every adoption (scripts/_bootstrap.require_passed_tests, the local drivers' check and
the daemon re-exec rule): a COMPLETE scripts/run_tests.py run over the whole tests/ tree with zero failed / error /
skipped, exit code 0, sources unchanged during the run, and its source_sha256 equal to the fingerprint of main AFTER the
fast-forward. Only the host differs: the suite ran on ONE authorized local MIG slice (CUDA_VISIBLE_DEVICES=<MIG UUID>) in
a clean detached worktree of the candidate (established local route: merge_adopt_local_20261106.py, here for the single
`main` branch). Nothing about the REMOTE gate changes: remote rows keep running only releases with a remote acceptance
manifest (release_chain.py: remote native suite + four isolation jobs); run release_chain.py for this commit once the
user lifts the remote MIG hold. This tool never creates, edits or removes any remote hold (runs/setup/remote-hold-admissions,
runs/remote/verification/HOLD, experiments/queue.yaml hosts.remote) and asserts they are byte-identical afterwards.

Atomic w.r.t. the drivers (local admission lock + remote driver lock). Fail-safe: on a fingerprint mismatch the evidence
is NOT adopted (the local drivers then keep refusing admission until resolved); main stays at the merged commit.

    .venv/bin/python -I runs/setup/tools/merge_adopt_local_main_20261111.py <candidate sha (40 hex)> \
        <expected main HEAD sha (40 hex)> <evidence worktree> [--dry-run]

The evidence worktree is normally checked out at the candidate itself. If bookkeeping commits were put on top of the
tested commit (e.g. runs/setup/tools registrars), the worktree may be at an ancestor E of the candidate provided that
`git diff E <candidate>` touches no fingerprinted path (s4d/ ops/ scripts/ tests/ configs/ mujoco_mig_setup.py
.gitmodules third_party/); the post-merge fingerprint equality below remains the deciding check either way.
"""
import fcntl
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/home/ws/ws/splatter4d')
commit, base, worktree = sys.argv[1], sys.argv[2], Path(sys.argv[3]).resolve()
DRY = '--dry-run' in sys.argv
assert re.fullmatch(r'[0-9a-f]{40}', commit) and re.fullmatch(r'[0-9a-f]{40}', base), 'full 40-hex shas required'
HOLDS = [ROOT / 'runs/setup/remote-hold-admissions', ROOT / 'runs/remote/verification/HOLD']
sys.path.insert(0, str(ROOT))
from ops.resources.gpu_guard import APPROVED_HOST_GPUS  # noqa: E402  (authorized local MIG UUIDs)


def git(*args, cwd=ROOT, check=True):
    return subprocess.run(['git', '-C', str(cwd), *args], capture_output=True, text=True, check=check).stdout.strip()


def holds_snapshot():
    import yaml
    snap = {str(p): (hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None) for p in HOLDS}
    snap['queue.hosts.remote'] = json.dumps(yaml.safe_load((ROOT / 'experiments/queue.yaml').read_text())
                                            .get('hosts', {}).get('remote'), sort_keys=True)
    return snap


# ---- evidence checks (before any lock or change)
evidence_path = worktree / 'docs/tests.json'
raw = evidence_path.read_bytes()
evidence = json.loads(raw)
tested = git('rev-parse', 'HEAD', cwd=worktree)
if tested != commit:  # bookkeeping on top of the tested commit: no fingerprinted path may differ
    assert subprocess.run(['git', '-C', str(ROOT), 'merge-base', '--is-ancestor', tested, commit]).returncode == 0, \
        'evidence commit is not an ancestor of the candidate'
    changed = git('diff', '--name-only', tested, commit).splitlines()
    fingerprinted = [c for c in changed if c.split('/')[0] in ('s4d', 'ops', 'scripts', 'tests', 'configs', 'third_party')
                     or c in ('mujoco_mig_setup.py', '.gitmodules')]
    assert not fingerprinted, f'candidate changes tested sources since the evidence commit: {fingerprinted[:8]}'
status = subprocess.run(['git', '-C', str(worktree), 'status', '--porcelain', '--ignore-submodules=none'],
                        capture_output=True, text=True, check=True).stdout
dirty = [line for line in status.splitlines() if not line[3:].startswith('docs/')]
assert not dirty, f'evidence worktree not clean outside docs/: {dirty[:5]}'
counts = evidence.get('counts', {})
assert evidence.get('full_suite_passed') is True and evidence.get('exit_code') == 0, 'suite did not pass'
assert not any(counts.get(k, 1) for k in ('failed', 'error', 'skipped')) and counts.get('passed', 0) > 0, counts
assert evidence.get('source_changed_during_suite') is False, 'sources changed during the suite'
assert evidence.get('pytest_args', [None])[0] == str(worktree / 'tests'), 'not a full run over tests/'
assert evidence.get('host') == 'local', evidence.get('host')
mapping = evidence.get('gpu_mapping') or []
assert len(mapping) == 1 and mapping[0].get('uuid') in APPROVED_HOST_GPUS['local'], mapping  # one authorized MIG slice
assert evidence.get('expected_commit') in (None, tested), evidence.get('expected_commit')
git('fetch', '-q', 'origin', 'main')
assert git('rev-parse', 'refs/remotes/origin/main') == commit, 'origin/main is not the candidate (push it first)'
assert subprocess.run(['git', '-C', str(ROOT), 'merge-base', '--is-ancestor', base, commit]).returncode == 0
before_holds = holds_snapshot()
print('evidence OK', counts, 'gpu', mapping[0]['uuid'], 'duration', round(evidence.get('duration_seconds', 0)), 's',
      flush=True)
if DRY:
    print('DRY RUN: no lock taken, nothing merged or adopted; holds', before_holds, flush=True)
    raise SystemExit(0)

with open(ROOT / 'runs/setup/stage2-driver.lock', 'w') as local_lock, \
        open(ROOT / 'runs/setup/remote-driver.lock', 'w') as remote_lock:
    t0 = time.time()
    fcntl.flock(local_lock, fcntl.LOCK_EX)
    fcntl.flock(remote_lock, fcntl.LOCK_EX)
    print(f'locks acquired after {time.time() - t0:.0f}s', flush=True)
    assert git('symbolic-ref', '--short', 'HEAD') == 'main'
    assert git('rev-parse', 'HEAD') == base, git('rev-parse', 'HEAD')
    git('merge', '--ff-only', '-q', commit)
    assert git('rev-parse', 'HEAD') == commit
    print('merged', commit[:7], flush=True)
    spec = importlib.util.spec_from_file_location('_bootstrap_main', ROOT / 'scripts/_bootstrap.py')
    sys.path.insert(0, str(ROOT / 'scripts'))
    boot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(boot)
    current = boot.source_fingerprints()
    if current != evidence.get('source_sha256'):
        diff = sorted(k for k in set(current) | set(evidence.get('source_sha256', {}))
                      if current.get(k) != evidence['source_sha256'].get(k))
        raise SystemExit(f'fingerprint mismatch after merge ({len(diff)} files, e.g. {diff[:8]}); evidence NOT adopted; '
                         f'main is at {commit[:7]}; local admissions stay refused until resolved')
    docs = ROOT / 'docs'
    shutil.copy2(docs / 'tests.json', docs / f"tests.before-adoption-{time.strftime('%Y%m%d-%H%M%S')}.json")
    adopted = {**evidence, 'adopted_evidence': {
        'host': 'local', 'commit': commit, 'tested_commit': tested, 'worktree': str(worktree), 'gpu': mapping[0]['uuid'],
        'report_sha256': hashlib.sha256(raw).hexdigest(), 'adopted_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'policy': 'user directive 2026-11-11 ~20:15 (unified pool; remote held for MIG reboot): the full suite of the '
                  'exact sources on one authorized local MIG slice is the LOCAL launch-gate evidence (same semantics as '
                  'require_passed_tests); the remote acceptance (release_chain.py) is still required before remote rows '
                  'use this commit'}}
    tmp = docs / 'tests.json.adopt.tmp'
    tmp.write_text(json.dumps(adopted, indent=2) + '\n')
    for name in ('tests.log', 'tests.junit.xml'):
        if (worktree / 'docs' / name).is_file():
            shutil.copy2(worktree / 'docs' / name, docs / name)
    tmp.replace(docs / 'tests.json')
    assert git('rev-parse', '--abbrev-ref', 'main@{upstream}') == 'origin/main'
    assert holds_snapshot() == before_holds, 'a remote hold changed during the adoption window'
    print('adopted local evidence', counts, 'at', time.strftime('%T'), '; remote holds unchanged', flush=True)
