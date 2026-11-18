"""Adopt a pushed local MIG16 candidate and its complete local suite evidence.

Same gates as merge_adopt_local_mig_20261112.py; validate GPU authorization using
this candidate, and leave the entire queue (including remote limits) untouched.
This tool is prepared for the coordinator; implementation never runs it.

2026-11-18 (user-approved release gate): evidence whose release_gate accepted a SELECTIVE run is adopted after
scripts/test_gate.adoption_check re-verifies it (same checkout and candidate commit, ancestor baseline with an untampered
record, no high-risk/unclassified change, every re-selected test covered, the candidate's own check); a complete run
keeps every check below, including exact shard coverage of tests/.
A descendant candidate may differ from the tested HEAD only in the three docs/tests evidence files;
the original HEAD is retained and re-verified, with candidate source equality proved separately.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from ops.resources.gpu_guard import APPROVED_HOST_GPUS  # noqa: E402


def git(root, *args):
    return subprocess.run(['git', '-C', str(root), *args], capture_output=True, text=True, check=True).stdout.rstrip()


EVIDENCE_FILES = ('docs/tests.json', 'docs/tests.junit.xml', 'docs/tests.log')


def evidence_commit(worktree: Path, commit: str, evidence: dict) -> str:
    """Allow evidence commits only after proving the executed code tree is identical.

    Do not rewrite report HEADs: the unchanged gate re-verifies the actual tested
    commit, and this independent proof binds it to the publication candidate.
    """
    produced = evidence.get('head') or commit  # compatibility with legacy complete reports
    if not re.fullmatch('[0-9a-f]{40}', produced):
        raise ValueError('invalid evidence HEAD')
    if produced != commit:
        git(worktree, 'merge-base', '--is-ancestor', produced, commit)
        changed = set(git(worktree, 'diff', '--name-only', '--no-renames', produced, commit).splitlines())
        if not changed or not changed <= set(EVIDENCE_FILES):
            raise ValueError('candidate differs from evidence HEAD outside the three evidence files')
        for path in changed:
            entry = git(worktree, 'ls-tree', commit, '--', path)
            if not entry.startswith('100644 blob '):
                raise ValueError('candidate evidence must be regular files')
        from ops.jobs.remote_backend import commit_source_sha256

        if evidence.get('source_sha256') != commit_source_sha256(worktree, commit):
            raise ValueError('evidence sources differ from candidate')
    return produced


@contextlib.contextmanager
def preserve_main_evidence(root: Path):
    """Back up all three files before cleaning only their unstaged changes for ff-merge.

    Staged evidence is refused: leave the coordinator's index intact. On any
    failure restore original bytes, mode and missing-file state. Successful
    adoption retains a durable backup of all three files under docs/.
    """
    paths = [root / name for name in EVIDENCE_FILES]
    if any(path.is_symlink() or path.parent.is_symlink() for path in paths):
        raise ValueError('refusing symlinked main evidence')
    if git(root, 'diff', '--cached', '--name-only', '--', *EVIDENCE_FILES):
        raise ValueError('main has staged evidence; preserve the index before adoption')
    backup = Path(tempfile.mkdtemp(prefix='tests.before-adoption-', dir=root / 'docs'))
    present = [path.exists() for path in paths]
    for path, exists in zip(paths, present):
        if exists:
            shutil.copy2(path, backup / path.name)
    (backup / 'manifest.json').write_text(json.dumps(dict(zip(EVIDENCE_FILES, present))) + '\n')
    try:
        git(root, 'restore', '--source=HEAD', '--worktree', '--', *EVIDENCE_FILES)
        yield backup
    except BaseException:
        for path, exists in zip(paths, present):
            if exists:
                shutil.copy2(backup / path.name, path)
            else:
                path.unlink(missing_ok=True)
        raise


def verify(worktree: Path, commit: str) -> dict:
    if git(worktree, 'rev-parse', 'HEAD') != commit:
        raise ValueError('evidence worktree must be at candidate')
    dirty = git(worktree, 'status', '--porcelain', '--ignore-submodules=none')
    if any(not line[3:].startswith('docs/') for line in dirty.splitlines()):
        raise ValueError('evidence worktree is dirty outside docs/')
    evidence = json.loads((worktree / 'docs/tests.json').read_bytes())
    produced = evidence_commit(worktree, commit, evidence)
    counts = evidence.get('counts', {})
    if (evidence.get('full_suite_passed') is not True or evidence.get('exit_code') != 0
            or any(counts.get(k, 1) for k in ('failed', 'error', 'skipped')) or counts.get('passed', 0) < 1
            or evidence.get('source_changed_during_suite') is not False or evidence.get('host') != 'local'
            or evidence.get('partition') not in (None, 'all')
            or evidence.get('expected_commit') not in (None, produced, commit)):
        raise ValueError('not passing complete local evidence')
    gate = evidence.get('release_gate')
    if gate is not None and gate.get('accepted') is not True:
        raise ValueError('release gate refused the evidence')
    if gate is not None and gate.get('mode') == 'selective':
        sys.path.insert(0, str(REPO / 'scripts'))
        import test_gate  # the adopting checkout's gate (scripts/test_gate.py)

        reasons = test_gate.adoption_check(evidence, worktree, produced)
        if reasons:
            raise ValueError('selective evidence refused: ' + '; '.join(reasons))
        mapping = evidence.get('gpu_mapping') or []
        if len(mapping) != 1 or mapping[0].get('uuid') not in APPROVED_HOST_GPUS['local']:
            raise ValueError('suite GPU not authorized by candidate')
        return evidence
    args = evidence.get('pytest_args') or [None]
    if isinstance(args[0], list):
        files = {a for shard in args for a in shard if a.startswith('tests/') and a.endswith('.py')}
        if files != {str(p.relative_to(worktree)) for p in (worktree / 'tests').rglob('test_*.py')}:
            raise ValueError('shards do not cover exactly tests/')
    elif args[0] != str(worktree / 'tests'):
        raise ValueError('not a full run over tests/')
    mapping = evidence.get('gpu_mapping') or []
    if len(mapping) != 1 or mapping[0].get('uuid') not in APPROVED_HOST_GPUS['local']:
        raise ValueError('suite GPU not authorized by candidate')
    return evidence


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('commit')
    p.add_argument('expected_main')
    p.add_argument('worktree', type=Path)
    p.add_argument('--root', type=Path, default=Path('/home/ws/ws/splatter4d'))
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if not all(re.fullmatch('[0-9a-f]{40}', value) for value in (a.commit, a.expected_main)):
        p.error('full 40-hex SHAs required')
    evidence = verify(a.worktree.resolve(), a.commit)
    git(a.root, 'fetch', '-q', 'origin', 'main')
    if git(a.root, 'rev-parse', 'refs/remotes/origin/main') != a.commit:
        p.error('origin/main must be the pushed candidate')
    git(a.root, 'merge-base', '--is-ancestor', a.expected_main, a.commit)
    if a.dry_run:
        print('DRY RUN: evidence OK', evidence['counts'])
        return
    with (a.root / 'runs/setup/stage2-driver.lock').open('a') as local, \
            (a.root / 'runs/setup/remote-driver.lock').open('a') as remote:
        fcntl.flock(local, fcntl.LOCK_EX)
        fcntl.flock(remote, fcntl.LOCK_EX)
        if git(a.root, 'symbolic-ref', '--short', 'HEAD') != 'main' or git(a.root, 'rev-parse', 'HEAD') != a.expected_main:
            p.error('main moved; integration and new evidence required')
        tracked = [a.root / 'experiments/queue.yaml', a.root / 'runs/setup/remote-hold-admissions',
                   a.root / 'runs/remote/verification/HOLD']
        before = {path: path.read_bytes() if path.exists() else None for path in tracked}
        # Recheck under the adoption locks before changing main's evidence.
        evidence = verify(a.worktree.resolve(), a.commit)
        with preserve_main_evidence(a.root):
            git(a.root, 'merge', '--ff-only', a.commit)
            # Match the post-merge source fingerprint exactly, as the existing adoption tool does.
            code = ('import sys; sys.path.insert(0,"scripts"); import _bootstrap; import json; '
                    'print(json.dumps(_bootstrap.source_fingerprints()))')
            current = json.loads(subprocess.run([str(a.root / '.venv/bin/python'), '-c', code], cwd=a.root,
                                                capture_output=True, text=True, check=True).stdout)
            if current != evidence['source_sha256']:
                raise ValueError('post-merge source mismatch: evidence NOT adopted; admissions remain held')
            docs = a.root / 'docs'
            stamp = time.strftime('%Y%m%d-%H%M%S')
            report = {**evidence, 'adopted_evidence': {'host': 'local', 'commit': a.commit,
                      'worktree': str(a.worktree.resolve()), 'adopted_at': stamp,
                      'report_sha256': hashlib.sha256((a.worktree / 'docs/tests.json').read_bytes()).hexdigest(),
                      'policy': 'user 2026-11-15 temporary local MIG16 expansion; '
                                + ('selective run accepted by the release gate (user 2026-11-17)'
                                   if (evidence.get('release_gate') or {}).get('mode') == 'selective'
                                   else 'complete local native suite')}}
            tmp = docs / 'tests.json.adopt.tmp'
            tmp.write_text(json.dumps(report, indent=2) + '\n')
            for name in ('tests.log', 'tests.junit.xml'):
                shutil.copy2(a.worktree / 'docs' / name, docs / name)
            tmp.replace(docs / 'tests.json')
        if any((path.read_bytes() if path.exists() else None) != data for path, data in before.items()):
            raise ValueError('queue or remote hold changed during adoption')
        print('adopted', a.commit, evidence['counts'])


if __name__ == '__main__':
    main()
