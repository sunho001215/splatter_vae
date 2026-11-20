"""Seal K3's private ffedec5 snapshot. This is preparation, never deployment.

No training, tests, image builds, registry edits, shared locks or pause changes.
Gitlink commits are recorded; absent submodules are not claimed materialized.
Read-only modes and content hashes seal source, not kernel immutable flags.
"""
import hashlib
import json
from pathlib import Path
import stat
import subprocess

ROOT = Path('/home/ws/ws/realworld-ops/k0-20261120/k3')
COMMIT = 'ffedec5464ba10b013f5071603b7bd6bc517b937'
RELEASE = ROOT / 'releases' / COMMIT
CODE = RELEASE / 'code'


def git(*args):
    return subprocess.run(['git', '-C', str(CODE), *args], check=True,
                          capture_output=True, timeout=30).stdout


def main():
    assert CODE.is_dir() and CODE.resolve() == CODE
    assert git('rev-parse', 'HEAD').decode().strip() == COMMIT
    assert not git('status', '--porcelain', '--untracked-files=all')
    files, links = {}, {}
    for entry in git('ls-files', '--stage', '-z').split(b'\0'):
        if not entry:
            continue
        meta, name = entry.split(b'\t', 1)
        mode, blob, stage = meta.decode().split()
        assert stage == '0'
        path = CODE / name.decode()
        if mode == '160000':
            links[name.decode()] = {'gitlink_commit': blob, 'materialized': False}
            continue
        assert not path.is_symlink() and path.is_file(), path
        files[name.decode()] = {'git_blob': blob, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                'bytes': path.stat().st_size}
    # Never follow a symlink or chmod an alternate's target in main's object store.
    paths = list(CODE.rglob('*'))
    for path in paths:
        assert not path.is_symlink(), path
    for path in paths:
        path.chmod(stat.S_IMODE(path.stat().st_mode) & ~0o222)
    CODE.chmod(stat.S_IMODE(CODE.stat().st_mode) & ~0o222)
    payload = {'commit': COMMIT, 'code_root': str(CODE), 'source_files': files, 'gitlinks': links,
               'seal': 'read-only mode bits plus sha256 inventory; no kernel immutable flag',
               'deployed': False, 'scheduler_accepted': False, 'image_created': False,
               'shared_object_store': '/home/ws/ws/splatter4d/.git/objects',
               'limitation': 'Private source snapshot only; runtime acceptance and isolated local routing remain blocked.'}
    output = RELEASE / 'snapshot.json'
    with output.open('x') as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    output.chmod(0o444)
    RELEASE.chmod(stat.S_IMODE(RELEASE.stat().st_mode) & ~0o222)
    print(json.dumps({'source_files': len(files), 'gitlinks': len(links), 'snapshot': str(output),
                      'sha256': hashlib.sha256(output.read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()
