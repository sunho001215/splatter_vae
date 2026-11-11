"""Drop the wrapper-written ``config.yaml`` from the two ours real-world pretraining rows (2026-11-11).

ops/containers/launch.py refuses numerical completion that relies on wrapper-only artifacts, and
config.yaml is one of them, so rwcal-20261111-ours-{drawer-open,mug-hanging}-pre were refused at
launch. Under stage2-driver.lock then registry.lock: back up the queue, remove exactly the
``  - config.yaml`` line inside each of those two rows, then verify that the YAML parses, the job
count and every other row are unchanged, and that the two rows differ only in required_results.
"""
import fcntl
import shutil
import sys
from pathlib import Path

import yaml

REPO = Path('/home/ws/ws/splatter4d')
QUEUE = REPO / 'experiments/queue.yaml'
IDS = ('rwcal-20261111-ours-drawer-open-pre', 'rwcal-20261111-ours-mug-hanging-pre')
BACKUP = QUEUE.with_name('queue.yaml.before-ours-pre-required-results-20261111')
EXPECTED = ['completion.json', 'metrics.jsonl', 'checkpoints/step_0200000.pt']


def edit(text):
    lines = text.splitlines(keepends=True)
    for job_id in IDS:
        starts = [i for i, line in enumerate(lines) if line.rstrip('\n') == f'  id: {job_id}']
        assert len(starts) == 1, (job_id, starts)
        start = starts[0]
        while not lines[start].startswith('- '):
            start -= 1
        end = start + 1
        while end < len(lines) and not lines[end].startswith('- '):
            end += 1
        hits = [i for i in range(start, end) if lines[i].rstrip('\n') == '  - config.yaml']
        assert len(hits) == 1, (job_id, hits)
        del lines[hits[0]]
    return ''.join(lines)


def main(dry_run):
    with (REPO / 'runs/setup/stage2-driver.lock').open('a') as admission:
        fcntl.flock(admission, fcntl.LOCK_EX)
        with (REPO / 'experiments/registry.lock').open('a') as registry:
            fcntl.flock(registry, fcntl.LOCK_EX)
            before_text = QUEUE.read_text()
            after_text = edit(before_text)
            before = yaml.safe_load(before_text)['jobs']
            after = yaml.safe_load(after_text)['jobs']
            assert len(before) == len(after), (len(before), len(after))
            for old, new in zip(before, after):
                if old.get('id') in IDS:
                    assert new['id'] == old['id']
                    assert new['required_results'] == EXPECTED, new['required_results']
                    assert {k: v for k, v in old.items() if k != 'required_results'} == \
                        {k: v for k, v in new.items() if k != 'required_results'}
                else:
                    assert old == new, old.get('id')
            print(f'jobs={len(after)} changed={list(IDS)} required_results={EXPECTED}')
            if dry_run:
                print('dry run: queue untouched')
                return
            if not BACKUP.exists():
                shutil.copy2(QUEUE, BACKUP)
            temporary = QUEUE.with_name(QUEUE.name + '.tmp-ours-pre-fix')
            temporary.write_text(after_text)
            with temporary.open('rb') as handle:
                import os
                os.fsync(handle.fileno())
            shutil.copymode(QUEUE, temporary)
            os.replace(temporary, QUEUE)
            print(f'written; backup {BACKUP}')


if __name__ == '__main__':
    main('--dry-run' in sys.argv[1:])
