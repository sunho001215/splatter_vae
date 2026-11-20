"""Read-only Claude observation after activation; finite advancing loss in all three.

Never imports torch, takes a lock, invokes a driver or signals a process.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import k0_isolated_release as release


def inspect(candidate, root=release.MAIN):
    grant = json.loads(candidate.read_bytes())
    authorization_sha = release.digest(candidate.read_bytes())
    ids = sorted(i for i in release.IDS if '-pre-ours-' in i)
    launched, exits = {}, {}
    for line in (root / 'experiments/registry.jsonl').open('rb'):
        if b'rwk0-20261120-' in line:
            event = json.loads(line)
            if event['id'] in ids and event['event'] == 'launched':
                launched[event['id']] = event
            if event['id'] in ids and event['event'] == 'exited':
                exits[event['id']] = event
    result = {}
    for identity in ids:
        launch = launched.get(identity, {})
        exit_event = exits.get(identity, {})
        if (launch.get('attempt', 0) > 2 or
                exit_event.get('attempt', 0) >= 2 and exit_event.get('code') != 0):
            raise SystemExit(f'{identity}: attempt 2 failed or another retry appeared; rollback future admission now.')
        policy_path = root / 'runs' / identity / 'isolated-execution.json'
        policy = json.loads(policy_path.read_bytes()) if policy_path.exists() else {}
        provenance = (launch.get('attempt') == 2 and launch.get('host') == 'local'
            and launch.get('commit') == release.COMMIT
            and launch.get('isolated_authorization_sha256') == authorization_sha
            and launch.get('execution_adapter_sha256') == grant['adapter_sha256']
            and policy.get('authorization_sha256') == authorization_sha
            and policy.get('execution_worker_sha256') == grant['worker_sha256']
            and policy.get('full_suite_claimed') is False)
        records = []
        path = root / 'runs' / identity / 'metrics.jsonl'
        if path.exists():
            for line in path.read_text().splitlines(keepends=True):
                if not line.endswith('\n'):  # writer may be between write and flush
                    continue
                row = json.loads(line)
                losses = {k: v for k, v in row.items() if k.startswith('loss/')}
                if losses:
                    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in losses.values()):
                        raise ValueError('nonfinite loss: ' + identity)
                    records.append({'step': row['step'], **losses})
        steps = [r['step'] for r in records]
        advancing = len(steps) >= 2 and steps[0] > 0 and all(a < b for a, b in zip(steps, steps[1:]))
        result[identity] = {'ok': provenance and advancing, 'attempt': launch.get('attempt'),
                            'provenance_ok': provenance, 'first_metrics': records[:2],
                            'last_step': steps[-1] if steps else None}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--timeout', type=int, default=600)
    args = parser.parse_args()
    if not 0 <= args.timeout <= 600:
        parser.error('timeout must be 0..600 seconds')
    deadline = time.monotonic() + args.timeout
    while True:
        report = inspect(args.candidate)
        print(json.dumps(report, sort_keys=True), flush=True)
        if all(row['ok'] for row in report.values()):
            return
        if time.monotonic() >= deadline:
            raise SystemExit('No finite advancing attempt-2 metrics from all three by deadline; rollback future admission.')
        time.sleep(min(15, max(0, deadline - time.monotonic())))


if __name__ == '__main__':
    main()
