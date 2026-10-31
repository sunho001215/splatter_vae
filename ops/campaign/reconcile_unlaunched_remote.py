"""Reconcile a remote job whose launch was refused before `docker run` (launcher exit != 0, no container id).

`scripts/jobs.py` records such a launch as unknown ("launch outcome requires reconciliation") and, once a confirmed
snapshot lacks the deterministic container, as "container missing from confirmed snapshot"; it never relaunches
and `jobs.py reset` refuses active remote jobs. This tool resolves exactly that case and nothing else: under the
registry lock it re-checks the registry state, waits out a minimum age, confirms on the remote that neither the
deterministic container (any state) nor the job's remote output directory exists, and only then appends a `reset`
event carrying the evidence, so the dispatcher may launch the next attempt. Never deletes anything.

    python -I reconcile_unlaunched_remote.py <job_id>
"""
import fcntl
import json
import shlex
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path('/home/ws/ws/splatter4d')
sys.path.insert(0, str(ROOT))
from ops.remote.access import RemoteClient  # noqa: E402
from ops.jobs import scheduler as jobs  # noqa: E402

MIN_AGE_SECONDS = 600
REASONS = {'launch outcome requires reconciliation', 'container missing from confirmed snapshot'}

job_id = sys.argv[1]
print('WAITING_FOR_REGISTRY_LOCK', job_id, flush=True)
with open(jobs.REGISTRY.with_suffix('.lock'), 'a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    print('REGISTRY_LOCK_ACQUIRED', job_id, flush=True)
    current = jobs.read_registry().get(job_id)
    assert current is not None, f'{job_id}: not in the registry'
    assert current.get('host') == 'remote' and current['status'] == 'launching', (job_id, current['status'])
    assert current.get('unknown') is True and current.get('reason') in REASONS, (job_id, current.get('reason'))
    age = time.time() - current['attempt_since']
    assert age >= MIN_AGE_SECONDS, f'{job_id}: launch attempt only {age:.0f}s old'
    container = current['container']
    assert container == f"s4d-{job_id}-a{current['attempt']}", container
    args = current['args']
    client = RemoteClient()
    if current.get('verification_purpose') is not None:
        # Typed verification: its only outputs are the attempt directory and runs/<id>; neither may exist.
        from ops.containers.verification import verification_output  # noqa: PLC0415

        remote_dir = verification_output(current['verification_purpose'], job_id, current['attempt'])
        control_dir = client.remote_path(f"releases/{current['commit']}/code")
        control_kind = 'accepted_release_dir'
        output_test = (f"if [ -e {shlex.quote(remote_dir)} ] || [ -e {shlex.quote(client.remote_path(f'runs/{job_id}'))} ]; "
                       "then echo exists; else echo absent; fi; ")
    elif '--output-dir' in args and '--run-dir' in args:
        remote_dir = client.remote_path(args[args.index('--output-dir') + 1])
        control_dir = client.remote_path(args[args.index('--run-dir') + 1])
        control_kind = 'trainer_run_dir'
    elif current.get('script') == 'scripts/train_rl.py':
        # A DrM trainer writes only into its --run-dir; its companion evaluator may already exist and is not touched.
        assert args[args.index('--run-dir') + 1] == f'runs/{job_id}', args
        remote_dir = client.remote_path(f'runs/{job_id}')
        control_dir = client.remote_path(f"releases/{current['commit']}/code")
        control_kind = 'accepted_release_dir'
    elif current.get('script') == 'scripts/evaluate.py':
        # The remote driver uploads only runs/<id>/config.yaml before a full-split evaluation launches; "absent"
        # then means no other file exists in the run directory (no evaluation output was ever produced).
        assert args[:2] == ['--run', f'runs/{job_id}'], args
        remote_dir = client.remote_path(f'runs/{job_id}')
        control_dir = client.remote_path(f"releases/{current['commit']}/code")
        control_kind = 'accepted_release_dir'
        output_test = (f"if [ -z \"$(find {shlex.quote(remote_dir)} -mindepth 1 ! -path {shlex.quote(remote_dir + '/config.yaml')} "
                       f"-print -quit 2>/dev/null)\" ]; then echo absent; else echo exists; fi; ")
    else:
        declared = next(job for job in jobs.load_queue()['jobs'] if job['id'] == job_id)
        assert declared['script'] == 'scripts/train.py' and '--output-root' in args and '--name' in args
        assert args[args.index('--output-root') + 1] == 'runs' and args[args.index('--name') + 1] == job_id
        remote_dir = client.remote_path(f'runs/{job_id}')
        control_dir = client.remote_path(f"releases/{current['commit']}/code")
        control_kind = 'accepted_release_dir'
    if current.get('script') != 'scripts/evaluate.py' and current.get('verification_purpose') is None:
        output_test = f"test -e {shlex.quote(remote_dir)} && echo exists || echo absent; "
    # A failed Docker query must fail the check, not look like an empty container listing.
    command = (f"set -e; matches=$(docker ps -a --no-trunc --filter name=^/{container}$ --format '{{{{.ID}}}}'); "
               'if [ -n "$matches" ]; then printf \'%s\\n\' "$matches" | wc -l; else echo 0; fi; '
               + output_test +
               f"test -d {shlex.quote(control_dir)} && echo exists || echo absent; date -u +%FT%TZ")
    checks = []
    for _ in range(2):
        result = client.checked_ssh(command, wait=False, idempotent=True, timeout=90)
        count, directory, control, stamp = result.stdout.split()
        checks.append({'containers_with_name': int(count), 'remote_output': directory, control_kind: control,
                       'remote_time_utc': stamp})
        if len(checks) == 1:
            time.sleep(5)
    assert all(c['containers_with_name'] == 0 and c['remote_output'] == 'absent' and c[control_kind] == 'exists'
               for c in checks), checks
    evidence = {'job_id': job_id, 'attempt': current['attempt'], 'container': container, 'gpu': current['gpu'],
                'reason': current['reason'], 'launch_attempt_age_seconds': round(age, 1),
                'remote_output_dir': remote_dir, control_kind: control_dir, 'checks': checks,
                'output_rule': ('no file besides the driver-uploaded config.yaml'
                                if current.get('script') == 'scripts/evaluate.py' else 'directory absent'),
                'conclusion': 'launch refused before docker run; deterministic container never existed'}
    jobs.record({'event': 'reset', 'id': job_id, 'reconciliation': evidence})
stamp = datetime.now().astimezone()
out = ROOT / 'runs/setup' / f'reconcile-{job_id}-{stamp:%Y%m%d-%H%M%S}.json'
out.write_text(json.dumps({'recorded_local': stamp.isoformat(), **evidence}, indent=2))
print('RECONCILED', job_id, out)
