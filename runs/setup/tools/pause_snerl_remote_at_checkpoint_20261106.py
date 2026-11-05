"""Lossless checkpoint pause of owned REMOTE SNeRL baseline pretraining containers (scripts/train_snerl.py), user
directive 2026-11-06 (stop the additional Meta-World environment expansion; standing authorization for owned
checkpoint pauses, 2026-10-10). SNeRL copy of ops/campaign/pause_remote_pretrain_at_checkpoint.py, adapted to the SNeRL
trainer's layout:

* only ``checkpoints/latest.pt`` (atomic tmp -> replace) holding ``global_step`` N, written when global_step reaches a
  multiple of ``train.save_every`` and BEFORE training step N runs; ``encoder.pt`` is exported right after it;
* the trainer logs ``metrics.jsonl`` (``{"step": ...}``) every ``train.i_print`` steps; resume with ``--resume auto``
  restores networks, optimiser, step and every RNG state, so a stop after the save loses no completed step.

Per job, one SSH session runs a watcher ON the remote host (no remote files written) that signals the container only
after (1) latest.pt holds global_step == target N (read from its pickle header), (2) encoder.pt was rewritten after it
and (3) no step beyond N + i_print is logged. The signal is ``docker kill --signal=TERM`` to the verified container id.
After the exit it checks latest.pt still holds N and records ``checkpoint_paused`` (intentional_stop: true) in
experiments/registry.jsonl, so the exit is not charged as a failure; runs/setup/admission-pauses.json keeps every driver
from relaunching it. A missed boundary moves the target to the next one; the final boundary is never a pause target.
Never removes containers or files.

    python -I pause_snerl_remote_at_checkpoint_20261106.py --reason <reason> <job id> [...]
    python -I pause_snerl_remote_at_checkpoint_20261106.py --reason <reason> --finalize <job id> [...]   (record only)
"""
import fcntl
import json
import os
import shlex
import sys
import threading
import time
from pathlib import Path

ROOT = Path('/home/ws/ws/splatter4d')
sys.path.insert(0, str(ROOT))
import yaml  # noqa: E402

from ops.jobs import scheduler as jobs  # noqa: E402
from ops.remote.access import RemoteClient  # noqa: E402

REGISTRY = ROOT / 'experiments/registry.jsonl'

WATCH = r'''
set -u
cd "$RUN" || exit 2
logged() { tail -c 20000 metrics.jsonl | grep -o '^{"step": [0-9]*' | tail -n1 | awk '{print $2}'; }
ckstep() { python3 -I -c '
import itertools, pickletools, sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as a:
    n = next(n for n in a.namelist() if n.endswith("/data.pkl"))
    ops = list(itertools.islice(pickletools.genops(a.read(n)), 7))
assert ops[-3][1] == "global_step", ops
print(ops[-1][1])
' "$1" 2>/dev/null || echo -1; }
cur=$(logged); target=$(( (cur / SAVE + 1) * SAVE ))
echo "ARMED $cur $target"
deadline=$(( $(date +%s) + LIMIT ))
last_m=$(stat -c %Y checkpoints/latest.pt)
while :; do
  if [ "$target" -ge "$STEPS" ]; then echo "FINAL_BOUNDARY $target"; exit 5; fi
  if [ "$(date +%s)" -gt "$deadline" ]; then echo "TIMEOUT $target"; exit 3; fi
  m=$(stat -c %Y checkpoints/latest.pt)
  if [ "$m" != "$last_m" ]; then
    s=$(ckstep checkpoints/latest.pt)
    if [ "$s" -eq "$target" ]; then
      # wait for the encoder export that follows the checkpoint (<= 60 s)
      for i in $(seq 1 300); do [ "$(stat -c %Y encoder.pt)" -ge "$m" ] && break; sleep 0.2; done
      l=$(logged)
      if [ "$l" -le $(( target + PRINT )) ]; then
        st=$(docker inspect --format '{{.Id}} {{.Name}} {{.State.Running}}' "$CID" 2>&1)
        if [ "$st" != "$CID /$NAME true" ]; then echo "IDENTITY $st"; exit 4; fi
        docker kill --signal=TERM "$CID" > /dev/null || exit 6
        echo "SIGNALLED $target $l $(date +%s.%N)"
        code=$(timeout 900 docker wait "$CID") || { echo "NO_EXIT"; exit 7; }
        echo "EXITED $code"
        echo "AFTER $(logged)"
        echo "STEP latest.pt $(ckstep checkpoints/latest.pt)"
        sha256sum checkpoints/latest.pt encoder.pt | sed 's/^/SHA /'
        exit 0
      fi
      echo "MISSED $target $l"; target=$(( target + SAVE ))
    elif [ "$s" -gt "$target" ]; then
      echo "MISSED $target (checkpoint at $s)"; target=$(( (s / SAVE + 1) * SAVE ))
    fi
    last_m=$m
  fi
  sleep 0.5
done
'''


def record(event):
    with REGISTRY.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with REGISTRY.open('a') as stream:
            stream.write(json.dumps({**event, 'time': time.time()}) + '\n')
            stream.flush()
            os.fsync(stream.fileno())


def pause(job_id, reason, out):
    client = RemoteClient()  # one client per thread
    current = jobs.read_registry()[job_id]
    assert current['host'] == 'remote' and current['status'] == 'running', (job_id, current['status'])
    attempt, name, cid = current['attempt'], current['container'], current['container_id']
    assert name == f's4d-{job_id}-a{attempt}' and len(cid) == 64, (name, cid)
    cfg = yaml.safe_load((ROOT / 'runs' / job_id / 'config.yaml').read_text())['train']
    save, steps, print_every = int(cfg['save_every']), int(cfg['max_steps']), int(cfg['i_print'])
    run = client.remote_path(f'runs/{job_id}')
    env = ' '.join(f'{k}={shlex.quote(str(v))}' for k, v in dict(
        RUN=run, CID=cid, NAME=name, SAVE=save, STEPS=steps, PRINT=print_every, LIMIT=3 * 3600).items())
    record({'event': 'checkpoint_pause_requested', 'id': job_id, 'host': 'remote', 'attempt': attempt,
            'container': name, 'container_id': cid, 'reason': reason, 'intentional_stop': True,
            'requested_signal': 'SIGTERM', 'save_every': save})
    result = client.ssh(env + ' bash -c ' + shlex.quote(WATCH), wait=True, idempotent=False, timeout=3 * 3600 + 1200)
    lines = result.stdout.splitlines()
    out[job_id] = {'code': result.returncode, 'lines': lines, 'stderr': result.stderr[-500:]}
    print(job_id, json.dumps(out[job_id]), flush=True)
    if result.returncode != 0:
        return
    target = int(next(line for line in lines if line.startswith('SIGNALLED')).split()[1])
    after = int(next(line for line in lines if line.startswith('AFTER')).split()[1])
    latest_step = int(next(line for line in lines if line.startswith('STEP')).split()[2])
    shas = {line.split()[2].rsplit('/', 1)[-1]: line.split()[1] for line in lines if line.startswith('SHA')}
    assert latest_step == target and after <= target + print_every, (latest_step, after, target)
    paused = {'event': 'checkpoint_paused', 'id': job_id, 'host': 'remote', 'attempt': attempt, 'container': name,
              'container_id': cid, 'reason': reason, 'intentional_stop': True, 'requested_signal': 'SIGTERM',
              'checkpoint_step': target, 'checkpoint': f'runs/{job_id}/checkpoints/latest.pt',
              'latest_sha256': shas['latest.pt'], 'encoder_sha256': shas['encoder.pt'],
              'last_logged_step': after, 'no_completed_steps_lost': True,
              'resume': 'scripts/train_snerl.py --resume auto from checkpoints/latest.pt (global_step, optimiser, RNG)',
              'container_exit_code': int(next(line for line in lines if line.startswith('EXITED')).split()[1]),
              'processes_exited': True}
    record(paused)
    with (ROOT / 'runs' / job_id / f'checkpoint_pause-{reason}.json').open('x') as stream:
        stream.write(json.dumps(paused, sort_keys=True) + '\n')
    print(json.dumps(paused), flush=True)


VERIFY = r'''
set -u
cd "$RUN" || exit 2
st=$(docker inspect --format '{{.Id}} {{.Name}} {{.State.Running}} {{.State.ExitCode}}' "$CID" 2>&1)
echo "STATE $st"
python3 -I -c '
import itertools, pickletools, sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as a:
    n = next(n for n in a.namelist() if n.endswith("/data.pkl"))
    ops = list(itertools.islice(pickletools.genops(a.read(n)), 7))
assert ops[-3][1] == "global_step", ops
print("STEP latest.pt", ops[-1][1])
' checkpoints/latest.pt
echo "MTIME $(stat -c %Y checkpoints/latest.pt) $(stat -c %Y encoder.pt)"
echo "AFTER $(tail -c 20000 metrics.jsonl | grep -o '^{"step": [0-9]*' | tail -n1 | awk '{print $2}')"
[ -f completion.json ] && echo "COMPLETED" || echo "NOT_COMPLETED"
sha256sum checkpoints/latest.pt encoder.pt | sed 's/^/SHA /'
'''


def finalize(job_id, reason):
    """Record the pause of a job whose remote watcher signalled it but whose result never reached this host (SSH lost
    after the signal): re-verify on the remote that the container exited with SIGTERM (143), not running, latest.pt
    holds a save boundary N, encoder.pt was exported with it, the last logged step is within one i_print of N, no
    completion.json;
    then record `checkpoint_paused` with the verified evidence."""
    client = RemoteClient()
    current = jobs.read_registry()[job_id]
    attempt, name, cid = current['attempt'], current['container'], current['container_id']
    assert name == f's4d-{job_id}-a{attempt}' and len(cid) == 64, (name, cid)
    cfg = yaml.safe_load((ROOT / 'runs' / job_id / 'config.yaml').read_text())['train']
    save, print_every = int(cfg['save_every']), int(cfg['i_print'])
    run = client.remote_path(f'runs/{job_id}')
    env = ' '.join(f'{k}={shlex.quote(str(v))}' for k, v in dict(RUN=run, CID=cid).items())
    lines = client.ssh(env + ' bash -c ' + shlex.quote(VERIFY), wait=True, idempotent=True, timeout=300).stdout.splitlines()
    state = next(line for line in lines if line.startswith('STATE')).split()
    step = int(next(line for line in lines if line.startswith('STEP')).split()[2])
    m_ck, m_enc = (int(x) for x in next(line for line in lines if line.startswith('MTIME')).split()[1:3])
    after = int(next(line for line in lines if line.startswith('AFTER')).split()[1])
    shas = {line.split()[2].rsplit('/', 1)[-1]: line.split()[1] for line in lines if line.startswith('SHA')}
    assert state[1:] == [cid, f'/{name}', 'false', '143'], state
    assert step % save == 0 and step - print_every <= after <= step + print_every and m_enc >= m_ck \
        and 'NOT_COMPLETED' in lines, \
        (step, after, m_ck, m_enc, lines)
    paused = {'event': 'checkpoint_paused', 'id': job_id, 'host': 'remote', 'attempt': attempt, 'container': name,
              'container_id': cid, 'reason': reason, 'intentional_stop': True, 'requested_signal': 'SIGTERM',
              'checkpoint_step': step, 'checkpoint': f'runs/{job_id}/checkpoints/latest.pt',
              'latest_sha256': shas['latest.pt'], 'encoder_sha256': shas['encoder.pt'], 'last_logged_step': after,
              'no_completed_steps_lost': True,
              'resume': 'scripts/train_snerl.py --resume auto from checkpoints/latest.pt (global_step, optimiser, RNG)',
              'container_exit_code': 143, 'processes_exited': True,
              'recorded_after_ssh_loss': 'the remote watcher signalled at the boundary; SSH dropped before its result '
                                         'reached this host; state re-verified on the remote by --finalize'}
    record(paused)
    with (ROOT / 'runs' / job_id / f'checkpoint_pause-{reason}.json').open('x') as stream:
        stream.write(json.dumps(paused, sort_keys=True) + '\n')
    print(json.dumps(paused), flush=True)


def main():
    args = sys.argv[1:]
    at = args.index('--reason')
    reason = args[at + 1]
    del args[at:at + 2]
    if args and args[0] == '--finalize':
        for job_id in args[1:]:
            finalize(job_id, reason)
        return
    out = {}
    threads = [threading.Thread(target=pause, args=(j, reason, out)) for j in args]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(json.dumps({'finished': True, 'ok': sorted(j for j, v in out.items() if v['code'] == 0),
                      'not_paused': sorted(j for j in args if out.get(j, {}).get('code') != 0)}), flush=True)


if __name__ == '__main__':
    main()
