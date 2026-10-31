"""Lossless checkpoint pause of owned REMOTE pretraining containers (user directive 2026-10-28: pause the paper ablations
at the nearest safe, complete checkpoint; standing authorization for owned checkpoint pauses, 2026-10-10).

Remote counterpart of pause_pretrain_at_checkpoint.py. Per job, one SSH session runs a watcher ON the remote host
(no remote files written) that signals the container only after (1) checkpoints/step_<N>.pt is published at the next
save boundary N (s4d/train/checkpoint.py writes step_<N>.pt then latest.pt, each by atomic rename, before the
validation at N), (2) the validation at N is in metrics.jsonl and (3) no training step after N is logged. The signal
is `docker kill --signal=TERM` to the verified container id (tini forwards it to scripts/remote/entry.py, which
forwards it to the trainer). After the exit it checks that no step after N was logged and that latest.pt holds step N,
and records `checkpoint_paused` (intentional_stop: true) in experiments/registry.jsonl, so the exit is not charged as a
failure. The job stays resumable from checkpoints/latest.pt (--resume auto); runs/setup/admission-pauses.json keeps
every driver from relaunching it. A missed boundary moves the target to the next one; the final boundary is never a
pause target (the run would simply complete). Never removes containers or files.

    python -I pause_remote_pretrain_at_checkpoint.py --reason <reason> <job id> [...]
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
from ops.remote.access import RemoteClient  # noqa: E402
from ops.jobs import scheduler as jobs  # noqa: E402
import yaml  # noqa: E402

REGISTRY = ROOT / 'experiments/registry.jsonl'

WATCH = r'''
set -u
cd "$RUN" || exit 2
logged() { tail -c 20000 log.txt | grep -o '\] step [0-9]* loss' | tail -n1 | awk '{print $3}'; }
cur=$(logged); target=$(( (cur / SAVE + 1) * SAVE ))
echo "ARMED $cur $target"
deadline=$(( $(date +%s) + LIMIT ))
while :; do
  if [ "$target" -ge "$STEPS" ]; then echo "FINAL_BOUNDARY $target"; exit 5; fi
  if [ "$(date +%s)" -gt "$deadline" ]; then echo "TIMEOUT $target"; exit 3; fi
  ck=checkpoints/step_$(printf %07d "$target").pt
  if [ -f "$ck" ] && tail -c 400000 metrics.jsonl | grep "^{\"step\": $target," | grep -q '"val/'; then
    l=$(logged)
    if [ "$l" -lt $(( target + LOGEVERY )) ]; then
      st=$(docker inspect --format '{{.Id}} {{.Name}} {{.State.Running}}' "$CID" 2>&1)
      if [ "$st" != "$CID /$NAME true" ]; then echo "IDENTITY $st"; exit 4; fi
      docker kill --signal=TERM "$CID" > /dev/null || exit 6
      echo "SIGNALLED $target $l $(date +%s.%N)"
      code=$(timeout 900 docker wait "$CID") || { echo "NO_EXIT"; exit 7; }
      echo "EXITED $code"
      echo "AFTER $(logged)"
      python3 -I -c '
import itertools, pickletools, sys, zipfile
for p in sys.argv[1:]:
    with zipfile.ZipFile(p) as a:
        assert a.testzip() is None
        n = next(n for n in a.namelist() if n.endswith("/data.pkl"))
        ops = list(itertools.islice(pickletools.genops(a.read(n)), 7))
    assert ops[-3][1] == "step"
    print("STEP", p, ops[-1][1])
' "$ck" checkpoints/latest.pt
      sha256sum "$ck" checkpoints/latest.pt | sed 's/^/SHA /'
      exit 0
    fi
    echo "MISSED $target $l"; target=$(( target + SAVE ))
  fi
  sleep 0.2
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
    cfg_path = ROOT / 'runs' / job_id / 'config.yaml'
    cfg = yaml.safe_load(cfg_path.read_text())['train']
    save, steps, log_every = int(cfg['save_every']), int(cfg['steps']), int(cfg['log_every'])
    assert save % int(cfg['eval_every']) == 0
    run = client.remote_path(f'runs/{job_id}')
    env = ' '.join(f'{k}={shlex.quote(str(v))}' for k, v in dict(
        RUN=run, CID=cid, NAME=name, SAVE=save, STEPS=steps, LOGEVERY=log_every, LIMIT=3 * 3600).items())
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
    steps_seen = {line.split()[1].rsplit('/', 1)[-1]: int(line.split()[2]) for line in lines if line.startswith('STEP')}
    shas = {line.split()[2].rsplit('/', 1)[-1]: line.split()[1] for line in lines if line.startswith('SHA')}
    ck = f'step_{target:07d}.pt'
    assert after < target + log_every and steps_seen == {ck: target, 'latest.pt': target}, (after, steps_seen)
    paused = {'event': 'checkpoint_paused', 'id': job_id, 'host': 'remote', 'attempt': attempt, 'container': name,
              'container_id': cid, 'reason': reason, 'intentional_stop': True, 'requested_signal': 'SIGTERM',
              'checkpoint_step': target, 'checkpoint': f'runs/{job_id}/checkpoints/{ck}',
              'checkpoint_sha256': shas[ck], 'latest_sha256': shas['latest.pt'],
              'validation_at_checkpoint_logged': True, 'last_logged_step': after, 'no_steps_lost': True,
              'container_exit_code': int(next(line for line in lines if line.startswith('EXITED')).split()[1]),
              'processes_exited': True}
    record(paused)
    with (ROOT / 'runs' / job_id / f'checkpoint_pause-{reason}.json').open('x') as stream:
        stream.write(json.dumps(paused, sort_keys=True) + '\n')
    print(json.dumps(paused), flush=True)


def main():
    args = sys.argv[1:]
    at = args.index('--reason')
    reason = args[at + 1]
    del args[at:at + 2]
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
