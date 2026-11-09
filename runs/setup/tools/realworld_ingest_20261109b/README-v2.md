# Upload boost v2 (2026-11-09, item 3)

This is the operation document shipped inside the reviewed code copy. The coordinator may also place it at
`/home/ws/ws/realworld-ops/ingest/upload-boost/v2/README-v2.md`. The implementation agent writes only inside `v2/code`;
the live ingest README and all v1 state are left untouched.

## Upload boost

Only this worker's episode uploads **local → 192.168.10.22** receive a boost. One rsync runs at a time, so the lease
is the aggregate budget. The original `ingest/upload-budget.json` stays byte-for-byte unchanged (600 KiB/s;
`0` still pauses). Downloads keep `runs/remote/access/download-budget.json` at 200 KiB/s. Mirrors use the base budget;
other uploaders, SSH configuration and network settings are unchanged.

The ladder is **original → 8192 → 16384 → 32768 KiB/s → `UNCAPPED`**. The last mode omits `--bwlimit` entirely.
Entry requires six recent measured, healthy samples at original, and ten minutes without hard failure. Recently
healthy finite levels (within six hours) can be re-entered after idle; UNCAPPED always requires the final ladder step.
Each step needs ten minutes at the current cap, two transfers started and verified at that cap, ten measured echoes,
the latest four echo/srtt measurements, a fresh throughput sample, and median throughput at least 80% of the cap.
If the cap does not bind, the controller holds and records the bottleneck.

These **absolute bounds were chosen by the coordinator**, replacing v1 baseline-relative bounds:

| Measurement, last eight samples at the current level | Bound |
| --- | --- |
| SSH control echo p90 | ≤250 ms |
| Any echo in that window | ≤1000 ms, no failures |
| Existing ControlMaster TCP srtt p90 | ≤150 ms |
| Dataset TCP retransmitted segments / data segments | ≤0.02 |

Soft breaches drop to the last recently healthy level below the blamed cap, or original. A failed/timed-out echo,
transfer error, hash conflict, remote outage or six zero-throughput samples is hard: original and cooldown.
Failures at each level expire after 30 minutes; until then a failed level has a 30-minute cooldown and two failures
cap the ladder below it. Hard events also require ten quiet minutes before entry. Measurements missing for 300 seconds
restore original; missing measurements never qualify entry/step-up.

Measurements retain v1 methods: no fresh TCP connections for probes. Echo runs `true` on the existing campaign
ControlMaster with `ControlMaster=no` and `ProxyCommand=false`; an absent master is skipped. Passive sock_diag reads
ControlMaster srtt and the dedicated rsync SSH connection's bytes acknowledged / retransmissions; five-second polls
accumulate short uploads. The controller reads worker events and status counts.

All mutable v2 state is in **`ingest/upload-boost/v2/state/`**: `policy.json`, `samples.jsonl`, `history.jsonl`,
`activations.jsonl`, `levels.json`, controller PID/lock, policy lock, `STOP`, and eventual `retired.json`.
The worker reads only that policy. V1 `retired.json` never gates v2. V1 `original-policy.json` and optional
`baseline.json` are read-only context; a fresh baseline is unnecessary for the absolute bounds.

An activation starts when eligible work exists, before a boost is granted. It records an ID, start time, exact prior
upload/download budget bytes (base64), existence and SHA256, levels reached and end reason in an append-only,
fsynced journal. Start/level/end snapshots share the ID; the final snapshot is the complete activation record.
It ends after ten minutes without eligible uploads (including v1's no-upload/backoff grace), on hard failure,
STOP/restore, worker absence/incompatible version/base change, missing evidence, controller exit, or six hours.
Expiry writes `expired`, clears the lease and holds original for **at least ten minutes**, persisted across restart,
before a new activation can begin. Idle writes `idle`, clears the lease and waits for later arrivals.
Retirement requires **every named task** in `ingest_worker.TASKS` (drawer-open, mug-hanging, covering, pick-and-place)
to have at least `TARGET_PER_TASK=100` transferred and verified. Lower status targets or unrelated task names cannot
satisfy retirement.

## Status and restore

```bash
T=/home/ws/ws/splatter4d/runs/setup/tools/realworld_ingest_20261109b
PY=/home/ws/ws/splatter4d/.venv/bin/python
STATE=/home/ws/ws/realworld-ops/ingest
$PY -I -B "$T/upload_boost.py" status --state-dir "$STATE"
$PY -I -B "$T/upload_boost.py" restore --state-dir "$STATE" --reason "coordinator restore"
```

Restore atomically writes state `restored`, null lease and a v2 STOP marker under the policy lock, then closes any
open activation records. The marker wins over a concurrent controller renewal. The independent worker watchdog checks
every two seconds, terminates a higher/uncapped rsync and resumes its part with `--partial --append-verify` at original.
Increases apply only to the next transfer. The base budget file never needs rewriting. Compare the activation's saved
SHA256/base64 bytes with the two current budget files when auditing restore. Do not overwrite coordinator changes.

## Recovery after crash, reboot, outage or worker death

Every active lease contains controller PID, `/proc/<pid>/stat` start ticks, hostname, boot ID, and wall-clock plus
`CLOCK_BOOTTIME` expiry, at most 150 seconds ahead (also bounded by activation age). The worker requires every check.
A dead controller or reused PID is rejected immediately at the next watchdog poll. A hung controller loses its lease
within 150 seconds; within another poll the running transfer is terminated (SIGKILL fallback after 30 seconds) and
resumed at original. Thus an already-lapsed lease is cut within one lease period even during validation.
Boot ID mismatch rejects pre-reboot leases; wall-clock rollback cannot extend the boot-time expiry. A controller
restart rewrites a stale `active` policy to `restored` **before querying the control channel**, closes interrupted
journal entries and requalifies from original. Persisted expiry rest and recent failure memory survive restart.

An outage/failed control echo restores original, ends the activation and invokes the normal retry/quiet-period rules.
A missing ControlMaster is not a new connection attempt; missing echo evidence eventually restores original.
Worker exit retains the existing unconditional rsync cleanup; SIGKILL/crash additionally delivers
`PR_SET_PDEATHSIG=SIGKILL` to rsync, with a check for death between fork and arming. Its local SSH pipe closes and
no local uncapped sender survives the worker. Reboot kills both. Partial bytes remain resumable, and the existing
remote sha256 + no-clobber publish checks remain in force. No global unlimited configuration is ever installed.

Automatic system-service startup is not installed by this change. If nothing restarts after a reboot, original is
still effective; the following recipe resumes operation. Runtime writes require the existing ingest state permissions.

## Exact coordinator restart recipe (do not run against the old tool directory)

The coordinator first creates the **new** directory below from this reviewed copy, including `ingest_worker.py`,
`upload_boost.py`, `episode_validator.py` and tests. Keep the running `realworld_ingest_20261109` directory unchanged.
These commands are an operator recipe, not actions performed by the implementation agent.

```bash
T=/home/ws/ws/splatter4d/runs/setup/tools/realworld_ingest_20261109b
PY=/home/ws/ws/splatter4d/.venv/bin/python
STATE=/home/ws/ws/realworld-ops/ingest
LOGS=/home/ws/ws/realworld-ops/logs
B="$STATE/upload-boost/v2/state"
test -f "$T/upload_boost.py" && test -f "$T/ingest_worker.py" || exit 1
cd "$STATE" || exit 1

# Stop any previous v2 controller, then request graceful stop of the current worker.
$PY -I -B "$T/upload_boost.py" restore --state-dir "$STATE" --reason "restart from reviewed v2 tool copy"
touch "$STATE/STOP"
# Wait for both ownership locks to be released; abort rather than start duplicate processes.
flock -w 900 "$STATE/worker.lock" true || exit 1
flock -w 300 "$B/controller.lock" true || exit 1
# Existing v1 controller is permanently retired; leave its files/STOP/retired record intact.
# Remove only the worker STOP and v2 STOP after both owners have exited.
rm -f "$STATE/STOP" "$B/STOP"

( exec setsid nohup "$PY" -I -B "$T/ingest_worker.py" --loop --interval 300 --state-dir "$STATE" \
    >> "$LOGS/ingest.log" 2>&1 < /dev/null ) > /dev/null 2>&1 &
( exec setsid nohup "$PY" -I -B "$T/upload_boost.py" run --state-dir "$STATE" \
    >> "$LOGS/upload-boost-v2.log" 2>&1 < /dev/null ) > /dev/null 2>&1 &

$PY -I -B "$T/ingest_worker.py" --status --state-dir "$STATE"
$PY -I -B "$T/upload_boost.py" status --state-dir "$STATE"
# Check worker.pid: worker == rw-ingest-worker/5 and argv names the new directory.
# Initial original/idle/standby is expected while the controller collects its healthy entry window.
```

A v2 retired marker intentionally prevents restart after all four tasks complete; do not remove it for ordinary
later batches. `status` works with no controller. If a prior STOP remains after an outage, follow the ownership-lock
checks above before clearing it. Do not use v1 restore commands or v1 safe_restart scripts for v2 policy management.

## Validation

Tests use fake clocks/remotes plus existing local rsync integration cases; they do not access SSH, network or GPUs.
All test-created data/temp files stay under `ingest/scratch/tests/`; `-B` (also on the nested worker-lock test
interpreter) prevents bytecode writes beside the code.
The ingest tests read a raw HDF5 template without modifying it.

```bash
cd /home/ws/ws/realworld-ops/ingest/upload-boost/v2/code
TMPDIR=/home/ws/ws/realworld-ops/ingest/scratch/tests \
  /home/ws/ws/splatter4d/.venv/bin/python -I -B -m pytest -q -p no:cacheprovider \
  test_upload_boost.py test_ingest.py
```

The known pre-existing `test_one_dropped_stereo_record_is_valid_with_warning` expects stale `adapter_refused` output;
its expectation is intentionally unchanged. Live throughput and latency qualification remain runtime work for the
coordinator: this implementation session starts/stops no worker or controller and makes no network measurements.
