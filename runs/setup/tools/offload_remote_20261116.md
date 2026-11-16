The offloader previews approved, untouched LOCAL S rows from the live coordinator.
It never launches, repins, edits registry events, clears pauses or moves a started
checkpoint. Preview has no network access. Execute is restricted to the adopted
coordinator copy; run it only after adopting this code there.

```bash
cd /home/ws/ws/splatter4d
.venv/bin/python -B runs/setup/tools/offload_remote_20261116.py
# Explicit estimates for a single pass (seconds until start, from now):
.venv/bin/python -B runs/setup/tools/offload_remote_20261116.py \
  --local-wait-seconds 1800 --remote-start-seconds 60
# Publish placement and inputs, with the same explicitly chosen estimates:
.venv/bin/python -B runs/setup/tools/offload_remote_20261116.py \
  --local-wait-seconds 1800 --remote-start-seconds 60 --execute
```

The estimates above are examples, not measured completion predictions. Local RAM
saturation does not determine a finite wait. There is deliberately no guessed
infinite local wait or zero remote delay. An absent forecast keeps the row local.
Remote delay must include the ready queue and driver cadence, after transfer.
For a persistent controller, provide a refreshed forecast from the coordinator:

```json
{"checked_at": 1794800000, "rows": {
  "rwv2-20261113-export-dynamo-covering-S": {
    "local_wait_seconds": 1800, "remote_start_seconds": 60
  }
}}
```

`checked_at` is a Unix timestamp. Forecasts expire after 300 seconds by default;
an upstream controller must refresh them. Missing IDs remain local. Bandwidth
defaults to 6,000,000 bytes/sec aggregate, adjustable with `--upload-mbps`.
Offline previews deduplicate shared inputs and charge cumulative transfers in
Preprocessing → Pretraining → FQL, Ours → baselines order. `--inventory FILE`
accepts `remote_plan` relative file entries or absolute remote target entries
with SHA256 values. Mirror origins marked locally generated never prove remote
presence. Execute always checks actual remote checksums and recalculates cost.

```bash
# After adoption and exact-pin remote acceptance; foreground persistent loop:
bash runs/setup/tools/offload_remote_loop_20261116.sh \
  --execute --forecast runs/setup/offload-forecast.json
# Stop after the current safe transfer; never signals a training process:
touch runs/setup/STOP-OFFLOAD-REMOTE
```

`OFFLOAD_INTERVAL_SECONDS`, `OFFLOAD_STOP_FILE` and `OFFLOAD_PYTHON` customize the
loop. A preexisting STOP prevents any pass; the Python tool checks it between
rows and input uploads. A failed pass leaves placement unchanged or fails closed
via ownership; the loop retries. No background process is started by installation.

Only exact `planned_commit` + `planned_image` remote acceptance is valid. The
offloader uses the standard `acceptance_manifest` validator; driver admission
still checks release evidence, pushed commit, dependency completion, roles,
pauses, foreign tenants, GPU/RAM/CPU/disk capacity and priority reservations.
Remote's disabled main-queue flag is retained: admission uses its normal dispatch
queue. A host-local forecast is not an admission bypass. Execute also checks fresh
guard-accounted resources both before upload and before publication, charging
other tenants, release reservations and already waiting higher-priority work.

Checksums, source/acceptance evidence and transfers are prepared outside admission
locks. Publication takes stage2-driver → remote-driver → registry nonblocking,
rereads the row/pin, ownership, dependencies, pauses and roles, and checks inode,
size, mtime and ctime of verified inputs/evidence. Backups are in
`runs/setup/offload-backup-<timestamp>/` (original queue, ownership, pool overlay
and transaction). Ownership precedes the atomic pool and queue rewrites. A crash
between these files can leave a held row requiring coordinator inspection of
that backup; automatic rollback could overwrite concurrent legitimate changes.

Upload stages live beside the destination on its filesystem, are resumable and
retained for retry. A no-clobber hard link publishes only missing files, with
checksums before and after. Conflicting existing content is refused. Symlinks
and changed sources are refused; no checkpoint or another user's file is replaced.
Receipt checks cannot prevent another authorized writer changing inputs later;
these published input paths must retain the campaign's immutable-input contract.

RGB dataset/training consumers and local native-depth descriptors use absolute
LOCAL paths. They remain held: the exact pinned remote mounts cannot consume them byte for
byte. A separately reviewed portable publication/mount contract and corresponding
accepted release are needed; changing argv or descriptors here would change
scientific identity. Other unknown input resolvers also fail closed.
Encoder artifact copies consume only their declared checkpoint, so their unused
RGB publication metadata does not require moving the publication.

On read-only inspection on 2026-11-16, the coordinator advanced the listed rows
from 48d92fa to a3a34db (same image 59082834…) during implementation, while the
remote role release was f7786e8. The final offline preview found no exact remote
acceptance manifest for a3a34db for all 15 listed waiting rows. Thus none can offload immediately
with the required exact pin. The six encoder exports (including drawer-v2 dynamo)
are the first small-input candidates once that acceptance and forecasts exist.
