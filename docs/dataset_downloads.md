# Large-scale DROID data downloads to the remote data disk (`ops/datasets/`)

Directive I (2026-11-10) items 6-7: the inputs of the large-scale DROID path (docs/droid_largescale.md) on the remote
host's `/data_8T` (ext4, 7.62 TB), under `/data_8T/kaist/sunho/`. Internet downloads only; the local <-> remote
transfer budget (`ops.remote.access`) is a separate policy and is not used for data.

## What is downloaded (pinned official sources; `ops/datasets/catalog.py`)

| Group | Source | Files | Download | On disk when done |
|---|---|---|---:|---:|
| `rlds` | `gs://gresearch/robotics/droid/1.0.1/` (public, not requester-pays; generation-pinned URLs) | 2048 shards + `dataset_info.json` + `features.json`; md5 from the GCS listing | 1,865,994,705,042 | same (used in place, never extracted) |
| `pw_depth` | `nvidia/PointWorld-DROID` @ `dd9aaeec` `droid/depth_320x180` | 252 parts of ONE `tar.zst` stream; LFS sha256 | 1,234,491,720,995 | ~1.24 TB of `*_depth.h5` (estimate: part-0000 expands x1.0044; plan uses +2 %) |
| `pw_flows` | same, `droid/flows-fs-optimized/shard-*` | 771 self-contained shards; LFS sha256 (= `_shards_manifest.json`) | 2,670,478,819,773 | 3,407,932,552,380 (`raw_member_bytes`, exact) |
| `pw_cameras` | same, `droid/cameras` | 1 package; LFS sha256 | 19,472,217 | 91,524,697 (42,935 JSON) |
| `pw_meta` | same | `LICENSE.pdf`, `README.md` (git blob ids), `_shards_manifest.json` | 4,021,082 | same |
| `droid3d` | `jaibrdhn/droid_3d_extrinsics` @ `69db9ccf` (**gated**) | `extrinsics.jsonl`, `camera_intrinsics.jsonl`, card; LFS-pointer blob ids | 302,780,254 | same |
| `hand_eye` | DepthWorld @ `6823c13d`, PointWorld code @ `3872ec6` | the two `gripper2wrist_transforms.json`; GitHub blob ids | 47,593 | same |

Not taken: `droid/1.0.0`, `droid_100`, PointWorld `droid/confidence` and helper scripts, DROID-3D `pseudolabels/`
and `extrinsics_missing.json` (unused by the pipeline). Every RLDS shard contains PointWorld episodes that join
DROID-3D, so there is no RLDS subset. Identity across sources: PointWorld member name = flow `uuid` = DROID-3D
`episode_uuid`; RLDS by the scene-path suffix (docs/droid_largescale.md section 4.1).

Layout (`root` = `/data_8T/kaist/sunho/datasets`): `droid/1.0.1/` (`data.rlds_data_root`),
`pointworld-droid/droid/{flows-fs-optimized,depth_320x180,cameras}/` (`data.pointworld_root`), `droid3d/`,
`.downloads/{tmp,pointworld-droid}/`, `.staging/`, `.state/`.

## Verification and publication

- Downloads (https only, also for redirects; a token goes only to huggingface.co): temp file -> size + official
  checksum (md5 / sha256 / git blob id / LFS pointer blob id) -> fsync -> atomic rename. Partials resume by byte range
  (re-hashed first); a server that ignores the range restarts the file. Bounded attempts (8) with exponential backoff
  (30 s .. 15 min); then `failed` (`retry` resets). A refused token is retried only after the token file changes.
- Flow shards: download -> extract into `.staging/pw_flows/<shard>` -> members must equal the shard's manifest list
  and `raw_member_bytes`, HDF5 signature on every file -> state `publishing` (saved) -> fsync + rename into the tree
  -> state `extracted` (saved) -> delete archive. A restart in `publishing` finishes the renames (no re-extraction).
- Depth: the parts stay until the whole stream is extracted (no index; a restart re-decompresses from part 0 and skips
  staged members) -> HDF5 signature on every member, member set compared with the flow episodes -> `publishing` ->
  publish -> `published` -> delete parts. A part that exhausts its attempts fails the stream (`retry` resets both).
- The episode selection is frozen once the first flow shard or the depth stream is extracted (`configure` refuses
  another one; the owner gates both groups if the config is edited by hand).
- Space: every download is admitted only if the claims of all admitted unfinished work plus its own leave
  `margin_fraction` x file-system size available (default 0.15; a lower value is honoured only on `/data_8T`, user
  decision 2026-11-10: 0.085 there, every other file system keeps 0.15 whatever the configuration says). Claims are upper bounds: a shard archive claims its
  full expansion; the depth stream claims, from its first part on, every remaining part plus its whole expansion. The
  flow group is refused while the final footprint of the enabled groups would cross the margin. Extraction stops
  while free space is below the margin, and all transfers pause (partials kept) while free space cannot cover the
  margin plus the outstanding claims (another writer); they resume by themselves.
- Restarts: `reconcile` restores unfinished records whose destination already verifies (crash between rename and
  state save), returns missing finished files to pending and deletes archives whose content was published. The
  owner keeps its lock until every worker and lane has stopped; `adopt` and `retry` hold the same lock.

## Concurrency and health

One HTTP connection per file; no tool with internal parallelism. Pool starts at `pool_initial` (3) files; every
`interval_s` (20 s) the owner samples throughput, PSI cpu/io/memory, available memory, data-disk utilisation, ICMP
RTT to 8.8.8.8 and to the gateway, the sshd sessions' kernel RTT and transfer errors (`.state/health.jsonl`). Healthy:
add one file, keep it only if throughput rises >= 7 %, else revert and hold 15 min. Degraded (internet RTT > 2.5x
baseline and > baseline + 60 ms, gateway RTT > 10 ms, io PSI > 25 %, memory PSI of OUR cgroup > 10 % (host-wide memory PSI is
only reported: on 2026-11-10 it came from another tenant), host MemAvailable < 10 %, disk > 90 % busy,
>= 3 errors per interval) twice in a row: drop one file; at the minimum pool, a per-process rate cap at 0.7x the
current rate, tightened while degraded, relaxed x1.3 per healthy period and removed. No fixed cap otherwise.
Local control-channel latency (fresh SSH connect + `true`, command RTT over the shared connection):
`scripts/datasets/droid_datasets_remote.py latency`.

## Commands

Remote (host `python3`, from the deployed copy; `<sha>` = deployed commit):

```bash
T=/data_8T/kaist/sunho/tools/<sha>/scripts/datasets/droid_datasets.py
python3 -I $T plan                      # official listings -> .state/plan.json + storage projection
python3 -I $T storage [configure options]  # final + peak table (current or proposed config; never writes)
python3 -I $T start                     # reconcile, then the detached owner (setsid, nice 10, ionice -c2 -n7)
python3 -I $T status [--full]           # live owner check + last status
python3 -I $T stop                      # graceful; partials kept
python3 -I $T configure --pointworld-selection all|<episode list> --enable pw_flows [--margin-fraction F]
                        [--schedule depth_first]   # refused if the final or the scheduled peak crosses the margin
python3 -I $T adopt --item droid3d/extrinsics.jsonl --source <existing copy>   # verified copy, source untouched
python3 -I $T retry                     # failed -> pending
```

Local: `.venv/bin/python -I scripts/datasets/droid_datasets_remote.py {deploy --sha S | latency | remote --sha S -- <cmd>
| stage-upload --kib N <files>}` (`stage-upload`: into `.downloads/adopt/`, which must resolve onto the /data_8T mount,
only if the file fits above the floor; explicit `--bwlimit`; partial name with delta transfer; sha256 verified on the
remote host; atomic no-replace publication; then `adopt`).

Space table (`storage`): final growth (credits for bytes already on disk), block rounding (one 4 KiB block per file
still to be written; measured 2.0-2.8 KB per extracted member, 6-8 KB per multi-GB file), the depth parts held until
the stream is published, in-flight flow shard archives at the maximum pool, one shard archive during its extraction;
peaks unordered and with `schedule: depth_first` (depth parts are scheduled before RLDS, and no flow shard is admitted
before the depth stream is published; the depth stream also claims its remaining parts and expansion from its first
part on). The owner refuses to start if a directory it writes resolves outside the root's file system.

Resume after a reboot or an outage: `start` (it refuses a second owner while the lock is held; a stale `owner.json`
after a crash or reboot is ignored because the lock is free; `downloading` items return to `pending` and continue
their partial files; `extracted` archives still present are deleted; staged members are kept).

Gated DROID-3D: the items stay `blocked` until `.state/secrets/hf_token` (mode 0600, an approved account's token)
exists, or until `adopt` installs verified copies.
