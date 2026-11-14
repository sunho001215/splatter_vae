# S/SF coordinator runbook — 2026-11-14

Implementer source: `0cb605b9fd24a8633907479082fd0ada599e48ca`, branch
`dsreset-20261113`, evidence worktree `/home/ws/ws/splatter4d/.worktrees/dsreset`.
Use HANDOFF.md for the final test result and docs-only release candidate. Main
`f2e2f8b` and current main/local driver fix `9c901bf` are merged. The implementer has not launched
preprocessing, contacted either remote machine, changed queue/registry/pauses,
or adopted main. All commands below are coordinator operations.

Start asset staging and host preparation immediately, independently of calibration
and training registration. After the source passes/adoption, register/start 248
phase-A jobs: drawer success98 + failure50, covering failure50, PNP failure50.
Reuse covering success98 and PNP success97. Drawer 77/79 are excluded by the binding
amendment, SHA256 `041042f1ff98d220b20dfd5ee159ed113eb79ad29a357134f246e8e5514ccbe7`.
Episode79 has coincident apparent 32-bit wraps but no recorded modulus or accepted
lossless repair contract; do not guess. Raw files remain unchanged.

Execution order: sections1/2 now; adoption commands in10 once evidence passes;
then3 starts A. Stage LoMa assets and old-success mirrors/proofs (4/6) while A runs.
Calibration jobs wait for A; B waits for its task's accepted rig; records/inputs
wait for B/QA; measurements wait for inputs; publication/461-row registration wait
for accepted reports. These are automatic artifact gates, not new human reviews.

## 1. Execution namespaces and immediately available assets

There is no Docker CLI/socket/systemd PID1 in the current campaign container.
The local server host binds `/ws/shkim` to its `/home/ws`. Host setup below must
be executed on that LOCAL HOST, not through Robot PC/`irsl`, and never in this
workspace container. Do not assume `.venv` is usable on the host: its interpreter
is inside the campaign container. Use the established prep image's interpreter
for the controller; it has the required scientific Python dependencies.

Inside the campaign container, use these variables for subsequent commands:

```bash
cd /home/ws/ws/splatter4d
export P=/home/ws/ws/realworld-ops/dsreset-20261113/pipeline
export A=/home/ws/ws/splatter4d/runs/rwv2-prep-assets
export R=/home/compu/kaist/sunho/realworld
export D=/home/compu/kaist/sunho/data/real_world
export X=$R/processed/dsreset-20261113
mkdir -p "$A/weights" "$A/canonical-realworld" /home/ws/data/realworld-processed/dsreset-20261113
```

`local-weight-audit.json` records the local checkpoint audit. First stage each
already-local file that matches its established SHA; download only missing pins:

```bash
PREP_WEIGHT_PHASE=A .venv/bin/python - <<'PY'
import hashlib, os, pathlib, shutil, yaml
from ops.remote.access import RemoteClient
root=pathlib.Path('runs/rwv2-prep-assets/weights').resolve()
server=yaml.safe_load(pathlib.Path('/home/ws/ws/realworld-ops/pre/tools/rwprep/configs/rwprep_server.yaml').read_text())
client=None
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
for row in server['weights']:
    if os.environ.get('PREP_WEIGHT_PHASE')=='A' and row['worker'] not in ('sam3','foundation_stereo'):continue
    rel=row['dest'].removeprefix('/ram/weights/'); dst=root/rel
    local=pathlib.Path('/home/ws/ws/validation-20261031/downloads')/rel
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.is_file() and sha(dst)==row['sha256']:continue
    if dst.exists():raise RuntimeError(f'conflicting staged weight: {dst}')
    if local.is_file() and sha(local)==row['sha256']:shutil.copyfile(local,dst)
    else:
        client=client or RemoteClient()
        client.rsync_download('/home/compu/kaist/sunho/realworld/weights/'+rel,dst)
    assert sha(dst)==row['sha256'],dst
print('Required phase weight pins verified')
PY
```

The three phase-A pins all match local files (audit recorded); this command needs
NO weight download. Repeat it with `PREP_WEIGHT_PHASE=B` during phase A to stage
the remaining TE/teacher pins before phase B. Teacher checkpoint downloads must
not delay phase-A launch.

The host should first inspect the already installed image
`s4d-rwprep-f7e075f51c79:20261106`. If absent, the coordinator can archive that exact
image on the remote with `RemoteClient.checked_ssh`, download the tar using
`rsync_download` into `$A/prep-image.tar` (shared bandwidth/checksum controls), then
run host `docker image load --input /home/ws/ws/splatter4d/runs/rwv2-prep-assets/prep-image.tar`.
Exact archive/download command, executed inside the campaign container only if needed:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
from ops.remote.access import RemoteClient
c=RemoteClient(); remote='/home/compu/kaist/sunho/realworld/dsreset-prep-image-20261114.tar'
c.checked_ssh('flock '+remote+'.lock sh -c '+__import__('shlex').quote(
    'test -f '+remote+' || { docker image save --output '+remote+'.tmp s4d-rwprep-f7e075f51c79:20261106 && mv -T '+remote+'.tmp '+remote+'; }'),idempotent=True,timeout=3600)
c.rsync_download(remote,Path('runs/rwv2-prep-assets/prep-image.tar').resolve())
PY
```

Do not fetch an image from an unpinned registry or print credentials. Image save
and staging may be slow; existing installed image and local matching weights are
the fastest route. LoMa upstream staging (section4) is independent of phase A.

## 2. Local host mounts and durable controller

Canonical paths must be identical on both hosts. This preserves existing compact
store paths and raw metadata, permits exact Ours-S reuse, and satisfies the standard
remote/local input resolver. Do not rewrite old H5 metadata or the reused S paths.
The host needs these bind mounts, propagated into the campaign namespace as well
as the controller. Verify `/ws/shkim` against `docker inspect` before installation.
All mount targets must be absent/empty, or an already verified identical mount;
never cover an existing unrelated directory.

Exact host preparation (source script lives on the existing host bind):

```bash
sudo bash /ws/shkim/ws/realworld-ops/dsreset-20261113/pipeline/prepare-prep-host.sh
docker image inspect --format '{{.Id}}' s4d-rwprep-f7e075f51c79:20261106 \
 > /home/ws/ws/realworld-ops/dsreset-20261113/pipeline/prep-image-id
docker update --restart unless-stopped 23cdfd5100ca3ded4557c748bc1623c5d0fd0e6a1b526190fde7d3c077af594f
```

The script validates the campaign container's current identity/mount, creates only
empty/identical targets, installs the exact fstab entries below, and repeats the
three canonical mounts in the campaign's current mount namespace. It refuses
changed mappings. It does not stop/recreate any container. `unless-stopped` keeps
this existing campaign container across unattended Docker/host restart.

| Host source | Target in host and campaign container | Mode |
|---|---|---|
| `/ws/shkim` | `/home/ws` (HOST only; already present in campaign) | rw |
| `$A/canonical-realworld` | `$R` | rw |
| `/home/ws/data/realworld-processed/dsreset-20261113` | `$X` | rw |
| `/home/ws/data/real_world` | `$D` | ro |

Install the host bind mounts persistently as four systemd mount units or verified
`/etc/fstab` entries, ordered parent before child. The following is the exact fstab
content after substituting these literal paths (append only once, after checking
targets). `mount --bind` alone does not survive reboot.

```fstab
/ws/shkim /home/ws none bind 0 0
/home/ws/ws/splatter4d/runs/rwv2-prep-assets/canonical-realworld /home/compu/kaist/sunho/realworld none bind,x-systemd.requires-mounts-for=/home/ws 0 0
/home/ws/data/realworld-processed/dsreset-20261113 /home/compu/kaist/sunho/realworld/processed/dsreset-20261113 none bind,x-systemd.requires-mounts-for=/home/compu/kaist/sunho/realworld 0 0
/home/ws/data/real_world /home/compu/kaist/sunho/data/real_world none bind,ro,x-systemd.requires-mounts-for=/home/ws 0 0
```

Create empty targets before mounting, `mount /home/ws`, `mount "$R"`, create
`$R/processed/dsreset-20261113`, then `mount "$X"` and `mount "$D"`. Inspect
`findmnt -M` for all four mounts. Mount propagation into the existing campaign
container is not assumed: use `nsenter --target <CURRENT campaign host PID> --mount`
to create the same three canonical mounts in its mount namespace, or run the
read/compile commands below through the new controller namespace. A coordinator
may instead use a shell in a separately provisioned management container with
these exact mounts and the same main/pipeline directories.

The campaign container observed in the supervisor inventory is
`23cdfd5100ca3ded4557c748bc1623c5d0fd0e6a1b526190fde7d3c077af594f`.
Revalidate identity and host mount source; do not reuse a stale PID. Its cgroup
must be mounted READ-ONLY at `/sys/fs/cgroup` in each dispatcher controller.
This is essential: `memory_budget.snapshot()` then reads the existing campaign's
actual charge, while prep sibling containers are reserved at their full limits.
Reading the controller's own cgroup would undercount training; reading the whole
host would incorrectly charge unrelated users against the 150GB campaign cap.
The controller also joins the campaign PID namespace, so session PSS is measured
against the same PIDs as the live registry; worker containers retain host PIDs.

Use `pipeline/launch-prep-controller.sh` on the LOCAL HOST. It validates the
campaign identity each start, resolves its current cgroup, mounts the same shared
paths, and uses the immutable prep image interpreter. Install this service on
the host, with the normal Docker binary path verified first:

```ini
# /etc/systemd/system/dsreset-prep@.service
[Unit]
Description=Admitted S/SF preprocessing queue %i
Requires=docker.service
After=docker.service
RequiresMountsFor=/home/ws /home/compu/kaist/sunho/realworld/processed/dsreset-20261113 /home/compu/kaist/sunho/data/real_world
[Service]
Type=simple
ExecStartPre=/bin/bash /home/ws/ws/realworld-ops/dsreset-20261113/pipeline/prepare-prep-host.sh
ExecStart=/bin/bash /home/ws/ws/realworld-ops/dsreset-20261113/pipeline/launch-prep-controller.sh %i
Restart=always
RestartSec=15
KillMode=process
TimeoutStopSec=20
[Install]
WantedBy=multi-user.target
```

The launcher uses no network. Docker daemon access is confined to the controller;
workers have no socket, read-only code/raw/weights, no network, exactly one MIG,
CPU/RAM/swap/PID/tmpfs limits. The controller queries all accepted local MIGs; it
does not perform model computation. Host and campaign must share the same main
lock and priority directory. Use one controller per immutable bundle.
Each host-controller tick additionally takes LIVE `experiments/registry.lock`,
which serializes the ordinary scheduler's local FQL admission. The implementation
takes stage2-driver.lock nonblocking inside that tick; a busy driver causes a
skip/retry, never a lock-order deadlock. Locks are released between15-second ticks.
Use this supplied launcher for local operations, rather than the bare daemon CLI.

## 3. Register and start phase A immediately after adoption

After adopting the passing candidate (section10), freeze a standalone clean clone
with an ordinary `.git` directory. Worktree git pointers are inaccessible in worker
containers. No submodule download is required for phase A's established workers.

```bash
cd /home/ws/ws/splatter4d
git clone --no-hardlinks /home/ws/ws/splatter4d "$P/prep-code"
git -C "$P/prep-code" checkout --detach "$(git rev-parse HEAD)"
# On the LOCAL HOST, record docker image inspect --format '{{.Id}}' s4d-rwprep-f7e075f51c79:20261106
# into $P/prep-image-id, a single immutable sha256:<64hex> line; paths are shared.
PREP_IMAGE=$(cat "$P/prep-image-id")
.venv/bin/python scripts/realworld_prep_register.py \
 --plan "$P/drawer-canonical-manifests/prep-plan.json" \
 --plan "$P/covering-canonical-manifests/prep-plan.json" \
 --plan "$P/pick-and-place-canonical-manifests/prep-plan.json" \
 --out "$P/local-phaseA" --processed "$X" --raw "$D" \
 --weights "$A/weights" --image "$PREP_IMAGE" --code "$P/prep-code" \
 --control-repo /home/ws/ws/splatter4d
```

Expected registration: **248**, launched=false. The real preview
`local-phaseA-preview/{jobs,prep-plan}.json` proves membership but is not a runnable
registered bundle. Do not use older `*-final-manifests` or provisional bundles.
Run a controller dry tick on the host before enabling its permanent service:

```bash
sudo bash /home/ws/ws/realworld-ops/dsreset-20261113/pipeline/launch-prep-controller.sh local-phaseA --dry-run
sudo install -m 0644 /home/ws/ws/realworld-ops/dsreset-20261113/pipeline/dsreset-prep@.service /etc/systemd/system/dsreset-prep@.service
sudo systemctl daemon-reload
sudo systemctl enable --now dsreset-prep@local-phaseA.service
journalctl -u dsreset-prep@local-phaseA.service -f
```

Phase A is ingest + native sensor depth + FoundationStereo + SAM3. Admission is
nonpreemptive: 16GiB GPU (established measured peak <=13.1GiB), 48GiB RAM, 4 CPUs,
32GiB reserved transient disk per job. Actual charge + remaining training growth
+ full prep/co-tenant RAM limits must fit 138e9 bytes inside the 150e9 cap, with
60GiB host reserve. One idle accepted 1g.24gb per job; no foreign GPU processes.
Load + new CPU <=85% quota, idle>=20%, CPU PSI<=75, >=15% disk free and32GiB floor.
RAM may correctly limit concurrency before all eight MIGs are occupied.

`state/STOP` stops new admission and drains/reconciles healthy workers. Remove only
this bundle's STOP to resume. State is `state/state.json`, attempt logs/results
under `state/<job>/a<N>`. Attempts<=3; ordinary stage fingerprints reuse complete
outputs. `unknown` means a previously running named container disappeared: inspect
its exact identity/output before repairing its state under the dispatcher lock.
Never delete state to retry or launch duplicate output writers. No auto cleanup.

Each dispatcher owns a separate durable hold under LIVE main
`runs/setup/rwprep-mig-priority.d/`. Ready/running/unknown prep blocks all NEW
campaign admission. Missing calibration dependencies do not reserve resources.
A crashed active controller stays fail-closed until restarted. A drained controller
cannot clear another controller's hold. Existing healthy trainers continue.

## 4. LoMa-G joint rigs — register while phase A runs

Run this exact coordinator command in the campaign container to stage sources
and caches through the existing checked, bandwidth-limited transport:

```bash
.venv/bin/python "$P/stage-prep-assets.py" upstream
```

It stages the pinned remote LoMa source at
`$A/calib-upstream/LoMa` (remote `realworld/calib-loma/upstream/LoMa`, commit415caed),
RoMa-v2 at `$A/calib-upstream/romav2` (remote
`realworld/calib-dw/upstream/romav2-95c9968`), and merged checkpoint caches at
`$A/calib-upstream/torch-home`. Cache source directories are remote
`realworld/calib-loma/torch-home` and `realworld/calib-dw/torch-home`.
Download archives through RemoteClient into `$A`, which is an authorized transfer
root. Merge identical checkpoint names only when their SHA256 agrees; refuse
conflicts. Preserve source commit manifests and weight SHA manifests. Required
LoMa checkpoint names include loma_G.pth, dedode_descriptor_G.pth,
dinov2_vitl14_pretrain.pth, dad.pth under torch-home/hub/checkpoints; also retain
RoMa's pinned model/backbone checkpoint set. Do not fetch model weights at runtime.

For each task, compile its independent session jobs:

```bash
for task in drawer-open-v2 covering pick-and-place; do
 .venv/bin/python scripts/realworld_dsreset_calibration.py plan \
  --phase-config "$P/local-phaseA/dispatcher.json" --task "$task" \
  --out "$P/local-calib-$task" --work "$X/calibration/$task" \
  --upstream "$A/calib-upstream"
done
# On LOCAL HOST, for each of the three bundles:
sudo systemctl enable --now dsreset-prep@local-calib-drawer-open-v2.service
sudo systemctl enable --now dsreset-prep@local-calib-covering.service
sudo systemctl enable --now dsreset-prep@local-calib-pick-and-place.service
```

They wait automatically for all task phase-A jobs in the external state file.
Drawer uses all148 new task episodes. Covering/PNP fit and verify the NEW failure
session only; their old success rigs/builds remain reused. Do not infer recording
period from an interleaved filename counter. Numeric aliases are confined to the
legacy solver and explicitly map back to label-inclusive identities on publication.

The graph runs RoMa matching, SAM-free init/static selection, visual Panda+Robotiq
mesh, LoMa r0 then r2, joint per-task solves, heldout validation, half/group/block/
LOCO/eight-start stability, native-depth independent scale solve, fixedness and
criteria-v2 pairwise/static evaluation. FoundationStereo remains the selected
depth source (S2M2 was retired by the established comparison). No old drawer rig,
cross-task prior or calibration.h5 enters fitting/selection. SAM is validation only.

Historical 4-process matching (~40GiB) does not fit. New matching is ONE process per
job, CPU rasterization, CUDA allocation cap20GiB, declaration21GiB + admission
margin. This split/cap is a conservative runnable bound, not a newly measured peak;
OOM fails/retries visibly and must not be solved by lowering science thresholds.
CPU solves reserve <=48GiB RAM and4 CPUs each; never launch the old `run_many ...4`.

Automatic acceptance retains existing G1-G6/criteria-v2: stability .35deg/8mm
(including image/radius backstop), heldout robot depth<=12mm and explained>=.70,
contour/o2m<=2px, epipolar<=1px/transfer<=3.75px, all12 SAM-free tolerance
perturbations p<.01 at frame AND episode level, fixedness and>=90% verifiable G5b,
plus timing/scale and provenance checks. See source criteria and
docs/realworld_calibration_release.md for exact conjunctions. Passing cameras
publish `$X/calibration/<task>/rig-<task>-accepted.json` + `.reference.npz`.
`automatic-decision.json` records held/validated. No new manual signoff is needed
on a pass; failed criteria must be corrected/re-run without forced acceptance.

## 5. Phase B and per-family QA — each task starts when its rig passes

Create a new task-only plan and a complete label-inclusive rig map from phase A;
the following derives them, so output roots stay identical:

```bash
.venv/bin/python - <<'PY'
import json,os,pathlib
p=pathlib.Path(os.environ['P']); x=pathlib.Path(os.environ['X'])
rows=json.loads((p/'local-phaseA/prep-plan.json').read_text())
for task in ('drawer-open-v2','covering','pick-and-place'):
    selected=[r for r in rows if r['episode'].startswith(task+'-')]
    rig=x/'calibration'/task/f'rig-{task}-accepted.json'
    (p/f'phaseB-{task}-plan.json').write_text(json.dumps(selected,indent=2))
    (p/f'phaseB-{task}-rig-map.json').write_text(json.dumps({r['episode']:str(rig) for r in selected},indent=2))
PY
task=drawer-open-v2 # repeat covering/pick-and-place independently when each auto-accepted rig exists
.venv/bin/python scripts/realworld_prep_register.py --phase B \
 --plan "$P/phaseB-$task-plan.json" --rig-map "$P/phaseB-$task-rig-map.json" \
 --out "$P/local-phaseB-$task" --processed "$X" --raw "$D" \
 --weights "$A/weights" --image "$(cat "$P/prep-image-id")" --code "$P/prep-code" \
 --extra-readonly "$X/calibration/$task" --control-repo /home/ws/ws/splatter4d
# LOCAL HOST:
sudo systemctl enable --now dsreset-prep@local-phaseB-drawer-open-v2.service
```

Phase B checks accepted camera status and hashes the rig/reference, then runs
rig-mode calibrate, bounded TrackEverything, track_motion, compact train_data,
contracts, QA `--families`. It declares7GiB GPU/16GiB RAM and sets
`S4D_TE_CUDA_MEMORY_CAP_GIB=5.5` (established <=5.61GiB GPU,<=6.2GB RAM). The old
40GiB TE path cannot run on these slices. Family QA and every current-build
calibration check remain gates. No FQL/pretraining is released merely by phase A.

## 6. Reused success mirrors, remote proof, NEW ingest snapshot

While phase A/B run, bring the existing195 success compact directories and their
completion records to their EXACT canonical local paths, read-only in workers.
Use `ops.jobs.realworld_local.fetch`/RemoteClient's verified directory inventory.
Existing roots are `$R/processed/calrel-6c3b9ec/covering` and
`$R/processed/calv2-4368f22/pick-and-place`. Only compact train_data directories
need copying for learning/measurement; keep live legacy ING records unchanged.
Stage the original rig/reference docs too if a downstream native builder needs
them. Verify file SHA256 against the remote before exposing mirrors. Do not copy
old obsolete drawer/mug datasets. Mirror downloads go under `$A` then canonical
local bind directories; RemoteClient rejects external pipeline transfer targets.

Run the following commands; the helper stages only the compact directories
actually referenced by the canonical covering/PNP S manifests.

```bash
.venv/bin/python "$P/stage-prep-assets.py" reused
.venv/bin/python "$P/stage-prep-assets.py" raw-proof
```

Generate independent remote raw checksums (all three SF manifests, same file
identity and task/label suffix), using `stage-prep-assets.py raw-proof`. It writes
`$P/remote-raw-hashes.json`, mapping local manifest raw_sha256 to the matching
verified remote SHA. This is read-only remote hashing; a missing/mismatching file
fails. Existing source verification markers alone do not invent this proof.

After all new B jobs pass, build the NEW snapshot (not live ING):

```bash
.venv/bin/python scripts/realworld_dsreset_snapshot.py \
 --source /home/ws/ws/realworld-ops/ingest/manifest.json \
 --manifest "$P/drawer-canonical-manifests/SF/manifest.json" \
 --manifest "$P/covering-canonical-manifests/SF/manifest.json" \
 --manifest "$P/pick-and-place-canonical-manifests/SF/manifest.json" \
 --prep-config "$P/local-phaseA/dispatcher.json" \
 --prep-config "$P/local-phaseB-drawer-open-v2/dispatcher.json" \
 --prep-config "$P/local-phaseB-covering/dispatcher.json" \
 --prep-config "$P/local-phaseB-pick-and-place/dispatcher.json" \
 --remote-hashes "$P/remote-raw-hashes.json" --out-dir "$P/ingest-snapshot"
```

Snapshot preserves exact reused success records, hashes new raws, runs standard
raw validation, collects actual stage/_COMPLETE + QA-v3 records, retains invalid
amended exclusions. It cannot waive missing or failed geometry/QA.

Create SESSION-SCOPED planning views, retaining original snapshot keys. This is
necessary because covering/PNP reused success records were built with other
accepted rig documents; do not try to validate them with the new failure rig:

```bash
.venv/bin/python - <<'PY'
import json,os,pathlib
p=pathlib.Path(os.environ['P']); m=json.loads((p/'ingest-snapshot/manifest.json').read_text())
for task in ('drawer-open-v2','covering','pick-and-place'):
    ids={r['episode'] for r in json.loads((p/f'phaseB-{task}-plan.json').read_text())}
    scoped={**m,'episodes':{k:e for k,e in m['episodes'].items() if e.get('episode_id') in ids}}
    (p/f'calibration-snapshot-{task}.json').write_text(json.dumps(scoped,indent=2))
PY
for task in drawer-open-v2 covering pick-and-place; do
 .venv/bin/python scripts/realworld_prep_followup.py records \
  --base-config "$P/local-phaseB-$task/dispatcher.json" --task "$task" \
  --rig "$X/calibration/$task/rig-$task-accepted.json" \
  --manifest "$P/calibration-snapshot-$task.json" --out "$P/local-records-$task" \
  --result "$X/calibration-records/$task"
done
# LOCAL HOST: enable dsreset-prep@local-records-<task>.service for each task.
# Once each plan.json has been automatically emitted, first dry-run then apply:
for task in drawer-open-v2 covering pick-and-place; do
 .venv/bin/python -m s4d.realworld.calib.release apply \
  --plan "$X/calibration-records/$task/plan.json" --state-dir "$P/ingest-snapshot"
 .venv/bin/python -m s4d.realworld.calib.release apply \
  --plan "$X/calibration-records/$task/plan.json" --state-dir "$P/ingest-snapshot" --apply
done
```

The ordinary release writer re-verifies keys, raw SHA and current calibrate build
before any write. Apply scoped plans to the full NEW snapshot directory; preserved
success validation records are not replaced. No period exceptions are fabricated.

## 7. Paired inputs and native scene measurement on the same admitted route

```bash
for task in drawer-open-v2 covering pick-and-place; do
 case "$task" in drawer-open-v2) bundle=drawer;; *) bundle=$task;; esac
 for variant in S SF; do
  .venv/bin/python scripts/realworld_dsreset_inputs.py \
   --manifest "$P/$bundle-canonical-manifests/$variant/manifest.json" \
   --ingest-manifest "$P/ingest-snapshot/manifest.json" \
   --out-dir "$P/learning-inputs/$task/$variant"
 done
 .venv/bin/python scripts/realworld_prep_followup.py measure \
  --base-config "$P/local-phaseB-$task/dispatcher.json" --task "$task" \
  --rig "$X/calibration/$task/rig-$task-accepted.json" \
  --inputs "$P/learning-inputs/$task/SF" --out "$P/local-measure-$task" \
  --result "$X/measurements/$task"
done
# LOCAL HOST: enable dsreset-prep@local-measure-<task>.service for each task.
```

Before starting measure controllers, include the mirrored success compact roots
as `extra_readonly` in the NEW measurement dispatcher configs; their default base
config only mounts the new processed root. Add these two roots for covering/PNP,
respectively, and keep `$D` raw mounted read-only. `stage-prep-assets.py mount-reuse`
does this before the first tick, refusing a state file that already exists:

```bash
.venv/bin/python "$P/stage-prep-assets.py" mount-reuse
```

The measurements use native RealSense alignment, retained compact calibration
poses, exact SF TRAIN membership (8 evenly spaced frames/camera/episode), union
of per-class .001/.999 quantiles +1cm, >=99% containment for robot/object/fixture.
Optical-axis lookat and4/median camera-distance pose_scale use scene_statistics.
Outputs `$X/measurements/<task>/scene.json` bind membership, manifest hash and
native overlay provenance. Both variants use the same accepted bounds/scene
settings. Registration consumes the measured drawer SNeRL runtime settings;
another manual source/config edit or release is unnecessary. No heldout episode
enters scene statistics. A failed measurement is a real automatic gate failure.

## 8. Publish verified canonical inputs before queue registration

Use RemoteClient's checked uploads for ONLY the new `$X` compact outputs, accepted
rig/reference documents and `$R/learning_inputs/dsreset/<task>/{S,SF}` five-file
bundles. Native measurements remain available at both local aliases (the example
spec names `/home/ws/data/realworld-processed/dsreset-20261113/measurements/...`).
Use `stage-prep-assets.py publish` for the exact manifest-selected outputs and
five-file bundles. It refuses existing different remote files; identical files
are reused. No `--delete`, no overwriting old success outputs, raw untouched.
The emitted canonical stores still name `$R` paths, not local mount sources.

Copy `registration-spec.example.json` to NEW `registration-spec.json`, replacing
each rig with `$R/processed/dsreset-20261113/calibration/<task>/rig-<task>-accepted.json`.
Keep both `reuse_ours_S=true` flags and the old input copies; the registrar will
re-prove hashes/splits/calibration builds/science against the actual live queue.
`stage-prep-assets.py spec` produces this concrete spec and refuses overwrites.

```bash
.venv/bin/python "$P/stage-prep-assets.py" publish
.venv/bin/python "$P/stage-prep-assets.py" spec
```

## 9. Register exactly461 new rows, then place without releasing partial inputs

First activate ONLY the standard named pause under the normal local/remote
admission locks, preserving every other pause. `stage-prep-assets.py pause` does
this with prefixes `["rwv2-20261113-"]`. It does not launch or stop anything.

```bash
.venv/bin/python "$P/stage-prep-assets.py" pause
```

```bash
cd /home/ws/ws/splatter4d
DSRESET_COMMIT=$(git rev-parse HEAD)
DSRESET_IMAGE=$(.venv/bin/python -c 'import json; print(json.load(open("runs/setup/role-release.json"))["image"])')
.venv/bin/python runs/setup/tools/register_dsreset_20261113.py \
 --spec "$P/registration-spec.json" --ingest-manifest "$P/ingest-snapshot/manifest.json" \
 --commit "$DSRESET_COMMIT" --image "$DSRESET_IMAGE" --release-source local \
 --out "$P/registration-dryrun.json"
# Check result has461 NEW rows (477 minus16 already equivalent Ours S producers).
.venv/bin/python runs/setup/tools/register_dsreset_20261113.py \
 --spec "$P/registration-spec.json" --ingest-manifest "$P/ingest-snapshot/manifest.json" \
 --commit "$DSRESET_COMMIT" --image "$DSRESET_IMAGE" --release-source local --execute
```

No preview substitutes for this complete dry run. The required pause makes append
safe; queue dependencies/shared barriers re-audit data. Ours follows Original200k
+ one full-state warm restart to400k; final400k only feeds FQL. Priorities for NEW
rows: prep/calib/qa -3; all pretraining/export ours-2, others-1; FQL ours0, others1.
Existing reused/running rows retain their exact priorities and release pins.

Run standard placement plan with `--allow-pause dsreset-await-placement-20261113`,
then verified fetch/place using the ordinary `ops/jobs/realworld_local.py` CLI:

```bash
.venv/bin/python -I ops/jobs/realworld_local.py plan --policy fit \
 --allow-pause dsreset-await-placement-20261113
# Fetch/place only IDs selected by that fit plan; the command takes explicit IDs:
.venv/bin/python -I ops/jobs/realworld_local.py fetch --priority dependency <selected-row-ids...>
.venv/bin/python -I ops/jobs/realworld_local.py place <selected-row-ids...> \
 --allow-pause dsreset-await-placement-20261113 --dry-run
.venv/bin/python -I ops/jobs/realworld_local.py place <selected-row-ids...> \
 --allow-pause dsreset-await-placement-20261113
.venv/bin/python -I ops/jobs/realworld_local.py verify
```

After complete mirrors, validated placement/dependencies/release pins and all prep
controllers drained (no active prep holds), `stage-prep-assets.py unpause` changes
ONLY that named pause to inactive. Keep discontinued-task and remote safety holds.

```bash
.venv/bin/python "$P/stage-prep-assets.py" unpause
```

Ensure the established supervisor is installed/reboot-persistent by following
`../supervisor/MIGRATION.md` and `../supervisor/install/INSTALL.md`; do not start
duplicate scheduler/drivers or terminate healthy jobs. The persistent drivers then
launch eligible pretraining before FQL; export/fqldata/FQL are dependency driven.
Local accepted release rows may start locally; remote placement still requires
the SAME commit's normal remote suite/isolation acceptance and lifted remote hold.

Offline eval barriers capture heldout trainer metrics. Subsequent completed policy
transfer/registration and SAFE NON-ACTUATING validation use the established
coordinator deployment workflow. No robot actuation, active-policy reload or5090
inference-server restart is part of this runbook.

## 10. Release/adoption and optional remote MIG work

Use final passing `docs/tests.json` in the frozen evidence worktree, including
full_suite_passed=true, exit0, zero failed/error/skipped, unchanged source hashes,
one authorized local MIG, exact test coverage and expected_commit. Historical
failed/incomplete reports are not evidence. Before the final suite main was
rechecked; if it advances in fingerprinted sources before adoption, merge/retest.

After the established reviewed push of the FINAL candidate to origin/main,
coordinator executes (network required, including dry-run):

```bash
cd /home/ws/ws/splatter4d
# Use the explicit final candidate in HANDOFF.md; evidence stays at tested source HEAD.
DSRESET_CANDIDATE=$(git rev-parse dsreset-20261113-release)
DSRESET_EXPECTED_MAIN=$(git rev-parse HEAD)
.venv/bin/python -I runs/setup/tools/merge_adopt_local_mig_20261112.py \
 "$DSRESET_CANDIDATE" "$DSRESET_EXPECTED_MAIN" /home/ws/ws/splatter4d/.worktrees/dsreset --dry-run
.venv/bin/python -I runs/setup/tools/merge_adopt_local_mig_20261112.py \
 "$DSRESET_CANDIDATE" "$DSRESET_EXPECTED_MAIN" /home/ws/ws/splatter4d/.worktrees/dsreset
```

Do not commit bookkeeping atop the frozen evidence worktree: the helper checks
expected_commit against its HEAD. A separate docs-only descendant candidate is
supported when no fingerprinted source differs from the evidence ancestor.

Remote optional acceleration: keep the old whole-GPU dispatcher's STOP in place.
Compile a disjoint episode subset on the REMOTE executing host with `--host remote`,
remote raw/tools/weights/assets/code, LIVE remote control queue/registry and accepted
16x1g.24gb inventory. The same load+CPU<=.85 admission deliberately refuses current
load~9000. Never weaken it. Partition IDs before compilation; never two writers for
one output. Run the same container/controller service on remote when admissible;
remote has no need for local raw alias mounts. Mirror its independent ACTIVE
priority hold to local main before any remote start, fail closed on disconnect,
and clear the local mirror only after proved remote drain. Until that coordinated
hold bridge and same-commit remote acceptance exist, use local only; neither is a
dependency of the immediate local route.

Remote registration is the same `realworld_prep_register.py` invocation as section3
with a NEW disjoint plan/output, `--host remote --raw /home/compu/kaist/sunho/data/real_world`,
remote processed/tools/weights paths, `--assets /home/compu/kaist/sunho/realworld/assets/validation-20261031`,
and its accepted standalone release clone. Before the first tick, set that remote
dispatcher's `admission_lock` to
`/home/compu/kaist/sunho/runtime/launch/admission.lock` (the actual standard remote
Docker launch guard lock, also used by established rwprep). The compiler's generic
control-repo stage2 lock is insufficient for cross-controller remote admission.
Run the remote daemon with the prep image interpreter, host PID/cgroup view and
host Docker access; use `rwprep_mig.py --config <remote bundle>/dispatcher.json`
for dry-run and `--apply --loop` in its enabled host service. Keep queue/registry
copies fresh and mirror its independent hold locally BEFORE the first applied tick.
For a fail-closed minimal bridge, keep a dedicated local `.d/*.json` active=true
for the entire immutable remote bundle, and clear it only after remote state proves
every row done and no live/unknown worker. This conservative bundle hold requires
no runtime networking in the workers. Re-establish it before every NEW bundle;
never clear another dispatcher's file. Partition before registration; a future
remote-admission opportunity does not authorize editing a live local immutable queue.
