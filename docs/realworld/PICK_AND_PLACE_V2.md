# Pick-and-place-v2 runbook: when the data arrives

User directive 2026-11-17: prepare ingestion → preprocessing → calibration → pretraining → FQL → evaluation so
experiments start promptly; **launch nothing before the data exists**; pick-and-place v1 stays discontinued
(`pick-and-place-v1-discontinued-20261117`). S only (100 successes, SF disabled), FQL seeds 0/1/2.

One entry point: `scripts/pnpv2.py` (`PY=/home/ws/ws/splatter4d/.venv/bin/python -I`).

| Command | What it does | Writes |
|---|---|---|
| `$PY scripts/pnpv2.py status` | arrival (complete marker or stable inventory), every stage's evidence, rows per wave, **the exact next command** | nothing |
| `$PY scripts/pnpv2.py check-data` | rehash + validate all recordings: 100 files, SHA vs Robot-PC list, no duplicates, recorder complete, 3 cameras, monotonic timing, actions, success-only, seed-0 90/10 split | nothing |
| `$PY scripts/pnpv2.py register --commit <main HEAD> --image <role image>` | dry run of the ready, unregistered waves; refuses unless check-data passes on the actual data | nothing |
| `... register ... --execute` | adopted main only: appends rows under `stage2-driver.lock` → `remote-driver.lock` → `registry.lock`, exact-ID pause `pnpv2-await-placement-20261117` first, pending journal + proof in `runs/setup/` | queue, pauses, proof |

## Stage graph

Upstream stages are the established producers (dsreset RUNBOOK sections 2–8), gated as evidence at
registration. Queue rows start at each view's readiness barrier.

| Stage | Producer (`status` prints the full command) | Evidence checked |
|---|---|---|
| arrival | operator's verified pull into `/home/ws/data/real_world/pick-and-place-v2/success` (+`success.{local,robotpc}.sha256`, `success.VERIFIED`) | marker with equal SHA lists, or unchanged inventory ≥600 s old; partial names (`*.part`, `.rsync-partial`, …) never complete |
| check-data | `pnpv2.py check-data` | see above |
| ingest | `<work>/arrival.json` (reviewed SAM3 prompts, robot URDF sha256, confirmer) then `pick_and_place_v2.py --entry manifest` | canonical manifest == check-data; 100 fresh prep configs |
| Phase A (100 jobs) | `realworld_prep_register.py --phase A --id-stem pnpv2-20261117-prepA` + host controller | all jobs `done`, ids use the v2 stem |
| RGB publication | `--entry publish-rgb -- --execute`, `--entry upload -- --view rgb` | 5-file bundle, RGB consumers pass, geometry refused, upload receipt |
| rig | `realworld_dsreset_calibration.py plan --id-stem pnpv2-20261117-calib` (waits for Phase A) | accepted `rig-pick-and-place-v2-accepted.json` + reference, every camera validated, no override |
| Phase B (100 jobs) | `--entry rig-map`, `realworld_prep_register.py --phase B --id-stem pnpv2-20261117-prepB` | all jobs `done` |
| geometry publication | `--entry geometry-remote`, remote tick, `--entry geometry-fetch`, `--entry upload -- --view geometry` | snapshot + per-episode `calibration_validated`, geometry consumers pass, RGB-identical split/outcomes, verified compact and canonical mirrors, upload receipt |
| scene | included in the geometry remote job; mirrored by `--entry geometry-fetch` | accepted train-only native measurement bound to the canonical manifest and rig |

Queue rows (90; ids `pnpv2-20261117-<stage>-<method>-pick-and-place-v2-S[-s<seed>]`):

| Wave | Needs | Rows |
|---|---|---|
| **rgb** (DynaMo, ReViWo, E2E CNN, E2E CNN noaug) | check-data, ingest, Phase A, RGB publication; **never** rig/Phase B | gate 1, pre 2, export 2, fqldata 4, fql 12, eval 12 = **33** |
| **geometry** (Ours orig/ch01 scratch400k, SinCro, SNeRL, 3D-MVP single/multiview) | the above + rig, Phase B, geometry publication, scene | gate 1, stats 1, native-depth 1, pre 5, export 5, fqldata 6, fql 18, eval 18, Ours task-length eval 2 = **57** |

Settings: frozen encoders FQL alpha 10 + `normalize_q_loss=true`; E2E CNN alpha 10, nq false (noaug: `p_aug=0`);
Ours = drawer-open-v2 S scratch400k (ch01 adds `loss.chamfer=0.1`), export step 400000; 3D-MVP single/multiview share
one native-depth pretrain/export; SNeRL/3D-MVP get the measured lookat/pose_scale/bounds via `--set`. Rows run
`scripts/pnpv2_<script>.py` (task installed process-locally, then the established script with the established argv;
a leftover `${pnpv2.*}` template field refuses). `ops/jobs/pnpv2_resolver.py` (installed by host_roles) maps them to
the established input resolvers; the registrar checks `needed_inputs` and host roles for both hosts.

## When the data arrives

1. `$PY scripts/pnpv2.py status` until `arrived: true`; then `$PY scripts/pnpv2.py check-data` (exit 0).
2. Look at a few recordings; write `/home/ws/ws/realworld-ops/pnpv2-20261117/arrival.json`
   (`{"schema":"pick-and-place-v2-arrival/1","task":"pick-and-place-v2","sam3_prompts":{...},"robot_urdf_sha256":"…","confirmed_by":"…"}`;
   v1 objects are only a reference in the spec). No repo edit or new release is needed.
3. Run each `next` command `status` prints, in order. Phase A and the rig dispatcher can be started together
   (the rig waits for Phase A); the RGB publication only needs Phase A.
4. As soon as `status` shows `rgb.ready_after_check_data: true`: dry run, then
   `register --wave rgb --commit $(git -C /home/ws/ws/splatter4d rev-parse HEAD) --image <role image> --execute`
   from the adopted main. Place exactly those ids with the established `ops/jobs/realworld_local.py place`
   and lift `pnpv2-await-placement-20261117` for them. These rows start while calibration/Phase B continue.
5. When the geometry wave is ready, repeat step 4 with `--wave geometry`.

## Remote Phase-A route (2026-11-18)

The local-host controller route needs the host Docker daemon, the canonical `/home/compu/...` bind mounts and free
local MIGs. From the campaign container, Phase A and the RGB publication therefore run on the REMOTE prep route
(`ops/jobs/rwprep_remote.py`, the drawer-open-v2 SAM3/Phase-B route), with the processed release
`pnpv2-20261117` as the only writable processed directory (`--release`; dsreset bundles keep `dsreset-20261113`):

1. `--entry raw-proof` once the raw upload is complete (every remote copy's sha256 equals the manifest).
2. `--entry phase-a-remote`: 100 jobs `pnpv2-20261117-prepA-<episode>` (ingest, native depth, FoundationStereo,
   SAM3), remote bundle `pnpv2-20261117-phaseA`, local attachment `<work>/phaseA` (what `status` reads).
3. `--entry rgb-remote`: one job `pnpv2-20261117-rgbpub-pick-and-place-v2-S` (bundle `pnpv2-20261117-rgb`, attachment
   `<work>/rgb-remote`). Its tick admits it only when the Phase-A state mirror marks all 100 jobs done; it runs
   `python -m ops.jobs.pnpv2_rgb remote` in the prep image: the same export/contracts/gate as `publish-rgb`, written
   into `<remote processed>/learning-inputs/pick-and-place-v2/S/rgb` through a private staging directory.
4. Ticks, detached, from main: `python -m ops.jobs.rwprep_remote tick --config <bundle>/dispatcher.json --apply --loop`.
5. `--entry rgb-fetch` copies only the bundle documents (never the episodes) into `<work>/publication/S/rgb`, then
   `--entry upload -- --view rgb` publishes them to the canonical directory with the receipt `status` checks.

Geometry on the same route: `--entry calib-remote` compiles the established LoMa-G plan (stem `pnpv2-20261117-calib`,
upstream `configs/dsreset-upstream-20261114`, automatic G1-G6, no override) inside the remote planner and attaches
`<work>/calib`; its view job waits on the Phase-A state mirror. After its publish job, `--entry rig-fetch` mirrors the
accepted rig + reference locally (what `status` checks) and `--entry phase-b-remote` compiles Phase B
(`pnpv2-20261117-prepB-*`) on the remote against that rig; `attach` gives every pnpv2 bundle the pnpv2 release.
After all Phase-B jobs are done, the geometry route is:

1. `--entry geometry-remote` (stage only): attachment `runs/pnpv2-20261117-geometry`, linked from
   `<work>/geometry-remote`. It freezes only this collection's ingest entries (default source
   `/home/ws/ws/realworld-ops/ingest/manifest.json`; optional `-- --source <absolute manifest>`), the canonical
   membership and source hashes. It uses the **current main commit**, because Phase A's older code lacks the worker.
2. From main, detach `python -m ops.jobs.rwprep_remote tick --config <work>/geometry-remote/dispatcher.json --apply --loop`.
   The single job `pnpv2-20261117-geopub-pick-and-place-v2-S` waits for the real Phase-A/B state mirrors; the worker
   rechecks exact membership, queue/config hashes, done states, STOPs and automatic rig acceptance. It independently
   rehashes remote raw files, calls the existing v2 snapshot adapter, plans calibration records, previews/applies them
   through `calib.release` to the **new snapshot**, and uses the same geometry publisher as `--entry publish`.
3. The immutable five-file geometry bundle is published under
   `<remote processed>/learning-inputs/pick-and-place-v2/S/geometry/`. It references the existing compact Phase-B stores,
   so no episode copies are made remotely. Snapshot, calibration plan, input binding, raw proof and measurement live
   under `<remote processed>/geometry-publication/pick-and-place-v2/S/`. The measurement uses `SampledNativeStore`
   (8 native frames/camera on the 90 training episodes, existing estimator and validation), then a completion marker
   hashes all publication documents, the snapshot, measurement, rig and reference. Private building directories are
   renamed only after validation; complete retries check hashes and gates; partial/conflicting output stays held.
4. After that job is done, `--entry geometry-fetch` uses `realworld_local.fetch` with dependency priority, expands the
   compact descriptor to every episode H5/completion file, and fully verifies the closure in `runs/realworld-inputs`.
   It preserves remote descriptors under `_remote`, checks the localized store's actual calibration gates, verifies the
   completion marker and canonical membership, then writes equality-only local copies of staging documents,
   `<work>/ingest-snapshot`, the rig/reference and `<processed>/measurements/pick-and-place-v2/scene.json`.
5. `--entry upload -- --view geometry` checks the fetch receipt and staging hashes before uploading verified copies
   from repository `runs/`. It keeps the established canonical `learning_inputs/pnpv2-20261117/.../S/geometry` path and
   upload receipt, and mirrors those canonical row-input paths with their compact closure so local rows resolve them.
   Status requires both the fetched processed closure and canonical mirror receipts for this remote route.
6. `scripts/pnpv2.py status` reports `geometry.ready_after_check_data` after the existing RGB, rig, Phase-B,
   geometry-publication and scene gates pass. Registration still requires a fresh successful `check-data`.

These entries do not tick existing controllers, register/place rows, edit pauses, or contact the Robot PC.
Keep the approved MIG environment used by other producer entries (the bootstrap also guards CPU-only commands).
The local-host `snapshot` / `publish` entries remain available and share the same publication implementation.

## Guarantees

- No row exists before check-data passes on the real recordings; each wave only after its inputs exist.
- v2 ids (rows and prep-dispatcher jobs) never start with a v1 prefix; the established producers' default
  `rwv2-20261113-*-pick-and-place-v2-*` ids would, hence `--id-stem`. The registrar refuses if the v1 pause is
  inactive, matches a v2 id, or any v1 queue row is not held by it; v2 rows never depend on or read a v1 run.
- No calibration acceptance override; S outcomes all success; RGB and geometry publications share one split.
- Nothing contacts the Robot PC; `status`/`check-data`/dry runs write nothing; uploads refuse to overwrite.
- Robot evaluation is out of scope: `eval` rows are the offline artifact check (1M steps, final held-out metrics).

Tests: `tests/test_pick_and_place_v2.py` (science parity, spec, runtime, readiness barrier) and
`tests/test_pnpv2_pipeline.py` (arrival, check-data, waves, registrar dry run/execute/refusals, v1 isolation,
producer stems, resolvers).

Remote geometry tests: `tests/test_pnpv2_geometry_route.py` (CPU shard; synthetic remote inventories and compact closure,
real snapshot/release writer integration marked `local_host_data`), plus the existing pnpv2 remote/pipeline tests.
