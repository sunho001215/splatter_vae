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
| geometry publication | `--entry raw-proof`, `--entry snapshot`, `realworld_prep_followup.py records --id-stem pnpv2-20261117-qa`, `calib.release apply`, `--entry publish`, `--entry upload -- --view geometry` | geometry consumers pass (per-episode `calibration_validated`), same split/outcomes as RGB, receipt |
| scene | `realworld_prep_followup.py measure --id-stem pnpv2-20261117-qa` | accepted train-only native measurement bound to the manifest and rig |

Queue rows (88; ids `pnpv2-20261117-<stage>-<method>-pick-and-place-v2-S[-s<seed>]`):

| Wave | Needs | Rows |
|---|---|---|
| **rgb** (DynaMo, ReViWo, E2E CNN, E2E CNN noaug) | check-data, ingest, Phase A, RGB publication; **never** rig/Phase B | gate 1, pre 2, export 2, fqldata 4, fql 12, eval 12 = **33** |
| **geometry** (Ours orig/ch01 scratch400k, SinCro, SNeRL, 3D-MVP single/multiview) | the above + rig, Phase B, geometry publication, scene | gate 1, stats 1, native-depth 1, pre 5, export 5, fqldata 6, fql 18, eval 18 = **55** |

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
