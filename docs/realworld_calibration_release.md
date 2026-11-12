# Real-world calibration release (`rig_loma`, 2026-11-08)

Release command: `python -m s4d.realworld.calibration_release` (the accepted release 6c3b9ec has only this path; in later code it is a module-object alias of `s4d.realworld.calib.release` with the original arguments and usage). Rig documents are handled by `s4d.realworld.calib.rig`, stages by `s4d.realworld.stages`, and robot geometry by `s4d.realworld.robotics`.

Status: release CODE (branch `rw-calib-rel-20261108`); no `calibration_validated` record has been written, no dataset
regenerated and no job registered - the coordinator runs the runbook below. Working notes and evidence:
`/home/ws/ws/realworld-ops/calib-loma/rel/` (NOTES.md, `rig/` published documents, `results/`). Calibration method and
held-out study: [realworld_rig_loma.md](realworld_rig_loma.md), `realworld-ops/calib-loma/loma/{FREEZE.md,
HELDOUT_REPORT.md}`.

## 1. What is released

One fixed rig per task (no per-episode pose refinement anywhere): the SAM-free LoMa-G rig calibration `rig_loma` of
frozen commit a935aef, FoundationStereo depth, poses of the `all` refit on all 100 episodes of the task. Published by
`python -m s4d.realworld.calib.loma.publish publish` as `calibration_rig` documents (format
`s4d.realworld.rig_calibration/1`, consumed by the calibrate rig mode):

| task | document (rel/rig/) | cameras | depth_scale fs / sensor / s2m2 (070988; 073201; 076054) |
|---|---|---|---|
| drawer-open | `rig-loma-drawer-open.json` (name `rig-loma-drawer-open-a935aef`, sha256 6a9be3db...) | all validated | 0.9968 / 0.9989 / 0.9979; 1.0142 / 1.0003 / 1.0149; 1.0249 / 1.0172 / 1.0267 |
| mug-hanging | `rig-loma-mug-hanging.json` (name `rig-loma-mug-hanging-a935aef`, sha256 d83637e7...) | 076054 validated; 070988 / 073201 held by G5a only, exception-eligible, exception EMPTY | 0.9948 / 0.9974 / 0.9955; 1.0116 / 0.9983 / 1.0120; 1.0252 / 1.0172 / 1.0271 |

Each document records: per camera `T_cam_base`, `depth_scale` {stage: exp(alpha)} (fs from the published solve;
sensor / s2m2 from their independent frozen solves, published only because their poses agree with fs within 0.05 deg /
1 mm - measured <= 0.015 deg / 0.22 mm), `checks` G1-G6, `held_by`, held-out validation, SAM G6 as `diagnostics.g6_sam`,
the per-task gripper-mount estimate (diagnostic, NOT applied: the frozen validation used the nominal mount);
`reference_sha256` (+ `.reference.npz`, the G5a registration reference of `calib-dw/results/fixedness-1seg-<task>`);
`fixedness_rule` `g5ab/1`, `fixedness_config` (the G5a configuration: 1 segment), `fixedness_evidence` (per camera and
episode: G5a registration and G5b epipolar class of the verdict), `fixedness_exception`, `episode_checks` (robot depth
<= 12 mm, explained >= 0.70 = rig_dw RigConfig), `robot_model` (`robotiq_2f85_register_3_230/2`, visual meshes),
`gate_version` and `provenance` (frozen commit a935aef8, sha256 of FREEZE.md, the frozen verdict JSON, full.json,
all.json, fixedness inputs and depth-source results; the user decision text). The publisher recomputes the G1-G6
verdict from the stored results and refuses to publish unless it equals the frozen verdict JSON exactly; it never
replaces an existing document.

## 2. Gate change (user decision update 2026-11-08 ~22:00 KST) and justification

`gate_version` = `rig_loma-release/1: G1-G5 = rig_dw DESIGN v1.1 (G5 amended), thresholds unchanged; G6 = SAM-free
corr_challenge primary; SAM G6 diagnostic`.

* **G6 primary = SAM-free `corr_challenge`** exactly as in frozen commit a935aef (`rig_loma_run.py` at that commit, now
  `calib/loma/run.py`; committed before the freeze record, untuned): 12 common rig perturbations of the G6 tolerance size (+-0.35 deg about the base axes
  through the workspace pivot, +-8 mm along them), scored on held-out LoMa-G robot 2D-3D correspondences; each must be
  rejected by frame AND episode one-sided sign tests (p < 0.01) with a frame majority. Held-out (fs): all 12 rejected
  for both tasks (drawer weakest rotx- 542/720 frames, 20/20 episodes). The publisher re-derives every `rejected` flag
  from the stored counts and refuses inconsistent records.
* **SAM G6 = diagnostic.** drawer-open failed it only on roty+ (242/445 frames, p = 0.036 > 0.01; episode test 18/20,
  p = 2e-4). Justification (FREEZE.md): the calibration is SAM-free by directive; SAM silhouettes are a different
  evidence type with their own boundary bias (+-2 px mask change = 1.4-2 % scale), so the SAM-based perturbation test is
  kept and reported, but the primary criterion uses the evidence the method is built on, on held-out episodes. Caveat
  kept from the held-out report: the SAM-free challenge is the same evidence type as the fit (held-out episodes only).
* G1-G5 unchanged (no threshold relaxed). Effective verdict: drawer-open all three cameras; mug-hanging 076054; 070988 /
  073201 pending the fixedness exception (section 3).

## 3. Camera fixedness in rig mode and the fixedness exception

Rig mode (`calibrate` with `rig_calibration=<document>`, `calib.rig.release_fixedness`) applies the verdict's rule
per episode instead of the strict 4-segment test (which fails its own reference, DESIGN v1.1 G5 amendment):

* G5a: the episode's whole-episode-median image is registered to the document's reference with its `fixedness_config`;
  motion is detected only when the registration is VERIFIABLE (>= 6 and >= 60 % consistent patches). An unverifiable
  registration is reported (`registration.verifiable = false`), never counted as moved (as in the verdict: drawer-open
  97-99 / 100 episodes are not verifiable). Registration metrics (`segments`, shifts, inliers) are always reported.
* G5b: the per-episode epipolar consistency of the rig solve stored in `fixedness_evidence`; an episode flagged `moved`,
  or an episode the document has no evidence for, is rejected.
* A camera published `held` with `held_by == ["G5a"]` is rejected in EVERY episode ("held by the registration-
  fixedness check G5a ...; no valid fixedness exception") until the document carries a valid exception; with a valid
  exception, episodes whose registration detects motion are accepted and the note "covered by the fixedness exception"
  is recorded. G5b violations are never covered.

**Exception procedure** (mug-hanging 070988 / 073201; FIXREG stream = independent pre-registered static-RGB feature
registration on the existing recordings):

1. FIXREG writes its method + criterion before results, then its verdict file(s).
2. If there is no credible evidence of physical camera movement, the coordinator records the decision and creates a NEW
   document (the published one is never modified; its sha256 is recorded as `supersedes`):

       python -m s4d.realworld.calib.loma.publish fill-exception --rig rel/rig/rig-loma-mug-hanging.json \
           --cameras 135122070988/color 135122073201/color --evidence <FIXREG verdict json> [<report>] \
           --decision accepted --decided-by coordinator --reference "<HANDOFF line / FIXREG report>" \
           --out rel/rig/rig-loma-mug-hanging-fx.json

   (sha256 of every evidence file is computed now; only eligible cameras; `--decision rejected` keeps them held.)
3. A valid exception = eligible camera listed, decision `accepted`, every evidence entry with path + sha256, decided_by /
   decided_at / reference present (`calib.rig.fixedness_exception_problem`). The new document has a new sha256, so
   every calibrate record built with it gets a new fingerprint and build id.
4. If FIXREG finds movement: do not fill; the cameras stay held and mug-hanging geometry consumers stay held.

### 3a. Recording periods (user decision 2026-11-10 ~13:00, covering)

The user's instruction was to "handle credible recording-period camera movement explicitly rather than forcing one
shared calibration". A task recorded in several sessions can therefore get one document per recording period:

    python -m s4d.realworld.calib.loma.publish split-periods --rig rel/rig/rig-loma-<task>[-fx].json \
        --periods period_steps.json --cameras <cameras with a credible period step> --evidence <FIXREG results> \
        [<addendum>] --decided-by <who> --reference "<addendum / decision>" --out-dir rel/rig/periods

The input `period_steps.json` (`rwx-period-steps/1`) comes from the pre-registered analysis
`realworld-ops/rwx/tools/period_steps.py`. It runs on the frozen FIXREG registrations. Per camera, estimator B, rigid
part only (scale is common-mode thermal), it computes:

- within-period first differences and drift;
- the step of each period's median versus the base period, i.e. the period that dominates the shared solve. The step
  needs a permutation test p < 0.001, a size >= 0.10 px, and estimator A in agreement.

`split-periods` always refuses a step camera when:

- its step lacks the recorded evidence (D >= 0.10 px, p < 0.001, estimator A agreeing);
- its omega does not reproduce;
- its step is above `MAX_PERIOD_ROT_DEG` = 0.035 deg (1/10 of the tolerance);
- it has a camera-specific scale flag.

It also refuses on the open problems of the pre-registered rule, unless the user's decision is passed as
`--user-override` (recorded with the problems in every period document):

- within-period movement of any camera;
- an inconclusive step or an unresolved scale flag of any camera;
- a missing decision table, or one that does not allow the split (`split_periods`, `period_pose_cameras`).

Otherwise every period document is the published document with these changes:

- only its period's episodes (`episodes`, G5b `fixedness_evidence`, `pool_frames`);
- a `period` block (base document name and sha256, evidence with sha256, decision);
- for a camera with a credible step, the pose `exp([omega]x) @ T_cam_base`, where omega is the camera rotation whose
  image field best matches the measured rigid step (`rotation_from_rigid`, least squares over FIXREG's grid). It is
  recorded in `period_pose` together with the base pose.

The verdict, depth scales, robot model, reference arrays and fixedness exception are unchanged; fill the exception
before splitting. The format is still `s4d.realworld.rig_calibration/1`, so the released rig mode needs no change.

- Rig mode rejects every episode of another period, because the document has no G5b evidence for it.
- Phase B selects each episode's period document in its calibrate config.
- `calib.release plan --rig <task>=<period A> --rig <task>=<period B>` checks the following and fails closed:
  - the documents come from ONE split-periods run of one published document (identical `period` blocks with a source
    name and sha256);
  - every period pose re-verifies, and every other camera carries the base pose;
  - each document's episodes and G5b evidence equal its period;
  - the periods partition exactly the task's valid recordings;
  - each recording is matched by its own episode id (`<task>-<index>`); its calibrate outputs must be that
    episode's, and its calibrate record must have been built with its period's document.
- The records name the period (`metrics.period`).

### 3b. Criteria v2 (`rig_loma-release/2`, agent CALSYNC, user directive 2026-11-12 ~20:40 KST)

The user asked to re-evaluate the calibration and camera-synchronization acceptance criteria from evidence, revise
overly conservative thresholds only with quantitative support, keep the original measurements and verdicts, and
document every change. `s4d/realworld/calib/criteria.py` implements release/2. Release/1 stays the DEFAULT of
`rig_loma_publish publish`. Release/2 is chosen explicitly with `--criteria v2 --static-eval S [--fixreg F]`. A
release/2 document still recomputes the release/1 verdict, checks it against the frozen verdict JSON and stores it,
together with the release/1 per-episode evidence, in its `criteria` block. Full numbers:
realworld-ops/rwx/CALSYNC-REPORT-20261112.md.

Measured on all 400 raw episodes (drawer-open, mug-hanging, covering, pick-and-place):

* The three D435 colour streams run free at 66.715 ms. There are no within-segment drops.
* Inter-camera phase offsets are constant within an episode (std <= 0.05 ms) and spread over 0-33 ms (half a period).
* Flange speed: p95 0.10 m/s, max 0.125 m/s. At the worst offset of 33 ms the flange therefore moves <= 4 mm, which is
  ~2.3 px at the 1.07 m camera distance (0.6 training px).

What changes (everything else is release/1). Every check fails closed on missing, malformed or non-finite evidence.

1. **Static-correspondence evidence for G4 / G5b: pairwise-nearest pairing.**
   * The problem: `romav2_match_worker` paired every camera with its frame nearest the reference camera's sample time
     and dropped pairs more than 20 ms apart. With constant offsets of 0-33 ms this dropped whole pairs for whole
     episodes. Two cameras on opposite sides of the reference were up to a full period apart even when their own
     nearest frames were < 10 ms apart; this is pick-and-place 000009-000012, where no pair was left, so "G5b:
     per-episode consistency incomplete" held all three cameras. A camera was G5b-verifiable (both of its pairs
     present) in only 35-92 % of the episodes, so the 20 ms rule weakened the fixedness test.
   * The new pairing: `--pairing pairwise` takes a's frame nearest the sample time and b's frame nearest a's frame,
     kept iff |dt| <= half the frame period. A larger gap means a missing frame or a pause; that (time, pair) is
     skipped and recorded.
   * `--supplement-of <release/1 matches>` adds only the missing (time, pair) combinations, each with its own seed:
     * the release/1 files stay bitwise unchanged;
     * the frames of every combination they already evaluated must equal the pairwise frames;
     * a supplement tolerance that would drop an existing combination is refused.
   * `criteria static-eval` re-evaluates G4 and G5b at the stored full-solve state from the union of both files.
     * Inputs: it validates them (`match_inputs`): exactly one release/1 file per episode; supplements bound to it by
       `supplement_of` sha256; no (time, pair) combination in two files; frames, camera order and dt checked against
       the episode's own timestamps. It hashes every input, the K sources and its code, and re-hashes them where they
       are (the remote job).
     * Publication: the publisher checks the strict schema (episodes and held-out set equal the solve's; path +
       sha256 per input; K-source and code identity), re-hashes every input and K-source file, re-checks each
       supplement's ancestry, and RECOMPUTES the static evaluation from those verified sources with its own code:
       every measurement, the dt table (derived from the files), the manifest and the code identity must be equal.
       It therefore runs where the evidence is (a remote CPU container, or with a complete mirror); any missing,
       changed or inconsistent input refuses publication. G4 needs finite, non-negative distances and >= 30 matches per
       pair.
     * Reproduction: it recomputes the release/1 values from the release/1 files alone. The publisher and `verdict`
       compare these with `full.json` (exact equality) instead of trusting a cached flag. The solves are reused (the
       estimator is unchanged).
   * Match files are validated frame by frame against the episode's own normalized timestamps:
     * camera order;
     * release/1 frames by the reference rule;
     * supplement frames by the pairwise plan, using half the measured frame period.
   * The G4 / G5b thresholds are unchanged. New and stricter:
     * every episode has evidence;
     * a camera's TWO pairs are named from the camera list (never discovered from the evidence). Each camera must be
       verifiable in >= 90 % of the episodes: both pairs present, finite, non-negative, >= 30 matches.
     * The dt extension is validated on the data (`dt_extension`):
       * the dt table reconciles exactly with the measurements;
       * per camera pair, the long-offset (> 20 ms) per-episode medians are compared with >= 10 short-offset
         measurements of the SAME pair (no pooling; fewer -> the pair fails). They may exceed them by <= 0.05 px at the
         median (upper bound of a 95 % bootstrap interval) and <= 0.10 px at p90;
       * no long-offset episode-pair may violate the per-episode G5b rule against the short-offset baseline
         (release/1 pick-and-place and covering: 0 violations).

       Over 0-20 ms the measured slope is -0.007..+0.014 px per 10 ms (935 episode-pairs).
2. **G1 as the stability radius about the published solution, plus an image-space check.**
   * The radius: every solve of the complete, unchanged stability set must lie within 0.35 deg and 8 mm (unchanged)
     of the published `all` pose.
   * Release/1 used the max PAIRWISE distance (diameter) between two perturbed solves, neither of them published.
     Keeping the numbers on the radius admits up to twice the former diameter. Also, 8 mm is ~1.2 training px at the
     measured 1.07 m camera distance (the derivation assumed 1.3 m). Hence the image-space constraint below, which is
     the binding downstream quantity.
   * The image check is an engineering BACKSTOP, not an independently validated acceptance envelope (the historical
     rig_dw rejections come from the rotation radius). Probe = a 16^3 lattice over the 3D-MVP scene box
     (`MVP3D_BOUNDS`, measured on drawer-open / mug-hanging / covering; a new task's containment is re-checked on its
     store before 3D-MVP), in front of the camera and inside the 480 x 480 training crop. For every stability solve, the reprojection displacement against
     the published pose must have:
     * p50 <= 1 training px (3.75 px);
     * p95 <= 2 training px (7.5 px). Release/1's separate one-training-px rotation and translation tolerances jointly
       admit two.
   * Probe sensitivity: on camera 073201 the p95 is 3.6-4.7 px depending on the depth weighting, in every task
     including the accepted drawer-open / mug-hanging rigs. A p95 at one training pixel is therefore not a robust
     gate. The p50 is stable (1.0-1.3 px).
   * It is not uniformly looser: an uncompensated 7.5 mm lateral translation passes the 8 mm radius and release/1 but
     fails the p50.
3. **G5a: unchanged.** The detection thresholds, the exception procedure and its FIXREG rule are release/1's.
   `g5a_materiality` reports decision support for a camera held by G5a alone, against 1/10 and 1/4 training px:
   * FIXREG per-episode 99 % bounds, corrected by the median and by the lowest observed injection recovery;
   * within-episode group maxima;
   * missing within-episode estimates;
   * unresolved candidates.

   Codex review 2 showed that these corrected bounds lack validated coverage near such budgets, so no eligibility is
   derived from them. The 1/10 level is below the instruments' resolution: it would hold 070988, which has no G5a flag
   (release/1 G5a passes it). A G5a-only camera therefore still needs the release/1 decision (coordinator or user), now with these
   numbers attached.

Sanity against known-bad or decided cases:

* rig_dw, held on 2026-11-08: 5 of 6 cameras still fail G1 v2 on rotation radius (0.51-0.84 deg). The sixth, drawer
  076054, was held by G3 anyway.
* G5a is unchanged, so earlier decisions (covering E1/E2, mug-hanging) are unaffected.
* Release/1-era evidence fails the new G5b coverage requirement (covering 78-92 %). Any re-release needs the
  pairwise supplement.

## 4. Per-camera depth scale (single source of truth = the accepted calibration record)

The rig solve estimates each camera's metric depth scale jointly with its pose (`log z = log d + alpha`); the pose is
consistent only with depth multiplied by exp(alpha) (device IR geometry: ~-0.3..+2.5 %, shared by FS / S2M2 / native,
s2m2/REPORT.md section 6). Rig mode copies the document's `depth_scale` map (and `robot_model`) into each ACCEPTED
`calibrate/<camera>.json`; consumers read it only through `pipeline`:

* `accepted_calibration_doc` / `accepted_calibration` (pose) / `calibration_depth_scale(root, ep, cam, stage)`: 1.0 for
  legacy records without a map or cameras without an accepted calibration; a record with a map but without `stage`
  raises (a scale is never borrowed from another depth source); factors outside 0.9-1.1 raise.
* `calibrated_view_depth` = `view_depth` x factor (raw `view_depth` / `load_view_depth` stay raw).
* Applied in: TrackEverything input depth (track2d / lift3d), training store (stored depth + gripper-visibility depth
  test of geometry cameras), window export (depth + gripper depth test), `vis_geometry` store, rwlearn
  `ProcessedEpisodeStore.depth`, viewer `view_rgbd`; track_motion receives it through lift3d.
* Fingerprints / versions: `calibrated_geometry_config` adds `depth_scale` {stage, factors, rule} (and `robot_model`)
  to a stage's config ONLY when a factor differs from 1 (or the robot model from the legacy one), and
  `calibrated_version` bumps the stage version with `+cal1` (`2-trackeverything+cal1`, `track_motion 1+cal1`,
  `train_data 2+cal1`, `export 1+cal1`, `vis_geometry 1+cal1`). Legacy calibrations keep their fingerprints, versions and
  bytes (train store byte-identity test), so existing outputs are never invalidated or rebuilt in place.
* The calibrate rig mode itself multiplies the episode depth of its `depth_stage` before the robot-depth checks and
  refuses a release document without a scale for that stage.

## 5. Gripper mapping (Robotiq 2F-85 register -> finger joint)

`robotics.model.GRIPPER_MAP_LEGACY` = `register_linear_255/1` (theta = reg / 255 x 0.8 rad) is wrong for this gripper;
`robotics.model.GRIPPER_MAP_2F85` = `robotiq_2f85_register_3_230/2` (theta = clip((reg - 3) / 227, 0, 1) x 0.8 rad). Evidence
(`rel/tools/gripper_evidence.py` -> `rel/results/gripper_register_evidence.json`, read-only scan of all 200 raw
recordings; `loma/notes/robot_model_check.md`; `loma/results/fz/g3corr_*.json`):

| evidence | observation |
|---|---|
| recorder convention | command closure c -> register 3 + 227 c (DROID; adapter note "Robotiq position 3 + 227 * closure") |
| open plateau | request 3 -> register exactly 3 with object status 3 (position reached) in 100 / 100 mug and 91 / 100 drawer episodes; the register equals the request whenever the position is reached (requests 4-62, 211) |
| closure plateau | every mug-hanging close command (request 230, all 100 episodes) stops at register 207-211 with object status 2 (contact while closing = object detected); no recording closes on nothing, so the closed end 230 rests on the convention + Robotiq (2F-85 reports ~227-230 fully closed in parallel mode) |
| URDF | pads touch at 0.80 rad, 85.0 mm gap at 0 |
| rendered pad width | legacy: open register 3 renders 84.2 mm (0.8 mm too closed), mug grasp (211) 15.9 mm gap (pads ~4 mm too open each); corrected: 85.0 mm and 7.8 mm |
| image evidence (held-out SAM, fs, supplementary) | mug fixed o2m 1.994 / 1.990 / 1.912 px -> 1.991 / 1.924 / 1.782 corrected (1.872 / 1.864 / 1.414 with the estimated mount); drawer unchanged (register <= 49) |

Implementation: `gripper_joint_value(..., mapping=)` (refuses the 2F-85 map for any other gripper / state semantics);
`RobotGeometry(meta, gripper_map=, surface=)` poses `q` with it; `closure()` (the proprioceptive closure stored for
learners) keeps the legacy map, so non-geometric consumers never change. The map reaches consumers only through the
accepted calibration record (`robot_model`, written by rig mode from the release document):
`calibrated_robot_geometry` in track_motion, export and the training store's gripper target points; the training
store's M5 set motion and probe state keep the legacy geometry (identical sets / proprioception). The map id enters
the fingerprint (`robot_model`) -> new build ids, version `+cal1`.

## 6. Visual meshes: fail closed + remote asset fix

* `robotics.visual.require_visual_meshes` raises `MissingVisualMeshError` listing every mesh geometry without a visual
  COLLADA mesh; `VisualSurface.build`, `calib.dw.frames.Surface` and `calib.loma.render.build_robot_mesh` fail closed
  unless `allow_collision_fallback=True` is passed explicitly (diagnostics / synthetic tests); `RobotSurface.
  build_visual` / `RobotGeometry(surface="visual")` never fall back. (Before: the remote asset copy had only collision
  meshes and every remote rig_dw run silently used them.)
* `scripts/realworld_add_visual_meshes.py` (stdlib, add-only; manifest `rel/tools/visual_meshes.sha256`): run once on
  2026-11-08 13:09 UTC on the remote host: source `calib-loma/assets/validation-20261031` -> target
  `/home/compu/kaist/sunho/realworld/assets/validation-20261031`: 20 files added (10 Franka fer `visual/*.dae`, 10
  Robotiq 2F-85 `visual/2f_85/*.dae`), every sha256 verified after the add, nothing overwritten or deleted, the two
  parent directories made writable only during the add and restored to 0555, new dirs 0555 / files 0444; the original
  `SHA256SUMS` still verifies 23 / 23. Report: `rel/results/visual_meshes_apply.json` (hashes = `loma/notes/
  visual_mesh_sha256.txt`).

## 7. calibration_validated records

`python -m s4d.realworld.calib.release plan|apply` (module docstring). Per task, every valid manifest episode
must have a done calibrate record (same sha256) whose on-disk record is a rig-mode build with EXACTLY the release
document (input hash + reference hash), build id = the manifest's, every document camera accepted with the document's
pose / depth scales / robot model, fixedness verified, and every document camera effectively validated (a G5a-held
camera only with a valid exception). Any failure refuses the whole task. Record: status done, `output_format`
`rw-calibration-validation/1`, `code_version` `calibration_validated@1+rig_release`, `input_sha256` = episode sha256,
`metrics` = {decision accepted, calibrate_build_id, cameras (serials), gate_version, shared_calibration {name, sha256,
reference_sha256}, evidence {frozen_commit, verdict_sha256, freeze_record_sha256, user_decision, calibrate_fingerprint,
fixedness_exception}, depth_scale, robot_model, per_camera {fixedness, held-out robot depth, explained, depth scale
applied}}. `apply` is a dry run unless `--apply`; it re-checks the live manifest (sha256, current calibrate build id)
and writes only through `ingest_worker.set_stage`.

## 8. Operator runbook (coordinator)

Preconditions: this branch released through the normal chain (`release_chain.py`), visual meshes present in the
asset copy (done), mug-hanging exception decided (section 3) before mug-hanging step 2. Never touch running jobs or the
existing release root; everything below writes NEW roots / records.

1. **Rig-mode calibrate per task into a new variant root that reuses depth and masks.** For every episode create
   `processed/calrel-<commit>/<task>/<ep>/episodes/<ep>/` with symlinks to the existing release stages
   (`normalized`, `derived/sensor_depth`, `derived/foundation_stereo_depth`, `derived/sam3_masks`; `run_stage` refuses
   to rebuild a symlinked stage, so they are reused, never modified). Root config = the release config with

       stages.calibrate: {evidence: rgbd, depth_stage: foundation_stereo_depth, robot_mask_stage: sam3_masks,
                          every: 2, max_frames: 200, landmark_links: [], overlays: 4,
                          rig_calibration: <rel/rig/rig-loma-<task>[-fx].json (copied to the remote)>}

   and run `s4d.realworld.cli run --config <cfg> --episode <ep> --stages calibrate` (container with the asset copy
   mounted; the document's robot model needs the visual meshes and fails closed otherwise). Expected per camera:
   `init_method rig_shared`, `status accepted`, `diagnostics.release`, `depth_scale`, `robot_model`.
2. **Records.** Sync the new calibrate stage records into the manifest (PRE dispatcher / `sync_stage_records.py`, so
   `stages.calibrate` names the new root and build id), then on the remote (read-only mounts):
   `python -m s4d.realworld.calib.release plan --manifest <copy> --rig <task>=<doc> --path-map ... --out
   plan.json`; locally `... apply --plan plan.json --state-dir /home/ws/ws/realworld-ops/ingest` (dry run), then with
   `--apply`. A refused task writes nothing; fix the cause, never edit records by hand.
3. **Regenerate geometry stages** in the same variant root (reusing depth / masks):
   `--stages track_everything,track_motion,train_data[,vis_geometry]` (versions `+cal1`, new build ids; the training
   store is written to `<root>/train_data/<ep>`), then the per-episode QA with families (rwprep qa, rwprep-qa/3) and
   sync the stage records (track2d, lift3d, track_motion, train_data, qa) into the manifest. The gate requires
   `calibration_validated.metrics.calibrate_build_id` = the current calibrate build id: rebuild order 1 -> 2 -> 3,
   and do not re-run calibrate afterwards (a new build id would void the records).
4. **Gate report:** `python -I scripts/realworld_learning.py gate --out <report>`; check `calibration_hold[task].passed`
   (100 / 100 validated) and that trainers' `check_store_calibration` sees the new stores (upstream calibrate = the
   validated build id, every store camera validated).
5. **Scheduler:** `python -I scripts/realworld_learning.py plan --tasks <passed tasks> --commit <release> --out
   plan.yaml` (rows only for consumers whose gate level passed), review, register through the normal scheduler
   workflow (`scripts/jobs.py`), priorities per the campaign rules. Running non-geometric experiments keep their
   existing stores (their non-geometric content is identical in the new stores).

## 9. Risks / limitations

* The gripper-mount estimate is not applied (nominal mount everywhere, as in the frozen SAM validation); gripper
  surface error up to ~3 mm / 1.5 deg.
* G5a is unverifiable for most drawer-open episodes and G5b lacks one camera pair in many mug episodes: per-episode
  fixedness rests on "no detected motion" (the verdict's rule), not on positive verification of every episode.
* Depth scales are per depth source; a new depth stage needs its own solve before it can be used with these poses.
* The verdict's G2 (robot depth <= 12 mm, explained >= 0.70) was evaluated pooled over the held-out frames; rig mode
  applies the same thresholds PER EPISODE (with the corrected gripper map and visual meshes, depth x scale). A single
  episode can therefore be rejected (e.g. little robot visibility); the builder then refuses the whole task and the
  coordinator must look at that episode - never relax `episode_checks` silently. Use `depth_stage:
  foundation_stereo_depth` in rig mode (the verdict's G2 depth); `sensor_depth` works with its own published scale.
* Downstream consumers that pose the robot (track_motion, training-store gripper targets, export) use the accepted
  record's robot model, i.e. the visual meshes: their containers need the asset copy with the visual meshes (added on
  the remote; the local copy has them) and fail closed otherwise.
* Not covered by the uncertainty spreads: common base-frame yaw / robot-scale confound, LoMa-G template bias.

## 10. Historical S2M2 inference identity (retired 2026-11-09)

The 2026-11-08 study did not adopt S2M2; FoundationStereo was kept. The inference worker, stage wrapper,
executable CLI registration and active pin were removed on 2026-11-09. Study evidence remains in
`realworld-ops/calib-loma/s2m2/REPORT.md`; the former pin block is preserved verbatim below for reproduction
with the frozen study source. It is historical, not an available inference backend. The isolated study
environment was outside the rwprep image (`calib-loma/s2m2/env/s2m2` on the remote host).

Published `s2m2_depth` scale entries remain part of the complete calibration documents and accepted records.
Their validation, existing-stage status recognition and generic depth-stage readers remain supported.

```yaml
s2m2:   # CANDIDATE own-rig depth (study 2026-11-08, realworld-ops/calib-loma/s2m2); not in the standard stage graph
  repo: https://github.com/junhong-3dv/s2m2
  commit: cbb5d169981e3086be1f959db5a8a0f139adcb17   # 2026-10-16 (default-branch HEAD on 2026-11-08)
  license: "CC BY-NC 4.0 (LICENSE; README: non-commercial research and education only)"
  environment: >-
    isolated s2m2 venv (uv 0.9.7, CPython 3.11.14 of the rwprep image; torch 2.10.0+cu129, torchvision 0.25.0+cu129,
    numpy 2.4.6, opencv-python-headless 5.0.0.93); upstream imported from <repo>/src (no pip install)
  checkpoint_source: https://huggingface.co/minimok/s2m2 (revision 880bc49f2a7cf54a42570f8550c46cb57454bac9, cc-by-nc-4.0)
  checkpoints:   # sha256 of the files = the Hugging Face LFS sha256
    CH128NTR1.pth: 93f7177d8159d6666506df3abb0cdd75473614fab0e39f25e397acade0f9e241   # S
    CH192NTR2.pth: 6e767e0977dd0a81a51021429f82b1c8451fd2534862170cb18a0019a33ae8ad   # M
    CH256NTR3.pth: 770e822272585f843288507a2274349f032f2ef8106d1370bdefa8849b37cf87   # L
    CH384NTR3.pth: 78dee08652ac6b6d42ff60d2ccd8184021b1e7beb93b091f379f82ba9f0b3032   # XL
  entrypoint_reference: >-
    src/s2m2/core/utils/model_utils.py load_model + run_stereo_matching (image_pad(32), autocast fp16, image_crop);
    model inputs float 0..255 3-channel; outputs disparity (left reference), occlusion (0 = occluded), confidence
  notes: >-
    worker s4d/realworld/workers/s2m2_worker.py loads the checkpoint strictly (upstream my_load_state_dict skips
    mismatched keys silently); grayscale IR replicated to 3 channels as model input only; right reference by flip-and-swap
```
