# Resume: real-data validation (items 5/6/8), refactor pass 2, merge into `splatter4d`

Written 2026-10-31 at 23:22 KST, when the user asked for all local work to stop within 20 minutes.
- No local process is running.
- The remote runner (`~/kaist/sunho/remote_runner`) was not touched. It stays the autonomous remote launcher; see
  `docs/RESTART_BRIEFING_20261031.md`.
- Nothing paused was resumed.
- Codex is unavailable; Claude does all reviews.

## 1. Branches (all pushed to origin, normal push)

| Branch | Head | Content | State |
|---|---|---|---|
| `realworld-preproc-20261031` | 81b202c | items 3–6 + DROID raw adapter + 7 integration fixes + diagnostics | real-data validated (dev run); **baseline run not finished** |
| `droid-largescale-20261031` | 9f0a80a | item 8 + 9 real-data integration fixes | real-data validated by the agent; suite 1283/0 on 87b7846 |
| `refactor2-20261031` | f2b6161 | refactor pass 2 (18 drivers into `ops/`) | suite 1277/0 |
| `integrate-20261031` | a014e4d | 127092c + merges of the three branches above + realworld `workers` package refactor (9abb93b) + test fix (a014e4d) | quick tests 166 passed; **full suite not yet run** |

`splatter4d` itself has only the bookkeeping commits; nothing is merged into it yet.
- Worktree of the integration branch: `.worktrees/integrate`.
- Worktree of the realworld branch: `.worktrees/realworld-preproc`.
- Reports:
  - `/home/ws/ws/validation-20261031/item8/REPORT.md`
  - `/home/ws/ws/validation-20261031/refactor2/REPORT.md`
  - `docs/realworld_droid_validation.md` on the realworld branch (stereo, tracks and SAM 3 results are in
    `docs/EXPERIMENT_LOG.md`, 2026-10-31 22:40–23:22).

## 2. What is complete

**Item 5 (dev run in `/home/ws/ws/validation-20261031/item5/root`, revision before 81b202c).** Diagnostics are in
`item5/diagnostics/*.json|png`.
- Rectification: matches the SDK to ≤ 0.06 px dy.
- FoundationStereo, both rigs and both references:
  - vs SIFT disparity: median −0.02 / −0.04 px;
  - left/right consistency: median 0.32 / 0.22 px;
  - cross-rig metric consistency: median 0.75 % / 0.82 %;
  - PointWorld depth: median 0.61 % / 0.50 %.
- AllTracker: temporal indexing OK; 84 % / 90 % of points valid after lifting.
- Lifting: static tracks drift a median 7 / 4 mm; agreement with the other rig's depth is a median 0.6 %.

**Item 6.**
- DINO worker on real frames: equals SNeRL's own `dino_target`, contract checked in `third_party/snerl`.
- SAM 3 stage, video mode:
  - grid aligned; box vs mask IoU 0.985;
  - temporal ids stable;
  - absences recorded explicitly;
  - SNeRL consumer cross-entropy finite.
- The `dino_snerl` **stage** has not run, because it needs the export stage; see §3.

**Item 8.** All checks are in the agent's report.
- Verified on real data: index, DROID-3D, join and geometry, batch contract, trainer smoke with resume, loader.
- Partial: the wrist pose chain and QA, and the anchor statistics.

**Refactor pass 2.** Done: `ops/` for the operational code, plus the realworld `workers` package.

## 3. What remains, in order

### 3.1 Item 5/6 baseline on the realworld branch head 81b202c

This adds the robot assets and runs the export and `dino_snerl` stages for the first time. The previous attempt was
stopped by the user during `stereo_depth`; ingest is complete in that root and is skipped on resume.

```bash
/home/ws/ws/validation-20261031/item5/run_all.sh /home/ws/ws/splatter4d/.worktrees/realworld-preproc baseline-81b202c
# about 50 min: stereo 34 min (MIG-8da55219), track2d 5, lift3d 1, export, sam3_masks ~8 (MIG-02b55a66),
# dino_snerl (MIG-423203d9), then the diagnostics (rectify, stereo, tracks, sam3; export in the repo venv)
```

- Run it as a tracked background task. Check memory first with
  `.venv/bin/python -I runs/setup/tools/memory_watch.py fits 20000000000`.
- If **export** fails, fix it on `realworld-preproc-20261031` in its own commit (`realworld integration fix: …`),
  merge into `integrate-20261031`, and rerun. Stages whose fingerprint matches are skipped.
- Robot assets:
  - URDF: `/home/ws/ws/validation-20261031/robot_assets/fr3_robotiq_2f85_droid.urdf` plus its `.manifest.json`.
  - The DROID mount (Franka external_gripper_example: xyz 0, yaw +π/2) is an **assumption**.
  - `diagnostics/export.json` reports `gripper_points_on_sam3_robot_mask` as the check of that assumption.
- After it passes, complete `docs/realworld_droid_validation.md` (export, DINO stage, status matrix) on the realworld
  branch and merge again.

### 3.2 Full suite on the integration branch

The gate records into that worktree's `docs/tests.*`.

```bash
cd /home/ws/ws/splatter4d/.worktrees/integrate
/home/ws/ws/splatter4d/.venv/bin/python -I /home/ws/ws/splatter4d/runs/setup/tools/memory_watch.py fits 40000000000
CUDA_VISIBLE_DEVICES=MIG-05a151e7-c6bd-5344-91cb-8bfb3a342631 nice -n 10 /home/ws/ws/splatter4d/.venv/bin/python -I scripts/run_tests.py
```

- Submodules and the DINO cache need to be present: `git submodule update --init` from `.git/modules` via
  `url.insteadOf`, and link `.cache/torch/hub/checkpoints/dino_deitsmall8_pretrain.pth` from main, as for refactor2.
- Commit the evidence (`docs/tests.json`, `.junit.xml`, `.log`). The sources must equal the tested revision.

### 3.3 Real-data checks on that exact revision

```bash
/home/ws/ws/validation-20261031/item5/run_all.sh /home/ws/ws/splatter4d/.worktrees/integrate merged-<sha>
/home/ws/ws/splatter4d/.venv/bin/python -I /home/ws/ws/validation-20261031/item5/compare_runs.py baseline-81b202c merged-<sha>
/home/ws/ws/validation-20261031/item8/rerun.sh /home/ws/ws/splatter4d/.worktrees/integrate merged-<sha> MIG-c3511b41-5f78-52f1-9b93-9c6794cdbbdd
```

- The item 8 rerun covers: manifest, `validate_batch` contract, loader workers 0/4/8, trainer smoke with stop/resume.
- Compare its outputs with `item8/checks/*` and `item8/checks/smoke/smoke_analysis.json`.
- Also run:
  - an import sweep of `s4d` and `ops`;
  - `--help` of `scripts/*.py`, `python -m s4d.realworld.cli --help` and `scripts/droid_ls_*.py --help`;
  - config loading of every `configs/**/*.yaml` through the trainer's loader.

### 3.4 Merge and push

The tested integration head fast-forwards `splatter4d`:

```bash
cd /home/ws/ws/splatter4d   # main checkout
git merge --ff-only integrate-20261031
timeout 120 git -c credential.helper= -c credential.helper='!gh auth git-credential' push -q origin splatter4d
```

- Fast-forward only, so the tested revision is the branch head. If `splatter4d` gained commits meanwhile, merge them
  into the integration branch first and repeat §3.2–3.3.
- Only bookkeeping files may be uncommitted in main at that point.
- Do not deploy to running experiments: remote releases stay pinned (no hot deployment).

### 3.5 Clean-checkout verification

```bash
git clone --branch splatter4d <origin> /home/ws/ws/validation-20261031/clean_clone
```

- Initialise the submodules.
- Run the import sweep, `--help` of the CLIs and the realworld/droid/ops tests, and record the HEAD sha.

## 4. Decisions for the user (from the item 8 agent)

1. **Wrist pose chain and QA.**
   - On FR3-295341-1326595, the published DROID-3D chain (with `dT_gw`) fits worse than the `dq`-only variant and
     than DROID's own wrist pose.
   - The 3 cm QA rejects 62/62 episodes, mostly because the fingers occlude the view.
   - Recommendation: compare on several robots and labs before changing anything.
2. **"csw4 on DROID" base keys.** The DROID base config keeps its own values: decoder 256/8, Gaussian parents
   1536/512, 16 slots × 384, consistency 0.1, depth_align scale, far 4.0, and 144×256 images. The Meta-World csw4
   base differs in all of these. Which base should apply?

## 5. Known issues (recorded, not fixed)

- `pause_rl_at_checkpoint` omits `intentional_stop: true`.
- `remote_campaign_driver --dry-run` writes state. A warning is in HANDOFF and the restart briefing.
- Wrist ZED Mini rectification has an unexplained 2.2–2.9 px horizontal offset to the SDK. The wrist is not used by
  the fixed-rig pipeline.
- SAM 3 found no detection for the prompt "gripper", nor for the brick in ext2. Both are recorded as absent.
