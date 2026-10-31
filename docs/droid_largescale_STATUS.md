# DROID large-scale data path: status (2026-10-31)

Branch `droid-largescale-20261031` (worktree `.worktrees/droid-largescale`), based on `2605ba2`. Not merged or pushed.
Design, schemas and conventions are in `docs/droid_largescale.md`.

## Done (code and tests)

| Piece | Files |
|---|---|
| TF-free TFRecord/Example walker and in-place RLDS step index (resumable, `--range` parallel) | `s4d/data/droid/largescale/{tfrecord,rlds_index}.py`, `scripts/droid_ls_build_rlds_index.py` |
| PointWorld schema checks, lazy bounded HDF5 access, strict tar-stream extractor | `pointworld_source.py`, `extract.py`, `scripts/droid_ls_extract_pointworld.py` |
| DROID-3D parsing, Franka FK, wrist poses (DROID-3D model, PointWorld recipe, original DROID hand-eye), exact static-camera track rebase | `calibration.py`, `geometry.py` |
| Identity, serial and frame-validated join; per-episode wrist QA (pose model and depth orientation); provenance manifest | `join.py`, `scripts/droid_ls_build_manifest.py` |
| Map-style lazy dataset (shared contract plus supervision-only `wrist_*`/`robot_*` fields) and `data.regime: droid_largescale` | `dataset.py`, `config.py`, `configs/droid/largescale.yaml`, one new `elif` in `scripts/train.py` |

`scripts/train.py` is the only existing file that changed: one new `elif` branch. Every existing regime and config is
untouched.

## Test results (CPU, `CUDA_VISIBLE_DEVICES=MIG-2ebd9e11-…`)

| Test files | Result |
|---|---|
| `tests/test_droid_largescale.py` | 40 passed |
| `tests/test_droid.py tests/test_droid_protocols.py tests/test_contract.py` (existing behaviour) | 91 passed |

The full suite was not run, by instruction. `scripts/train.py` refuses to train until `require_passed_tests()` sees a
recorded full-suite pass.

## Verified, assumed, blocked

See the table in `docs/droid_largescale.md` §6. In short:

- **Verified from released bytes:** the RLDS schema.
- **Verified on the real sample and in the publisher's code:** the PointWorld schema and conventions.
- **Verified from the card and the authors' code only:** DROID-3D. The data files are gated and no bytes were read.
- **Unknown, decided per episode from data:** the PointWorld wrist depth (orientation, intrinsics, source). The
  wrist QA decides orientation, and a failing episode only loses its wrist fields.

## Explicit limitations

1. **DROID-3D not checked against real bytes.** No real `extrinsics.jsonl` or `camera_intrinsics.jsonl` line was
   read; access needs a valid HF token for the gated repo. Field names come from the card and DepthWorld.
2. **Hand-eye `mean_mat` frame conflict.**
   - DepthWorld composes it with flange FK `FK(q+dq)`.
   - PointWorld's generator defines it against `cartesian_position`.
   - The two differ by the controller's flange→tool offset, which DROID does not document.
   - Mitigations: the manifest records `frames.flange_to_cartesian_m`, and wrist QA rejects a misplaced camera
     (tested). The asset is not vendored; its path is passed explicitly.
3. **Gripper points under `droid3d`.** They assume `cartesian_position` = controller FK of `q` with the same DH
   (`FK(q+dq) · inv(FK(q)) · T_base_ee`). `dT_gw` (a hand-eye correction) is not applied to the gripper.
4. **External K stays PointWorld's in both modes.** Its tracks and depth were produced on that grid. That K was
   written as ZED K × 0.25 without a half-pixel shift (≤0.375 px from the continuous optical centre).
   `K_droid3d_minus_pw_max_px` is recorded per episode.
5. **Wrist track supervision is visibility only.** PointWorld releases no wrist tracks. `wrist_track_weight` marks
   which external tracks the wrist camera sees at each pair source time, by depth test. No wrist-native 2D tracks
   exist to re-lift.
6. **Wrist poses vary per frame.** The contract's training views have one pose per window, so the wrist is not a
   training view. Its aligned per-frame fields are supervision-only extras that the current trainer ignores
   (bit-identical loss, tested). The encoder input contract is unchanged.
7. **Index-less depth package.** It must be extracted once (about 1.23 TB). The flows can be extracted per shard
   using `_shards_manifest.json`.
8. **Unmeasured on real data:** throughput, corpus join rate, wrist QA pass rate, PointWorld depth HDF5 chunking.
9. **Placeholders:**
   - `model.anchor_stats` is `null` in the overlay; workspace statistics for the joined corpus are a TODO.
   - `gripper_offset_m = -0.019` was fitted on one sample episode.

## Remaining steps (in order)

1. Index the RLDS release in place.
   - Command: `scripts/droid_ls_build_rlds_index.py --rlds-root <droid/1.0.1> --out <new dir outside the release>`.
   - Run it in `--range` slices, then finalize.
   - This needs explicit authorization to *read* the protected DROID root.
   - The first shard confirms the assumed Example wire layout (float32 packed state).
2. Download the DROID-3D `extrinsics.jsonl` (100 MB) and `camera_intrinsics.jsonl` (203 MB) with a valid token, and
   the DepthWorld `gripper2wrist_transforms.json`.
   - Spot-check five records against §2.3.
   - Reconcile the hand-eye frame using `flange_to_cartesian_m` and wrist QA on a few episodes.
3. Fetch and extract PointWorld: cameras (19.5 MB), flows shards, and the depth stream.
   - Command: `zstd -dc … | scripts/droid_ls_extract_pointworld.py --dest <root>`.
   - Check disk first: about 3.9 TB total.
4. Build the manifest: `scripts/droid_ls_build_manifest.py … --calibration droid3d`.
   - Review the report's rejection reasons, the `uuid_matches_folder` rate, the wrist QA residuals and orientation
     decisions, and `K_droid3d_minus_pw_max_px`.
5. Compute corpus workspace statistics for `model.anchor_stats`. Refit or validate `gripper_offset_m` on several
   episodes.
6. Run the full test suite to satisfy the training gate.
7. Run a short single-GPU overfit on a few manifest episodes, then measure throughput with `num_workers` 8.
