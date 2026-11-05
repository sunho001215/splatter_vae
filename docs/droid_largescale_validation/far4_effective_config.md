# DROID far4 profiles: `render.far` 4.0 (2026-11-05, before any training launch)

User directive (2026-11-05): use `render.far: 4.0` for the applicable real-world and DROID pretraining
configurations. Version it explicitly. Keep depth validity, Chamfer targets, rendering and synthetic views consistent
with it. Do not apply it to Meta-World. Do not change running or completed experiments or relabel outputs. DROID
pretraining uses DROID RGB plus the depth and tracks supplied by PointWorld; there is no new depth estimation or tracking.

## Versions

| Profile | File (use alone) | `render.far` | Status |
|---|---|---|---|
| **DROID csw4, far4** | `configs/droid/csw4_largescale_far4.yaml` | 4.0 | new; resolved: `csw4_largescale_far4_resolved.yaml` |
| **DROID ViT-S-sized, far4** | `configs/droid/vits_largescale_far4.yaml` | 4.0 | new; resolved: `vits_largescale_far4_resolved.yaml` |
| DROID csw4 / ViT-S-sized | `csw4_largescale.yaml` / `vits_largescale.yaml` | 3.0 | unchanged (any run that resolved them keeps its meaning) |
| LEGACY DROID-base smoke | `pretrain.yaml` + `largescale.yaml` | 4.0 (already) | unchanged; no `loss.depth_valid_far`: far there bounds rendering and the visibility regulariser only |
| Meta-World | `configs/metaworld/base.yaml` | 3.0 | untouched |

- `tests/test_droid_far4_profiles.py` checks the following:
  - each far4 file resolves to its far-3.0 profile with exactly one difference, `render.far` 3.0 → 4.0;
  - ViT-S far4 differs from csw4 far4 exactly as ViT-S differs from csw4;
  - no Meta-World file sets another far plane, and every resolved Meta-World csw4 task composition keeps 3.0.
- Run names: `droid_largescale-csw4far4-<tag>-<seed>`, `droid_largescale-vitsfar4-<tag>-<seed>`. The run's
  `config.yaml` and `provenance.json` record the resolved value, and the run log records the config path.
- Scope checks:
  - `experiments/registry.jsonl` and `queue.yaml` hold no job with a DROID training config (0 of 13,100 records).
    Nothing running or completed resolves these files.
  - **No real-world pretraining configuration exists.** `configs/realworld/*.yaml` are preprocessing or validation
    pipelines, and `scripts/train.py` has the regimes `metaworld`, `droid` and `droid_largescale` only.
    `s4d.realworld.dataset.RealworldCacheDataset` is used only by contract checks and diagnostics. No real-world
    regime was invented. A future one needs its own far4 profile.

## What reads `render.far` (DROID large-scale, `loop.forward_losses`)

No other key carries the depth range. Every consumer below takes `near, far = cfg["render"]`. The profiles set
`loss.depth_valid_far: true`. The test runs one `forward_losses` step of each profile (far 3.0 and far 4.0) on fixture
windows. It checks the renderer planes, the depth mask, the Chamfer targets, the track weights, the synthetic cameras and
views, and the usage diagnostics. Depth bands at 3.5 m and 4.5 m and tracks at camera z 3.5 m and 4.5 m are planted in
the windows.

| Consumer | Code | Rule | 3.0 → 4.0 |
|---|---|---|---|
| Depth L1 and coverage masks; `depth_absrel`, `coverage_valid_mean` | `loop.py` `valid` | 0 < d < far (∧ observed) | depth in (3, 4) m becomes a target |
| Chamfer target clouds | `losses/chamfer.py::target_clouds` | 0 < d < far | points from (3, 4) m enter; `chamfer_excluded_beyond_far` drops |
| Track motion weights | `loop.py` (track branch) | source camera z < far | inert: no PointWorld track lies beyond 1.53 m (below) |
| Synthetic cameras (`training_cameras` rig) | `augment.jitter_training_cameras` | look-at = median first-frame depth in (near, far) | look-at moves outward only in deep scenes |
| Synthetic views (RGB/depth/coverage targets, invariance inputs) | `diag/heldout.fused_training_clouds` → `splat_batched` | 0 < d ≤ far | (3, 4] m points are splatted |
| Renderer (training, synthetic-view, feature, orbit, evaluation renders) | `model/render.py` → gsplat `far_plane` | Gaussian centres with camera z beyond far are culled | Gaussians at 3–4 m render |
| Usage diagnostics | `losses/occlusion.usage_diagnostics` | 0 < D < far | metrics only |
| Evaluation panels | `train/evaluate.py` | depth colour range [near, far]; error scale 0.1 (far − near) | visual only |

Inactive in both profiles (weight 0 or absent), but they read the same value: `depth_hard`, `visibility`, `occlusion`,
`loss.self_render`, and the per-frame pilots (`train/perframe.py`).

**Kept separate.** These are source or sensor validity, or are unrelated to the render range. None changed.

- Released depth: 0 and 65535 mm are invalid (`depth_mm_to_m`).
- PointWorld track validity: `scene_visibility & scene_depth_valid_mask`; the mask is 0 < d ≤ 4.0 m upstream. The
  dataset also applies a 2 cm `depth_test`.
- Crop motion thresholds (`data.crop.motion_min_m` / `motion_max_m`): track motion, not depth range.
- Workspace anchor statistics (`scripts/compute_workspace_stats.py`) use all valid depth with no far cut. The same
  statistics serve both versions.
- Chamfer diagnostic bands 0.9 / 1.3 m: metrics only.
- Real-world preprocessing `sensor_depth` / `foundation_stereo_depth` `min_depth_m` / `max_depth_m` (0.1 / 6.0 m):
  source validity of another pipeline, not a training render range.

**Boundary.** The released depth is integer millimetres.

| Rule | Treatment of exactly 4000 mm |
|---|---|
| Depth / Chamfer masks (strict `<`) | excluded |
| Synthetic views (`<=`, as with 3.0) | included |
| PointWorld track validity (≤ 4000 mm) | included |

This affects 18,815 training-window pixels: 6 × 10⁻⁶ of valid ext depth. The shared code was not changed, because that
would also change Meta-World.

## Measurement on the released inputs

Source: `scripts/droid_ls_depth_range.py` (read-only), run on all 294 AUTOLab episodes of
`item8/r2/manifests/manifest_droid3d_autolab294_v2.jsonl` and their extracted PointWorld tree. This is every DROID depth
episode available locally. The remote host holds no DROID data. All stored frames were measured, not a subsample.
Output: `far4_depth_range_autolab294.json`.

| External cameras | all 3.91 G stored pixels | training windows (stride 3, weighted by windows per frame) |
|---|---|---|
| valid fraction of pixels | 91.3 % | 91.4 % |
| valid depth ≤ 3 m | 99.11 % | 98.89 % |
| valid depth in (3, 4] m | **0.53 %** | **0.68 %** |
| valid depth > 4 m | 0.36 % | 0.43 % |
| percentiles 95 / 99 / 99.5 / 99.9 | 1.35 / 2.85 / 3.73 / 4.73 m | 1.41 / 3.13 / 3.90 / 4.69 m |
| largest valid value | 65.52 m | 65.52 m |
| saturation code 65535 | 6,843 px (58 of 588 camera-episodes) | 5,541 px |

The far-3.0 profiles' earlier subsample figures (1.04 % beyond 3 m, 0.43 % beyond 4 m) agree.

- **Concentration.** Depth in (3, 4] m is concentrated in a few camera views.
  - Median camera-episode: 0.012 %. 90th percentile: 0.34 %.
  - 18 of 588 camera-episodes (3.1 %) exceed 5 %, at most 21 %.
  - By recording date: 2023-11-03 has 16.7 % in (3, 4] and 10.4 % beyond 4 m. 2023-10-21, 10-27 and 12-02 have
    0.03–0.17 % in (3, 4].
- **Wrist depth** (unused by the losses): 0.012 % in (3, 4] m, 0.009 % beyond 4 m.
- **Temporal stability** (static cameras; consecutive canonical frames 0.13 s apart; median / 90th percentile of
  |Δd| / d):

  | Depth bin | Median | 90th percentile |
  |---|---|---|
  | 0.5–1 m | 0.33 % | 2.8 % |
  | 3–3.5 m | 0.33 % | 8.4 % |
  | 3.5–4 m | 0.48 % | 5.8 % |
  | 4–5 m | 0.4 % | 3.5–3.9 % |
  | beyond 5 m | 10–33 % | — |

  Depth up to 4–5 m is temporally stable. Beyond 5 m it is not.
- **Cross-camera consistency.** Each camera's depth is lifted with the training K and calibration and compared with the
  other external camera's depth (median |z/d − 1|; selection: z ≤ d + 2 cm):

  | Source depth | DROID-3D (training) median | DROID-3D 90th pct | PointWorld calibration median |
  |---|---|---|---|
  | 0.5–1 m | 0.63 % | 3.3 % | 0.98 % |
  | 1–1.5 m | 1.5 % | 6.4 % | 2.0 % |
  | 2.5–3 m | 2.9 % | 5.7 % | 4.6 % |
  | 3–3.5 m | 2.7 % | 5.5 % | 4.9 % |
  | 3.5–4 m | 3.1 % | 5.3 % | 4.7 % |
  | 4–4.5 m | 3.2 % | 5.2 % | 2.8 % |

  At 3–4 m the two cameras' surfaces disagree by about 3 % median, roughly 0.1 m. That is about 5 times the 0.5–1 m
  value. The fused two-camera Chamfer targets in that range therefore carry about 10 cm of inter-camera spread (voxel
  1 cm). There are 1.05 M selected overlap points at 3–4 m, against 117 M at 0.5–1 m.
- **Tracks.** Visible, depth-valid PointWorld track points (1.08 G) have camera z ≤ 1.53 m (99.9th percentile 1.40 m).
  None lie beyond 2 m. The largest clip `initial_depth` value is exactly 4.000 m.

## Upstream limitations

1. **No cap below 4 m in the released depth.** The stored depth goes to 65.52 m, plus the 65535 code. The 4 m
   cutoff is fully represented in the inputs.
2. **PointWorld's own pipeline caps depth at 4.0 m.** `convert_2d_flows_to_3d.py` sets `MAX_DEPTH_M = 4.0` (NVlabs
   PointWorld 3872ec6). This defines `scene_depth_valid_mask` = 0 < d ≤ 4000 mm and sanitises `initial_depth` to
   ≤ 4.0 m; the measured maximum is 4.000. far 4.0 coincides with the source's own validity cap.
   - The released depth generator (`compute_depth.py`, FoundationStereo on ZED 2 stereo) stores fx·b/disparity in
     uint16 mm with no range clip.
   - Its documented resolution is 640×360. At that resolution 1 px of disparity is about 0.29 m at 3 m and 0.51 m at
     4 m.
   - How the released 320×180 package was downsampled is not in the released code.
3. **No motion supervision at range.** PointWorld seeds tracks only inside its workspace mask, so no track exists
   beyond 1.53 m. The far change adds static geometry (depth, coverage, Chamfer, synthetic views) and no motion targets.
4. **Accuracy at 3–4 m is lower than near the robot.** It is about 3 % cross-camera (about 0.1 m), against 0.6 % at
   0.5–1 m. PointWorld's calibration is worse in this range than DROID-3D's.
5. **Coverage.** These numbers come from one lab: AUTOLab, 294 episodes on four dates, from the first sequential depth
   part. The (3, 4] share is dominated by one date. Corpus statistics remain to be measured; the depth package is 252
   sequential parts. Re-run the script on any larger extraction before relying on the share.

## Before a far4 launch

- Give the four data sources, `model.anchor_stats` (unchanged by far) and `train.steps`.
- Pass the release gate on the merged sources, then run the trainer smoke with the far4 file, as `item8/r2/smoke_csw4.sh`.
- Use far4 run names. Never resume a far-3.0 run with a far4 config. The resolved config, and therefore
  `config_sha256`, differs.
