# Real-world preprocessing status (2026-11-06)

Robot: Franka Panda with the DROID-style Robotiq 2F-85 installation (ISO-50 coupling kept, nominal; no wrist camera).
Standard pipeline (`configs/realworld/mug_hanging.yaml`, [realworld_preprocessing.md](realworld_preprocessing.md)):
RGB for appearance and tracking; FoundationStereo on each RealSense IR pair registered to the colour grid (B1, selected
by the 2026-11-05 five-way comparison, section 9) for lifting/export; native depth + SAM 3 robot masks as calibration
evidence only; robot-grounded calibration; single-view TrackEverything per camera (B1 depth, K, accepted pose); the
noise-aware `track_motion` filter; export of moving tracks. Alternatives (MapAnything, AllTracker + lifting, filtered
B1, comparison runners, the raw-DROID route) are retired (section 10). DROID pretraining does not use this package.
Hosts: bulk preprocessing on the remote host; the RTX 5090 inference server for single-episode inference and
visualization. TrackEverything details: [realworld_trackeverything.md](realworld_trackeverything.md); historical DROID
evidence: [realworld_droid_validation.md](realworld_droid_validation.md).

| Component | Validated | Awaiting |
|---|---|---|
| RGB-D calibration (Panda, SAM 3 masks, 13 starts) | real episodes (mug-hanging 0/1, drawer-open, pick-and-place, covering), all cameras accepted; bit-identical speed-up (bounded KD-tree threads, parallel starts) | measured coupling; translation spread ~1 cm at equal held-out residual |
| IR stereo (`foundation_stereo_depth`, B1) | real episodes; conventions reviewed; LR-tap and threshold-precision regressions | independent depth reference |
| TrackEverything single-view on B1 | real episodes; matched to AllTracker windows/queries on mug-hanging episode 1 (historical comparison) | - |
| `track_motion` + export (moving tracks) | real episodes; contract checks (loader, `validate_batch`, model forward) | training on the export |
| Viewer (viser recordings, browser checks, videos) | real episodes | - |
| Depth variants B0, B1, E1, E2, E3 | historical (episode 1, frozen calibration; section 9) | - |

Tests do not invoke upstream models or use the GPU; `mock_worker.py` always stamps `is_mock`. The package isolation
test remains enforced. Shared stages of the historical comparison roots are symlinks to their base root and are never
rebuilt there (`io.run_stage` refuses symlinked stage directories).
