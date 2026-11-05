# Real-world preprocessing status (2026-11-05, evening)

Robot: Franka Panda with the DROID-style Robotiq 2F-85 installation (ISO-50 coupling kept, nominal; no wrist camera).
Depth for lifting/export on the RealSense rigs: FoundationStereo on the device infrared pairs (B1), selected by the
five-way comparison on episode 1 (section 9 of [realworld_preprocessing.md](realworld_preprocessing.md)); native depth
remains the calibration evidence; MapAnything remains available (`mapanything_depth`, also the stereo/DROID route).
Tracks: AllTracker + lifting (default) or single-view TrackEverything (`track_everything`, comparison/optional), both
followed by the noise-aware `track_motion` filter. Historical DROID evidence:
[realworld_droid_validation.md](realworld_droid_validation.md); TrackEverything details:
[realworld_trackeverything.md](realworld_trackeverything.md).

| Component | Validated | Awaiting |
|---|---|---|
| RGB-D calibration (Panda, SAM 3 masks, 13 starts) | real episodes 0 and 1, all cameras accepted; bit-identical speed-up (bounded KD-tree threads, parallel starts) | measured coupling; translation spread ~1 cm at equal held-out residual |
| IR stereo (`foundation_stereo_depth`, `foundation_stereo_filtered`) | real episode 1; conventions reviewed; LR-tap and threshold-precision regressions | other rigs/episodes |
| Depth variants B0, B1, E1, E2, E3 | real episode 1 (frozen calibration, consistency report) | independent depth reference |
| `track_motion` + export (moving tracks) | real episodes 0 and 1 | training on the export |
| TrackEverything single-view | real episode 1 on B1, matched to AllTracker windows/queries | more episodes |
| Viewer (viser recordings, browser checks, videos) | episodes 0 and 1 | - |
| Stereo/DROID MapAnything route | earlier real DROID validation | new large-scale DROID run |

Tests do not invoke upstream models or use the GPU; `mock_worker.py` always stamps `is_mock`. The package isolation
test remains enforced. Shared stages of depth-variant roots are symlinks to the base root and are never rebuilt there
(`io.run_stage` refuses); comparisons refuse stale calibration provenance.
