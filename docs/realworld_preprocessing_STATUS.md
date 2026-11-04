# Real-world preprocessing status (2026-11-05)

FoundationStereo removed on 2026-11-05, replaced by MapAnything.

Maintained branch: `realworld-fsrm-20261105`. Design, stage layout, worker pins and commands are in
[realworld_preprocessing.md](realworld_preprocessing.md). Historical DROID evidence is linked from
[realworld_droid_validation.md](realworld_droid_validation.md).

| Component | Validated | Awaiting |
|---|---|---|
| Stereo shared grid | Record-only inputs; Bouguet rectification of both eyes; K and metric `T_rr_rl`; missing-grid refusal | New real DROID run |
| `mapanything_rig_depth` | CPU mock, one posed pair per rig/time before world calibration; rotated-camera depth accuracy | Actual MapAnything stereo quality |
| `mapanything_depth` stereo views | Shared images/K; accepted left pose and composed right pose; optional right outputs | Actual multirig model run |
| RGB-D | Existing recorder, sensor registration, calibration evidence and lifting-depth selection retained | New real-data validation for this change |
| Calibration/tracking/lifting/export | CPU synthetic and contract coverage, mock refusal, infrared opt-in | New stereo model residuals |
| Local/remote stage dispatch | Shared runner-based implementation; local file protocol | Remote execution of new stereo route |
| Own robot mount | Explicit provisional measurements and provenance retained | Measured coupling and gripper mapping |

`mock_worker.py` always stamps `is_mock`. Its `infer_views` echoes input depth; without depth it only accepts
exactly two rectified views and uses the CPU matcher as a test oracle. Production has no matching fallback.
Tests do not invoke upstream models or use the GPU. The package isolation test remains enforced.

Review decisions: rig depth saves both streams; world depth includes right eyes by default but only primary views
are consumed; stereo raw/rectified images retain channel order until model input conversion; nonmetric rigs are
skipped; missing metric-view calibration is refused. Mixed stereo/RGB-D automatic depth selection and differing
view sizes remain unsupported. Completed legacy depth caches are not reused under the new stage names.
