# Ball-hiding BH3: 99 eligible successes

Authority: coordinator's 2026-11-21 Codex BH3 directive. Exclude
success/episode_000008.h5 for every method. Preserve the original file and
strict Panda adapter rejection; no timestamp/depth repair or calibration waiver.

Read-only local evidence: SHA256
364c3eff063cd9ab495a2455a9bd7392dba8a8a3583de199e74b29858833c322,
cam_135122073201, stereo/depth_ok = 0 at zero-based frame 121 of 150.
All t_valid and stereo/t_valid entries in all three cameras are nonzero;
the other cameras' depth_ok entries are nonzero. The adapter's read_meta
rejects any zero in those arrays before normalization.

The companion rw-gate-amendment/1 record binds all 100 original hashes,
99 eligible and one excluded. Arrival still requires the complete verified
100-file inventory. Every method uses configs/realworld/tasks/ball-hiding-split99.json.
Its 89/10 membership is recomputed on the sorted eligible list by the existing
dsreset.paired_manifests protocol: numpy default_rng seeded by
[0, int(sha256(b'ball-hiding').hexdigest()[:8], 16)], first ceil(0.1 * 99)
permutation entries for validation. store.split_episodes uses the same rule.
Drawer-open-v2's 98-success amendment documents this hashed task-seed/ceil
protocol; covering's amendment explicitly names store.split_episodes.

Old normalize, Phase A, RGB and calibration bundles remain immutable.
New RGB/calibration bundles require only the 99 eligible upstream jobs.
Phase B and geometry retain all ordinary automatic calibration and QA checks.
The geometry snapshot retains a raw-invalid evidence row for the exclusion;
its stores, rig, measurement and learning inputs contain exactly 99 episodes.
