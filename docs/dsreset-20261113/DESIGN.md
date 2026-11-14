# Paired real-world S / SF pipeline

The immutable episode manifest identifies `(task, label, filename)`, sha256 and
split. Labels come from the directory, never from motion or an absent outcome
attribute. Each label has its own seeded permutation. Success uses the established hashed
per-task seed and ceil validation count; failure has a separate label stream. S is the exact success
subset of SF in both partitions; therefore S validation never enters SF training.
A supplied legacy success split must be an exact disjoint filename partition.

Raw QA checks complete recorder status, action/state correspondence, strictly
monotonic robot/gripper/camera and device clocks, three camera streams, intrinsics
and synchronized valid action transitions. Partial files and discarded/ are never
opened. Final manifests require complete verified pulls and exact 100/50 raw counts.
Existing success roots may use their exact verified ingest SHA set instead of a
new pull marker; each local file is rehashed. Excluded old recordings use the
accepted amendment hash and must remain invalid in current ingest evidence.
Existing documented covering-98 and pick-and-place-97 gate amendments can be
supplied explicitly: all included/excluded success hashes must match, exclusions
remain recorded with `split=excluded`, and the same accepted success set is used
in S and SF. No automatic exclusion or timestamp repair is permitted.

`--prep-only-available` instead emits a provisional inventory and ordinary pipeline
configs for available complete episodes. It reports raw-QA rejections and emits no
training manifests, outcomes or splits. This lets phase A proceed during transfers.
The drawer-v2 smoke found success episode_000077.h5 has cameras/events only and
episode_000079.h5 has a nonmonotonic device clock. The binding 2026-11-14 directive
excludes both with `drawer-open-v2-98-20261114.json`, hash-bound to all100 successes,
the rejection evidence and amendment document. Raw data stays unchanged. Apparent
32-bit clock wraps in79 lack a recorder modulus/accepted repair contract. Final
drawer S98/SF148 manifests exist; no rerecording or human membership review waits.

The new host-local MIG dispatcher consumes the generated per-episode configs.
Its reusable implementation is `ops/jobs/rwprep_mig.py`, with a thin command in
`ops/campaign/rwprep_mig.py`. The old remote whole-GPU dispatcher remains stopped.
The standard queue rejects local upstream prep and lacks its worker environment;
a dedicated admitted queue is the narrow compatible route. Phase A is
`ingest,sensor_depth,foundation_stereo_depth,sam3_masks`; phase B applies the accepted
rig and runs TrackEverything, track_motion and train_data, followed by the existing
per-family QA and calibration-validation record procedures. Raw QA is preliminary;
it cannot release the ordinary full-data/geometry gates. The registration tool
requires completed learning inputs; its shared prep/calib/qa rows recheck evidence
at consumption time. They do not replace the bulk dispatcher or fit a rig.

Drawer-open-v2 requires its own LoMa-G camera-to-base rig session calibration.
Recorded device inter-stream extrinsics do not supply that calibration. The legacy
ending-check registry now marks drawer-v2 as needing image review until its own
thresholds are measured; ending flags never change the manifest label. Failures
reuse their own task/session's accepted rig after serial/fixedness/coverage checks.
PNP and covering can have period-specific rigs; use `--rig-map`, keyed by the new
label-inclusive episode IDs. Do not infer a failure's period from its numeric
counter alone. The new drawer prep template has task-appropriate SAM prompts and
an intentionally nonexistent rig placeholder. Phase A can run before acceptance.

Reuse of success train_data is explicit and bound to raw sha256. `--reuse-ingest`
discovers existing completed train_data records; final learning input creation
still requires their current validation, per-family QA and accepted calibration.
Failure outputs use separate label directories. Label-aware identity joins allow
overlapping success/failure counters and continue to reject duplicates within a
label or duplicate content. Existing success tensor/identity metadata is unchanged.

`scripts/realworld_dsreset_inputs.py` builds each variant from exact live ingest
hashes, projects ordinary `gate.evaluate` onto that immutable membership and binds
actual compact store identities. It writes a new directory, never the live ingest
manifest. The registrar rechecks live gate sections, train_data paths, paired
membership, splits and explicit boolean outcomes before publishing anything.

The retained method set is ours, sincro, reviwo, dynamo, snerl, mvp3d_single,
mvp3d_multiview, e2e_cnn, e2e_cnn_noaug. None is retired for these tasks; memory
notes retire S2M2 and obsolete RL algorithms instead. The two MVP encoders share
pretraining; E2E has no representation-pretraining/export stage. Every method uses
the same science config, seeds, budgets and evaluation for S/SF, with only
manifest-derived data/statistics and run paths changed. Existing registered Ours S chains are reused only after exact raw hash, split,
compact-path, accepted calibration-build/camera, Original recipe and FQL-science
proofs. Real covering-S and PNP-S proofs pass. All other methods receive paired
rows. A mismatch refuses reuse; no existing queue row is rewritten.

Ours uses the Original architecture and its original 200k recipe, followed by one
full-state 200k→400k continuation: warmup 2500 steps from 1e-5 to 2.5e-4, cosine to
1e-5 at 400k. The continuation preserves model, AdamW, scheduler and RNG state using
the existing validated continuation implementation. Only the final 400k checkpoint
feeds export/fqldata/FQL. Six pinned task/variant baseline configs make the input
and science hashes reviewable. Train-only workspace stats are dependency outputs.
The merged adopted Ours400k release already binds validated covering-S/PNP-S
200k sources to 400k continuations. Their complete S chains are reused after the
proof above; all other new Ours variants have explicit 200k and 400k producer rows.
The implementation shares main’s --schedule-config interface and continuation400k.yaml.

3D-MVP uses native RealSense depth overlays, never FoundationStereo depth for its
learning input. Registration requires an accepted native-bounds report for the SF
training partition, bound to its manifest digest and exact train IDs. The same
measured bounds are used in both variants. Report schema is documented in
COORDINATOR.md. Drawer-v2 SNeRL lookat/pose_scale and scene bounds remain null until
measured. Admitted native-depth SF-TRAIN jobs publish manifest-bound reports;
registration supplies identical runtime settings to both variants. No further
source/config release or human signoff is needed after an automatic measurement
pass, and no old drawer values are copied.

FQL already implements the required explicit-outcome handling. Preserve reward
`success(next)-1`: ordinary steps receive -1 and reaching success receives 0, as in
the existing success-only convention. This time cost applies identically to both
labels. Only a success episode has a reachable final success observation. Failure
ends as truncation, receives no success reward, and keeps mask 1 to bootstrap from
its real final observation. No transition crosses an episode, pause, dropped frame
or absent state. Keep BC/flow on all existing valid transitions: unsuccessful and
recovery actions remain behavior data, while Q distinguishes outcomes. The legacy
`terminal` tensor denotes a contiguous-chain boundary; the training mask denotes
MDP termination. Keeping that tensor unchanged preserves S compatibility.

Frozen-encoder FQL uses alpha=10 and normalize_q_loss=true. E2E retains the existing
CNN configuration; only noaug sets p_aug=0. Tests compare every S dataset tensor
against the current success-only builder and check SF reward/mask/boundary behavior.
A 500-update tiny CPU FQL run ranks successful actions above failed actions at two
matched states. This validates plumbing, not real-world policy performance. An
existing PNP fqldata binary was not available locally for the optional real-output
regression; the synthetic comparison exercises the current production builder.

The full graph template has 477 rows for all tasks/methods/seeds. With the two
proven registered Ours S chains, 461 new rows remain; reused producers keep their
existing IDs, releases and dependencies. Dry-run is the default;
`--preview` validates only the graph/argv and can never execute. Publication requires
the adopted main copy, exact local or remote release pin, unchanged input hashes,
an active `dsreset-await-placement-20261113` pause, locks and an atomic journal.
It appends queue rows and never registry events. Both-host admission and local input
resolvers are checked before publication. Local-release rows still need remote
acceptance of the same commit before remote placement. Nothing here launches,
places or stops campaign jobs, deploys a policy, or contacts the Robot PC.

Export rows validate/copy baseline encoder artifacts; Ours uses its standard final
checkpoint exporter. Eval rows verify the final policy budget and capture the
trainer's final held-out offline metrics with the policy hash. They do not perform
new rollouts, claim robot success, transfer/register policies, reload a server or
actuate anything. The coordinator's established safe deployment/validation workflow
remains a separate downstream action after those policies exist.
