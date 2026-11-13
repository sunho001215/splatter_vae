# Paired S / SF real-world reset

The immutable manifest is the authority for `(task, label, filename)`, raw sha256,
label and episode split. Directory labels override the recorder task string; no
outcome inference from motion or an absent success attribute is permitted.
Success and failure are shuffled independently with the same seed. S is the exact
success subset of SF in both partitions. Legacy success splits can be supplied as
an exact filename partition; otherwise new S rows are required for a fair comparison.

`realworld_dsreset.py` performs raw timing QA and generates standard pipeline
configs and a prep plan. It never runs the plan. Exact counts and `.VERIFIED`
markers are required. It ignores partial files and never recurses into discarded/.
The ordinary pipeline still performs full ingest/image QA, B1 FoundationStereo,
SAM3, LoMa-G calibration application, per-camera TrackEverything, track-motion,
and compact train_data generation. Raw timing QA does not replace those gates.

Drawer-open-v2 is a new task rig session and requires new LoMa-G calibration and
scene measurements. Failure episodes use their task/session's validated rig (PNP
can select accepted period-specific documents using --rig-map). Template SAM
prompts must describe the task. Missing calibration fails at the ordinary geometry
gate; preprocessing phase A can run ingest/sensor/FS/SAM before that gate.

Reuse is explicit: success `train_data` paths must bind to the same raw sha256.
Downstream input creation must validate their existing QA/calibration records.
Failure outputs live in separate label directories. No copied success tensors or
failure/success numeric-counter collisions. Native RealSense depth remains the
3D-MVP source; its native-depth bounds must be rechecked per task/variant.

Per method the S and SF science config must be identical, including seeds, budget,
architecture and FQL settings; only store/split/outcomes/manifest inputs differ.
Use Original Ours (200k baseline), SinCro, ReViWo, DynaMo, SNeRL, 3D-MVP single and
multiview, E2E CNN and noaug. No requested method is retired; S2M2 and obsolete RL
algorithms are excluded. Frozen FQL uses alpha 10 + normalize_q_loss=true; E2E
retains its established configuration, with p_aug=0 for the noaug method.

FQL already accepts explicit boolean per-episode outcomes. Keep its established
reward `success(next)-1` (OGBench single-task convention): -1 ordinary step, 0 on
reaching success. This time cost is identical for successes and failures, not a
failure label on every action. Success exists only at a reachable final recorded
observation of a success episode. Failures end as truncations with mask 1 and
bootstrap from their real final observation; they never receive absorbing success.
Do not synthesize transitions across episodes, pauses, dropped frames or absent
states. Keep BC/flow on every existing valid transition: demonstrations of recovery
and unsuccessful actions are still behavior data, and the Q term distinguishes
outcomes. The legacy `terminal` array records chain boundaries, not MDP termination;
keep it unchanged for S regression compatibility. The training mask is authoritative.
