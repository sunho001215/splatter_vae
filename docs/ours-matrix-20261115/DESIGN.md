# New real-world scratch matrix (2026-11-15)

Twelve new runs: drawer-open-v2 / covering / pick-and-place × S / SF ×
Original / Original+Chamfer0.1. Each has one scratch 400k pretrain, final-400k
encoder export, frozen fqldata, FQL seeds 0–2, and task-aware length evaluation
once all three FQL runs finish. That is **84 new matrix rows**. A missing
pick-and-place continuation evaluation adds one row; existing continuation rows
are audited and retained byte-for-byte. Covering already has a task-aware evaluator.

Each committed snapshot derives from the task's established Original recipe.
The sole schedule change is `train.steps=400000`; the original cosine code uses
that total horizon, warmup 10000, peak 5e-4 and minimum 1e-5. There is no restart,
original_steps, alternate cycle, imported checkpoint or 200k downstream artifact.
`--resume auto` only retries the same fresh run's private directory; registration
refuses any pre-existing proposed run path or queue/registry ID. The only arm
change is `loss.chamfer=0.1`; Original keeps its implicit zero. The existing
`s4d/losses/chamfer.py::chamfer_loss` is bidirectional and the unchanged training
loop uses the existing cloud voxel/point defaults (0.01 m, 4096 points).
RGB, SSIM, hard depth, depth gradient, motion, invariance, consistency, coverage,
visibility, 50% motion-aware tube masking, augmentation and architecture stay intact.
The registrar validates the exact resolved differences on every plan, including
saved/queued Original covering/PNP science and saved seed-specific FQL science.

FQL inherits the actual rwnq Ours argv: alpha 10, normalize_q_loss true, batch 256,
1M offline steps, original intervals, seeds 0–2. Only experiment names and the
correct task/variant dataset, split and gate paths change. Resource declarations
and the 8 CPU / 1g.24gb footprints derive from the established rows. Priorities
are prep -4 (existing dsreset producers), matrix pre -3, all downstream -1.
No running work is preempted.

Four covering/PNP S rows reuse the exact locally mirrored rwv2 S learning inputs
and existing statistics. The registrar verifies their real local closures and
completed dependencies, without SSH or release placement. This establishes input
readiness; actual capacity is rechecked by the official placement transaction.
Drawer S and all SF chains wait on canonical dsreset QA/publication and per-variant
train-only stats. These IDs may be absent until the separate dsreset publication
registration; the scheduler's missing dependencies fail closed and become ready
on normal producer completion. The dsreset publication worker must set `ours_input_barriers_only=true` in
its registration spec to suppress new legacy Ours learning chains, keep these
standard IDs, publish the accepted five-file bundles, mirror them and close out
local producers through the existing route. No hashes or completion evidence are
invented for unpublished data. The label-aware dsreset train/export/dataset/FQL
wrappers are used only for those new chains. A new isolated evaluation wrapper
runs the unchanged `scripts/evaluate_ours_task_length.py` after installing the same
label-aware runtime. Existing placed evaluator code remains unchanged.

Registration defaults to stdout-only. Execute is restricted to adopted main,
matching passing full-suite fingerprints and release image, fresh queue/registry
and input hashes, empty run paths, no prior journal/proof, and one active exact-ID
placement hold. Publication reuses the reviewed append-only transaction. The
new placement helper and finite loop reuse normal locks, scope verification,
mirror/dependency closeout, priority/capacity accounting and per-ID hold lifting.
They do not launch training. Existing controllers and their files are unchanged.

The per-seed `deployment_hook` contract names task, dataset, arm, protocol,
checkpoint, seed, exact policy path, linked evaluation ID and owner
`robot-pc-pipeline`. The registrar proof also lists all six continuation hooks
without editing their existing rows. The other worker should consume the proof,
verify normal FQL/policy and evaluation closeouts, hash policy bytes and deduplicate
by full identity and SHA before export/transfer/registration. This candidate
supplies hooks; it does not install or run the separate Robot PC worker.

See RUNBOOK.md for exact coordinator commands and REMOTE-RECOVERY.md for VPN recovery.
