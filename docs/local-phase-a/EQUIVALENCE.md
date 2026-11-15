# Local phase-A equivalence

Qualified: **False**. Coverage: cpu-only.

Depth limits: each frame mean abs <= 0.002 m, p99 abs <= 0.01 m, p99 rel <= 0.01; valid-mask IoU >= 0.995.
Raw SAM frame/prompt/instance IDs and counts must match. Raw instance-mask IoU uses the same limits; score abs <= 0.005, box abs <= 1 px.
Semantic foreground class/frame IoU: mean >= 0.99 and minimum >= 0.95; class presence exact.
Absent-in-both classes are counted separately. Ingest arrays and sensor depth must be bitwise equal.
HDF5 container bytes and completion IDs/times are not compared. Ingest contract mapping changes only episode_id, raw path and provenance.created_utc; raw SHA/size and every other field remain exact.

| Episode | Stage | Result | Numbers |
|---|---|---|---|
| covering-success-episode_000000 | foundation_stereo_depth | BLOCKED/FAIL | NOT RUN: designated MIG occupied by healthy FQL trainer PID 993329 (7388 MiB); short GPU gate deferred under the idle-ish condition. |
| covering-success-episode_000000 | ingest | PASS | bitwise payloads equal |
| covering-success-episode_000000 | sam3_masks | BLOCKED/FAIL | NOT RUN: designated MIG occupied by healthy FQL trainer PID 993329 (7388 MiB); short GPU gate deferred under the idle-ish condition. |
| covering-success-episode_000000 | sensor_depth | PASS | bitwise equal; max abs=0 m; max rel=0; valid IoU=1 |
| pick-and-place-success-episode_000002 | foundation_stereo_depth | BLOCKED/FAIL | NOT RUN: designated MIG occupied by healthy FQL trainer PID 993329 (7388 MiB); short GPU gate deferred under the idle-ish condition. |
| pick-and-place-success-episode_000002 | ingest | PASS | bitwise payloads equal |
| pick-and-place-success-episode_000002 | sam3_masks | BLOCKED/FAIL | NOT RUN: designated MIG occupied by healthy FQL trainer PID 993329 (7388 MiB); short GPU gate deferred under the idle-ish condition. |
| pick-and-place-success-episode_000002 | sensor_depth | PASS | bitwise equal; max abs=0 m; max rel=0; valid IoU=1 |

FS absolute/relative error and SAM3 mask IoU: **unmeasured**. Bulk registration is disabled.
