# Drawer-open-v2 98-success amendment (2026-11-14)

Authority: user continuation directive and binding NAMING.md raw-schema amendment.
Exclude success episode_000077.h5 (no action/robot/gripper schema) and
success episode_000079.h5 (nonmonotonic device clocks) from BOTH S and SF.
No recording is deleted or modified. S=98 success; SF=98 success + 50 failure.
The exact 100-success recording set, both rejected hashes and validator reasons
are bound by the companion rw-gate-amendment/1 JSON. Included success hashes must
match across S/SF; split via the canonical hashed per-task seed/ceil protocol.

79: four clocks jump together near 2^32 and adding 2^32 yields 66720 us.
This is strong wrap evidence, retained in drawer-raw-rejections.json. The raw
float64 fields declare no counter modulus. A candidate unwrap alone is not an
accepted ingest/normalization contract preserving raw identity through every
gate, so this release uses the explicitly authorized exclusion and does not
introduce a task-specific timestamp repair. This matches covering's established
wrap exclusions and avoids changing the paired protocol to salvage one episode.

This amendment authorizes membership only. Ordinary per-family QA, accepted
per-task/session rigs and build-bound calibration validation remain mandatory.
