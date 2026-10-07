# splatter4d Meta-World campaign — progress

Last updated: 2026-10-07 (campaign start)

## Current phase
**Phase A — unblock the runtime.** Nothing else may run until Phase A gates pass.

## Facts discovered at campaign start
- `/home/ws/ws/hierarchical_splatter` and `/home/ws/ws/droid_training` no longer exist on disk.
  The reference code is read from git commit `c0abf56` (`origin/Dynamic3D`, identical to the recorded
  reference HEAD) extracted read-only into `.cache/reference/Dynamic3D` (gitignored).
  SinCro and ReViWo were git submodules; their sources must be fetched separately.
- The old venv only contained ruff plus `reference_runtime.pth`, which pointed at the deleted reference venv.
- Hardware: 384 CPU cores, 503 GB RAM, 5.1 TB free on `/home/ws`, CUDA 12.9 nvcc, GPUs 4/5 idle.
- `docs/SPEC.md` is missing; the original spec text is available to the agent from the previous session transcript.
- `CLAUDE.md` with the standing rules is missing from the repo; auto mode blocked the agent from creating it.

## Blocker (needs the user)
The Phase A environment build (`uv sync --extra dev`, which compiles gsplat and fused-ssim from source for sm_120)
was denied by the Claude Code auto-mode classifier. Every later phase needs this environment: the venv currently
has no torch, MuJoCo or Meta-World. See the final message of the session for the exact command.

## Work done without the runtime (untested until the environment exists)
- DrQ-v2 port: `s4d/rl/{replay,encoders,agent,env,evaluate}.py`, `scripts/train_rl.py`, `configs/rl/`.
- Scheduler: `scripts/jobs.py` (queue `experiments/queue.yaml`, registry `experiments/registry.jsonl`).
- Tests: `tests/test_rl.py`, `tests/test_jobs.py` (+ subprocess workers).
- Protocol: `docs/RL_PROTOCOL.md`. Baseline sources fetched to `.cache/reference/{sincro,ReViWo}` at the pinned commits.
- Encoder export now records the pretraining frame strides.

## Job table
| id | GPU | status | log | W&B |
|---|---|---|---|---|
| (none yet) | | | | |

## Next actions
1. Standalone uv environment with gsplat and fused-ssim built from source for sm_120 (blocked: needs permission).
2. Full test suite with zero errors; GPU isolation check on both UUIDs; 50-step rendered pilot run on GPU 4 then GPU 5.
3. Phase B data collection.
