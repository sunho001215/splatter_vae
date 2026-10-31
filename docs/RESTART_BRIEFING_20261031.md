# Restart briefing — local shutdown 2026-10-31 23:00 KST

Machine-readable state: `runs/setup/restart-state-20261031.json`. Both files are also in
`~/kaist/sunho/remote_runner/` on the remote. The details below are in `docs/EXPERIMENT_LOG.md` (2026-10-31 entries).

## 1. Who controls what during the outage
- **Remote:** `~/kaist/sunho/remote_runner/remote_runner.py` is the only launcher of remote jobs. It runs under the
  host's `python3`, detached (setsid/nohup), and reads `plan.json`. `commands/` holds the pre-built launch commands,
  identical to the local driver's except for the GPU placeholder.
  - Records: `events.jsonl` (append-only) and `heartbeat.json` (rewritten every pass, about 60 s).
  - GPU load limit: 5.0 units per GPU (temporary amendment; the local driver uses 4.0), adjustable in
    `gpu_load_limit.txt`. Each launch command's own guards still apply: foreign GPU processes, RAM reserve, disk floor,
    per-GPU job and memory caps.
- **Local:** every campaign controller is stopped: the scheduler daemon, the remote and local drivers, the outage
  guard, media sync, replay cleanup, the memory monitors and the watcher. `experiments/HOLD` remains.

## 2. Restart procedure (in this order; do not just restart every loop)
1. Check the remote: `cat ~/kaist/sunho/remote_runner/heartbeat.json` and `tail events.jsonl`, plus
   `docker ps -a --filter label=s4d.managed=true`.
2. Stop the runner and import its launches before any local controller starts:
   - `.venv/bin/python -I runs/setup/tools/reconcile_remote_runner_20261031.py --stop`
   - `.venv/bin/python -I runs/setup/tools/reconcile_remote_runner_20261031.py` (dry run; check that there are no
     CONFLICT lines)
   - `.venv/bin/python -I runs/setup/tools/reconcile_remote_runner_20261031.py --apply`
3. Check the local resources (`memory_watch.py line`, disk, MIG list) and the paused checkpoints (section 3).
4. Restart the scheduler daemon first (`scripts/jobs.py daemon --interval 30`). It syncs results and verifies the
   remote jobs that finished during the outage. Then restart the outage guard, the remote campaign driver loop, the
   local driver loops, the media sync, the replay cleanup and the monitors. Use the same commands as before; see
   PROGRESS and the EXPERIMENT_LOG.
5. The paused local jobs resume automatically. Each is a crashed row with an intentional stop and no charged failure;
   the local driver relaunches it, with `--resume auto` for pretraining and `checkpoints/latest.pt` plus the replay
   snapshot for FlashSAC.

## 3. Paused local jobs (lossless, intentional-stop events)
See `paused_local_jobs` in the state file for the exact checkpoints, sha256 values and commands. FlashSAC:
`ds-flashsac-snerl-coffee-push-s2001` (300k), `ds-flashsac-mvp3d-multiview-drawer-open-s2000` (400k) and
`ds-flashsac-mvp3d-single-drawer-open-s2000` (500k), with the evaluators of the two mvp3d rows. Each has
`checkpoints/latest.pt` and `replay/ram_snapshot.npz`.

## 4. Self-render v4 pilot (priority) during the outage
- hammer and pick-place were paused at 170k and 180k, uploaded and migrated (`migrated` event, release e8ad5ac). The
  runner resumes them on the remote, then runs their exports and FlashSAC seeds 2000–2002.
- bin-picking finished locally; its export ran locally, and `encoder.pt` was uploaded with a sha256 check. Its three
  FlashSAC seeds run on the remote.

## 5. Development work (not merged, not deployed; verified 2026-10-31 20:45–21:00)
| Item | Worktree / branch | State | Evidence | Remaining |
|---|---|---|---|---|
| 3 Real-world episode contract + adapter | `.worktrees/realworld-preproc` / `realworld-preproc-20261031` (15 commits, 35 new files) | implemented, synthetic-validated | `tests/test_realworld_*` 43 passed (re-run 20:31) | first real episode + real adapter; RealSense model and stream types |
| 4 Robot-grounded calibration | same | implemented per camera and episode (DepthWorld adaptation, simplifications documented) | 10 calibration tests: pose within 0.3° / 5 mm; rejection gates | exact Robotiq URDF/meshes, measured mount; landmark detection on real images; gate tuning |
| 5 FoundationStereo + AllTracker | same | numerical parts implemented and tested; upstream workers written, never executed | stereo 10 tests (flip-and-swap right reference, alignment); lifting tests | build pinned envs (`configs/realworld/upstream_pins.yaml`), run once on real frames |
| 6 DINO / SAM 3 (optional) | same | stages and workers written; consumer contract taken from SNeRL (`dino_vits8`, layer 11 key, stride 4) | 3 baseline-stage tests (mock); workers not executed | run workers in isolated envs |
| 7 Behaviour-preserving refactor (now: `ops` package) | `ops-refactor-20261031`, **merged into main as b185ee0** (21:27) | done: operational code moved to `ops/{remote,jobs,resources,containers}`; old `s4d.*`/`scripts/*` paths are aliases of the same module objects | full native suite 1246 passed / 0 failed / 0 errors / 0 skipped on the exact merged sources (`docs/tests.json`); reconcile-tool and local-driver dry runs on the merged code | phase 2: move the active drivers out of `runs/setup/tools` and consider `scripts/remote/*`; remote releases still use their accepted commits, so the next release chain includes `ops/` |
| 8 DROID + PointWorld + DROID-3D, wrist cameras | `.worktrees/droid-largescale` / `droid-largescale-20261031` (5 commits) | implemented on schema-faithful fixtures; blocked on data | 40 new tests passed (re-run 20:20); 91 existing DROID/contract tests passed per the agent | needs user go-ahead to read the protected DROID directory, a valid HF token (gated DROID-3D), PointWorld extraction (about 3.9 TB), full suite, throughput run |

## 6. Code layout after the merge
See `docs/DEVELOPER.md`. Every restart command in section 2 keeps its old path: the files at old locations are
aliases or wrappers for the implementations in `ops/`. The remote runner and the accepted remote releases do not
use the main checkout.

## 7. Open decisions for the user
- DROID large-scale: reading the protected DROID directory, the PointWorld extraction (about 3.9 TB) and a valid
  Hugging Face token for the gated DROID-3D extrinsics are needed before real-data validation.
