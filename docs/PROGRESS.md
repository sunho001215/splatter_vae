# splatter4d Meta-World campaign — progress

Last updated: 2026-10-07 18:45

## Current phase
**Phase B (data collection) running; Phase C RL timing running.** Phase A complete (EXPERIMENT_LOG E0).
RL algorithm: DrM (official code @ 989732d6) for every RL run (E4).

## Event handling
- Event watcher: `python3 -I scripts/watch_events.py --once` as a background task; exits on the first poll with
  events (wake-up), then re-armed. Cursor: `experiments/watch_state.json`.
- Hourly check: user `/loop`, session cron job `af519ff6` at :47 (expires after 7 days; re-create before then).
- Scheduler daemon: `scripts/jobs.py daemon --interval 30` (pid in `experiments/daemon.pid`, log
  `experiments/daemon.log`), restarted 18:34 after fixing its session-id check.

## Decisions applied (see docs/EXPERIMENT_LOG.md)
- E1: pretraining strides {2,4,6} uniform; validation at strides 2 and 6; RL uses the reference spacing for all methods.
- E4: DrM replaces DrQ-v2 for every RL run; no per-task overrides for the campaign tasks.
- E5: seeded evaluation resets are history-free (Meta-World 3.0 ignores reset seeds).
- Replay: RAM fp16 latents for frozen encoders; disk memmap frames for CNN with a prefetch thread.

## Job table
| id | GPU | status | log | W&B |
|---|---|---|---|---|
| collect-door-open | GPU 4 | running (230/250) | runs/collect-door-open/console.log | - |
| collect-hammer | GPU 5 | running (218/250) | runs/collect-hammer/console.log | - |
| collect-peg-unplug-side | GPU 4 | running (158/250) | runs/collect-peg-unplug-side/console.log | - |
| collect-stick-push | GPU 5 | running (202/250) | runs/collect-stick-push/console.log | - |
| collect-pick-place | GPU 4 | running (216/250) | runs/collect-pick-place/console.log | - |
| collect-peg-insert-side | GPU 5 | running (190/250) | runs/collect-peg-insert-side/console.log | - |
| collect-shelf-place | GPU 4 | running (173/250) | runs/collect-shelf-place/console.log | - |
| collect-bin-picking | GPU 5 | running (203/250) | runs/collect-bin-picking/console.log | - |
| split-/stats-/check-<task> (24) | any | pending (after each collection) | runs/<id>/console.log | - |
| timing-cnn-c1 | GPU 4 | running (old code; 22 fps) | runs/timing-cnn-c1/console.log | disabled |
| timing-cnn-c1-eval | GPU 4 | pending (eval throughput) | runs/timing-cnn-c1-eval/console.log | disabled |
| timing2-cnn-c1, -c4-{0..3}, -c8-{0..7} | GPU 4 | pending (concurrency on new code) | runs/<id>/console.log | disabled |

## Disk
| time | free on /home/ws |
|---|---|
| 2026-10-07 17:00 | 5578 GB |
| 2026-10-07 18:45 | 5.0 TB |
| 2026-10-07 19:02 | 5.0 TB (after timing-cnn-c1 finished; deleted runs/timing-cnn-c1/replay/, 0.98 GB) |

## Next actions
1. Collections finish -> splits, workspace stats, D2/D3 geometry checks and sanity panels run as dependent jobs.
2. Timing results -> docs/COMPUTE_PLAN.md (per-GPU concurrency, GPU-hours per run).
3. Phase C pretraining gates on hammer: M2 overfit (one episode, 3k steps), 500-step timing.
4. Stage 0: DrM + CNN, seed 2000, hammer and shelf-place (compare hammer with the DrM paper).
