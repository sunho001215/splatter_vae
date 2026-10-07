# splatter4d Meta-World campaign — progress

Last updated: 2026-10-07 18:15

## Current phase
**Phase A complete (2026-10-07 18:15).** Standalone env, native gsplat on sm_120, full suite 187/187, GPU isolation,
50-step rendered runs on GPU 4 and GPU 5 all pass (EXPERIMENT_LOG E0). Next: Phase B data collection.

## Event handling
- Event watcher: `python3 -I scripts/watch_events.py --once` as a background task. It exits on the first poll
  with events, waking the agent; the agent handles the events and restarts it. The cursor
  `experiments/watch_state.json` prevents lost or repeated events.
- Heartbeat: user `/loop` as session cron job `af519ff6`, every hour at :47 (replaces cron `806ed702`); each firing
  checks this file and the registry, runs `python3 -I scripts/heartbeat.py`, and fixes anything stuck.
  Auto-expires after 7 days; re-create it before then.
- 17:10 heartbeat: OK (watcher alive, no jobs yet, gsplat compiling, 5577 GB free).
- Build stall watcher: background loop writing `runs/setup/build_watch.log` every 5 minutes.

## Decisions applied (see docs/EXPERIMENT_LOG.md)
- E1: pretraining strides {2,4,6} uniform; validation at strides 2 and 6; RL uses the reference spacing for all methods.
- E4: DrM (official code @ 989732d6) replaces DrQ-v2 for every RL run; E3 schedules removed.
- Replay: RAM latents for frozen encoders, disk memmap frames for CNN; evaluation in a companion job per run.

## Job table
| id | GPU | status | log | W&B |
|---|---|---|---|---|
| (none yet; scheduler daemon starts after Phase A) | | | | |

## Disk
| time | free on /home/ws |
|---|---|
| 2026-10-07 17:00 | 5578 GB |

## Next actions
1. Build finishes -> native gsplat rasterization on each GPU, full test suite (0 errors / 0 failures),
   GPU isolation check, 50-step rendered pretraining on GPU 4 then GPU 5, runtime_versions.json.
2. Commit and push the Phase A milestone; start the scheduler daemon.
3. Phase B collection of all eight tasks through the scheduler.
4. Stage 0: DrM + CNN, seed 2000, hammer and shelf-place (compare hammer with the DrM paper).
