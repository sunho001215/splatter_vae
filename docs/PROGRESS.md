# splatter4d Meta-World campaign — progress

Last updated: 2026-10-08 13:08

## Current phase
- **Phase A** complete (E0). **Phase B** complete: 8 tasks, 162 GB, D2/D3 pass (EXPERIMENT_LOG "Phase B result").
- **Phase C** in progress: RL timing done (CNN GPU-bound ~80 steps/s/GPU; frozen ~190 steps/s/GPU with 8 runs);
  pretraining 0.22 s/step at 8 loader workers (loader-bound; 32 workers now); `docs/COMPUTE_PLAN.md` written.
  M2 overfit gate FAILED after three diagnosed iterations (G1-G3; recorded in RESULTS.md).
- **Stage 0** done: DrM + CNN hammer 0.70 train-camera / 0.02 held-out success at 1M; shelf-place 0 (no reward ever: sparse v3 reward). See EXPERIMENT_LOG.
- **Stage 1** in progress: base pretraining (200k schedule) on hammer and pick-place at ~120k; improvement iterations
  1 (decoder dim 256) and 2 (lambda_dyn 4) at 70-77k of their 200k schedules, compared with base at 100k.
- **Pretraining length (user update, 2026-10-08).** Ours: default 300k; the final length (200k/300k/400k) is chosen on
  hammer and pick-place by the pre-registered rule in EXPERIMENT_LOG ("Pretraining length") and fixed before
  `method-frozen-v1`, for all 8 tasks and all ablations. SinCro: exactly 300k steps on every task. ReViWo: reference
  100 001 steps. Configs updated (`base.yaml` train.steps 300000; SinCro `max_global_steps: 300000`); the six running
  200k-schedule jobs pin `train.steps=200000` in the queue. Suite 204/204 on the new configs.
- **Stage 4 prep** done: SinCro and ReViWo ported (merge 67f5984, `docs/BASELINES.md`); suite 199/199. Measured cost
  under load: SinCro 0.53 s/step (300k steps ~45 h/task, ~15 GPU-days for 8 tasks), ReViWo 0.58 s/step (100k steps
  ~16 h/task). SinCro's budget is settled by the user's 300k decision.

- **Mid-campaign review (user, ~11:40)** integrated: one EXPERIMENT_LOG entry per item with designs and decision
  rules fixed before any run. Code is developed in the worktree `.worktrees/review` (branch `review-items`) so the
  main checkout's test evidence stays valid for the scheduler; it is merged once the full suite passes there.
  - Item 1a done: near/trajectory held-out sets rendered for all 8 tasks by replaying stored states (training-camera
    replay reproduces the stored frames; goal sites and the moved shelf body are restored from the stored data),
    `/home/ws/data/metaworld/splatter4d_v1/heldout_sets/<task>.hdf5` (1.4-2.3 GB per task, ~14 GB in total).
  - Item 3a done: push-back fails (expert 0.64); ranking plate-slide, assembly, coffee-push, drawer-open, lever-pull,
    sweep-into; 3b CNN runs on plate-slide and assembly queue after the merge.
  - Items 1b/1c, 2, 4 code written with tests (held-out diagnostics, crop / synthetic views / self-render, decoder
    options, moving-pixel motion normalisation, depth-hard boost, `train.stop_step`); suite running in the worktree.
  - Item 5: queued hammer proxies run 400k agent steps; base-100k hammer proxies rerun at 400k (running).

## Event handling
- Event watcher: `python3 -I scripts/watch_events.py --once` as a background task, re-armed after each event.
- Hourly check: user `/loop`, session cron job `af519ff6` at :47 (expires after 7 days).
- Scheduler daemon: `scripts/jobs.py daemon --interval 30` (pid `experiments/daemon.pid`, log `experiments/daemon.log`).
  `touch experiments/HOLD` while editing or testing sources (every file in `s4d/ scripts/ tests/ configs/` is
  fingerprinted by the test gate); remove it after the suite passes.

## Heartbeats
- 20:13: HEARTBEAT_OK — 8 running jobs active (last activity < 3 min), watcher and scheduler alive, 5375 GB free.
  hammer pretraining at 5k/200k (0.175 s/step), pick-place at 2.9k (0.34 s/step), gate arms at 2.0k (3a) / 1.3k (3b).

- 21:20: HEARTBEAT_OK — 7 running jobs. Two Stage 0 evaluations took ~20 min (20:40-21:03) while another test run
  shared GPU 5; back to ~80-100 s and catching up. Heartbeat now counts eval companions' writes in the training run
  (they looked idle 25+ min). Progress: hammer pretraining 24k/200k (PSNR 23.7, retrieval 0.73 at 20k), pick-place
  12k/200k, gate 3b at 7k (PSNR 25.4, rel EPE 0.38 s2), Stage 0 at 230k (hammer train-camera success 0.23).

- 22:08: HEARTBEAT_OK — 9 running jobs active, watcher and scheduler alive, 5352 GB free. Screens at ~1.4k/6k
  (no difference from the reference yet); hammer base 39k/200k (0.26 s/step while sharing GPU 4 with the screens).

- 22:15 GPU 5 process audit: all 29 GPU processes belong to active registered jobs (Stage 0 train 864132/864134,
  their eval companions 1018246/1018247 with 12 spawned render workers each, pick-place pretraining 1255925); no
  orphans, no collectors, no foreign processes.

- 23:10: HEARTBEAT_PROBLEM 1 — the watcher had been dead for 25 min because I did not re-arm it after the
  `screen-s1-motion20` completion event; restarted (the registry cursor replays missed events). 10 running jobs OK.
  GPU 4 at 98% with 5 pretraining jobs (~0.4 s/step each); available RAM 90 GB (loader shared memory) — no further
  memory-heavy launches until some finish. Stage 0 at ~430k/1M (~6 h left).

- 00:08: HEARTBEAT_OK, but Stage 0 had slowed to 5 steps/s (page-cache thrashing from pretraining loaders). Loader
  workers reduced to 12/8; each pretraining run restarts after its next checkpoint (hammer base and it1-hammer done at
  00:13; pick-place base at 50k, it1-pick-place and it2-hammer at 10k, it2-pick-place at 20k pending). See log entry.

- 00:57: all six pretraining runs restarted from their checkpoints with 12/8 loader workers (no steps lost);
  available RAM 76 -> ~160 GB; Stage 0 back to ~25 steps/s.

- 01:10: HEARTBEAT_OK — 10 running jobs, 184 GB RAM available, Stage 0 at ~29 steps/s (hammer 573k, shelf-place
  553k). A second simultaneous ~20 min Stage 0 evaluation spike (both ended 01:04; first one 21:03) — evaluators
  caught up within one snapshot each time; watching for a third occurrence before investigating further.
  Pretraining: hammer base 67k, pick-place base 55k, iterations 16-20k (all ~0.4 s/step).

- 02:10: HEARTBEAT_OK, but available RAM 25 GB from other tenants' growth (~290 GB anon not ours); Stage 0 hammer at
  1.8 steps/s. Pausing the 4 iteration runs at their 30k checkpoints (held behind `hold-memory` in the queue).

- 02:27: host OOM killer killed both base pretraining runs (other tenants' memory). Orphaned loader workers
  terminated; scheduler now kills leftover session processes. Iterations pausing at 30k; base runs resume from
  70k / 60k with 8 workers once all four are paused (scheduler HOLD until then).

- 02:55: all four iteration runs paused at their 30k checkpoints (held behind `hold-memory`); base runs resumed at
  02:53 from 70k (hammer) and 60k (pick-place) with 8 loader workers. Available RAM 76 GB; resume the iterations when
  it stays above ~120 GB.

- 03:18: resume fast-forward fixed (index-level skip); pick-place base restarted onto it. Available RAM ~300 GB
  (other tenants released memory) -> iterations released and resumed from 30k. 10 jobs running.

- 04:10: HEARTBEAT_OK — 10 jobs, 189 GB RAM available. Eval spikes investigated: 9 per evaluator since 20:40, mostly
  simultaneous on both (e.g. 03:44-04:05), i.e. a shared external cause: every job inherits nice 5 from the agent
  shell, so CPU-bound eval workers yield to other tenants under host load (load 110-230). Not reniced (that would
  compete with other users); evaluators catch up, results unaffected. Stage 0 at 782k / 752k (~2.5 h left);
  base pretraining 81k / 70k (100k export in ~2-3 h); iterations at 37.5k.

- 05:10: HEARTBEAT_OK — 10 jobs. Stage 0 at 882k / 851k; hammer evaluator 6 snapshots behind (catches up after
  training). Queued the base-encoder DrM proxy: `export-s1-base-{task}-100k` waits for the 100k checkpoint, then
  `s1-proxy-base-{task}-s{1000,1001}` (200k agent steps) and their evaluators start on GPU 5.

- 06:08: HEARTBEAT_OK — 12 jobs (incl. 2 export jobs waiting for 100k). Stage 0 hammer 983k (train-camera success
  0.62-0.64), shelf-place 949k (0.0); hammer base 98.4k, pick-place base 92.1k, iterations ~54k.

- 07:08: HEARTBEAT_OK — 14 jobs, 155 GB RAM available, 5.4 TB free. Base pretraining 107k / 105k; iterations ~63k.
  Base-encoder proxy (100k encoder) on hammer at 100k agent steps: train-camera success 0.42 / 0.34, held-out 0.07 /
  0.04 (seeds 1000 / 1001); for reference the Stage 0 CNN had 0.02 / 0.00 at 100k (different seed; informal).
  Pick-place proxies at ~50k: 0.0 so far.

- 08:08: HEARTBEAT_OK — 8 jobs + 4 export jobs. Base hammer proxies done (final train-camera 0.13 / 0.29, peak 0.44
  near 100k then decline); base pick-place s1001 done (0.0), s1000 finishing. Queued iteration proxies: exports wait
  for the iterations' 100k checkpoints (~3 h), then 8 DrM proxy runs + evaluators on GPU 5. RAM available 93 GB.

- 08:45: OOM cascade from other tenants (5 pretraining runs killed). Scheduler now admits jobs only when host RAM
  allows (`ram_gb` + 40 GB reserve). Hammer base relaunched; pick-place base, it1 x2, it2-pick-place waiting for RAM.

- 09:20: more OOM kills (it1-pick-place, it2-hammer). OOM priorities added (iterations most killable). Running: both
  base runs, it1-hammer; waiting for RAM: it1-pick-place, it2-hammer, it2-pick-place (79 GB available, need 80).

- 09:21: HEARTBEAT_OK — running: base hammer 116k, base pick-place 115k, it1-hammer 75k (0.22-0.25 s/step) + 4 export
  waiters; waiting for host RAM (78 GB available, 80 needed): it1-pick-place (70k ckpt), it2-hammer (70k), it2-pick-place (70k).

- 09:23: host RAM freed up (~104 GB available); the scheduler readmitted it1-pick-place (attempt 6, resumes from
  70k) and it2-hammer (attempt 4, from 70k) on GPU 4. it2-pick-place still waits (64 GB available after the launches).

- 09:28: it2-pick-place readmitted on GPU 4 (attempt 4, from 70k). GPU 4 then ran five pretraining jobs at 98%
  utilisation (0.32-0.35 s/step) while GPU 5 ran one (0.30 s/step), so its queue entry was pinned to GPU 5 and the job
  was stopped (SIGTERM to its own session) one minute after resuming; relaunched on GPU 5 at 09:30 (attempt 5, from
  the same 70k checkpoint).

- 10:15: queued full-split `scripts/evaluate.py` runs (job dirs `runs/fulleval-*`, outputs kept out of the training
  runs) for base/it1/it2 at 100k (iteration comparison; in-training validation is too noisy, see EXPERIMENT_LOG) and
  base at 200k, plus the base-200k exports and DrM proxies (seeds 1000/1001). Base 100k evaluations started on GPU 5.

- 10:17: HEARTBEAT_OK — 14 running jobs (6 pretraining, 6 export waiters, 2 full-split evaluations), watcher and
  scheduler alive, 5400 GB free, 217 GB RAM available. Base 126k/200k (0.27 s/step, 200k at ~15:45); it1-hammer 85k,
  it1-pick-place 79k, it2-hammer 80k, it2-pick-place 80k (100k at ~11:45-12:15). Nothing stuck.

- 10:20: full-split evaluations of the base encoders at 100k done (4.5 min each; ~2000 windows per stride). hammer /
  pick-place at s2: moving PSNR 25.1 / 16.2 dB, held-out PSNR 14.8 / 14.6, rel. EPE 0->2 0.69 / 0.98, train-camera
  retrieval 0.83 / 0.95, hand-velocity probe R² 0.64 / 0.26. The in-training validation (first 8 batches) is biased,
  not just noisy: it gave rel. EPE 0.47 for hammer at 100k vs 0.69 on the full split. Iteration and length
  comparisons use the full split only.

- 11:07: HEARTBEAT_OK — 12 running jobs (6 pretraining, 6 export waiters), watcher and scheduler alive, 5378 GB
  free, 166 GB RAM available. Base hammer 135k / pick-place 140k (200k at ~17:00 / ~16:00); it2-pick-place 95k and
  it1-hammer 94k (100k at ~11:30-11:40), it2-hammer 89k and it1-pick-place 88k (~12:10). Nothing stuck.

- 12:19-12:43: review code merged twice (d3f7364, ac02dd0; suites 222/222 and 223/223). The second merge vectorises
  the oracle and synthetic-view splatting (a per-window loop made a full evaluation ~45 min; now ~13 min). Launched:
  3b DrM + CNN on plate-slide and assembly (light evaluation), full-split evaluations with the new metrics (base,
  it1, it2 at 100k; S1 at 6k), gate screens g-ref and 4a-4c. Declared GPU memory set to measured values (pretraining
  ~4.5 GB actual, declared 7; evaluations 5; CNN 5; frozen RL 2; evaluators 3) and slots raised to 20 per GPU, because
  the inflated declarations (12 per pretraining run) blocked GPU 4 at 1 GB of real use per 3 declared. Pending proxies
  (item 5) moved to priority 3, after the screens, as the review orders. Both GPUs run at ~95 %, so everything is
  slower (CNN ~14 agent steps/s).

- 12:43: HEARTBEAT_OK — 36 running jobs active, watcher and scheduler alive, 5350 GB free, 96 GB RAM available.
  Contention: with 6 evaluations, 4 gate screens and 2 CNN runs added, pretraining slowed from ~0.3 to 0.6-0.8 s/step
  (base hammer 149k, pick-place 159k; it1/it2 102-115k past their 100k comparison point); evaluations end within
  ~30 min. Two old-named evaluations (fulleval-s1-it2-hammer, fulleval-s1-it1-pick-place) launched with the merged
  code and duplicate their fulleval2 twins; left to finish (~20 min of GPU).

- 13:07: HEARTBEAT_OK — 36 running jobs active, watcher and scheduler alive, 5350 GB free, 200 GB RAM available.
  No exits since 12:41: the six full-split evaluations (base, it1, it2 at 100k) have finished stride 2 and are on
  stride 6 (GPU contention); gate screens 4d/4e wait for GPU 4's declared-memory budget (81 of 90 GB).

## Job table (running or pending)
| id | GPU | status | log | W&B |
|---|---|---|---|---|
| stage0-drm-cnn-hammer-s2000 (+ -eval) | GPU 5 | running | runs/stage0-drm-cnn-hammer-s2000/console.log | splatter4d-rl / stage0-drm-cnn-hammer-s2000 |
| stage0-drm-cnn-shelf-place-s2000 (+ -eval) | GPU 5 | running | runs/stage0-drm-cnn-shelf-place-s2000/console.log | splatter4d-rl / stage0-drm-cnn-shelf-place-s2000 |
| s1-pretrain-hammer-base | GPU 4 | running (200k) | runs/s1-pretrain-hammer-base/console.log, runs/pretrain/s1-pretrain-hammer-base/ | splatter4d-metaworld / s1-pretrain-hammer-base |
| s1-pretrain-pick-place-base | GPU 5 | running (200k) | runs/s1-pretrain-pick-place-base/console.log, runs/pretrain/s1-pretrain-pick-place-base/ | splatter4d-metaworld / s1-pretrain-pick-place-base |
| export-s1-base-* -> s1-proxy-base-{task}-s{1000,1001} | GPU 5 | hammer done; pick-place s1001 done, s1000 finishing |
| export-s1-it{1,2}-*-100k -> s1-proxy-it{1,2}-*-{task}-s{1000,1001} | GPU 5 | waiting for 100k checkpoints | runs/<id>/console.log | splatter4d-rl |
| s1-it1-decdim256-{hammer,pick-place} (iteration 1, 200k schedule, compared at 100k) | GPU 4 | running (resumed 03:17 from 30k) | runs/pretrain/s1-it1-*/log.txt | splatter4d-metaworld |
| s1-it2-lambdadyn4-{hammer,pick-place} (iteration 2) | GPU 4 | running (resumed 03:17 from 30k) | runs/pretrain/s1-it2-*/log.txt | splatter4d-metaworld |

Completed: all `collect-/split-/stats-/check-<task>` (32), timing runs (`timing-*`, `timing2-*`), gate it1/it2.

## Disk
| time | free on /home/ws |
|---|---|
| 2026-10-07 17:00 | 5578 GB |
| 2026-10-07 18:45 | 5.0 TB |
| 2026-10-07 19:02 | 5.0 TB (after timing-cnn-c1 finished; deleted runs/timing-cnn-c1/replay/, 0.98 GB) |
| 2026-10-07 19:55 | 4.9 TB (timing runs finished; their replays are kept: no final evaluation) |
| 2026-10-08 06:30 | 4.9 TB (stage0-drm-cnn-hammer-s2000 completed with final eval at 1M; deleted its replay/, 46 GB) |
| 2026-10-08 06:50 | 4.9 TB (stage0-drm-cnn-shelf-place-s2000 completed with final eval at 1M; deleted its replay/, 46 GB) |
| 2026-10-08 07:35 | 5.0 TB (s1-proxy-base-hammer-s1001 completed with final eval at 200k; deleted its replay/, 0.56 GB) |
| 2026-10-08 07:45 | 5.0 TB (s1-proxy-base-hammer-s1000 completed with final eval at 200k; deleted its replay/, 0.56 GB) |
| 2026-10-08 08:08 | 5.0 TB (s1-proxy-base-pick-place-s1001 completed with final eval at 200k; deleted its replay/, 0.56 GB) |
| 2026-10-08 08:47 | 5.0 TB (s1-proxy-base-pick-place-s1000 completed with final eval at 200k; deleted its replay/, 0.56 GB) |
| 2026-10-08 13:46 | 4.9 TB (s1-proxy-it2-lambdadyn4-pick-place-s1000 completed with final eval at 200k; deleted its replay/, 0.58 GB) |
| 2026-10-08 13:48 | 4.9 TB (s1-proxy-it2-lambdadyn4-pick-place-s1001 completed with final eval at 200k; deleted its replay/, 0.58 GB) |

## Next actions
1. ~12:30: iterations reach 100k -> exports -> DrM proxies (seeds 1000/1001, 200k agent steps); compare with base at
   100k (pretraining metrics at strides 2 and 6 + proxies) -> choose the method configuration C; stop iteration runs
   that do not help (the winner continues to 200k as its A200).
2. Launch C with the default 300k schedule on hammer and pick-place (A300); exports at 100k/200k/300k.
3. ~16:00: base runs reach 200k -> export -> DrM proxies; full-split `scripts/evaluate.py` on the A200 endpoints.
4. Length decision by the pre-registered rule (A200 vs A300) -> EXPERIMENT_LOG; then Stage 1 final DrM RL at L with
   seeds 1000-1002 (1M steps) and the Stage 1 report to the user.
