# splatter4d Meta-World campaign — progress

Last updated: 2026-10-10 13:10

## Current phase
- **Done:** Phase A (E0), Phase B (8 tasks, D2/D3 pass), Phase C (timing, `docs/COMPUTE_PLAN.md`; M2 overfit gate failed
  after three diagnosed iterations, recorded), Stage 0 (DrM + CNN hammer 0.70 / shelf-place 0), Stage 4 prep (SinCro,
  ReViWo ported), mid-campaign review items 1-5 (EXPERIMENT_LOG; coffee-push replaces shelf-place; synthetic near
  views adopted into the method configuration C).
- **Stage 1 (in progress), user directives of 2026-10-09** (all rules pre-registered in EXPERIMENT_LOG before results):
  - **S1 (first, counted iteration 4):** R = C, V1 = synthetic views as invariance positives only, V2 = C +
    self-render; 2 pretraining seeds each on hammer and pick-place at 100k of the 200k schedule, full-split
    evaluation; margins from R's two seeds. Decides the C on which items 1-2 run. 4 of 10 new runs running.
  - **Item 1 (depth redesign):** code done and tested (occlusion/free-space loss, far-plane depth validity, usage
    diagnostics, visible-only CD-centers, GPU time; worktree df1df24). D0-D4 x 2 seeds x 2 tasks queue after S1.
  - **Item 2 (M3D):** code done and tested (Gaussian-space motion loss, binned relative EPE, EPE in mm; worktree
    648ff5e). Runs on the item-1 winner (2 seeds x 2 tasks) + 6-seed hammer RL proxies per variant.
  - **Item 3 (prefetch):** benchmark script written; its first launch failed on an output-path check (worktree script
    writing into the main checkout's `runs/`), relaunched from the main checkout after the merge.
  - **S2-S6:** 6-seed hammer RL proxies for RL-based decisions; base300k/synth300k stopped (context only; length study
    redone for the final configuration); S4 CNN pick-place seed 2000 to 1M running; S5 workspace-statistics anchors
    kept; S6 gate screens retired.
  - `method-frozen-v1` only after items 1-2 (then item 1e leave-one-out and the A200/A300 length study).
- **Pretraining lengths:** ours default 300k, final L chosen on hammer/pick-place before the freeze (same L for all
  8 tasks and ablations); SinCro exactly 300k; ReViWo 100 001 (reference).

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

- 13:54: iteration-1 runs ended at 110k / 100k (rejected configuration; `train.stop_step` + SIGTERM, both exited 0).
  Gate reference done (6k); S1 not promoted. Item 2 screens crop x2 and synth-hammer running (0.8 s/step under load).

- 14:07: HEARTBEAT_OK — 33 running jobs active, watcher and scheduler alive, 5344 GB free, 261 GB RAM available.
  Gate 4a diverged (PSNR 21 -> 7.4 dB near step 1000, never recovered; not promoted); 4b 5.4k, 4c 5.8k, 4d 3.4k,
  4e 3.1k of 6k, all stable (PSNR 24.2-24.9). Item 2 screens: crop 3.4k (0.7-0.8 s/step), synth 0.7-0.8k
  (1.0-1.1 s/step), self-render just started; ~20-29 h to 100k at the current contention. Base hammer 154k,
  pick-place 165k; it2 hammer 111k, pick-place 123k.

- 15:07: HEARTBEAT_OK — 32 running jobs active, watcher and scheduler alive, 5342 GB free, 263 GB RAM available.
  Item 2 screens: crop 7.7k / 8.6k, synth 4.4k / 4.3k, self-render 4.2k / 3.6k (0.67-0.95 s/step). Item 3b at 300k
  target: plate-slide 115k (train-camera success 0.65 at 100k), assembly 131k (0.00 at 125k). Hammer 400k proxies:
  base 295k, it1 254-277k, it2 59-91k. Base pretraining 159k / 170k; it2 116k / 127k. Second-seed gate reference 1.3k.

- 16:07: HEARTBEAT_OK — 32 running jobs active, watcher and scheduler alive, 5334 GB free, 255 GB RAM available.
  No exits since 14:50 (long runs). Item 2 screens 7.7-13.9k; hammer 400k proxies: base 378k, it1 320-343k, it2
  132-157k; 3b plate-slide 0.89 at 150k (too easy), assembly 0.00 at 150k -> coffee-push and drawer-open CNN runs
  queued now (ranking order). Base pretraining 164k / 174k.

- 17:07: HEARTBEAT_OK — 26 running jobs active, watcher and scheduler alive, 5318 GB free, 272 GB RAM available.
  Item 2 screens: crop 16.8k / 18.5k, synth 11.4k / 11.2k, self-render 12.7k / 12.1k (0.67-1.08 s/step). 3b at
  300k target: plate-slide 1.00 at 200k (too easy), assembly 0.00 at 200k, coffee-push 0.11 and drawer-open 0.08 at
  25k. Hammer 400k proxies: base done, it1 s1000 done / s1001 391k, it2 209-228k. Base pretraining 169k / 179k;
  it2 126k / 137k.

- 18:07: HEARTBEAT_OK — 24 running jobs active, watcher and scheduler alive, 5317 GB free, 159 GB RAM available.
  Item 2 screens: crop 22.4k / 23.2k, synth 14.8k / 14.7k, self-render 17.9k / 17.5k. 3b: plate-slide 0.94 at 250k,
  assembly 0.00 at 250k, coffee-push 0.29 and drawer-open 0.44 at 75k. it2 hammer proxies 302k / 314k.
  Base pretraining 174k / 185k; it2 131k / 142k.

- 19:07: HEARTBEAT_OK — 17 running jobs active, watcher and scheduler alive, 5345 GB free, 158 GB RAM available.
  Fewer jobs, faster steps (0.38-0.88 s/step). Item 2 screens: crop 29.1k / 28.0k, synth 18.2k / 18.1k, self-render
  23.9k / 23.6k. 3b: plate-slide (0.89) and assembly (0.00) fail at 300k; coffee-push 0.17 and drawer-open 0.38 at
  125k. it2 hammer proxies trained to 400k (s1001 evaluation finishing). Base pretraining 179k / 192k; it2 136k / 149k.

- 19:49: base pick-place pretraining done (200k; A200). A300 runs for base queued (`s1-pretrain-*-base300k`).

- 20:07: HEARTBEAT_OK — 18 running jobs active, watcher and scheduler alive, 5339 GB free, 176 GB RAM available.
  Item 2 screens: crop 39.9k / 34.4k, synth 22.9k / 22.7k, self-render 33.1k / 33.0k. 3b: coffee-push 0.17 and
  drawer-open 0.92 at 200k. Base hammer 185k/200k; base300k runs at 1.8k / 2.0k (0.41 s/step). Base pick-place 200k
  evaluated (PSNR +0.4 dB over 100k; retrieval and EPE flat); its 200k proxies running.

- 21:07: HEARTBEAT_OK — 18 running jobs active, watcher and scheduler alive, 5328 GB free, 127 GB RAM available.
  Item 2 screens: crop 48.0k / 40.4k, synth 27.3k / 27.1k, self-render 40.4k / 40.2k. 3b: coffee-push 0.42 at 250k,
  drawer-open 0.94 at 275k (300k within ~30 min). Base hammer 192k/200k; base300k runs 10.0k / 10.3k.

- 21:38: item 3c decided — coffee-push replaces shelf-place (CNN 0.18 at 300k; plate-slide 0.89 and drawer-open 1.00
  too easy, assembly 0.00). Verification gate passed (20/20); pilot -> collection -> split / stats / D2-D3 checks ->
  held-out sets queued.

- 22:07: HEARTBEAT_OK — 11 running jobs active, watcher and scheduler alive, 5350 GB free, 123 GB RAM available.
  Base hammer pretraining done (200k; A200 for both tasks); its full evaluation and export running, 400k hammer
  proxies follow. Item 2 screens: crop 56.2k / 49.4k, synth 33.8k / 33.5k, self-render 47.9k / 47.8k (0.27-0.47
  s/step). base300k runs 18.5k / 18.8k. coffee-push collection 24/250 episodes (~2.5 h total).

- 23:07: HEARTBEAT_OK — 13 running jobs active, watcher and scheduler alive, 5343 GB free, 132 GB RAM available.
  Item 2 screens: crop 64.3k / 58.8k, synth 42.1k / 41.8k, self-render 55.3k / 55.4k. base300k 26.1k / 26.3k.
  base-200k hammer proxies 102k / 103k of 400k. coffee-push collection 77/250 (~4 h total, done ~02:00).

- 2026-10-09 00:07: HEARTBEAT_OK — 13 running jobs active, watcher and scheduler alive, 5343 GB free, 149 GB RAM
  available. Item 2 screens: crop 73.0k / 70.6k, synth 51.0k / 50.6k, self-render 63.3k / 63.3k (0.31-0.45 s/step;
  crop reaches 100k ~03:00, synth ~06:30). base300k 35.1k / 35.2k. base-200k hammer proxies 230k / 232k (0.49 / 0.36
  at 220k). coffee-push collection 191/250 (done ~01:15).

- 01:09: HEARTBEAT_OK — 12 running jobs active, watcher and scheduler alive, 5332 GB free, 144 GB RAM available.
  coffee-push data done (250 episodes, D2/D3 pass, stats done); its held-out render failed the replay check (target
  marker sites not in qpos) — fixed in the worktree (exact replay on coffee-push, other 8 tasks unchanged), suite
  running there, then merge and re-render. The watcher was down 01:00-01:07 (not re-armed after one event; replayed
  the missed events on restart). Item 2 screens: crop 82.0k / 82.9k, synth 60.1k / 60.0k, self-render 71.4k / 71.4k.
  base300k 44.4k / 44.7k; base-200k hammer proxies 360k / 365k.

- 02:07: HEARTBEAT_OK — 8 running jobs (both GPUs at 97-98 %), watcher and scheduler alive, 5330 GB free, 141 GB RAM
  available. Item 2 screens: crop 90.6k / 95.8k, synth 71.0k / 70.5k, self-render 79.3k / 79.2k. base300k 53.1k /
  53.5k. coffee-push fully ready (held-out sets rendered). Queued for the length study: exports at 100k/200k/300k of
  the base300k runs (waiting for the checkpoints), their full-split evaluations, hammer proxies (400k agent steps,
  seeds 1000/1001) at each point, pick-place proxies at 300k (reported, not counted).

- 03:07: HEARTBEAT_OK — 13 running jobs (7 training, 6 export waiters), watcher and scheduler alive, 5336 GB free,
  155 GB RAM available. Item 2: crop pick-place done and evaluated (fails 2d on pick-place: trajectory retrieval
  +0.00, CD-render +15 %); crop hammer 99.8k; synth 86.0k / 84.8k; self-render 87.1k / 87.1k (100k ~04:00).
  base300k 62.3k / 62.7k.

- 04:07: HEARTBEAT_OK — 11 running jobs, watcher and scheduler alive, 5326 GB free, 185 GB RAM available. Item 2:
  crop rejected (2d); synth hammer evaluated (trajectory retrieval 0.33 -> 0.74, hand-position R² 0.49 -> 0.89,
  CD-render p90 halved); synth pick-place done at 100k, evaluation running; self-render ~95k.

- 04:45: item 2d decided — synthetic near views adopted (passes on both tasks); crop and self-render rejected.
  Synthetic views abandon the dynamic group on pick-place (relative EPE 1.00, dynamic share 0.06): raised with the
  user. Queued: counted iteration 3 = synth (exports at 100k + hammer 400k / pick-place 200k proxies); length study
  for C = base + synth (screen runs continue to 200k = A200; synth300k runs = A300; exports, full evaluations and
  proxies follow). Base A300 runs continue as context / fallback.

- 05:07: HEARTBEAT_OK — 21 running jobs, watcher and scheduler alive, 5320 GB free, 61 GB RAM available (other
  tenants high). Iteration 3 proxies running (hammer 400k, pick-place 200k, seeds 1000/1001). Synth A200 continuations
  at 102.6k (hammer) and just resumed (pick-place); synth300k hammer started, pick-place waits for host RAM. base300k
  86.4k / 86.3k. Awaiting the user's call on the synth motion collapse (following the rule meanwhile).

- 06:07: HEARTBEAT_OK — 22 running jobs, watcher and scheduler alive, 5323 GB free, 201 GB RAM available. Synth A200
  continuations 112.8k / 110.3k; synth300k 8.0k / 6.7k; base300k 96.4k / 96.3k (base300k 100k exports next).
  Iteration-3 proxies: hammer 188-199k (0.07-0.23), pick-place 149-181k (~0.0).

- 07:07: HEARTBEAT_OK — 20 running jobs, watcher and scheduler alive, 5338 GB free, 193 GB RAM available. Synth A200
  continuations 124.2k / 121.6k (200k ~14:00); synth300k 15.1k / 13.6k (0.50-0.53 s/step; 300k in ~40 h with base300k
  sharing the GPUs); base300k 105.9k / 106.2k (context / fallback). Iteration-3 hammer proxies 338-356k (0.19-0.59);
  base300k@100k hammer proxies 52-55k.

- 08:07: HEARTBEAT_OK — 16 running jobs, watcher and scheduler alive, 5332 GB free, 95 GB RAM available. Synth A200
  continuations 137.3k / 133.9k; synth300k 22.0k / 20.5k; base300k 115.1k / 115.6k. Iteration 3 proxies done (no
  hammer RL gain within two seeds). Awaiting the user's call on the synth motion collapse.

- 11:18: HEARTBEAT_PROBLEM — host memory exhausted (6 GB available, swap full; pretraining >200 s/step); OOM kills of
  base300k@100k hammer proxies + evaluators, base300k hammer and synth300k pick-place pretraining. Paused both synth300k
  runs (held behind `hold-memory`); available back to ~36 GB, runs at ~0.5 s/step. Killed jobs restart from
  checkpoints as RAM admits. See EXPERIMENT_LOG "Fourth host-memory incident".

- 11:37-11:51: second OOM round (all pretraining killed); pretraining jobs now declare 30 GB RAM; base300k pick-place
  held behind `hold-memory` with the synth300k runs (concurrency 3: synth to-200k x2, base300k hammer). Scheduler fix
  merged (dfc5270, suite 224/224): jobs launched in the last 10 min count their declared RAM against available host
  memory; daemon restarted on it (pid in `experiments/daemon.pid`).

- 12:07: HEARTBEAT_OK — 13 running jobs, watcher and scheduler alive, 5344 GB free, 86 GB RAM available. Three
  pretraining runs at 0.23-0.26 s/step: synth to-200k 147.4k / 135.8k (200k ~16:00 / ~17:00), base300k hammer 115.8k.
  Held: base300k pick-place, synth300k x2 (memory; user decision on synth pending).

- 13:01: host memory recovered (245 GB available). Released synth300k hammer (the rule's length-study critical path;
  resumes from 20k) -> four pretraining runs. synth300k pick-place follows when a to-200k run finishes; base300k
  pick-place stays held (fallback) until the user decides on synthetic views.

- 13:07: HEARTBEAT_OK — 10 running jobs, watcher and scheduler alive, 5346 GB free, 226 GB RAM available. synth
  to-200k 161.8k / 151.3k; base300k hammer 133.1k; synth300k hammer 21.2k. Memory ample, so synth300k pick-place is
  released now (critical path of the length study) instead of waiting for a to-200k run to finish. base300k pick-place
  stays held (fallback).

- 14:07: HEARTBEAT_OK — 11 running jobs (5 pretraining), watcher and scheduler alive, 5345 GB free, 208 GB RAM
  available. synth to-200k 172.9k / 166.5k (200k ~16:30 / ~16:10); base300k hammer 148.1k; synth300k 32.8k / 34.5k
  (0.24-0.30 s/step; 300k in ~20-22 h).

- 15:07: HEARTBEAT_OK — 11 running jobs, watcher and scheduler alive, 5344 GB free, 220 GB RAM available. synth
  to-200k 184.4k / 181.5k (200k ~16:20 / ~16:15); base300k hammer 162.9k; synth300k 44.3k / 48.9k.

- 16:07: HEARTBEAT_OK — 11 running jobs, watcher and scheduler alive, 5331 GB free, 119 GB RAM available. synth
  to-200k 195.2k / 195.8k (200k ~16:30); base300k hammer 175.9k; synth300k 55.0k / 62.7k.

- 17:30: HEARTBEAT_OK — watcher and scheduler alive, 4.9 TB free, ~190 GB RAM available. All 10 S1 runs running.
  Items 1-3 code merged (ddd43dc, suite 239/239). GPU 4 had received 6 of the 10 S1 runs (my suite and smoke tests on
  GPU 5 blocked launches there) and ran them at ~0.9 s/step vs ~0.44 on GPU 5; the two youngest (synthinv-seed1 and
  synthsr-seed0 pick-place, 200 steps, no checkpoint) were restarted from step 0 pinned to GPU 5 (same seeds, so the
  same training). Loader benchmark relaunched from the main checkout (GPU 4).

- 18:07: HEARTBEAT_OK — watcher and scheduler alive, 4.9 TB free, 104 GB RAM available (host at 311 GB used). Split
  defect found and fixed (EXPERIMENT_LOG): the six seed-1 S1 runs were stopped and rerun on the fixed split
  (`s1v-*-seed1-split0-*`, merge 81692be, suite 240/240); 2 started, 4 + the two reference re-evaluations wait for host
  RAM. Seed-0 S1 runs at 4.4k-8.2k (0.30-0.49 s/step); synth200k proxies hammer 163k/166k (of 400k), pick-place s1000
  192k; S4 CNN pick-place 80k; loader benchmark at 6 of 12 settings.

- 19:07: HEARTBEAT_OK — watcher and scheduler alive, 4.9 TB free, 165 GB RAM available. Item 3 decided and merged
  (prefetch 2 x 6 workers; `ram_gb` 13 / 15 from measured PSS; reported). All 10 S1 runs running: seed 0 at
  10.9k-16.2k, seed 1 (fixed split) at 1.6k-7.2k; GPU 4 runs at 0.73-0.81 s/step next to the CNN run and two proxies,
  GPU 5 runs at 0.45-0.52. Slowest S1 run reaches 100k in ~20 h (sooner once the hammer proxies finish, 265k/270k of
  400k). S4 CNN pick-place 150k.

- 20:07: HEARTBEAT_OK — watcher and scheduler alive, 4.9 TB free, 117 GB RAM available. S1 seed 0 at 15.9k-24.1k,
  seed 1 (fixed split) at 5.7k-14.3k (GPU 4 runs 0.75-0.89 s/step, GPU 5 0.44-0.51); synth200k hammer proxies
  350k/356k of 400k; S4 CNN pick-place 212k (success 0 so far).

- 21:07: HEARTBEAT_OK — watcher and scheduler alive, 4.9 TB free, 135 GB RAM available. Hammer proxies done (context
  logged; replays deleted). S1 seed 0 at 21.3k-32.0k, seed 1 at 10.2k-21.6k (0.45-0.72 s/step; GPU 4 faster since the
  proxies ended); slowest reaches 100k in ~18 h (~15:00 tomorrow). S4 CNN pick-place 279k.

- 22:07: HEARTBEAT_OK — watcher and scheduler alive, 4.8 TB free, 142 GB RAM available, 12 jobs running. S1 seed 0
  at 27.1k-40.0k, seed 1 at 15.3k-29.3k (0.45-0.76 s/step). S4 CNN pick-place 351k, first non-zero success (0.07 on
  the training cameras at 350k).

- 23:07: HEARTBEAT_OK — watcher and scheduler alive, 4.8 TB free, 158 GB RAM available, 12 jobs running. S1 seed 0
  at 32.9k-48.0k, seed 1 at 20.4k-36.8k (0.43-0.74 s/step). S4 CNN pick-place 424k, training-camera success 0.16.

- 2026-10-10 00:07: HEARTBEAT_OK — watcher and scheduler alive, 4.8 TB free, 154 GB RAM available, 12 jobs running.
  S1 seed 0 at 38.8k-55.9k, seed 1 at 25.6k-44.3k (0.39-0.70 s/step; slowest at 100k in ~14 h). S4 CNN pick-place
  496k, training-camera success 0.18.

- 01:07: HEARTBEAT_OK — watcher and scheduler alive, 4.8 TB free, 250 GB RAM available, 12 jobs running. S1 seed 0
  at 44.8k-64.3k, seed 1 at 30.7k-51.9k (0.42-0.75 s/step). S4 CNN pick-place 572k, training-camera success 0.24.
  Spare host memory is not used for baseline pretraining now: both GPUs carry 5 S1 runs each and extra GPU load would
  delay the S1 -> item 1 -> item 2 critical path.

- 02:07: HEARTBEAT_OK — watcher and scheduler alive, 4.8 TB free, 189 GB RAM available, 12 jobs running. S1 seed 0
  at 50.7k-72.5k, seed 1 at 35.9k-59.5k (0.42-0.68 s/step; slowest at 100k ~13:30). S4 CNN pick-place 648k, success
  0.22.

- 03:07: HEARTBEAT_OK with one slowdown — watcher and scheduler alive, 4.8 TB free, but host RAM available fell to
  44-47 GB (host 353 GB used; other tenants grew, page cache shrank from ~255 to ~143 GB). S1 runs unaffected (seed 0
  at 57.8k-80.5k, seed 1 at 41.1k-66.8k, 0.44-0.61 s/step). The S4 CNN pick-place run (693k, success 0.42 at 690k) is
  not stalled but slowed from ~15 to 2.5 agent steps/s: its 33 GB frame memmap no longer stays in page cache and
  every random frame fault reads ~1.5 MB through the 2 MB device readahead (805 MB/s of reads, 538 major faults/s).
  The run is left as is (still a Stage 4 candidate seed). Fix for future CNN runs: MADV_RANDOM on the frame memmap
  (I/O hint only, sampled bytes unchanged; worktree commit, suite running).

- 03:30: host-memory incident — S4 CNN pick-place trainer OOM-killed at 693k (exit 137; available RAM 2-6 GB from
  other tenants' growth). Held until the S1 runs reach 100k, then resumes from its 650k checkpoint. Replay readahead fix merged
  (25793e0, suite 241/241). All 10 S1 runs unaffected; RAM available back to 40-69 GB. Details in EXPERIMENT_LOG.

- 04:07: HEARTBEAT_OK — watcher and scheduler alive, 4.8 TB free, 191 GB RAM available. S1 seed 0 at 66.3k-88.7k,
  seed 1 at 48.1k-74.1k (0.43-0.51 s/step); first S1 run reaches 100k ~05:30, the slowest ~11:30; their evaluations
  start automatically (configs in place). CNN pick-place trainer held (incident 03:16).

- 05:18: host-memory incident 6 — three S1 seed-1 runs and the CNN eval companion OOM-killed (host 400 GB used, swap
  full, pressure 82 %); I stopped the two slowest seed-1 runs to end the thrash (seed-0 runs back to 0.32-0.53
  s/step). The five resume from checkpoints (80k, 4 x 50k) when 75 GB is available. Watcher re-armed (it had exited on
  the crash events). Reference evaluations R seed 0 done for both tasks.

- 06:58: host memory still exhausted by other tenants (415 GB used, pressure 91-95 %). Two more S1 runs OOM-killed
  (synthsr-seed0-pick-place 80.3k, synth-seed1-split0-hammer 82.2k; resume from 80k). 3 S1 runs running, 7 waiting
  for 75 GB available. S1 completion now depends on when host memory frees.

- 07:07: watcher and scheduler alive; memory pressure gone (0 % over 60 s) but only 49 GB available (host 392 GB
  used), below the 75 GB the scheduler needs to resume an S1 run. First S1 run done (`s1v-synthinv-seed0-hammer`
  100k); its evaluation waits for RAM. Two S1 runs training fast on the emptied GPUs (0.18-0.20 s/step:
  synthsr-seed0-hammer 89.7k, synthinv-seed0-pick-place 78.2k). The 60 GB reserve is kept for now: the last two
  incidents came from 100+ GB jumps by other tenants within an hour.

- 07:49: host memory recovered (104-123 GB available); the scheduler resumed all seven waiting S1 runs from their
  checkpoints between 07:19 and 07:49. 9 S1 runs training, 1 done (synthinv-seed0-hammer, evaluated). CNN pick-place
  trainer stays held until S1 ends.

- 08:07: HEARTBEAT_OK — watcher and scheduler alive, 4.8 TB free, 107 GB RAM available, pressure ~1 %. S1: 2 done
  and evaluated (synthinv-seed0-hammer, synthsr-seed0-hammer) + both R seed-0 evaluations; 8 training (seed 0
  pick-place 89.3k / 91.4k, seed-1 hammer 53.1k-87.0k, seed-1 pick-place 52.2k-54.4k; 0.29-0.52 s/step). Slowest
  reaches 100k ~15:00, S1 decision ~15:30. Item-1 runs wait for S1 (it may change C).

- 09:07: HEARTBEAT_OK — watcher and scheduler alive, 4.8 TB free, 172 GB RAM available. S1: seed-0 pick-place runs
  at 99.1k / 99.8k (evaluations next), seed-1 hammer 60.3k-95.8k, seed-1 pick-place 59.1k-64.8k (0.30-0.50 s/step);
  4 of 12 evaluations done. Slowest at 100k ~14:50.

- 10:07: HEARTBEAT_OK — watcher and scheduler alive, 4.8 TB free, 188 GB RAM available, no pressure. S1: 8 of 12
  evaluations done; 4 seed-1 runs left (68.1k-77.1k, 0.22-0.34 s/step; slowest at 100k ~13:10). S1 margins for
  hammer recorded (R seed 1 shows the motion collapse on hammer). CNN pick-place resumed at 10:08 from 650k on
  GPU 5 (memory recovered; GPU 5 had one S1 run).

- 11:07: HEARTBEAT_OK — watcher and scheduler alive, 4.8 TB free, 185 GB RAM available. S1 seed-1 runs at
  79.0k-90.0k (0.24-0.36 s/step; slowest at 100k ~12:50). CNN pick-place 766k at ~46 agent steps/s (page cache warm
  again; 1M in ~1.5 h).

- 12:07: HEARTBEAT_OK — watcher and scheduler alive, 4.8 TB free, 201 GB RAM available. S1: 9 of 12 evaluations
  done; last three runs at 89.9k-94.8k (100k by ~13:00). CNN pick-place 883k, training-camera success 0.61 at 880k.

- 13:10: **S1 decided** — neither V1 nor V2 qualifies (both fail (c); V1 also misses (a) with pick-place dynamic share
  0.37 < 0.5); C unchanged, collapse reported as a limitation (it is a per-seed event in every variant). **Item 1
  queued** on C: D0-D4 x 2 seeds x 2 tasks (`i1-*`, 2 x 6 loader, `ram_gb` 13) + 20 full-split evaluations (configs
  in place for the started runs); 13 of 20 runs started at 13:05, the rest wait for host RAM. S4 CNN pick-place
  finished: 0.675 training cameras at 1M (replay deleted).

## Job table (running or pending)
| id | GPU | status | log |
|---|---|---|---|
| i1-{d0..d4}-seed{0,1}-{hammer,pick-place} (item 1, 100k of the 200k schedule, 2 x 6 loader) | 4/5 | 13 running, 7 waiting for host RAM (started 13:05) | runs/pretrain/i1-*/log.txt |
| fulleval-i1-*-100k (20 full-split evaluations) | any | wait for their runs | runs/fulleval-i1-*/console.log |
| base300k / synth300k pretraining and their export/proxy waiters | - | stopped and held (S3; checkpoints kept) | - |

Completed: data collection and checks, timing runs, Stage 0, review screens and proxies, S1 (10 runs + 12
evaluations), item-3 benchmark, S4 CNN pick-place (see Disk table and EXPERIMENT_LOG).

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
| 2026-10-08 16:14 | 4.9 TB (s1-proxy-it1-decdim256-pick-place-s1000 completed with final eval at 200k; deleted its replay/, 0.58 GB) |
| 2026-10-08 16:30 | 4.9 TB (s1-proxy400-base-hammer-s1001 completed with final eval at 400k; deleted its replay/, 0.58 GB) |
| 2026-10-08 16:31 | 4.9 TB (s1-proxy400-base-hammer-s1000 completed with final eval at 400k; deleted its replay/, 0.58 GB) |
| 2026-10-08 16:46 | 4.9 TB (s1-proxy-it1-decdim256-pick-place-s1001 completed with final eval at 200k; deleted its replay/, 0.58 GB) |
| 2026-10-08 17:01 | 4.9 TB (s1-proxy-it1-decdim256-hammer-s1000 completed with final eval at 400k; deleted its replay/, 0.58 GB) |
| 2026-10-08 17:18 | 4.9 TB (s1-proxy-it1-decdim256-hammer-s1001 completed with final eval at 400k; deleted its replay/, 0.58 GB) |
| 2026-10-08 18:41 | 4.9 TB (s3b-cnn-assembly-s2000 completed with final eval at 300k; deleted its replay/, 14 GB used of a 49 GB sparse memmap) |
| 2026-10-08 19:05 | 4.9 TB (s3b-cnn-plate-slide-s2000 completed with final eval at 300k; deleted its replay/, ~14 GB) |
| 2026-10-08 19:06 | 4.9 TB (s1-proxy-it2-lambdadyn4-hammer-s1000 completed with final eval at 400k; deleted its replay/, 0.58 GB) |
| 2026-10-08 19:10 | 4.9 TB (s1-proxy-it2-lambdadyn4-hammer-s1001 completed with final eval at 400k; deleted its replay/, 0.58 GB) |
| 2026-10-08 21:30 | 4.9 TB (s3b-cnn-drawer-open-s2000 completed with final eval at 300k; deleted its replay/, ~14 GB) |
| 2026-10-08 21:38 | 4.9 TB (s3b-cnn-coffee-push-s2000 completed with final eval at 300k; deleted its replay/, ~14 GB) |
| 2026-10-08 21:51 | 4.9 TB (s1-proxy-base200k-pick-place-s1000 completed with final eval at 200k; deleted its replay/, 0.58 GB) |
| 2026-10-08 22:07 | 4.9 TB (s1-proxy-base200k-pick-place-s1001 completed with final eval at 200k; deleted its replay/, 0.58 GB) |
| 2026-10-09 01:27 | 4.9 TB (s1-proxy-base200k-hammer-s1001 completed with final eval at 400k; deleted its replay/, 0.58 GB) |
| 2026-10-09 01:31 | 4.9 TB (s1-proxy-base200k-hammer-s1000 completed with final eval at 400k; deleted its replay/, 0.58 GB) |
| 2026-10-09 06:19 | 4.9 TB (s1-proxy-it3-synth-pick-place-s1000 completed with final eval at 200k; deleted its replay/, 0.58 GB) |
| 2026-10-09 06:37 | 4.9 TB (s1-proxy-it3-synth-pick-place-s1001 completed with final eval at 200k; deleted its replay/, 0.58 GB) |
| 2026-10-09 07:26 | 4.9 TB (s1-proxy-it3-synth-hammer-s1000 completed with final eval at 400k; deleted its replay/, 0.58 GB) |
| 2026-10-09 07:37 | 4.9 TB (s1-proxy-it3-synth-hammer-s1001 completed with final eval at 400k; deleted its replay/, 0.58 GB) |
| 2026-10-09 13:03 | 4.9 TB (s1-proxy-base300k-100k-hammer-s1001 completed with final eval at 400k; deleted its replay/, 0.58 GB) |
| 2026-10-09 13:05 | 4.9 TB (s1-proxy-base300k-100k-hammer-s1000 completed with final eval at 400k; deleted its replay/, 0.58 GB) |
| 2026-10-09 18:01 | 4.9 TB (s1-proxy-synth200k-pick-place-s1001 completed with final eval at 200k; deleted its replay/, 0.56 GB) |
| 2026-10-09 18:15 | 4.9 TB (s1-proxy-synth200k-pick-place-s1000 completed with final eval at 200k; deleted its replay/, 0.56 GB) |
| 2026-10-09 20:39 | 4.9 TB (s1-proxy-synth200k-hammer-s1001 completed with final eval at 400k; deleted its replay/, 0.56 GB) |
| 2026-10-09 20:43 | 4.9 TB (s1-proxy-synth200k-hammer-s1000 completed with final eval at 400k; deleted its replay/, 0.56 GB) |
| 2026-10-10 12:56 | 4.8 TB (s4-drm-cnn-pick-place-s2000 completed with final eval at 1M; deleted its replay/, 46 GB apparent / ~33 GB used) |

## Next actions
1. Item 1: runs reach 100k (~1-1.5 days with 13-20 concurrent runs) -> 20 evaluations -> margins from D0's two seeds
   (written before comparing) -> item-1 rule -> report 1d to the user.
2. Item 2: M3D on the item-1 winner (2 seeds x 2 tasks; `analysis/queue_variants.py item2`) + 6-seed hammer RL
   proxies for M2D and M3D -> item-2 rule -> report 2b.
3. Item 1e leave-one-out, length study (A200/A300 of the final configuration, 6-seed RL), `method-frozen-v1`, Stage 1
   final RL (seeds 1000-1002, 1M).
