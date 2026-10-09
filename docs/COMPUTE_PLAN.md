# Compute plan (Meta-World campaign)

Measured on GPU 4 / GPU 5 (RTX PRO 6000 Blackwell, 96 GB each), 384 CPU cores, ~140 GB RAM available to this
campaign (the host is shared), 5.0 TB free disk. All numbers from scheduler jobs in `runs/`.

## Measurements

| Workload | Configuration | Throughput | GPU memory | RAM | CPU | Source |
|---|---|---|---|---|---|---|
| Pretraining (ours) | batch 16, 6 cameras x 3 times, 8 loader workers | 0.22 s/step alone (0.06-0.10 s waiting for data); 0.25-0.35 s/step with 2-5 runs per GPU | 3.2 GB | ~6 GB + workers | loader-bound | `runs/pretrain/timing-pretrain-hammer` |
| Pretraining loader | one window (6+4 cameras, 3 times, motion for 6 pairs) | 76 ms/window/core; 16-window batch: 258 / 178 / 152 ms with 8 / 16 / 24 workers | - | - | 1 core/worker | loader benchmark (E-log G1) |
| DrM + CNN, alone | 128x128, batch 256, update every 2 steps | 52 agent steps/s; update 30 ms, sampling hidden by prefetch | 1.7 GB + renderer | 3.2 GB | 6 cores | `runs/timing2-cnn-c1` |
| DrM + CNN, 4 per GPU | same | 20 steps/s each, 80 total: the GPU saturates | same | same | 3 cores each | `runs/timing2-cnn-c4-*` |
| DrM + frozen splatter4d, alone | latents in RAM | 39 steps/s (GPU shared with 2 CNN runs) | 0.13 GB + renderer | 2.2 GB (0.5 GB replay) | 1 core | `runs/timing-s4d-c1` |
| DrM + frozen splatter4d, 8 per GPU | same | 24 steps/s each, ~190 total | same | same | 1 core each | `runs/timing-s4d-c8-*` |
| Evaluation (companion job) | 240 episodes, 6 workers x 8 envs | 25 s per snapshot (100 snapshots per run: ~42 min) | ~3 GB while evaluating | ~4 GB | 6 cores while evaluating | `runs/timing-cnn-c1-eval` |
| Renderer | one MuJoCo EGL renderer (scene textures) | - | ~0.44 GB | - | - | GPU 5 process table |

## Concurrency per GPU

- **CNN (DrM, end-to-end):** GPU-bound. A second concurrent run already costs ~25% per run, so at most 2 CNN runs per
  GPU, and CNN runs are spread over both GPUs. One 1M-step CNN run needs ~3.5 GPU-hours of saturated GPU time.
- **Frozen encoders (ours, SinCro, ReViWo):** latency-bound (one core, small kernels). Up to 12 per GPU; per-run speed
  ~20-25 steps/s when mixed with other work, ~12 h per 1M-step run.
- **Pretraining:** one run nearly saturates a GPU at full loader speed; 32 loader workers per run. At most 2 per GPU
  when mixed with RL.
- Admission in `experiments/queue.yaml`: per-GPU `max_jobs` 16 and `max_mem_gb` 80 (declared per job), so renderer
  memory and pretraining stay below the 92% alarm of the watcher.

## Planned runs, cost and priority tier

Tiers (finish in order if compute runs short): 1 = Ours vs CNN on all 8 tasks; 2 = SinCro and ReViWo on all 8 tasks;
3 = RL ablations; 4 = remaining representation ablations, K sweep, extra seeds.

**Pretraining lengths (user update, 2026-10-08).** Ours: 300k steps by default; the final length L is chosen on
hammer and pick-place before `method-frozen-v1` (200k, 300k or 400k by the pre-registered rule in
`docs/EXPERIMENT_LOG.md`) and used for all 8 tasks and every ablation. SinCro: exactly 300k steps on every task.
ReViWo: 100 001 steps (reference). Costs below assume L = 300k; measured pretraining speed under sharing is
0.25-0.35 s/step, i.e. ~21-29 h of wall-clock per 300k run with 2-3 runs per GPU (~12-18 GPU-hours each).

| Stage | Runs | Count | Est. GPU-hours (each) | Est. GPU-hours (total) | Tier |
|---|---|---|---|---|---|
| 0 | DrM + CNN sanity (hammer, shelf-place), seed 2000 | 2 | 3.5 | 7 (done) | 1 |
| 1 | pretraining, ours (base), hammer and pick-place, 200k schedule (A200 of the length study) | 2 | 8-12 | 24 | 1 |
| 1 | improvement iterations (<= 8): pretraining 100k + RL proxy 2 seeds x 2 tasks | <= 8 x (2 + 4) | 5 + 4 x 1 | <= 72 | 1 |
| 1 | pretraining, method configuration, hammer and pick-place, 300k schedule (A300) | 2 | 12-18 | 30 | 1 |
| 1 | length-study DrM proxies (100k/200k/300k of A300, A200), 2 seeds x 2 tasks, + full-split evaluations | 16 | 0.5 (shared) | 8 | 1 |
| 1 | contingency: L = 400k, or a 200k run for a combined configuration | 2 | 8-24 | <= 48 | 1 |
| 1 | DrM + ours at L, hammer and pick-place, seeds 1000-1002 | 6 | 1 (shared) | 6 | 1 |
| 2 | representation ablations at L, hammer and pick-place (no coverage, single group, no motion, T=1, mis-scaled x2) | 12 | 12-18 | 180 | 3 (tier 4 if short) |
| 2 | RL ablation encoders not shared with the list above (no InfoNCE) at L | 2 | 12-18 | 30 | 3 |
| 2 | DrM RL ablations (T=1, no motion, no InfoNCE, single group) x 2 tasks x 3 seeds | 24 | 1 (shared) | 24 | 3 |
| 3 | pretraining at L, frozen method, the 6 remaining tasks (development encoders reused) | 6 | 12-18 | 90 | 1 |
| 3 | DrM + ours, 8 tasks x seeds 2000-2002 | 24 | 1 (shared) | 24 | 1 |
| 4 | DrM + CNN, 8 tasks x 3 seeds (Stage 0 runs reused only if code and protocol are unchanged) | 22-24 | 3.5 | 84 | 1 |
| 4 | SinCro pretraining, 8 tasks, exactly 300k steps (0.53 s/step measured on a shared GPU, + 300 validations) | 8 | ~45 | ~360 | 2 |
| 4 | ReViWo pretraining, 8 tasks, 100 001 steps (0.58 s/step measured on a shared GPU) | 8 | ~16 | ~130 | 2 |
| 4 | DrM + SinCro / ReViWo, 8 tasks x 3 seeds | 48 | 1 (shared) | 48 | 2 |
| 5 | evaluation companions, probes, figures | all | small | ~20 | - |

Estimated total: ~700 GPU-hours excluding SinCro/ReViWo pretraining, ~1 190 GPU-hours including them (~25 days on
two GPUs at full packing; pretraining and frozen-encoder RL overlap, so wall-clock is lower). Tier 1 alone is ~420
GPU-hours. SinCro at 300k costs ~15 GPU-days instead of ~25 at the reference 500k. The Phase D improvement budget
stays capped at 8 iterations or 25% of the total GPU budget, whichever comes first. If L = 200k, the stage 2 and
stage 3 pretraining rows shrink by a third; if L = 400k, they grow by a third.

## Mid-campaign review additions (2026-10-08)

| Item | Runs | Count | Est. GPU-hours | Tier |
|---|---|---|---|---|
| 1a | held-out-set ground truth (replayed states, 24 cameras, validation episodes), 8 tasks + replacement | 9 | ~1.5 | 1 |
| 1b/1c | full-split evaluations with oracle, Chamfer, near/trajectory retrieval and probes (~5-10 min each) | ~40 | ~5 | 1 |
| 3b | DrM + CNN, 300k agent steps, light evaluation, top-2 reserve tasks (next 2 if neither qualifies) | 2-4 | 1 each | 1 |
| 3c | collection (250 episodes), split, statistics, D2/D3 and held-out sets for the replacement task | 1 | ~2 | 1 |
| 2 | viewpoint screens (crop, synthetic near views, self-render) x 2 tasks, 100k steps on the 200k schedule | 6 | ~5 each, 30 | 1 (improvement budget) |
| 4 | gate screens (reference + 4a-4e, 6k steps) | 6 | ~0.5 each, 3 | 1 (improvement budget) |
| 2/4 | counted iterations combining winners (100k x 2 tasks + proxies) | <= 2 | ~12 each | 1 (improvement budget) |
| 5 | hammer proxies to 400k agent steps for every compared encoder (2 seeds) | ~10 encoders | ~+1 each (shared) | 1 |

Added cost ~80-90 GPU-hours, all inside the Phase D improvement budget (8 counted iterations or 25 % of the GPU
budget). Shelf-place's place in Stages 3/4 is taken by the replacement task at the same cost.

## User directives additions (2026-10-09)

Validation standard: 2 pretraining seeds per variant on hammer and pick-place at 100k of the 200k schedule
(`train.stop_step=100000`) + full-split evaluation; RL decisions use 6 hammer seeds at 400k agent steps. Measured
pretraining speed under the current sharing: 0.25-0.35 s/step, i.e. ~8-10 h of wall-clock per 100k run; host RAM, not
the GPUs, limits concurrency (each pretraining run is declared at `ram_gb` 30 until item 3 measures it; 5-7 fit
next to the other tenants).

| Item | Runs | Count | Est. GPU-hours | Tier |
|---|---|---|---|---|
| S1 | V1/V2 x 2 seeds + R seed 1, 2 tasks, 100k (R seed 0 reused) + full-split evaluations | 10 + 12 | ~85 | 1 (iteration 4) |
| 1 | D0-D4 x 2 seeds x 2 tasks, 100k + full-split evaluations | 20 + 20 | ~170 | 1 (directive budget) |
| 2 | M3D x 2 seeds x 2 tasks, 100k (M2D = item-1 winner, reused) + evaluations | 4 + 4 | ~35 | 1 |
| 2 | hammer RL proxies, M2D and M3D x 6 seeds, 400k agent steps, full evaluation protocol | 12 | ~8 (shared) | 1 |
| 1e | leave-one-out of redundant terms (candidates pre-registered after item 2), ~3-4 variants x 2 seeds x 2 tasks | <= 16 | <= 140 | 1 |
| length | A200 and A300 of the final configuration (2 tasks) + 6-seed hammer proxies at 100k/200k/300k and A200 | 4 + 24 | ~90 + 16 | 1 |
| 3 | loader benchmark (6 prefetch x worker settings, 2 passes) | 1 | ~1 | 1 |
| S4 | DrM + CNN pick-place seed 2000, 1M agent steps (a Stage 4 seed if code and protocol stay unchanged) | 1 | 3.5 | 1 |

Added cost ~550 GPU-hours (user authorisation of 2026-10-09 for items 1-2). At 5-7 concurrent pretraining runs the
critical path to `method-frozen-v1` is roughly S1 ~1 day, item 1 ~2 days, item 2 ~0.5-1 day, item 1e ~1.5 days and
the length study ~1.5 days. SinCro and ReViWo pretraining do not depend on our method and may fill spare host memory at
lower priority before the freeze (no baseline RL before the freeze).

## Disk

CNN replay: one 128x128 frame per state, 1M states -> 46 GB per run (sparse until written); deleted once the run is
complete and its final evaluation written. Frozen-encoder replay: <1 GB RAM snapshot per run. Snapshots: ~20 MB (CNN)
/ ~4 MB (frozen) every 10k steps. New runs are not launched below 300 GB free.
