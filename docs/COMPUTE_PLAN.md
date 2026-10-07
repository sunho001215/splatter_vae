# Compute plan (Meta-World campaign)

Measured on GPU 4 / GPU 5 (RTX PRO 6000 Blackwell, 96 GB each), 384 CPU cores, ~140 GB RAM available to this
campaign (the host is shared), 5.0 TB free disk. All numbers from scheduler jobs in `runs/`.

## Measurements

| Workload | Configuration | Throughput | GPU memory | RAM | CPU | Source |
|---|---|---|---|---|---|---|
| Pretraining (ours) | batch 16, 6 cameras x 3 times, 8 loader workers | 0.22 s/step (0.06-0.10 s waiting for data) | 3.2 GB | ~6 GB + workers | loader-bound | `runs/pretrain/timing-pretrain-hammer` |
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

| Stage | Runs | Count | Est. GPU-hours (each) | Est. GPU-hours (total) | Tier |
|---|---|---|---|---|---|
| 0 | DrM + CNN sanity (hammer, shelf-place), seed 2000 | 2 | 3.5 | 7 | 1 |
| 1 | pretraining, ours, hammer and pick-place, 200k steps | 2 | 8-12 | 24 | 1 |
| 1 | DrM + ours, hammer and pick-place, seeds 1000-1002 | 6 | 1 (shared) | 6 | 1 |
| 1 | improvement iterations (<= 8): pretraining 100k + RL proxy 2 seeds x 2 tasks | <= 8 x (2 + 4) | 5 + 4 x 1 | <= 72 | 1 |
| 2 | representation ablations, hammer and pick-place (no coverage, single group, no motion, T=1, mis-scaled x2) at 200k | 12 | 8-12 | 120 | 3 (tier 4 if short) |
| 2 | RL ablation encoders not shared with the list above (no InfoNCE) | 2 | 8-12 | 20 | 3 |
| 2 | DrM RL ablations (T=1, no motion, no InfoNCE, single group) x 2 tasks x 3 seeds | 24 | 1 (shared) | 24 | 3 |
| 3 | pretraining, frozen method, 8 tasks (the 2 development encoders are retrained only if the method changed) | 6-8 | 8-12 | 80 | 1 |
| 3 | DrM + ours, 8 tasks x seeds 2000-2002 | 24 | 1 (shared) | 24 | 1 |
| 4 | DrM + CNN, 8 tasks x 3 seeds (Stage 0 runs reused only if code and protocol are unchanged) | 22-24 | 3.5 | 84 | 1 |
| 4 | SinCro pretraining, 8 tasks (original hyperparameters; NeRF-based, cost measured before launch) | 8 | to measure | to measure | 2 |
| 4 | ReViWo pretraining, 8 tasks (100k steps) | 8 | to measure | to measure | 2 |
| 4 | DrM + SinCro / ReViWo, 8 tasks x 3 seeds | 48 | 1 (shared) | 48 | 2 |
| 5 | evaluation companions, probes, figures | all | small | ~20 | - |

Estimated total excluding SinCro/ReViWo pretraining: ~530 GPU-hours, i.e. ~11 days on two GPUs at full packing;
tier 1 alone is ~300 GPU-hours. The Phase D improvement budget is capped at 8 iterations or 25% of the total GPU
budget (~130 GPU-hours), whichever comes first. Frozen-encoder RL runs are packed alongside pretraining, so their
GPU-hours overlap rather than add.

## Disk

CNN replay: one 128x128 frame per state, 1M states -> 46 GB per run (sparse until written); deleted once the run is
complete and its final evaluation written. Frozen-encoder replay: <1 GB RAM snapshot per run. Snapshots: ~20 MB (CNN)
/ ~4 MB (frozen) every 10k steps. New runs are not launched below 300 GB free.
