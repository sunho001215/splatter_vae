# Compute plan (Meta-World campaign)

Measured on GPU 4 / GPU 5 (RTX PRO 6000 Blackwell, 96 GB each), 384 CPU cores on a shared host. Historical
availability was ~140 GB RAM and 5.0 TB free disk; neither is a guaranteed campaign allocation. Current conservative
memory correction and pending acceptance are below. Workload numbers come from scheduler jobs in `runs/`.

## Hard campaign budgets (user update, 2026-10-10)

- **Local hard ceiling: 150,000,000,000 bytes (150 decimal GB), steady target 130,000,000,000 bytes.**
  This includes every campaign process, evaluation/test/export transient, HDF5/memmap file cache, loader shared memory,
  pinned memory and retained allocator growth. Charge the entire container `memory.current`, cross-checked against
  whole-cgroup/tree PSS; use the greater value. No reclaimable file cache is excluded. PSS already includes its mapped
  shmem/file components; do not add those components again. Container-wide accounting deliberately includes other
  work in this container rather than claiming unsupported exclusive attribution.
- User speed-up update at22:44 returns the independent **host-MemAvailable reserve to60GiB**, not128GiB; keep the local disk floor. Legacy queue `ram_gb` and host
  reserve fields are GiB (`2**30`); new hard limits/targets are explicit integer bytes, never 150 GiB masquerading as
  150 GB. Admission must reserve full active workload peaks plus non-job/cache/kernel/background charge, newly admitted
  peaks and a transient margin below the hard limit; the ten-minute host-available launch ramp remains additional.
- **Remote campaign ceiling: 1,000,000,000,000 bytes (1 decimal TB), provisional steady target 900 GB.** Keep its
  separate host reserve and the recomputed at-least-15%-free SSD floor, plus per-run future disk growth. Measure full
  owned-container cgroup charge without Docker working-set cache subtraction. Remote is the default destination for
  item 2, wide RL sweeps, Stage 4 CNN/SinCro/ReViWo and future pretraining batches once native/comparability gates pass.
  Review snapshot and fresh launch-time checks retain an additional outside-container upper bound:
  `max(0, (MemTotal - MemFree) * 1024 - owned_container_charge)`. This intentionally reserves all nonfree host RAM,
  including foreign usage, rather than assuming host-side transfers/cache/runtime are free. It can over-restrict
  admission; missing/invalid cgroups or host-memory fields hold new launches without erasing observed job state.
- All 20 active item-1 configurations were independently inspected: they **already use six workers and prefetch 2**.
  There is no remaining 4x8-to-2x6 saving to obtain by redundantly restarting these jobs. No numerical setting changed.
- Proposed immediate reduction: checkpoint-pause the complete hammer decision group (D0-D4, both seeds; ten runs),
  preserving every completed optimizer step and the original 100k stop on the 200k schedule. Approximately 9 GB PSS
  per run supports a whole-group pause rather than splitting comparison placement. Old checkpoints are not a safe
  substitute for a fresh boundary: no rollback/recomputation may be claimed as lossless. If measured release is still
  insufficient, additional checkpoint pauses are needed, not experiment cancellation or cache exclusions.
- **Standing authorization received 22:23:** lossless owned-session checkpoint pauses/resumes and bounded remote
  bootstrap explicitly approved. The initial controller `bv3z5vw1s` stopped after its first teardown wait; successor
  `bjhmh9q29` completed exit 0 with all ten pauses independently verified (nine at 40k, D2 seed 0 at 50k), never
  rolled back to old checkpoints. Earlier denials are historical. Actual measured headroom and complete live-PID
  coverage admitted only the bounded frozen-candidate native suite at 00:50 on 2026-10-11; no above-cap launch.
  Review-only admission/heartbeat/pause-replay code remains unaccepted and excluded from this seven-file candidate.
- Migration is allowed only after real cross-host step-0/short-run equivalence **and resumed-checkpoint same-next-step
  loss within numerical noise**, with identical data/config/protocol, preserved optimizer/RNG/scheduler/data position,
  and every variant/both seeds of a task moving together. Registry host-step segments and checkpoint hashes must be
  explicit. Intentional pauses are not infrastructure failures or completed experiments. Until then, paused work waits;
  none of the required experiments is removed, shortened, reseeded or numerically retuned.
- Admission and measured margins are not a kernel-enforced all-times cap; checkpoint pauses/native peak acceptance
  and ongoing footprint monitoring are still necessary. No cgroup/device/host/VPN settings were changed.

Historical hard-budget snapshot (2026-10-10 22:38): **215.09 GB** full container charge versus **184.43 GB** whole-cgroup
PSS, including **77.72 GB** Pss_Shmem; cgroup file **106.45 GB** includes **78.81 GB** shmem. This violates the required
150 GB ceiling; it is not an accepted operating allocation. `runs/setup/memory-ceiling-20261010-2240.json` records exact
bytes at observed_at 22:38, ten armed pause requests and zero verified pauses. Earlier 21:47, 22:06 and 22:10 snapshots remain unchanged
in `runs/setup/memory-ceiling-20261010.json` and `runs/setup/memory-ceiling-20261010-2206.json`. A temporary unavailable
live-process PSS sample failed closed; no partial PSS estimate was treated as complete. Six pure accounting and
thirteen admission/replay checks passed without native fixtures or worker launches; this is not full acceptance.
New heavy **local** launches remain held. User22:44speed-up removes localRAMand unfinishedreviewcode as dependencies
for remote-only suite/isolation/work from clean deployed200e5e6. Bounded local verification<=15GBmay use measured
headroom under150GB. Fresh equivalence is oneC/D0hammerseed0case for200steps and measured same-host remote-repeat
noise (unsmoothed total-lossRMSE), not twenty fixed-tolerance configurations. Only migrated runs require original
localcheckpointnext50step comparisons under the same criterion. IndependentStage4CNN/SinCro/ReViWowork starts
after the nonmigration gates; the whole hammergroup resumeslocallyunderbudgetor migrates after its checkpointgate,
whichever is faster. The superseded22:23protocol and before-result user decision are preserved in EXPERIMENT_LOG.

## Completed four-hour memory window (2026-10-11 00:13)

Original evidence: `runs/setup/memory-window-20261010-2015.jsonl` and
`runs/setup/memory-window-20261010-2015.summary.json`. Actual window: **2026-10-10 20:12:40 to
2026-10-11 00:12:59 +09:00**, 14,419.60 seconds, **102 samples**. The observer completed exit 0;
it is no longer running. The nominal wait is 120 seconds plus scanning time: observed timestamp gaps
were 109.41–161.94 seconds, median **142.59 seconds**, with a maximum scan duration of 41.94 seconds.

| Observed quantity | Window result | Interpretation |
|---|---|---|
| Full container memory.current maximum | **227,567,730,688 bytes (227.57 GB)** | All charged cache/kernel/background work included; no cache subtraction |
| Aggregate owned-session PSS maximum | **188,319,444,992 bytes (188.32 GB)** | A different sample from the full-charge maximum; not a simultaneous pair |
| Per-run sampled PSS-peak range | **9,146,406,912–10,773,317,632 bytes (9.15–10.77 GB)** | Largest: D4 seed 1 hammer; latest metric at that sample was step 30k, not independent phase identification |
| Cgroup shmem maximum | **82,727,145,472 bytes (82.73 GB)** | Already included in cgroup file charge and mapped PSS; never added again |
| Minimum host MemAvailable | **31,325,167,616 bytes (31.33 GB)** | Shared-host observation, not an allocation guarantee |

At the **22:26:29** full-charge peak, all twenty jobs were active: owned PSS was **182.67 GB**, cgroup
anon **109.95 GB**, file **114.58 GB**, including shmem **80.17 GB**. That sample had no observation errors.
The 44.90 GB difference between full charge and owned PSS is not exclusively attributed to any allocator or job.
One process exited during the 23:38:37 scan (PID 2969774, ProcessLookupError); preserve that limitation.
The final sample had fifteen active jobs, charge **175.81 GB**, owned PSS **138.85 GB**, and no observation errors.

This is **not a lifetime peak**, cannot resolve brief transients, and does not prove exclusive evaluation-phase
peaks. RSS/HWM double-count shared worker pages; PSS already includes mapped shmem/file pages; zero VmPin/Locked
does not establish absence of CUDA-registered pinned memory. Existing **13 GiB** pretraining declarations provide
about 30% margin over the largest sampled per-run PSS, but do not alone reserve the full container cache/background
charge. No additional workload is admitted by dividing 150 GB by those declarations, and complete-footprint/native
acceptance remains pending. No scientific configuration, seed, worker count or schedule changed.

By 00:30, **eight hammer pauses are independently verified**: seven at 40k and D2 seed 0 at 50k, preserving the
100k stop and 200k schedule. D1 seed 0 and D4 seed 0 remain armed under controller `bjhmh9q29`, PID 2961453;
ten pick-place trainers continue. Charge-only evidence at 00:30 is **150,773,706,752 bytes (150.77 GB)** in
`runs/setup/campaign-health-20261011-0030.json`; a preceding observation was 148.88 GB, so momentary crossing below
150 GB is not steady cap compliance. Main HOLD stays. The existing readiness waiter requires <=135 GB plus the
60 GiB host reserve and 15 GB suite reservation; fresh complete PSS, actual foreign-GPU checks, candidate fingerprints
and accepted scheduler admission still precede the full frozen-candidate suite. No duplicate observer is started.

## Measurements

| Workload | Configuration | Throughput | GPU memory | RAM | CPU | Source |
|---|---|---|---|---|---|---|
| Pretraining (ours) | batch 16, 6 cameras x 3 times, 8 loader workers | 0.22 s/step alone (0.06-0.10 s waiting for data); 0.25-0.35 s/step with 2-5 runs per GPU | 3.2 GB | ~6 GB + workers | loader-bound | `runs/pretrain/timing-pretrain-hammer` |
| Pretraining loader | one window (6+4 cameras, 3 times, motion for 6 pairs) | 76 ms/window/core; 16-window batch: 258 / 178 / 152 ms with 8 / 16 / 24 workers | - | - | 1 core/worker | loader benchmark (E-log G1) |
| Pretraining memory (item 3) | C configuration, batch 16, whole process tree | prefetch x workers 4x8: 11.3 GB peak PSS; 2x6 (default since 2026-10-09): 9.6 GB; ~0.8 GB per worker, prefetch depth <= 0.5 GB | - | `ram_gb` 13 (2x6) / 15 (4x8) = 1.25 x peak PSS | - | `runs/bench-loader-synth-hammer` |
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
- **Pretraining:** current default is 6 loader workers with prefetch factor 2 (item 3). Item 1 currently packs 10
  runs per GPU, with measured training steps around 0.72-0.94 s and both GPUs at 96% utilisation. This is operational
  packing, not a comparison of loss-variant efficiency; the final hard-pass timing comparison remains pending.
- Production admission in `experiments/queue.yaml`: per-GPU `max_jobs` 20 and `max_mem_gb` 90 (declared per job),
  60 GiB host-RAM reserve and a 10-minute RAM ramp window. The additional 150 decimal GB hard campaign budget is
  mandatory but its admission implementation is still review-only; HOLD blocks new launches. Item-1 training jobs
  declare 6 GiB GPU memory and 13 GiB RAM; those historical declarations still await complete-footprint acceptance.
  At 2026-10-10 17:16, the 20 process trees held 175 GB PSS (7.98-9.38 GB each) after their 10k evaluations; none was
  still evaluating at the snapshot, so the transient evaluation-time peak remains unmeasured.

## Shared-host memory correction (2026-10-10; window completed 2026-10-11)

The historic 13 GB declaration above is based on a short PSS benchmark, not a complete long-run footprint. It must
not justify further packing on this shared host. At approximately 20:10, host MemAvailable was 60.9 GB, swap was
almost exhausted, our container's `memory.current` was 213.7 GB, and its cgroup reported roughly 110.9 GB anonymous,
99.9 GB file (including 80.6 GB shmem), and 2.8 GB kernel memory. These components must not be added twice. The two
training HDF5 files alone had 18.8 GB resident page cache, shared across their runs rather than per-worker copies.
This is a snapshot, not a measured lifetime peak or exclusive attribution of all container memory to the campaign.

- The four-hour read-only window is complete; its measured extrema, actual cadence and limitations are recorded
  above, with both original artifacts retained. It sampled exact owned sessions, RSS/PSS components, Locked/VmPin/HWM,
  latest logged metrics, unique HDF5 residency without warming files, host memory, cgroup charge/stat/events and pressure.
  New heavy local launches remain held pending actual hard-budget headroom, not pending this completed observer.
- PSS already contains mapped shared memory and file pages. Do not add those again; shared dataset/replay cache is
  reserved once at host level (or explicitly apportioned), and kernel/allocator retention receives a measured margin.
  RSS/HWM are diagnostic and double-count shared worker pages. Zero VmPin/Locked is not proof that CUDA-pinned pages
  are absent; driver-registered pinned memory is not completely visible in proc. Bound this uncertainty conservatively
  and do not present allocator-specific attribution or a lifetime peak that the observer cannot measure.
- Sample across additional scheduled evaluations and post-evaluation training, inspect retained anonymous growth,
  then set workload `ram_gb` from observed peaks plus margin. Extend observation if footprint continues growing or
  an evaluation is not covered, but do not postpone the new hard-cap reduction for the observer. The latest user
  directive authorizes checkpoint-bound validated 2x6 transitions and intentional pauses; speculative pinning/allocator
  or numerical changes remain prohibited. All current item-1 jobs already have the validated 2x6 configuration.
- The historical 128 GiB reserve accepted at 21:03 was explicitly superseded by the user's 22:44 directive:
  operational main and review queues now use **60 GiB**, retaining the ten-minute ramp. This independent host reserve
  does not replace the 150 decimal GB campaign ceiling. Final workload declarations still require complete-footprint
  peaks and margins; short verification <=15 GB may use measured current headroom below the cap. Lossless checkpoint
  pauses/resumes and bounded cross-host bootstrap are explicitly authorized; earlier permission denials are historical.
- The validated six-worker/prefetch-two setting remains the future default. Fewer workers, capped/reused pinning and
  allocator arena limits require independent validation before adoption; do not apply speculative fixes to running
  jobs. Admission will leave explicit shared-tenant headroom, not merely avoid our own OOM.
- After real remote native/isolation/data/equivalence acceptance, prefer its larger RAM for many-worker pretraining,
  CNN disk replay and wide seed sweeps; all comparison arms stay on one host or first pass cross-host equivalence.

## Planned runs, cost and priority tier

Tiers (finish in order if compute runs short): 1 = Ours vs CNN on all 8 tasks; 2 = SinCro and ReViWo on all 8 tasks;
3 = RL ablations; 4 = remaining representation ablations, K sweep, extra seeds.

**Pretraining lengths (decision, 2026-10-13).** Ours: **L = 200k**, selected by the unchanged registered
C + D0 + M3D A200/A300 rule (RESULTS; `runs/setup/length-verification-20261013.json`), fixed for all eight tasks and
every ours ablation. The historical 300k base default is overridden explicitly for all final runs. Reuse the two
verified A200 development encoders; pretrain the remaining six tasks at 200k. SinCro: exactly 300k steps on every task.
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

**Prospective seed policy, user decision2026-10-10 23:00:** item1anditem2retain two pretraining seeds per variant
and development task,100kof the original200kschedule (`train.stop_step=100000`) and full-split evaluation. Item2
also reports per-seed collapse and actual dynamic share at10kand100k. Comparisons after item2that are not yet
registered/launched (item1eleave-one-out, final length study, later screens) default to **one pretraining seed per
variant/task (seed0)**, with exact margins recorded before launch from max(existing reference seed-pair spread,
existing floors). If the adopted configuration still collapses in any item2seed, report those numbers before any
such launch and let the user decide seed count again. Existing registered comparisons are not changed; RL
proxies retain all **six seeds**,400kagentstepsand the existing RL rule. Final RL seeds and baseline protocols are
unchanged. New downstream pretraining budgets below assume the no-residual-collapse condition is satisfied. Measured
pretraining speed under the current sharing: 0.25-0.35 s/step, i.e. ~8-10 h of wall-clock per 100k run; host RAM, not
the GPUs, limits concurrency (item 3: a pretraining run needs ~10-12 GB PSS, declared as `ram_gb` 13 at the new
2x6 loader default or 15 at 4x8; RSS (~25 GB) double-counts shared worker pages; ~8-12 runs fit next to the other
tenants).

| Item | Runs | Count | Est. GPU-hours | Tier |
|---|---|---|---|---|
| S1 | V1/V2 x 2 seeds + R seed 1, 2 tasks, 100k (R seed 0 reused) + full-split evaluations | 10 + 12 | ~85 | 1 (iteration 4) |
| 1 | D0-D4 x 2 seeds x 2 tasks, 100k + full-split evaluations | 20 + 20 | ~170 | 1 (directive budget) |
| 2 | M3D x 2 seeds x 2 tasks, 100k (M2D = item-1 winner, reused) + evaluations | 4 + 4 | ~35 | 1 |
| 2 | hammer RL proxies, M2D and M3D x 6 seeds, 400k agent steps, full evaluation protocol | 12 | ~8 (shared) | 1 |
| 1e | new leave-one-out comparisons after item2, ~3-4variants x1pretrainingseed x2tasks, subject to collapse exception | <=8 | <=70 | 1 |
| length | new A200/A300finalconfiguration comparisons x1pretrainingseed x2tasks, plus unchanged6-seedhammerproxies at100k/200k/300kandA200 | 4 +24 | ~90 +16 | 1 |
| 3 | loader benchmark (6 prefetch x worker settings, 2 passes) | 1 | ~1 | 1 |
| S4 | DrM + CNN pick-place seed 2000, 1M agent steps (a Stage 4 seed if code and protocol stay unchanged) | 1 | 3.5 | 1 |

Added cost ~550 GPU-hours (user authorisation of 2026-10-09 for items 1-2). At 5-7 concurrent pretraining runs the
critical path to `method-frozen-v1` is roughly S1 ~1 day, item 1 ~2 days, item 2 ~0.5-1 day, item 1e ~1.5 days and
the length study ~1.5 days. The user's 2026-10-10 speed-up directive permits independent Stage 4 CNN DrM
(8 tasks x seeds 2000–2002) remotely before method freeze once the nonmigration v2 gates pass. SinCro and ReViWo
pretraining may start then on already transferred tasks, with other tasks transferred as SSD admission allows.
Other baseline RL ordering remains unchanged; item 2 and its RL proxies still wait for the item-1 decision.

## Disk

CNN replay: one 128x128 frame per state, 1M states -> 46 GB per run (sparse until written); deleted once the run is
complete and its final evaluation written. Frozen-encoder replay: <1 GB RAM snapshot per run. Snapshots: ~20 MB (CNN)
/ ~4 MB (frozen) every 10k steps. New local runs are not launched below 300 GB free.

## Second host (authorized 2026-10-10; admission remains held until verification)

| Host | Eligible GPUs | Host RAM | GPU limits | Disk admission |
|---|---|---|---|---|
| Main container | original GPU 4/5 UUIDs only | required 150 GB campaign hard ceiling / 130 GB steady target; deployed 60 GiB host reserve, 10-minute ramp; new launches held, cap code review-only | 20 slots / 90 GB per GPU are ceilings, not packing targets | 300 GB minimum free plus future-growth reservations |
| Remote Docker | four explicit remote UUIDs in `configs/hosts/remote.yaml` | required campaign cap 1 decimal TB / provisional 900 GB steady target; approximately 1.5 TB total, 120 GiB reserve, 10-minute ramp; admission disabled | initial ceilings 20 slots / 90 GiB per GPU; exclude foreign processes | 15% total floor, initially 1.129 TB, plus future-growth reservations |

Remote capacity is not currently accepted campaign capacity: immutable runtime, full native suite, all four
one-UUID CUDA/EGL checks, data checksums and cross-host equivalence must pass first. The main container's new-CUDA
outage does not justify restarting its 20 healthy item-1 runs. At 18:48 those runs were at 17.4k-24.35k; the watcher
was alive and the scheduler remained dead/held. This is a progress snapshot, not a new timing estimate.

After acceptance, keep both comparison arms/all seeds on one host wherever possible. Under the superseding
2026-10-10 speed-up directive, first independent remote work is Stage 4 CNN DrM (8 tasks x seeds 2000–2002) and
SinCro/ReViWo pretraining on already transferred tasks, after the genuine nonmigration v2 gates. The complete paused
hammer group may move only after its 50-update checkpoint-continuation gates, or resume locally once the entire group
fits under budget. Item-2 paired pretraining and both six-seed hammer RL arms still wait for the item-1 decision;
leave-one-out, length selection, freeze and method-dependent Stage 3 work retain their order. The extra capacity does
not expand the experiment/iteration budget, change numerical protocols or allow nondevelopment RL to guide decisions.

Keep the validated 6-worker/prefetch-2 loader and original batch size. Historical 13 GB RAM declarations are
provisional and must be replaced by complete-footprint long-window peaks plus margin before new heavy admission.
Increase packing only from accepted footprint, GPU memory and steady throughput, respecting remote CPU/data I/O,
shared-tenant headroom and the launch ramp.
No remote throughput has been measured yet; local timings must not be relabeled as remote evidence. Keep the two
1M-step CNN runs/GPU ceiling until measured remote contention justifies a scheduling change. Reserve 46 GB replay
per admitted CNN run, plus snapshots and data/image transfer storage; 8 concurrent CNN runs reserve at least 368 GB
before other growth. Completed literal replay directories alone are eligible for cleanup after final evaluation.

Remote image native payloads come from audited installed production binaries, not cached wheels with different
hashes. Driver versions differ (main 580.178.04, remote 580.95.05); CUDA/EGL equivalence is a required gate. See
`docs/REMOTE.md` for identity, transfers, isolation and outage mechanics. Queue/registry and all decision results
remain on the main server.
