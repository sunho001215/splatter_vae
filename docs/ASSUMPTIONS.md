# Assumptions and uncertainties

Updated 2026-10-07. Status is **confirmed**, **refuted**, or **open**. Confirmation is restricted to the stated evidence, not the unfinished research project. Acceptance thresholds remain unchanged. See [RESULTS.md](RESULTS.md) for criterion-by-criterion outcomes.

## Environment and safety

| ID | Assumption | Final status and evidence |
|---|---|---|
| E1 | Workspace is `/home/ws`, with the new repo under `ws/splatter4d` and the two old repos read-only. | **Confirmed** paths. Reference preservation is not fully confirmed: a baseline untracked file disappeared. See S1. |
| E2 | Only the two specified Blackwell GPUs may be used, selected by exact UUID. | **Confirmed** inventory and CUDA/EGL isolation in `gpu_isolation.json`. GPU 4 is `GPU-76871c16-ff1d-f7b1-cdf5-fabbaf9df8ce`; GPU 5 is `GPU-d09f0338-71b9-d915-3c7f-e99754a3b639`. MuJoCo exposes exactly one UUID. GPU 6 is prohibited. |
| E3 | Python 3.10, uv, torch 2.10.0+cu129, gsplat, fused-SSIM, MuJoCo and Meta-World can provide a standalone environment. | Runtime versions **confirmed**, standalone compatibility **open**. Lock resolves without native builds. Current venv reads existing reference packages through `reference_runtime.pth`; it is not standalone. Native gsplat fails. |
| E4 | `data/metaworld` was empty at setup; new data uses `splatter4d_v1`. | **Confirmed** initial state. It now contains the eight pilots and earlier hammer/pick-place workspace statistics. No full task dataset or split manifests exist. |
| E5 | The obsolete Stage-0 cache and original DROID are distinct, nonsymlink roots; the obsolete cache can be safely deleted. | Paths **confirmed** by realpath/lstat. Safe deletion **open/blocked**. Historical cache size was approximately 662 GB, not a new exact-byte measurement. The cache is intact. Recursive open-file tools are unavailable and installation/deletion-script changes were denied. |
| E6 | Disk capacity is sufficient for pilot data and compact sample downloads. | **Confirmed** for completed work, approximately 3.6 TB free. Full-data capacity/throughput is **open**. `data_inventory.json` labels extrapolation as an estimate. |
| E7 | GitHub authentication and HTTPS access are available. | **Confirmed** by authenticated clone and gh status. Repository delivery is recorded in RESULTS.md. No old repository history is included in the orphan branch. |
| E8 | Hugging Face and official DINOv2 downloads are reachable. | **Confirmed** real files and hashes. No downloaded code or teacher generation was executed. |
| S1 | Both reference repositories remain byte-for-byte and git-status unchanged. | **Open; git-status equality refuted** for `hierarchical_splatter`. Both HEADs and the DROID reference status are unchanged. The baseline untracked `download_pointworld_all.py` is now absent. Cause is unknown. No restoration or reference write was attempted. No pre-session exhaustive filesystem snapshot exists. |
| S2 | Native-extension permission denials may be bypassed by fallback JIT or another tool. | **Refuted**. Production renderer fails before gsplat initialization/JIT. No denied build, tool installation, shared-data write, or deletion is retried through an alternate route. |
| S3 | A previously passing suite is sufficient after source edits. | **Refuted**. Long collection and training require a full passing suite with identical source/config fingerprints. Native errors remain errors, not skips. |

## Meta-World and camera geometry

| ID | Assumption | Final status and evidence |
|---|---|---|
| M1 | Installed Meta-World uses `<task>-v3` and `Meta-World/MT1`. | **Confirmed**, Meta-World 3.0.0. All requested tasks exist; no replacements. |
| M2 | Observation is 39-dimensional, with hand xyz, gripper, object blocks, previous state, and goal. | **Confirmed** installed implementation and collected fields. Probes use hand xyz and first-object xyz. Hand velocity divides finite displacement by recorded stride times `dt_seconds`. |
| M3 | Scripted policies are available through `ENV_POLICY_MAP`. | **Confirmed** actual noise-free runs on all eight tasks. |
| M4 | MuJoCo depth is metric camera-z, not ray length. | **Confirmed** installed renderer and measured multi-view lifting checks. |
| M5 | Segmentation contains geom id/type and maps to rigid body ids. | **Confirmed** stored poses/body-id fields and synthetic exact-motion tests. Segmentation is used only inside target construction, never as a foreground loss mask. |
| M6 | The free-camera rig becomes OpenCV after the OpenGL axis conversion and shares robot-base coordinates. | **Confirmed** ten-camera transform check, maximum position residual about 3e-8 m. Collection explicitly asserts identity robot-base body. |
| M7 | Continuous array centres `(column+.5,row+.5)` and principal point `(W/2,H/2)` are appropriate for MuJoCo. | **Confirmed for pilots** by unchanged D2/D3 median gates on all eight tasks. No convention change was needed. |
| M8 | The specified 250-episode collection mixture works and supplies enough motion. | Task success/visibility **confirmed**, full-mixture coverage **open**. Twenty noise-free episodes per task passed; only five shuffled-mixture pilot episodes per task were saved. Visibility was sampled every ten simulation steps. |
| M9 | Per-body poses yield consistent rigid 3D motion and exact zero world-body motion. | **Confirmed for pilots**: all D2 medians below 3 mm; 941,478 tested world-body pixels have exactly zero displacement. Full-dataset verification remains **open**. |
| M10 | Pilot data may stand in for the requested full dataset and saved splits. | **Refuted**. Full collection is gated by failing native tests. New six-task statistics and all pilot split writes were denied. Existing hammer/pick-place statistics were not regenerated. |
| M11 | Sixty-four distinct retrieval episodes can be selected from the validation split alone. | **Refuted** for the prescribed 96/4 split of 250 episodes. Evaluator uses all task episodes, including training episodes, for this distinct-episode cross-camera diagnostic. This is not unseen-episode retrieval. Fewer than 64 episodes reports the protocol unavailable. Probes still fit training episodes and evaluate validation episodes. |

## Gaussian rendering and losses

| ID | Assumption | Final status and evidence |
|---|---|---|
| G1 | Actual gsplat RGB+ED returns alpha-normalized expected camera-z depth. | **Open** native acceptance. Wrapper semantics are implemented and tested synthetically. The native opaque-Gaussian test errors at extension import. |
| G2 | Actual gsplat arbitrary-feature splats preserve constant world displacements and intended gradient detachment. | **Open** native acceptance. Installed API/source and synthetic wrapper tests support implementation; CUDA tests cannot execute. |
| G3 | Networks use bf16 while rendering/losses use FP32, and only intended tensors receive geometry/motion gradients. | **Confirmed** implementation and synthetic full-loss backward tests. Actual rendered training behavior is **open**. Hard depth updates centers only; motion render geometry and opacity are detached. |
| G4 | Fused SSIM provides a per-pixel map and D-SSIM is `(1-SSIM)/2`. | **Confirmed** actual installed CUDA extension: perfect-image zero loss and non-perfect finite nonzero gradients pass. |
| G5 | Dynamic alpha share may be measured only at the first time, and an empty motion region has perfect EPE. | **Refuted**. Alpha share aggregates all three times on score >0.5. Empty regions produce unavailable/NaN metrics with explicit support counts, not a passing zero. |
| G6 | Research representation can use averaged slots instead of flattened slots. | **Refuted** for policy/probe/retrieval state. Those flatten all K slots. Only the contrastive invariant head averages slots, as specified. |

## PointWorld-DROID and robot conventions

| ID | Assumption | Final status and evidence |
|---|---|---|
| P1 | PointWorld has publicly downloadable flat episode paths with a permissive code-style license. | Public real release **confirmed**; flat download layout and Apache dataset license **refuted**. Files are in streamed packaged archives. NVIDIA dataset license restricts derivatives to non-commercial use. Complete license is in `pointworld_license/LICENSE.pdf`. |
| P2 | PointWorld extrinsics are camera-to-base. | **Refuted**. They are OpenCV world-to-camera. Robot-base world coordinates are **confirmed for the selected sample** by independently matched raw end-effector state and depth/gripper projections. |
| P3 | Canonical indices are every second raw step, with raw `(t,t+6,t+12)` corresponding to canonical `(i,i+3,i+6)`. | **Confirmed for sample**: 127 raw steps, 64 canonical steps, and all 77 stored clip states verified. Frame timing is not exactly 15 Hz. Depth uses measured nearest timestamps. |
| P4 | Original RLDS can be independently matched by both metadata paths. | **Confirmed** sample 1/1 at ordinal 21512. An old index supplied only a path/ordinal hint. Raw original data was reread and matched; old Stage-0 teachers were not used. This is not a corpus match rate. |
| P5 | Scene flows are displacements and track ids can cross clip boundaries. | **Refuted**. They are tracked world positions. Differences are within one half-open numerical clip; ids are clip-local. Wrist camera is excluded. |
| P6 | Native and resized cache intrinsics use the same centre convention. | **Refuted**. Native intrinsics use integer centres. Conversion adds .5 to principal points before per-axis resize into continuous-centre cache intrinsics. Wrong initial cache is preserved under `coordinate_v1_do_not_train`, never used for training. |
| P7 | Track projection into the other camera is sufficient dense depth-reprojection acceptance. | **Refuted**. New dense validation lifts every valid source-depth pixel, transforms to world/target camera, and nearest-z splats. Selected median 1.5513%; unfiltered 2.2385%; both pass 3%. Visibility uses target teacher depth, not an independent oracle. |
| P8 | Sample validation is cross-episode generalization. | **Refuted**. Thirty training windows and five disjoint-timeline validation windows come from one episode. No held-out static cameras are fabricated. |
| R1 | Raw EE pose is xyz in meters plus extrinsic XYZ Euler radians. | **Confirmed** independent xyz/quaternion comparison. Quaternion order is xyzw. Thirty-two approximate palm/finger points are generated from raw state, not substituted PointWorld poses. |
| R2 | Raw gripper closure is 0=open, 1=closed with 85 mm stroke. | **Confirmed** upstream convention, independent raw comparison and held-out depth fit. PointWorld angle equals raw normalized closure times .725. Opening is local EE y; approach is local +z. Training-only constant offset is -19 mm. |
| P9 | Loading DINOv2 proves learned representation quality. | **Refuted**. Official pretrained weights and bitwise export/reload were verified on real corrected RGB. New state/temporal/projection parameters are untrained. No rendered overfit ran. |
| P10 | Short overfits should silently shorten temporal/depth warmups. | **Refuted**. Original 20k temporal ramp and 5k depth warmup remain. DROID 2k overfit only reaches temporal ramp .1 and stays in depth mode none. M2 3k Meta-World likewise retains the original ramp. These may make the requested overfit gates difficult; no criteria are relaxed. |

## Compute budget, deviations, and open work

| ID | Assumption | Final status and evidence |
|---|---|---|
| C1 | Full Meta-World runs fit comfortably and reach at least four iterations per second, enabling GPU sharing. | **Open**. No 500-step timing or training ran. Do not infer capacity from hardware size or encoder-only smokes. |
| C2 | CPU/Gloo distributed equality proves NCCL rendered DROID equality. | **Refuted**. Gloo verifies global InfoNCE loss/gradients, RNG restoration and evaluation synchronization only. Actual NCCL 50-step training and rendered step-0 agreement remain blocked. The comparison protocol uses deterministic FP32/no-mask/no-dropout settings, not ordinary bf16 stochastic training. |

- Phase 0 sequencing was not followed initially. Source scaffolding was written while shell safety-classifier failures blocked setup. Clone/orphan setup was later restored without overwriting work. This deviation is not retroactively reported as compliance.
- First obtainable reference git baseline was recorded on local 2026-10-07. Preservation before that baseline is unverified. The later missing reference file is explicitly reported, without assigning a cause.
- The copied MuJoCo helper remains byte-identical. It contains its own `/tmp` fallback paths. Verified direct EGL discovery avoids relying on an unverified fallback; project-created caches stay in the new repository.
- Three diagnosed already-compiled gsplat alternatives failed with different undefined PyTorch symbols. No fourth compatibility workaround or source/JIT build was pursued. Native tests remain errors.
- New shared Meta-World statistics/split writes were denied before execution. Deletion safety-tool installation and a destructive-script rewrite were also denied. These permissions are not inferred from agent reports or path validation.
- The old adapter filenames differ from the suggested skeleton only in naming. Optional multi-task training was deferred instead of publishing an unsupported config. No RL agents, unrelated baselines, or full DROID pretraining were implemented or run.
