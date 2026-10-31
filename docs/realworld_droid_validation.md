# Real-data validation of the real-world pipeline on a raw DROID episode (2026-10-31)

This validates items 5 (FoundationStereo + AllTracker, 3D lifting) and 6 (SNeRL DINO teacher, SAM 3 masks). It runs
the actual upstream models on genuine stereo pairs through this repository's own entry points:
`python -m s4d.realworld.cli run` with the `droid_raw_svo` adapter and `scripts/realworld_droid_diagnostics.py`.

**What it can and cannot establish.** DROID uses ZED 2 RGB stereo (external) and ZED Mini (wrist) cameras, on a
Franka with a Robotiq 2F-85.
- It validates the model integrations, the stereo/rectification geometry, units and conventions, timestamps, tracking
  and lifting on real images.
- It cannot establish the future RealSense stream configuration: infrared pairs, emitter, stream modes and the
  RealSense SDK intrinsics.
- It cannot establish the Panda–Robotiq mounting geometry of our own rig.
- One episode shows functional integration, not robustness.

## Data (read-only inputs; every output under `/home/ws/ws/validation-20261031/item5`)

| Input | Source | Size / hash |
|---|---|---|
| Raw episode `AUTOLab+0d4edc83+2023-10-21-19h-07m-04s` (the historical sample of `docs/droid_validation/`) | `https://storage.googleapis.com/gresearch/robotics/droid_raw/1.0.1/AUTOLab/success/2023-10-21/Sat_Oct_21_19:07:04_2023/` | 34.8 MB; `SHA256SUMS` beside the files |
| ZED factory calibration `SN22008760.conf`, `SN24400334.conf`, `SN18026681.conf` | `https://calib.stereolabs.com/?SN=<serial>` (the file the ZED SDK downloads) | 1.2–1.6 kB each; `SHA256SUMS` |
| DROID-3D extrinsics and intrinsics | HF dataset `jaibrdhn/droid_3d_extrinsics`, revision 69db9ccf9a51 | `extrinsics.jsonl` 100 MB, `camera_intrinsics.jsonl` 203 MB |
| PointWorld-DROID depth of the same episode (reference depth only) | `/home/ws/data/pointworld_droid_sample` (revision dd9aaeec) | sha256 c1c7e10c… |

The SVO files contain each ZED's raw side-by-side left|right HEVC stream, which is a genuine stereo pair from one
rig. The MP4 files are the SDK-rectified left views and serve as a reference. Decoding uses PyAV in an isolated
environment, so no ZED SDK is needed.

## Findings

### Format and timing
- SVO v1 layout: a header (`-11, width, height, serial, fps`), then `[uint64 ts_ns, uint32 type, uint32 size]` frame
  records of HEVC, interleaved with sensor blocks. 127 records per camera (key frames 0 and 120), each 2560×720.
- Each record timestamp equals `<serial>_frame_received` (ms) of `trajectory.h5`. All 127 frames map to trajectory
  steps 0–126; step 127 has no frame, which matches the 127-step RLDS episode.

### Factory calibration reading
The convention was decided against the SDK's own output. Eight hypotheses were tested (`rect_hypotheses_*.json`):

| Reading | ext1 L/R row error (median \|dy\|) | ext1 vs SDK MP4 (median \|d\|) |
|---|---|---|
| **R = Rodrigues(RX, CV, RZ) left→right, right centre (B, −TY, −TZ), `[*_DISTO]` rational model** | **0.41 px** | **0.32 px** |
| same, 5-coefficient `[*_CAM_HD]` model | 0.44 px | 1.44 px |
| same, centre (B, +TY, +TZ) | 0.36 px | 1.56 px |
| transposed rotation | 2.8 px | 1.5 px |

- Baselines equal DROID-3D's SDK baselines exactly: 120.058 mm, 120.155 mm and 63.061 mm.
- The wrist ZED Mini uses a 4-coefficient Kannala–Brandt model. Rows align at 0.7 px, but there is a 2.2–2.9 px
  horizontal offset to the SDK. This is not resolved; the wrist is not part of this fixed-rig run.

### The pipeline's own rectification (`diagnostics/rectify.json`)
- `geometry.stereo_rectify` + `rectify_map` + `remap` on the ingested real frames.
- R1/R2 equal OpenCV's `stereoRectify` to 2×10⁻⁶ °.
- With the SDK's rectified K (converted to the continuous convention, +0.5) and our R1, the rectified left view
  matches the SDK MP4:
  - ext1: dx 0.003 px, dy −0.055 px (median over 6501 SIFT matches);
  - ext2: dx +0.40 px, dy −0.05 px.
- With the pipeline's default K_rect, the left/right row residual is median |dy| 0.40 px (ext1) and 0.38 px (ext2).

<!-- results of the model stages are appended below as they are produced -->

## Models, environments and provenance

Every model runs in its own isolated uv environment under `/home/ws/ws/validation-20261031/envs` and is called
through the worker protocol. The environments use torch 2.10.0+cu129 because the GPUs are Blackwell (sm_120). The
upstream FoundationStereo pin, torch 2.4.1, has no sm_120 kernels; that is the main environment deviation.

| Model | Upstream commit | Weights (sha256) | Environment notes |
|---|---|---|---|
| FoundationStereo ViT-L `23-51-11` | NVlabs/FoundationStereo 6e88068 | official Google Drive file, 3,298,527,334 B (60e79bde…) | py3.11, torch 2.10. No xformers, so the vendored DINOv2 layers fall back to standard attention; flash-attn is unused because upstream calls SDPA. The `torch.hub` DINOv2 code is pre-populated at facebookresearch/dinov2 7764ea0f, since upstream would fetch the unpinned default branch. timm downloads `edgenext_small` pretrained weights at construction; the strict checkpoint load overwrites them. open3d needs libusb, supplied from `libusb-package` via `LD_LIBRARY_PATH`. |
| AllTracker | aharley/alltracker e755313 | HF `aharley/alltracker` rev c8ba3122 (ffd9ebcf…) | py3.12, torch 2.10, opencv-python-headless 4.10.0.84 (the headless build of the pinned version). torchvision `convnext_tiny` ImageNet weights are downloaded for initialisation only; the strict load overwrites them. |
| SAM 3 | facebookresearch/sam3 f1647e8 | gated HF `facebook/sam3` rev 3c879f39 `sam3.pt` (9999e234…) | py3.12, torch 2.10, numpy 1.26, setuptools < 81 (upstream imports `pkg_resources`). Run with `HF_HUB_OFFLINE=1`. |
| DINO ViT-S/8 (SNeRL teacher) | facebookresearch/dino 7c446df (`third_party/dino`) | `dino_deitsmall8_pretrain.pth` (55c8b267…, enforced) | The repository venv: the SNeRL consumer's own environment. |
| SVO decoder | PyAV 18.1.0 (libavcodec 62.28.102) | — | py3.11 media environment |

The SNeRL contract was checked in the pinned upstream (`third_party/snerl` d01533b, `nerf_pretrain/run_nerf.py`):
- `ViTExtractor('dino_vits8', 4)` at line 662;
- `(target − 0.45)/0.225`, then `extract_descriptors(…, 11, 'key', False).view(1, 31, 31, 384)`, then bilinear
  `F.interpolate` to (128, 128), at lines 712–715.

`s4d.baselines.snerl.model.dino_target` and the worker reproduce this exactly. The worker checks equality on the
first image of every call.

## Item 5: FoundationStereo through `stereo_depth` (`diagnostics/stereo.json`, `depth_*.png`, `cross_*.png`)

- **Run.** Both rigs, 127 frames each, left and right references, in 33.6 min on one MIG 1g.24gb slice. That is 16
  worker calls per rig, one per chunk and reference. Every record carries the model identity: commit 6e88068,
  checkpoint sha256 60e79bde…, `is_mock: false`.
- **Rectification and intrinsics.** The rectification ran inside the stage. The depth grid's K is the pipeline's
  K_rect: fx 533.17 / 533.93 px, the left raw principal point, continuous convention. Depth is metric camera-z in
  metres, with 0 marking invalid pixels.

| Check (every 10th frame) | ext1 | ext2 |
|---|---|---|
| valid fraction, left / right reference | 0.913 / 0.940 | 0.945 / 0.883 |
| median depth (m) | 0.868 | 0.721 |
| FoundationStereo − SIFT disparity at independent L/R matches: median, median \|·\| (px) | −0.024, 0.41 | −0.036, 0.49 |
| right-reference flip-and-swap vs left reference, \|d_R(x − d_L) − d_L\|: median px, fraction < 1 px | 0.32, 0.75 | 0.22, 0.86 |
| PointWorld-DROID full depth (320×180), signed median, relative median, fraction < 2 % | −1.4 mm, 0.61 %, 0.80 | −2.6 mm, 0.50 %, 0.91 |

**Cross-camera metric consistency (independent rigs, joined only by the DROID-3D extrinsics).** Each rig's left depth
is lifted with its pose, z-buffered into the other rig, and compared with that rig's own stereo depth.
- ext1→ext2: median signed −1.0 mm, median relative 0.75 %; 74.8 % of pixels below 2 %.
- ext2→ext1: median signed +3.6 mm, median relative 0.82 %.
- The tails come from occlusion edges and non-overlapping regions, which are included in the statistic.

This checks units, the baseline, rectified K and the w2c convention end to end.
- If DROID-3D's view matrix had been read as c2w, the lifted points would not land on the other camera's image at all.
- A baseline or fx error of x % would bias every comparison by x %.

The PointWorld depth is probably itself a stereo-network output, so it is a consistency reference, not ground truth.
