# TrackEverything single-view stage

`s4d.realworld.trackeverything_stage.track_everything` is an opt-in replacement for
both `pipeline.track2d` and `pipeline.lift3d`. It writes their ordinary stage names,
per-camera filenames, keys, point IDs, query frames, and original colour-grid pixel
coordinates. `track_motion`, `export` (either track source), and `viewer_data` use
the results without changes. Use a separate output root when retaining an
AllTracker comparison; the two stages occupy the same output locations.

The prerequisites in that root are normalized ingestion, our metric depth stage,
and accepted camera calibrations. RGB views with metric rig geometry participate;
other views get explicit skip records. There is no automatic switch in the
existing pipeline or server campaign.

Example wiring (recipe only; no model code was executed during implementation):

```python
from s4d.realworld.runners import LocalRunner
from s4d.realworld.trackeverything_stage import track_everything
from s4d.realworld.workers import WorkerSpec

spec = WorkerSpec(
    "trackeverything",
    python="/opt/envs/trackeverything/bin/python",
    repo_dir="/opt/upstream/trackeverything",  # must have this package directory name
    expected_commit="dfb29b505cf2ea954f8697fb612c2429668dd3a0",
    checkpoint="/ram/weights/trackeverything/model.pth",  # existing local file, never a hub ID
)
records = track_everything(
    root, episode_id, spec, LocalRunner(workdir), depth_stage="sensor_depth",
    window=16, step=8, grid_stride=16, model_hw=(384, 512), num_iters=3,
)
```

The same runner interface supports a persistent remote worker: one request is one
camera clip, not one query window. Workers cache the model and checkpoint hash per
process/session. `LocalRunner` uses a fresh subprocess per request, like the other
workers, so it reloads weights between clips; use a persistent runner for real runs.
The worker validates the local checkpoint before any upstream import. Only the
single-view `trackeverything.demo_hf` API is imported; the package parent goes on
`sys.path` as required by upstream README.md:58-59.

Conventions, verified against TrackEverything commit
`dfb29b505cf2ea954f8697fb612c2429668dd3a0`:

| Decision | Adapter location | Upstream reference |
| --- | --- | --- |
| Mandatory metric camera-z depth, continuous K, OpenCV world-to-camera pose; geometry estimation bypassed | `workers/trackeverything_worker.py:136` | `demo_hf.py:752-800` |
| Select `geometry_backend="pi3"` for cam0/scale preprocessing; estimator stays lazy | `workers/trackeverything_worker.py:105` | `demo_hf.py:488-508`, `626-641` |
| RGB Lanczos and depth exact nearest; 480x640 to 384x512 scales both first K rows by 0.8 | `trackeverything_stage.py:70` | `utils/common_utils.py:214-231` uses arange+0.5; `demo_hf.py:583-585` is the scaling we bypass |
| Original query-grid centres `(col+.5,row+.5)` use the same continuous resize affine, without snapping | `trackeverything_stage.py:183`, `198` | `demo_hf.py:820-840` preserves supplied query coordinates with input resolution enabled |
| Undo scale first, then pose: `X_base = inv(E0) [norm_factor * X_pred; 1]`, with original metric E0 | `workers/trackeverything_worker.py:55` | `demo_hf.py:190-192`, `626-641` |
| Mask frames before the exact query birth, including earlier frames in its birth window | `workers/trackeverything_worker.py:74` | `models/track_everything.py:526-532`, `1385-1401` |
| Visibility is sigmoid(logit), confidence is 1; OUT precedes OCCLUDED; 3D positions are NaN unless VALID | `trackeverything_stage.py:89` | `models/track_everything.py:477-479`, `1390-1401`; repository `tracking.STATUS` |
| Temporary forward hook obtains dynamic logits without upstream edits, then removes itself even on failure | `workers/trackeverything_worker.py:112` | `training/datatypes.py:320`, `models/track_everything.py:1403-1413`, `1578-1618`; `demo_hf.py:896-900` omits them |

Queries use exactly `pipeline.chunk_queries(n, window, step)` on each camera's
full timeline and `query_grid(original_hw, grid_stride, grid_stride//2)`. For the
640x480 view these are the original AllTracker query pixels. They generally become
subpixel coordinates on the model grid; replacing them with a new integer grid
would change the queried rays. We preserve the original rays.

A pause is any interval greater than 1.5 times that camera's median period. Clips
never cross pauses or cameras. Segments up to 64 frames use one clip. Longer
segments overlap enough to contain each entire query window in exactly one clip;
the first query after a pause can be off the segment boundary. Query identities
stay global. A window crossing a pause retains its original frame list, with all
post-pause entries NOT_TRACKED. `window` must be at most 64.

Each clip is saved through `ChunkStore.put` (arrays plus hash-checked manifest,
fsync) before `runner.ack`. The cache survives either stage failing and is cleaned
only after both ordinary `run_stage` swaps succeed. Stages are individually
atomic; the pair is not a single atomic transaction. After interruption, rerun
`track_everything` to finish the pair before downstream processing. Config,
upstream records, WorkerSpec, and explicit mock opt-in participate in fingerprints.

The worker returns optional `dyn_logit` in the clip response and records its
availability. Ordinary track chunks intentionally retain exactly the existing
keys; no dynamic field is added to their schema. The stage consumes visibility,
not dynamic classification, and discards clip caches on successful completion.

Validation uses `synthetic.py` ray-cast geometry exposed as RGB-D, a file-protocol
oracle returning normalized cam0 predictions, and a fake torch module to check
the hook. It does not execute any upstream model code. It checks metric recovery,
births, visibility/status, grid/resize rays, pause boundaries, bounded overlapping
clips, durable saves, resuming either stage, and unchanged motion/export/viewer
consumers. The existing pipeline suite is also run.

Open validation items:

- Real checkpoint inference and target-server memory/runtime have not been tested.
  The environment file is an inference recipe stub, not a validated transitive
  lock. `setup_envs.sh` documents torch 2.8.0+cu129, Python 3.10, the pt28cu129
  torch_scatter wheel, and a matching-ABI local wheel fallback if sm_120 fails.
  No setup, downloads, or server actions were performed.
- Upstream still filters supplied depth (`demo_hf.py:618-624`,
  `datasets/data_ops.py:1229-1252`). Its sparse query depth lookup rounds continuous
  x/y to integer indices and substitutes neighbouring mean depth or normalized
  depth 1 when zero (`demo_hf.py:223-232`). We preserve the required public API
  and continuous rays. Whether to reject queries born on invalid supplied depth
  rather than accept that upstream fallback remains a policy choice; the model's
  finite prediction/visibility determines the present output status.
