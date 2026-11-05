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
camera's query window, with exactly the frames sent to AllTracker. Workers cache the model and checkpoint hash per
process/session. `LocalRunner` uses a fresh subprocess per request, like the other
workers, so it reloads weights between clips; use a persistent runner for real runs.
The worker validates the local checkpoint before any upstream import. Only the
single-view `trackeverything.demo_hf` API is imported; the package parent goes on
`sys.path` as required by upstream README.md:58-59.

Conventions, verified against TrackEverything commit
`dfb29b505cf2ea954f8697fb612c2429668dd3a0`:

| Decision | Adapter location | Upstream reference |
| --- | --- | --- |
| Mandatory metric camera-z depth, continuous K, OpenCV world-to-camera pose; geometry estimation bypassed | `workers/trackeverything_worker.py:handle` | `demo_hf.py:752-800` |
| Select `geometry_backend="pi3"` for cam0/scale preprocessing; estimator stays lazy | `workers/trackeverything_worker.py:_model` | `demo_hf.py:488-508`, `626-641` |
| RGB Lanczos and depth exact nearest; 480x640 to 384x512 scales both first K rows by 0.8 | `trackeverything_stage.py:resize_inputs` | `utils/common_utils.py:214-231` uses arange+0.5; `demo_hf.py:583-585` is the scaling we bypass |
| Original query-grid centres `(col+.5,row+.5)` use the same continuous resize affine, without snapping | `trackeverything_stage.py:track_everything` | `demo_hf.py:820-840` preserves supplied query coordinates with input resolution enabled |
| Query depth uses `depth[t, floor(y), floor(x)]`, preserving the continuous ray; missing depth is never invented | `workers/trackeverything_worker.py:continuous_query_unprojection` | Replaces `demo_hf.py:196-246`; upstream rounds at :223-226 and fills missing depth at :228-232 |
| Undo scale first, then pose: `X_base = inv(E0) [norm_factor * X_pred; 1]`, with original metric E0 | `workers/trackeverything_worker.py:predictions_to_base` | `demo_hf.py:190-192`, `626-641` |
| Mask frames before the exact query birth, including earlier frames in its birth window | `workers/trackeverything_worker.py:predictions_to_base` | `models/track_everything.py:526-532`, `1385-1401` |
| Visibility is sigmoid(logit), confidence is 1; OUT precedes OCCLUDED; 3D positions are NaN unless VALID | `trackeverything_stage.py:query_chunks` | `models/track_everything.py:477-479`, `1390-1401`; repository `tracking.STATUS` |
| Temporary forward hook obtains dynamic logits without upstream edits, then removes itself even on failure | `workers/trackeverything_worker.py:infer_queries` | `training/datatypes.py:320`, `models/track_everything.py:1403-1413`, `1578-1618`; `demo_hf.py:896-900` omits them |

Queries use exactly `pipeline.chunk_queries(n, window, step)` on each camera's
full timeline and `query_grid(original_hw, grid_stride, grid_stride//2)`. For the
640x480 view these are the original AllTracker query pixels. They generally become
subpixel coordinates on the model grid; replacing them with a new integer grid
would change the queried rays. We preserve the original rays.

One clip is one `chunk_queries` window, at most 16 frames (one TE model window),
including frames after a recording pause. Inference never splits at pauses. Both
trackers use the same downstream `track_motion` pause segmentation.

Queries require valid birth depth on our original grid and at their exact resized
lookup cell. Both use lift3d's 3x3 test: every neighbour must be finite, positive
and within 5% of the centre. Unsupported queries are excluded from worker inputs;
their identities remain in the output with NO_DEPTH or EDGE at birth and
NOT_TRACKED elsewhere. Each camera records `queries`, `unsupported_birth_depth`
and `unsupported_birth_edge`. A clip with no supported queries skips inference.
Our worker replaces upstream `_unproject_2d_to_3d` in memory, using floor lookup
and rejecting missing depth. Upstream's boundary-depth replacement of the dense
geometry inputs remains TE preprocessing (`demo_hf.py:619-624`,
`datasets/data_ops.py:1244-1245`) and is disclosed in worker/record metadata.

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
births, visibility/status, grid/resize rays and query-depth lookup, unsupported
births, identical AllTracker inference frames across pauses, durable saves, resuming either stage, and motion/export/viewer
consumers. The existing pipeline suite is also run.

Open validation items:

- Real checkpoint inference and target-server memory/runtime have not been tested.
  The environment file is an inference recipe stub, not a validated transitive
  lock. `setup_envs.sh` documents torch 2.8.0+cu129, Python 3.10, the pt28cu129
  torch_scatter wheel, and a matching-ABI local wheel fallback if sm_120 fails.
  No setup, downloads, or server actions were performed.
