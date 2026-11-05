"""Single-view TrackEverything worker; upstream imports happen only inside ``_model``.

``track_queries`` takes RGB uint8 (T,H,W,3), metric camera-z ``depth`` (T,H,W), continuous
``K`` (T,3,3), OpenCV world-to-camera ``w2c`` (T,4,4), and ``queries`` (N,3) [t,x,y].
H,W must be multiples of 16. Outputs: ``xyz_base`` (T,N,3), visibility/dynamic LOGITS
(T,N), and ``birth_t`` (N,). All predictions before exact query birth are NaN.

References below refer to ayushjain1144/trackeverything at dfb29b505cf2ea954f8697fb612c2429668dd3a0.
No geometry estimator is loaded: all three geometry inputs are mandatory, and backend
``pi3`` selects the cam0/scale preprocessing rather than the VGGT-Omega special case.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _protocol import git_commit, main, sha256  # noqa: E402

_CACHE = {}


def validate_inputs(inputs):
    video, depth, K, w2c, queries = (np.asarray(inputs[k]) for k in ("video", "depth", "K", "w2c", "queries"))
    if video.ndim != 4 or video.shape[-1] != 3 or video.dtype != np.uint8:
        raise ValueError("video must be (T,H,W,3) uint8 RGB, one camera only")
    t, h, w = video.shape[:3]
    if not 0 < t <= 16 or h <= 0 or w <= 0 or h % 16 or w % 16:
        raise ValueError("clips need 1..16 frames (one query window) and H,W divisible by 16")
    if depth.shape != (t, h, w) or depth.dtype != np.float32 or not np.isfinite(depth).all() or (depth < 0).any():
        raise ValueError("depth must be finite nonnegative (T,H,W) float32 camera-z metres; 0 invalid")
    if K.shape != (t, 3, 3) or w2c.shape != (t, 4, 4) or not np.isfinite(K).all() or not np.isfinite(w2c).all():
        raise ValueError("K and w2c must be finite (T,3,3) and (T,4,4)")
    if queries.ndim != 2 or queries.shape[1] != 3 or not len(queries) or not np.isfinite(queries).all():
        raise ValueError("queries must be nonempty finite (N,3) [t,x,y]")
    birth = queries[:, 0]
    if ((birth != np.floor(birth)) | (birth < 0) | (birth >= t)).any():
        raise ValueError("query birth must be an integer clip frame")
    if ((queries[:, 1:] < 0) | (queries[:, 1:] >= [w, h])).any():
        raise ValueError("query pixels must lie inside the resized image")
    x, y = np.floor(queries[:, 1:]).astype(int).T
    if (depth[birth.astype(int), y, x] <= 0).any():
        raise ValueError("queries require valid birth depth; depth invention is disabled")
    return video, depth, K, w2c, queries


def continuous_query_unprojection(query_points_2d, depths, intrinsics, extrinsics):
    """Our replacement for dfb29b5 demo_hf.py:196-246 (_unproject_2d_to_3d).

    Upstream :223-226 rounds/clamps pixels and :228-232 invents missing depths. Continuous coordinates instead
    select depth[t, floor(y), floor(x)] while retaining the original ray. Invalid support is an error, never filled.
    Geometry depth still undergoes TE's boundary replacement (demo_hf.py:619-624; datasets/data_ops.py:1244-1245).
    """
    import torch

    q = torch.as_tensor(query_points_2d, device=depths.device, dtype=depths.dtype)
    t = q[:, 0].long()
    xy = q[:, 1:]
    h, w = depths.shape[1:]
    if (not torch.isfinite(q).all() or (q[:, 0] != t).any() or (t < 0).any() or (t >= len(depths)).any()
            or (xy < 0).any() or (xy >= q.new_tensor([w, h])).any()):
        raise ValueError("queries must have finite in-bounds continuous pixels and integer birth frames")
    x, y = torch.floor(xy).long().unbind(-1)
    z = depths[t, y, x]
    if not (torch.isfinite(z) & (z > 0)).all():
        raise ValueError("query birth depth is invalid; depth invention is disabled")
    uv_h = torch.cat([xy, torch.ones_like(q[:, :1])], -1)
    rays = torch.linalg.solve(intrinsics[t], uv_h[..., None]).squeeze(-1)
    cam_h = torch.cat([rays * z[:, None], torch.ones_like(q[:, :1])], -1)
    world = torch.linalg.solve(extrinsics[t], cam_h[..., None]).squeeze(-1)[:, :3]
    return torch.cat([q[:, :1], world], -1).unsqueeze(0)


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value, np.float32)


def predictions_to_base(query_coords, query_visibs, norm_factor, w2c, queries, dynamic_logits=None):
    """Undo cam0 alignment and scale, then mask *exact* birth, including inside the birth window.

    demo_hf.py:626-641 uses E'_t = E_t inv(E_0); :190-192 divides depth and pose
    translation by norm. Thus X_base = inv(E_0) [norm * X_pred; 1], using the
    ORIGINAL metric E_0, never the returned normalized extrinsic. Sparse outputs
    are (B,T,N,3)/(B,T,N), demo_hf.py:896-898. Earlier windows contain zero coordinates
    and logit 0 (models/track_everything.py:1385-1401), which must not become visible.
    """
    t, n = len(w2c), len(queries)
    coords, vis = _numpy(query_coords), _numpy(query_visibs)
    if coords.shape != (1, t, n, 3) or vis.shape != (1, t, n):
        raise ValueError("unexpected TrackEverything query prediction shape")
    norm = float(norm_factor)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("norm_factor must be finite and positive")
    E_inv = np.linalg.inv(np.asarray(w2c[0], np.float64))
    xyz = ((coords[0].astype(np.float64) * norm) @ E_inv[:3, :3].T + E_inv[:3, 3]).astype(np.float32)
    birth = np.asarray(queries[:, 0], np.int64)
    pre = np.arange(t)[:, None] < birth[None]
    xyz[pre] = np.nan
    vis = vis[0].copy()
    vis[pre] = np.nan
    out = {"xyz_base": xyz, "vis_logit": vis, "birth_t": birth}
    if dynamic_logits is not None:
        dyn = _numpy(dynamic_logits)
        if dyn.shape != (1, t, n):
            raise ValueError("unexpected query_dynamic_logits shape")
        out["dyn_logit"] = dyn[0].copy()
        out["dyn_logit"][pre] = np.nan
    return out


def _model(repo, checkpoint, device):
    """Local weights only (demo_hf.py:449-466); README.md:58-59 requires the package parent."""
    repo, checkpoint = Path(repo).resolve(), Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise ValueError("checkpoint must be an existing local file; hub downloads are disabled")
    if repo.name != "trackeverything" or not (repo / "demo_hf.py").is_file():
        raise ValueError("repo_dir must be the pinned upstream directory named trackeverything")
    key = (str(repo), str(checkpoint), device)
    if key not in _CACHE:
        sys.path.insert(0, str(repo.parent))
        from trackeverything import demo_hf

        if Path(demo_hf.__file__).resolve() != repo / "demo_hf.py":
            raise RuntimeError("TrackEverything import does not match repo_dir")
        demo_hf._unproject_2d_to_3d = continuous_query_unprojection
        identity = {"name": "TrackEverything", "repo": "ayushjain1144/trackeverything",
                    "commit": git_commit(str(repo)), "checkpoint": str(checkpoint),
                    "checkpoint_sha256": sha256(str(checkpoint)), "is_mock": False}
        tracker = demo_hf.TrackEverything.from_pretrained(str(checkpoint), device=device, geometry_backend="pi3")
        if tracker.model.seq_len != 16:
            raise ValueError("expected checkpoint seq_len=16 (non-overlapping upstream windows)")
        _CACHE[key] = tracker, identity
    return _CACHE[key]


def infer_queries(tracker, video, depth, K, w2c, queries, num_iters):
    """Capture Prediction.query_dynamic_logits without editing upstream.

    The model returns (Prediction, train_data), models/track_everything.py:1578-1618;
    Prediction carries query_dynamic_logits (training/datatypes.py:320), but the
    sparse demo wrapper omits it (demo_hf.py:896-900). A temporary forward hook
    retains only those logits, and is removed even when inference raises.
    """
    captured = {}

    def capture(_module, _args, result):
        captured["dynamic"] = getattr(result[0], "query_dynamic_logits", None)

    hook = tracker.model.register_forward_hook(capture)
    try:
        output = tracker.infer(video, depths=depth, intrinsics=K, extrinsics=w2c,
                               query_points_2d=queries, mode="sparse", num_iters=num_iters,
                               use_input_resolution=True)
        arrays = predictions_to_base(output.query_coords, output.query_visibs, output.norm_factor,
                                     w2c, queries, captured.get("dynamic"))
        return arrays, float(output.norm_factor)
    finally:
        hook.remove()


def handle(request, inputs):
    started = time.monotonic()
    if request["task"] != "track_queries" or request["params"].get("mode", "sparse") != "sparse":
        raise ValueError("only single-view track_queries mode=sparse is supported")
    video, depth, K, w2c, queries = validate_inputs(inputs)
    num_iters = int(request["params"].get("num_iters", 3))
    if num_iters < 1:
        raise ValueError("num_iters must be positive")
    tracker, identity = _model(request["repo_dir"], request["checkpoint"], "cuda")
    import torch

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    infer_started = time.monotonic()
    arrays, norm = infer_queries(tracker, video, depth, K, w2c, queries, num_iters)
    torch.cuda.synchronize()
    return identity, arrays, {"norm_factor": norm, "inference_s": time.monotonic() - infer_started,
                              "elapsed_s": time.monotonic() - started,
                              "max_memory_allocated": int(torch.cuda.max_memory_allocated()),
                              "max_memory_reserved": int(torch.cuda.max_memory_reserved()),
                              "dynamic_logits_available": "dyn_logit" in arrays, "single_view": True,
                              "geometry": "supplied metric depth, continuous K, OpenCV world-to-camera",
                              "query_depth_lookup": "depth[t, floor(y), floor(x)] on resized grid; no invention",
                              "geometry_preprocessing": "TE boundary-depth replacement retained: "
                                                        "demo_hf.py:619-624, datasets/data_ops.py:1244-1245",
                              "geometry_backend": "pi3 (estimator bypassed)", "torch": torch.__version__}


if __name__ == "__main__":
    main(handle)
