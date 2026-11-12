"""Single-view TrackEverything worker; upstream imports happen only inside ``_model``.

``track_queries`` takes RGB uint8 (T,H,W,3), metric camera-z ``depth`` (T,H,W), continuous
``K`` (T,3,3), OpenCV world-to-camera ``w2c`` (T,4,4), and ``queries`` (N,3) [t,x,y].
H,W must be multiples of 16. Outputs: ``xyz_base`` (T,N,3), visibility/dynamic LOGITS
(T,N), and ``birth_t`` (N,). All predictions before exact query birth are NaN.

References below refer to ayushjain1144/trackeverything at dfb29b505cf2ea954f8697fb612c2429668dd3a0.
No geometry estimator is loaded: all three geometry inputs are mandatory, and backend
``pi3`` selects the cam0/scale preprocessing rather than the VGGT-Omega special case.

Provenance (2026-11-06, user decision item 3: normal fast GPU inference, pinned and recorded): every response's
``meta.runtime`` records the GPU (name, UUID, capability), NVIDIA driver, CUDA runtime / driver API, cuDNN, torch,
the precision/determinism flags in effect after inference, the sha256 of every torch.hub checkpoint TE can load
(its DINOv2 ViT-S/14 trunk is constructed from torch.hub before model.pth overwrites it) and the seed. The inference
path draws no random numbers (upstream ``randperm`` calls are training-only: models/track_everything.py:888 and the
drop-path blocks check ``self.training``); torch/numpy/random are still seeded with ``SEED`` before every request so
that any hidden draw is order-independent. GPU kernels stay nondeterministic (no deterministic mode).

CUDA memory (2026-11-12, agent TEMEM; the 1g.24gb MIG fit): one request's LIVE peak (weights + activations,
``max_memory_allocated``) was <= 4.15 GiB over all 20,802 calrel-6c3b9ec clips, but the persistent stream worker's
caching allocator never returned freed blocks while the 96-GB device had room, so its RESERVED memory grew
monotonically over an episode's ~70-90 requests (shapes vary per clip: query count, voxelized scene) to 21 GiB median
and 49.8 GiB maximum. After every request (also a failed one, whose traceback is dropped first so that the frames
holding request tensors are freed) this worker returns the cached, unused blocks to the driver
(``torch.cuda.empty_cache``); the loaded model stays resident. ``S4D_TE_CUDA_MEMORY_CAP_GIB`` caps the caching
allocator (``torch.cuda.set_per_process_memory_fraction``); it is REQUIRED on a MIG instance (a shared device: an
uncapped allocator could take memory that co-tenants were promised) and optional elsewhere. At the cap the allocator
frees its cached blocks and retries; if it still cannot back an allocation (live tensors plus fragmentation, segment
rounding, library workspaces) the request fails with an out-of-memory error - a failed request, never a different
result. Neither changes a requested computation; ``meta.memory`` and the runtime flags record the policy, the
allocator configuration and the per-request peaks.
"""

from __future__ import annotations

import gc
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _protocol import git_commit, main, sha256  # noqa: E402

_CACHE = {}
SEED = 0
_RUNTIME = {}
_MEMORY = {}
CUDA_MEMORY_CAP_ENV = "S4D_TE_CUDA_MEMORY_CAP_GIB"


def runtime_provenance(torch) -> dict:
    """Hardware/runtime identity of this worker process (static part cached) + the flags in effect now."""
    if not _RUNTIME:
        props = torch.cuda.get_device_properties(0)
        try:
            driver = Path("/proc/driver/nvidia/version").read_text().splitlines()[0].strip()
        except (OSError, IndexError):
            driver = None
        hub = Path(torch.hub.get_dir()) / "checkpoints"
        weights = {p.name: sha256(str(p.resolve())) for p in sorted(hub.glob("*")) if p.is_file()} if hub.is_dir() else {}
        api = getattr(torch._C, "_cuda_getDriverVersion", None)
        _RUNTIME.update(gpu={"name": props.name, "uuid": str(props.uuid), "capability": [props.major, props.minor],
                             "total_memory": int(props.total_memory)},
                        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), nvidia_driver=driver,
                        cuda_driver_api=int(api()) if api else None, cuda_runtime=torch.version.cuda,
                        cudnn=torch.backends.cudnn.version(), torch=torch.__version__, python=sys.version.split()[0],
                        torch_hub_dir=str(hub), torch_hub_checkpoints_sha256=weights)
    flags = {"cudnn_enabled": torch.backends.cudnn.enabled, "cudnn_benchmark": torch.backends.cudnn.benchmark,
             "cudnn_deterministic": torch.backends.cudnn.deterministic,
             "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
             "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
             "float32_matmul_precision": torch.get_float32_matmul_precision(),
             "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
             **{k: os.environ.get(k) for k in ("CUBLAS_WORKSPACE_CONFIG", "NVIDIA_TF32_OVERRIDE",
                                               "PYTORCH_CUDA_ALLOC_CONF", CUDA_MEMORY_CAP_ENV)}}
    return {**_RUNTIME, "flags": flags, "seed": SEED, "deterministic_mode": False}


class RequestFailed(RuntimeError):
    """A failed request, re-raised as text after its traceback (and the request tensors it kept alive) was dropped and
    the CUDA cache was released; the message carries the original exception type, message and traceback."""


def memory_policy(torch) -> dict:
    """Apply (once per process) and describe this worker's CUDA memory policy: cached blocks are released after every
    request; ``S4D_TE_CUDA_MEMORY_CAP_GIB`` (GiB) caps the caching allocator of device 0 - required on a MIG instance,
    optional elsewhere. A missing (on MIG) or invalid cap is an error (every request fails), never ignored."""
    if not _MEMORY:
        raw = os.environ.get(CUDA_MEMORY_CAP_ENV, "").strip()
        props = torch.cuda.get_device_properties(0)
        total = int(props.total_memory)
        mig = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip().startswith("MIG-") or " MIG " in f" {props.name} "
        cap = fraction = None
        if mig and not raw:
            raise ValueError(f"{CUDA_MEMORY_CAP_ENV} is required on a MIG instance (shared device); set it to the GiB "
                             "this worker may reserve")
        if raw:
            try:
                cap = float(raw)
            except ValueError:
                cap = math.nan
            if not math.isfinite(cap) or cap <= 0:
                raise ValueError(f"{CUDA_MEMORY_CAP_ENV} must be a positive number of GiB, got {raw!r}")
            fraction = min(1.0, cap * 2**30 / total)
            torch.cuda.set_per_process_memory_fraction(fraction, 0)
        _MEMORY.update(release_cached_after_request=True, cuda_memory_cap_gib=cap, memory_fraction=fraction,
                       device_total_memory=total, mig_instance=mig,
                       pytorch_cuda_alloc_conf=os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))
    return dict(_MEMORY)


def release_cuda_cache(torch) -> dict:
    """Return every cached, unused CUDA block to the driver; live tensors (the loaded model) stay. Cyclic garbage that
    may still reference request tensors is collected first."""
    gc.collect()
    torch.cuda.empty_cache()
    return {"allocated_after_release": int(torch.cuda.memory_allocated()),
            "reserved_after_release": int(torch.cuda.memory_reserved())}


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
    import torch

    failure = None
    try:
        memory = memory_policy(torch)
        tracker, identity = _model(request["repo_dir"], request["checkpoint"], "cuda")
        torch.manual_seed(SEED)  # also seeds every CUDA device
        np.random.seed(SEED)
        random.seed(SEED)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        infer_started = time.monotonic()
        arrays, norm = infer_queries(tracker, video, depth, K, w2c, queries, num_iters)
        torch.cuda.synchronize()
        inference_s = time.monotonic() - infer_started
        peak_allocated, peak_reserved = int(torch.cuda.max_memory_allocated()), int(torch.cuda.max_memory_reserved())
    except Exception as exc:  # keep only text: the traceback's frames hold the request's tensors
        failure = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    released = release_cuda_cache(torch)  # also after a failed request: never keep the request's cache
    if failure is not None:
        raise RequestFailed(failure)
    memory.update(released)
    return identity, arrays, {"norm_factor": norm, "inference_s": inference_s,
                              "elapsed_s": time.monotonic() - started,
                              "max_memory_allocated": peak_allocated, "max_memory_reserved": peak_reserved,
                              "memory": memory,
                              "dynamic_logits_available": "dyn_logit" in arrays, "single_view": True,
                              "geometry": "supplied metric depth, continuous K, OpenCV world-to-camera",
                              "query_depth_lookup": "depth[t, floor(y), floor(x)] on resized grid; no invention",
                              "geometry_preprocessing": "TE boundary-depth replacement retained: "
                                                        "demo_hf.py:619-624, datasets/data_ops.py:1244-1245",
                              "geometry_backend": "pi3 (estimator bypassed)", "torch": torch.__version__,
                              "runtime": runtime_provenance(torch)}


if __name__ == "__main__":
    main(handle)
