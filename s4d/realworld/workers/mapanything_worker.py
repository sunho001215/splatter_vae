"""MapAnything worker (runs in the separate ``mapanything`` environment with the pinned upstream checkout).

Upstream: facebookresearch/map-anything (``MapAnything.from_pretrained`` + ``preprocess_inputs`` + ``model.infer``),
checkpoint ``facebook/map-anything`` (``config.json`` + ``model.safetensors``) at a pinned Hugging Face revision, loaded
from a local directory (RAM-backed in the inference container; never a persistent model cache).

task ``infer_views``: one or more synchronized frame SETS of the same cameras, each processed as one multi-view scene.

inputs
  ``images``  (S,V,H,W,3) uint8 RGB on each view's pinhole grid (all views share H, W)
  ``K``       (V,3,3) float intrinsics of that grid in the REPOSITORY convention (continuous pixels, centre of the
              top-left pixel at (0.5, 0.5))
  ``cam2world`` (V,4,4) float, OpenCV axes, metres, world = robot base (``inv(T_cam_base)``)
  ``depth``   (S,V,H,W) float32 metric camera-z (0 = invalid), optional (noisy metric sensor depth)
  ``joint``   (S,) bool, optional: False processes that set's views one by one (never as one static scene)
params ``model_hw`` [h, w] (multiples of 14; 640x480 -> [392, 518]), ``hf_dir``, ``hf_revision``, ``use_depth``,
  ``use_pose``, ``apply_mask``, ``mask_edges``, ``amp_dtype``.

Geometry: this worker resizes with ONE explicit full-image pixel affine (``x' = s x`` in continuous coordinates,
corners to corners: LANCZOS for images as upstream, exact nearest for depth), scales K accordingly, converts to
MapAnything's pixel convention (rays from ``arange(W)``: pixel centres at integers, i.e. K_continuous - 0.5) and
asserts that upstream ``preprocess_inputs`` is then the identity (no second resize/crop). Outputs are mapped back to
the input grid with the exact inverse nearest lookup.

outputs (S,V,H,W) on the input grid: ``depth`` float32 camera-z (0 where invalid), ``valid`` bool (MapAnything mask
& finite & > 0), ``conf`` float16, ``non_ambiguous`` bool; per view ``cam2world_pred`` (S,V,4,4), ``K_pred_model``
(S,V,3,3) (model grid, MapAnything convention), ``metric_scale`` (S,V); ``depth_model``-grid outputs are not
returned.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _protocol import git_commit, main, sha256  # noqa: E402

_CACHE = {}


def nearest_index(n_dst: int, n_src: int):
    """Source index of each destination pixel for a full-image resize in continuous coordinates (exact nearest:
    the source pixel containing the destination pixel centre)."""
    import numpy as np

    return np.clip(np.floor((np.arange(n_dst) + 0.5) * (n_src / n_dst)).astype(np.int64), 0, n_src - 1)


def resize_nearest(a, out_hw):
    rows, cols = nearest_index(out_hw[0], a.shape[-2]), nearest_index(out_hw[1], a.shape[-1])
    return a[..., rows[:, None], cols[None, :]]


def scaled_K(K, src_hw, dst_hw):
    import numpy as np

    S = np.diag([dst_hw[1] / src_hw[1], dst_hw[0] / src_hw[0], 1.0])
    return S @ np.asarray(K, float)


def to_mapanything_K(K_continuous):
    K = K_continuous.copy()
    K[0, 2] -= 0.5
    K[1, 2] -= 0.5
    return K


def _model(hf_dir: str, device: str):
    key = (hf_dir, device)
    if key not in _CACHE:
        from mapanything.models import MapAnything  # upstream

        model = MapAnything.from_pretrained(hf_dir).to(device)
        model.eval()
        _CACHE[key] = model
    return _CACHE[key]


def handle(request, inputs):
    import numpy as np
    import torch
    from mapanything.utils.image import preprocess_inputs  # upstream
    from PIL import Image

    params = request["params"]
    repo = Path(request["repo_dir"]).resolve()
    hf_dir = params["hf_dir"]
    weights = str(Path(hf_dir) / "model.safetensors")
    model = _model(hf_dir, "cuda")
    images, K_in, c2w = inputs["images"], np.asarray(inputs["K"], float), np.asarray(inputs["cam2world"], float)
    S, V, H, W = images.shape[:4]
    depth = inputs.get("depth")
    joint = inputs.get("joint", np.ones(S, bool))
    h, w = (int(x) for x in params["model_hw"])
    if h % 14 or w % 14:
        raise ValueError(f"model_hw {h}x{w} is not a multiple of the 14-px patch")
    K_model = [to_mapanything_K(scaled_K(K_in[v], (H, W), (h, w))) for v in range(V)]
    use_depth = bool(params.get("use_depth", True)) and depth is not None
    out = {"depth": np.zeros((S, V, H, W), np.float32), "valid": np.zeros((S, V, H, W), bool),
           "conf": np.zeros((S, V, H, W), np.float16), "non_ambiguous": np.zeros((S, V, H, W), bool),
           "cam2world_pred": np.zeros((S, V, 4, 4), np.float32), "K_pred_model": np.zeros((S, V, 3, 3), np.float32),
           "metric_scale": np.zeros((S, V), np.float32)}
    checks = {"preprocess_identity_max_K_err": 0.0, "preprocess_identity_max_depth_err": 0.0}
    torch.cuda.reset_peak_memory_stats()
    for s in range(S):
        views = []
        for v in range(V):
            img = np.asarray(Image.fromarray(images[s, v]).resize((w, h), Image.LANCZOS))
            view = {"img": img, "intrinsics": K_model[v].astype(np.float32),
                    "camera_poses": c2w[v].astype(np.float32), "is_metric_scale": torch.tensor([True])}
            if use_depth:
                view["depth_z"] = resize_nearest(depth[s, v], (h, w)).astype(np.float32)
            views.append(view)
        groups = [list(range(V))] if bool(joint[s]) else [[v] for v in range(V)]
        for group in groups:
            processed = preprocess_inputs([views[v] for v in group])
            for v, pv in zip(group, processed):  # upstream preprocessing must not resize or crop again
                if tuple(pv["img"].shape[-2:]) != (h, w):
                    raise RuntimeError(f"preprocess_inputs changed the grid to {tuple(pv['img'].shape[-2:])}")
                checks["preprocess_identity_max_K_err"] = max(checks["preprocess_identity_max_K_err"], float(
                    np.abs(pv["intrinsics"][0].numpy() - K_model[v]).max()))
                if use_depth:
                    checks["preprocess_identity_max_depth_err"] = max(checks["preprocess_identity_max_depth_err"], float(
                        np.abs(pv["depth_z"][0].numpy() - views[v]["depth_z"]).max()))
            with torch.no_grad():
                preds = model.infer(processed, memory_efficient_inference=False, use_amp=True,
                                    amp_dtype=params.get("amp_dtype", "bf16"),
                                    apply_mask=bool(params.get("apply_mask", True)),
                                    mask_edges=bool(params.get("mask_edges", True)), apply_confidence_mask=False,
                                    ignore_calibration_inputs=False, ignore_depth_inputs=not use_depth,
                                    ignore_pose_inputs=not bool(params.get("use_pose", True)),
                                    ignore_depth_scale_inputs=False, ignore_pose_scale_inputs=False)
            for v, p in zip(group, preds):
                d = p["depth_z"][0, ..., 0].float().cpu().numpy()
                m = p["mask"][0, ..., 0].bool().cpu().numpy() & np.isfinite(d) & (d > 0)
                out["depth"][s, v] = resize_nearest(np.where(m, d, 0.0), (H, W))
                out["valid"][s, v] = resize_nearest(m, (H, W))
                out["conf"][s, v] = resize_nearest(p["conf"][0].float().cpu().numpy(), (H, W))
                out["non_ambiguous"][s, v] = resize_nearest(p["non_ambiguous_mask"][0].bool().cpu().numpy(), (H, W))
                out["cam2world_pred"][s, v] = p["camera_poses"][0].float().cpu().numpy()
                out["K_pred_model"][s, v] = p["intrinsics"][0].float().cpu().numpy()
                out["metric_scale"][s, v] = float(p["metric_scaling_factor"].reshape(-1)[0])
    if checks["preprocess_identity_max_K_err"] > 1e-3 or checks["preprocess_identity_max_depth_err"] > 0:
        raise RuntimeError(f"upstream preprocessing was not the identity: {checks}")
    if weights not in _CACHE:
        _CACHE[weights] = sha256(weights)
    identity = {"name": "MapAnything", "repo": "facebookresearch/map-anything", "commit": git_commit(str(repo)),
                "checkpoint": weights, "checkpoint_sha256": _CACHE[weights],
                "hf_repo": params.get("hf_repo", "facebook/map-anything"), "hf_revision": params.get("hf_revision"),
                "is_mock": False}
    meta = {"model_hw": [h, w], "input_hw": [H, W], "sets": S, "views": V, "joint": [bool(x) for x in joint],
            "use_depth": use_depth, "use_pose": bool(params.get("use_pose", True)), "torch": torch.__version__,
            "amp_dtype": params.get("amp_dtype", "bf16"), "pixel_convention_model": "integer centres (K - 0.5)",
            "checks": checks, "max_memory_allocated": int(torch.cuda.max_memory_allocated())}
    return identity, out, meta


if __name__ == "__main__":
    main(handle)
