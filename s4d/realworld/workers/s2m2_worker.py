"""S2M2 worker (separate ``s2m2`` environment; imports the pinned upstream checkout ``junhong-3dv/s2m2``).

Candidate own-rig depth backend (study 2026-11-08, ``realworld-ops/calib-loma/s2m2``), parallel to the FoundationStereo
worker with the SAME task and output contract so that the depth stage can switch by name. Written against the pinned
upstream ``src/s2m2/core/utils/model_utils.py`` (``load_model``, ``run_stereo_matching``) and
``src/s2m2/core/utils/image_utils.py`` (``image_pad``/``image_crop``, factor 32): the model is built with the upstream
size table (S/M/L/XL = feature channels 128/192/256/384, transformers 1/2/3/3), ``use_positivity`` = not
``allow_negative`` (upstream default: positive disparities only), ``refine_iter`` = ``num_refine`` (upstream default 3),
images as float 0..255 3-channel tensors (the model normalises internally to [-1, 1]), forward under
``torch.amp.autocast(float16)`` (upstream ``run_stereo_matching``) or float32 (``precision: fp32``). Differences to the
upstream loader, all on the safe side: the checkpoint is loaded with ``weights_only=True`` exactly as upstream, but every
model tensor must be present with its shape (upstream ``my_load_state_dict`` silently skips shape mismatches, loads
``strict=False``; here a missing or mis-shaped model tensor is an error, and the checkpoint's extra tensors of modules
the public implementation does not build are listed in the response meta), and upstream's module-level ``open3d`` import
(used only by its point-cloud helper) is avoided by not importing ``model_utils``.

task ``ir_pair_disparity``: inputs ``left``, ``right`` (N,H,W) uint8 device-rectified infrared pairs (grayscale
replicated to 3 channels as MODEL INPUT ONLY); outputs ``disparity_left`` (left reference, ``x_left - x_right``) and
``disparity_right`` (right reference by flip-and-swap: flip(right) as left input, flip(left) as right input, result
flipped back), both (N,H,W) float32 on the given grid, plus the model's own per-pixel ``occlusion_left`` (upstream:
0 = occluded) and ``confidence_left`` (upstream: probability that the disparity error is < 4 px), (N,H,W) float16,
for diagnostics. No resizing (the caller passes native resolution; 640x480 needs no padding).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _protocol import git_commit, main, sha256  # noqa: E402

_MODELS = {}
SIZES = {"S": (128, 1), "M": (192, 2), "L": (256, 3), "XL": (384, 3)}  # upstream model_utils.load_model table


def checkpoint_name(model_type: str) -> str:
    ch, ntr = SIZES[model_type]
    return f"CH{ch}NTR{ntr}.pth"


def gray_to_model_input(img):
    """(H,W) uint8 grayscale -> (H,W,3) uint8 replicated channels (model input only; the IR values are unchanged)."""
    import numpy as np

    img = np.asarray(img)
    if img.ndim != 2 or img.dtype != np.uint8:
        raise ValueError(f"expected an (H,W) uint8 infrared image, got {img.shape}/{img.dtype}")
    return np.repeat(img[..., None], 3, -1)


def flip_and_swap(disparity_fn, left3, right3):
    """Right-referenced disparity from a left-reference matcher: flip(right) as left input, flip(left) as right input,
    output flipped back (the same construction as the FoundationStereo worker and ``stereo.flip_and_swap``)."""
    import numpy as np

    out = disparity_fn(np.ascontiguousarray(right3[:, ::-1]), np.ascontiguousarray(left3[:, ::-1]))
    return tuple(np.ascontiguousarray(o[:, ::-1]) for o in out)


def _model(repo: Path, ckpt: Path, params: dict):
    import torch

    model_type = str(params.get("model_type", "XL"))
    key = (str(ckpt), model_type, int(params.get("num_refine", 3)), bool(params.get("allow_negative", False)))
    if key not in _MODELS:
        if checkpoint_name(model_type) != ckpt.name:
            raise ValueError(f"model_type {model_type} needs {checkpoint_name(model_type)}, got {ckpt.name}")
        src = str(repo / "src")
        if src not in sys.path:
            sys.path.insert(0, src)
        from s2m2.core.model.s2m2 import S2M2  # upstream

        ch, ntr = SIZES[model_type]
        model = S2M2(feature_channels=ch, dim_expansion=1, num_transformer=ntr, use_positivity=not key[3],
                     refine_iter=key[2])
        state = torch.load(ckpt, map_location="cpu", weights_only=True)["state_dict"]
        own = model.state_dict()
        missing = sorted(set(own) - set(state))
        unexpected = sorted(set(state) - set(own))
        shape = sorted(k for k in set(own) & set(state) if tuple(own[k].shape) != tuple(state[k].shape))
        if missing or shape:
            raise RuntimeError(f"S2M2 checkpoint does not match the model: missing {missing[:5]} ({len(missing)}), "
                               f"shape {shape[:5]} ({len(shape)})")
        # The released checkpoints also carry tensors of modules this public implementation does not build (README:
        # the attention-based refinement was replaced by a UNet; observed: disp_init.layer_norm1.*,
        # feature_flow_attn.*). They are unused; every model tensor is loaded (missing == 0, shapes equal).
        res = model.load_state_dict({k: v for k, v in state.items() if k in own}, strict=True)
        assert not res.missing_keys and not res.unexpected_keys
        model.cuda().eval()
        _MODELS[key] = (model, {"model_type": model_type, "num_refine": key[2], "allow_negative": key[3],
                                "n_parameters": int(sum(p.numel() for p in model.parameters())),
                                "checkpoint_unused_keys": unexpected})
    return _MODELS[key]


def _infer(model, left3, right3, precision: str):
    """left3/right3 (H,W,3) uint8 -> (disparity float32, occlusion float32, confidence float32), (H,W) each
    (upstream ``run_stereo_matching`` call sequence: image_pad(32) -> forward -> image_crop)."""
    import torch
    from s2m2.core.utils.image_utils import image_crop, image_pad  # upstream

    H, W = left3.shape[:2]
    img0 = torch.as_tensor(left3).permute(2, 0, 1)[None].cuda()
    img1 = torch.as_tensor(right3).permute(2, 0, 1)[None].cuda()
    img0, img1 = image_pad(img0, 32), image_pad(img1, 32)  # float, 0..255 (no-op padding at 640x480)
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=precision == "fp16", dtype=torch.float16):
        disp, occ, conf = model(img0, img1)
    return tuple(image_crop(x, (H, W)).float().cpu().numpy().reshape(H, W).astype("float32")
                 for x in (disp, occ, conf))


def handle(request, inputs):
    import time

    import numpy as np
    import torch

    repo = Path(request["repo_dir"]).resolve()
    ckpt = Path(request["checkpoint"])
    if not ckpt.is_file():
        raise FileNotFoundError("S2M2 needs the local checkpoint named in the request (no download)")
    params = request["params"]
    precision = str(params.get("precision", "fp16"))
    if precision not in ("fp16", "fp32"):
        raise ValueError(precision)
    torch.backends.cudnn.benchmark = False  # deterministic kernel choice across calls
    model, info = _model(repo, ckpt, params)
    torch.cuda.reset_peak_memory_stats()
    task = request["task"]
    if task != "ir_pair_disparity":
        raise ValueError(task)
    left, right = np.asarray(inputs["left"]), np.asarray(inputs["right"])
    if left.ndim != 3 or left.dtype != np.uint8 or left.shape != right.shape:
        raise ValueError(f"ir_pair_disparity expects (N,H,W) uint8 pairs, got {left.shape}/{left.dtype}")
    fn = lambda a, b: _infer(model, a, b, precision)  # noqa: E731
    dl, dr, occ, conf, times = [], [], [], [], []
    for a, b in zip(left, right):
        a3, b3 = gray_to_model_input(a), gray_to_model_input(b)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        d, o, c = fn(a3, b3)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        dl.append(d), occ.append(o), conf.append(c)
        dr.append(flip_and_swap(fn, a3, b3)[0])
    outputs = {"disparity_left": np.stack(dl), "disparity_right": np.stack(dr),
               "occlusion_left": np.stack(occ).astype(np.float16), "confidence_left": np.stack(conf).astype(np.float16)}
    if ("sha", str(ckpt)) not in _MODELS:
        _MODELS[("sha", str(ckpt))] = sha256(str(ckpt))
    identity = {"name": "S2M2", "repo": "junhong-3dv/s2m2", "commit": git_commit(str(repo)),
                "checkpoint": str(ckpt), "checkpoint_sha256": _MODELS[("sha", str(ckpt))], "is_mock": False}
    meta = {"reference": "left (+ right by flip-and-swap)", **info, "precision": precision,
            "torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(),
            "input": "grayscale IR replicated to 3 channels, float 0..255, native resolution, image_pad(32)",
            "left_forward_s": [float(t) for t in times],
            "max_memory_allocated": int(torch.cuda.max_memory_allocated())}
    return identity, outputs, meta


if __name__ == "__main__":
    main(handle)
