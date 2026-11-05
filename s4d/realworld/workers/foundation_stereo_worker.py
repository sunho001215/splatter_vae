"""FoundationStereo worker (separate ``foundation_stereo`` environment; imports the pinned upstream checkout).

Restored 2026-11-05 as an optional backend for the controlled depth comparison (it had been removed at 379bca5).
Written against NVlabs/FoundationStereo ``scripts/run_demo.py`` at the pinned commit
(configs/realworld/upstream_pins.yaml): model built from the checkpoint's ``cfg.yaml``, ``InputPadder(divis_by=32)``,
``model.forward(img0, img1, iters=valid_iters, test_mode=True)`` (or ``run_hierachical``) under autocast, images as
float 0..255 3-channel tensors. Pretrained inference only.

task ``ir_pair_disparity``: inputs ``left``, ``right`` (N,H,W) uint8 device-rectified infrared pairs (grayscale
replicated to 3 channels as MODEL INPUT ONLY); outputs ``disparity_left`` (left reference, ``x_left - x_right``)
and ``disparity_right`` (right reference by flip-and-swap: flip(right) as left input, flip(left) as right input,
result flipped back), both (N,H,W) float32 on the given grid. No resizing (the caller passes native resolution).
task ``stereo_disparity``: inputs ``left``, ``right`` (N,H,W,3) uint8 rectified pairs; output ``disparity``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _protocol import git_commit, main, sha256  # noqa: E402

_MODELS = {}


def _model(repo: Path, ckpt: Path, params: dict):
    import torch
    from omegaconf import OmegaConf

    key = (str(ckpt), int(params.get("valid_iters", 32)), int(params.get("hiera", 0)))
    if key not in _MODELS:
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from core.foundation_stereo import FoundationStereo  # upstream

        cfg = OmegaConf.load(ckpt.parent / "cfg.yaml")
        if "vit_size" not in cfg:
            cfg["vit_size"] = "vitl"
        cfg["valid_iters"] = key[1]
        cfg["hiera"] = key[2]
        args = OmegaConf.create(cfg)
        # Offline construction: upstream builds its EdgeNeXt stem with timm pretrained=True (an ImageNet download) and
        # DINOv2 from the torch.hub code cache (pretrained_dino=False). Every weight is then overwritten by the official
        # checkpoint, loaded STRICTLY below, so constructing EdgeNeXt without the download changes no final weight.
        import timm

        create = timm.create_model
        timm.create_model = lambda name, *a, **kw: create(name, *a, **{**kw, "pretrained": False})
        try:
            model = FoundationStereo(args)
        finally:
            timm.create_model = create
        # torch >= 2.6 defaults to weights_only=True; the official checkpoint also pickles numpy scalars (global_step,
        # epoch). Its sha256 is pinned in the server config and reported in the identity below.
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model"], strict=True)
        model.cuda().eval()
        _MODELS[key] = (model, args)
    return _MODELS[key]


def _disparity(model, args, left3, right3):
    """left3/right3 (H,W,3) uint8 -> left-reference disparity (H,W) float32 (run_demo.py's call sequence)."""
    import torch
    from core.utils.utils import InputPadder  # upstream

    H, W = left3.shape[:2]
    img0 = torch.as_tensor(left3).cuda().float()[None].permute(0, 3, 1, 2)
    img1 = torch.as_tensor(right3).cuda().float()[None].permute(0, 3, 1, 2)
    padder = InputPadder(img0.shape, divis_by=32, force_square=False)
    img0, img1 = padder.pad(img0, img1)
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=True):
        if args.hiera:
            disp = model.run_hierachical(img0, img1, iters=args.valid_iters, test_mode=True, small_ratio=0.5)
        else:
            disp = model.forward(img0, img1, iters=args.valid_iters, test_mode=True)
    return padder.unpad(disp.float()).cpu().numpy().reshape(H, W).astype("float32")


def handle(request, inputs):
    import numpy as np
    import torch

    repo = Path(request["repo_dir"]).resolve()
    ckpt = Path(request["checkpoint"])
    if not ckpt.is_file():
        raise FileNotFoundError("FoundationStereo needs the local checkpoint named in the request (no download)")
    params = request["params"]
    model, args = _model(repo, ckpt, params)
    torch.cuda.reset_peak_memory_stats()
    task = request["task"]
    if task == "ir_pair_disparity":
        left, right = np.asarray(inputs["left"]), np.asarray(inputs["right"])
        if left.ndim != 3 or left.dtype != np.uint8 or left.shape != right.shape:
            raise ValueError(f"ir_pair_disparity expects (N,H,W) uint8 pairs, got {left.shape}/{left.dtype}")
        dl, dr = [], []
        for a, b in zip(left, right):
            a3, b3 = np.repeat(a[..., None], 3, -1), np.repeat(b[..., None], 3, -1)
            dl.append(_disparity(model, args, a3, b3))
            flipped = _disparity(model, args, np.ascontiguousarray(b3[:, ::-1]), np.ascontiguousarray(a3[:, ::-1]))
            dr.append(np.ascontiguousarray(flipped[:, ::-1]))
        outputs = {"disparity_left": np.stack(dl), "disparity_right": np.stack(dr)}
    elif task == "stereo_disparity":
        outputs = {"disparity": np.stack([_disparity(model, args, a, b)
                                          for a, b in zip(inputs["left"], inputs["right"])])}
    else:
        raise ValueError(task)
    if ("sha", str(ckpt)) not in _MODELS:  # hash the 3.3 GB checkpoint once per worker session
        _MODELS[("sha", str(ckpt))] = sha256(str(ckpt))
    identity = {"name": "FoundationStereo", "repo": "NVlabs/FoundationStereo", "commit": git_commit(str(repo)),
                "checkpoint": str(ckpt), "checkpoint_sha256": _MODELS[("sha", str(ckpt))], "is_mock": False}
    meta = {"reference": "left (+ right by flip-and-swap)" if task == "ir_pair_disparity" else "left",
            "valid_iters": int(args.valid_iters), "hiera": int(args.hiera), "torch": torch.__version__,
            "input": "grayscale IR replicated to 3 channels, float 0..255, native resolution, InputPadder(32)",
            "max_memory_allocated": int(torch.cuda.max_memory_allocated())}
    return identity, outputs, meta


if __name__ == "__main__":
    main(handle)
