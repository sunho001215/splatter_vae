"""MOCK worker for protocol tests. Its outputs are NOT model outputs and are always flagged ``is_mock``.

tasks:
  infer_views       depth-input echo, or CPU SAD depth oracle for exactly two rectified rig views
  echo_tracks       returns the tracks given in the inputs (an oracle; synthetic tests only)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repository root (-I drops the script dir)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _protocol import main  # noqa: E402


def handle(request, inputs):
    import numpy as np

    task, params = request["task"], request["params"]
    model = {"name": "mock", "repo": None, "commit": None, "checkpoint": None, "checkpoint_sha256": None,
             "is_mock": True}
    if task == "infer_views":
        images, Ks, c2w = inputs["images"], inputs["K"], inputs["cam2world"]
        S, V, H, W = images.shape[:4]
        if "depth" in inputs:
            depth = np.asarray(inputs["depth"], np.float32).copy()
        else:
            if V != 2 or not np.asarray(inputs.get("joint", np.ones(S, bool))).all():
                raise ValueError("mock infer_views without depth needs exactly two rectified rig views")
            T = np.linalg.inv(c2w[1]) @ c2w[0]
            if (not np.allclose(T[:3, :3], np.eye(3), atol=1e-7)
                    or not np.allclose(T[1:3, 3], 0, atol=1e-7) or T[0, 3] >= 0
                    or not np.allclose(Ks[0][:2, :2], Ks[1][:2, :2])
                    or not np.isclose(Ks[0][1, 2], Ks[1][1, 2])):
                raise ValueError("mock infer_views requires a horizontal rectified stereo rig")
            baseline = -T[0, 3]
            depth = np.zeros((S, V, H, W), np.float32)
            for k in range(S):
                for v in range(2):
                    a, b = images[k, v], images[k, 1 - v]
                    if v:
                        a, b = a[:, ::-1], b[:, ::-1]
                    shift = block_match(a, b, int(params.get("max_disparity", 32)))
                    if v:
                        shift = shift[:, ::-1]
                    corrected = shift - (Ks[0, 0, 2] - Ks[1, 0, 2])
                    x = np.arange(W)[None, :] + 0.5
                    other_x = x + shift if v else x - shift
                    valid = np.isfinite(corrected) & (corrected > 1e-3) & (other_x >= 0) & (other_x <= W)
                    np.divide(Ks[v, 0, 0] * baseline, corrected, out=depth[k, v], where=valid)
        valid = np.isfinite(depth) & (depth > 0)
        depth[~valid] = 0
        return model, {"depth": depth, "valid": valid, "conf": valid.astype(np.float16),
                       "non_ambiguous": valid.copy(),
                       "cam2world_pred": np.broadcast_to(c2w, (S, V, 4, 4)).astype(np.float32).copy(),
                       "K_pred_model": np.broadcast_to(Ks, (S, V, 3, 3)).astype(np.float32).copy(),
                       "metric_scale": np.ones((S, V), np.float32)}, {"oracle": True, "sets": S, "views": V}
    if task == "dino_snerl":  # deterministic fake tokens with the consumer's grid shape (31x31x384 at 128x128)
        images = inputs["images"]
        h = (images.shape[1] - 8) // 4 + 1
        w = (images.shape[2] - 8) // 4 + 1
        tokens = np.stack([np.full((h, w, 384), float(im.mean()), np.float32) for im in images])
        return model, {"tokens": tokens}, {"output": "tokens", "verified_against_consumer": None}
    if task == "segment_images":  # one fake instance per prompt on frame 0 only; others absent
        images = inputs["images"]
        H, W = images.shape[1:3]
        P = len(request["params"]["prompts"])
        masks = np.zeros((P, H, W), bool)
        for p in range(P):
            masks[p, p * 4:(p + 1) * 4, :] = True
        count = np.zeros((len(images), P), np.int32)
        count[0] = 1
        return model, {"masks": masks, "scores": np.linspace(0.9, 0.6, P).astype(np.float32),
                       "boxes_xyxy": np.zeros((P, 4), np.float32), "frame": np.zeros(P, np.int32),
                       "prompt_index": np.arange(P, dtype=np.int32), "instance_id": np.zeros(P, np.int32),
                       "count": count}, {"prompts": request["params"]["prompts"]}
    if task == "echo_tracks":
        return model, {k: inputs[k] for k in ("flows", "visconf")}, {"oracle": True}
    raise ValueError(f"unknown task {task!r}")


def block_match(left, right, max_disparity: int = 32, window: int = 5):
    """Test-only integer SAD oracle with parabolic subpixel refinement; never used as a model fallback."""
    import numpy as np
    from scipy.ndimage import uniform_filter

    L = np.asarray(left, np.float64)
    R = np.asarray(right, np.float64)
    if L.ndim == 3:
        L, R = L.mean(-1), R.mean(-1)
    H, W = L.shape
    cost = np.full((max_disparity + 1, H, W), np.inf)
    for d in range(max_disparity + 1):
        diff = np.full((H, W), np.inf)
        diff[:, d:] = np.abs(L[:, d:] - R[:, : W - d])
        filled = np.where(np.isfinite(diff), diff, 1e3)
        cost[d] = uniform_filter(filled, size=window, mode="nearest")
        cost[d][:, :d] = np.inf
    best = np.argmin(cost, axis=0)
    disp = best.astype(np.float64)
    inner = (best > 0) & (best < max_disparity)
    ii, jj = np.nonzero(inner)
    c0 = cost[best[ii, jj] - 1, ii, jj]
    c1 = cost[best[ii, jj], ii, jj]
    c2 = cost[best[ii, jj] + 1, ii, jj]
    denom = c0 - 2 * c1 + c2
    ok = np.isfinite(denom) & (denom > 1e-9)
    disp[ii[ok], jj[ok]] += 0.5 * (c0[ok] - c2[ok]) / denom[ok]
    disp[~np.isfinite(cost.min(0))] = np.nan
    return disp.astype(np.float32)


if __name__ == "__main__":
    main(handle)
