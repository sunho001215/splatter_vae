"""MOCK worker for protocol tests. Its outputs are NOT model outputs and are always flagged ``is_mock``.

tasks:
  stereo_disparity  CPU SAD block matching (s4d.realworld.stereo.block_match) on rectified pairs
  echo_tracks       returns the tracks given in the inputs (an oracle; synthetic tests only)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repository root (-I drops the script dir)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _protocol import main  # noqa: E402


def handle(request, inputs):
    import numpy as np

    from s4d.realworld.stereo import block_match

    task, params = request["task"], request["params"]
    model = {"name": "mock", "repo": None, "commit": None, "checkpoint": None, "checkpoint_sha256": None,
             "is_mock": True}
    if task == "stereo_disparity":
        left, right = inputs["left"], inputs["right"]
        disp = np.stack([block_match(a, b, int(params.get("max_disparity", 32))) for a, b in zip(left, right)])
        return model, {"disparity": disp}, {"reference": "left", "matcher": "block_match (CPU SAD)"}
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


if __name__ == "__main__":
    main(handle)
