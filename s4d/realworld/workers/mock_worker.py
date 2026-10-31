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
    if task == "echo_tracks":
        return model, {k: inputs[k] for k in ("flows", "visconf")}, {"oracle": True}
    raise ValueError(f"unknown task {task!r}")


if __name__ == "__main__":
    main(handle)
