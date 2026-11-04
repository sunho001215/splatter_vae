"""This repository's compute kernels (``s4d.realworld.kernels``) behind the worker protocol.

Runs in the repository environment (locally ``.venv``; in the inference container ``/opt/envs/s4d``), so CPU-heavy
pipeline compute (depth registration, calibration, lifting) can run next to the model workers. The identity
``commit`` is the checkout's HEAD: callers pin it with ``WorkerSpec.expected_commit`` so the server-side code is the
code that was reviewed locally.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))  # repository root (-I drops the script dir)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _protocol import git_commit, main  # noqa: E402


def handle(request, inputs):
    from s4d.realworld.kernels import run_kernel

    outputs, meta = run_kernel(request["task"], request["params"], inputs)
    identity = {"name": "s4d_kernels", "repo": "splatter4d", "commit": git_commit(str(REPO)), "checkpoint": None,
                "checkpoint_sha256": None, "is_mock": False}
    return identity, outputs, meta


if __name__ == "__main__":
    main(handle)
