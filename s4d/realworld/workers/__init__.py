"""Isolated upstream-model workers: a file-based subprocess protocol.

Upstream models (FoundationStereo, AllTracker, SAM 3, DINO) are never imported into this package. Each runs as
``<python of its own pinned environment> s4d/realworld/workers/<name>_worker.py --request req.json``; inputs and
outputs travel as ``.npz`` files next to the request. The response must name the model identity (repo commit,
checkpoint + sha256, ``is_mock``), which every stage copies into its completion record. Commit mismatches against
``configs/realworld/upstream_pins.yaml`` are errors.

Protocol ``s4d.realworld.worker/1``::

    request.json  {"protocol", "task", "params", "inputs": "inputs.npz", "outputs": "outputs.npz",
                   "repo_dir", "checkpoint", "expected_commit"}
    response.json {"protocol", "ok", "error", "model": {name, repo, commit, checkpoint, checkpoint_sha256,
                   is_mock}, "outputs": "outputs.npz", "meta": {...}}
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

PROTOCOL = "s4d.realworld.worker/1"
WORKER_DIR = Path(__file__).resolve().parent  # this package directory holds the worker scripts (as before the move)


class WorkerError(RuntimeError):
    pass


@dataclass(frozen=True)
class WorkerSpec:
    name: str  # foundation_stereo / alltracker / sam3 / dino_snerl / mock
    python: str  # interpreter of the isolated, pinned environment
    script: str | None = None  # defaults to s4d/realworld/workers/<name>_worker.py
    repo_dir: str | None = None  # upstream checkout at the pinned commit
    expected_commit: str | None = None
    checkpoint: str | None = None
    env: dict = field(default_factory=dict)
    timeout_s: float = 3600.0

    def script_path(self) -> Path:
        return Path(self.script) if self.script else WORKER_DIR / f"{self.name}_worker.py"


def call_worker(spec: WorkerSpec, task: str, params: dict, arrays: dict, workdir: Path,
                allow_mock: bool = False) -> tuple[dict, dict]:
    """Run one request; returns (response, output arrays). Raises WorkerError on any protocol/identity problem."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    np.savez(workdir / "inputs.npz", **arrays)
    request = {"protocol": PROTOCOL, "task": task, "params": params, "inputs": "inputs.npz",
               "outputs": "outputs.npz", "repo_dir": spec.repo_dir, "checkpoint": spec.checkpoint,
               "expected_commit": spec.expected_commit}
    (workdir / "request.json").write_text(json.dumps(request, indent=2))
    response_path = workdir / "response.json"
    if response_path.exists():
        response_path.unlink()
    env = {k: v for k, v in os.environ.items() if k in ("PATH", "HOME", "CUDA_VISIBLE_DEVICES", "LANG", "TMPDIR")}
    env.update({k: str(v) for k, v in spec.env.items()})
    proc = subprocess.run([spec.python, "-I", str(spec.script_path()), "--request", str(workdir / "request.json"),
                           "--response", str(response_path)], cwd=workdir, env=env, capture_output=True, text=True,
                          timeout=spec.timeout_s)
    (workdir / "worker.log").write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr)
    if not response_path.is_file():
        raise WorkerError(f"{spec.name}: no response (exit {proc.returncode}); see {workdir / 'worker.log'}")
    response = json.loads(response_path.read_text())
    if response.get("protocol") != PROTOCOL:
        raise WorkerError(f"{spec.name}: protocol {response.get('protocol')!r} != {PROTOCOL!r}")
    if not response.get("ok"):
        raise WorkerError(f"{spec.name}: {response.get('error')}")
    model = response.get("model") or {}
    if model.get("is_mock") and not allow_mock:
        raise WorkerError(f"{spec.name}: mock output refused (allow_mock=False)")
    if spec.expected_commit and model.get("commit") != spec.expected_commit:
        raise WorkerError(f"{spec.name}: upstream commit {model.get('commit')} != pinned {spec.expected_commit}")
    with np.load(workdir / response["outputs"]) as data:
        outputs = {k: data[k] for k in data.files}
    response["spec"] = asdict(spec)
    return response, outputs
