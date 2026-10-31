"""Shared request/response helpers for worker scripts. Standard library + numpy only (importable from any env)."""

from __future__ import annotations

import argparse
import json
import subprocess
import traceback
from pathlib import Path

import numpy as np

PROTOCOL = "s4d.realworld.worker/1"


def git_commit(repo_dir: str | None) -> str | None:
    if not repo_dir:
        return None
    try:
        return subprocess.run(["git", "-C", repo_dir, "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def sha256(path: str | None) -> str | None:
    if not path or not Path(path).is_file():
        return None
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main(handler) -> None:
    """``handler(request, inputs) -> (model_identity, outputs, meta)``; always writes a response."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--request", required=True)
    ap.add_argument("--response", required=True)
    args = ap.parse_args()
    request = json.loads(Path(args.request).read_text())
    workdir = Path(args.request).parent
    response = {"protocol": PROTOCOL, "ok": False, "error": None, "model": None, "outputs": request["outputs"],
                "meta": {}}
    try:
        if request.get("protocol") != PROTOCOL:
            raise ValueError(f"protocol {request.get('protocol')!r}")
        with np.load(workdir / request["inputs"]) as data:
            inputs = {k: data[k] for k in data.files}
        model, outputs, meta = handler(request, inputs)
        np.savez(workdir / request["outputs"], **outputs)
        response.update(ok=True, model=model, meta=meta)
    except Exception as exc:  # reported, never swallowed
        response["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    tmp = Path(args.response).with_suffix(".tmp")
    tmp.write_text(json.dumps(response, indent=2))
    tmp.replace(args.response)
