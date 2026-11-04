"""Shared file/stream worker helpers. Standard library + numpy only, with no s4d imports."""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import os
import re
import resource
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import BinaryIO

import numpy as np

if __package__:
    from ._framing import MAX_PAYLOAD, FrameError, pack_arrays, read_frame, unpack_arrays, write_frame
else:
    from _framing import MAX_PAYLOAD, FrameError, pack_arrays, read_frame, unpack_arrays, write_frame

PROTOCOL = "s4d.realworld.worker/1"


def git_commit(repo_dir: str | None) -> str | None:
    """Resolve HEAD using git, or loose/packed refs (including linked worktrees)."""
    if not repo_dir:
        return None
    try:
        commit = subprocess.run(["git", "-C", repo_dir, "rev-parse", "HEAD"], capture_output=True, text=True,
                                check=True).stdout.strip()
        if re.fullmatch(r"[0-9a-fA-F]{40}", commit):
            return commit.lower()
    except (OSError, subprocess.CalledProcessError):
        pass
    try:
        gitdir = Path(repo_dir) / ".git"
        if gitdir.is_file():
            prefix, target = gitdir.read_text().strip().split(":", 1)
            if prefix != "gitdir":
                return None
            gitdir = (gitdir.parent / target.strip()).resolve()
        roots = [gitdir]
        if (gitdir / "commondir").is_file():
            roots.append((gitdir / (gitdir / "commondir").read_text().strip()).resolve())
        head = (gitdir / "HEAD").read_text().strip()
        if re.fullmatch(r"[0-9a-fA-F]{40}", head):
            return head.lower()
        if not head.startswith("ref: refs/"):
            return None
        ref = head[5:].strip()
        if ".." in Path(ref).parts:
            return None
        for root in roots:
            loose = root / ref
            if loose.is_file():
                commit = loose.read_text().strip()
                if re.fullmatch(r"[0-9a-fA-F]{40}", commit):
                    return commit.lower()
            packed = root / "packed-refs"
            if packed.is_file():
                for line in packed.read_text().splitlines():
                    parts = line.split()
                    if len(parts) == 2 and parts[1] == ref and re.fullmatch(r"[0-9a-fA-F]{40}", parts[0]):
                        return parts[0].lower()
    except (OSError, ValueError, UnicodeError):
        pass
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


def json_safe(value):
    """Strict-JSON copy: non-finite floats become the strings "nan" / "inf" / "-inf" (headers are written with
    ``allow_nan=False``; a diagnostic such as an infinite covariance must not kill the session), numpy scalars and
    arrays become Python values."""
    import math

    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")
    return value


def resources() -> dict:
    """Peak RSS and cgroup/CUDA memory in bytes; never import torch for telemetry."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = {"ru_maxrss": int(rss if sys.platform == "darwin" else rss * 1024)}
    cgroup = Path("/sys/fs/cgroup")
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            if line.startswith("0::"):
                candidate = cgroup / line[3:].lstrip("/")
                if candidate.is_dir():
                    cgroup = candidate
                break
    except OSError:
        pass
    memory = {}
    for name in ("memory.current", "memory.peak", "memory.max"):
        try:
            value = (cgroup / name).read_text().strip()
            memory[name] = int(value) if value != "max" else value
        except (OSError, ValueError):
            pass
    if memory:
        result["cgroup"] = memory
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            if torch.cuda.is_initialized():
                result["cuda"] = {"max_memory_allocated": int(torch.cuda.max_memory_allocated()),
                                  "max_memory_reserved": int(torch.cuda.max_memory_reserved()),
                                  "mem_get_info": list(torch.cuda.mem_get_info())}
        except (RuntimeError, AssertionError, OSError):
            pass
    return result


def _mount_for(path: str, mountinfo: str) -> dict:
    resolved = str(Path(path).resolve())
    best = {"path": resolved, "mountpoint": None, "filesystem": None, "ok": False}
    for line in mountinfo.splitlines():
        before, _, after = line.partition(" - ")
        fields, types = before.split(), after.split()
        if len(fields) < 5 or not types:
            continue
        mount = re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), fields[4])
        if resolved == mount or resolved.startswith(mount.rstrip("/") + "/"):
            if best["mountpoint"] is None or len(mount) >= len(best["mountpoint"]):
                best.update(mountpoint=mount, filesystem=types[0], ok=types[0] in ("tmpfs", "ramfs"))
    return best


def ram_backed() -> dict:
    """Fail closed unless cwd, an explicitly set TMPDIR and the effective temporary directory that libraries use
    (``tempfile.gettempdir()``, e.g. /tmp when TMPDIR is unset) all live on tmpfs/ramfs."""
    import tempfile

    try:
        mountinfo = Path("/proc/self/mountinfo").read_text()
        paths = {"cwd": _mount_for(os.getcwd(), mountinfo)}
        if os.environ.get("TMPDIR") is not None:
            tmpdir = os.environ["TMPDIR"]
            paths["TMPDIR"] = _mount_for(tmpdir, mountinfo)
            if not tmpdir or not Path(tmpdir).is_dir() or not os.access(tmpdir, os.W_OK):
                paths["TMPDIR"].update(ok=False, error="TMPDIR must be an existing writable directory")
        paths["gettempdir"] = _mount_for(tempfile.gettempdir(), mountinfo)
        return {"ok": all(item["ok"] for item in paths.values()),
                "override": os.environ.get("S4D_WORKER_ALLOW_DISK") == "1", **paths}
    except OSError as exc:
        return {"ok": False, "override": os.environ.get("S4D_WORKER_ALLOW_DISK") == "1", "error": str(exc)}


def _gpu() -> dict | None:
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return None
    count = torch.cuda.device_count()
    devices = []
    for index in range(count):
        props = torch.cuda.get_device_properties(index)
        devices.append({"name": props.name, "uuid": str(props.uuid), "total_memory": int(props.total_memory)})
    return {"device_count": count, "devices": devices}


def _source_commit() -> str | None:
    for root in Path(__file__).resolve().parents:
        if (root / ".git").exists():
            return git_commit(str(root))
    return None


def serve_stream(handler, instream: BinaryIO, outstream: BinaryIO, *, worker_name: str) -> int:
    """Serve one frame at a time, requiring an ACK before the next request.

    ``S4D_WORKER_MAX_PAYLOAD`` configures the inbound/outbound byte limit (default 2 GiB).
    """
    with contextlib.redirect_stdout(sys.stderr):
        return _serve_stream(handler, instream, outstream, worker_name=worker_name)


def _serve_stream(handler, instream: BinaryIO, outstream: BinaryIO, *, worker_name: str) -> int:
    ram = ram_backed()

    def error(message: str, **fields) -> None:
        write_frame(outstream, {"type": "error", "protocol": PROTOCOL, "ok": False, "error": message, **fields})

    if not ram["ok"] and not ram["override"]:
        error("stream worker requires RAM-backed cwd and TMPDIR", ram_backed=ram)
        return 4
    pending = None
    try:
        max_payload = int(os.environ.get("S4D_WORKER_MAX_PAYLOAD", MAX_PAYLOAD))
        if max_payload < 0:
            raise ValueError("negative payload limit")
        while True:
            header, payload = read_frame(instream, max_payload=max_payload)
            kind = header.get("type")
            if kind == "bye":
                return 0
            if pending is not None:
                if kind != "ack" or type(header.get("seq")) is not int or header["seq"] != pending or payload:
                    raise FrameError(f"expected ack for seq {pending}")
                pending = None
                continue
            if kind == "hello" and not payload:
                hello = {"type": "hello", "protocol": PROTOCOL, "transport": "stream/1", "worker": worker_name,
                         "pid": os.getpid(), "source_commit": _source_commit(), "ram_backed": ram}
                gpu = _gpu()
                if gpu is not None:
                    hello["gpu"] = gpu
                hello["resources"] = resources()
                write_frame(outstream, hello)
                continue
            if kind != "request" or type(header.get("seq")) is not int or not isinstance(header.get("chunk_id"), str):
                raise FrameError("expected request with integer seq and string chunk_id")
            started = time.monotonic()
            response = {"type": "response", "protocol": PROTOCOL, "seq": header["seq"],
                        "chunk_id": header["chunk_id"], "ok": False, "error": None, "model": None,
                        "outputs": "outputs.npz", "meta": {}}
            packed = b""
            try:
                request = {"protocol": PROTOCOL, "task": header["task"], "params": header["params"],
                           "inputs": "inputs.npz", "outputs": "outputs.npz", "repo_dir": header.get("repo_dir"),
                           "checkpoint": header.get("checkpoint"), "expected_commit": header.get("expected_commit")}
                inputs = unpack_arrays(payload)
                del payload
                model, outputs, meta = handler(request, inputs)
                packed = pack_arrays(outputs)
                if len(packed) > max_payload:
                    raise FrameError(f"output payload exceeds limit {max_payload}")
                response.update(ok=True, model=model, meta=dict(meta))
            except Exception as exc:  # reported, never swallowed; failed requests also need an ACK
                packed = b""
                response["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            response["meta"]["resources"] = resources()
            response["elapsed_s"] = time.monotonic() - started
            write_frame(outstream, json_safe(response), packed)
            pending = header["seq"]
            # Drop episode arrays before waiting for the caller's durable-save acknowledgement.
            payload = packed = b""
            inputs = outputs = None
    except EOFError:
        return 0
    except (FrameError, ValueError) as exc:
        error(str(exc))
        return 3


def main(handler) -> None:
    """``handler(request, inputs) -> (model_identity, outputs, meta)``; file or stream mode."""
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--request")
    mode.add_argument("--stream", action="store_true")
    ap.add_argument("--response")
    args = ap.parse_args()
    if args.stream:
        if args.response:
            ap.error("--stream cannot be combined with --response")
        outstream = sys.stdout.buffer
        worker_name = Path(sys.argv[0]).stem.removesuffix("_worker")
        raise SystemExit(serve_stream(handler, sys.stdin.buffer, outstream, worker_name=worker_name))
    if not args.response:
        ap.error("--request requires --response")
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
