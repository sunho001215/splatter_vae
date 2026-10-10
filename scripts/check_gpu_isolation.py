"""Verify pinned native CUDA/EGL work; Docker requires --pid=host."""

from __future__ import annotations

import argparse
import json
import os
import re
import select
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))

from s4d.gpu_guard import ALLOWED_GPU_UUIDS, APPROVED_HOST_GPUS, HOST_NAME, validate_runtime_path  # noqa: E402

CHILD = r"""
import os, sys, json
sys.path.insert(0, %(repo)r)
sys.path.insert(0, %(scripts)r)
import _bootstrap
from s4d.gpu_guard import enforce_allowed_gpus, enforce_mujoco_egl_device
mapping = enforce_allowed_gpus()
import mujoco_mig_setup
egl_dev = enforce_mujoco_egl_device()
import mujoco
import metaworld
import torch
import gymnasium as gym
x = torch.randn(1024, 1024, device="cuda") @ torch.randn(1024, 1024, device="cuda")
torch.cuda.synchronize()
env = gym.make("Meta-World/MT1", env_name="hammer-v3", seed=0)
env.reset(seed=0)
renderer = mujoco.Renderer(env.unwrapped.model, height=128, width=128)
cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.lookat[:] = (0.0, 0.6, 0.0); cam.distance = 1.0; cam.azimuth = 0.0; cam.elevation = -45.0
renderer.update_scene(env.unwrapped.data, camera=cam)
frame = renderer.render()
info = {"pid": os.getpid(), "mapping": mapping, "egl_device": egl_dev,
        "frame_mean": float(frame.mean()), "matmul": float(x.float().mean()),
        "native_cuda_passed": True, "native_egl_passed": True}
os.write(%(ready_fd)d, json.dumps(info).encode() + b"\n")
os.close(%(ready_fd)d)
sys.stdin.read(1)
renderer.close()
env.close()
"""


def pid_namespace_evidence(host_pid_namespace: bool) -> dict:
    container = Path("/.dockerenv").exists()
    if (container or HOST_NAME == "remote") and not host_pid_namespace:
        raise RuntimeError("Docker isolation validation requires --pid=host and --host-pid-namespace.")
    status = Path("/proc/self/status").read_text()
    nspid = next((list(map(int, line.split()[1:])) for line in status.splitlines() if line.startswith("NSpid:")), [])
    if len(nspid) != 1 or nspid[0] != os.getpid():
        raise RuntimeError("Cannot match NVIDIA host PIDs from a nested or unverified PID namespace.")
    return {
        "container": container,
        "host_pid_namespace_asserted": host_pid_namespace,
        "pid_namespace": os.readlink("/proc/self/ns/pid"),
        "nspid": nspid,
    }


def gpu_index_to_uuid() -> dict[str, str]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        capture_output=True, text=True, check=True, timeout=60,
    ).stdout
    return {idx.strip(): uuid.strip() for idx, uuid in (line.split(",") for line in out.splitlines())}


def pid_gpus(pid: int, index_to_uuid: dict[str, str]) -> tuple[set[str], str, str]:
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
        capture_output=True, text=True, check=True, timeout=60,
    ).stdout
    found = set()
    for line in apps.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0] == str(pid):
            found.add(parts[1])
    table = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True, timeout=60).stdout
    in_processes = False
    for line in table.splitlines():
        if "Processes:" in line:
            in_processes = True
            continue
        if in_processes:
            match = re.match(r"\|\s+(\d+)\s+\S+\s+\S+\s+(\d+)\s+([CG+]+)\s", line)
            if match and int(match.group(2)) == pid:
                found.add(index_to_uuid[match.group(1)])
    return found, table, apps


def run_child(env_value: str | None) -> tuple[subprocess.Popen, int]:
    env = dict(os.environ)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.pop("MUJOCO_EGL_DEVICE_ID", None)
    if env_value is not None:
        env["CUDA_VISIBLE_DEVICES"] = env_value
    env["MUJOCO_GL"] = "egl"
    env["PYOPENGL_PLATFORM"] = "egl"
    read_fd, write_fd = os.pipe()
    code = CHILD % {"repo": str(REPO), "scripts": str(REPO / "scripts"), "ready_fd": write_fd}
    try:
        proc = subprocess.Popen(
            [sys.executable, "-I", "-B", "-c", code], env=env, cwd=REPO,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            pass_fds=(write_fd,), text=True,
        )
    except BaseException:
        os.close(read_fd)
        raise
    finally:
        os.close(write_fd)
    return proc, read_fd


def stop_child(proc: subprocess.Popen) -> str:
    if proc.poll() is None:
        proc.terminate()
    try:
        return proc.communicate(timeout=30)[0]
    except subprocess.TimeoutExpired:
        proc.kill()
        return proc.communicate(timeout=30)[0]


def check_uuid(uuid: str, index_to_uuid: dict[str, str]) -> dict:
    proc, read_fd = run_child(uuid)
    result = {"uuid": uuid, "pid": proc.pid, "passed": False}
    try:
        if not select.select([read_fd], [], [], 300)[0]:
            raise RuntimeError("Child readiness timed out.")
        raw = os.read(read_fd, 65536)
        if not raw:
            raise RuntimeError("Child exited before native CUDA/EGL readiness.")
        info = json.loads(raw)
        if info["pid"] != proc.pid or proc.poll() is not None:
            raise RuntimeError("Child PID or lifetime does not match its readiness record.")
        seen, table, apps = pid_gpus(proc.pid, index_to_uuid)
        result.update(
            gpus_seen=sorted(seen), passed=seen == {uuid}, child=info,
            full_process_table=table, compute_table=apps,
        )
    except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        os.close(read_fd)
        result["child_log"] = stop_child(proc)
    return result


def check_rejected(value: str | None) -> dict:
    proc, read_fd = run_child(value)
    ready = b""
    try:
        out = proc.communicate(timeout=300)[0]
        ready = os.read(read_fd, 65536)
        rejected = proc.returncode != 0 and not ready and "[gpu_guard] FATAL:" in out
    except subprocess.TimeoutExpired:
        out = stop_child(proc)
        rejected = False
    finally:
        os.close(read_fd)
    return {"cuda_visible_devices": value, "rejected": rejected, "child_log": out}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        epilog="Verify docker inspect HostConfig.PidMode=host before accepting Docker isolation evidence.",
    )
    ap.add_argument("--gpu", choices=ALLOWED_GPU_UUIDS)
    ap.add_argument("--output-dir", default=os.environ.get("S4D_TEST_EVIDENCE_DIR", str(REPO / "docs")))
    ap.add_argument("--host-pid-namespace", action="store_true")
    args = ap.parse_args()
    output = validate_runtime_path(Path(args.output_dir))
    if REPO not in output.parents:
        os.environ.setdefault("S4D_CACHE_ROOT", str(output / "cache"))
    from _bootstrap import guard_gpus, source_fingerprints

    report = {"passed": False, "host": HOST_NAME, "source_sha256": source_fingerprints(), "results": []}
    output.mkdir(parents=True, exist_ok=True)
    try:
        report["pid_matching"] = pid_namespace_evidence(args.host_pid_namespace)
        report["parent_mapping"] = guard_gpus()
        if args.gpu and [item["uuid"] for item in report["parent_mapping"]] != [args.gpu]:
            raise RuntimeError("--gpu requires that same single UUID in CUDA_VISIBLE_DEVICES.")
        index_to_uuid = gpu_index_to_uuid()
        for uuid in (args.gpu,) if args.gpu else ALLOWED_GPU_UUIDS:
            result = check_uuid(uuid, index_to_uuid)
            report["results"].append(result)
            print(f"[{'PASS' if result['passed'] else 'FAIL'}] {uuid}: {result.get('gpus_seen', result.get('error'))}")
        other_host = "remote" if HOST_NAME == "local" else "local"
        for bad in ("0", None, APPROVED_HOST_GPUS[other_host][0]):
            result = check_rejected(bad)
            report["results"].append(result)
            print(f"[{'PASS' if result['rejected'] else 'FAIL'}] CUDA_VISIBLE_DEVICES={bad!r} rejected")
        report["passed"] = all(item.get("passed", item.get("rejected", False)) for item in report["results"])
    except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    report["source_changed_during_check"] = source_fingerprints() != report["source_sha256"]
    if report["source_changed_during_check"]:
        report["passed"] = False
        report["error"] = "Sources changed during isolation verification."
    (output / "gpu_isolation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
