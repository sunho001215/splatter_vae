"""Verify that CUDA work and MuJoCo EGL rendering land only on the pinned GPU.

For every allowed UUID the script starts a child pinned to that UUID. The child
allocates a CUDA tensor and renders one Meta-World frame through EGL, then
waits. While it is alive the parent reads ``nvidia-smi`` (compute apps and the
full process table, because EGL contexts show up as graphics ``G`` entries) and
asserts the child's PID appears only on the intended GPU. The script also
checks that ``CUDA_VISIBLE_DEVICES=0`` and an unset variable are rejected.

Run from the repository root (no GPU is needed by the parent itself):
    python scripts/check_gpu_isolation.py
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from s4d.gpu_guard import ALLOWED_GPU_UUIDS, enforce_allowed_gpus  # noqa: E402

PARENT_MAPPING = enforce_allowed_gpus()

CHILD = r"""
import os, sys, time, json
sys.path.insert(0, %(repo)r)
from s4d.gpu_guard import enforce_allowed_gpus, enforce_mujoco_egl_device
mapping = enforce_allowed_gpus()
import mujoco_mig_setup  # noqa: F401  (must precede mujoco / metaworld)
egl_dev = enforce_mujoco_egl_device()
import torch
x = torch.randn(1024, 1024, device="cuda") @ torch.randn(1024, 1024, device="cuda")
torch.cuda.synchronize()
import gymnasium as gym
import metaworld  # noqa: F401
import mujoco
env = gym.make("Meta-World/MT1", env_name="hammer-v3", seed=0)
env.reset(seed=0)
model, data = env.unwrapped.model, env.unwrapped.data
renderer = mujoco.Renderer(model, height=128, width=128)
cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.lookat[:] = (0.0, 0.6, 0.0); cam.distance = 1.0; cam.azimuth = 0.0; cam.elevation = -45.0
renderer.update_scene(data, camera=cam)
frame = renderer.render()
with open(%(ready)r, "w") as f:
    json.dump({"pid": os.getpid(), "mapping": mapping, "egl_device": egl_dev,
               "frame_mean": float(frame.mean()), "matmul": float(x.float().mean())}, f)
time.sleep(%(hold)d)
"""


def gpu_index_to_uuid() -> dict[str, str]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    table = {}
    for line in out.splitlines():
        idx, uuid = [p.strip() for p in line.split(",")]
        table[idx] = uuid
    return table


def pid_gpus(pid: int, index_to_uuid: dict[str, str]) -> set[str]:
    """UUIDs of every GPU on which ``pid`` has a compute or graphics context."""
    found: set[str] = set()
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    for line in apps.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0] == str(pid):
            found.add(parts[1])
    table = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True).stdout
    in_processes = False
    for line in table.splitlines():
        if "Processes:" in line:
            in_processes = True
            continue
        if not in_processes:
            continue
        m = re.match(r"\|\s+(\d+)\s+\S+\s+\S+\s+(\d+)\s+([CG+]+)\s", line)
        if m and int(m.group(2)) == pid:
            found.add(index_to_uuid[m.group(1)])
    return found


def run_child(env_value: str | None, hold: int, ready: Path) -> subprocess.Popen:
    env = dict(os.environ)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    if env_value is not None:
        env["CUDA_VISIBLE_DEVICES"] = env_value
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    code = CHILD % {"repo": str(REPO), "ready": str(ready), "hold": hold}
    return subprocess.Popen(
        [sys.executable, "-B", "-c", code], env=env, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )


def main() -> int:
    index_to_uuid = gpu_index_to_uuid()
    results = []
    ok = True
    with tempfile.TemporaryDirectory(dir=REPO / "docs", prefix="isolation-") as tmp:
        for uuid in ALLOWED_GPU_UUIDS:
            ready = Path(tmp) / f"{uuid}.json"
            proc = run_child(uuid, hold=20, ready=ready)
            deadline = time.time() + 300
            while not ready.exists() and proc.poll() is None and time.time() < deadline:
                time.sleep(0.5)
            if not ready.exists():
                out = proc.communicate()[0]
                print(f"[FAIL] child for {uuid} never became ready:\n{out}")
                ok = False
                continue
            seen = pid_gpus(proc.pid, index_to_uuid)
            table = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True).stdout
            compute = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
            info = json.loads(ready.read_text())
            proc.terminate()
            out = proc.communicate()[0]
            passed = seen == {uuid}
            ok &= passed
            results.append(
                {
                    "uuid": uuid,
                    "pid": proc.pid,
                    "gpus_seen": sorted(seen),
                    "passed": passed,
                    "child": info,
                    "full_process_table": table,
                    "compute_table": compute,
                }
            )
            print(f"[{'PASS' if passed else 'FAIL'}] {uuid}: pid {proc.pid} seen on {sorted(seen)}")
            print(out.strip())
        for bad in ("0", None):
            ready = Path(tmp) / "bad.json"
            proc = run_child(bad, hold=1, ready=ready)
            out = proc.communicate(timeout=300)[0]
            rejected = proc.returncode != 0 and not ready.exists()
            ok &= rejected
            label = "unset" if bad is None else repr(bad)
            results.append({"cuda_visible_devices": label, "rejected": rejected})
            print(f"[{'PASS' if rejected else 'FAIL'}] CUDA_VISIBLE_DEVICES={label} rejected={rejected}")
            print("\n".join(out.strip().splitlines()[-3:]))
    report = {"passed": ok, "parent_mapping": PARENT_MAPPING, "results": results}
    (REPO / "docs" / "gpu_isolation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
