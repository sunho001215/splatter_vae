"""Fail-closed GPU isolation guard.

Every entry point calls :func:`enforce_allowed_gpus` before importing torch,
gsplat, or MuJoCo. Devices are always selected by UUID; bare integers are
rejected because their meaning differs between nvidia-smi, CUDA, and EGL.
"""

from __future__ import annotations

import os
import sys

ALLOWED_GPU_UUIDS = (
    "GPU-76871c16-ff1d-f7b1-cdf5-fabbaf9df8ce",  # physical GPU 4
    "GPU-d09f0338-71b9-d915-3c7f-e99754a3b639",  # physical GPU 5
)


class GPUIsolationError(RuntimeError):
    """Raised when the process could touch a GPU outside the allowed set."""


def _normalize_uuid(value: str) -> str:
    value = str(value).strip().lower()
    return value[4:] if value.startswith("gpu-") else value


def _die(message: str) -> None:
    sys.stderr.write(f"[gpu_guard] FATAL: {message}\n")
    sys.stderr.flush()
    raise GPUIsolationError(message)


def visible_device_uuids() -> list[str]:
    """Return the entries of CUDA_VISIBLE_DEVICES after validating them."""
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None or not raw.strip():
        _die("CUDA_VISIBLE_DEVICES is unset or empty; refusing to run.")
    entries = [item.strip() for item in raw.split(",")]
    if any(not item for item in entries):
        _die(f"CUDA_VISIBLE_DEVICES={raw!r} contains an empty entry.")
    allowed = {_normalize_uuid(u) for u in ALLOWED_GPU_UUIDS}
    for item in entries:
        if not item.startswith("GPU-") or _normalize_uuid(item) not in allowed:
            _die(f"CUDA_VISIBLE_DEVICES entry {item!r} is not an allowed GPU UUID. Allowed: {list(ALLOWED_GPU_UUIDS)}.")
    if len({_normalize_uuid(item) for item in entries}) != len(entries):
        _die(f"CUDA_VISIBLE_DEVICES={raw!r} lists a device twice.")
    return entries


def enforce_allowed_gpus(verbose: bool = True) -> list[dict]:
    """Validate the environment, then import torch and validate the visible devices.

    Returns the resolved mapping ``[{"index", "uuid", "name"}, ...]``.
    """
    entries = visible_device_uuids()
    import torch  # noqa: PLC0415  (deliberately imported after the env check)

    count = torch.cuda.device_count()
    if count != len(entries):
        _die(f"torch sees {count} CUDA devices but CUDA_VISIBLE_DEVICES lists {len(entries)}.")
    if count == 0:
        _die("torch.cuda.device_count() is 0; CUDA is not available in this process.")
    allowed = {_normalize_uuid(u) for u in ALLOWED_GPU_UUIDS}
    mapping = []
    for index in range(count):
        props = torch.cuda.get_device_properties(index)
        uuid = _normalize_uuid(str(props.uuid))
        if uuid not in allowed:
            _die(f"Visible CUDA device {index} has UUID {uuid} which is not allowed.")
        if uuid != _normalize_uuid(entries[index]):
            _die(f"Visible CUDA device {index} is {uuid} but CUDA_VISIBLE_DEVICES[{index}] is {entries[index]}.")
        mapping.append({"index": index, "uuid": f"GPU-{uuid}", "name": props.name})
    if verbose:
        for item in mapping:
            print(f"[gpu_guard] cuda:{item['index']} -> {item['uuid']} ({item['name']})")
    return mapping


def enforce_mujoco_egl_device() -> int:
    """Confirm MuJoCo's EGL device is the one that maps to visible CUDA ordinal 0.

    Must be called right after ``import mujoco_mig_setup`` (which selects the
    device) and before any MuJoCo rendering. Reuses the ctypes helpers of
    ``mujoco_mig_setup`` to query ``EGL_CUDA_DEVICE_NV``.
    """
    import ctypes  # noqa: PLC0415

    import mujoco_mig_setup as mms  # noqa: PLC0415

    if len(visible_device_uuids()) != 1:
        _die("MuJoCo rendering requires exactly one allowed UUID in CUDA_VISIBLE_DEVICES.")
    selected = os.environ.get("MUJOCO_EGL_DEVICE_ID")
    if selected is None or not str(selected).strip().isdigit():
        _die("MUJOCO_EGL_DEVICE_ID is not set; mujoco_mig_setup could not map the GPU.")
    selected_idx = int(selected)
    devices = mms._query_egl_devices_ctypes()
    if not 0 <= selected_idx < len(devices):
        _die(f"MUJOCO_EGL_DEVICE_ID={selected_idx} is outside the {len(devices)} EGL devices.")
    egl = mms._load_egl()
    query = mms._get_egl_ext_function(
        egl,
        b"eglQueryDeviceAttribEXT",
        ctypes.c_uint32,
        [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_long)],
    )
    cuda_idx = ctypes.c_long(-1)
    ok = query(devices[selected_idx], mms.EGL_CUDA_DEVICE_NV, ctypes.byref(cuda_idx))
    if not ok or cuda_idx.value != 0:
        _die(
            f"EGL device {selected_idx} reports CUDA ordinal {cuda_idx.value if ok else 'n/a'}, expected visible ordinal 0."
        )
    import mujoco.egl as mujoco_egl  # noqa: PLC0415

    if mujoco_egl.create_initialized_egl_device_display is not mms.create_initialized_egl_device_display_full:
        _die("mujoco.egl was not patched by mujoco_mig_setup; import order is wrong.")
    print(f"[gpu_guard] MuJoCo EGL device {selected_idx} -> visible CUDA ordinal 0")
    return selected_idx
