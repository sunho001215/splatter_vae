"""Fail-closed GPU isolation guard.

Every entry point calls :func:`enforce_allowed_gpus` before importing torch,
gsplat, or MuJoCo. Devices are always selected by UUID; bare integers are
rejected because their meaning differs between nvidia-smi, CUDA, and EGL.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import MappingProxyType

import yaml

APPROVED_HOST_GPUS = MappingProxyType(
    {
        "local": (
            "GPU-76871c16-ff1d-f7b1-cdf5-fabbaf9df8ce",
            "GPU-d09f0338-71b9-d915-3c7f-e99754a3b639",
        ),
        "remote": (
            "GPU-4391fcee-537f-d408-d138-10d8a3866eea",
            "GPU-6122c7a9-eaa6-b539-6005-f20e339da4ea",
            "GPU-9e9f1e97-b2ca-04e0-eb2a-8035398daa79",
            "GPU-ca65e1b7-c0ac-6757-7d3f-816c3119a4c1",
        ),
    }
)
APPROVED_RUNTIME_ROOTS = MappingProxyType({"local": None, "remote": Path("/home/compu/kaist/sunho")})
REPO = Path(__file__).resolve().parents[1]


class GPUIsolationError(RuntimeError):
    """Raised when the process could touch a GPU outside the allowed set."""


def _normalize_uuid(value: str) -> str:
    value = str(value).strip().lower()
    return value[4:] if value.startswith("gpu-") else value


def _die(message: str) -> None:
    sys.stderr.write(f"[gpu_guard] FATAL: {message}\n")
    sys.stderr.flush()
    raise GPUIsolationError(message)


def load_host_config(path: Path) -> dict:
    path = Path(path)
    if not path.is_absolute():
        _die("S4D_HOST_CONFIG must name an absolute host-config path.")
    try:
        config = yaml.safe_load(path.read_text())
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        _die(f"Cannot read host config {path} ({type(exc).__name__}).")
    if not isinstance(config, dict):
        _die("Host config must be a mapping.")
    name = config.get("name")
    if not isinstance(name, str) or name not in APPROVED_HOST_GPUS:
        _die("Host config name must be exactly local or remote.")
    uuids = config.get("gpu_uuids")
    approved = APPROVED_HOST_GPUS[name]
    if (
        not isinstance(uuids, list)
        or any(not isinstance(uuid, str) for uuid in uuids)
        or len(uuids) != len(approved)
        or set(uuids) != set(approved)
    ):
        _die(f"Host config {name} must contain exactly its approved GPU UUID set.")
    root = config.get("runtime_root")
    approved_root = APPROVED_RUNTIME_ROOTS[name]
    if root != (str(approved_root) if approved_root is not None else None):
        _die(f"Host config {name} has an unauthorized runtime_root.")
    return {"name": name, "gpu_uuids": approved, "runtime_root": approved_root}


HOST_CONFIG_PATH = Path(os.environ.get("S4D_HOST_CONFIG", str(REPO / "configs/hosts/local.yaml")))
HOST_CONFIG = MappingProxyType(load_host_config(HOST_CONFIG_PATH))
HOST_NAME = HOST_CONFIG["name"]
ALLOWED_GPU_UUIDS = HOST_CONFIG["gpu_uuids"]
RUNTIME_ROOT = HOST_CONFIG["runtime_root"]


def validate_runtime_path(path: Path, *, repository: Path | None = None, cache: bool = False) -> Path:
    resolved = Path(path).expanduser().resolve()
    repository = REPO.resolve() if repository is None else Path(repository).resolve()
    if repository in resolved.parents:
        cache_root = repository / ".cache"
        if cache and resolved != cache_root and cache_root not in resolved.parents:
            _die("Repository caches must remain inside .cache.")
        return resolved
    if RUNTIME_ROOT is not None and RUNTIME_ROOT in resolved.parents:
        return resolved
    _die("Runtime output must be beneath the repository or the explicitly approved host runtime_root.")


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
