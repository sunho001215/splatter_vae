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

from ops.resources.mig_inventory import INVENTORY_RELATIVE, REMOTE_PHYSICAL_GPUS, InventoryError, load_allowlist

REPO = Path(__file__).resolve().parents[2]


def _remote_mig_allowlist(path):
    """(instances or None, error): None = no MIG allowlist file (the physical GPUs stay authorized); an unreadable
    or invalid file authorizes NO remote device (fail closed), with the reason in REMOTE_MIG_ERROR."""
    try:
        return load_allowlist(path), None
    except InventoryError as exc:
        return (), str(exc)
    except Exception as exc:  # noqa: BLE001 - any malformed content: no remote device, never an import failure
        return (), f"{type(exc).__name__}: {exc}"


# Remote MIG (user directive 2026-11-11): configs/hosts/remote-mig.yaml, written from a verified discovery report by
# ops/resources/mig_inventory.py render and committed with the release that is verified on exactly those instances.
REMOTE_MIG_ALLOWLIST = REPO / INVENTORY_RELATIVE
REMOTE_MIG_INSTANCES, REMOTE_MIG_ERROR = _remote_mig_allowlist(REMOTE_MIG_ALLOWLIST)
REMOTE_MIG_ACTIVE = REMOTE_MIG_INSTANCES is not None
# The parents of the remote instances: never a launch target while MIG is active (a physical UUID of a MIG-enabled
# GPU gives CUDA no device; launch.py refuses it explicitly).
REMOTE_MIG_PARENTS = tuple(REMOTE_PHYSICAL_GPUS) if REMOTE_MIG_ACTIVE else ()
REMOTE_DEVICES = tuple(REMOTE_PHYSICAL_GPUS) if not REMOTE_MIG_ACTIVE else tuple(
    item["uuid"] for item in REMOTE_MIG_INSTANCES)
# remote MIG UUID -> (parent GPU UUID, GPU instance ID, MIG device index on the parent), as MIG_DEVICES for local
REMOTE_MIG_DEVICES = MappingProxyType({item["uuid"]: (item["parent"], item["gpu_instance_id"], item["device"])
                                      for item in (REMOTE_MIG_INSTANCES or ())})

APPROVED_HOST_GPUS = MappingProxyType(
    {
        # MIG 1g.24gb instances of local GPUs 4/5 (user, 2026-10-19; GPU 6 revoked), in nvidia-smi -L order.
        "local": (
            "MIG-8da55219-bd6b-5260-8be1-3b2a76ac733c",
            "MIG-c4ec2a95-30cc-5398-82bf-d9b11787fc70",
            "MIG-423203d9-5801-5892-a489-30f16a5a2d4a",
            "MIG-02b55a66-ed41-53c2-9004-c3f5619d6369",
            "MIG-05a151e7-c6bd-5344-91cb-8bfb3a342631",
            "MIG-b50e85c3-8498-5ef0-906e-5eda9db34b90",
            "MIG-2ebd9e11-f2e0-5e22-870a-7779d44ec10a",
            "MIG-c3511b41-5f78-52f1-9b93-9c6794cdbbdd",
        ),
        # the four physical GPUs, or their verified MIG instances once configs/hosts/remote-mig.yaml exists
        "remote": REMOTE_DEVICES,
        # Real-world inference server (user, 2026-11-04): the one designated RTX 5090, inside the RAM-only inference
        # container (ops/inference_server). Preprocessing, inference and bounded tests only - never a training-pool host.
        "inference": ("GPU-9bcf38b6-8240-e910-6548-ca715643dc11",),
    }
)
LOCAL_MIG_PARENTS = (
    "GPU-76871c16-ff1d-f7b1-cdf5-fabbaf9df8ce",
    "GPU-d09f0338-71b9-d915-3c7f-e99754a3b639",
)
# MIG UUID -> (parent GPU UUID, GPU instance ID, MIG device index on the parent). CUDA reports the parent GPU's
# UUID for a MIG device, and EGL lists one device per MIG instance in this order, all mapping to CUDA ordinal 0.
MIG_DEVICES = MappingProxyType(
    {
        uuid: (LOCAL_MIG_PARENTS[index // 4], 3 + index % 4, index % 4)
        for index, uuid in enumerate(APPROVED_HOST_GPUS["local"])
    }
)
APPROVED_RUNTIME_ROOTS = MappingProxyType({"local": None, "remote": Path("/home/compu/kaist/sunho"),
                                           "inference": Path("/ram/s4d-runtime")})  # RAM tmpfs in the container


class GPUIsolationError(RuntimeError):
    """Raised when the process could touch a GPU outside the allowed set."""


def _normalize_uuid(value: str) -> str:
    value = str(value).strip().lower()
    return value[4:] if value.startswith("gpu-") else value


def cuda_uuid(entry: str) -> str:
    """The UUID CUDA reports for a CUDA_VISIBLE_DEVICES entry: a MIG device reports its parent GPU's UUID."""
    if entry in MIG_DEVICES:
        return MIG_DEVICES[entry][0]
    return REMOTE_MIG_DEVICES[entry][0] if entry in REMOTE_MIG_DEVICES else entry


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
        _die("Host config name must be exactly local, remote or inference.")
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
        if not item.startswith(("GPU-", "MIG-")) or _normalize_uuid(item) not in allowed:
            _die(f"CUDA_VISIBLE_DEVICES entry {item!r} is not an allowed GPU UUID. Allowed: {list(ALLOWED_GPU_UUIDS)}.")
    if len({_normalize_uuid(item) for item in entries}) != len(entries):
        _die(f"CUDA_VISIBLE_DEVICES={raw!r} lists a device twice.")
    if len(entries) > 1 and any(item in MIG_DEVICES or item in REMOTE_MIG_DEVICES for item in entries):
        _die(f"CUDA_VISIBLE_DEVICES={raw!r}: CUDA exposes only one MIG device per process.")
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
    allowed = {_normalize_uuid(cuda_uuid(u)) for u in ALLOWED_GPU_UUIDS}
    mapping = []
    for index in range(count):
        props = torch.cuda.get_device_properties(index)
        uuid = _normalize_uuid(str(props.uuid))
        if uuid not in allowed:
            _die(f"Visible CUDA device {index} has UUID {uuid} which is not allowed.")
        if uuid != _normalize_uuid(cuda_uuid(entries[index])):
            _die(f"Visible CUDA device {index} is {uuid} but CUDA_VISIBLE_DEVICES[{index}] is {entries[index]}.")
        mig = entries[index] in MIG_DEVICES or entries[index] in REMOTE_MIG_DEVICES
        mapping.append({"index": index, "uuid": entries[index] if mig else f"GPU-{uuid}",
                        "name": props.name})
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
