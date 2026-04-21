"""
MuJoCo MIG EGL Auto-Setup
=========================

Automatically maps CUDA_VISIBLE_DEVICES (MIG UUID) to the correct
MUJOCO_EGL_DEVICE_ID by querying nvidia-smi for the parent GPU,
then matching it against EGL device enumeration.

Usage:
    # Option 1: Import at the top of your script (before importing mujoco)
    import mujoco_mig_setup
    import mujoco  # now EGL will use the correct device

    # Option 2: Use as a command-line wrapper
    CUDA_VISIBLE_DEVICES=MIG-xxxxx python -c "import mujoco_mig_setup; import mujoco; ..."

    # Option 3: Call setup() explicitly
    from mujoco_mig_setup import setup
    setup()
    import mujoco

Requires:
    - CUDA_VISIBLE_DEVICES set to a MIG UUID (e.g., MIG-87f2fc44-...)
    - MUJOCO_GL=egl (set automatically if not already set)
    - nvidia-smi available in PATH
"""

import os
import re
import subprocess
import ctypes
from ctypes import byref, c_int, c_void_p, c_char_p, c_uint32
from typing import Optional

# ---------------------------------------------------------------------------
# EGL constants
# ---------------------------------------------------------------------------
EGL_NO_DISPLAY = c_void_p(0)
EGL_TRUE = 1
EGL_SUCCESS = 0x3000
EGL_PLATFORM_DEVICE_EXT = 0x313F
EGL_CUDA_DEVICE_NV = 0x323A

# ---------------------------------------------------------------------------
# Low-level EGL helpers (ctypes, no PyOpenGL dependency)
# ---------------------------------------------------------------------------

def _load_egl():
    try:
        egl = ctypes.CDLL("libEGL.so.1")
    except OSError as e:
        raise RuntimeError(f"Failed to load libEGL.so.1: {e}")
    egl.eglGetProcAddress.restype = c_void_p
    egl.eglGetProcAddress.argtypes = [c_char_p]
    egl.eglGetError.restype = c_int
    egl.eglGetError.argtypes = []
    egl.eglInitialize.restype = c_uint32
    egl.eglInitialize.argtypes = [c_void_p, c_void_p, c_void_p]
    return egl


def _get_egl_ext_function(egl, name: bytes, restype, argtypes):
    ptr = egl.eglGetProcAddress(name)
    if not ptr:
        raise RuntimeError(f"{name.decode()} not available from eglGetProcAddress")
    fn_type = ctypes.CFUNCTYPE(restype, *argtypes)
    return fn_type(ptr)


def _query_egl_devices_ctypes():
    """Returns a list of raw EGLDeviceEXT handles as c_void_p."""
    egl = _load_egl()
    eglQueryDevicesEXT = _get_egl_ext_function(
        egl,
        b"eglQueryDevicesEXT",
        c_uint32,
        [c_int, ctypes.POINTER(c_void_p), ctypes.POINTER(c_int)],
    )
    max_devices = 64
    devices = (c_void_p * max_devices)()
    num_devices = c_int(0)
    ok = eglQueryDevicesEXT(max_devices, devices, byref(num_devices))
    if not ok:
        err = egl.eglGetError()
        raise RuntimeError(f"eglQueryDevicesEXT failed, eglGetError=0x{err:04x}")
    return [devices[i] for i in range(num_devices.value)]


def create_initialized_egl_device_display_full():
    """
    Replacement for mujoco.egl.create_initialized_egl_device_display
    using ctypes-only full EGL device enumeration.
    """
    egl = _load_egl()
    eglGetPlatformDisplayEXT = _get_egl_ext_function(
        egl,
        b"eglGetPlatformDisplayEXT",
        c_void_p,
        [c_int, c_void_p, c_void_p],
    )

    all_devices = _query_egl_devices_ctypes()
    print(f"[mujoco_mig_setup] Full EGL device count = {len(all_devices)}")

    selected_device = os.environ.get("MUJOCO_EGL_DEVICE_ID", None)
    if selected_device is None:
        candidates = all_devices
    else:
        device_idx = int(selected_device)
        if not 0 <= device_idx < len(all_devices):
            raise RuntimeError(
                f"The MUJOCO_EGL_DEVICE_ID environment variable must be an integer "
                f"between 0 and {len(all_devices)-1} (inclusive), got {device_idx}."
            )
        candidates = all_devices[device_idx : device_idx + 1]

    for device in candidates:
        display = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, device, None)
        err = egl.eglGetError()
        if (
            display
            and int(ctypes.cast(display, ctypes.c_void_p).value or 0) != 0
            and err == EGL_SUCCESS
        ):
            initialized = egl.eglInitialize(display, None, None)
            err = egl.eglGetError()
            if initialized == EGL_TRUE and err == EGL_SUCCESS:
                return display

    return EGL_NO_DISPLAY


# ---------------------------------------------------------------------------
# nvidia-smi helpers: MIG UUID -> parent GPU index
# ---------------------------------------------------------------------------

def _parse_nvidia_smi_l():
    """
    Parse `nvidia-smi -L` and return:
      - MIG UUID -> parent GPU index
      - MIG UUID -> local MIG device index on that GPU
      - GPU index -> number of MIG devices on that GPU

    Example nvidia-smi -L output:
        GPU 0: NVIDIA RTX PRO 6000 ... (UUID: GPU-fe480a9b-...)
        GPU 2: NVIDIA RTX PRO 6000 ... (UUID: GPU-c952875c-...)
          MIG 2g.48gb  Device 0: (UUID: MIG-87f2fc44-...)
          MIG 2g.48gb  Device 1: (UUID: MIG-e07350af-...)
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True, text=True, timeout=10,
        )
        output = result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        raise RuntimeError(f"Failed to run nvidia-smi -L: {e}")

    mig_to_gpu = {}
    mig_to_local_device_idx = {}
    gpu_to_mig_count = {}
    current_gpu_idx = None

    for line in output.splitlines():
        # Match: GPU 0: ... (UUID: GPU-xxxx)
        gpu_match = re.match(r"^GPU\s+(\d+):", line)
        if gpu_match:
            current_gpu_idx = int(gpu_match.group(1))
            gpu_to_mig_count.setdefault(current_gpu_idx, 0)
            continue

        # Match short-form MIG UUIDs emitted by `nvidia-smi -L`.
        mig_match = re.search(
            r"MIG .* Device\s+(\d+):\s+\(UUID:\s+(MIG-[A-Za-z0-9\-]+)\)",
            line,
        )
        if mig_match and current_gpu_idx is not None:
            local_device_idx = int(mig_match.group(1))
            mig_uuid = mig_match.group(2)
            mig_to_gpu[mig_uuid] = current_gpu_idx
            mig_to_local_device_idx[mig_uuid] = local_device_idx
            gpu_to_mig_count[current_gpu_idx] = max(
                gpu_to_mig_count[current_gpu_idx], local_device_idx + 1
            )

    return mig_to_gpu, mig_to_local_device_idx, gpu_to_mig_count


def _get_gpu_uuid_to_index_map():
    """Build a dict mapping GPU UUID strings (GPU-...) to physical GPU indices."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        output = result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        raise RuntimeError(f"Failed to query GPU UUIDs via nvidia-smi: {e}")

    gpu_uuid_to_idx = {}
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit():
            gpu_uuid_to_idx[parts[1]] = int(parts[0])
    return gpu_uuid_to_idx


def _get_gpu_to_cuda_ordinal_map():
    """
    Build the CUDA ordinal that EGL reports for each physical GPU.

    On MIG-enabled systems, EGL_CUDA_DEVICE_NV aligns with the first CUDA
    ordinal owned by that physical GPU, not with the physical GPU index.
    """
    _, _, gpu_to_mig_count = _parse_nvidia_smi_l()
    gpu_to_cuda_ordinal = {}
    next_cuda_ordinal = 0

    for gpu_idx in sorted(gpu_to_mig_count):
        gpu_to_cuda_ordinal[gpu_idx] = next_cuda_ordinal
        mig_count = gpu_to_mig_count[gpu_idx]
        next_cuda_ordinal += mig_count if mig_count > 0 else 1

    return gpu_to_cuda_ordinal


def _get_parent_gpu_index(cuda_visible: str) -> int:
    """
    Given a CUDA_VISIBLE_DEVICES value (MIG UUID), return the parent GPU index.
    Supports formats: MIG-xxxx, MIG-GPU-xxxx/x/x, or just the UUID part.
    """
    # Normalize: take the first device if comma-separated
    device_id = cuda_visible.split(",")[0].strip()

    # If CUDA_VISIBLE_DEVICES uses the long-form MIG identifier,
    # e.g. MIG-GPU-<GPU-UUID>/<GI>/<CI>, resolve the parent GPU UUID directly.
    mig_gpu_match = re.match(r"^(MIG-(GPU-[^/]+))(?:/.*)?$", device_id)
    if mig_gpu_match:
        parent_gpu_uuid = mig_gpu_match.group(2)
        gpu_uuid_map = _get_gpu_uuid_to_index_map()
        if parent_gpu_uuid in gpu_uuid_map:
            return gpu_uuid_map[parent_gpu_uuid]
        raise RuntimeError(
            f"Could not resolve parent GPU UUID '{parent_gpu_uuid}' from MIG device '{device_id}'.\n"
            f"Known GPU UUIDs: {list(gpu_uuid_map.keys())}"
        )

    # If it looks like a short-form MIG UUID
    if device_id.startswith("MIG-"):
        mig_map, _, _ = _parse_nvidia_smi_l()
        if device_id in mig_map:
            return mig_map[device_id]
        # Try partial match (nvidia-smi sometimes uses short UUIDs)
        for uuid, gpu_idx in mig_map.items():
            if device_id in uuid or uuid in device_id:
                return gpu_idx
        raise RuntimeError(
            f"Could not find parent GPU for MIG UUID '{device_id}'.\n"
            f"Known MIG instances: {list(mig_map.keys())}"
        )

    # If it's a plain integer, just return it
    if device_id.isdigit():
        return int(device_id)

    # If it's a GPU UUID
    if device_id.startswith("GPU-"):
        try:
            gpu_uuid_map = _get_gpu_uuid_to_index_map()
            if device_id in gpu_uuid_map:
                return gpu_uuid_map[device_id]
        except RuntimeError:
            pass
        raise RuntimeError(f"Could not resolve GPU UUID '{device_id}' to an index.")

    raise RuntimeError(f"Unrecognized CUDA_VISIBLE_DEVICES format: '{device_id}'")


def _get_mig_local_device_index(cuda_visible: str) -> Optional[int]:
    """Return the local MIG device index when CUDA_VISIBLE_DEVICES names a short MIG UUID."""
    device_id = cuda_visible.split(",")[0].strip()
    if not device_id.startswith("MIG-") or device_id.startswith("MIG-GPU-"):
        return None

    _, mig_to_local_device_idx, _ = _parse_nvidia_smi_l()
    if device_id in mig_to_local_device_idx:
        return mig_to_local_device_idx[device_id]

    for uuid, local_idx in mig_to_local_device_idx.items():
        if device_id in uuid or uuid in device_id:
            return local_idx

    return None


def _find_egl_device_index_for_visible_cuda(
    visible_cuda_idx: int = 0, mig_local_device_idx: Optional[int] = None
) -> Optional[int]:
    """
    Resolve the EGL device from the CUDA ordinal visible inside the current
    process.

    When CUDA_VISIBLE_DEVICES is set, EGL_CUDA_DEVICE_NV is re-indexed to that
    visible CUDA namespace. On a single visible MIG device, the matching EGL
    devices report cuda_idx == 0 while unrelated devices often fail the query.
    """
    egl = _load_egl()
    all_devices = _query_egl_devices_ctypes()

    try:
        eglQueryDeviceAttribEXT = _get_egl_ext_function(
            egl,
            b"eglQueryDeviceAttribEXT",
            c_uint32,
            [c_void_p, c_int, ctypes.POINTER(ctypes.c_long)],
        )
    except RuntimeError:
        return None

    matching_device_indices = []
    for i, device in enumerate(all_devices):
        cuda_idx = ctypes.c_long(-1)
        ok = eglQueryDeviceAttribEXT(device, EGL_CUDA_DEVICE_NV, byref(cuda_idx))
        if ok and cuda_idx.value == visible_cuda_idx:
            matching_device_indices.append(i)

    if not matching_device_indices:
        return None

    if mig_local_device_idx is not None and mig_local_device_idx < len(matching_device_indices):
        selected_idx = matching_device_indices[mig_local_device_idx]
        print(
            f"[mujoco_mig_setup] EGL device {selected_idx} matches "
            f"visible CUDA device {visible_cuda_idx} and MIG local index {mig_local_device_idx}"
        )
        return selected_idx

    selected_idx = matching_device_indices[0]
    print(
        f"[mujoco_mig_setup] EGL device {selected_idx} matches "
        f"visible CUDA device {visible_cuda_idx}"
    )
    return selected_idx


# ---------------------------------------------------------------------------
# EGL device index <-> GPU index mapping
# ---------------------------------------------------------------------------

def _find_egl_device_index_for_gpu(target_gpu_idx: int, mig_local_device_idx: Optional[int] = None) -> int:
    """
    Enumerate EGL devices and find which EGL device index corresponds to
    the given physical GPU index, using EGL_CUDA_DEVICE_NV attribute.

    On MIG systems, EGL_CUDA_DEVICE_NV aligns with the first CUDA ordinal for
    the parent GPU. When multiple EGL devices share that ordinal, we use the
    local MIG device index as a stable tie-breaker when available.
    """
    egl = _load_egl()
    all_devices = _query_egl_devices_ctypes()
    gpu_to_cuda_ordinal = _get_gpu_to_cuda_ordinal_map()
    target_cuda_ordinal = gpu_to_cuda_ordinal.get(target_gpu_idx, target_gpu_idx)

    try:
        eglQueryDeviceAttribEXT = _get_egl_ext_function(
            egl,
            b"eglQueryDeviceAttribEXT",
            c_uint32,
            [c_void_p, c_int, ctypes.POINTER(ctypes.c_long)],
        )

        matching_device_indices = []
        for i, device in enumerate(all_devices):
            cuda_idx = ctypes.c_long(-1)
            ok = eglQueryDeviceAttribEXT(device, EGL_CUDA_DEVICE_NV, byref(cuda_idx))
            if ok and cuda_idx.value == target_cuda_ordinal:
                matching_device_indices.append(i)

        if matching_device_indices:
            if mig_local_device_idx is not None and mig_local_device_idx < len(matching_device_indices):
                selected_idx = matching_device_indices[mig_local_device_idx]
                print(
                    f"[mujoco_mig_setup] EGL device {selected_idx} matches "
                    f"physical GPU {target_gpu_idx} via CUDA ordinal {target_cuda_ordinal} "
                    f"and MIG local index {mig_local_device_idx}"
                )
                return selected_idx

            selected_idx = matching_device_indices[0]
            print(
                f"[mujoco_mig_setup] EGL device {selected_idx} matches "
                f"physical GPU {target_gpu_idx} via CUDA ordinal {target_cuda_ordinal}"
            )
            return selected_idx
    except RuntimeError:
        # eglQueryDeviceAttribEXT not available; fall through
        pass

    # Fallback: assume EGL device order matches GPU index order
    if 0 <= target_gpu_idx < len(all_devices):
        print(
            f"[mujoco_mig_setup] Falling back to EGL device index = "
            f"GPU index = {target_gpu_idx}"
        )
        return target_gpu_idx

    raise RuntimeError(
        f"Cannot map GPU index {target_gpu_idx} to an EGL device. "
        f"Only {len(all_devices)} EGL devices found."
    )


# ---------------------------------------------------------------------------
# Main setup
# ---------------------------------------------------------------------------

def setup():
    """
    Auto-detect the correct MUJOCO_EGL_DEVICE_ID from CUDA_VISIBLE_DEVICES
    and apply the MuJoCo EGL monkey-patch.

    Call this BEFORE `import mujoco`.
    """
    # Ensure EGL backend
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not cuda_visible:
        print("[mujoco_mig_setup] CUDA_VISIBLE_DEVICES not set; skipping auto-config.")
        _patch_mujoco_egl()
        return

    print(f"[mujoco_mig_setup] CUDA_VISIBLE_DEVICES = {cuda_visible}")
    if cuda_visible.split(",")[0].strip().startswith("MIG-"):
        print(
            "[mujoco_mig_setup] NOTE: EGL can only target the parent physical GPU, "
            "not an individual MIG slice. MuJoCo rendering may share the same "
            "physical GPU as PyTorch, but it cannot be pinned to the exact MIG instance."
        )

    try:
        mig_local_device_idx = _get_mig_local_device_index(cuda_visible)
        if mig_local_device_idx is not None:
            print(f"[mujoco_mig_setup] Local MIG device index = {mig_local_device_idx}")

        # First prefer the CUDA-visible namespace from the current process.
        egl_device_idx = _find_egl_device_index_for_visible_cuda(
            visible_cuda_idx=0, mig_local_device_idx=mig_local_device_idx
        )
        if egl_device_idx is not None:
            print(f"[mujoco_mig_setup] Setting MUJOCO_EGL_DEVICE_ID = {egl_device_idx}")
            os.environ["MUJOCO_EGL_DEVICE_ID"] = str(egl_device_idx)
            _patch_mujoco_egl()
            return

        parent_gpu_idx = _get_parent_gpu_index(cuda_visible)
        print(f"[mujoco_mig_setup] Parent GPU index = {parent_gpu_idx}")

        egl_device_idx = _find_egl_device_index_for_gpu(parent_gpu_idx, mig_local_device_idx)
        print(f"[mujoco_mig_setup] Setting MUJOCO_EGL_DEVICE_ID = {egl_device_idx}")

        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(egl_device_idx)
    except RuntimeError as e:
        print(f"[mujoco_mig_setup] WARNING: Auto-detection failed: {e}")
        print("[mujoco_mig_setup] Falling back to default EGL device selection.")

    _patch_mujoco_egl()


def _patch_mujoco_egl():
    """Replace MuJoCo's EGL display creation with our ctypes-based version."""
    try:
        import mujoco.egl as mujoco_egl
        mujoco_egl.create_initialized_egl_device_display = (
            create_initialized_egl_device_display_full
        )
        print("[mujoco_mig_setup] MuJoCo EGL patch applied successfully.")
    except ImportError:
        print(
            "[mujoco_mig_setup] WARNING: mujoco.egl not found. "
            "Patch not applied. Make sure mujoco is installed."
        )


# ---------------------------------------------------------------------------
# Auto-setup on import
# ---------------------------------------------------------------------------
setup()
