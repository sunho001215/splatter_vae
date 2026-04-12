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

def _parse_mig_to_gpu_map():
    """
    Parse `nvidia-smi -L` to build a dict:  MIG-UUID -> parent GPU index.

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
    current_gpu_idx = None

    for line in output.splitlines():
        # Match: GPU 0: ... (UUID: GPU-xxxx)
        gpu_match = re.match(r"^GPU\s+(\d+):", line)
        if gpu_match:
            current_gpu_idx = int(gpu_match.group(1))
            continue

        # Match:   MIG ...  (UUID: MIG-xxxx)
        mig_match = re.search(r"\(UUID:\s+(MIG-[0-9a-f\-]+)\)", line)
        if mig_match and current_gpu_idx is not None:
            mig_uuid = mig_match.group(1)
            mig_to_gpu[mig_uuid] = current_gpu_idx

    return mig_to_gpu


def _get_parent_gpu_index(cuda_visible: str) -> int:
    """
    Given a CUDA_VISIBLE_DEVICES value (MIG UUID), return the parent GPU index.
    Supports formats: MIG-xxxx, MIG-GPU-xxxx/x/x, or just the UUID part.
    """
    # Normalize: take the first device if comma-separated
    device_id = cuda_visible.split(",")[0].strip()

    # If it looks like a MIG UUID
    if device_id.startswith("MIG-"):
        mig_map = _parse_mig_to_gpu_map()
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
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10,
            )
            for line in result.stdout.splitlines():
                parts = line.split(",")
                if len(parts) == 2 and parts[1].strip() == device_id:
                    return int(parts[0].strip())
        except Exception:
            pass
        raise RuntimeError(f"Could not resolve GPU UUID '{device_id}' to an index.")

    raise RuntimeError(f"Unrecognized CUDA_VISIBLE_DEVICES format: '{device_id}'")


# ---------------------------------------------------------------------------
# EGL device index <-> GPU index mapping
# ---------------------------------------------------------------------------

def _find_egl_device_index_for_gpu(target_gpu_idx: int) -> int:
    """
    Enumerate EGL devices and find which EGL device index corresponds to
    the given physical GPU index, using EGL_CUDA_DEVICE_NV attribute.

    Falls back to using the GPU index directly if the attribute query fails.
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

        for i, device in enumerate(all_devices):
            cuda_idx = ctypes.c_long(-1)
            ok = eglQueryDeviceAttribEXT(device, EGL_CUDA_DEVICE_NV, byref(cuda_idx))
            if ok and cuda_idx.value == target_gpu_idx:
                print(
                    f"[mujoco_mig_setup] EGL device {i} matches "
                    f"CUDA device {target_gpu_idx} (via EGL_CUDA_DEVICE_NV)"
                )
                return i
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

    try:
        parent_gpu_idx = _get_parent_gpu_index(cuda_visible)
        print(f"[mujoco_mig_setup] Parent GPU index = {parent_gpu_idx}")

        egl_device_idx = _find_egl_device_index_for_gpu(parent_gpu_idx)
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