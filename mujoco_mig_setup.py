"""Select the proved EGL device for one allowlisted CUDA UUID before MuJoCo imports."""

from __future__ import annotations

import ctypes
import os
from ctypes import byref, c_char_p, c_int, c_uint32, c_void_p

from s4d.gpu_guard import GPUIsolationError, visible_device_uuids

EGL_NO_DISPLAY = c_void_p(0)
EGL_TRUE = 1
EGL_SUCCESS = 0x3000
EGL_PLATFORM_DEVICE_EXT = 0x313F
EGL_CUDA_DEVICE_NV = 0x323A

_SELECTED_EGL_DEVICE_ID: int | None = None
_SELECTED_GPU_UUID: str | None = None


def _load_egl():
    try:
        egl = ctypes.CDLL("libEGL.so.1")
    except OSError as exc:
        raise RuntimeError(f"Failed to load libEGL.so.1: {exc}") from exc
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
    return ctypes.CFUNCTYPE(restype, *argtypes)(ptr)


def _query_egl_devices_ctypes():
    egl = _load_egl()
    query = _get_egl_ext_function(
        egl, b"eglQueryDevicesEXT", c_uint32,
        [c_int, ctypes.POINTER(c_void_p), ctypes.POINTER(c_int)],
    )
    devices = (c_void_p * 64)()
    count = c_int(0)
    if not query(len(devices), devices, byref(count)):
        raise RuntimeError(f"eglQueryDevicesEXT failed, eglGetError=0x{egl.eglGetError():04x}")
    return list(devices[:count.value])


def _single_visible_uuid() -> str:
    entries = visible_device_uuids()
    if len(entries) != 1:
        raise GPUIsolationError("MuJoCo rendering requires exactly one allowed UUID in CUDA_VISIBLE_DEVICES.")
    return entries[0]


def _find_egl_device_index_for_visible_cuda(visible_cuda_idx: int = 0) -> int | None:
    egl = _load_egl()
    devices = _query_egl_devices_ctypes()
    query = _get_egl_ext_function(
        egl, b"eglQueryDeviceAttribEXT", c_uint32,
        [c_void_p, c_int, ctypes.POINTER(ctypes.c_long)],
    )
    matching = []
    for index, device in enumerate(devices):
        cuda_index = ctypes.c_long(-1)
        if query(device, EGL_CUDA_DEVICE_NV, byref(cuda_index)) and cuda_index.value == visible_cuda_idx:
            matching.append(index)
    if not matching:
        return None
    if len(matching) != 1:
        raise GPUIsolationError(f"Multiple EGL devices map to visible CUDA ordinal {visible_cuda_idx}; refusing EGL order.")
    return matching[0]


def create_initialized_egl_device_display_full():
    uuid = _single_visible_uuid()
    selected = os.environ.get("MUJOCO_EGL_DEVICE_ID")
    if selected is None or not selected.strip().isascii() or not selected.strip().isdigit():
        raise GPUIsolationError("MUJOCO_EGL_DEVICE_ID must be the proved numeric EGL device index.")
    device_index = int(selected)
    if (
        _SELECTED_EGL_DEVICE_ID is None or device_index != _SELECTED_EGL_DEVICE_ID
        or _SELECTED_GPU_UUID is None or uuid.lower() != _SELECTED_GPU_UUID.lower()
    ):
        raise GPUIsolationError("EGL device or CUDA UUID changed after proved selection.")
    egl = _load_egl()
    devices = _query_egl_devices_ctypes()
    if not 0 <= device_index < len(devices):
        raise GPUIsolationError(f"MUJOCO_EGL_DEVICE_ID={device_index} is outside the {len(devices)} EGL devices.")
    query = _get_egl_ext_function(
        egl, b"eglQueryDeviceAttribEXT", c_uint32,
        [c_void_p, c_int, ctypes.POINTER(ctypes.c_long)],
    )
    cuda_index = ctypes.c_long(-1)
    if not query(devices[device_index], EGL_CUDA_DEVICE_NV, byref(cuda_index)) or cuda_index.value != 0:
        raise GPUIsolationError("Selected EGL device no longer maps to visible CUDA ordinal 0.")
    get_display = _get_egl_ext_function(
        egl, b"eglGetPlatformDisplayEXT", c_void_p, [c_int, c_void_p, c_void_p],
    )
    display = get_display(EGL_PLATFORM_DEVICE_EXT, devices[device_index], None)
    if not display or egl.eglGetError() != EGL_SUCCESS:
        raise RuntimeError(f"Failed to create the proved EGL device {device_index} display.")
    if egl.eglInitialize(display, None, None) != EGL_TRUE or egl.eglGetError() != EGL_SUCCESS:
        raise RuntimeError(f"Failed to initialize the proved EGL device {device_index} display.")
    return display


def setup():
    global _SELECTED_EGL_DEVICE_ID, _SELECTED_GPU_UUID
    _SELECTED_EGL_DEVICE_ID = None
    _SELECTED_GPU_UUID = None
    os.environ.pop("MUJOCO_EGL_DEVICE_ID", None)
    uuid = _single_visible_uuid()
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    device_index = _find_egl_device_index_for_visible_cuda(visible_cuda_idx=0)
    if device_index is None:
        raise GPUIsolationError("No EGL device maps to visible CUDA ordinal 0; refusing default EGL selection.")
    _SELECTED_EGL_DEVICE_ID = device_index
    _SELECTED_GPU_UUID = uuid
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(device_index)
    print(f"[mujoco_mig_setup] EGL device {device_index} -> visible CUDA ordinal 0 ({uuid})")
    _patch_mujoco_egl()


def _patch_mujoco_egl():
    import mujoco.egl as mujoco_egl

    mujoco_egl.create_initialized_egl_device_display = create_initialized_egl_device_display_full
    print("[mujoco_mig_setup] MuJoCo EGL patch applied successfully.")


setup()
