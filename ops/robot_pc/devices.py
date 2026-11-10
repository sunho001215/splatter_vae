"""Device-busy guard of the live dry run: refuse to open cameras when that could interfere with a running recording
or teleoperation session. Read-only inspection of /proc and /tmp; it opens no device. Standard library only.

Busy when any of:
* a process other than this one has a video / USB device node open (``/proc/<pid>/fd`` links to ``/dev/video*`` or
  ``/dev/bus/usb/*``; processes of other users whose fds cannot be read are reported as ``unreadable``);
* a process command line matches a recorder / teleop / RealSense tool pattern (``RECORDER_PATTERNS``, from the
  recordings: software ``panda_teleop`` 2.0.0, its collector IPC ``teleop_collector_requests``);
* the collector's request IPC socket exists (``/tmp/teleop_collector_requests.ipc``, recorded ``robot.request_endpoint``).
The camera pipeline start itself is the last check: librealsense refuses a device that is already streaming.
"""

from __future__ import annotations

import os
from pathlib import Path

RECORDER_PATTERNS = ("panda_teleop", "teleop_collector", "teleop", "realsense-viewer", "rs-record", "rs-capture",
                     "switch_controller")
COLLECTOR_IPC = "/tmp/teleop_collector_requests.ipc"
DEVICE_PREFIXES = ("/dev/video", "/dev/bus/usb/")


class DevicesBusy(RuntimeError):
    pass


def scan(proc_root: str = "/proc", *, patterns=RECORDER_PATTERNS, ipc_paths=(COLLECTOR_IPC,),
         self_pid: int | None = None) -> dict:
    """{"holders": [(pid, device, cmdline)], "recorders": [(pid, cmdline)], "ipc": [path], "unreadable": n}."""
    me = os.getpid() if self_pid is None else int(self_pid)
    holders, recorders, unreadable = [], [], 0
    for d in Path(proc_root).iterdir():
        if not d.name.isdigit() or int(d.name) == me:
            continue
        try:
            cmd = (d / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
        except OSError:
            continue
        low = cmd.lower()
        if cmd and any(p in low for p in patterns) and "robot_pc" not in low:
            recorders.append((int(d.name), cmd))
        try:
            fds = list((d / "fd").iterdir())
        except OSError:
            unreadable += 1
            continue
        for fd in fds:
            try:
                target = os.readlink(fd)
            except OSError:
                continue
            if target.startswith(DEVICE_PREFIXES):
                holders.append((int(d.name), target, cmd))
    return {"holders": holders, "recorders": recorders, "ipc": [p for p in ipc_paths if Path(p).exists()],
            "unreadable": unreadable}


def require_free(**kw) -> dict:
    """``scan`` and raise ``DevicesBusy`` naming every finding; returns the scan when nothing is busy."""
    s = scan(**kw)
    reasons = [f"pid {p} holds {dev} ({cmd[:80]})" for p, dev, cmd in s["holders"]]
    reasons += [f"pid {p} looks like a recorder/teleop process ({cmd[:80]})" for p, cmd in s["recorders"]]
    reasons += [f"collector IPC {p} exists (a running or crashed teleop collector)" for p in s["ipc"]]
    if reasons:
        raise DevicesBusy("refusing live acquisition (it could interfere with a recording): " + "; ".join(reasons))
    return s
