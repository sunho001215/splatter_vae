"""Direct robot PC -> 5090 route: one ``ssh -N -L`` forward to the policy server's container, managed by pid file
(docs/robot_pc.md "Direct route"). The private key and the pinned host keys are FILES placed by the user/coordinator
(``secrets/ssh_5090_ed25519``, 0600; ``secrets/known_hosts_5090``); this module only checks their presence and mode and
passes their paths to ssh - it never reads, prints or logs them. The 5090's authorized_keys entry for that key allows
exactly ``permitopen="<container IP>:47100"`` and no shell. Standard library only.

* ``start``: refuses if the key / known hosts are missing or the key is not 0600, if the local port is a switch_controller
  channel or already bound, or if a tunnel pid file is live; starts ssh in its own session, waits for the listener and
  the server's WELCOME, else stops it again (fail closed).
* ``status``: pid alive (and its command line is this tunnel's ssh), listener accepts, WELCOME received.
* ``stop``: SIGTERM (SIGKILL after 5 s) through a pidfd to the pid from the pid file, only if that process's full argv
  is exactly this tunnel's command, its working directory is the root and (when recorded) its start time matches
  (never ``pkill -f``).
"""

from __future__ import annotations

import os
import signal
import socket
import stat
import subprocess
import time
from pathlib import Path

from ops.robot_pc.safety import ExecutionRefused, admit_endpoint

TARGET = "irsl@143.248.142.98"
CONTAINER = "172.17.0.2"  # shkim_rw on the 5090 (the authorized_keys permitopen names it)
KEY, KNOWN_HOSTS = "secrets/ssh_5090_ed25519", "secrets/known_hosts_5090"


class TunnelError(RuntimeError):
    pass


def paths(root, port: int) -> dict:
    root = Path(root).expanduser()
    return {"root": root, "key": root / KEY, "known_hosts": root / KNOWN_HOSTS,
            "pid": root / f"logs/tunnel-direct-{port}.pid", "log": root / f"logs/tunnel-direct-{port}.log"}


def command(root, *, port: int = 47100, container: str = CONTAINER, remote_port: int = 47100,
            target: str = TARGET) -> list:
    """Exactly the coordinator-verified command (relative key / known-hosts paths, run from ``root``)."""
    return ["ssh", "-N", "-i", KEY, "-o", "IdentitiesOnly=yes", "-o", "IdentityAgent=none",
            "-o", "PasswordAuthentication=no", "-o", "KbdInteractiveAuthentication=no",
            "-o", f"UserKnownHostsFile={KNOWN_HOSTS}", "-o", "StrictHostKeyChecking=yes",
            "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3",
            "-o", "ControlMaster=no", "-o", "ControlPath=none",
            "-L", f"127.0.0.1:{int(port)}:{container}:{int(remote_port)}", target]


def _listening(port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def _argv(pid: int) -> list:
    try:
        raw = Path(f"/proc/{int(pid)}/cmdline").read_bytes()
    except OSError:
        return []
    return [a.decode(errors="replace") for a in raw.split(b"\0") if a]


def _start_ticks(pid: int) -> str | None:
    """Process start time (clock ticks since boot, /proc/<pid>/stat field 22): guards against pid reuse."""
    try:
        stat_line = Path(f"/proc/{int(pid)}/stat").read_text()
    except OSError:
        return None
    return stat_line[stat_line.rindex(")") + 2:].split()[19]


def _is_our_tunnel(pid: int, root, port: int, container: str = CONTAINER, start_ticks: str | None = None) -> bool:
    """Exactly this tunnel: the full argv equals ``command(...)``, the working directory is ``root`` (the key paths are
    relative to it) and, when recorded, the process start time matches (no pid reuse)."""
    if _argv(pid) != command(root, port=port, container=container):
        return False
    try:
        if Path(os.readlink(f"/proc/{int(pid)}/cwd")) != Path(root).expanduser().resolve():
            return False
    except OSError:
        return False
    return start_ticks is None or _start_ticks(pid) == start_ticks


def _pid(p: dict) -> tuple[int | None, str | None]:
    """(pid, recorded start ticks or None) from the pid file ("<pid>" or "<pid> <start ticks>")."""
    try:
        parts = p["pid"].read_text().split()
        return int(parts[0]), (parts[1] if len(parts) > 1 else None)
    except (OSError, ValueError, IndexError):
        return None, None


def preflight(root, port: int, container: str = CONTAINER) -> dict:
    admit_endpoint("127.0.0.1", port)
    p = paths(root, port)
    for k in ("key", "known_hosts"):
        if not p[k].is_file():
            raise TunnelError(f"{p[k]} missing (placed by the user / coordinator; never by this tool)")
    if stat.S_IMODE(p["key"].stat().st_mode) & 0o077:
        raise TunnelError(f"{p['key']} must be mode 0600")
    pid, ticks = _pid(p)
    if pid is not None and _is_our_tunnel(pid, root, port, container, ticks):
        raise TunnelError(f"a tunnel is already running (pid {pid}, {p['pid']})")
    if _listening(port):
        raise TunnelError(f"127.0.0.1:{port} is already bound (another tunnel / relay)")
    return p


def welcome(port: int, timeout: float = 3.0) -> dict | None:
    from ops.policy_serving import protocol as P

    try:
        sock = socket.create_connection(("127.0.0.1", int(port)), timeout=timeout)
    except OSError:
        return None
    wire = P.Wire(sock)
    try:
        msg = wire.recv(timeout)
        return None if msg is None or msg.type != P.WELCOME else {"server": msg.get("server"),
                                                                  "policies": len(msg.get("policies") or {})}
    except (OSError, TimeoutError, ConnectionError, P.ProtocolError):
        return None
    finally:
        wire.close()


def status(root, port: int = 47100, container: str = CONTAINER) -> dict:
    p = paths(root, port)
    pid, ticks = _pid(p)
    alive = pid is not None and _is_our_tunnel(pid, root, port, container, ticks)
    listening = _listening(port)
    w = welcome(port) if listening else None
    return {"pid": pid, "pid_alive_ssh_tunnel": alive, "listening": listening, "welcome": w,
            "ok": bool(alive and listening and w)}


def start(root, *, port: int = 47100, container: str = CONTAINER, wait_s: float = 10.0) -> dict:
    p = preflight(root, port, container)
    p["log"].parent.mkdir(parents=True, exist_ok=True)
    with open(p["log"], "ab") as log:
        proc = subprocess.Popen(command(root, port=port, container=container), cwd=p["root"].resolve(),
                                stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
    try:
        p["pid"].write_text(f"{proc.pid} {_start_ticks(proc.pid)}\n")
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline and proc.poll() is None and not _listening(port):
            time.sleep(0.2)
        st = status(root, port, container)
        if not st["ok"]:
            raise TunnelError(f"tunnel did not come up (exit {proc.poll()}, see {p['log']}): {st}")
        return st
    except BaseException:  # never leave our ssh behind: terminate, kill, reap through the retained handle
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=3)
        p["pid"].unlink(missing_ok=True)
        raise


def stop(root, port: int = 47100, container: str = CONTAINER) -> dict:
    """SIGTERM (then SIGKILL after 5 s) through a pidfd opened only after the identity check (no pid-reuse race)."""
    p = paths(root, port)
    pid, ticks = _pid(p)
    if pid is None:
        return {"stopped": False, "reason": "no pid file"}
    pidfd = getattr(os, "pidfd_open", None)  # Linux >= 5.3 / CPython >= 3.9 (the robot PC: 3.12); else os.kill
    try:
        fd = pidfd(pid) if pidfd else None
    except OSError:
        p["pid"].unlink(missing_ok=True)
        return {"stopped": False, "reason": f"pid {pid} is not running (stale pid file removed)"}

    def send(sig):
        signal.pidfd_send_signal(fd, sig) if fd is not None else os.kill(pid, sig)

    try:
        if not _is_our_tunnel(pid, root, port, container, ticks):
            raise ExecutionRefused(f"pid {pid} is not this tunnel's ssh ({' '.join(_argv(pid))[:80]!r}); not signalled")
        send(signal.SIGTERM)
        for _ in range(25):
            if not _is_our_tunnel(pid, root, port, container, ticks):
                break
            time.sleep(0.2)
        else:
            send(signal.SIGKILL)
            time.sleep(0.2)
        if _is_our_tunnel(pid, root, port, container, ticks):
            raise TunnelError(f"pid {pid} survived SIGKILL")
    finally:
        if fd is not None:
            os.close(fd)
    p["pid"].unlink(missing_ok=True)
    return {"stopped": True, "pid": pid}
