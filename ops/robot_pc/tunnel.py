"""Direct robot PC -> 5090 route: one ``ssh -N -L`` forward to the policy server's container, managed by pid file
(docs/robot_pc.md "Direct route"). The private key and the pinned host keys are FILES placed by the user/coordinator
(``secrets/ssh_5090_ed25519``, 0600; ``secrets/known_hosts_5090``); this module only checks their presence and mode and
passes their paths to ssh - it never reads, prints or logs them. The 5090's authorized_keys entry for that key allows
exactly ``permitopen="<container IP>:47100"`` and no shell. Standard library only.

* ``start``: refuses if the key / known hosts are missing or the key is not 0600, if the local port is a switch_controller
  channel or already bound, or if a tunnel pid file is live; starts ssh in its own session, waits for the listener and
  the server's WELCOME, else stops it again (fail closed).
* ``status``: pid alive (and its command line is this tunnel's ssh), listener accepts, WELCOME received.
* ``stop``: SIGTERM to the pid from the pid file after checking that pid's command line (never ``pkill -f``).
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


def _cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{int(pid)}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return ""


def _is_our_tunnel(pid: int, port: int) -> bool:
    cmd = _cmdline(pid)
    return cmd.startswith("ssh ") and f"-L 127.0.0.1:{int(port)}:" in cmd and " -N " in f" {cmd} "


def _pid(p: dict) -> int | None:
    try:
        return int(p["pid"].read_text().strip())
    except (OSError, ValueError):
        return None


def preflight(root, port: int) -> dict:
    admit_endpoint("127.0.0.1", port)
    p = paths(root, port)
    for k in ("key", "known_hosts"):
        if not p[k].is_file():
            raise TunnelError(f"{p[k]} missing (placed by the user / coordinator; never by this tool)")
    if stat.S_IMODE(p["key"].stat().st_mode) & 0o077:
        raise TunnelError(f"{p['key']} must be mode 0600")
    pid = _pid(p)
    if pid is not None and _is_our_tunnel(pid, port):
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


def status(root, port: int = 47100) -> dict:
    p = paths(root, port)
    pid = _pid(p)
    alive = pid is not None and _is_our_tunnel(pid, port)
    listening = _listening(port)
    w = welcome(port) if listening else None
    return {"pid": pid, "pid_alive_ssh_tunnel": alive, "listening": listening, "welcome": w,
            "ok": bool(alive and listening and w)}


def start(root, *, port: int = 47100, container: str = CONTAINER, wait_s: float = 10.0) -> dict:
    p = preflight(root, port)
    p["log"].parent.mkdir(parents=True, exist_ok=True)
    with open(p["log"], "ab") as log:
        proc = subprocess.Popen(command(root, port=port, container=container), cwd=p["root"], stdin=subprocess.DEVNULL,
                                stdout=log, stderr=log, start_new_session=True)
    p["pid"].write_text(f"{proc.pid}\n")
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline and proc.poll() is None and not _listening(port):
        time.sleep(0.2)
    st = status(root, port)
    if not st["ok"]:
        stop(root, port)
        raise TunnelError(f"tunnel did not come up (exit {proc.poll()}, see {p['log']}): {st}")
    return st


def stop(root, port: int = 47100) -> dict:
    p = paths(root, port)
    pid = _pid(p)
    if pid is None:
        return {"stopped": False, "reason": "no pid file"}
    if not _is_our_tunnel(pid, port):
        raise ExecutionRefused(f"pid {pid} is not this tunnel's ssh ({_cmdline(pid)[:80]!r}); not signalled")
    os.kill(pid, signal.SIGTERM)
    for _ in range(25):
        if not _is_our_tunnel(pid, port):
            break
        time.sleep(0.2)
    p["pid"].unlink(missing_ok=True)
    return {"stopped": True, "pid": pid}
