"""Robotiq 2F-85 on the robot PC: the MEASURED gripper state for the policy (Robotiq position register gPO, the
training proprio ``gripper/position``) and, in execution mode only, the closure command - with this process as the ONLY
Modbus master on the serial port.

Protocol, rates and conventions are those of the robot PC's teleop collector (panda-teleop/teleop_collector/
teleop_collector/gripper.py, read-only; ported, not imported): Modbus RTU 115200 8N1, slave 9, status FC03 at 0x07D0
(3 registers: ACTION/STATUS, reserved, gFLT, gPR, gPO, gCU), command FC16 at 0x03E8, 50 Hz cycle, deadband 2 counts,
1 s keepalive, closure c <-> register 3 + 227 c (DROID / Polymetis).

Single owner (no second Modbus master): ``open_exclusive`` refuses while any other process has the port open or a
teleop collector / switch_controller-side gripper user runs (/proc scan), takes an advisory ``flock`` on a lock file
and opens the port with ``exclusive=True`` (advisory flock), then explicitly applies TIOCEXCL and rechecks holders.
Unreadable process fds are counted as unverifiable ownership. ``read_only=True``
(dry run / validation) sends ONLY status reads: no activation, no command - the fingers cannot move.
"""

from __future__ import annotations

import collections
import fcntl
import logging
import os
import termios
import threading
import time
from dataclasses import dataclass
from pathlib import Path

CMD_REGISTER, STATUS_REGISTER = 0x03E8, 0x07D0
BAUDRATE, SLAVE_ID = 115200, 9
RATE_HZ = 50
DEADBAND = 2
POS_OPEN, POS_CLOSED = 3, 230
PORT = "/dev/ttyUSB0"
LOCK = "/tmp/s4d-robot-pc-gripper.lock"
log = logging.getLogger(__name__)


def crc16_modbus(data: bytes) -> bytes:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return bytes([crc & 0xFF, crc >> 8])


def read_request(slave: int = SLAVE_ID) -> bytes:
    body = bytes([slave, 0x03, STATUS_REGISTER >> 8, STATUS_REGISTER & 0xFF, 0x00, 0x03])
    return body + crc16_modbus(body)


def write_request(position: int, speed: int, force: int, slave: int = SLAVE_ID, activate=True, go=True) -> bytes:
    action = (1 if activate else 0) | (8 if go else 0)
    body = bytes([slave, 0x10, CMD_REGISTER >> 8, CMD_REGISTER & 0xFF, 0x00, 0x03, 0x06,
                  action, 0x00, 0x00, position & 0xFF, speed & 0xFF, force & 0xFF])
    return body + crc16_modbus(body)


@dataclass(frozen=True)
class GripperStatus:
    t_ns: int  # host CLOCK_MONOTONIC when the reply was read
    activated: int
    goto: int
    status: int  # gSTA 3 = activated
    object: int  # gOBJ 3 = at target, 2 / 1 contact
    fault: int
    pos_request_echo: int
    position: int  # gPO 0..255: the MEASURED register (training proprio)
    current: int


def parse_status(reply: bytes, t_ns: int) -> GripperStatus:
    d = reply[3:9]
    b0 = d[0]
    return GripperStatus(t_ns, b0 & 1, (b0 >> 3) & 1, (b0 >> 4) & 3, (b0 >> 6) & 3, d[2], d[3], d[4], d[5])


def closure_to_position(closure: float) -> int:
    c = min(1.0, max(0.0, float(closure)))
    return int(round(POS_OPEN + c * (POS_CLOSED - POS_OPEN)))


class PortBusy(RuntimeError):
    pass


class BadReply(OSError):
    """A malformed or unsolicited Modbus response (possible second master)."""


def port_holders(port: str = PORT, proc_root: str = "/proc", self_pid: int | None = None) -> dict:
    """Known holders and a separate count of fd entries/directories we cannot verify."""
    me, real, out = os.getpid() if self_pid is None else self_pid, os.path.realpath(port), []
    unverifiable = 0
    for d in Path(proc_root).iterdir():
        if not d.name.isdigit() or int(d.name) == me:
            continue
        try:
            for fd in (d / "fd").iterdir():
                try:
                    if os.path.realpath(os.readlink(fd)) == real:
                        out.append(int(d.name))
                        break
                except PermissionError:
                    unverifiable += 1
                except OSError:
                    continue  # process/fd may have exited
        except PermissionError:
            unverifiable += 1
        except OSError:
            continue
    return {"holders": out, "unverifiable": unverifiable}


class RobotiqRTU:
    """One serial connection, opened exclusively (single Modbus master)."""

    def __init__(self, port: str = PORT, *, serial_module=None, lock_path: str = LOCK, timeout: float = 0.05,
                 check_holders=port_holders, ioctl=fcntl.ioctl, listen_s: float = 0.3, read_source=None):
        from ops.robot_pc import devices

        busy = devices.scan(ipc_paths=())["recorders"]
        def scan_holders():
            report = check_holders(port)
            # List-returning probes remain usable by callers/tests.
            return report if isinstance(report, dict) else {"holders": report, "unverifiable": 0}

        ownership = scan_holders()
        holders = ownership["holders"]
        self.unverifiable = ownership["unverifiable"]
        if busy or holders:
            raise PortBusy(f"{port}: refusing a second Modbus master (holders {holders}, recorder/teleop {busy})")
        self._lock = open(lock_path, "a")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self._lock.close()
            raise PortBusy(f"{port}: another s4d gripper owner holds {lock_path}") from exc
        try:
            if serial_module is None:
                import serial as serial_module
            self.ser = serial_module.Serial(port, BAUDRATE, bytesize=8, parity="N", stopbits=1, timeout=timeout,
                                            write_timeout=0.05, exclusive=True)
            ioctl(self.ser.fileno(), termios.TIOCEXCL)
            ownership = scan_holders()  # TIOCEXCL doesn't evict pre-existing opens
            self.unverifiable = max(self.unverifiable, ownership["unverifiable"])
            if ownership["holders"]:
                raise PortBusy(f"{port}: holders after TIOCEXCL: {ownership['holders']}")
            self._read = self.ser.read if read_source is None else read_source
            self.consecutive_bad_replies = 0
            self.ser.reset_input_buffer()
            deadline = time.monotonic() + float(listen_s)
            while time.monotonic() < deadline:
                if self._read(1):
                    raise PortBusy("bus traffic without our request: another Modbus master?")
            if self.unverifiable:
                log.warning("%s: ownership unverifiable for %d /proc fd entries/directories", port,
                            self.unverifiable)
        except Exception:
            try:
                if hasattr(self, "ser"):
                    self.ser.close()
            finally:
                self._release()
            raise

    def _transact(self, request: bytes, reply_len: int) -> bytes:
        self.ser.reset_input_buffer()
        self.ser.write(request)
        reply = self._read(reply_len)
        if len(reply) != reply_len or reply[0] != request[0] or reply[1] != request[1] or \
                crc16_modbus(reply[:-2]) != reply[-2:] or \
                (request[1] == 0x03 and reply[2] != 6) or \
                (request[1] == 0x10 and reply[:6] != request[:6]):
            self.consecutive_bad_replies += 1
            raise BadReply(f"Robotiq: bad reply (CRC / length / echo mismatch; {len(reply)}/{reply_len} bytes)")
        self.consecutive_bad_replies = 0
        return reply

    def read_status(self) -> GripperStatus:
        return parse_status(self._transact(read_request(), 11), time.monotonic_ns())

    def write(self, position: int, speed: int, force: int) -> None:
        self._transact(write_request(position, speed, force), 8)

    def _release(self) -> None:
        try:
            fcntl.flock(self._lock, fcntl.LOCK_UN)
        finally:
            self._lock.close()

    def close(self) -> None:
        try:
            self.ser.close()
        finally:
            self._release()


class GripperWorker(threading.Thread):
    """Owns the connection at 50 Hz: status every cycle (measured register samples with host times), and - only when
    ``read_only`` is False and a target was set - the closure command (deadband, keepalive). It never activates the
    gripper: a gripper that is not activated (gSTA != 3) is reported, and commands are withheld."""

    # speed: the collector's teleop value (config/default.yaml:63, rSP; DROID goto(speed=0.05) -> 146); force: the
    # MAXIMUM rFR register value 255 (0..255, default.yaml:64 / gripper.py write_request; user directive K)
    def __init__(self, device_factory, *, read_only: bool = True, speed: int = 146, force: int = 255,
                 capacity: int = 100_000):
        super().__init__(daemon=True, name="gripper")
        self.factory, self.read_only, self.speed, self.force = device_factory, bool(read_only), int(speed), int(force)
        self.samples: collections.deque = collections.deque(maxlen=capacity)  # (t_ns, GripperStatus)
        self.sent: collections.deque = collections.deque(maxlen=capacity)  # (t_ns, register)
        self.holds: list = []  # (t_ns, register, reason)
        self.latest: GripperStatus | None = None
        self.error: str | None = "connecting"
        self._target: int | None = None
        self._expires_ns = 0
        self._target_lock = threading.Lock()
        self._generation = 0
        self._hold_pending = None
        self._has_written = False
        self.error_latched = False
        self._bad_replies = 0
        self._halt = threading.Event()
        self._frozen = False
        self.hold_failures: list = []  # (t_ns, reason, error) of failed physical holds (kept for the verdict)
        self.unverifiable = 0
        self._last_sent, self._last_write = None, 0.0
        self.origin = "measured: Robotiq gPO register (Modbus status read by this process)"

    def set_closure(self, closure: float, expires_monotonic_ns: int) -> None:
        if self.read_only:
            raise PermissionError("gripper worker is read-only (dry run)")
        with self._target_lock:
            if not self._halt.is_set() and not self._frozen:
                target, expires = closure_to_position(closure), int(expires_monotonic_ns)
                if target != self._target:
                    self._generation += 1  # a new target (an expiry renewal alone keeps the pending write cycle)
                self._target, self._expires_ns = target, expires

    def freeze(self) -> None:
        """Session shutdown: clear the target and refuse every later set_closure (a late executor tick included)."""
        with self._target_lock:
            self._frozen = True
        self.clear_target()

    def clear_target(self) -> None:
        with self._target_lock:
            self._target, self._expires_ns = None, 0
            self._generation += 1
            self._hold_pending = self._hold_pending or "clear_target (session abort)"

    def healthy(self) -> bool:
        s = self.latest
        return bool(s is not None and 0 <= time.monotonic_ns() - s.t_ns < 100_000_000 and
                    s.activated == 1 and s.status == 3 and s.fault == 0 and self.error is None)

    def stop(self, join_timeout: float = 1.0) -> None:
        with self._target_lock:
            if not self._halt.is_set():
                self._target, self._expires_ns = None, 0
                self._generation += 1
                self._hold_pending = self._hold_pending or "stop"
                self._halt.set()
        if self.ident is not None and threading.current_thread() is not self:
            self.join(timeout=join_timeout)
            if self.is_alive():
                log.error("gripper worker did not exit within %.3f s", join_timeout)

    def _physical_hold(self, device, reason: str) -> None:
        s = self.latest
        if self.read_only or not self._has_written or s is None or \
                not 0 <= time.monotonic_ns() - s.t_ns < 100_000_000 or \
                s.activated != 1 or s.status != 3 or s.fault != 0:
            return
        if reason == "expiry" and s.object != 0:
            return  # expiry while stationary / in contact (gOBJ 1-3): holding at gPO could relax a grasp
        # One FC16 only. The worker owns all I/O; an already-started transaction finishes before this hold.
        try:
            device.write(s.position, self.speed, self.force)
        except Exception as exc:
            # The request may have reached the device: never retry this stop/expiry event.
            self.error = f"physical hold failed: {type(exc).__name__}: {exc}"
            self.hold_failures.append((time.monotonic_ns(), reason, self.error))
            self._check_bad_replies(device)
            log.exception("gripper physical hold failed")
            return
        t = time.monotonic_ns()
        self.sent.append((t, s.position))
        self.holds.append((t, s.position, reason))
        self._last_sent, self._last_write = s.position, time.monotonic()

    def _take_hold(self):
        with self._target_lock:
            reason, self._hold_pending = self._hold_pending, None
            return reason

    def _check_bad_replies(self, device) -> bool:
        self._bad_replies = getattr(device, "consecutive_bad_replies", 0)
        if self._bad_replies >= 3:
            self.error_latched = True
            self.error = "3 consecutive bad replies: " + (self.error or "CRC / length / echo mismatch")
        return self.error_latched

    def run(self) -> None:
        period = 1.0 / RATE_HZ
        device = None
        try:
            while not self._halt.is_set():
                t0 = time.monotonic()
                try:
                    if device is None:
                        device = self.factory()
                        self.unverifiable = getattr(device, "unverifiable", 0)
                        if hasattr(device, "consecutive_bad_replies"):
                            device.consecutive_bad_replies = self._bad_replies
                    s = device.read_status()
                    self.latest = s
                    self.samples.append((s.t_ns, s))
                    self.error = f"gripper fault 0x{s.fault:02X}" if s.fault else None
                    # Only the mailbox decision is locked. Callers never wait on a serial read/write.
                    with self._target_lock:
                        if self._target is not None and time.monotonic_ns() >= self._expires_ns:
                            self._target, self._expires_ns = None, 0
                            self._generation += 1
                            self._hold_pending = self._hold_pending or "expiry"
                        target, generation = self._target, self._generation
                    reason = self._take_hold()
                    if reason is not None:
                        self._physical_hold(device, reason)
                        if self.error_latched:
                            return
                    with self._target_lock:
                        write = (generation == self._generation and not self._halt.is_set() and
                                 self._hold_pending is None and not self.read_only and target is not None and
                                 time.monotonic_ns() < self._expires_ns and s.activated == 1 and
                                 s.status == 3 and s.fault == 0 and (
                                     self._last_sent is None or abs(target - self._last_sent) > DEADBAND or
                                     (target != self._last_sent and t0 - self._last_write > 0.2) or
                                     t0 - self._last_write > 1.0))
                    if write:
                        # A lost acknowledgement does not prove the position command was not applied.
                        self._has_written = True
                        device.write(target, self.speed, self.force)
                        self._last_sent, self._last_write = target, t0
                        self.sent.append((time.monotonic_ns(), target))
                    self._check_bad_replies(device)
                except BadReply as exc:
                    self.error = f"{type(exc).__name__}: {exc}"
                    if self._check_bad_replies(device):
                        self.clear_target()
                        return
                    self.clear_target()
                except Exception as exc:  # noqa: BLE001 - reported; retried after 1 s (never crashes the app)
                    self.error = f"{type(exc).__name__}: {exc}"
                    self.clear_target()  # a recovered connection needs a NEW session command
                    if isinstance(exc, PortBusy):
                        return
                    if device is not None:
                        try:
                            reason = self._take_hold()
                            if reason is not None:
                                self._physical_hold(device, reason)
                        except Exception:
                            log.exception("gripper physical hold after I/O failure failed")
                        finally:
                            latched = self._check_bad_replies(device)
                            device.close()
                            device = None
                        if latched:
                            return
                    self._halt.wait(1.0)
                    continue
                self._halt.wait(max(0.0, period - (time.monotonic() - t0)))
        finally:
            if device is not None:
                try:
                    reason = self._take_hold()
                    if reason is not None:
                        self._physical_hold(device, reason)
                except Exception as exc:
                    self.error = f"physical hold failed: {type(exc).__name__}: {exc}"
                    log.exception("gripper physical hold failed")
                finally:
                    device.close()

    def at_or_before(self, t_ns: int):
        """(t_ns, register) of the latest MEASURED sample at or before ``t_ns`` (the training binding)."""
        for t, s in reversed(list(self.samples)):
            if t <= t_ns:
                return t, float(s.position)
        raise LookupError("no gripper sample at or before the observation")

    def rows(self, after_ns: int, up_to_ns: int) -> list:
        return [(t, float(s.position)) for t, s in list(self.samples) if after_ns < t <= up_to_ns]
