"""Host and network health signals and the adaptive pool/rate controller of the downloader.

Signals (all read on the remote host, standard library only): aggregate throughput; pressure stall information
(``/proc/pressure/{cpu,io,memory}`` "some avg10"); available memory; utilisation of the data disk
(``/proc/diskstats`` io ticks); ICMP round-trip time to an internet host and to the default gateway (the internet
path is the one the downloads load; the control channel from the local site reaches this host over the same
uplink, 10.10.10.x via 192.168.10.1); the smoothed RTT the kernel keeps for established sshd sessions
(informational: it only updates while a session carries data); transfer errors.

Controller (``decide``): no fixed bandwidth cap. Start with a pool of ``pool_initial`` files; while healthy, add one
file at a time and keep it only if aggregate throughput grows (>= ``min_gain``), otherwise revert and hold the
plateau; on degradation remove one file at a time; only if the pool is already at its minimum and degradation
persists, apply a per-process rate cap below the current rate, relaxed step-wise (and removed) once healthy again.
"""

from __future__ import annotations

import os
import re
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

DEFAULTS = {
    "interval_s": 20.0,
    "pool_initial": 3,
    "pool_min": 1,
    "pool_max": 8,
    "min_gain": 0.07,            # a scale-up must raise aggregate throughput by 7 %
    "plateau_hold_s": 900.0,     # after a reverted scale-up, wait this long before trying again
    "healthy_patience": 3,       # consecutive healthy samples before any scale-up or cap relaxation
    "degraded_patience": 2,      # consecutive degraded samples before acting
    "wan_rtt_factor": 2.5,       # degraded: internet RTT > factor x baseline AND > baseline + wan_rtt_add_ms
    "wan_rtt_add_ms": 60.0,
    "gw_rtt_ms": 10.0,           # degraded: gateway RTT above this
    "psi_io_some": 25.0,         # degraded: io pressure "some avg10" above this (percent)
    "psi_mem_some": 10.0,
    "mem_avail_min_frac": 0.05,
    "disk_util_max": 0.90,
    "errors_max": 3,             # degraded: this many transfer errors within one interval
    "cap_factor": 0.7,           # rate cap = factor x current rate (and x factor again while still degraded)
    "cap_relax": 1.3,
    "cap_min_bps": 5e6,
    "ping_wan": "8.8.8.8",
}


def pool_bounds(cfg: dict) -> tuple[int, int]:
    """(minimum, maximum) concurrent files, normalised the same way for reporting and execution."""
    low = max(1, int(cfg["pool_min"]))
    return low, max(low, int(cfg["pool_max"]))


def read_psi(kind: str, root: Path = Path("/proc/pressure")) -> float | None:
    try:
        text = (root / kind).read_text()
    except OSError:
        return None
    match = re.search(r"^some avg10=([0-9.]+)", text, re.M)
    return float(match.group(1)) if match else None


def meminfo(path: Path = Path("/proc/meminfo")) -> dict:
    values = {}
    for line in path.read_text().splitlines():
        key, _, rest = line.partition(":")
        values[key] = int(rest.split()[0]) * 1024
    return values


def _unescape(field: str) -> str:
    return re.sub(r"\\([0-7]{3})", lambda m: chr(int(m.group(1), 8)), field)


def mount_entry(path: str, mountinfo: Path = Path("/proc/self/mountinfo")) -> tuple[str, str | None]:
    """(mount point, source device) of the file system holding ``path``: the path is resolved first (symlinks and
    ``..`` cannot point the lookup at another file system) and mountinfo escapes (``\\040``) are decoded."""
    path = os.path.realpath(path)
    best, device = "", None
    for line in mountinfo.read_text().splitlines():
        left, _, right = line.partition(" - ")
        mount_point, source = _unescape(left.split()[4]), _unescape(right.split()[1])
        if (path == mount_point or path.startswith(mount_point.rstrip("/") + "/")) and len(mount_point) > len(best):
            best, device = mount_point, source
    return best, device


def mount_device(path: str, mountinfo: Path = Path("/proc/self/mountinfo")) -> str | None:
    """Kernel name (``nvme1n1p1``) of the block device holding ``path``."""
    device = mount_entry(path, mountinfo)[1]
    return device.rsplit("/", 1)[1] if device and device.startswith("/dev/") else None


def disk_io_ticks(device: str, path: Path = Path("/proc/diskstats")) -> int | None:
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) > 12 and fields[2] == device:
            return int(fields[12])  # milliseconds spent doing I/O
    return None


def default_gateway(path: Path = Path("/proc/net/route")) -> str | None:
    for line in path.read_text().splitlines()[1:]:
        fields = line.split()
        if len(fields) > 2 and fields[1] == "00000000":
            raw = bytes.fromhex(fields[2])[::-1]
            return ".".join(str(b) for b in raw)
    return None


def ping_ms(host: str | None, count: int = 5, run=subprocess.run) -> float | None:
    if not host:
        return None
    try:
        result = run(["ping", "-n", "-q", "-c", str(count), "-i", "0.2", "-W", "1", host], capture_output=True,
                     text=True, timeout=count + 5)
    except (OSError, subprocess.TimeoutExpired):
        return None
    match = re.search(r"= [0-9.]+/([0-9.]+)/", result.stdout)
    return float(match.group(1)) if match else None


def sshd_rtt_ms(run=subprocess.run) -> float | None:
    try:
        result = run(["ss", "-tin", "state", "established", "( sport = :22 )"], capture_output=True, text=True,
                     timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    values = [float(v) for v in re.findall(r"\brtt:([0-9.]+)/", result.stdout)]
    return statistics.median(values) if values else None


class Probe:
    """Samples the signals; disk utilisation and throughput are rates between consecutive samples."""

    def __init__(self, data_path: str, ping_wan: str | None, clock=time.monotonic):
        self.device = mount_device(data_path)
        self.ping_wan, self.gateway = ping_wan, default_gateway()
        self.clock = clock
        self._last = None

    def sample(self, total_bytes: int, errors: int) -> dict:
        now, ticks = self.clock(), disk_io_ticks(self.device) if self.device else None
        mem = meminfo()
        out = {
            "time": time.time(), "psi_cpu": read_psi("cpu"), "psi_io": read_psi("io"), "psi_mem": read_psi("memory"),
            "mem_avail_frac": mem.get("MemAvailable", 0) / max(1, mem.get("MemTotal", 1)),
            "wan_rtt_ms": ping_ms(self.ping_wan), "gw_rtt_ms": ping_ms(self.gateway, count=3),
            "sshd_rtt_ms": sshd_rtt_ms(), "errors": errors, "throughput_bps": None, "disk_util": None,
        }
        if self._last is not None:
            dt = max(1e-6, now - self._last[0])
            out["throughput_bps"] = (total_bytes - self._last[1]) / dt
            if ticks is not None and self._last[2] is not None:
                out["disk_util"] = min(1.0, (ticks - self._last[2]) / 1000.0 / dt)
        self._last = (now, total_bytes, ticks)
        return out


def degraded_reasons(sample: dict, baseline: dict, cfg: dict) -> list[str]:
    reasons = []
    wan, base = sample.get("wan_rtt_ms"), baseline.get("wan_rtt_ms")
    if wan is not None and base is not None and wan > cfg["wan_rtt_factor"] * base and wan > base + cfg["wan_rtt_add_ms"]:
        reasons.append(f"wan_rtt {wan:.0f} ms (baseline {base:.0f})")
    if wan is None and base is not None:
        reasons.append("wan_ping_lost")
    if (sample.get("gw_rtt_ms") or 0) > cfg["gw_rtt_ms"]:
        reasons.append(f"gateway_rtt {sample['gw_rtt_ms']:.1f} ms")
    if (sample.get("psi_io") or 0) > cfg["psi_io_some"]:
        reasons.append(f"psi_io {sample['psi_io']}")
    if (sample.get("psi_mem") or 0) > cfg["psi_mem_some"]:
        reasons.append(f"psi_mem {sample['psi_mem']}")
    if sample.get("mem_avail_frac", 1.0) < cfg["mem_avail_min_frac"]:
        reasons.append("low_memory")
    if (sample.get("disk_util") or 0) > cfg["disk_util_max"]:
        reasons.append(f"disk_util {sample['disk_util']:.2f}")
    if sample.get("errors", 0) >= cfg["errors_max"]:
        reasons.append(f"errors {sample['errors']}")
    return reasons


@dataclass
class Control:
    pool: int
    cap_bps: float | None = None
    healthy: int = 0
    degraded: int = 0
    plateau_until: float = 0.0
    trial_from: float | None = None   # mean throughput before the scale-up under evaluation
    history: list = field(default_factory=list)  # recent throughput samples since the last change

    def as_dict(self) -> dict:
        data = asdict(self)
        data["history"] = data["history"][-6:]
        return data


def decide(control: Control, sample: dict, baseline: dict, cfg: dict, now: float, work_left: int) -> tuple[str, str]:
    """Update ``control`` from one sample; returns (action, reason). Actions: hold, scale_up, scale_down,
    revert_scale_up, cap, cap_tighten, cap_relax, cap_remove."""
    reasons = degraded_reasons(sample, baseline, cfg)
    if sample.get("throughput_bps") is not None:
        control.history = (control.history + [sample["throughput_bps"]])[-12:]
    if reasons:
        control.degraded += 1
        control.healthy = 0
        if control.degraded < cfg["degraded_patience"]:
            return "hold", "degraded once: " + "; ".join(reasons)
        control.degraded = 0
        control.trial_from = None
        control.history = []
        rate = sample.get("throughput_bps") or 0.0
        if control.pool > cfg["pool_min"]:
            control.pool -= 1
            return "scale_down", "; ".join(reasons)
        if control.cap_bps is None:
            control.cap_bps = max(cfg["cap_min_bps"], cfg["cap_factor"] * rate)
            return "cap", "; ".join(reasons)
        control.cap_bps = max(cfg["cap_min_bps"], cfg["cap_factor"] * min(control.cap_bps, rate or control.cap_bps))
        return "cap_tighten", "; ".join(reasons)
    control.degraded = 0
    control.healthy += 1
    if control.healthy < cfg["healthy_patience"]:
        return "hold", "healthy"
    recent = control.history[-cfg["healthy_patience"]:]
    mean = statistics.fmean(recent) if recent else 0.0
    if control.cap_bps is not None:
        control.healthy = 0
        control.cap_bps *= cfg["cap_relax"]
        if mean and control.cap_bps > 2.0 * mean:
            control.cap_bps = None
            return "cap_remove", "healthy"
        return "cap_relax", "healthy"
    if control.trial_from is not None:
        before, control.trial_from = control.trial_from, None
        control.healthy = 0
        if mean < before * (1.0 + cfg["min_gain"]):
            control.pool = max(cfg["pool_min"], control.pool - 1)
            control.plateau_until = now + cfg["plateau_hold_s"]
            control.history = []
            return "revert_scale_up", f"throughput {mean / 1e6:.1f} MB/s vs {before / 1e6:.1f} MB/s before"
        return "hold", f"scale-up kept: {before / 1e6:.1f} -> {mean / 1e6:.1f} MB/s"
    if control.pool < cfg["pool_max"] and now >= control.plateau_until and work_left > control.pool and mean > 0:
        control.trial_from = mean
        control.pool += 1
        control.healthy = 0
        control.history = []
        return "scale_up", f"healthy at {mean / 1e6:.1f} MB/s"
    return "hold", "healthy"
