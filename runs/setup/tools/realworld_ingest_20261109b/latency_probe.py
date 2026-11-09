"""SSH-path latency probe (same method as the 2026-10-27 download-budget measurement: TCP connect time to the remote's
port 22), plus a fresh ``ssh ... echo ok`` round trip (new TCP connection + password auth + exec, i.e. what an
interactive login waits for). Records which campaign rsync transfers were running during the measurement.

  python -I latency_probe.py --label idle [--tcp 40] [--ssh 5] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path("/home/ws/ws/splatter4d")
OUT = Path("/home/ws/ws/realworld-ops/ingest/latency")


def running_transfers() -> list[str]:
    out = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            argv = Path("/proc", name, "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if argv and os.path.basename(argv[0].decode(errors="replace")) == "rsync":
            args = [a.decode(errors="replace") for a in argv if a]
            bw = next((a for a in args if a.startswith("--bwlimit")), "no-bwlimit")
            ops = [a for a in args[1:] if not a.startswith("-") and "ssh" not in a]
            direction = "upload" if ops and "@" in ops[-1] else "download" if any("@" in a for a in ops) else "?"
            out.append(f"pid {name} {direction} {bw} {ops[-1][-80:] if ops else ''}")
    return sorted(out)


def tcp_samples(host: str, n: int, gap: float) -> list[float]:
    vals = []
    for _ in range(n):
        t = time.perf_counter()
        try:
            with socket.create_connection((host, 22), timeout=10):
                vals.append((time.perf_counter() - t) * 1e3)
        except OSError:
            vals.append(float("inf"))
        time.sleep(gap)
    return vals


def ssh_samples(n: int, gap: float) -> list[float]:
    sys.path.insert(0, str(REPO))
    from ops.remote.access import RemoteClient

    c = RemoteClient()
    transport, args, i = [], c.ssh_args(), 0
    while i < len(args):
        if args[i] == "-o" and args[i + 1].split("=")[0] in ("ControlMaster", "ControlPersist", "ControlPath"):
            i += 2
            continue
        transport.append(args[i])
        i += 1
    cmd = [*transport, "-o", "ControlMaster=no", "-o", "ControlPath=none", c.target, "echo ok"]
    vals = []
    for _ in range(n):
        t = time.perf_counter()
        r = subprocess.run(cmd, env=c._environment(), capture_output=True, text=True, timeout=120)
        vals.append((time.perf_counter() - t) * 1e3 if r.stdout.strip() == "ok" else float("inf"))
        time.sleep(gap)
    return vals


def summary(v: list[float]) -> dict:
    ok = sorted(x for x in v if x != float("inf"))
    if not ok:
        return {"n": len(v), "failed": len(v)}
    return {"n": len(v), "failed": len(v) - len(ok), "median_ms": round(statistics.median(ok), 1),
            "p90_ms": round(ok[min(len(ok) - 1, int(0.9 * len(ok)))], 1), "min_ms": round(ok[0], 1),
            "max_ms": round(ok[-1], 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--host", default="192.168.10.22")
    ap.add_argument("--tcp", type=int, default=40)
    ap.add_argument("--ssh", type=int, default=5)
    ap.add_argument("--gap", type=float, default=1.0)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    started = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    before = running_transfers()
    tcp = tcp_samples(a.host, a.tcp, a.gap)
    ssh = ssh_samples(a.ssh, a.gap) if a.ssh else []
    doc = {"label": a.label, "started_at": started,
           "finished_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
           "tcp_connect_22": summary(tcp), "ssh_echo_fresh": summary(ssh) if ssh else None,
           "tcp_samples_ms": [round(x, 1) for x in tcp], "ssh_samples_ms": [round(x, 1) for x in ssh],
           "transfers_before": before, "transfers_after": running_transfers()}
    a.out.mkdir(parents=True, exist_ok=True)
    path = a.out / f"{a.label}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({k: doc[k] for k in ("label", "tcp_connect_22", "ssh_echo_fresh", "transfers_before")},
                     indent=1))
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
