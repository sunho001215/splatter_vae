"""Upload boost v2: worker dataset uploads local -> 192.168.10.22 (2026-11-09 ~21:40, item 3).

The original upload-budget.json (600 KiB/s) is NEVER modified. Downloads (200 KiB/s), other uploads and network
settings are unchanged. v1 history is read-only; ALL v2 state is in upload-boost/v2/state/. A valid short lease grants
one shared dataset-upload budget: original -> 8192 -> 16384 -> 32768 KiB/s -> explicit "UNCAPPED" (no --bwlimit).
0 in upload-budget.json always means pause. Entry requires a measured healthy window; step-up retains v1's 10-minute
dwell, >=2 transfers started+verified at that cap, >=10 echoes, fresh measurements and >=80% cap-bound throughput.
Recently healthy capped levels may be re-entered after idle; UNCAPPED must always be reached via the ladder.

Measurements use EXISTING connections only: mux-only SSH true (ProxyCommand=false; absent master is skipped), passive
ControlMaster TCP srtt and worker rsync TCP bytes acknowledged / retransmissions via sock_diag, plus worker events.
Coordinator ABSOLUTE limits over the last 8 samples at the level: echo p90 <=250 ms, NO echo >1000 ms or failed;
ControlMaster srtt p90 <=150 ms; data retransmissions / data segments <=0.02. No baseline-relative limits or majority
exemptions. A failed echo, transfer error/sha256 conflict, stall or outage is hard -> original + cooldown; other breaches
are soft -> last healthy level below the blamed cap, else original. Missing measurements never qualify as healthy and
restore after 300 s. Per-level failure memory expires after 30 minutes (two recent failures cap the ladder below it).

An activation begins when eligible work exists, records exact previous upload/download budget bytes + sha256 BEFORE
any grant, and ends at idle grace (10 min without eligible work OR uploads), hard failure, STOP/restore, controller exit,
or age 6 h. Expiry holds original for >=10 min, persisted across restart, before reactivation. Start/level/end snapshots
are appended and fsynced to activations.jsonl (activation id, start, previous settings, levels reached, end reason).
Each lease carries PID + /proc start ticks + host + boot_id, wall and CLOCK_BOOTTIME expiry (<=150 s). A dead/reused
controller, reboot, STOP, or expired lease means original. The independent 2-s worker watchdog kills and resumes rsync
with --partial --append-verify at a reduction, even during validation; raises wait for the next transfer. rsync gets
PR_SET_PDEATHSIG=SIGKILL with a parent-race check, plus normal kill-on-exit. A hung controller loses its lease.
Controller startup first rewrites stale active policy to restored and closes interrupted activation journal entries.
Retire ONLY when every task in ingest_worker.TASKS has >=TARGET_PER_TASK transferred+verified; v1 retirement is ignored.

  python -I upload_boost.py run
  python -I upload_boost.py status
  python -I upload_boost.py restore [--reason TEXT]
See README-v2.md for recovery and the coordinator's restart recipe from realworld_ingest_20261109b.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import fcntl
import hashlib
import json
import os
import re
import socket
import statistics
import struct
import subprocess
import sys
import time
import traceback
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path

TOOL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOL_DIR))
import ingest_worker as W  # noqa: E402

REMOTE_HOST, REMOTE_USER = "192.168.10.22", "compu"  # = ops.remote.access.REMOTE_HOST/REMOTE_USER (not secrets)
TARGET = f"{REMOTE_USER}@{REMOTE_HOST}"
ACCESS_DIR = W.REPO / "runs/remote/access"  # RemoteClient state dir: ControlPath cm-%C of the campaign's master
LEVELS = (0, *W.BOOST_LEVELS_KIB)  # 0 = original budget (no boost)
UNCAPPED = W.UNCAPPED
SAMPLE_S = 30.0
POLL_S = 5.0                 # passive upload-connection polls between samples (no network traffic)
IDLE_SAMPLE_S = 60.0
LEASE_S = 150.0              # renewed every sample by a live, healthy controller; worker refuses > W.BOOST_MAX_LEASE_S
DWELL_S = 600.0              # minimum time at a level before the next step up
MIN_VERIFIED_AT_LEVEL = 2    # transfers started and verified at the level's cap during the dwell
CAP_BOUND = 0.80             # median in-flight throughput must reach this fraction of the cap to step up
WINDOW = 8                   # last samples at this level; absolute bounds, no majority exemption
COOLDOWN_S = 1800.0          # bounded by failure memory; two recent failures cap the ladder below it
ENTRY_QUIET_S = 600.0        # no hard failure for this long before (re-)entering the boost
IDLE_GRACE_S = 600.0         # no eligible upload for this long -> original budget (idle)
FAILURE_MEMORY_S = 1800.0   # every failure expires, including persisted failures
MAX_ACTIVATION_S = 6 * 3600.0
EXPIRY_REST_S = 600.0
REQUALIFY_S = 6 * 3600.0     # a level proven healthy this recently may be re-entered directly after standby
ECHO_TIMEOUT_S = 10.0
SLOW_ECHO_MS = 1000.0        # any echo above this in the window is a breach
BASELINE_MIN_ECHOES = 20
HARD_EVENTS = {"transfer_error", "remote_unreachable", "remote_error", "pass_error",
               "remote_conflict"}  # remote_conflict includes a final-file sha256 mismatch after publish
STALL_SAMPLES = 6            # no bytes acknowledged for 6 samples (~3 min) while rsync runs = stall (rsync's own
                             # --timeout=300 ends a real stall later; a resume's prefix verification takes seconds)
MISSING_S = 300.0            # boosted without a successful echo / data-flow measurement this long -> original
ENTRY_SAMPLES = 6            # (re-)entry needs this many recent samples at the original budget, all measured, no breach


def now_iso(t: float | None = None) -> str:
    return W.now_iso(t)


def log(msg: str) -> None:
    print(f"[{now_iso()}] {msg}", flush=True)


def med(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 1) if xs else None


def pct(xs, q: float):
    xs = sorted(x for x in xs if x is not None)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


# ------------------------------------------------------------------------------------------------ passive TCP info


def tcp_flows(host: str = REMOTE_HOST, port: int = 22) -> dict[int, dict]:
    """{socket inode: tcp_info subset} of established IPv4 TCP connections to host:port (netlink sock_diag, the
    interface `ss` uses; read-only, no packets sent)."""
    sock = socket.socket(socket.AF_NETLINK, socket.SOCK_RAW, 4)  # NETLINK_SOCK_DIAG
    sock.settimeout(5.0)
    try:
        req = struct.pack("=BBBBI", socket.AF_INET, socket.IPPROTO_TCP, 1 << 1, 0, 1 << 1) + b"\0" * 48
        sock.sendto(struct.pack("=IHHII", 16 + len(req), 20, 0x301, 1, 0) + req, (0, 0))  # SOCK_DIAG_BY_FAMILY, DUMP
        out, want = {}, socket.inet_aton(host)
        while True:
            data, off = sock.recv(1 << 16), 0
            while off < len(data):
                length, kind = struct.unpack_from("=IH", data, off)
                if kind == 3:  # NLMSG_DONE
                    return out
                if kind == 2:
                    raise OSError("sock_diag error")
                body = data[off + 16:off + length]
                off += (length + 3) & ~3
                if struct.unpack_from("!H", body, 6)[0] != port or body[24:28] != want:
                    continue
                inode = struct.unpack_from("=I", body, 68)[0]
                a = 72
                while a + 4 <= len(body):
                    alen, atype = struct.unpack_from("=HH", body, a)
                    if alen < 4:
                        break
                    if atype == 2 and alen - 4 >= 160:  # INET_DIAG_INFO: struct tcp_info
                        info = body[a + 4:a + alen]
                        u = struct.unpack_from("=24I", info, 8)
                        out[inode] = {"srtt_ms": u[15] / 1000, "rttvar_ms": u[16] / 1000, "cwnd": u[18],
                                      "total_retrans": u[23],
                                      "bytes_acked": struct.unpack_from("=Q", info, 120)[0],
                                      "data_segs_out": struct.unpack_from("=I", info, 156)[0],
                                      "delivery_rate": struct.unpack_from("=Q", info, 160)[0],
                                      "busy_us": struct.unpack_from("=Q", info, 168)[0],
                                      "rwnd_limited_us": struct.unpack_from("=Q", info, 176)[0],
                                      "sndbuf_limited_us": struct.unpack_from("=Q", info, 184)[0]}
                    a += (alen + 3) & ~3
    finally:
        sock.close()


def socket_inodes(pid: int) -> set[int]:
    out = set()
    with contextlib.suppress(OSError):
        for fd in os.listdir(f"/proc/{pid}/fd"):
            with contextlib.suppress(OSError):
                m = re.fullmatch(r"socket:\[(\d+)\]", os.readlink(f"/proc/{pid}/fd/{fd}"))
                if m:
                    out.add(int(m.group(1)))
    return out


def children(pid: int) -> list[int]:
    out = []
    for name in os.listdir("/proc"):
        if name.isdigit():
            with contextlib.suppress(OSError, IndexError):
                if int(Path(f"/proc/{name}/stat").read_text().rsplit(")", 1)[1].split()[1]) == pid:
                    out.append(int(name))
    return out


def cpu_ticks(pid: int) -> int | None:
    try:
        f = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return int(f[11]) + int(f[12])  # utime + stime
    except (OSError, IndexError, ValueError):
        return None


def proc_comm(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/comm").read_text().strip()
    except OSError:
        return ""


# ------------------------------------------------------------------------------------------- control channel (mux)


def control_path() -> str:
    out = subprocess.run(["ssh", "-G", "-F", "/dev/null", "-o", f"ControlPath={ACCESS_DIR}/cm-%C", TARGET],
                         capture_output=True, text=True, timeout=10, stdin=subprocess.DEVNULL).stdout
    m = re.search(r"^controlpath (\S+)$", out, re.M)
    if not m:
        raise RuntimeError("cannot resolve the campaign ControlPath")
    return m.group(1)


def master_pid(cp: str) -> int | None:
    """pid of the live campaign ControlMaster (local socket query only)."""
    if not os.path.exists(cp):
        return None
    r = subprocess.run(["ssh", "-F", "/dev/null", "-o", f"ControlPath={cp}", "-O", "check", TARGET],
                       capture_output=True, text=True, timeout=10, stdin=subprocess.DEVNULL)
    m = re.search(r"pid=(\d+)", r.stderr)
    return int(m.group(1)) if r.returncode == 0 and m else None


def mux_echo_ms(cp: str) -> tuple[float | None, str]:
    """Round trip of `true` over the existing master (new channel, no new TCP connection, no authentication)."""
    cmd = ["ssh", "-F", "/dev/null", "-T", "-o", "BatchMode=yes", "-o", "ControlMaster=no", "-o", f"ControlPath={cp}",
           "-o", "ProxyCommand=false", TARGET, "true"]
    t = time.perf_counter()
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=ECHO_TIMEOUT_S, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return None, "timeout"
    ms = (time.perf_counter() - t) * 1e3
    return (round(ms, 1), "ok") if r.returncode == 0 else (None, f"exit {r.returncode}")


# ------------------------------------------------------------------------------------------------------- sampling


class Sampler:
    """One measurement per call: worker/transfer state, control-channel latency, passive TCP stats, new events."""

    def __init__(self, state_dir: Path):
        self.dir = state_dir
        self.cp = control_path()
        self.events_path = state_dir / "events.jsonl"
        self.events_off = self.events_path.stat().st_size if self.events_path.exists() else 0
        self.flow_prev = None   # (t, inode, bytes_acked, total_retrans, data_segs_out, cpu ticks) of the last poll
        self.flow_last = None   # latest tcp_info snapshot of the worker's upload connection
        self.acc: dict = {}     # cap -> accumulated {"bytes", "secs", "retrans", "segs", "cpu_s"} until the next sample

    def new_events(self) -> list[dict]:
        try:
            with open(self.events_path, "rb") as f:
                f.seek(self.events_off)
                chunk = f.read()
        except OSError:
            return []
        end = chunk.rfind(b"\n") + 1
        self.events_off += end
        out = []
        for line in chunk[:end].splitlines():
            with contextlib.suppress(ValueError):
                out.append(json.loads(line))
        return out

    def worker(self) -> dict:
        """Running ingest worker (version), in-flight rsync, counts of eligible work, base budget."""
        info = {"alive": False, "version": None, "rsync_pid": None, "kib_cap": None, "key": None}
        with contextlib.suppress(OSError, ValueError):
            pid_doc = json.loads((self.dir / "worker.pid").read_text())
            pid = int(pid_doc["pid"])
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes()
            info.update(alive=b"ingest_worker.py" in cmd and pid_doc.get("running", True) is not False,
                        version=pid_doc.get("worker"), pid=pid)
        with contextlib.suppress(OSError, ValueError, KeyError):
            cur = json.loads((self.dir / "rsync-current.json").read_text())
            if b"rsync" in Path(f"/proc/{int(cur['pid'])}/cmdline").read_bytes():
                info.update(rsync_pid=int(cur["pid"]), kib_cap=cur["kib_per_s"], key=cur.get("key"))
        status = load_json(self.dir / "status.json") or {}
        counts = status.get("counts") or {}
        info["remaining"] = max(0, (status.get("last_pass") or {}).get("pending_transfers", 0))
        info["counts"] = {t: {"valid": c.get("valid"), "transferred": c.get("transferred"), "target": c.get("target")}
                          for t, c in counts.items()}
        info["all_done"] = all_tasks_done(counts)
        info["base_kib"] = W.base_budget_kib(self.dir)
        return info

    def sample(self, measure: bool = True) -> dict:
        t = time.time()
        s = {"t": round(t, 1), "at": now_iso(t), **self.worker()}
        if not measure:
            self.flow_prev, self.flow_last = None, None
            return s
        mpid = master_pid(self.cp)
        s["master"] = mpid is not None
        if mpid:
            s["echo_ms"], s["echo"] = mux_echo_ms(self.cp)
            if s["echo_ms"] is None and master_pid(self.cp) is None:
                s["echo"], s["master"] = "master vanished", False  # not a path failure
        try:
            flows = tcp_flows()
        except OSError:
            flows = {}
        if mpid:
            ctrl = [flows[i] for i in socket_inodes(mpid) if i in flows]
            s["ctrl_srtt_ms"] = ctrl[0]["srtt_ms"] if ctrl else None
        self.poll_flows(flows)
        last = self.flow_last
        if last and s.get("rsync_pid"):
            s.update(data_srtt_ms=last["srtt_ms"], data_cwnd=last["cwnd"], data_retrans_total=last["total_retrans"],
                     data_rwnd_limited_us=last["rwnd_limited_us"], data_sndbuf_limited_us=last["sndbuf_limited_us"])
        return self.consume_rates(s)

    def consume_rates(self, s: dict) -> dict:
        """Move the accumulated upload-connection statistics into sample `s` (and reset them)."""
        acc = self.acc
        if acc:  # the cap with the most in-flight time since the last sample (completed uploads included)
            cap, a = max(acc.items(), key=lambda kv: kv[1]["secs"])
            if a["secs"] >= 5.0:  # in-flight time covered by consecutive polls of the same upload connection
                s.update(rate_cap=cap, rate_kib=round(a["bytes"] / 1024 / a["secs"], 1), rate_secs=round(a["secs"], 1),
                         cpu_frac=round(a["cpu_s"] / a["secs"], 3))
            s.update(retrans=sum(x["retrans"] for x in acc.values()), segs=sum(x["segs"] for x in acc.values()))
        self.acc = {}
        return s

    def poll_flows(self, flows: dict | None = None) -> None:
        """Passive poll (every POLL_S between samples): accumulate bytes acknowledged / retransmissions / CPU of the
        worker's upload connection, so transfers shorter than SAMPLE_S are still measured."""
        rp = cap = None
        with contextlib.suppress(OSError, ValueError, KeyError, TypeError):
            cur = json.loads((self.dir / "rsync-current.json").read_text())
            rp, cap = int(cur["pid"]), cur["kib_per_s"]
        ssh = [c for p in (children(rp) if rp else []) for c in [p, *children(p)] if proc_comm(c) == "ssh"]
        if flows is None and ssh:
            try:
                flows = tcp_flows()
            except OSError:
                flows = {}
        data = [(i, (flows or {})[i]) for p in ssh for i in socket_inodes(p) if i in (flows or {})]
        if not data:
            self.flow_prev, self.flow_last = None, None
            return
        t = time.time()
        inode, d = data[0]
        ticks = (cpu_ticks(rp) or 0) + sum(cpu_ticks(p) or 0 for p in ssh)
        p = self.flow_prev
        if p and p[1] == inode and t > p[0] and cap is not None:
            a = self.acc.setdefault(cap, {"bytes": 0, "secs": 0.0, "retrans": 0, "segs": 0, "cpu_s": 0.0})
            a["bytes"] += d["bytes_acked"] - p[2]
            a["retrans"] += d["total_retrans"] - p[3]
            a["segs"] += d["data_segs_out"] - p[4]
            a["cpu_s"] += (ticks - p[5]) / os.sysconf("SC_CLK_TCK")
            a["secs"] += t - p[0]
        self.flow_prev = (t, inode, d["bytes_acked"], d["total_retrans"], d["data_segs_out"], ticks)
        self.flow_last = d


# ----------------------------------------------------------------------------------------------------- controller


def load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def summarize(samples: list[dict], cap: int | None = None) -> dict:
    """Latency/throughput statistics over samples (those with an in-flight transfer at `cap` for the data columns)."""
    fly = [s for s in samples if s.get("kib_cap") and (cap is None or s["kib_cap"] == cap)]
    rated = [s for s in samples if s.get("rate_cap", s.get("kib_cap")) and
             (cap is None or s.get("rate_cap", s.get("kib_cap")) == cap)]
    segs = sum(s.get("segs") or 0 for s in rated)
    return {"n": len(samples), "n_in_flight": len(fly),
            "echo_ms": {"median": med(s.get("echo_ms") for s in samples), "p90": pct([s.get("echo_ms") for s in samples], .9),
                        "n": sum(s.get("echo_ms") is not None for s in samples),
                        "failed": sum(s.get("echo") not in (None, "ok", "master vanished") for s in samples)},
            "ctrl_srtt_ms": {"median": med(s.get("ctrl_srtt_ms") for s in samples),
                             "p90": pct([s.get("ctrl_srtt_ms") for s in samples], .9)},
            "data_srtt_ms": {"median": med(s.get("data_srtt_ms") for s in fly),
                             "p90": pct([s.get("data_srtt_ms") for s in fly], .9)},
            "rate_kib": {"median": med(s.get("rate_kib") for s in rated),
                         "p10": pct([s.get("rate_kib") for s in rated], .1),
                         "n": sum(s.get("rate_kib") is not None for s in rated)},
            "retrans_frac": round(sum(s.get("retrans") or 0 for s in rated) / segs, 5) if segs else None,
            "cpu_frac": med(s.get("cpu_frac") for s in rated)}


def run_span(transfers: list[dict], cap: int, max_gap: float = 120.0) -> tuple[float, list[float]]:
    """Wall time covered by back-to-back transfers at `cap` (a run breaks at another cap or a gap > max_gap) and the
    gaps inside runs; aggregate throughput = bytes / span (includes verification and per-transfer overhead)."""
    span, gaps, start, prev = 0.0, [], None, None
    for x in sorted(transfers, key=lambda x: x["t_start"]):
        same = x["cap"] == cap and x["final_cap"] == cap
        if same and prev is not None and start is not None and x["t_start"] - prev <= max_gap:
            gaps.append(round(x["t_start"] - prev, 1))
        elif same:
            if start is not None:
                span += prev - start
            start = x["t_start"]
        else:
            if start is not None:
                span += prev - start
            start = None
        prev = x["t_end"]
    if start is not None:
        span += prev - start
    return span, gaps


def limits(baseline: dict | None = None) -> dict:
    """Coordinator's v2 absolute bounds; baseline is retained for historical reports only."""
    return {"echo_ms": 250.0, "ctrl_srtt_ms": 150.0, "retrans_frac": 0.02, "slow_echo_ms": 1000.0}


def rank(level) -> int:
    return LEVELS.index(level)


def all_tasks_done(counts: dict) -> bool:
    return all(counts.get(task, {}).get("transferred", 0) >= W.TARGET_PER_TASK for task in W.TASKS)


def append_record(path: Path, doc: dict) -> None:
    with open(path, "a+") as f:
        f.seek(0, os.SEEK_END)
        if f.tell():
            f.seek(f.tell() - 1)
            if f.read(1) != "\n":
                f.write("\n")  # separate a torn crash write before appending a recoverable record
        f.write(json.dumps(doc, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def activation_records(bdir: Path) -> dict:
    latest = {}
    path = bdir / "activations.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            with contextlib.suppress(ValueError, KeyError):  # ignore only a torn trailing write
                rec = json.loads(line)
                latest[rec["activation_id"]] = rec
    return latest


def close_activation(bdir: Path, activation: dict | None, reason: str, now: float) -> None:
    """Called under policy.lock. Idempotent across force-restore, exit and startup recovery."""
    if not activation:
        return
    latest = activation_records(bdir).get(activation["activation_id"], activation)
    if latest.get("end_reason") is None:
        append_record(bdir / "activations.jsonl", {**latest, "event": "end", "end": now_iso(now),
                      "end_unix": now, "end_reason": reason})


def previous_settings(state_dir: Path) -> dict:
    out = {}
    for label, path in (("upload-budget.json", state_dir / "upload-budget.json"),
                        ("download-budget.json", ACCESS_DIR / "download-budget.json")):
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            out[label] = {"path": str(path), "exists": False, "sha256": None, "bytes_base64": None}
        else:
            out[label] = {"path": str(path), "exists": True, "sha256": hashlib.sha256(data).hexdigest(),
                          "bytes_base64": base64.b64encode(data).decode("ascii")}
    return out


class Controller:
    def __init__(self, state_dir: Path, sampler=None, clock=time.time):
        self.dir = state_dir
        self.bdir = state_dir / W.BOOST_STATE
        self.bdir.mkdir(parents=True, exist_ok=True)
        self.clock = clock
        self.sampler = sampler
        self.baseline = load_json(state_dir / "upload-boost/baseline.json")
        self.limits = limits()
        orig = load_json(state_dir / "upload-boost/original-policy.json") or {}
        self.original_kib = (orig.get("original") or {}).get("dataset_upload_kib_per_s")
        now = clock()
        self.identity = W.controller_identity()
        self.activation = None
        self.state = "idle"
        self.expired_until = 0.0
        self.expired_until_boot = 0.0
        self.level = 0
        self.level_since = now
        self.reason = "controller start"
        self.samples: deque = deque(maxlen=200)
        self.at_level: list[dict] = []      # samples since level_since
        self.transfers: list[dict] = []     # verified transfers (from events) with their starting cap
        self.starts: dict = {}              # key -> transfer_start event
        # quiet period: the latest hard failure in the worker's events / this controller's history (a new controller
        # does not need to wait ENTRY_QUIET_S again when the record shows it already passed)
        self.last_hard_at = last_hard_failure(state_dir, now)
        self.last_echo_at = self.last_ctrl_at = self.last_rate_at = self.last_idle_at = now
        self.eligible = True
        self.idle_since = None
        self.stalled = 0
        old_policy = load_json(self.bdir / "policy.json") or {}
        # Restore BEFORE any measurement or new grant. The controller.lock is held by CLI run.
        with open(self.bdir / "policy.lock", "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if old_policy.get("state") == "active":
                old_policy.update(state="restored", aggregate_kib_per_s=None, lease=None,
                                  reason="startup recovery: stale active policy after controller failure", updated_at=now_iso(now))
                W.atomic_write_json(self.bdir / "policy.json", old_policy)
            close_activation(self.bdir, old_policy.get("activation"), "startup recovery: interrupted controller", now)
            for rec in activation_records(self.bdir).values():
                close_activation(self.bdir, rec, "startup recovery: interrupted controller", now)
        self.expired_until = old_policy.get("expired_until_unix", 0.0)
        if old_policy.get("controller", {}).get("boot_id") == self.identity["boot_id"]:
            self.expired_until_boot = old_policy.get("expired_until_boottime", 0.0)
        # A crash at the 6h boundary must not erase the mandatory rest interval.
        for rec in activation_records(self.bdir).values():
            if rec.get("end_reason", "").startswith("startup recovery") and rec["end_unix"] >= rec["start_unix"] + MAX_ACTIVATION_S:
                self.expired_until = max(self.expired_until, rec["end_unix"] + EXPIRY_REST_S)
        if type(self.original_kib) is not int or self.original_kib != W.DEFAULT_KIB:
            raise SystemExit("v1 original-policy.json must record the original 600 KiB/s budget")
        self.last_hard_at = max(self.last_hard_at, old_policy.get("last_hard_at", 0.0))
        mem = old_policy.get("memory") or {}
        fresh = now - (mem.get("saved_at") or 0) < REQUALIFY_S
        self.healthy = {(UNCAPPED if k == UNCAPPED else int(k)): v for k, v in (mem.get("healthy") or {}).items()} if fresh else {}
        self.failures = {(UNCAPPED if k == UNCAPPED else int(k)): [t for t in v if now - t < FAILURE_MEMORY_S]
                         for k, v in (mem.get("failures") or {}).items()} if fresh else {}
        self.history: list = []

    # -------------------------------------------------------------------------------------------- assessment
    def note_events(self, events: list[dict]) -> list[str]:
        hard = []
        for ev in events:
            kind = ev.get("event")
            if kind == "transfer_start":
                self.starts[ev.get("key")] = ev
            elif kind == "transfer_verified" and ev.get("sent_bytes"):
                st = self.starts.pop(ev.get("key"), None)
                if st:
                    self.transfers.append({"key": ev["key"], "cap": st.get("kib_per_s"), "final_cap": ev.get("kib_per_s"),
                                           "t_start": datetime.fromisoformat(st["t"]).timestamp(),
                                           "t_end": datetime.fromisoformat(ev["t"]).timestamp(),
                                           "bytes": ev["sent_bytes"], "rsync_s": ev.get("rsync_s"),
                                           "duration_s": ev.get("duration_s")})
            elif kind in HARD_EVENTS:
                detail = str(ev.get("error", ""))[:160]
                hard.append(f"{kind}{' (sha256 mismatch)' if 'sha256' in detail else ''}: {ev.get('key', '')} {detail}")
        return hard

    def assess(self, s: dict, hard: list[str]) -> tuple[list[str], list[str]]:
        """(hard, soft) reasons from this sample, the recent window and the worker's events."""
        hard = list(hard)
        if s.get("echo") == "timeout":
            hard.append(f"control channel: echo timed out (> {ECHO_TIMEOUT_S:.0f} s)")
        if s.get("echo", "ok") not in ("ok", "master vanished"):
            hard.append(f"control channel: echo failed ({s.get('echo')})")
        stalled = s.get("rate_kib") is not None and s["rate_kib"] <= 0
        self.stalled = self.stalled + 1 if stalled else 0
        if self.stalled >= STALL_SAMPLES:
            hard.append(f"transfer stalled: no bytes acknowledged for {self.stalled} samples ({s.get('key')})")
        soft = []
        lim = self.limits
        win = self.at_level[-WINDOW:]  # samples at the current level only (incl. this one)
        echo90 = pct([x.get("echo_ms") for x in win], .9)
        if echo90 is not None and echo90 > lim["echo_ms"]:
            soft.append(f"control echoes p90 {echo90} ms > {lim['echo_ms']} ms")
        if any((x.get("echo_ms") or 0) > lim["slow_echo_ms"] for x in win):
            soft.append("control echo > 1000 ms in window")
        srtt90 = pct([x.get("ctrl_srtt_ms") for x in win], .9)
        if srtt90 is not None and srtt90 > lim["ctrl_srtt_ms"]:
            soft.append(f"ControlMaster srtt p90 {srtt90} ms > {lim['ctrl_srtt_ms']} ms")
        segs = sum(x.get("segs") or 0 for x in win)
        if segs:
            frac = sum(x.get("retrans") or 0 for x in win) / segs
            if frac > lim["retrans_frac"]:
                soft.append(f"upload retransmissions {frac:.2%} > {lim['retrans_frac']:.2%}")
        return hard, soft

    def note_evidence(self, s: dict) -> None:
        """Times of the last successful measurements (kept across level changes)."""
        if s.get("echo_ms") is not None:
            self.last_echo_at = s["t"]
        if s.get("ctrl_srtt_ms") is not None:
            self.last_ctrl_at = s["t"]
        if s.get("rate_kib") is not None:
            self.last_rate_at = s["t"]
        if s.get("rsync_pid") is None:
            self.last_idle_at = s["t"]

    def missing_evidence(self, now: float) -> str | None:
        """While boosted: the health judgement needs fresh measurements (absence never counts as healthy)."""
        if now - self.last_echo_at > MISSING_S:
            return f"no control-channel measurement for {now - self.last_echo_at:.0f} s"
        if now - self.last_ctrl_at > MISSING_S:
            return f"no ControlMaster srtt measurement for {now - self.last_ctrl_at:.0f} s"
        if now - max(self.last_rate_at, self.last_idle_at) > MISSING_S:
            return f"uploads running but no data-flow measurement for {now - self.last_rate_at:.0f} s"
        return None

    def entry_ready(self, soft: list[str], now: float) -> tuple[bool, str]:
        """A demonstrated healthy window at the original budget: ENTRY_SAMPLES recent samples, all with an echo, no
        soft breach, no hard failure for ENTRY_QUIET_S (counted from controller start)."""
        win = [x for x in self.at_level[-ENTRY_SAMPLES:] if now - x["t"] <= ENTRY_SAMPLES * IDLE_SAMPLE_S]
        if now - self.last_hard_at < ENTRY_QUIET_S:
            return False, "hard failure too recent"
        if len(win) < ENTRY_SAMPLES or any(x.get("echo_ms") is None or x.get("ctrl_srtt_ms") is None for x in win):
            return False, f"entry window: {sum(x.get('echo_ms') is not None for x in win)}/{ENTRY_SAMPLES} measured"
        if soft:
            return False, "entry window unhealthy: " + "; ".join(soft)
        return True, f"healthy window of {ENTRY_SAMPLES} samples at the original budget"

    def prune_failures(self, now: float) -> None:
        self.failures = {k: [t for t in ts if now - t < FAILURE_MEMORY_S]
                         for k, ts in self.failures.items() if any(now - t < FAILURE_MEMORY_S for t in ts)}

    def blocked(self, kib, now: float) -> bool:
        self.prune_failures(now)
        return (any(now - t < COOLDOWN_S for t in self.failures.get(kib, []))
                or any(len(ts) >= 2 for k, ts in self.failures.items() if rank(k) <= rank(kib)))

    def ceiling(self, now: float):
        self.prune_failures(now)
        return max([k for k in LEVELS if k == 0 or not any(
            len(ts) >= 2 for j, ts in self.failures.items() if rank(j) <= rank(k))], key=rank)

    def fail(self, kib: int | str, now: float) -> None:
        self.failures.setdefault(kib, []).append(now)

    def step_up_ready(self, now: float) -> tuple[bool, str]:
        """The current level qualified as healthy: dwell, verified transfers at its cap, cap-bound throughput."""
        lvl = self.level
        if lvl == UNCAPPED:
            return False, "uncapped ceiling"
        done = [x for x in self.transfers if x["cap"] == lvl and x["final_cap"] == lvl and x["t_start"] >= self.level_since]
        rates = [s["rate_kib"] for s in self.at_level if s.get("rate_cap") == lvl and s.get("rate_kib") is not None]
        if now - self.level_since < DWELL_S:
            return False, "dwell"
        if any(s.get("echo_ms") is None or s.get("ctrl_srtt_ms") is None for s in self.at_level[-4:]):
            return False, "latest latency samples missing"
        if not any(s.get("rate_kib") is not None and s.get("rate_cap") == lvl for s in self.at_level[-4:]):
            return False, "no fresh throughput sample at this cap"
        if len(done) < MIN_VERIFIED_AT_LEVEL:
            return False, f"{len(done)}/{MIN_VERIFIED_AT_LEVEL} transfers verified at {lvl} KiB/s"
        if sum(s.get("echo_ms") is not None for s in self.at_level) < 10:
            return False, "too few latency samples"
        r = med(rates)
        if r is None or r < CAP_BOUND * lvl:
            return False, f"not cap-bound: median in-flight {r} KiB/s < {CAP_BOUND:.0%} of {lvl} (bottleneck below cap)"
        return True, f"healthy {now - self.level_since:.0f} s, {len(done)} transfers, median {r} KiB/s"

    def begin_activation(self, now: float) -> bool:
        # Record exact previous settings durably before any boost, even if entry stays at original.
        activation = {"activation_id": uuid.uuid4().hex, "start": now_iso(now), "start_unix": now,
                      "start_boottime": W.boottime(), "previous_settings": previous_settings(self.dir),
                      "levels_reached": [0], "end_reason": None}
        with open(self.bdir / "policy.lock", "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if (self.bdir / "STOP").exists():
                return False
            append_record(self.bdir / "activations.jsonl", {**activation, "event": "start"})
        self.activation = activation
        self.state = "standby"
        return True

    def end_activation(self, reason: str, now: float, state: str = "idle") -> None:
        if self.level:
            self.set_level(0, reason, {"t": now})
        self.reason, self.state = reason, state
        # Revoke the lease first, then finish the journal under the same restore lock.
        with open(self.bdir / "policy.lock", "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not (self.bdir / "STOP").exists():
                W.atomic_write_json(self.bdir / "policy.json", self.policy_doc(state, now))
            close_activation(self.bdir, self.activation, reason, now)
        self.activation = None
        self.at_level = []

    def decide(self, s: dict, events: list[dict]) -> None:
        now = s["t"]
        hard, soft = self.assess(s, self.note_events(events))
        self.prune_failures(now)
        if hard:
            self.last_hard_at = now
        flying = s.get("rsync_pid") is not None
        eligible = flying or s.get("remaining", 0) > 0
        self.eligible = eligible
        if flying:
            self.idle_since = None
        elif self.idle_since is None:
            self.idle_since = now
        idle_for = 0.0 if self.idle_since is None else now - self.idle_since
        self.note_evidence(s)
        stop_reason = None
        if all_tasks_done(s.get("counts") or {}):
            stop_reason = "all expected episodes transferred and verified"
        elif not s.get("alive"):
            stop_reason = "ingest worker not running"
        elif s.get("version") != W.WORKER_VERSION:
            stop_reason = f"worker {s.get('version')} cannot read the boost lease"
        elif s.get("base_kib") != self.original_kib:
            stop_reason = f"base budget {s.get('base_kib')} != recorded original {self.original_kib} (coordinator)"
        if self.activation and (now - self.activation["start_unix"] >= MAX_ACTIVATION_S or
                                W.boottime() - self.activation["start_boottime"] >= MAX_ACTIVATION_S):
            self.expired_until = now + EXPIRY_REST_S
            self.expired_until_boot = W.boottime() + EXPIRY_REST_S
            self.end_activation("activation expired after 6 h", now, "expired")
            return
        if stop_reason:
            self.end_activation(stop_reason, now)
            return
        if hard:
            if self.level:
                self.fail(self.blamed_level(), now)
            self.end_activation("hard: " + "; ".join(hard), now)
            return
        if now < self.expired_until or W.boottime() < self.expired_until_boot:
            self.state = "expired"
            return
        # v1's no-upload grace also handles pending files held in retry/backoff.
        if idle_for >= IDLE_GRACE_S:
            self.end_activation(f"no eligible dataset transfer (no upload for {idle_for:.0f} s)", now)
            return
        if not self.activation:
            if not eligible:
                self.state = "idle"
                return
            if not self.begin_activation(now):
                return
        if self.level and soft:
            blamed = self.blamed_level()
            self.fail(blamed, now)
            ok = [k for k, t in self.healthy.items() if rank(k) < rank(blamed) and rank(k) <= rank(self.level) and
                  now - t < REQUALIFY_S and not self.blocked(k, now)]
            target = max(ok, key=rank) if ok else 0
            self.set_level(target, "soft: " + "; ".join(soft) + f" (blamed cap {blamed})", s)
        elif self.level and self.missing_evidence(now):
            self.end_activation("missing evidence: " + self.missing_evidence(now), now)
        elif self.level == 0 and eligible:
            ready, detail = self.entry_ready(soft, now)
            ok = [k for k, t in self.healthy.items() if k != UNCAPPED and now - t < REQUALIFY_S and not self.blocked(k, now)]
            entry = max(ok, key=rank) if ok else LEVELS[1]
            if ready and not self.blocked(entry, now):
                self.set_level(entry, f"entry: {detail}", s)
        elif self.level:
            ready, detail = self.step_up_ready(now)
            if ready:
                self.healthy[self.level] = now
                nxt = LEVELS[rank(self.level) + 1]
                if not self.blocked(nxt, now):
                    self.set_level(nxt, f"step up: {detail}", s)
            elif detail.startswith("not cap-bound") and now - self.level_since >= DWELL_S:
                self.note_once(f"hold at {self.level}: {detail}")
        self.state = "active" if self.level else "standby" if self.activation else "idle"

    def blamed_level(self):
        caps = [x.get("kib_cap") for x in list(self.samples)[-WINDOW:] if x.get("kib_cap") in LEVELS[1:]]
        return max(caps, key=rank) if caps else self.level

    def note_once(self, text: str) -> None:
        if getattr(self, "_noted", None) != text:
            self._noted = text
            self.record({"event": "note", "level_kib": self.level, "detail": text})

    def set_level(self, kib: int | str, why: str, s: dict) -> None:
        prev = self.level
        stats = summarize(self.at_level, prev or None)
        self.record({"event": "level", "from_kib": prev, "to_kib": kib, "reason": why,
                     "seconds_at_previous": round(s["t"] - self.level_since), "previous_level_stats": stats})
        self.level, self.level_since, self.reason, self.at_level = kib, s["t"], why, []
        if self.activation and kib not in self.activation["levels_reached"]:
            self.activation["levels_reached"].append(kib)
            with open(self.bdir / "policy.lock", "a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                latest = activation_records(self.bdir).get(self.activation["activation_id"], {})
                if latest.get("end_reason") is None and not (self.bdir / "STOP").exists():
                    append_record(self.bdir / "activations.jsonl", {**self.activation, "event": "level"})

    def record(self, rec: dict) -> None:
        rec = {"t": now_iso(self.clock()), **rec}
        self.history.append(rec)
        with open(self.bdir / "history.jsonl", "a") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
        log(f"{rec['event']}: " + json.dumps({k: v for k, v in rec.items() if k not in ("t", "event",
                                                                                     "previous_level_stats")})[:600])

    # --------------------------------------------------------------------------------------------- policy record
    def policy_doc(self, state: str, now: float) -> dict:
        active = state == "active"
        return {"schema": W.BOOST_SCHEMA, "state": state, "aggregate_kib_per_s": self.level if active else None,
                "original_kib_per_s": self.original_kib, "levels_kib": list(W.BOOST_LEVELS_KIB),
                "ceiling_kib": self.ceiling(now), "scope": "ingest worker dataset uploads local->remote (one shared "
                "aggregate budget); downloads, other uploads and networking unchanged",
                "lease": {**self.identity, "renewed_at": now_iso(now), "expires_at": now_iso(now + LEASE_S),
                          "expires_at_unix": min(now + LEASE_S, self.activation["start_unix"] + MAX_ACTIVATION_S),
                          "expires_at_boottime": min(W.boottime() + LEASE_S,
                              self.activation["start_boottime"] + MAX_ACTIVATION_S)} if active else None,
                "activation": self.activation, "expired_until_unix": self.expired_until,
                "expired_until_boottime": self.expired_until_boot,
                "reason": self.reason, "level_since": now_iso(self.level_since), "limits": self.limits,
                "last_hard_at": self.last_hard_at,
                "controller": {**self.identity,
                               "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]},
                "memory": {"saved_at": now, "healthy": {str(k): v for k, v in self.healthy.items()},
                           "failures": {str(k): v for k, v in self.failures.items()}},
                "updated_at": now_iso(now)}

    def write_policy(self, state: str) -> bool:
        """Atomic write under policy.lock; refused once STOP exists (a force-restore wins over a running loop)."""
        with open(self.bdir / "policy.lock", "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if (self.bdir / "STOP").exists():
                return False
            W.atomic_write_json(self.bdir / "policy.json", self.policy_doc(state, self.clock()))
            return True

    def write_levels(self, s: dict) -> None:
        """Per-level measurement summary (samples + verified transfers) for reports."""
        out = {}
        hist = [json.loads(x) for x in (self.bdir / "samples.jsonl").read_text().splitlines()[-20000:]] \
            if (self.bdir / "samples.jsonl").exists() else []
        for cap in sorted({x.get("kib_cap") for x in hist if x.get("kib_cap")}, key=W.budget_order):
            at = [x for x in hist if x.get("kib_cap") == cap]
            xs = [x for x in self.transfers if x["cap"] == cap and x["final_cap"] == cap]
            span, gaps = run_span(self.transfers, cap)
            out[str(cap)] = {**summarize(at, cap), "transfers": len(xs),
                             "gap_between_transfers_s_median": med(gaps),
                             "transfer_rsync_kib_median": med(x["bytes"] / 1024 / x["rsync_s"] for x in xs if x["rsync_s"]),
                             "transfer_total_kib_median": med(x["bytes"] / 1024 / x["duration_s"] for x in xs
                                                              if x["duration_s"]),
                             "verify_overhead_s_median": med(x["duration_s"] - x["rsync_s"] for x in xs
                                                             if x["duration_s"] and x["rsync_s"]),
                             "aggregate_kib": round(sum(x["bytes"] for x in xs) / 1024 / span, 1) if span > 0 else None}
        W.atomic_write_json(self.bdir / "levels.json", {"updated_at": now_iso(s["t"]), "baseline": self.baseline,
                                                        "limits": self.limits, "by_cap_kib": out})

    # ----------------------------------------------------------------------------------------------------- loop
    def tick(self) -> str:
        busy = bool(self.level) or self.eligible  # keep measuring through short gaps between uploads
        try:
            s = self.sampler.sample(measure=busy)
            s["level_kib"] = self.level
            self.samples.append(s)
            self.at_level.append(s)
            self.decide(s, self.sampler.new_events())
            with open(self.bdir / "samples.jsonl", "a") as f:
                f.write(json.dumps(s, sort_keys=True) + "\n")
        except Exception as exc:  # noqa: BLE001 - an unmeasurable state is never boosted
            log("tick failed:\n" + traceback.format_exc())
            s = {"t": self.clock()}
            self.end_activation(f"controller error: {type(exc).__name__}: {exc}"[:300], s["t"])
        if all_tasks_done(s.get("counts") or {}):
            return "retire"
        state = self.state
        if (self.bdir / "STOP").exists() or not self.write_policy(state):
            return "stop"
        return state

    def run(self) -> int:
        self.write_policy("idle")
        self.record({"event": "controller_start", "pid": os.getpid(), "limits": self.limits,
                     "memory": {"healthy": {str(k): v for k, v in self.healthy.items()},
                           "failures": {str(k): v for k, v in self.failures.items()}}})
        last_levels = 0.0
        try:
            while True:
                if (self.bdir / "STOP").exists():
                    self.reason = "STOP file"
                    return 0
                state = self.tick()
                if state == "retire":
                    retire(self.dir, self)
                    return 0
                if state == "stop":
                    return 0
                if time.time() - last_levels > 300:
                    with contextlib.suppress(Exception):
                        self.write_levels(self.samples[-1])
                    last_levels = time.time()
                wait_s = SAMPLE_S if state == "active" or self.eligible else IDLE_SAMPLE_S
                end = time.time() + wait_s
                while time.time() < end:
                    time.sleep(min(POLL_S, max(0.0, end - time.time())))
                    if self.level or self.eligible:
                        with contextlib.suppress(Exception):
                            self.sampler.poll_flows()
        finally:
            with contextlib.suppress(Exception):
                if not (self.bdir / "retired.json").exists():
                    self.end_activation(f"controller exit ({self.reason})", self.clock(), "restored")
            with contextlib.suppress(Exception):
                self.write_levels(self.samples[-1] if self.samples else {"t": time.time()})
            self.record({"event": "controller_stop", "reason": self.reason})


def last_hard_failure(state_dir: Path, now: float, tail_bytes: int = 2 << 20) -> float:
    """Unix time of the latest hard failure within ENTRY_QUIET_S before `now`, from the tails of the worker's
    events.jsonl (HARD_EVENTS) and the controller's history.jsonl (hard/soft level drops); if a file cannot be read,
    `now` (a quiet period must then be observed)."""
    latest = 0.0
    for path, hit in ((state_dir / "events.jsonl", lambda r: r.get("event") in HARD_EVENTS),
                      (state_dir / W.BOOST_STATE / "history.jsonl", lambda r: r.get("event") == "level" and
                       str(r.get("reason", "")).startswith(("hard:", "soft:", "missing evidence")))):
        try:
            with open(path, "rb") as f:
                off = max(0, f.seek(0, 2) - tail_bytes)
                f.seek(off)
                lines = f.read().splitlines()[1 if off else 0:]  # drop a partial first line only
        except FileNotFoundError:
            continue
        except OSError:
            return now
        for line in lines:
            with contextlib.suppress(ValueError, TypeError, AttributeError):
                r = json.loads(line)
                if hit(r):
                    latest = max(latest, datetime.fromisoformat(r["t"]).timestamp())
    return latest if now - latest < ENTRY_QUIET_S else 0.0


def budget_sha(state_dir: Path) -> str | None:
    p = state_dir / "upload-budget.json"
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def retire(state_dir: Path, ctl: Controller) -> None:
    orig = load_json(state_dir / "upload-boost/original-policy.json") or {}
    recorded = ((orig.get("files") or {}).get(str(state_dir / "upload-budget.json")) or {}).get("sha256")
    if not ctl.samples or not all_tasks_done(ctl.samples[-1].get("counts") or {}):
        raise ValueError("retirement requires every TASKS entry at TARGET_PER_TASK verified")
    ctl.end_activation("retired: every task transferred and verified", ctl.clock(), "retired")
    ctl.level, ctl.reason = 0, "retired: every task transferred and verified"
    with open(ctl.bdir / "policy.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        W.atomic_write_json(ctl.bdir / "policy.json", ctl.policy_doc("retired", ctl.clock()))
    doc = {"retired_at": now_iso(), "counts": ctl.samples[-1].get("counts") if ctl.samples else None,
           "upload_budget_sha256": budget_sha(state_dir), "recorded_original_sha256": recorded,
           "original_restored": budget_sha(state_dir) == recorded,
           "note": "boost permanently retired; the worker uses upload-budget.json (original policy) from now on"}
    W.atomic_write_json(ctl.bdir / "retired.json", doc)
    ctl.record({"event": "retired", **{k: v for k, v in doc.items() if k != "counts"}})


# ------------------------------------------------------------------------------------------------------------ CLI


def cmd_observe(state_dir: Path, minutes: float, label: str, write_baseline: bool) -> int:
    """Measurement only (no policy change): samples every SAMPLE_S for `minutes`."""
    sampler = Sampler(state_dir)
    out = []
    end = time.time() + minutes * 60
    while time.time() < end:
        out.append(sampler.sample())
        nxt = time.time() + SAMPLE_S
        while time.time() < nxt:
            time.sleep(POLL_S)
            sampler.poll_flows()
    caps = sorted({s.get("kib_cap") for s in out if s.get("kib_cap")}, key=W.budget_order)
    doc = {"label": label, "started_at": out[0]["at"], "finished_at": out[-1]["at"], "kib_caps_seen": caps,
           **summarize(out, caps[0] if len(caps) == 1 else None), "method": __doc__.split("Measurements use")[1][:900]}
    bdir = state_dir / W.BOOST_STATE
    bdir.mkdir(parents=True, exist_ok=True)
    W.atomic_write_json(bdir / f"measure-{label}.json", {**doc, "samples": out})
    if write_baseline:
        if len(caps) != 1 or doc["echo_ms"]["n"] < BASELINE_MIN_ECHOES:
            raise SystemExit(f"baseline needs one cap in flight and >= {BASELINE_MIN_ECHOES} echoes: {caps} "
                             f"{doc['echo_ms']}")
        W.atomic_write_json(bdir / "baseline.json", {**doc, "kib_cap": caps[0]})
    print(json.dumps({k: v for k, v in doc.items() if k != "method"}, indent=1))
    return 0


def cmd_status(state_dir: Path) -> int:
    bdir = state_dir / W.BOOST_STATE
    pol = load_json(bdir / "policy.json") or {}
    base = W.base_budget_kib(state_dir)
    level, why = W.boost_level(state_dir, base)
    print(json.dumps({"in_effect_kib_per_s": level or base, "source": "boost" if level else "original/base",
                      "why": why, "policy_state": pol.get("state"), "policy_level": pol.get("aggregate_kib_per_s"),
                      "reason": pol.get("reason"), "level_since": pol.get("level_since"),
                      "lease": pol.get("lease"), "ceiling_kib": pol.get("ceiling_kib"),
                      "controller": load_json(bdir / "controller.pid"), "retired": (bdir / "retired.json").exists(),
                      "levels": (load_json(bdir / "levels.json") or {}).get("by_cap_kib")}, indent=1))
    return 0


def cmd_restore(state_dir: Path, reason: str) -> int:
    """Force-restore: STOP the controller and write a non-active record (the worker falls back within ~2 s)."""
    bdir = state_dir / W.BOOST_STATE
    bdir.mkdir(parents=True, exist_ok=True)
    with open(bdir / "policy.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        (bdir / "STOP").write_text(json.dumps({"at": now_iso(), "reason": reason}) + "\n")
        pol = load_json(bdir / "policy.json") or {"schema": W.BOOST_SCHEMA}
        pol.update(state="restored", aggregate_kib_per_s=None, lease=None, reason=f"force-restore: {reason}",
                   updated_at=now_iso())
        W.atomic_write_json(bdir / "policy.json", pol)
        for rec in activation_records(bdir).values():
            close_activation(bdir, rec, f"force-restore: {reason}", time.time())
    with open(bdir / "history.jsonl", "a") as f:
        f.write(json.dumps({"t": now_iso(), "event": "force_restore", "reason": reason}) + "\n")
    print(f"restored: original budget in effect; controller asked to stop ({bdir / 'STOP'})")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=("run", "status", "restore", "observe"))
    ap.add_argument("--state-dir", type=Path, default=W.STATE_DIR)
    ap.add_argument("--minutes", type=float, default=15.0)
    ap.add_argument("--label", default="observe")
    ap.add_argument("--write-baseline", action="store_true")
    ap.add_argument("--reason", default="manual")
    a = ap.parse_args(argv)
    if a.command == "status":
        return cmd_status(a.state_dir)
    if a.command == "restore":
        return cmd_restore(a.state_dir, a.reason)
    bdir = a.state_dir / W.BOOST_STATE
    if a.command == "run" and (bdir / "retired.json").exists():
        print("v2 boost retired: nothing to do")
        return 0
    os.setpriority(os.PRIO_PROCESS, 0, max(15, os.getpriority(os.PRIO_PROCESS, 0)))
    if a.command == "observe":
        return cmd_observe(a.state_dir, a.minutes, a.label, a.write_baseline)
    bdir.mkdir(parents=True, exist_ok=True)
    fd = open(bdir / "controller.lock", "a+")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"another boost controller is running ({load_json(bdir / 'controller.pid')})") from None
    import signal

    def stop(signum, _frame):
        raise SystemExit(f"signal {signum}")
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    ctl = Controller(a.state_dir)  # recover stale grants before Sampler can query any control channel
    ctl.sampler = Sampler(a.state_dir)
    W.atomic_write_json(bdir / "controller.pid", {"pid": os.getpid(), "started_at": now_iso(), "argv": sys.argv,
                                                  "code_sha256": hashlib.sha256(Path(__file__).read_bytes())
                                                  .hexdigest()[:16]})
    return ctl.run()


if __name__ == "__main__":
    sys.exit(main())
