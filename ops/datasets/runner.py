"""The single owner process: download workers, extraction lanes, space admission, health control, status.

Layout under ``root`` (``/data_8T/kaist/sunho/datasets`` on the remote host):

- ``droid/1.0.1/`` (RLDS, used in place), ``pointworld-droid/droid/{flows-fs-optimized,depth_320x180,cameras}/``
  (``data.pointworld_root``), ``pointworld-droid/{LICENSE.pdf,README.md}``, ``droid3d/`` (+ ``hand_eye/``);
- ``.downloads/tmp/`` partial files (resumed by byte range), ``.downloads/pointworld-droid/`` verified archives
  awaiting extraction, ``.staging/`` extraction staging (same file system: publication is a rename);
- ``.state/``: ``plan.json`` (immutable), ``config.json`` (hot-reloaded), ``state.json``, ``owner.lock`` /
  ``owner.json``, ``status.json``, ``progress.jsonl``, ``health.jsonl``, ``runner.out``, ``STOP`` (stop request),
  ``secrets/hf_token`` (optional, mode 0600, never logged).

Gates: the PointWorld flow shards and the depth extraction wait for ``pointworld_selection`` ("pending" until the
coordinator decides "all" or an episode list) and, for flows, ``groups.pw_flows`` plus a final-fit check above the
margin. Every download and extraction is admitted only if the space all admitted unfinished steps may still claim plus
its own claim leaves the margin available.
"""

from __future__ import annotations

import collections
import json
import os
import signal
import threading
import time
from pathlib import Path

from ops.datasets import archives, health, storage
from ops.datasets.catalog import GROUPS, HF_HOST
from ops.datasets.state import FINISHED, OwnerLock, StateStore, fsync_dir, read_json, write_json_atomic
from ops.datasets.transfer import AuthRequired, Interrupted, RateLimiter, TransferError, download

DEFAULT_CONFIG = {
    "groups": {g: g != "pw_flows" for g in GROUPS},
    "pointworld_selection": "pending",
    "margin_fraction": 0.15,
    "max_attempts": 8,
    "max_extract_attempts": 3,
    "controller": {},
}
BIG_GROUPS = ("pw_depth", "rlds", "pw_flows")


def load_config(state_dir: Path) -> dict:
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    stored = read_json(state_dir / "config.json", {}) or {}
    for key, value in stored.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            config[key].update(value)
        else:
            config[key] = value
    return config


def resolve_selection(value, plan: dict, root: Path) -> dict | None | str:
    """"pending" stays "pending"; "all" -> None; a path -> the selected flow episodes per shard and overall."""
    if value in ("pending", "all"):
        return value if value == "pending" else None
    path = Path(value) if Path(value).is_absolute() else root / value
    wanted = {line.strip() for line in path.read_text().splitlines() if line.strip()}
    per_shard = {shard: len(wanted.intersection(members)) for shard, members in plan["flow_members"].items()}
    total = sum(len(m) for m in plan["flow_members"].values())
    return {"episodes": wanted, "per_shard": per_shard, "fraction": sum(per_shard.values()) / max(1, total),
            "source": str(path)}


def temp_path(root: Path, item: dict) -> Path:
    return root / ".downloads" / "tmp" / (item["id"].replace("/", "__") + ".part")


class Runner:
    def __init__(self, root: Path, tools_sha: str = "", probe=None, clock=time.monotonic):
        self.root = Path(root)
        self.state_dir = self.root / ".state"
        self.plan = read_json(self.state_dir / "plan.json")
        if not self.plan:
            raise SystemExit("no plan.json: run the plan command first")
        self.items = self.plan["items"]
        self.by_id = {i["id"]: i for i in self.items}
        self.state = StateStore(self.state_dir / "state.json", self.by_id)
        self.tools_sha = tools_sha
        self.lock = OwnerLock(self.state_dir)
        self.limiter = RateLimiter(None)
        self.stop_event = threading.Event()
        self.clock = clock
        self.probe = probe
        self.counter_lock = threading.Lock()
        self.log_lock = threading.Lock()
        self.sched_lock = threading.Lock()
        self.bytes_total = 0
        self.errors = 0
        self.inflight: dict[str, dict] = {}
        self.workers: dict[int, tuple[threading.Thread, threading.Event]] = {}
        self.lanes: dict[str, dict] = {}
        self.staged = {"pw_depth": 0}
        self.recent_errors = collections.deque(maxlen=20)
        self.rates = collections.deque(maxlen=200)  # (monotonic time, bytes_total)
        self.space_wait = None
        self.config = load_config(self.state_dir)
        self.cfg = {**health.DEFAULTS, **self.config.get("controller", {})}
        self.control = health.Control(pool=int(self.cfg["pool_initial"]))
        self.selection = "pending"
        self.gates = {}

    # ---- logging -------------------------------------------------------------------------------------------------
    def log(self, event: str, **fields) -> None:
        line = json.dumps({"time": time.time(), "event": event, **fields}, sort_keys=True)
        with self.log_lock, open(self.state_dir / "progress.jsonl", "a") as out:
            out.write(line + "\n")

    # ---- configuration and gates ---------------------------------------------------------------------------------
    def reload_config(self) -> None:
        self.config = load_config(self.state_dir)
        self.cfg = {**health.DEFAULTS, **self.config.get("controller", {})}
        gates = {}
        try:
            self.selection = resolve_selection(self.config["pointworld_selection"], self.plan, self.root)
        except OSError as error:
            self.selection = "pending"
            gates["selection"] = f"unreadable selection: {error}"
        if self.selection == "pending":
            gates["pw_flows"] = "pointworld_selection pending (coordinator decision)"
            gates["pw_depth_extract"] = "pointworld_selection pending (coordinator decision)"
        elif not self.config["groups"].get("pw_flows"):
            gates["pw_flows"] = "group disabled"
        else:
            fit = self.fit()
            if not fit["final_fits"]:
                gates["pw_flows"] = f"final footprint exceeds the margin by {fit['deficit']} bytes"
        self.gates = gates
        token = self.hf_token()
        for item_id in self.state.by_status("blocked"):
            if token and self.by_id[item_id].get("auth") == "hf":
                self.state.update(item_id, status="pending", error=None)

    def fit(self) -> dict:
        groups = [g for g in GROUPS if self.config["groups"].get(g)]
        selection = None if self.selection in (None, "pending") else self.selection
        return storage.plan_fit(self.items, groups, storage.fs_usage(self.root), self.config["margin_fraction"],
                                self.state.items, selection)

    def idle(self) -> float:
        return min(5.0, float(self.cfg["interval_s"]))

    def group_open(self, group: str) -> bool:
        return bool(self.config["groups"].get(group)) and group not in self.gates

    def hf_token(self) -> str | None:
        path = self.state_dir / "secrets" / "hf_token"
        try:
            if path.stat().st_mode & 0o077:
                self.gates["hf_token"] = "secrets/hf_token must not be readable by others (chmod 600)"
                return None
            return path.read_text().strip() or None
        except FileNotFoundError:
            return None

    def auth_for(self, item: dict):
        if item.get("auth") != "hf":
            return None
        token = self.hf_token()
        if not token:
            raise AuthRequired("gated source: no Hugging Face token at .state/secrets/hf_token")
        return (HF_HOST, f"Bearer {token}")

    # ---- space accounting ----------------------------------------------------------------------------------------
    def expansion(self, item: dict) -> int:
        selection = None if self.selection in (None, "pending") else self.selection
        return storage.final_bytes(item, selection) if "archive" in item else 0

    def stream_reserved(self) -> int:
        depth = [i for i in self.items if i["group"] == "pw_depth"]
        if self.state.section("stream").get("status") == "published":
            return 0
        if not any(self.state.items[i["id"]]["status"] in ("downloading", "downloaded") for i in depth):
            return 0
        return max(0, sum(self.expansion(i) for i in depth) - self.staged["pw_depth"])

    def outstanding(self) -> int:
        total = 0
        for item_id, info in list(self.inflight.items()):
            total += max(0, int(self.by_id[item_id]["size"]) - info["on_disk"])
        for item_id in self.state.by_status("downloading", "downloaded"):
            item = self.by_id[item_id]
            if item.get("archive", {}).get("kind") == "single":  # staged members are counted again: conservative
                total += self.expansion(item)
        return total + self.stream_reserved()

    def growth(self, item: dict) -> int:
        temp = temp_path(self.root, item)
        have = temp.stat().st_size if temp.exists() else 0
        claim = int(item["size"]) - have
        kind = item.get("archive", {}).get("kind")
        if kind == "single":
            claim += self.expansion(item)
        elif kind == "stream" and not self.stream_reserved():
            claim += sum(self.expansion(i) for i in self.items if i["group"] == "pw_depth")
        return claim

    # ---- scheduling ----------------------------------------------------------------------------------------------
    def candidates(self) -> list[dict]:
        now = time.time()
        ready = [i for i in self.items if self.group_open(i["group"])
                 and self.state.items[i["id"]]["status"] == "pending"
                 and self.state.items[i["id"]].get("not_before", 0) <= now and i["id"] not in self.inflight]
        small = [i for i in ready if i["group"] not in BIG_GROUPS]
        queues = {g: [i for i in ready if i["group"] == g] for g in BIG_GROUPS}
        order = list(BIG_GROUPS)
        start = int(now) % len(order)
        order = order[start:] + order[:start]
        interleaved = []
        for k in range(max((len(q) for q in queues.values()), default=0)):
            interleaved += [queues[g][k] for g in order if k < len(queues[g])]
        return small + interleaved

    def next_item(self) -> dict | None:
        with self.sched_lock:
            usage = storage.fs_usage(self.root)
            margin = storage.margin_bytes(usage["size"], self.config["margin_fraction"])
            outstanding = self.outstanding()
            for item in self.candidates():
                claim = self.growth(item)
                if storage.admissible(usage["avail"], outstanding, claim, margin):
                    temp = temp_path(self.root, item)
                    self.inflight[item["id"]] = {"on_disk": temp.stat().st_size if temp.exists() else 0,
                                                 "started": time.time(), "fetched": 0}
                    self.state.update(item["id"], status="downloading")
                    self.space_wait = None
                    return item
                if self.space_wait != item["id"]:
                    self.space_wait = item["id"]
                    self.log("space_wait", item=item["id"], avail=usage["avail"], outstanding=outstanding,
                             claim=claim, margin=margin)
                return None
        return None

    def on_bytes(self, item_id: str, amount: int) -> None:
        with self.counter_lock:
            self.bytes_total += amount
            info = self.inflight.get(item_id)
            if info is not None:
                info["on_disk"] += amount
                info["fetched"] += amount

    def download_one(self, item: dict, stop: threading.Event) -> None:
        item_id, temp = item["id"], temp_path(self.root, item)
        temp.parent.mkdir(parents=True, exist_ok=True)
        record = self.state.get(item_id)
        try:
            download(item, temp, limiter=self.limiter, on_bytes=lambda n: self.on_bytes(item_id, n),
                     should_stop=lambda: stop.is_set() or self.stop_event.is_set(), auth=self.auth_for(item))
            final = self.root / item["dest"]
            final.parent.mkdir(parents=True, exist_ok=True)
            os.replace(temp, final)
            fsync_dir(final.parent)
            status = "downloaded" if "archive" in item else "done"
            self.state.update(item_id, status=status, error=None, verified=item["checksum"], completed=time.time())
            self.log("verified", item=item_id, status=status, size=item["size"])
        except Interrupted:
            self.state.update(item_id, status="pending")
        except AuthRequired as error:
            self.state.update(item_id, status="blocked", error=str(error))
            self.log("blocked", item=item_id, error=str(error))
        except (TransferError, OSError) as error:
            attempts = int(record.get("attempts", 0)) + 1
            with self.counter_lock:
                self.errors += 1
            self.recent_errors.append({"time": time.time(), "item": item_id, "error": str(error)[:300]})
            if attempts >= int(self.config["max_attempts"]):
                self.state.update(item_id, status="failed", attempts=attempts, error=str(error)[:500])
                self.log("failed", item=item_id, attempts=attempts, error=str(error)[:500])
            else:
                delay = min(900.0, 30.0 * 2 ** (attempts - 1))
                self.state.update(item_id, status="pending", attempts=attempts, error=str(error)[:500],
                                  not_before=time.time() + delay)
                self.log("retry", item=item_id, attempts=attempts, delay_s=delay, error=str(error)[:300])
        finally:
            self.inflight.pop(item_id, None)

    def worker(self, stop: threading.Event) -> None:
        while not stop.is_set() and not self.stop_event.is_set():
            item = self.next_item()
            if item is None:
                stop.wait(self.idle())
                continue
            self.download_one(item, stop)

    def set_pool(self, size: int) -> None:
        for index in sorted(self.workers):
            thread, stop = self.workers[index]
            if not thread.is_alive():
                del self.workers[index]
            elif index >= size:
                stop.set()
        for index in range(size):
            if index not in self.workers:
                stop = threading.Event()
                thread = threading.Thread(target=self.worker, args=(stop,), name=f"download-{index}", daemon=True)
                self.workers[index] = (thread, stop)
                thread.start()

    # ---- extraction lanes ----------------------------------------------------------------------------------------
    def pointworld_root(self) -> Path:
        return self.root / "pointworld-droid" / "droid"

    def archive_lane(self) -> None:
        while not self.stop_event.is_set():
            ready = [self.by_id[i] for i in self.state.by_status("downloaded")
                     if self.by_id[i].get("archive", {}).get("kind") == "single"
                     and (self.by_id[i]["group"] == "pw_cameras" or "pw_flows" not in self.gates)]
            if not ready:
                self.stop_event.wait(self.idle())
                continue
            self.extract_single(ready[0])

    def extract_single(self, item: dict) -> None:
        item_id, archive = item["id"], self.root / item["dest"]
        name = item["archive"].get("shard", "cameras")
        staging = self.root / ".staging" / item["group"] / name
        selection = None if self.selection in (None, "pending") else self.selection
        self.lanes[item_id] = {"started": time.time()}
        try:
            if item["group"] == "pw_flows":
                members = self.plan["flow_members"][name]
                episodes = None if selection is None else set(members) & selection["episodes"]
                archives.extract_stream(archives.file_chunks([archive], self.stop_event.is_set), staging,
                                        item["archive"]["kinds"], episodes, self.stop_event.is_set)
                report = archives.verify_flow_shard(staging, sorted(episodes) if episodes is not None else members,
                                                    item["archive"]["expanded"] if selection is None else None)
            else:
                archives.extract_stream(archives.file_chunks([archive], self.stop_event.is_set), staging,
                                        item["archive"]["kinds"], None, self.stop_event.is_set)
                report = archives.verify_cameras(staging, item["archive"]["members"], item["archive"]["expanded"])
            for kind in item["archive"]["kinds"]:
                archives.publish(staging, kind, self.pointworld_root())
            archives.clear_staging(staging)
            self.state.update(item_id, status="extracted", extracted=report)
            self.state.save(force=True)
            archive.unlink(missing_ok=True)
            self.log("extracted", item=item_id, **report)
        except archives.ExtractionStopped:
            pass
        except (archives.ExtractionError, OSError, ValueError) as error:
            attempts = int(self.state.get(item_id).get("extract_attempts", 0)) + 1
            status = "failed" if attempts >= int(self.config["max_extract_attempts"]) else "downloaded"
            self.state.update(item_id, status=status, extract_attempts=attempts, error=f"extract: {error}"[:500])
            self.log("extract_error", item=item_id, attempts=attempts, error=str(error)[:500])
            if status == "downloaded":
                self.stop_event.wait(60.0)
        finally:
            self.lanes.pop(item_id, None)

    def depth_parts(self):
        """Paths of the depth parts in order; waits for each to be downloaded and verified."""
        for item in sorted((i for i in self.items if i["group"] == "pw_depth"), key=lambda i: i["archive"]["index"]):
            while self.state.items[item["id"]]["status"] != "downloaded":
                if self.stop_event.is_set():
                    return
                self.lanes["stream"]["waiting_for"] = item["id"]
                self.stop_event.wait(self.idle())
            self.lanes["stream"]["reading"] = item["id"]
            yield self.root / item["dest"]

    def stream_lane(self) -> None:
        staging = self.root / ".staging" / "pw_depth"
        if not any(i["group"] == "pw_depth" for i in self.items):
            return
        while not self.stop_event.is_set():
            stream = self.state.section("stream")
            depth = [i for i in self.items if i["group"] == "pw_depth"]
            if stream.get("status") == "published":
                self.finish_stream(depth)
                return
            if "pw_depth_extract" in self.gates or self.state.items[depth[0]["id"]]["status"] != "downloaded":
                self.stop_event.wait(self.idle())
                continue
            selection = None if self.selection in (None, "pending") else self.selection
            self.lanes["stream"] = {"started": time.time()}
            try:
                archives.extract_stream(archives.file_chunks(self.depth_parts(), self.stop_event.is_set), staging,
                                        ["depth_320x180"], None if selection is None else selection["episodes"],
                                        self.stop_event.is_set)
                flows = {e for members in self.plan["flow_members"].values() for e in members}
                report = archives.verify_depth(staging, flows if selection is None else flows & selection["episodes"])
                archives.publish(staging, "depth_320x180", self.pointworld_root())
                self.state.set_section("stream", {"status": "published", "report": report, "time": time.time()})
                self.state.save(force=True)
                self.log("stream_published", **report)
            except archives.ExtractionStopped:
                return
            except (archives.ExtractionError, OSError, ValueError) as error:
                attempts = int(stream.get("attempts", 0)) + 1
                self.state.set_section("stream", {**stream, "attempts": attempts, "error": str(error)[:500]})
                self.log("stream_error", attempts=attempts, error=str(error)[:500])
                if attempts >= int(self.config["max_extract_attempts"]):
                    return
                self.stop_event.wait(60.0)
            finally:
                self.lanes.pop("stream", None)

    def finish_stream(self, depth: list[dict]) -> None:
        for item in depth:
            if self.state.items[item["id"]]["status"] != "extracted":
                self.state.update(item["id"], status="extracted")
        self.state.save(force=True)
        for item in depth:
            (self.root / item["dest"]).unlink(missing_ok=True)
        archives.clear_staging(self.root / ".staging" / "pw_depth")

    # ---- reconciliation, status, main loop -----------------------------------------------------------------------
    def reconcile(self) -> dict:
        report = collections.Counter()
        for item in self.items:
            record = self.state.items[item["id"]]
            final = self.root / item["dest"]
            if record["status"] == "downloading":
                self.state.update(item["id"], status="pending")
                report["downloading_to_pending"] += 1
            elif record["status"] in ("done", "downloaded"):
                if not final.is_file() or final.stat().st_size != int(item["size"]):
                    self.state.update(item["id"], status="pending", error="file missing at reconcile")
                    report["missing_to_pending"] += 1
            elif record["status"] == "extracted" and final.exists():
                final.unlink()
                report["leftover_archive_deleted"] += 1
        report["partial_temp_files"] = len(list((self.root / ".downloads" / "tmp").glob("*.part"))) \
            if (self.root / ".downloads" / "tmp").is_dir() else 0
        report["staging_dirs"] = len(list((self.root / ".staging").glob("*/*"))) \
            if (self.root / ".staging").is_dir() else 0
        self.state.save(force=True)
        return dict(report)

    def staged_bytes(self) -> int:
        folder = self.root / ".staging" / "pw_depth" / "depth_320x180"
        if not folder.is_dir():
            return 0
        with os.scandir(folder) as entries:
            return sum(e.stat().st_size for e in entries if e.is_file())

    def throughput(self, window: float) -> float | None:
        now = self.clock()
        points = [p for p in self.rates if now - p[0] <= window]
        if len(points) < 2 or points[-1][0] == points[0][0]:
            return None
        return (points[-1][1] - points[0][1]) / (points[-1][0] - points[0][0])

    def status(self, sample: dict | None, baseline: dict, action: tuple[str, str] | None) -> dict:
        usage = storage.fs_usage(self.root)
        margin = storage.margin_bytes(usage["size"], self.config["margin_fraction"])
        groups = {}
        remaining = 0
        for group in GROUPS:
            members = [i for i in self.items if i["group"] == group]
            counts = collections.Counter(self.state.items[i["id"]]["status"] for i in members)
            done = sum(int(i["size"]) for i in members if self.state.items[i["id"]]["status"] in
                       ("done", "downloaded", "extracted"))
            total = sum(int(i["size"]) for i in members)
            groups[group] = {"files": len(members), "status": dict(counts), "download_done": done,
                             "download_total": total, "enabled": bool(self.config["groups"].get(group)),
                             "gate": self.gates.get(group)}
            if self.group_open(group):
                remaining += total - done
        rate = self.throughput(600.0)
        return {
            "time": time.time(), "owner": {"pid": os.getpid(), "tools_sha": self.tools_sha},
            "pool": self.control.pool, "cap_bps": self.control.cap_bps, "control": self.control.as_dict(),
            "last_action": action, "gates": self.gates, "fs": {**usage, "margin": margin,
                                                               "above_margin": usage["avail"] - margin},
            "groups": groups,
            "inflight": [{"item": k, "on_disk": v["on_disk"], "size": self.by_id[k]["size"],
                          "rate_bps": v["fetched"] / max(1e-3, time.time() - v["started"])}
                         for k, v in list(self.inflight.items())],
            "lanes": {k: dict(v) for k, v in list(self.lanes.items())},
            "stream": self.state.section("stream"), "staged_depth_bytes": self.staged["pw_depth"],
            "throughput_bps": {"1min": self.throughput(60.0), "10min": rate, "1h": self.throughput(3600.0)},
            "remaining_open_download_bytes": remaining,
            "eta_s": remaining / rate if rate else None, "health": sample, "baseline": baseline,
            "errors_total": self.errors, "recent_errors": list(self.recent_errors)[-5:],
            "space_wait": self.space_wait, "selection": self.config["pointworld_selection"],
            "margin_fraction": self.config["margin_fraction"],
        }

    def finished(self) -> bool:
        statuses = {r["status"] for r in self.state.items.values()}
        return statuses <= set(FINISHED) | {"failed"} and not self.inflight

    def run(self) -> int:
        if not self.lock.acquire(tools_sha=self.tools_sha, root=str(self.root)):
            print(json.dumps({"error": "another owner holds the lock", "owner": self.lock.owner()}))
            return 3
        try:
            if threading.current_thread() is threading.main_thread():
                signal.signal(signal.SIGTERM, lambda *_: self.stop_event.set())
                signal.signal(signal.SIGINT, lambda *_: self.stop_event.set())
            (self.state_dir / "STOP").unlink(missing_ok=True)
            report = self.reconcile()
            self.reload_config()
            self.log("owner_start", pid=os.getpid(), tools_sha=self.tools_sha, reconcile=report, gates=self.gates,
                     fit=self.fit()["deficit"])
            probe = self.probe or health.Probe(str(self.root), self.cfg.get("ping_wan"))
            probe.sample(0, 0)
            baseline = probe.sample(0, 0)  # before any transfer
            baseline = {k: baseline[k] for k in ("wan_rtt_ms", "gw_rtt_ms", "psi_io", "psi_cpu", "psi_mem",
                                                 "sshd_rtt_ms", "disk_util")}
            self.state.set_section("baseline", baseline)
            lanes = [threading.Thread(target=self.archive_lane, name="archive-lane", daemon=True),
                     threading.Thread(target=self.stream_lane, name="stream-lane", daemon=True)]
            for lane in lanes:
                lane.start()
            self.set_pool(self.control.pool)
            action = None
            errors_seen = 0
            while not self.stop_event.is_set():
                self.stop_event.wait(float(self.cfg["interval_s"]))
                if (self.state_dir / "STOP").exists():
                    self.log("stop_requested")
                    self.stop_event.set()
                    break
                self.reload_config()
                self.staged["pw_depth"] = self.staged_bytes()
                with self.counter_lock:
                    total, errors = self.bytes_total, self.errors
                self.rates.append((self.clock(), total))
                sample = probe.sample(total, errors - errors_seen)
                errors_seen = errors
                work_left = len(self.candidates()) + len(self.inflight)
                action = health.decide(self.control, sample, baseline, self.cfg, time.time(), work_left)
                if action[0] != "hold":
                    self.log("control", action=action[0], reason=action[1], pool=self.control.pool,
                             cap_bps=self.control.cap_bps)
                if self.limiter.rate != self.control.cap_bps:
                    self.limiter.set_rate(self.control.cap_bps)
                status = self.status(sample, baseline, action)
                # another writer pushed the disk far below the margin: interrupt transfers (partials are kept)
                emergency = status["fs"]["above_margin"] < -status["fs"]["margin"] // 2
                if emergency:
                    self.log("space_emergency", above_margin=status["fs"]["above_margin"])
                self.set_pool(0 if emergency else self.control.pool)
                with open(self.state_dir / "health.jsonl", "a") as out:
                    out.write(json.dumps({"sample": sample, "pool": self.control.pool, "cap_bps": self.control.cap_bps,
                                          "action": action, "inflight": len(self.inflight),
                                          "above_margin": status["fs"]["above_margin"]}, sort_keys=True) + "\n")
                write_json_atomic(self.state_dir / "status.json", status, indent=1)
                self.state.save()
                if self.finished():
                    self.log("all_finished")
                    break
            self.stop_event.set()
            for thread, stop in list(self.workers.values()):
                stop.set()
                thread.join(timeout=120)
            for lane in lanes:
                lane.join(timeout=300)
            self.state.save(force=True)
            write_json_atomic(self.state_dir / "status.json", {**self.status(None, baseline, action), "stopped": True},
                              indent=1)
            self.log("owner_exit", pid=os.getpid())
            return 0
        finally:
            (self.state_dir / "STOP").unlink(missing_ok=True)
            self.lock.release()
