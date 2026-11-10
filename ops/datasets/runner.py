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
coordinator decides "all" or an episode list; frozen once extraction has begun) and, for flows, ``groups.pw_flows``
plus a final-fit check above the margin.

Space: every download is admitted only if the space all admitted unfinished work may still claim plus its own claim
leaves the margin available. Claims are upper bounds: a shard archive claims its full expansion even under a
selection; the depth stream claims, once its first part is admitted, every remaining part plus the whole expansion
(the parts stay until the stream is published). Extraction stops when available space falls below the margin, and
all transfers are interrupted while available space cannot cover the margin plus the outstanding claims.

Publication: verified staging -> state ``publishing`` (saved) -> fsync + rename -> state ``extracted``/``published``
(saved) -> archive deleted. A restart in ``publishing`` resumes the renames instead of extracting again.
"""

from __future__ import annotations

import collections
import hashlib
import json
import os
import re
import signal
import threading
import time
import traceback
from pathlib import Path

from ops.datasets import archives, health, storage
from ops.datasets.catalog import GROUPS, HF_HOST
from ops.datasets.state import FINISHED, OwnerLock, StateStore, fsync_dir, read_json, write_json_atomic
from ops.datasets.transfer import AuthRequired, Interrupted, RateLimiter, TransferError, download, hash_file

DEFAULT_CONFIG = {
    "groups": {g: g != "pw_flows" for g in GROUPS},
    "pointworld_selection": "pending",
    "margin_fraction": 0.15,
    "schedule": "round_robin",  # or "depth_first": depth parts before RLDS and flow shards
    "max_attempts": 8,
    "max_extract_attempts": 3,
    "controller": {},
}
BIG_GROUPS = ("pw_depth", "rlds", "pw_flows")
TOKEN = re.compile(r"[A-Za-z0-9_.\-]{8,512}")
SHUTDOWN_JOIN_S = 600.0


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


def selection_fingerprint(selection) -> str | None:
    if selection == "pending":
        return None
    if selection is None:
        return "all"
    return hashlib.sha256("\n".join(sorted(selection["episodes"])).encode()).hexdigest()


def temp_path(root: Path, item: dict) -> Path:
    return root / ".downloads" / "tmp" / (item["id"].replace("/", "__") + ".part")


def token_fingerprint(token: str | None) -> str | None:
    return hashlib.sha256(token.encode()).hexdigest()[:16] if token else None


def staged_depth_bytes(root: Path) -> int:
    folder = Path(root) / ".staging" / "pw_depth" / "depth_320x180"
    total = 0
    try:
        with os.scandir(folder) as entries:
            for entry in entries:
                try:
                    total += entry.stat().st_size
                except FileNotFoundError:  # renamed by a concurrent publication
                    pass
    except FileNotFoundError:
        return 0
    return total


def on_disk_bytes(root: Path, items: list[dict], state_items: dict, inflight: dict | None = None) -> dict:
    """Bytes of unfinished items already on disk: held archives (downloaded/publishing) and partial temp files."""
    out = {}
    for item in items:
        status = state_items.get(item["id"], {}).get("status", "pending")
        if status in ("done", "extracted"):
            continue
        if status in ("downloaded", "publishing"):
            out[item["id"]] = int(item["size"])
        elif inflight and item["id"] in inflight:
            out[item["id"]] = inflight[item["id"]]["on_disk"]
        else:
            temp = temp_path(root, item)
            if temp.exists():
                out[item["id"]] = temp.stat().st_size
    return out


class Runner:
    def __init__(self, root: Path, tools_sha: str = "", probe=None, clock=time.monotonic):
        self.root = Path(root)
        self.state_dir = self.root / ".state"
        self.plan = read_json(self.state_dir / "plan.json")
        if not self.plan:
            raise SystemExit("no plan.json: run the plan command first")
        self.items = self.plan["items"]
        self.by_id = {i["id"]: i for i in self.items}
        self.depth = sorted((i for i in self.items if i["group"] == "pw_depth"), key=lambda i: i["archive"]["index"])
        self.load_state()
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
        self.workers: list[tuple[threading.Thread, threading.Event]] = []
        self.worker_serial = 0
        self.lanes: dict[str, dict] = {}
        self.staged = {"pw_depth": 0}
        self.recent_errors = collections.deque(maxlen=20)
        self.rates = collections.deque(maxlen=200)  # (monotonic time, bytes_total)
        self.space_wait = None
        self.paused_for_space = False
        self.config = load_config(self.state_dir)
        self.cfg = {**health.DEFAULTS, **self.config.get("controller", {})}
        self.control = health.Control(pool=int(self.cfg["pool_initial"]))
        self.selection = "pending"
        self.margin = storage.DEFAULT_MARGIN
        self.gates = {}

    def load_state(self) -> None:
        self.state = StateStore(self.state_dir / "state.json", self.by_id)

    # ---- logging -------------------------------------------------------------------------------------------------
    def log(self, event: str, **fields) -> None:
        line = json.dumps({"time": time.time(), "event": event, **fields}, sort_keys=True)
        with self.log_lock, open(self.state_dir / "progress.jsonl", "a") as out:
            out.write(line + "\n")

    # ---- configuration and gates ---------------------------------------------------------------------------------
    def reload_config(self) -> None:
        self.config = load_config(self.state_dir)
        self.cfg = {**health.DEFAULTS, **self.config.get("controller", {})}
        self.margin = storage.effective_margin(float(self.config["margin_fraction"]),
                                               health.mount_entry(str(self.root))[0])
        low, high = int(self.cfg["pool_min"]), max(int(self.cfg["pool_min"]), int(self.cfg["pool_max"]))
        self.control.pool = min(high, max(low, self.control.pool))
        gates = {}
        try:
            self.selection = resolve_selection(self.config["pointworld_selection"], self.plan, self.root)
        except OSError as error:
            self.selection = "pending"
            gates["selection"] = f"unreadable selection: {error}"
        frozen = self.state.section("selection_lock")
        if self.selection == "pending":
            gates["pw_flows"] = "pointworld_selection pending (coordinator decision)"
            gates["pw_depth_extract"] = "pointworld_selection pending (coordinator decision)"
        elif frozen and frozen.get("fingerprint") != selection_fingerprint(self.selection):
            reason = f"selection differs from the one extraction started with ({frozen.get('value')}); revert it"
            gates["pw_flows"] = gates["pw_depth_extract"] = reason
        elif not self.config["groups"].get("pw_flows"):
            gates["pw_flows"] = "group disabled"
        else:
            fit = self.fit()
            if not fit["final_fits"]:
                gates["pw_flows"] = f"final footprint exceeds the margin by {fit['deficit']} bytes"
        token = self.hf_token(gates)
        self.gates = gates
        current = token_fingerprint(token)
        for item_id in self.state.by_status("blocked"):
            # retry a gated file only with a different credential than the one it was refused with
            if token and self.by_id[item_id].get("auth") == "hf" and self.state.items[item_id].get("token") != current:
                self.state.update(item_id, status="pending", error=None)

    def freeze_selection(self) -> None:
        if not self.state.section("selection_lock"):
            self.state.set_section("selection_lock", {"fingerprint": selection_fingerprint(self.selection),
                                                      "value": self.config["pointworld_selection"],
                                                      "time": time.time()})
            self.state.save(force=True)

    def fit(self) -> dict:
        groups = [g for g in GROUPS if self.config["groups"].get(g)]
        selection = None if self.selection in (None, "pending") else self.selection
        return storage.plan_fit(self.items, groups, storage.fs_usage(self.root), self.margin, self.state.items,
                                selection, on_disk_bytes(self.root, self.items, self.state.items, self.inflight),
                                {"pw_depth": self.staged["pw_depth"]})

    def idle(self) -> float:
        return min(5.0, float(self.cfg["interval_s"]))

    def group_open(self, group: str) -> bool:
        return bool(self.config["groups"].get(group)) and group not in self.gates

    def hf_token(self, gates: dict | None = None) -> str | None:
        """The token, or None if absent, readable by others or malformed (control characters would otherwise reach
        an HTTP header error message)."""
        path = self.state_dir / "secrets" / "hf_token"
        gates = self.gates if gates is None else gates
        try:
            if path.stat().st_mode & 0o077:
                gates["hf_token"] = "secrets/hf_token must not be readable by others (chmod 600)"
                return None
            token = path.read_text().strip()
        except FileNotFoundError:
            return None
        if not TOKEN.fullmatch(token):
            gates["hf_token"] = "secrets/hf_token is malformed (one token, no spaces or control characters)"
            return None
        return token

    def auth_for(self, item: dict):
        if item.get("auth") != "hf":
            return None
        token = self.hf_token({})
        if not token:
            raise AuthRequired("gated source: no usable Hugging Face token at .state/secrets/hf_token")
        return (HF_HOST, f"Bearer {token}")

    # ---- space accounting ----------------------------------------------------------------------------------------
    @staticmethod
    def claim_expansion(item: dict) -> int:
        """Upper bound of an archive's extracted bytes (a selection only shrinks the real need)."""
        return int(item["archive"]["expanded"]) if "archive" in item else 0

    def partial_bytes(self, item: dict) -> int:
        info = self.inflight.get(item["id"])
        if info is not None:
            return info["on_disk"]
        temp = temp_path(self.root, item)
        return temp.stat().st_size if temp.exists() else 0

    def stream_active(self) -> bool:
        if self.state.section("stream").get("status") in ("publishing", "published"):
            return False
        return any(self.state.items[i["id"]]["status"] != "pending" for i in self.depth) or self.staged["pw_depth"] > 0

    def stream_claim(self) -> int:
        """Everything the depth stream may still add: remaining bytes of every part not yet downloaded plus the
        expansion not yet staged."""
        if self.state.section("stream").get("status") in ("publishing", "published"):
            return 0
        remaining = sum(int(i["size"]) - self.partial_bytes(i) for i in self.depth
                        if self.state.items[i["id"]]["status"] in ("pending", "downloading"))
        return max(0, remaining) + max(0, sum(self.claim_expansion(i) for i in self.depth) - self.staged["pw_depth"])

    def outstanding(self) -> int:
        total = 0
        for item_id, info in list(self.inflight.items()):
            if self.by_id[item_id]["group"] != "pw_depth":
                total += max(0, int(self.by_id[item_id]["size"]) - info["on_disk"])
        for item_id in self.state.by_status("downloading", "downloaded"):
            item = self.by_id[item_id]
            if item.get("archive", {}).get("kind") == "single":  # staged members are counted again: conservative
                total += self.claim_expansion(item)
        return total + (self.stream_claim() if self.stream_active() else 0)

    def growth(self, item: dict) -> int:
        if item["group"] == "pw_depth":
            return 0 if self.stream_active() else self.stream_claim()
        claim = int(item["size"]) - self.partial_bytes(item)
        if item.get("archive", {}).get("kind") == "single":
            claim += self.claim_expansion(item)
        return claim

    def space(self) -> tuple[int, int, int]:
        """(avail, outstanding, margin); the claims are read before the free space so a concurrent write is counted
        twice rather than not at all."""
        with self.counter_lock:
            outstanding = self.outstanding()
        usage = storage.fs_usage(self.root)
        return usage["avail"], outstanding, storage.margin_bytes(usage["size"], self.margin)

    def space_low(self) -> bool:
        usage = storage.fs_usage(self.root)
        return usage["avail"] < storage.margin_bytes(usage["size"], self.margin)

    # ---- scheduling ----------------------------------------------------------------------------------------------
    def candidates(self) -> list[dict]:
        now = time.time()
        ready = [i for i in self.items if self.group_open(i["group"])
                 and self.state.items[i["id"]]["status"] == "pending"
                 and self.state.items[i["id"]].get("not_before", 0) <= now and i["id"] not in self.inflight]
        small = [i for i in ready if i["group"] not in BIG_GROUPS]
        queues = {g: [i for i in ready if i["group"] == g] for g in BIG_GROUPS}
        first = []
        if self.config.get("schedule") == "depth_first":
            first = queues.pop("pw_depth")
        order = list(queues)
        start = int(now) % len(order)
        order = order[start:] + order[:start]
        interleaved = []
        for k in range(max((len(q) for q in queues.values()), default=0)):
            interleaved += [queues[g][k] for g in order if k < len(queues[g])]
        return small + first + interleaved

    def next_item(self) -> dict | None:
        """The first candidate whose claim fits; candidates that do not fit are skipped (a small file waiting for
        space must not block a depth part that would let the stream finish and free its parts)."""
        if self.paused_for_space:
            return None
        with self.sched_lock:
            avail, outstanding, margin = self.space()
            rejected = None
            for item in self.candidates():
                claim = self.growth(item)
                if storage.admissible(avail, outstanding, claim, margin):
                    with self.counter_lock:
                        self.inflight[item["id"]] = {"on_disk": self.partial_bytes(item), "started": time.time(),
                                                     "fetched": 0}
                    self.state.update(item["id"], status="downloading")
                    return item
                rejected = rejected or (item["id"], claim)
            if rejected and self.space_wait != rejected[0]:
                self.space_wait = rejected[0]
                self.log("space_wait", item=rejected[0], avail=avail, outstanding=outstanding, claim=rejected[1],
                         margin=margin)
        return None

    def on_progress(self, item_id: str, delta: int, on_disk: int) -> None:
        with self.counter_lock:
            self.bytes_total += delta
            info = self.inflight.get(item_id)
            if info is not None:
                info["on_disk"] = on_disk
                info["fetched"] += delta

    def download_one(self, item: dict, stop: threading.Event) -> None:
        item_id, temp = item["id"], temp_path(self.root, item)
        temp.parent.mkdir(parents=True, exist_ok=True)
        record = self.state.get(item_id)
        try:
            download(item, temp, limiter=self.limiter, on_progress=lambda d, n: self.on_progress(item_id, d, n),
                     should_stop=lambda: stop.is_set() or self.stop_event.is_set(), auth=self.auth_for(item))
            self.install(item, temp)
            self.log("verified", item=item_id, status=self.state.items[item_id]["status"], size=item["size"])
        except Interrupted:
            self.state.update(item_id, status="pending")
        except AuthRequired as error:
            self.state.update(item_id, status="blocked", error=str(error), token=token_fingerprint(self.hf_token({})))
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
            with self.counter_lock:
                self.inflight.pop(item_id, None)

    def install(self, item: dict, verified: Path) -> None:
        """Atomic rename of a verified, fsynced file to its destination; the state follows."""
        final = self.root / item["dest"]
        final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(verified, final)
        fsync_dir(final.parent)
        status = "downloaded" if "archive" in item else "done"
        self.state.update(item["id"], status=status, error=None, verified=item["checksum"], completed=time.time())

    def worker(self, stop: threading.Event) -> None:
        while not stop.is_set() and not self.stop_event.is_set():
            item = self.next_item()
            if item is None:
                stop.wait(self.idle())
                continue
            self.download_one(item, stop)

    def set_pool(self, size: int) -> None:
        """Exactly ``size`` workers that are not stopping (a stopping worker finishes in the background)."""
        self.workers = [(t, s) for t, s in self.workers if t.is_alive()]
        active = [(t, s) for t, s in self.workers if not s.is_set()]
        for _, stop in active[size:]:
            stop.set()
        for _ in range(size - len(active)):
            stop = threading.Event()
            self.worker_serial += 1
            thread = threading.Thread(target=self.worker, args=(stop,), name=f"download-{self.worker_serial}",
                                      daemon=True)
            self.workers.append((thread, stop))
            thread.start()

    # ---- extraction lanes ----------------------------------------------------------------------------------------
    def pointworld_root(self) -> Path:
        return self.root / "pointworld-droid" / "droid"

    def extraction_stop(self) -> bool:
        return self.stop_event.is_set() or self.space_low()

    def archive_lane(self) -> None:
        while not self.stop_event.is_set():
            ready = [self.by_id[i] for i in self.state.by_status("publishing", "downloaded")
                     if self.by_id[i].get("archive", {}).get("kind") == "single"
                     and (self.by_id[i]["group"] == "pw_cameras" or "pw_flows" not in self.gates
                          or self.state.items[i]["status"] == "publishing")]
            if not ready or self.space_low():
                self.stop_event.wait(self.idle())
                continue
            self.extract_single(ready[0])

    def extract_single(self, item: dict) -> None:
        item_id, archive = item["id"], self.root / item["dest"]
        name = item["archive"].get("shard", "cameras")
        staging = self.root / ".staging" / item["group"] / name
        self.lanes[item_id] = {"started": time.time()}
        try:
            members, expected = None, None
            if item["group"] == "pw_flows":
                self.freeze_selection()
                selection = None if self.selection in (None, "pending") else self.selection
                members = self.plan["flow_members"][name]
                if selection is not None:
                    members = sorted(set(members) & selection["episodes"])
                expected = item["archive"]["expanded"] if selection is None else None
            if self.state.items[item_id]["status"] == "downloaded":
                archives.extract_stream(archives.file_chunks([archive], self.extraction_stop), staging,
                                        item["archive"]["kinds"], None if expected is not None or members is None
                                        else set(members), self.extraction_stop)
                if item["group"] == "pw_flows":
                    report = archives.verify_flow_shard(staging, members, expected)
                else:
                    report = archives.verify_cameras(staging, item["archive"]["members"], item["archive"]["expanded"])
                self.state.update(item_id, status="publishing", extracted=report)
                self.state.save(force=True)
            for kind in item["archive"]["kinds"]:
                archives.publish(staging, kind, self.pointworld_root())
            if item["group"] == "pw_flows":
                report = archives.verify_published(self.pointworld_root(), "flows-fs-optimized",
                                                   [f"{m}_flows.h5" for m in members], expected)
            else:
                report = self.state.items[item_id].get("extracted", {})
            archives.clear_staging(staging)
            self.state.update(item_id, status="extracted", extracted=report)
            self.state.save(force=True)
            archive.unlink(missing_ok=True)
            self.log("extracted", item=item_id, **report)
        except archives.ExtractionStopped:
            pass
        except (archives.ExtractionError, OSError, ValueError) as error:
            attempts = int(self.state.get(item_id).get("extract_attempts", 0)) + 1
            status = self.state.items[item_id]["status"]
            if attempts >= int(self.config["max_extract_attempts"]):
                status = "failed"
            self.state.update(item_id, status=status, extract_attempts=attempts, error=f"extract: {error}"[:500])
            self.log("extract_error", item=item_id, attempts=attempts, error=str(error)[:500])
            if status != "failed":
                self.stop_event.wait(60.0)
        finally:
            self.lanes.pop(item_id, None)

    def depth_parts(self):
        """Paths of the depth parts in order; waits for each to be downloaded and verified; a part that has
        exhausted its attempts ends the stream with an error."""
        for item in self.depth:
            while (status := self.state.items[item["id"]]["status"]) != "downloaded":
                if status == "failed":
                    raise archives.ExtractionError(f"{item['id']} failed to download (run retry)")
                if self.extraction_stop():
                    return
                self.lanes["stream"]["waiting_for"] = item["id"]
                self.stop_event.wait(self.idle())
            self.lanes["stream"]["reading"] = item["id"]
            yield self.root / item["dest"]

    def stream_lane(self) -> None:
        staging = self.root / ".staging" / "pw_depth"
        if not self.depth:
            return
        while not self.stop_event.is_set():
            stream = self.state.section("stream")
            if stream.get("status") == "published":
                self.finish_stream()
                return
            if stream.get("status") == "failed":
                return
            if stream.get("status") != "publishing" and (
                    "pw_depth_extract" in self.gates or self.state.items[self.depth[0]["id"]]["status"] != "downloaded"
                    or self.space_low()):
                self.stop_event.wait(self.idle())
                continue
            self.lanes["stream"] = {"started": time.time()}
            try:
                if stream.get("status") != "publishing":
                    self.freeze_selection()
                    selection = None if self.selection in (None, "pending") else self.selection
                    archives.extract_stream(archives.file_chunks(self.depth_parts(), self.extraction_stop), staging,
                                            ["depth_320x180"], None if selection is None else selection["episodes"],
                                            self.extraction_stop)
                    flows = {e for members in self.plan["flow_members"].values() for e in members}
                    report = archives.verify_depth(staging, flows if selection is None
                                                   else flows & selection["episodes"])
                    stream = {"status": "publishing", "report": report, "time": time.time()}
                    self.state.set_section("stream", stream)
                    self.state.save(force=True)
                archives.publish(staging, "depth_320x180", self.pointworld_root())
                self.state.set_section("stream", {**stream, "status": "published", "time": time.time()})
                self.state.save(force=True)
                self.log("stream_published", **stream["report"])
            except archives.ExtractionStopped:
                if self.stop_event.is_set():
                    return
                self.log("stream_paused_for_space")
            except (archives.ExtractionError, OSError, ValueError) as error:
                attempts = int(stream.get("attempts", 0)) + 1
                failed = attempts >= int(self.config["max_extract_attempts"]) or "failed to download" in str(error)
                self.state.set_section("stream", {**stream, "attempts": attempts, "error": str(error)[:500],
                                                  **({"status": "failed"} if failed else {})})
                self.state.save(force=True)
                self.log("stream_error", attempts=attempts, failed=failed, error=str(error)[:500])
                if failed:
                    return
                self.stop_event.wait(60.0)
            finally:
                self.lanes.pop("stream", None)

    def finish_stream(self) -> None:
        for item in self.depth:
            if self.state.items[item["id"]]["status"] != "extracted":
                self.state.update(item["id"], status="extracted")
        self.state.save(force=True)
        for item in self.depth:
            (self.root / item["dest"]).unlink(missing_ok=True)
        archives.clear_staging(self.root / ".staging" / "pw_depth")

    # ---- reconciliation, status, main loop -----------------------------------------------------------------------
    def reconcile(self) -> dict:
        """After a crash, a reboot or a retry: unfinished records whose destination is already complete and verifies
        against the official checksum are restored instead of downloaded again; finished records whose files are
        missing go back to pending; archives whose content was published are deleted."""
        report = collections.Counter()
        for item in self.items:
            record = self.state.items[item["id"]]
            final = self.root / item["dest"]
            complete = final.is_file() and final.stat().st_size == int(item["size"])
            if record["status"] in ("downloading", "pending", "failed") and complete:
                if hash_file(final, item["checksum"], int(item["size"])):
                    self.state.update(item["id"], status="downloaded" if "archive" in item else "done", error=None)
                    report["published_file_restored"] += 1
                    continue
            if record["status"] == "downloading":
                self.state.update(item["id"], status="pending")
                report["downloading_to_pending"] += 1
            elif record["status"] in ("done", "downloaded", "publishing") and not complete:
                self.state.update(item["id"], status="pending", error="file missing at reconcile")
                report["missing_to_pending"] += 1
            elif record["status"] == "extracted" and final.exists():
                final.unlink()
                report["leftover_archive_deleted"] += 1
        tmp = self.root / ".downloads" / "tmp"
        report["partial_temp_files"] = len(list(tmp.glob("*.part"))) if tmp.is_dir() else 0
        staging = self.root / ".staging"
        report["staging_dirs"] = len(list(staging.glob("*/*"))) if staging.is_dir() else 0
        self.state.save(force=True)
        return dict(report)

    def throughput(self, window: float) -> float | None:
        now = self.clock()
        points = [p for p in self.rates if now - p[0] <= window]
        if len(points) < 2 or points[-1][0] == points[0][0]:
            return None
        return (points[-1][1] - points[0][1]) / (points[-1][0] - points[0][0])

    def status(self, sample: dict | None, baseline: dict, action: tuple[str, str] | None) -> dict:
        avail, outstanding, margin = self.space()
        usage = storage.fs_usage(self.root)
        groups = {}
        remaining = 0
        for group in GROUPS:
            members = [i for i in self.items if i["group"] == group]
            counts = collections.Counter(self.state.items[i["id"]]["status"] for i in members)
            done = sum(int(i["size"]) for i in members if self.state.items[i["id"]]["status"] in
                       ("done", "downloaded", "publishing", "extracted"))
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
            "last_action": action, "gates": self.gates,
            "fs": {**usage, "margin": margin, "above_margin": avail - margin, "outstanding_claims": outstanding,
                   "paused_for_space": self.paused_for_space},
            "groups": groups,
            "inflight": [{"item": k, "on_disk": v["on_disk"], "size": self.by_id[k]["size"],
                          "rate_bps": v["fetched"] / max(1e-3, time.time() - v["started"])}
                         for k, v in list(self.inflight.items())],
            "lanes": {k: dict(v) for k, v in list(self.lanes.items())},
            "stream": self.state.section("stream"), "staged_depth_bytes": self.staged["pw_depth"],
            "selection_lock": self.state.section("selection_lock"),
            "throughput_bps": {"1min": self.throughput(60.0), "10min": rate, "1h": self.throughput(3600.0)},
            "remaining_open_download_bytes": remaining,
            "eta_s": remaining / rate if rate else None, "health": sample, "baseline": baseline,
            "errors_total": self.errors, "recent_errors": list(self.recent_errors)[-5:],
            "space_wait": self.space_wait, "selection": self.config["pointworld_selection"],
            "margin_fraction": self.margin, "schedule": self.config.get("schedule"),
        }

    def finished(self) -> bool:
        statuses = {r["status"] for r in self.state.items.values()}
        return statuses <= set(FINISHED) | {"failed"} and not self.inflight

    def tick(self, probe, baseline: dict, errors_seen: int) -> tuple[tuple[str, str], int, dict]:
        self.reload_config()
        self.staged["pw_depth"] = staged_depth_bytes(self.root)
        with self.counter_lock:
            total, errors = self.bytes_total, self.errors
        self.rates.append((self.clock(), total))
        sample = probe.sample(total, errors - errors_seen)
        work_left = len(self.candidates()) + len(self.inflight)
        action = health.decide(self.control, sample, baseline, self.cfg, time.time(), work_left)
        if action[0] != "hold":
            self.log("control", action=action[0], reason=action[1], pool=self.control.pool,
                     cap_bps=self.control.cap_bps)
        if self.limiter.rate != self.control.cap_bps:
            self.limiter.set_rate(self.control.cap_bps)
        status = self.status(sample, baseline, action)
        # our own writes stop while the free space cannot cover the margin plus everything already claimed (another
        # writer used the disk); partial files are kept and transfers resume when space returns
        paused = status["fs"]["above_margin"] < status["fs"]["outstanding_claims"]
        if paused != self.paused_for_space:
            self.paused_for_space = paused
            self.log("space_pause" if paused else "space_resume", above_margin=status["fs"]["above_margin"],
                     outstanding=status["fs"]["outstanding_claims"])
        self.set_pool(0 if paused else self.control.pool)
        with open(self.state_dir / "health.jsonl", "a") as out:
            out.write(json.dumps({"sample": sample, "pool": self.control.pool, "cap_bps": self.control.cap_bps,
                                  "action": action, "inflight": len(self.inflight),
                                  "above_margin": status["fs"]["above_margin"]}, sort_keys=True) + "\n")
        write_json_atomic(self.state_dir / "status.json", status, indent=1)
        self.state.save()
        return action, errors, status

    def run(self) -> int:
        if not self.lock.acquire(tools_sha=self.tools_sha, root=str(self.root)):
            print(json.dumps({"error": "another owner holds the lock", "owner": self.lock.owner()}))
            return 3
        lanes: list[threading.Thread] = []
        baseline, action = {}, None
        try:
            self.load_state()  # under the lock: no stale snapshot from before another owner or CLI mutation
            if threading.current_thread() is threading.main_thread():
                signal.signal(signal.SIGTERM, lambda *_: self.stop_event.set())
                signal.signal(signal.SIGINT, lambda *_: self.stop_event.set())
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
            errors_seen = 0
            while not self.stop_event.is_set():
                self.stop_event.wait(float(self.cfg["interval_s"]))
                if (self.state_dir / "STOP").exists():
                    self.log("stop_requested")
                    break
                try:
                    action, errors_seen, _ = self.tick(probe, baseline, errors_seen)
                except Exception as error:  # noqa: BLE001  (one bad interval must not end the owner)
                    self.log("tick_error", error=f"{type(error).__name__}: {error}"[:300],
                             where=traceback.format_exc(limit=3)[-600:])
                    continue
                if self.finished():
                    self.log("all_finished")
                    break
            return 0
        finally:
            self.stop_event.set()
            for _thread, stop in self.workers:
                stop.set()
            alive = [t for t, _ in self.workers] + lanes
            deadline = time.monotonic() + SHUTDOWN_JOIN_S
            for thread in alive:
                thread.join(timeout=max(0.0, deadline - time.monotonic()))
            stuck = [t.name for t in alive if t.is_alive()]
            self.state.save(force=True)
            if baseline or action:
                write_json_atomic(self.state_dir / "status.json",
                                  {**self.status(None, baseline, action), "stopped": True}, indent=1)
            self.log("owner_exit", pid=os.getpid(), stuck_threads=stuck)
            (self.state_dir / "STOP").unlink(missing_ok=True)
            if stuck:  # never release ownership while a writer may still run: the lock dies with the process
                os._exit(4)
            self.lock.release()
