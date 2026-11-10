"""Remote dataset downloader (ops/datasets): transfer/verification, catalog, storage, state, health control, archives,
the owner process end to end against a local HTTP server, the remote CLI and the deployment command."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from ops.datasets import archives, catalog, cli, control, health, runner, state, storage, transfer

REPO = Path(__file__).resolve().parents[1]
HDF5 = archives.HDF5_SIGNATURE


# ---- a local HTTP server with byte ranges, redirects and an authorization check ----------------------------------
class Server:
    def __init__(self):
        self.files: dict[str, bytes] = {}
        self.ignore_range: set[str] = set()
        self.token: str | None = None
        self.requests: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):  # noqa: N802
                outer.requests.append({"path": self.path, "headers": dict(self.headers)})
                path = self.path.split("?")[0]
                if path.startswith("/redirect/"):
                    target = f"http://localhost:{outer.port}/{path[len('/redirect/'):]}"
                    self.send_response(302)
                    self.send_header("Location", target)
                    self.end_headers()
                    return
                if path.startswith("/gated/") and self.headers.get("Authorization") != f"Bearer {outer.token}":
                    self.send_response(401)
                    self.end_headers()
                    return
                name = path.split("/", 2)[2] if path.startswith("/gated/") else path.lstrip("/")
                if name not in outer.files:
                    self.send_response(404)
                    self.end_headers()
                    return
                body = outer.files[name]
                start = 0
                header = self.headers.get("Range")
                if header and name not in outer.ignore_range:
                    start = int(header.split("=")[1].split("-")[0])
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{len(body) - 1}/{len(body)}")
                else:
                    self.send_response(200)
                self.send_header("Content-Length", str(len(body) - start))
                self.end_headers()
                self.wfile.write(body[start:])

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def url(self, name: str) -> str:
        return f"http://127.0.0.1:{self.port}/{name}"


@pytest.fixture
def server():
    srv = Server()
    yield srv
    srv.httpd.shutdown()


def md5(data: bytes) -> dict:
    return {"md5": hashlib.md5(data).hexdigest()}


def item_for(srv: Server, name: str, data: bytes, **extra) -> dict:
    srv.files[name] = data
    return {"id": f"t/{name}", "url": srv.url(name), "size": len(data), "checksum": md5(data), **extra}


def fetch(item, temp, stop=lambda: False, auth=None):
    return transfer.download(item, temp, limiter=transfer.RateLimiter(None), on_bytes=lambda n: None,
                             should_stop=stop, auth=auth)


# ---- transfer --------------------------------------------------------------------------------------------------
def test_verifier_kinds_match_published_checksums():
    data = b"hello\n"
    for checksum in ({"md5": hashlib.md5(data).hexdigest()}, {"sha256": hashlib.sha256(data).hexdigest()},
                     {"git_sha1": "ce013625030ba8dba906f756967f9e9ca394464a"}):  # `git hash-object` of "hello\n"
        verifier = transfer.Verifier(checksum, len(data))
        verifier.update(data)
        assert verifier.matches()
    pointer = transfer.LFS_POINTER.format(sha256=hashlib.sha256(data).hexdigest(), size=len(data)).encode()
    oid = hashlib.sha1(b"blob %d\0" % len(pointer) + pointer).hexdigest()
    verifier = transfer.Verifier({"lfs_pointer_sha1": oid}, len(data))
    verifier.update(data)
    assert verifier.matches()
    assert transfer.gcs_md5_hex(base64.b64encode(bytes.fromhex("00ff" * 8)).decode()) == "00ff" * 8


def test_download_resumes_a_partial_file_by_byte_range(server, tmp_path):
    data = os.urandom(3 << 20)
    item = item_for(server, "a.bin", data)
    temp = tmp_path / "a.part"
    temp.write_bytes(data[:1_000_000])
    assert fetch(item, temp) == len(data) - 1_000_000
    assert temp.read_bytes() == data
    assert server.requests[-1]["headers"]["Range"] == "bytes=1000000-"


def test_server_ignoring_the_range_restarts_from_zero(server, tmp_path):
    data = os.urandom(2 << 20)
    item = item_for(server, "b.bin", data)
    server.ignore_range.add("b.bin")
    temp = tmp_path / "b.part"
    temp.write_bytes(data[:500])
    assert fetch(item, temp) == len(data) and temp.read_bytes() == data


def test_checksum_mismatch_removes_the_file_and_interrupt_keeps_it(server, tmp_path):
    data = os.urandom(3 << 20)
    bad = item_for(server, "c.bin", data, checksum={"md5": "0" * 32})
    with pytest.raises(transfer.TransferError, match="checksum mismatch"):
        fetch(bad, tmp_path / "c.part")
    assert not (tmp_path / "c.part").exists()
    good = item_for(server, "d.bin", data)
    calls = []
    with pytest.raises(transfer.Interrupted):
        fetch(good, tmp_path / "d.part", stop=lambda: calls.append(1) or len(calls) > 1)
    assert 0 < (tmp_path / "d.part").stat().st_size < len(data)
    fetch(good, tmp_path / "d.part")
    assert (tmp_path / "d.part").read_bytes() == data


def test_authorization_goes_only_to_its_host(server, tmp_path, monkeypatch):
    data = b"x" * 1000
    plain = {"id": "p", "url": server.url("e.bin"), "size": len(data), "checksum": md5(data)}
    server.files["e.bin"] = data
    fetch(plain, tmp_path / "p.part", auth=("127.0.0.1", "Bearer secret"))
    assert "Authorization" not in server.requests[-1]["headers"]  # never over plain http
    monkeypatch.setattr(transfer, "AUTH_SCHEMES", ("https", "http"))  # the test server speaks http
    server.files["e.bin"] = data
    item = {"id": "e", "url": server.url("redirect/e.bin"), "size": len(data), "checksum": md5(data)}
    fetch(item, tmp_path / "e.part", auth=("127.0.0.1", "Bearer secret"))
    first, second = server.requests[-2:]
    assert first["headers"].get("Authorization") == "Bearer secret"
    assert "Authorization" not in second["headers"]  # localhost is another host
    server.token = "tok"
    gated = {"id": "g", "url": server.url("gated/e.bin"), "size": len(data), "checksum": md5(data)}
    with pytest.raises(transfer.AuthRequired):
        fetch(gated, tmp_path / "g.part")
    fetch(gated, tmp_path / "g.part", auth=("127.0.0.1", "Bearer tok"))


def test_rate_limiter_paces_to_the_rate():
    now = [0.0]
    slept = []

    def sleep(seconds):
        slept.append(seconds)
        now[0] += seconds

    limiter = transfer.RateLimiter(1000.0, clock=lambda: now[0], sleep=sleep)
    for _ in range(10):
        limiter.acquire(500)
    assert 4.0 <= now[0] <= 5.0  # 5000 bytes at 1000 B/s, the first burst free
    limiter.set_rate(None)
    before = now[0]
    limiter.acquire(10**9)
    assert now[0] == before


# ---- catalog -----------------------------------------------------------------------------------------------------
def test_pointworld_selection_takes_only_what_the_pipeline_reads():
    group = catalog.pointworld_group
    assert group("droid/depth_320x180/package.tar.zst.part-0251") == "pw_depth"
    assert group("droid/flows-fs-optimized/shard-000770/package.tar.zst.part-0000") == "pw_flows"
    assert group("droid/cameras/package.tar.zst.part-0000") == "pw_cameras"
    assert group(catalog.FLOW_MANIFEST) == group("LICENSE.pdf") == "pw_meta"
    for skipped in ("droid/confidence/package.tar.zst.part-0000", "recover_dataset_from_parts.sh", ".gitattributes"):
        assert group(skipped) is None
    masked = {"oid": "a" * 40, "lfs": {"oid": "*" * 64, "size": 3}}
    assert catalog.hf_checksum(masked) == {"lfs_pointer_sha1": "a" * 40}
    assert catalog.hf_checksum({"oid": "b" * 40}) == {"git_sha1": "b" * 40}
    assert catalog.hf_checksum({"oid": "b" * 40, "lfs": {"oid": "c" * 64}}) == {"sha256": "c" * 64}


def fake_sources():
    shards = [{"name": f"robotics/droid/1.0.1/droid_101-train.tfrecord-{k:05d}-of-02048", "size": "10",
               "md5Hash": base64.b64encode(b"\x01" * 16).decode(), "generation": "7"} for k in range(2048)]
    shards += [{"name": f"robotics/droid/1.0.1/{n}", "size": "5", "md5Hash": base64.b64encode(b"\x02" * 16).decode(),
                "generation": "7"} for n in ("dataset_info.json", "features.json")]
    manifest = {"shards": [{"shard_id": f"shard-{k:06d}", "member_count": 2, "raw_member_bytes": 300 + k,
                            "members": [f"droid/flows-fs-optimized/LAB+u{k}+2023-01-0{j + 1}-10h-00m-00s_flows.h5"
                                        for j in range(2)],
                            "parts": [{"size_bytes": 100 + k, "sha256": f"{k:064x}"}]} for k in range(3)]}
    manifest_bytes = json.dumps(manifest).encode()
    pw = [{"type": "file", "path": "LICENSE.pdf", "size": 9, "oid": "1" * 40},
          {"type": "file", "path": ".gitattributes", "size": 9, "oid": "2" * 40},
          {"type": "file", "path": "droid/confidence/package.tar.zst.part-0000", "size": 9, "oid": "3" * 40,
           "lfs": {"oid": "4" * 64}},
          {"type": "file", "path": catalog.FLOW_MANIFEST, "size": len(manifest_bytes), "oid": "5" * 40,
           "lfs": {"oid": hashlib.sha256(manifest_bytes).hexdigest()}},
          {"type": "file", "path": "droid/cameras/package.tar.zst.part-0000", "size": 50, "oid": "6" * 40,
           "lfs": {"oid": "7" * 64}}]
    pw += [{"type": "file", "path": f"droid/depth_320x180/package.tar.zst.part-{k:04d}", "size": 1000, "oid": "8" * 40,
            "lfs": {"oid": f"{k + 100:064x}"}} for k in range(2)]
    pw += [{"type": "file", "path": f"droid/flows-fs-optimized/shard-{k:06d}/package.tar.zst.part-0000",
            "size": 100 + k, "oid": "9" * 40, "lfs": {"oid": f"{k:064x}"}} for k in range(3)]
    d3d = [{"type": "file", "path": n, "size": 4, "oid": "a" * 40, "lfs": {"oid": "*" * 64}}
           for n in ("extrinsics.jsonl", "camera_intrinsics.jsonl")]
    d3d += [{"type": "file", "path": "README.md", "size": 3, "oid": "b" * 40},
            {"type": "file", "path": "pseudolabels/refined_all.npz", "size": 3, "oid": "c" * 40}]
    calls = []

    def get(url):
        calls.append(url)
        if "storage/v1" in url:
            if "pageToken" not in url:
                return {"items": shards[:1000], "nextPageToken": "p2"}, {}
            if "pageToken=p2" in url:
                return {"items": shards[1000:2000], "nextPageToken": "p3"}, {}
            return {"items": shards[2000:]}, {}
        if catalog.POINTWORLD_REPO in url:
            if "cursor" not in url:
                return pw[:4], {"link": f'<{url}&cursor=x>; rel="next"'}
            return pw[4:], {}
        if catalog.DROID3D_REPO in url:
            return d3d, {}
        if "api.github.com" in url:
            return {"sha": "d" * 40, "size": 12}, {}
        raise AssertionError(url)

    return get, (lambda url, checksum, size: manifest_bytes), calls


def test_build_plan_from_official_listings():
    get, get_bytes, calls = fake_sources()
    plan = catalog.build_plan(get=get, get_bytes=get_bytes)
    groups = {}
    for item in plan["items"]:
        groups.setdefault(item["group"], []).append(item)
    assert {g: len(v) for g, v in groups.items()} == {"pw_meta": 2, "pw_cameras": 1, "pw_depth": 2, "pw_flows": 3,
                                                     "droid3d": 3, "hand_eye": 2, "rlds": 2050}
    shards = [i for i in groups["rlds"] if ".tfrecord-" in i["id"]]
    assert len(shards) == 2048 and all("generation=7" in i["url"] and i["checksum"] == {"md5": "01" * 16}
                                       for i in shards)
    assert all(catalog.POINTWORLD_REVISION in i["url"] for i in groups["pw_depth"] + groups["pw_flows"])
    assert [i["archive"]["expanded"] for i in groups["pw_flows"]] == [300, 301, 302]
    assert groups["pw_depth"][1]["archive"] == {"kind": "stream", "kinds": ["depth_320x180"], "index": 1,
                                                "expanded": round(1000 * catalog.DEPTH_RATIO * catalog.DEPTH_SAFETY),
                                                "exact": False}
    assert {i["checksum"].popitem()[0] for i in groups["droid3d"]} == {"lfs_pointer_sha1", "git_sha1"}
    assert all(i["auth"] == "hf" for i in groups["droid3d"])
    assert plan["flow_members"]["shard-000001"] == ["LAB+u1+2023-01-01-10h-00m-00s", "LAB+u1+2023-01-02-10h-00m-00s"]
    assert sum("cursor=x" in c for c in calls) == 1 and sum("pageToken" in c for c in calls) == 2


def test_listing_disagreeing_with_the_shard_manifest_is_refused():
    get, get_bytes, _ = fake_sources()
    entries = catalog.hf_listing(catalog.POINTWORLD_REPO, catalog.POINTWORLD_REVISION, get=get)
    manifest = json.loads(get_bytes(None, None, None))
    manifest["shards"][0]["parts"][0]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="disagree"):
        catalog.pointworld_items(entries, manifest)


# ---- storage -----------------------------------------------------------------------------------------------------
def storage_items():
    return [
        {"id": "r", "group": "rlds", "size": 100},
        {"id": "d0", "group": "pw_depth", "size": 50, "archive": {"kind": "stream", "expanded": 60, "exact": False}},
        {"id": "d1", "group": "pw_depth", "size": 50, "archive": {"kind": "stream", "expanded": 60, "exact": False}},
        {"id": "f0", "group": "pw_flows", "size": 40, "archive": {"kind": "single", "shard": "s0", "expanded": 80,
                                                                 "members": 4, "exact": True}},
        {"id": "f1", "group": "pw_flows", "size": 30, "archive": {"kind": "single", "shard": "s1", "expanded": 60,
                                                                 "members": 2, "exact": True}},
    ]


def test_projection_final_and_peak():
    proj = storage.projection(storage_items(), ("rlds", "pw_depth", "pw_flows"))
    assert proj["rlds"] == {"files": 1, "download": 100, "final": 100, "peak": 100, "estimate": False}
    assert proj["pw_depth"]["final"] == 120 and proj["pw_depth"]["peak"] == 220  # all parts held until the end
    assert proj["pw_flows"]["final"] == 140 and proj["pw_flows"]["peak"] == 180  # one shard archive at a time
    selection = {"per_shard": {"s0": 1, "s1": 2}, "fraction": 0.5}
    assert storage.final_bytes(storage_items()[3], selection) == 20
    assert storage.final_bytes(storage_items()[1], selection) == 30
    fit = storage.plan_fit(storage_items(), ("rlds", "pw_depth", "pw_flows"), {"size": 1000, "avail": 500}, 0.15,
                           {"r": {"status": "done"}})
    assert fit["margin"] == 150 and fit["usable"] == 350 and fit["final_growth"] == 260
    assert fit["final_fits"] and fit["peak_growth"] == 360 and not fit["peak_fits"]
    tight = storage.plan_fit(storage_items(), ("rlds", "pw_depth", "pw_flows"), {"size": 1000, "avail": 400}, 0.15)
    assert tight["deficit"] == 360 - 250
    assert storage.admissible(1000, 300, 550, 150) and not storage.admissible(1000, 300, 551, 150)


# ---- state -------------------------------------------------------------------------------------------------------
def test_state_store_and_single_owner_lock(tmp_path):
    store = state.StateStore(tmp_path / "state.json", ["a", "b"], min_interval=3600)
    store.update("a", status="done")
    store.save(force=True)
    store.update("b", status="failed")
    store.save()  # throttled
    again = state.StateStore(tmp_path / "state.json", ["a", "b", "c"])
    assert again.get("a")["status"] == "done" and again.get("b")["status"] == "pending"
    assert again.get("c") == {"status": "pending", "attempts": 0}
    lock = state.OwnerLock(tmp_path)
    assert lock.acquire(tools_sha="x")
    assert not state.OwnerLock(tmp_path).acquire()
    assert state.OwnerLock(tmp_path).held_by_other()
    live = state.OwnerLock(tmp_path).live_owner()
    assert live["pid"] == os.getpid() and live["tools_sha"] == "x"
    lock.release()
    assert state.OwnerLock(tmp_path).live_owner() is None and not lock.held_by_other()


# ---- health ------------------------------------------------------------------------------------------------------
def sample(rate, **extra):
    return {"throughput_bps": rate, "wan_rtt_ms": 40.0, "gw_rtt_ms": 0.3, "errors": 0, "mem_avail_frac": 0.9,
            "psi_io": 0.0, **extra}


def test_controller_scales_up_keeps_gains_reverts_plateaus():
    cfg = {**health.DEFAULTS, "pool_initial": 2}
    control = health.Control(pool=2)
    base = {"wan_rtt_ms": 40.0}
    actions = [health.decide(control, sample(20e6), base, cfg, 0, 100)[0] for _ in range(3)]
    assert actions == ["hold", "hold", "scale_up"] and control.pool == 3
    actions = [health.decide(control, sample(30e6), base, cfg, 0, 100)[0] for _ in range(3)]
    assert actions[-1] == "hold" and control.pool == 3  # +50 %: kept
    actions = [health.decide(control, sample(30e6), base, cfg, 0, 100)[0] for _ in range(3)]
    assert actions[-1] == "scale_up" and control.pool == 4
    actions = [health.decide(control, sample(30.5e6), base, cfg, 10, 100)[0] for _ in range(3)]
    assert actions[-1] == "revert_scale_up" and control.pool == 3 and control.plateau_until == 10 + 900
    actions = [health.decide(control, sample(30e6), base, cfg, 20, 100)[0] for _ in range(3)]
    assert actions[-1] == "hold" and control.pool == 3  # plateau hold


def test_controller_backs_off_then_caps_then_relaxes():
    cfg = dict(health.DEFAULTS)
    control = health.Control(pool=2)
    base = {"wan_rtt_ms": 40.0}
    slow = sample(50e6, wan_rtt_ms=200.0)
    assert health.degraded_reasons(slow, base, cfg) == ["wan_rtt 200 ms (baseline 40)"]
    assert [health.decide(control, slow, base, cfg, 0, 100)[0] for _ in range(2)] == ["hold", "scale_down"]
    assert control.pool == 1
    assert [health.decide(control, slow, base, cfg, 0, 100)[0] for _ in range(2)] == ["hold", "cap"]
    assert control.cap_bps == pytest.approx(35e6)
    assert [health.decide(control, slow, base, cfg, 0, 100)[0] for _ in range(2)] == ["hold", "cap_tighten"]
    assert control.cap_bps == pytest.approx(0.7 * 35e6)
    actions = [health.decide(control, sample(24e6), base, cfg, 0, 100)[0] for _ in range(3)]
    assert actions[-1] == "cap_relax" and control.cap_bps == pytest.approx(0.7 * 35e6 * 1.3)
    for _ in range(9):
        health.decide(control, sample(10e6), base, cfg, 0, 100)
    assert control.cap_bps is None
    assert health.degraded_reasons(sample(1, errors=3, psi_io=30.0, gw_rtt_ms=20.0), base, cfg) == [
        "gateway_rtt 20.0 ms", "psi_io 30.0", "errors 3"]


def test_host_signal_parsers(tmp_path):
    (tmp_path / "io").write_text("some avg10=12.50 avg60=1.00 avg300=0.00 total=1\nfull avg10=3.00\n")
    assert health.read_psi("io", tmp_path) == 12.5 and health.read_psi("cpu", tmp_path) is None
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text("1 0 259:2 / / rw - ext4 /dev/nvme0n1p3 rw\n2 1 259:5 / /data_8T rw - ext4 /dev/nvme1n1p1 rw\n")
    assert health.mount_device("/data_8T/kaist/sunho", mountinfo) == "nvme1n1p1"
    assert health.mount_device("/home/x", mountinfo) == "nvme0n1p3"
    route = tmp_path / "route"
    route.write_text("Iface\tDestination\tGateway\n ens6\t00000000\t010AA8C0\t0003\n")
    assert health.default_gateway(route) == "192.168.10.1"
    out = "rtt min/avg/max/mdev = 0.201/0.330/0.765/0.217 ms\n"
    assert health.ping_ms("h", run=lambda *a, **k: subprocess.CompletedProcess(a, 0, out, "")) == 0.33


# ---- archives ----------------------------------------------------------------------------------------------------
def tar_zst(members: dict[str, bytes]) -> bytes:
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return subprocess.run(["zstd", "-q", "-c"], input=raw.getvalue(), capture_output=True, check=True).stdout


EPISODES = ["AUTOLab+0d4edc83+2023-10-21-19h-07m-04s", "ILIAD+t3d58310+2023-05-01-10h-00m-00s"]


def test_flow_shard_extraction_verification_and_publication(tmp_path):
    members = {f"droid/flows-fs-optimized/{e}_flows.h5": HDF5 + os.urandom(1000 + k) for k, e in enumerate(EPISODES)}
    archive = tmp_path / "shard.tar.zst"
    archive.write_bytes(tar_zst(members))
    staging = tmp_path / "staging"
    report = archives.extract_stream(archives.file_chunks([archive]), staging, ["flows-fs-optimized"])
    assert report["written"] == 2
    expected = sum(len(v) for v in members.values())
    assert archives.verify_flow_shard(staging, EPISODES, expected) == {"members": 2, "bytes": expected}
    with pytest.raises(archives.ExtractionError, match="raw_member_bytes"):
        archives.verify_flow_shard(staging, EPISODES, expected + 1)
    with pytest.raises(archives.ExtractionError, match="differ"):
        archives.verify_flow_shard(staging, EPISODES[:1], None)
    final = tmp_path / "final"
    assert archives.publish(staging, "flows-fs-optimized", final) == 2
    assert archives.publish(staging, "flows-fs-optimized", final) == 0  # idempotent
    assert sorted(p.name for p in (final / "flows-fs-optimized").iterdir()) == sorted(Path(m).name for m in members)


def test_depth_stream_across_parts_and_a_stop(tmp_path):
    members = {f"droid/depth_320x180/{e}_depth.h5": HDF5 + os.urandom(200_000) for e in EPISODES}
    blob = tar_zst(members)
    parts = []
    for k, start in enumerate(range(0, len(blob), len(blob) // 3 + 1)):
        parts.append(tmp_path / f"part-{k}")
        parts[-1].write_bytes(blob[start:start + len(blob) // 3 + 1])
    staging = tmp_path / "staging"
    with pytest.raises(archives.ExtractionStopped):
        archives.extract_stream(archives.file_chunks(parts), staging, ["depth_320x180"], should_stop=lambda: True)
    archives.extract_stream(archives.file_chunks(parts), staging, ["depth_320x180"])
    report = archives.verify_depth(staging, set(EPISODES) | {"LAB+x+2023-01-01-00h-00m-00s"})
    assert report["members"] == 2 and report["flow_episodes_without_depth"] == 1
    with pytest.raises(archives.ExtractionError, match="truncated|zstd"):
        archives.extract_stream(archives.file_chunks(parts[:2]), tmp_path / "s2", ["depth_320x180"])


def test_cameras_verification(tmp_path):
    members = {f"droid/cameras/{e}_cameras.json": json.dumps({"uuid": e}).encode() for e in EPISODES}
    archive = tmp_path / "cams.tar.zst"
    archive.write_bytes(tar_zst(members))
    archives.extract_stream(archives.file_chunks([archive]), tmp_path / "st", ["cameras"])
    size = sum(len(v) for v in members.values())
    assert archives.verify_cameras(tmp_path / "st", 2, size)["members"] == 2
    with pytest.raises(archives.ExtractionError):
        archives.verify_cameras(tmp_path / "st", 3, size)


# ---- the owner process end to end --------------------------------------------------------------------------------
class FakeProbe:
    def sample(self, total, errors):
        return {"wan_rtt_ms": 40.0, "gw_rtt_ms": 0.3, "psi_io": 0.0, "psi_cpu": 0.0, "psi_mem": 0.0,
                "sshd_rtt_ms": 20.0, "disk_util": 0.1, "mem_avail_frac": 0.9, "errors": errors,
                "throughput_bps": 1e6}


def build_world(srv: Server, root: Path) -> dict:
    flows = {f"droid/flows-fs-optimized/{e}_flows.h5": HDF5 + os.urandom(3000) for e in EPISODES}
    depth_blob = tar_zst({f"droid/depth_320x180/{e}_depth.h5": HDF5 + os.urandom(50_000) for e in EPISODES})
    cams = {f"droid/cameras/{e}_cameras.json": json.dumps({"uuid": e}).encode() for e in EPISODES}
    half = len(depth_blob) // 2
    items = [
        item_for(srv, "rlds/shard0", os.urandom(2 << 20), group="rlds", dest="droid/1.0.1/shard0"),
        item_for(srv, "rlds/info.json", b"{}", group="rlds", dest="droid/1.0.1/dataset_info.json"),
        item_for(srv, "pw/cams", tar_zst(cams), group="pw_cameras", dest=".downloads/pw/cams",
                 archive={"kind": "single", "kinds": ["cameras"], "expanded": sum(map(len, cams.values())),
                          "exact": True, "members": 2}),
        item_for(srv, "pw/d0", depth_blob[:half], group="pw_depth", dest=".downloads/pw/d0",
                 archive={"kind": "stream", "kinds": ["depth_320x180"], "index": 0, "expanded": half * 2,
                          "exact": False}),
        item_for(srv, "pw/d1", depth_blob[half:], group="pw_depth", dest=".downloads/pw/d1",
                 archive={"kind": "stream", "kinds": ["depth_320x180"], "index": 1, "expanded": half * 2,
                          "exact": False}),
        item_for(srv, "pw/f0", tar_zst(flows), group="pw_flows", dest=".downloads/pw/f0",
                 archive={"kind": "single", "kinds": ["flows-fs-optimized"], "shard": "shard-000000",
                          "expanded": sum(map(len, flows.values())), "exact": True, "members": 2}),
    ]
    srv.files["d3d"] = b"calibration"
    items.append({"id": "droid3d/x", "group": "droid3d", "url": srv.url("gated/d3d"), "size": 11, "auth": "hf",
                  "checksum": md5(b"calibration"), "dest": "droid3d/x"})
    srv.token = "secret-token"
    plan = {"items": items, "flow_members": {"shard-000000": sorted(EPISODES)}}
    (root / ".state").mkdir(parents=True)
    state.write_json_atomic(root / ".state" / "plan.json", plan)
    return plan


def run_owner(root: Path, **kwargs) -> tuple[runner.Runner, threading.Thread]:
    owner = runner.Runner(root, tools_sha="test", probe=FakeProbe())
    thread = threading.Thread(target=owner.run, daemon=True)
    thread.start()
    return owner, thread


def wait_for(predicate, timeout=60.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.1)
    return False


def test_owner_downloads_gates_extracts_and_publishes(server, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "HF_HOST", "127.0.0.1")
    monkeypatch.setattr(transfer, "AUTH_SCHEMES", ("https", "http"))
    root = tmp_path / "datasets"
    build_world(server, root)
    config = {**runner.DEFAULT_CONFIG, "margin_fraction": 0.0, "controller": {"interval_s": 0.2}}
    state.write_json_atomic(root / ".state" / "config.json", config)
    owner, thread = run_owner(root)
    status = lambda item: owner.state.items[item]["status"]  # noqa: E731
    assert wait_for(lambda: status("t/rlds/shard0") == "done" and status("t/pw/cams") == "extracted")
    assert wait_for(lambda: status("droid3d/x") == "blocked")
    assert status("t/pw/f0") == "pending" and owner.gates["pw_flows"].startswith("pointworld_selection pending")
    assert wait_for(lambda: status("t/pw/d1") == "downloaded")
    time.sleep(0.5)
    assert owner.state.section("stream") == {}  # depth extraction waits for the selection decision
    secret = root / ".state" / "secrets"
    secret.mkdir()
    (secret / "hf_token").write_text("secret-token\n")
    (secret / "hf_token").chmod(0o600)
    state.write_json_atomic(root / ".state" / "config.json",
                            {**config, "pointworld_selection": "all", "groups": {"pw_flows": True}})
    thread.join(timeout=60)
    assert not thread.is_alive()
    statuses = {k: v["status"] for k, v in owner.state.items.items()}
    assert statuses == {"t/rlds/shard0": "done", "t/rlds/info.json": "done", "t/pw/cams": "extracted",
                        "t/pw/d0": "extracted", "t/pw/d1": "extracted", "t/pw/f0": "extracted", "droid3d/x": "done"}
    pw = root / "pointworld-droid" / "droid"
    assert len(list((pw / "flows-fs-optimized").iterdir())) == len(list((pw / "depth_320x180").iterdir())) == 2
    assert len(list((pw / "cameras").iterdir())) == 2
    assert not any((root / ".downloads" / "pw").iterdir()) and not (root / ".staging" / "pw_depth").exists()
    assert (root / "droid3d" / "x").read_bytes() == b"calibration"
    events = [json.loads(line)["event"] for line in (root / ".state" / "progress.jsonl").read_text().splitlines()]
    assert {"owner_start", "verified", "blocked", "extracted", "stream_published", "all_finished"} <= set(events)
    assert "secret-token" not in (root / ".state" / "progress.jsonl").read_text()
    assert json.loads((root / ".state" / "status.json").read_text())["stopped"] is True
    assert not state.OwnerLock(root / ".state").held_by_other()


def test_space_admission_and_reconciliation(server, tmp_path, monkeypatch):
    root = tmp_path / "datasets"
    plan = build_world(server, root)
    state.write_json_atomic(root / ".state" / "config.json", {**runner.DEFAULT_CONFIG, "margin_fraction": 0.0})
    owner = runner.Runner(root, probe=FakeProbe())
    owner.reload_config()
    monkeypatch.setattr(storage, "fs_usage", lambda path: {"size": 10**9, "avail": 100})
    assert owner.next_item() is None
    assert json.loads((root / ".state" / "progress.jsonl").read_text().splitlines()[-1])["event"] == "space_wait"
    monkeypatch.setattr(storage, "fs_usage", lambda path: {"size": 10**9, "avail": 10**9})
    first = owner.next_item()  # small groups first
    assert first["id"] == "t/pw/cams" and owner.state.items["t/pw/cams"]["status"] == "downloading"
    # a crash leaves "downloading" and an archive whose content was already published
    leftover = root / plan["items"][5]["dest"]
    leftover.parent.mkdir(parents=True)
    leftover.write_bytes(b"x")
    owner.state.update("t/pw/f0", status="extracted")
    owner.state.update("t/rlds/shard0", status="done")
    owner.state.save(force=True)
    again = runner.Runner(root, probe=FakeProbe())
    report = again.reconcile()
    assert report["downloading_to_pending"] == 1 and report["leftover_archive_deleted"] == 1
    assert report["missing_to_pending"] == 1 and not leftover.exists()


# ---- command line and deployment ---------------------------------------------------------------------------------
def test_cli_configure_refuses_to_cross_the_margin_and_adopt_verifies(server, tmp_path, monkeypatch, capsys):
    root = tmp_path / "datasets"
    build_world(server, root)
    usage = {"size": 10**7, "avail": 3_000_000}
    monkeypatch.setattr(storage, "fs_usage", lambda path: usage)
    args = ["--root", str(root)]
    assert cli.main(["configure", *args, "--enable", "pw_flows", "--pointworld-selection", "all",
                     "--margin-fraction", "0.15"]) == 2
    assert "refused" in capsys.readouterr().out
    assert cli.main(["configure", *args, "--enable", "pw_flows", "--pointworld-selection", "all",
                     "--margin-fraction", "0.0"]) == 0
    capsys.readouterr()
    source = tmp_path / "copy"
    source.write_bytes(b"calibration")
    assert cli.main(["adopt", *args, "--item", "droid3d/x", "--source", str(source)]) == 0
    assert (root / "droid3d" / "x").read_bytes() == source.read_bytes() == b"calibration"
    source.write_bytes(b"calibratioN")
    assert cli.main(["adopt", *args, "--item", "droid3d/x", "--source", str(source)]) == 2
    lock = state.OwnerLock(root / ".state")
    assert lock.acquire()
    try:
        capsys.readouterr()
        assert cli.main(["start", *args]) == 0  # this process is the live owner: no second start
        out = json.loads(capsys.readouterr().out)
        assert out["started"] is False and out["running"] is True and out["owner"]["pid"] == os.getpid()
    finally:
        lock.release()


def test_remote_side_is_standard_library_only():
    """The deployed modules run on the remote host's python3 (no numpy/h5py/torch outside the containers)."""
    import ast

    allowed_first_party = {"ops.datasets", "ops.datasets.archives", "ops.datasets.catalog", "ops.datasets.health",
                           "ops.datasets.runner", "ops.datasets.state", "ops.datasets.storage",
                           "ops.datasets.transfer", "ops.datasets.cli", "s4d.data.droid.largescale.extract"}
    files = [p for p in (REPO / "ops/datasets").glob("*.py") if p.name != "control.py"]
    files += [REPO / "s4d/data/droid/largescale/extract.py", REPO / "s4d/data/__init__.py"]
    for path in files:
        for node in ast.walk(ast.parse(path.read_text())):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else (
                [node.module] if isinstance(node, ast.ImportFrom) and node.level == 0 else [])
            for name in names:
                top = name.split(".")[0]
                assert top in sys.stdlib_module_names or name in allowed_first_party or (
                    path.name == "__init__.py" and name.startswith("ops.resources")), (path, name)
    assert set(control.DEPLOY_PATHS) >= {"ops/datasets", "s4d/data/droid/largescale/extract.py"}


def test_remote_entry_point_runs_isolated():
    result = subprocess.run([sys.executable, "-I", str(REPO / "scripts/datasets/droid_datasets.py"), "--help"],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0 and result.stdout.startswith("usage: droid_datasets.py")


def test_deploy_command_unpacks_verifies_and_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(control, "TOOLS_ROOT", str(tmp_path / "tools"))
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w:gz") as tar:
        info = tarfile.TarInfo("ops/datasets/x.py")
        info.size = 3
        tar.addfile(info, io.BytesIO(b"abc"))
    manifest = f"{hashlib.sha256(b'abc').hexdigest()}  ops/datasets/x.py\n"
    sha = "a" * 40
    command = control.deploy_command(sha, raw.getvalue(), manifest)
    assert subprocess.run(["bash", "-c", command], capture_output=True, text=True).stdout.strip() == "deployed-new"
    dest = tmp_path / "tools" / sha
    assert (dest / "ops/datasets/x.py").read_bytes() == b"abc" and (dest / "DEPLOYED_SHA").read_text() == sha
    assert not os.access(dest / "ops/datasets/x.py", os.W_OK) or os.geteuid() == 0
    assert subprocess.run(["bash", "-c", command], capture_output=True, text=True).stdout.strip() == "deployed-existing"
    bad = control.deploy_command(sha[:-1] + "b", raw.getvalue(), "f" * 64 + "  ops/datasets/x.py\n")
    assert subprocess.run(["bash", "-c", bad], capture_output=True, text=True).returncode != 0
    assert not (tmp_path / "tools" / (sha[:-1] + "b")).exists()
    assert [p.name for p in (tmp_path / "tools").iterdir()] == [sha]  # the failed staging copy was removed
    with pytest.raises(ValueError):
        control.deploy_command("short", b"", "")
    args = ["sshpass", "-e", "ssh", "-o", "ConnectTimeout=15", "-o", "ControlMaster=auto", "-o", "ControlPersist=600",
            "-o", "ControlPath=/x/cm-%C"]
    assert control.fresh_ssh_args(args) == ["sshpass", "-e", "ssh", "-o", "ConnectTimeout=15", "-o",
                                            "ControlMaster=no", "-o", "ControlPath=none"]
    assert control.remote_command(sha, ["status", "--full"])[3:] == ["status", "--root", control.DATA_ROOT, "--full"]
