"""Tests for the real-world ingestion worker. Raw data is only READ (one real episode is the template); every file the
tests create lives under /home/ws/ws/realworld-ops/ingest/scratch/tests/ (removed afterwards).

  /home/ws/ws/splatter4d/.venv/bin/python -I -m pytest -q -p no:cacheprovider \
      /home/ws/ws/splatter4d/runs/setup/tools/realworld_ingest_20261106/test_ingest.py
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path, PurePosixPath

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import episode_validator as V  # noqa: E402
import ingest_worker as W  # noqa: E402

TEMPLATE = Path("/home/ws/data/real_world/drawer-open/episode_000017.h5")  # read-only template (fully paired)
SCRATCH = Path("/home/ws/ws/realworld-ops/ingest/scratch/tests")
N_FRAMES = 36
OLD = time.time() - 3600


def _copy_reduced(src: Path, dst: Path, n: int = N_FRAMES, drop_stereo_row: int | None = None,
                  episode_index: int | None = None) -> Path:
    """A smaller but complete format_version 2 episode: first ``n`` camera frames of every camera, robot/action/omega/
    gripper rows up to the last kept camera frame, stats_json rows updated; optionally one stereo record dropped."""
    with h5py.File(src, "r", locking=False) as f, h5py.File(dst, "w") as g:
        for k, v in f.attrs.items():
            g.attrs[k] = v
        stats = json.loads(f.attrs["stats_json"])
        t_end = max(float(f["cameras"][c]["t"][n - 1]) for c in f["cameras"]) + 0.05
        for grp in ("robot", "action", "omega"):
            keep = int(np.searchsorted(f["robot/t"][()], t_end))
            for name, ds in f[grp].items():
                g.create_dataset(f"{grp}/{name}", data=ds[:keep], maxshape=(None,) + ds.shape[1:], chunks=True,
                                 dtype=ds.dtype)
        stats["rows"]["robot"] = keep
        gk = int(np.searchsorted(f["gripper/t"][()], t_end))
        for name, ds in f["gripper"].items():
            g.create_dataset(f"gripper/{name}", data=ds[:gk], maxshape=(None,), chunks=True, dtype=ds.dtype)
        stats["rows"]["gripper"] = gk
        for grp in ("events", "segments"):
            for name, ds in f[grp].items():
                g.create_dataset(f"{grp}/{name}", data=ds[()], maxshape=(None,), chunks=True, dtype=ds.dtype)
        for cam in f["cameras"]:
            src_g, dst_g = f["cameras"][cam], g.create_group(f"cameras/{cam}")
            for k, v in src_g.attrs.items():
                dst_g.attrs[k] = v
            for name, ds in src_g.items():
                if isinstance(ds, h5py.Dataset):
                    dst_g.create_dataset(name, data=ds[:n], maxshape=(None,), chunks=True, dtype=ds.dtype)
            rows = [i for i in range(n) if not (drop_stereo_row is not None and cam == sorted(f["cameras"])[0]
                                                 and i == drop_stereo_row)]
            for name, ds in src_g["stereo"].items():
                dst_g.create_dataset(f"stereo/{name}", data=ds[:n][rows], maxshape=(None,), chunks=True,
                                     dtype=ds.dtype)
            stats["rows"][f"cameras/{cam}"] = n
            stats["rows"][f"cameras/{cam}/stereo"] = len(rows)
        g.attrs["stats_json"] = json.dumps(stats)
        if episode_index is not None:
            g.attrs["episode_index"] = episode_index
    os.utime(dst, (OLD, OLD))
    return dst


@pytest.fixture(scope="session")
def base(tmp_path_factory) -> Path:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    d = SCRATCH / f"base-{os.getpid()}"
    d.mkdir(exist_ok=True)
    out = _copy_reduced(TEMPLATE, d / "episode_000017.h5")
    yield out
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture()
def work(request) -> Path:
    d = SCRATCH / (re.sub(r"[^A-Za-z0-9_.-]+", "_", request.node.name)[:80] + f"-{os.getpid()}")
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


def variant(base: Path, work: Path, name: str = "episode_000017.h5", mutate=None, old: bool = True) -> Path:
    dst = work / name
    shutil.copyfile(base, dst)
    if mutate:
        with h5py.File(dst, "r+", locking=False) as f:
            mutate(f)
    if old:
        os.utime(dst, (OLD, OLD))
    return dst


def check(path: Path, grace_s: float = 0.0, **kw):  # ctime cannot be back-dated: grace 0 unless tested
    return V.validate(path, task_names={"drawer-opening"}, grace_s=grace_s, **kw)


CAM = "cameras/cam_135122070988"


# --------------------------------------------------------------------------------------------------- validator


def test_reduced_episode_valid(base):
    r = check(base)
    assert r["status"] == "valid", r["reasons"]
    assert r["warnings"] == [] and r["adapter"]["ok"], (r["warnings"], r["adapter"])
    assert r["sha256"] == V.sha256_file(base)
    assert r["summary"]["camera"]["cam_135122070988"]["colour_frames"] == N_FRAMES


@pytest.mark.parametrize("fraction", [0.4, 0.98])
def test_truncated_copy_is_incomplete(base, work, fraction):
    dst = work / "episode_000017.h5"
    size = base.stat().st_size
    with open(base, "rb") as src, open(dst, "wb") as out:
        out.write(src.read(int(size * fraction)))
    os.utime(dst, (OLD, OLD))
    r = check(dst)
    assert r["status"] == "incomplete" and "HDF5 open failed" in r["reasons"][0], r


def test_recent_file_is_waiting(base, work):
    r = check(variant(base, work, old=False), grace_s=120)
    assert r["status"] == "waiting" and "sha256" not in r
    r = check(variant(base, work, name="episode_000018.h5", old=True), grace_s=120)  # mtime back-dated (scp -p /
    assert r["status"] == "waiting"  # rsync -t), but ctime records the write: still inside the grace period


@pytest.mark.parametrize("mutate, expect", [
    (lambda f: f.attrs.__setitem__("status", "recording"), "completion signal"),
    (lambda f: f.attrs.__delitem__("end_unix"), "completion signal"),
])
def test_missing_completion_signal_is_incomplete(base, work, mutate, expect):
    r = check(variant(base, work, mutate=mutate))
    assert r["status"] == "incomplete" and expect in r["reasons"][0]


def _resize(path, n):
    def m(f):
        f[path].resize((n,))
    return m


def _corrupt(path, i, data=b"not an image"):
    def m(f):
        f[path][i] = np.frombuffer(data, np.uint8)
    return m


def _swap_t(path):
    def m(f):
        t = f[path][()]
        t[5], t[6] = t[6], t[5]
        f[path][...] = t
    return m


def _scale_t(f):
    for cam in f["cameras"]:
        for p in ("t", "stereo/t"):
            t = f["cameras"][cam][p][()]
            f["cameras"][cam][p][...] = t[0] + (t - t[0]) * 2.0


def _stats_rows(f):
    s = json.loads(f.attrs["stats_json"])
    s["rows"]["robot"] += 1
    f.attrs["stats_json"] = json.dumps(s)


@pytest.mark.parametrize("mutate, expect", [
    (lambda f: f.__delitem__(f"{CAM}/stereo/right"), "stereo/right: missing"),
    (lambda f: f.__delitem__("action/flange_target"), "action/flange_target: missing"),
    (lambda f: f.__delitem__("events"), "events: group missing"),
    (_resize(f"{CAM}/stereo/depth", N_FRAMES - 1), "different row counts"),
    (_resize(f"{CAM}/color", N_FRAMES - 1), "different row counts"),
    (_corrupt(f"{CAM}/color", N_FRAMES // 2), "jpeg start/end markers"),
    (_corrupt(f"{CAM}/stereo/depth", N_FRAMES - 1, b"\x89PNG\r\n\x1a\n" + b"\0" * 64), "png16 start/end markers"),
    (_swap_t(f"{CAM}/t"), "not strictly increasing"),
    (_swap_t("robot/t"), "robot/t"),
    (_scale_t, "median within-segment interval"),
    (lambda f: f.attrs.__setitem__("format_version", 1), "is not panda_teleop_episode v2"),
    (_stats_rows, "!= stats_json rows"),
    (lambda f: f.__delitem__("cameras/cam_135122076054"), "2 cameras"),
])
def test_integrity_failures(base, work, mutate, expect):
    r = check(variant(base, work, mutate=mutate))
    assert r["status"] == "failed", r
    assert any(expect in x for x in r["reasons"]), r["reasons"]


def test_valid_jpeg_markers_but_undecodable_middle_frame(base, work):
    bad = b"\xff\xd8" + b"\x00" * 200 + b"\xff\xd9"
    r = check(variant(base, work, mutate=_corrupt(f"{CAM}/color", N_FRAMES // 2, bad)))
    assert r["status"] == "failed" and any("decode failed" in x for x in r["reasons"]), r["reasons"]


def test_task_name_and_index_mismatch_are_warnings(base, work):
    def m(f):
        f.attrs["task_name"] = "pick-and-place"
        f.attrs["episode_index"] = 99
    r = check(variant(base, work, mutate=m))
    assert r["status"] == "valid"
    assert {"task_name_mismatch", "index_name_mismatch"} <= set(r["warning_codes"])


def test_one_dropped_stereo_record_is_valid_with_warning(work):
    p = _copy_reduced(TEMPLATE, work / "episode_000017.h5", drop_stereo_row=20)
    r = check(p)
    assert r["status"] == "valid", r["reasons"]
    assert {"colour_stereo_unpaired", "dropped_frames", "adapter_refused"} <= set(r["warning_codes"])
    assert r["summary"]["camera"]["cam_135122070988"]["unpaired_colour_frames"] == [20]


def test_validator_never_writes(base, work):
    p = variant(base, work)
    os.chmod(p, 0o444)
    before = V.stat_fingerprint(p), V.sha256_file(p)
    assert check(p)["status"] == "valid"
    assert (V.stat_fingerprint(p), V.sha256_file(p)) == before


# ------------------------------------------------------------------------------------------------------ worker


class FakeClient:
    """Stands in for RemoteClient: local 'remote' root, same path policy, no credentials."""

    def __init__(self, root: Path):
        self.root = str(root)
        self.target = "local"
        self.credentials = types.SimpleNamespace(password="\x00never")

    def remote_path(self, path, allow_root=False):
        posix = PurePosixPath(self.root) / str(path)
        assert ".." not in posix.parts and PurePosixPath(self.root) in posix.parents
        return str(posix)

    def ensure_remote_path(self, rel, create_parent=False, directory=False, wait=True):
        Path(self.remote_path(rel)).mkdir(parents=True, exist_ok=True)

    def check_tunnel(self, wait=True):
        if (Path(self.root) / "UNREACHABLE").exists():
            raise W_access.RemoteUnreachable("down")
        return True

    def _environment(self):
        return dict(os.environ)

    def _redact(self, text):
        return text


W_access = types.SimpleNamespace(RemoteUnreachable=type("RemoteUnreachable", (RuntimeError,), {}),
                                 RemoteCommandError=type("RemoteCommandError", (RuntimeError,), {}))


class LocalShellRemote(W.RemoteOps):
    """The real RemoteOps shell scripts (probe/sha256/publish/remove), executed locally with bash."""

    def __init__(self, root: Path):
        self.client = FakeClient(root)
        self._access = W_access
        self.target = "local"
        self.rsync_calls = []

    def _ssh(self, command, idempotent=True, timeout=300):
        try:
            self.client.check_tunnel()
        except W_access.RemoteUnreachable as exc:
            raise W.Unreachable(str(exc)) from None
        r = subprocess.run(["bash", "-c", command], capture_output=True, text=True, timeout=timeout)
        if r.returncode:
            raise W.RemoteError(f"exit {r.returncode}: {r.stderr}")
        return r.stdout

    def rsync_command(self, local, remote, kib, append=True):
        cmd, _ = super().rsync_command.__func__(_RsyncShim(self), local, remote, kib, append)
        i = cmd.index("-e")
        cmd = cmd[:i] + cmd[i + 2:]
        cmd[-1] = remote  # local destination
        self.rsync_calls.append(cmd)
        return cmd, dict(os.environ)


class _RsyncShim:
    def __init__(self, outer):
        self.client = types.SimpleNamespace(ssh_args=lambda: ["sshpass", "-e", "ssh", "-o", "ControlMaster=auto",
                                                              "-o", "ControlPath=/x", "-o", "ControlPersist=600"],
                                            _environment=lambda: {}, credentials=outer.client.credentials)
        self.target = "local"


def make_raw(work: Path, base: Path, names: dict) -> Path:
    raw = work / "raw" / "drawer-open"
    raw.mkdir(parents=True)
    for name, src in names.items():
        shutil.copyfile(src or base, raw / name)
        os.utime(raw / name, (OLD, OLD))
    return raw.parent


FAST = {"kib_per_s": 500000, "coordinator_override": True}  # tests only: local copies at full speed


def worker(work: Path, raw: Path, remote=None, **kw) -> W.Worker:
    state = work / "state"
    state.mkdir(exist_ok=True)
    if not (state / "upload-budget.json").exists():
        (state / "upload-budget.json").write_text(json.dumps(FAST))
    kw.setdefault("grace_s", 0.0)
    return W.Worker(state, raw_root=raw, tasks={"drawer-open": {"drawer-opening"}},
                    remote=remote or LocalShellRemote(work / "remote"), mirror=True, **kw)


def remote_file(work: Path, name: str) -> Path:
    return work / "remote" / "data/real_world/drawer-open" / name


def test_pass_transfers_verifies_and_is_idempotent(base, work):
    raw = make_raw(work, base, {"episode_000017.h5": None})
    w = worker(work, raw)
    res = w.run_pass()
    e = w.state.episodes["drawer-open/episode_000017.h5"]
    assert res["transferred"] == 1 and e["transfer"]["status"] == "verified"
    final = remote_file(work, "episode_000017.h5")
    assert V.sha256_file(final) == e["sha256"] == e["transfer"]["remote_sha256"]
    assert oct(final.stat().st_mode & 0o777) == "0o444"
    assert not list((final.parent / ".incoming").iterdir())
    assert "--bwlimit=500000" in w.remote.rsync_calls[0]
    c = json.loads((work / "state/manifest.json").read_text())["counts"]["drawer-open"]
    assert (c["received"], c["valid"], c["transferred"], c["target"]) == (1, 1, 1, 100)
    mirror = json.loads((work / "remote/data/real_world/ingest-manifest.json").read_text())
    assert "drawer-open/episode_000017.h5" in mirror["episodes"]
    n = len(w.remote.rsync_calls)
    w2 = worker(work, raw)  # restart from the manifest
    assert n == 2  # episode + manifest mirror
    assert w2.run_pass()["transferred"] == 0
    assert not any("episode_000017" in c[-2] for c in w2.remote.rsync_calls)
    events = [json.loads(x)["event"] for x in (work / "state/events.jsonl").read_text().splitlines()]
    assert events.count("transfer_verified") == 1


def test_partial_part_is_resumed(base, work):
    raw = make_raw(work, base, {"episode_000017.h5": None})
    part = remote_file(work, ".incoming/episode_000017.h5.part")
    part.parent.mkdir(parents=True)
    data = base.read_bytes()
    part.write_bytes(data[: len(data) // 2])
    w = worker(work, raw)
    w.run_pass()
    e = w.state.episodes["drawer-open/episode_000017.h5"]
    assert e["transfer"]["status"] == "verified" and e["transfer"]["resumed_from_bytes"] == len(data) // 2
    assert remote_file(work, "episode_000017.h5").read_bytes() == data


def test_interrupted_rsync_then_resume(base, work):
    raw = make_raw(work, base, {"episode_000017.h5": None})
    (work / "state").mkdir()
    (work / "state/upload-budget.json").write_text(json.dumps({"kib_per_s": 2000, "coordinator_override": True}))
    w = worker(work, raw)
    orig = w._run_rsync

    def interrupted(local, part, kib, entry):  # kill the transfer after ~1 s: --partial keeps the bytes
        cmd, env = w.remote.rsync_command(local, part, kib)
        p = subprocess.Popen(cmd, env=env)
        time.sleep(1.5)
        p.terminate()
        p.wait()
        return None, ""
    w._run_rsync = interrupted
    w.run_pass()
    e = w.state.episodes["drawer-open/episode_000017.h5"]
    part = remote_file(work, ".incoming/episode_000017.h5.part")
    assert e["transfer"]["status"] == "partial" and 0 < part.stat().st_size < base.stat().st_size
    (work / "state/upload-budget.json").write_text(json.dumps(FAST))
    w._run_rsync = orig
    w.run_pass()
    assert e["transfer"]["status"] == "verified" and e["transfer"]["resumed_from_bytes"] > 0
    assert V.sha256_file(remote_file(work, "episode_000017.h5")) == e["sha256"]


def test_corrupt_part_is_detected_and_retransferred(base, work):
    raw = make_raw(work, base, {"episode_000017.h5": None})
    part = remote_file(work, ".incoming/episode_000017.h5.part")
    part.parent.mkdir(parents=True)
    part.write_bytes(b"\0" * base.stat().st_size)
    os.utime(part, (OLD, OLD))
    w = worker(work, raw)
    w.run_pass()
    e = w.state.episodes["drawer-open/episode_000017.h5"]
    assert e["transfer"]["sha_mismatches"] == 1 and e["transfer"]["status"] == "partial" and not part.exists()
    e["transfer"]["next_attempt_unix"] = 0
    w.run_pass()
    assert e["transfer"]["status"] == "verified"


def test_existing_remote_copy_never_overwritten(base, work):
    raw = make_raw(work, base, {"episode_000017.h5": None})
    final = remote_file(work, "episode_000017.h5")
    final.parent.mkdir(parents=True)
    final.write_bytes(b"other content")
    w = worker(work, raw)
    w.run_pass()
    e = w.state.episodes["drawer-open/episode_000017.h5"]
    assert "remote_conflict" in e["flags"] and final.read_bytes() == b"other content"
    assert w.state.doc["counts"]["drawer-open"]["valid"] == 0  # blocked until resolved


def test_existing_identical_remote_copy_is_adopted(base, work):
    raw = make_raw(work, base, {"episode_000017.h5": None})
    final = remote_file(work, "episode_000017.h5")
    final.parent.mkdir(parents=True)
    shutil.copyfile(base, final)
    w = worker(work, raw)
    w.run_pass()
    e = w.state.episodes["drawer-open/episode_000017.h5"]
    assert e["transfer"]["status"] == "verified" and not any("episode_000017" in c[-2] for c in w.remote.rsync_calls)


def test_duplicates_and_index_conflicts(base, work):
    other = _copy_reduced(TEMPLATE, work / "other.h5", n=N_FRAMES + 2)  # different content, same episode_index 17
    raw = make_raw(work, base, {"episode_000017.h5": None, "episode_000018.h5": None,
                                "episode_000019.h5": other})
    w = worker(work, raw, transfer=False)
    w.run_pass()
    eps = w.state.episodes
    assert "duplicate_content" in eps["drawer-open/episode_000018.h5"]["flags"]
    assert "index_conflict" in eps["drawer-open/episode_000017.h5"]["flags"]
    assert "index_conflict" in eps["drawer-open/episode_000019.h5"]["flags"]
    c = w.state.doc["counts"]["drawer-open"]
    assert (c["received"], c["valid"]) == (3, 0) and len(c["blocked"]) == 3
    (work / "state/overrides.json").write_text(json.dumps(
        {"drawer-open/episode_000017.h5": {"decision": "accept", "reason": "test"}}))
    w.run_pass()
    assert w.state.doc["counts"]["drawer-open"]["valid"] == 1


def test_budget_zero_pauses_transfers(base, work):
    raw = make_raw(work, base, {"episode_000017.h5": None})
    (work / "state").mkdir()
    (work / "state/upload-budget.json").write_text(json.dumps({"kib_per_s": 0}))
    w = worker(work, raw)
    res = w.run_pass()
    assert res["remote"].startswith("paused") and not w.remote.rsync_calls
    (work / "state/upload-budget.json").write_text(json.dumps({"kib_per_s": 5000}))
    assert w.upload_kib() == 600  # capped without coordinator_override
    (work / "state/upload-budget.json").unlink()
    assert w.upload_kib() == 600  # authorized default


def test_unreachable_remote_is_survived(base, work):
    raw = make_raw(work, base, {"episode_000017.h5": None})
    (work / "remote").mkdir()
    (work / "remote/UNREACHABLE").write_text("")
    w = worker(work, raw)
    assert w.run_pass()["remote"] == "unreachable"
    assert w.state.doc["counts"]["drawer-open"]["valid"] == 1
    (work / "remote/UNREACHABLE").unlink()
    assert w.run_pass()["transferred"] == 1


def test_source_missing_and_failed_then_invalid(base, work):
    bad = variant(base, work, name="bad.h5", mutate=lambda f: f.__delitem__(f"{CAM}/stereo/left"))
    raw = make_raw(work, base, {"episode_000017.h5": None, "episode_000020.h5": bad})
    w = worker(work, raw, transfer=False, invalid_after_s=0.0)
    w.run_pass()
    eps = w.state.episodes
    assert eps["drawer-open/episode_000020.h5"]["validation"]["status"] == "invalid"
    os.remove(raw / "drawer-open/episode_000017.h5")
    w.run_pass()
    assert "source_missing" in eps["drawer-open/episode_000017.h5"]["flags"]
    c = w.state.doc["counts"]["drawer-open"]
    assert (c["received"], c["valid"], c["invalid"]) == (1, 0, 1)


def test_retrying_before_invalid_window(base, work):
    bad = variant(base, work, name="bad.h5", mutate=lambda f: f.__delitem__(f"{CAM}/stereo/left"))
    raw = make_raw(work, base, {"episode_000020.h5": bad})
    w = worker(work, raw, transfer=False)
    w.run_pass()
    assert w.state.episodes["drawer-open/episode_000020.h5"]["validation"]["status"] == "retrying"


def test_preprocessing_fields_survive_worker_saves(base, work):
    raw = make_raw(work, base, {"episode_000017.h5": None})
    w = worker(work, raw, transfer=False)
    w.run_pass()
    key = "drawer-open/episode_000017.h5"
    sha = w.state.episodes[key]["sha256"]
    W.set_stage(work / "state", key, "calibration", {"status": "done", "code_version": "calib-x1",
                                                     "output_format": "rw-calib/1", "input_sha256": sha,
                                                     "started_at": "t0", "finished_at": "t1"})
    w.run_pass()
    doc = json.loads((work / "state/manifest.json").read_text())
    assert (doc["schema"], doc["schema_version"]) == ("rw-ingest-manifest", 2)
    pre = doc["episodes"][key]["preprocessing"]
    assert (pre["schema"], pre["schema_version"], pre["status"]) == ("rw-preprocess", 1, "done")
    rec = pre["stages"]["calibration"]
    assert rec["stage"] == "calibration" and rec["stage_schema_version"] == 1 and rec["failure"] is None
    assert rec["code_version"] == "calib-x1" and rec["output_format"] == "rw-calib/1" and rec["updated_at"]
    for bad in ({"status": "done", "code_version": "v", "output_format": "f", "input_sha256": "0" * 64},
                {"status": "done", "output_format": "f"},                 # no code_version
                {"status": "done", "code_version": "v"},                  # done without output_format
                {"status": "failed", "code_version": "v"},                # failed without failure
                {"status": "done", "code_version": "v", "output_format": "f", "version": "old"}):  # unknown field
        with pytest.raises(ValueError):
            W.set_stage(work / "state", key, "calibration", bad)
    W.set_stage(work / "state", key, "trackeverything", {"status": "failed", "code_version": "te-1",
                                                         "failure": {"message": "oom", "at": "t2"}})
    w.run_pass()
    pre = json.loads((work / "state/manifest.json").read_text())["episodes"][key]["preprocessing"]
    assert pre["status"] == "failed" and pre["stages"]["calibration"]["status"] == "done"


def test_manifest_v1_is_migrated(work):
    (work / "state").mkdir()
    v1 = {"schema": "rw-ingest-manifest/1", "episodes": {"drawer-open/episode_000001.h5": {
        "preprocessing": {"schema": "rw-preprocess-stage/1", "status": "pending", "stages": {}}}}}
    (work / "state/manifest.json").write_text(json.dumps(v1))
    st = W.State(work / "state")
    assert (st.doc["schema"], st.doc["schema_version"]) == ("rw-ingest-manifest", 2)
    pre = st.episodes["drawer-open/episode_000001.h5"]["preprocessing"]
    assert (pre["schema"], pre["schema_version"]) == ("rw-preprocess", 1) and st.doc["migrations"]
    st.save()  # the save merges the (v1) on-disk preprocessing block: it must come back migrated
    disk = json.loads((work / "state/manifest.json").read_text())
    assert disk["schema_version"] == 2
    assert disk["episodes"]["drawer-open/episode_000001.h5"]["preprocessing"]["schema"] == "rw-preprocess"
    v2_mixed = {"schema": "rw-ingest-manifest", "schema_version": 2, "episodes": {"drawer-open/episode_000002.h5": {
        "preprocessing": {"schema": "rw-preprocess-stage/1", "status": "pending", "stages": {}}}}}
    (work / "state/manifest.json").write_text(json.dumps(v2_mixed))
    assert W.State(work / "state").episodes["drawer-open/episode_000002.h5"]["preprocessing"]["schema"] == \
        "rw-preprocess"
    (work / "state/manifest.json").write_text(json.dumps({"schema": "rw-ingest-manifest", "schema_version": 9}))
    with pytest.raises(ValueError):
        W.State(work / "state")


def test_quiet_when_nothing_changes(base, work, capsys):
    raw = make_raw(work, base, {"episode_000017.h5": None})
    w = worker(work, raw, transfer=False)
    w.run_pass()
    capsys.readouterr()
    w.run_pass()
    w.run_pass()
    assert capsys.readouterr().out == ""
    w._last_summary_at -= W.HEARTBEAT_S
    w.run_pass()
    assert "heartbeat:" in capsys.readouterr().out


def test_second_worker_is_refused(work):
    (work / "state").mkdir()
    with W.worker_lock(work / "state"):
        r = subprocess.run([sys.executable, "-I", "-c", "import sys; sys.path.insert(0, %r); import ingest_worker as W;"
                            " from pathlib import Path\nwith W.worker_lock(Path(%r)): pass"
                            % (str(Path(W.__file__).parent), str(work / "state"))], capture_output=True, text=True)
        assert r.returncode != 0 and "another ingest worker" in r.stderr


def test_rsync_command_uses_dedicated_connection():
    shim = _RsyncShim(types.SimpleNamespace(client=types.SimpleNamespace(
        credentials=types.SimpleNamespace(password="\x00never"))))
    cmd, _ = W.RemoteOps.rsync_command(shim, Path("/x/episode_000001.h5"),
                                       "/r/data/real_world/drawer-open/.incoming/episode_000001.h5.part", 600)
    transport = shlex.split(cmd[cmd.index("-e") + 1])
    assert "ControlMaster=no" in transport and "ControlPath=none" in transport
    assert not any(x.startswith(("ControlPath=/", "ControlMaster=auto", "ControlPersist")) for x in transport)
    assert "--bwlimit=600" in cmd and "--partial" in cmd and "--append-verify" in cmd


def test_override_strictness_and_retry(base, work):
    raw = make_raw(work, base, {"episode_000017.h5": None})
    w = worker(work, raw, transfer=False)
    for value, expect in ((False, 600), ("false", 600), (1, 600), (True, 6000)):
        (work / "state/upload-budget.json").write_text(json.dumps({"kib_per_s": 6000, "coordinator_override": value}))
        assert w.upload_kib() == expect, value
    w.run_pass()
    key = "drawer-open/episode_000017.h5"
    e = w.state.episodes[key]
    e["flags"]["transfer_failed_repeatedly"] = "test"
    e["transfer"].update(status="failed", sha_mismatches=3)
    assert w.transfer_candidates({}) == []
    assert [x["key"] for x in w.transfer_candidates({key: {"decision": "retry"}})] == [key]
    assert e["transfer"]["status"] == "pending" and "transfer_failed_repeatedly" not in e["flags"]


def test_part_vanishing_before_publish_is_retried(base, work):
    raw = make_raw(work, base, {"episode_000017.h5": None})
    w = worker(work, raw)
    publish = w.remote.publish

    def vanish(part, final):
        os.remove(part)
        return publish(part, final)
    w.remote.publish = vanish
    w.run_pass()
    e = w.state.episodes["drawer-open/episode_000017.h5"]
    assert e["transfer"]["status"] == "partial" and not e["flags"] and "disappeared" in e["transfer"]["last_error"]
    w.remote.publish = publish
    e["transfer"]["next_attempt_unix"] = 0
    w.run_pass()
    assert e["transfer"]["status"] == "verified"


def test_one_camera_much_shorter_fails(base, work):
    cam = "cameras/cam_135122076054"

    def m(f):
        for grp in (f[cam], f[cam]["stereo"]):
            for ds in grp.values():
                if isinstance(ds, h5py.Dataset):
                    ds.resize((N_FRAMES - 6,))
        s = json.loads(f.attrs["stats_json"])
        s["rows"][cam] = s["rows"][cam + "/stereo"] = N_FRAMES - 6
        f.attrs["stats_json"] = json.dumps(s)
    r = check(variant(base, work, mutate=m))
    assert r["status"] == "failed" and any("differ across cameras" in x for x in r["reasons"]), r["reasons"]


def test_new_files_are_scanned_during_a_long_upload(base, work, monkeypatch):
    import threading

    raw = make_raw(work, base, {"episode_000017.h5": None})
    (work / "state").mkdir()
    (work / "state/upload-budget.json").write_text(json.dumps({"kib_per_s": 4000, "coordinator_override": True}))
    monkeypatch.setattr(W, "SCAN_DURING_TRANSFER_S", 1.0)
    w = worker(work, raw, max_transfers=1)

    def arrive():
        shutil.copyfile(_copy_reduced(TEMPLATE, work / "new.h5", n=N_FRAMES + 1, episode_index=18),
                        raw / "drawer-open/episode_000018.h5")
    threading.Timer(2.0, arrive).start()
    w.run_pass()
    eps = w.state.episodes
    assert eps["drawer-open/episode_000017.h5"]["transfer"]["status"] == "verified"
    new = eps["drawer-open/episode_000018.h5"]
    assert new["validation"]["status"] == "valid"
    assert new["validation"]["checked_at_unix"] < time.time() - 1  # validated while the first upload was running


def _drop_color_intrinsics(f):
    g = f[CAM]
    cal = json.loads(g.attrs["calibration_json"])
    del cal["streams"]["color"]["intrinsics"]
    g.attrs["calibration_json"] = json.dumps(cal)


def _flip_extrinsic(f):
    g = f[CAM]
    cal = json.loads(g.attrs["calibration_json"])
    e = cal["extrinsics"]["color_from_depth"]
    e["source"], e["target"] = e["target"], e["source"]
    g.attrs["calibration_json"] = json.dumps(cal)


@pytest.mark.parametrize("mutate, expect", [(_drop_color_intrinsics, "stream color incomplete"),
                                            (_flip_extrinsic, "extrinsic color_from_depth")])
def test_malformed_calibration_fails(base, work, mutate, expect):
    r = check(variant(base, work, mutate=mutate))
    assert r["status"] == "failed" and any(expect in x for x in r["reasons"]), r["reasons"]


def test_unexplained_adapter_refusal_fails_and_unavailable_adapter_warns(base, work, monkeypatch):
    p = variant(base, work)
    monkeypatch.setattr(V, "_adapter_check", lambda path: {"ok": False, "available": True, "errors": ["KeyError: x"],
                                                           "warnings": []})
    r = check(p)
    assert r["status"] == "failed" and any("not explained" in x for x in r["reasons"])
    monkeypatch.setattr(V, "_adapter_check", lambda path: {"ok": None, "available": False,
                                                           "errors": ["ImportError: y"], "warnings": []})
    r = check(p)
    assert r["status"] == "valid" and "adapter_unavailable" in r["warning_codes"]


def test_conflict_found_during_upload_blocks_publication(base, work, monkeypatch):
    import threading

    raw = make_raw(work, base, {"episode_000017.h5": None})
    (work / "state").mkdir()
    (work / "state/upload-budget.json").write_text(json.dumps({"kib_per_s": 4000, "coordinator_override": True}))
    monkeypatch.setattr(W, "SCAN_DURING_TRANSFER_S", 1.0)
    w = worker(work, raw, max_transfers=1)

    def arrive():  # different content, same episode_index 17 -> index conflict with the file being uploaded
        shutil.copyfile(_copy_reduced(TEMPLATE, work / "new.h5", n=N_FRAMES + 1),
                        raw / "drawer-open/episode_000018.h5")
    threading.Timer(2.0, arrive).start()
    w.run_pass()
    e = w.state.episodes["drawer-open/episode_000017.h5"]
    assert "index_conflict" in e["flags"] and e["transfer"]["status"] == "partial"
    assert "blocked before publish" in e["transfer"]["hold_reason"]
    assert not remote_file(work, "episode_000017.h5").exists()
    assert remote_file(work, ".incoming/episode_000017.h5.part").stat().st_size == base.stat().st_size


def _cal(mutator):
    def m(f):
        g = f[CAM]
        cal = json.loads(g.attrs["calibration_json"])
        mutator(cal)
        g.attrs["calibration_json"] = json.dumps(cal)
    return m


@pytest.mark.parametrize("mutator, expect", [
    (lambda c: c.__setitem__("depth_scale", float("inf")), "depth_scale"),
    (lambda c: c["streams"]["depth"]["intrinsics"].__setitem__("distortion_coeffs", [float("nan")] * 5),
     "distortion_coeffs"),
    (lambda c: c["streams"]["color"]["intrinsics"].__setitem__("distortion_coeffs", [0.0] * 3), "distortion_coeffs"),
])
def test_nonfinite_calibration_fails(base, work, mutator, expect):
    r = check(variant(base, work, mutate=_cal(mutator)), run_adapter=False)
    assert r["status"] == "failed" and any(expect in x for x in r["reasons"]), r["reasons"]


def test_known_condition_does_not_mask_other_adapter_errors(work, monkeypatch):
    p = _copy_reduced(TEMPLATE, work / "episode_000017.h5", drop_stereo_row=20)
    assert check(p)["status"] == "valid"
    monkeypatch.setattr(V, "_adapter_check", lambda path: {"ok": False, "available": True, "warnings": [],
                                                           "errors": ["KeyError: 'intrinsics'"]})
    r = check(p)
    assert r["status"] == "failed" and any("not explained" in x for x in r["reasons"])


def test_retry_override_acts_once_per_record(base, work):
    raw = make_raw(work, base, {"episode_000017.h5": None})
    w = worker(work, raw, transfer=False)
    w.run_pass()
    key = "drawer-open/episode_000017.h5"
    e = w.state.episodes[key]
    o = {key: {"decision": "retry", "at": "1"}}
    e["flags"]["transfer_failed_repeatedly"] = "x"
    assert w.transfer_candidates(o)
    e["flags"]["transfer_failed_repeatedly"] = "x again"
    assert not w.transfer_candidates(o)                      # same record: not re-applied
    assert w.transfer_candidates({key: {"decision": "retry", "at": "2"}})  # a new record acts once more


def test_migration_backfills_verification_snapshot(work):
    (work / "state").mkdir()
    ep = {"sha256": "a" * 64, "size": 5, "episode_index": 3,
          "transfer": {"status": "verified", "remote_sha256": "a" * 64},
          "preprocessing": {"schema": "rw-preprocess-stage/1", "status": "pending", "stages": {}}}
    other = {**ep, "sha256": "b" * 64, "transfer": {"status": "verified", "remote_sha256": "a" * 64}}
    (work / "state/manifest.json").write_text(json.dumps({"schema": "rw-ingest-manifest/1", "episodes": {
        "drawer-open/episode_000003.h5": ep, "drawer-open/episode_000004.h5": other}}))
    eps = W.State(work / "state").episodes
    assert eps["drawer-open/episode_000003.h5"]["transfer"]["verified_size"] == 5
    assert eps["drawer-open/episode_000004.h5"]["transfer"]["verified_episode_index"] is None
