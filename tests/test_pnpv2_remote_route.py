"""pick-and-place-v2 remote prep route: release-parametrized rwprep_remote, remote RGB publication, producer entries.

Offline: fake SSH client, fake git, no Docker, no network; dsreset defaults must stay unchanged.
"""

import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from ops.jobs import pnpv2_rgb
from ops.jobs import rwprep_remote as R
from ops.jobs.rwprep_mig import validate
from s4d.rwlearn.pnpv2_spec import SPEC, TASK, load_spec

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "c" * 40
IMAGE = "sha256:" + "d" * 64


# ------------------------------------------------------------------------------------------ processed release
@pytest.mark.parametrize("release,other", [("dsreset-20261113", "pnpv2-20261117"),
                                           ("pnpv2-20261117", "dsreset-20261113")])
def test_only_the_named_release_is_writable(tmp_path, release, other):
    root = tmp_path / "rw"
    (root / "processed" / other).mkdir(parents=True)
    broad = f"type=bind,src={root}/processed,dst={root}/processed"
    args = R.protect_processed(["docker", "run", "--mount", broad, "image"], str(root), release)
    mounts = [args[n + 1] for n, x in enumerate(args) if x == "--mount"]
    target = root / "processed" / release
    assert broad + ",readonly" in mounts
    assert mounts.count(f"type=bind,src={target},dst={target}") == 1
    assert not any(f"src={root / 'processed' / other}," in m for m in mounts)


def test_default_release_is_unchanged_and_unknown_release_refused(tmp_path):
    root = tmp_path / "rw"
    default = R.protect_processed(["docker", "run", "image"], str(root))
    explicit = R.protect_processed(["docker", "run", "image"], str(root), "dsreset-20261113")
    assert default == explicit
    with pytest.raises(ValueError, match="unknown processed release"):
        R.protect_processed(["docker", "run", "image"], str(root), "dsreset-20261113/../x")
    with pytest.raises(ValueError, match="unknown processed release"):
        R.release_of({"release": "drawer"})
    assert R.release_of({}) == "dsreset-20261113"


def test_guard_wrapper_mounts_the_configured_release(tmp_path, monkeypatch):
    """The embedded remote guard honours spec['release'] (absent = dsreset)."""
    import shutil

    raw = tmp_path / "raw"
    raw.write_bytes(b"raw")
    for release in (None, "pnpv2-20261117"):
        spec = dict(deadline=time.time() + 600, code="/code", commit="abc", worker_job="/configs/job.json",
                    stop_path=str(tmp_path / "STOP"), storage_root=str(tmp_path / "rw"),
                    inputs={str(raw): R.digest(raw)},
                    slot_guard=dict(uuid="MIG-x", ram_bytes=8 * R.GIB, reserve_bytes=120 * R.GIB))
        if release:
            spec["release"] = release
        job = dict(name="n", episode="ep", gpu="MIG-x", docker_args=["docker", "run", "w", "j"])
        calls = []

        def run(*args, calls=calls):
            calls.append(args)
            if args[:2] == ("docker", "ps"):
                return ""
            if args[:2] == ("git", "-C"):
                return "abc" if "rev-parse" in args else ""
            if args[:2] == ("docker", "run"):
                return "cid"
            raise AssertionError(args)

        monkeypatch.setattr(shutil, "disk_usage", lambda _: SimpleNamespace(free=50, total=100))
        monkeypatch.setattr(sys, "argv", ["guard", json.dumps(job)])
        env = dict(ROUTE_SPEC=spec, run=run, decide=lambda d: {"admit": True, "reasons": []}, json=json, sys=sys,
                   Path=Path, protect_processed=R.protect_processed)
        exec(R.GUARD_WRAPPER, env)
        command = next(c for c in calls if c[:2] == ("docker", "run"))
        target = tmp_path / "rw/processed" / (release or "dsreset-20261113")
        assert f"type=bind,src={target},dst={target}" in command


def test_planner_mounts_the_configured_release(tmp_path, monkeypatch):
    from ops.remote import launch_hold

    queue = tmp_path / "queue.yaml"
    queue.write_text("hosts:\n  remote:\n    cpu_thread_budget: 42\n    host_ram_reserve_gb: 220\n")
    monkeypatch.setattr(R, "LIVE_QUEUE", queue)
    monkeypatch.setattr(R, "RW", str(tmp_path / "rw"))
    monkeypatch.setattr(launch_hold, "check_shell", lambda: ":")
    monkeypatch.setattr(launch_hold, "locked", lambda body: body)
    launched = []

    def checked(command, **kw):
        source = shlex.split(command)[-1]
        library, body = source.split("\ndata=", 1)
        ns = {}
        exec(library, ns)
        ns["decide"] = lambda data: {"admit": True, "reasons": []}
        exec("\ndata=" + body, ns)
        return SimpleNamespace(stdout="planned")

    monkeypatch.setattr(subprocess, "run", lambda args, **kw: launched.append(args))
    client = SimpleNamespace(checked_ssh=checked)
    for release in (None, "pnpv2-20261117"):
        config = dict(code="/code", image=IMAGE, commit=COMMIT, **({"release": release} if release else {}))
        assert R.planner(client, config, ["-m", "planner"]) == "planned"
        target = tmp_path / "rw/processed" / (release or "dsreset-20261113")
        assert f"type=bind,src={target},dst={target}" in launched[-1]


# ------------------------------------------------------------------------------------------------ staging
class FakeGit:
    """git rev-parse/branch/clone for the staging clone; no repository is touched."""

    def __init__(self):
        self.cloned = []

    def check_output(self, args, **kw):
        if args[-1] == "HEAD":
            return COMMIT + "\n"
        if "--show-current" in args:
            return "main\n"
        raise AssertionError(args)

    def run(self, args, **kw):
        assert args[:2] == ["git", "clone"]
        Path(args[-1]).mkdir(parents=True)
        self.cloned.append(args)


@pytest.fixture
def staging(tmp_path, monkeypatch):
    git = FakeGit()
    monkeypatch.setattr(R, "subprocess", SimpleNamespace(check_output=git.check_output, run=git.run))
    monkeypatch.setattr(R, "REPO", tmp_path / "repo")
    tools = tmp_path / "tools"
    (tools / "rwprep/configs").mkdir(parents=True)
    (tools / "rwprep/configs/rwprep_server.yaml").write_text("weights: []\n")
    monkeypatch.setattr(R, "TOOLS", tools)
    monkeypatch.setattr(R, "established", lambda: (SimpleNamespace(push_tools=lambda c: "tools-1"), None))
    published = []
    monkeypatch.setattr(R, "publish_archive", lambda c, archive, dest, commit="": published.append(dest))
    client = SimpleNamespace(checked_ssh=lambda *a, **kw: SimpleNamespace(stdout=IMAGE + "\n"))
    return SimpleNamespace(client=client, published=published, tmp=tmp_path)


def plan_file(tmp_path, episodes):
    rows = []
    for ep in episodes:
        cfg = tmp_path / "cfg" / f"{ep}.yaml"
        cfg.parent.mkdir(parents=True, exist_ok=True)
        task, label, number = ep.rsplit("-", 2)
        raw = f"/home/ws/data/real_world/{task}/{label}/{number}.h5"
        cfg.write_text(yaml.safe_dump({"output_root": "x", "adapter": {"config": {"episodes": {ep: raw}}},
                                       "stages": {"calibrate": {}}}))
        rows.append({"episode": ep, "config": str(cfg), "raw_sha256": "e" * 64})
    path = tmp_path / "prep-plan.json"
    path.write_text(json.dumps(rows))
    return path


def test_pnpv2_phase_a_bundle_uses_its_release_stem_and_remote_name(staging):
    ep = "pick-and-place-v2-success-episode_000000"
    plan = plan_file(staging.tmp, [ep])
    out = staging.tmp / "work/phaseA"
    with pytest.raises(ValueError, match="job-id stem"):
        R.prepare([plan], out, staging.client, release="pnpv2-20261117", name="pnpv2-20261117-phaseA")
    with pytest.raises(ValueError, match="must start with pnpv2-20261117-"):
        R.prepare([plan], out, staging.client, release="pnpv2-20261117", stem="pnpv2-20261117-prepA")
    result = R.prepare([plan], out, staging.client, release="pnpv2-20261117", stem="pnpv2-20261117-prepA",
                       name="pnpv2-20261117-phaseA")
    remote = R.RW + "/configs/pnpv2-20261117-phaseA"
    assert result["remote"] == remote and result["launched"] is False
    config = json.loads((out / "dispatcher.json").read_text())
    jobs = json.loads((out / "jobs.json").read_text())
    remote_config = json.loads((out / "remote-dispatcher.json").read_text())
    assert config["release"] == "pnpv2-20261117" and config["remote"] == remote
    assert remote_config["processed"] == R.RW + "/processed/pnpv2-20261117"
    assert [j["id"] for j in jobs] == [f"pnpv2-20261117-prepA-{ep}"]
    cfg_key = next(k for k in jobs[0]["inputs"] if k.endswith(".yaml"))
    assert cfg_key == f"{remote}/configs/{ep}.yaml"
    assert f"{R.RAW}/{TASK}/success/episode_000000.h5" in jobs[0]["inputs"]
    compiled = yaml.safe_load((out / "configs" / f"{ep}.yaml").read_text())
    assert compiled["output_root"] == f"{R.RW}/processed/pnpv2-20261117/{TASK}/success/episode_000000"
    assert staging.published == [R.RW + "/code/dsreset-" + COMMIT, remote]


def test_dsreset_default_staging_is_unchanged(staging):
    ep = "drawer-open-v2-success-episode_000001"
    plan = plan_file(staging.tmp, [ep])
    with pytest.raises(ValueError, match="must start with dsreset-"):
        R.prepare([plan], staging.tmp / "x-bundle", staging.client)
    out = staging.tmp / "dsreset-test-bundle"
    R.prepare([plan], out, staging.client)
    config = json.loads((out / "dispatcher.json").read_text())
    remote_config = json.loads((out / "remote-dispatcher.json").read_text())
    assert "release" not in config and config["remote"] == R.RW + "/configs/dsreset-test-bundle"
    assert remote_config["processed"] == R.RW + "/processed/dsreset-20261113"
    assert [j["id"] for j in json.loads((out / "jobs.json").read_text())] == [
        f"rwv2-20261113-prep-ours-{ep}-phaseA"]


def _attach_client(docs):
    return SimpleNamespace(checked_ssh=lambda *a, **kw: SimpleNamespace(stdout=json.dumps(docs)))


def _jobs(prefix):
    return [{"id": prefix + "x", "gpu_profile": "1g.24gb", "gpu_gib": 2, "ram_gib": 32, "cpus": 4,
             "inputs": {"/m": "a" * 64}, "steps": [{"kind": "python", "argv": ["-m", "x"]}]}]


def test_attach_accepts_only_known_release_bundles(tmp_path):
    base = {"commit": COMMIT, "code": "/code", "tools_id": "t", "image": IMAGE, "release": "pnpv2-20261117"}
    with pytest.raises(ValueError, match="isolated"):
        R.attach(R.RW + "/configs/rwv2-other", tmp_path / "a", None, base)
    docs = {"dispatcher.json": {"commit": COMMIT, "release": "pnpv2-20261117"}, "jobs.json": _jobs("pnpv2-20261117-")}
    R.attach(R.RW + "/configs/pnpv2-20261117-rgb", tmp_path / "b", _attach_client(docs), base)
    assert json.loads((tmp_path / "b/dispatcher.json").read_text())["release"] == "pnpv2-20261117"
    # A dsreset-prefixed bundle can never be attached to the pnpv2 release (or the reverse).
    with pytest.raises(ValueError, match="release prefix"):
        R.attach(R.RW + "/configs/dsreset-x", tmp_path / "c", _attach_client(docs | {"dispatcher.json": {
            "commit": COMMIT}}), base)
    assert not (tmp_path / "c").exists()


def test_attach_inherits_the_base_release_and_keeps_dsreset_unchanged(tmp_path):
    """Phase B compiled by realworld_prep_register has no release key: it must write the base bundle's release."""
    pn = {"commit": COMMIT, "code": "/code", "tools_id": "t", "image": IMAGE, "release": "pnpv2-20261117"}
    docs = {"dispatcher.json": {"commit": COMMIT, "host": "remote"}, "jobs.json": _jobs("pnpv2-20261117-prepB-")}
    R.attach(R.RW + "/configs/pnpv2-20261117-phaseB", tmp_path / "b", _attach_client(docs), pn)
    assert R.release_of(json.loads((tmp_path / "b/dispatcher.json").read_text())) == "pnpv2-20261117"
    with pytest.raises(ValueError, match="differs from its base"):
        R.attach(R.RW + "/configs/pnpv2-20261117-x", tmp_path / "x", _attach_client(
            {**docs, "dispatcher.json": {"commit": COMMIT, "release": "dsreset-20261113"}}), pn)
    ds = {k: v for k, v in pn.items() if k != "release"}
    docs = {"dispatcher.json": {"commit": COMMIT, "host": "remote"}, "jobs.json": _jobs("rwv2-20261113-calib-")}
    R.attach(R.RW + "/configs/dsreset-calib-x", tmp_path / "d", _attach_client(docs), ds)
    assert "release" not in json.loads((tmp_path / "d/dispatcher.json").read_text())


# ------------------------------------------------------------------------------------------ remote RGB route
@pytest.fixture
def spec():
    return load_spec(ROOT / SPEC)


def test_remote_rgb_job_is_valid_bound_and_gated_on_the_phase_a_mirror(spec, tmp_path, staging):
    out = tmp_path / "work/rgb-remote"
    out.mkdir(parents=True)
    (out / "manifest.json").write_text("{}")
    phase = R.RW + "/configs/pnpv2-20261117-phaseA"
    ids = [f"pnpv2-20261117-prepA-{TASK}-success-episode_{i:06d}" for i in range(3)]
    job = pnpv2_rgb.remote_job(spec, out, phase, ids)
    validate({"host": "remote", "image": IMAGE, "commit": COMMIT}, [job])
    assert job["external_states"] == {f"{phase}/state/state.json": ids}
    assert job["steps"][0]["argv"][:3] == ["-m", "ops.jobs.pnpv2_rgb", "remote"]
    assert pnpv2_rgb.remote_out(spec).startswith(R.RW + "/processed/pnpv2-20261117/")
    result = R.stage([job], out, staging.client, commit=COMMIT, release="pnpv2-20261117",
                     name=pnpv2_rgb.RGB_BUNDLE, image=IMAGE, files=("manifest.json",))
    staged = json.loads((out / "jobs.json").read_text())[0]
    assert result["remote"] == R.RW + "/configs/" + pnpv2_rgb.RGB_BUNDLE
    assert set(staged["inputs"]) == {result["remote"] + "/manifest.json"}
    assert staged["steps"][0]["argv"][4] == result["remote"] + "/manifest.json"


def test_remote_raw_path_keeps_the_suffix():
    row = {"path": f"/home/ws/data/real_world/{TASK}/success/episode_000007.h5"}
    assert pnpv2_rgb.remote_raw(row) == f"{pnpv2_rgb.REMOTE_RAW}/{TASK}/success/episode_000007.h5"


def test_phase_a_roots_require_every_v2_job_done_and_normalized(tmp_path):
    bundle = tmp_path / "phaseA"
    (bundle / "state").mkdir(parents=True)
    (bundle / "configs").mkdir()
    eps = [f"{TASK}-success-episode_{i:06d}" for i in range(2)]
    jobs = []
    for ep in eps:
        root = tmp_path / "processed" / ep
        (root / "episodes" / ep / "normalized").mkdir(parents=True)
        (root / "episodes" / ep / pnpv2_rgb.STAGE_COMPLETE).write_text("{}")
        cfg = bundle / "configs" / f"{ep}.yaml"
        cfg.write_text(yaml.safe_dump({"output_root": str(root)}))
        jobs.append({"id": f"pnpv2-20261117-prepA-{ep}", "episode": ep, "inputs": {str(cfg): "a" * 64}})
    (bundle / "jobs.json").write_text(json.dumps(jobs))
    state = {"jobs": {j["id"]: {"status": "done"} for j in jobs}}
    (bundle / "state/state.json").write_text(json.dumps(state))
    assert set(pnpv2_rgb.phase_a_roots(bundle, eps)) == set(eps)
    state["jobs"][jobs[0]["id"]]["status"] = "running"
    (bundle / "state/state.json").write_text(json.dumps(state))
    with pytest.raises(ValueError, match="Phase A not complete"):
        pnpv2_rgb.phase_a_roots(bundle, eps)
    with pytest.raises(ValueError, match="episodes differ"):
        pnpv2_rgb.phase_a_roots(bundle, eps[:1])


def test_remote_build_refuses_a_non_canonical_output(tmp_path):
    with pytest.raises(ValueError, match="canonical release directory"):
        pnpv2_rgb.remote_build(tmp_path / "m.json", tmp_path / "phaseA", tmp_path / "out")


# ------------------------------------------------------------------------------------------ producer entries
def test_phase_a_remote_refuses_before_every_remote_raw_copy_is_verified(spec, tmp_path, monkeypatch):
    import scripts.pick_and_place_v2 as P

    spec["roots"]["work"] = str(tmp_path / "work")
    ingest = tmp_path / "work/ingest/S"
    ingest.mkdir(parents=True)
    rows = [{"sha256": f"{i:064x}"} for i in range(3)]
    (ingest / "manifest.json").write_text(json.dumps({"episodes": rows}))
    monkeypatch.setattr(P, "_check_data", lambda spec, a: {"passed": True})
    monkeypatch.setattr(P, "ATTACH_ROOT", tmp_path / "runs")
    called = []
    monkeypatch.setattr(R, "prepare", lambda *a, **kw: called.append(kw) or {"launched": False})
    monkeypatch.setattr(R, "RemoteClient", lambda: None)
    with pytest.raises(ValueError, match="raw-proof first"):
        P.phase_a_remote(spec, None, [])
    proof = tmp_path / "work/remote-raw-hashes.json"
    proof.write_text(json.dumps({r["sha256"]: r["sha256"] for r in rows[:2]}))
    with pytest.raises(ValueError, match="differs"):
        P.phase_a_remote(spec, None, [])
    proof.unlink()
    proof.write_text(json.dumps({r["sha256"]: r["sha256"] for r in rows}))
    P.phase_a_remote(spec, None, [])
    assert called == [{"release": "pnpv2-20261117", "stem": "pnpv2-20261117-prepA", "name": "pnpv2-20261117-phaseA"}]
    # The attachment lives in the transfer root (repository runs/); the work root links to it.
    assert (tmp_path / "work/phaseA").readlink() == tmp_path / "runs/pnpv2-20261117-phaseA"
    with pytest.raises(ValueError, match="immutable"):
        P.phase_a_remote(spec, None, [])


def test_status_points_at_the_remote_phase_a_route(spec, tmp_path):
    from ops.jobs import pnpv2_registration as reg

    ev = {k: {"status": "done"} for k in ("arrival", "check-data", "ingest")}
    ev.update({k: {"status": "pending"} for k in ("phaseA", "rgb-publication", "rig", "phaseB",
                                                  "geometry-publication", "scene")})
    nxt = reg.next_steps(spec, ev)
    assert any("phase-a-remote" in c for c in nxt["phaseA"]) and any("raw-proof" in c for c in nxt["phaseA"])
    assert not any("LOCAL HOST" in c for c in nxt["phaseA"])


# --------------------------------------------------------------------------------- geometry route entries
def test_calib_remote_plans_on_the_remote_and_attaches(spec, tmp_path, monkeypatch):
    import scripts.pick_and_place_v2 as P

    spec["roots"]["work"] = str(tmp_path / "work")
    phase = tmp_path / "work/phaseA"
    phase.mkdir(parents=True)
    base = {"remote": R.RW + "/configs/pnpv2-20261117-phaseA", "release": "pnpv2-20261117", "commit": COMMIT}
    (phase / "dispatcher.json").write_text(json.dumps(base))
    calls = []
    monkeypatch.setattr(R, "RemoteClient", lambda: "client")
    monkeypatch.setattr(P, "ATTACH_ROOT", tmp_path / "runs")
    monkeypatch.setattr(R, "planner", lambda client, config, argv: calls.append(("plan", config, argv)) or "{}")
    monkeypatch.setattr(R, "attach", lambda remote, out, client, b: calls.append(("attach", remote, out, b)) or {})
    P.calib_remote(spec, None, [])
    (_, config, argv), (_, remote, out, b) = calls
    assert config == base == b and argv[0] == "-c" and remote == R.RW + "/configs/pnpv2-20261117-calib"
    assert argv[2:8] == [base["remote"] + "/dispatcher.json", TASK, remote,
                         f"{R.RW}/processed/pnpv2-20261117/calibration/{TASK}", P.CALIB_UPSTREAM,
                         "pnpv2-20261117-calib"]
    assert out == tmp_path / "runs/pnpv2-20261117-calib"
    assert (tmp_path / "work/calib").readlink() == out
    with pytest.raises(ValueError, match="immutable"):
        P.calib_remote(spec, None, [])


def test_calibration_planner_code_keeps_the_frozen_plan_and_adds_readonly_mounts(tmp_path, monkeypatch):
    import scripts.pick_and_place_v2 as P
    import scripts.realworld_dsreset_calibration as C

    def plan(phase, task, out, work, upstream, stem=None):
        assert stem == "pnpv2-20261117-calib"
        out.mkdir()
        (out / "jobs.json").write_text(json.dumps([{"id": "j"}]))
        (out / "dispatcher.json").write_text(json.dumps({"extra_readonly": [upstream]}))

    monkeypatch.setattr(C, "plan", plan)
    out = tmp_path / "calib"
    argv = ["x", "phase.json", TASK, str(out), "work", "/up", "pnpv2-20261117-calib", json.dumps(["/ro"])]
    for _ in range(2):  # re-running is idempotent
        monkeypatch.setattr(sys, "argv", argv)
        exec(P._CALIB_PLAN, {})
        assert json.loads((out / "dispatcher.json").read_text())["extra_readonly"] == ["/up", "/ro"]
    (out / "state").mkdir()
    (out / "state/state.json").write_text("{}")
    monkeypatch.setattr(sys, "argv", argv[:-1] + [json.dumps(["/other"])])
    with pytest.raises(AssertionError, match="already used"):
        exec(P._CALIB_PLAN, {})


def test_phase_b_planner_code_binds_one_rig_for_every_episode(tmp_path, monkeypatch):
    import scripts.pick_and_place_v2 as P

    base = tmp_path / "phaseA"
    base.mkdir()
    keys = ("processed", "raw", "weights", "image", "tools", "assets")
    (base / "dispatcher.json").write_text(json.dumps({k: f"/{k}" for k in keys}))
    (base / "prep-plan.json").write_text(json.dumps([{"episode": "e1"}, {"episode": "e2"}]))
    out = tmp_path / "phaseB"
    ran = []

    def run(args, check):
        ran.append(args)
        out.mkdir()
        (out / "jobs.json").write_text("[]")

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(sys, "argv", ["x", str(base), str(out), "/rig.json", "pnpv2-20261117-prepB", "/calib"])
    exec(P._PHASE_B_PLAN, {})
    rigmap = tmp_path / "phaseB.rig-map.json"
    assert json.loads(rigmap.read_text()) == {"e1": "/rig.json", "e2": "/rig.json"}
    args = ran[0]
    assert args[1:5] == ["scripts/realworld_prep_register.py", "--phase", "B", "--id-stem"]
    assert "pnpv2-20261117-prepB" in args and args[args.index("--rig-map") + 1] == str(rigmap)
    assert all(args[args.index("--" + k) + 1] == f"/{k}" for k in keys)
    monkeypatch.setattr(sys, "argv", ["x", str(base), str(out), "/other-rig.json", "pnpv2-20261117-prepB", "/c"])
    with pytest.raises(AssertionError, match="rig map differs"):
        exec(P._PHASE_B_PLAN, {})


def test_rig_fetch_and_phase_b_refuse_before_an_accepted_rig(spec, tmp_path, monkeypatch):
    import scripts.pick_and_place_v2 as P

    spec["roots"]["work"] = str(tmp_path / "work")
    spec["roots"]["processed"] = str(tmp_path / "processed")
    (tmp_path / "work/calib/state").mkdir(parents=True)
    (tmp_path / "work/calib/state/state.json").write_text(json.dumps({"jobs": {}}))
    monkeypatch.setattr(R, "RemoteClient", lambda: pytest.fail("no remote access before acceptance"))
    with pytest.raises(ValueError, match="publish is not done"):
        P.rig_fetch(spec, None, [])
    with pytest.raises(ValueError, match="task rig not accepted"):
        P.phase_b_remote(spec, None, [])


def test_upload_stages_copies_inside_the_transfer_root_and_writes_the_receipt(spec, tmp_path, monkeypatch):
    import hashlib

    import scripts.pick_and_place_v2 as P
    from ops.remote import access
    from s4d.rwlearn.pnpv2_spec import BUNDLE_FILES, values

    spec["roots"]["work"] = str(tmp_path / "work")
    monkeypatch.setattr(P, "ATTACH_ROOT", tmp_path / "runs")
    p = values(spec)
    local = Path(p["rgb_dir_local"])
    local.mkdir(parents=True)
    for _, name in BUNDLE_FILES:
        (local / name).write_text(name)
    remote_files, uploads = {}, []

    class Client:
        def _remote_checksum(self, target, allow_missing=False):
            return remote_files.get(target)

        def rsync_upload(self, source, target):
            assert Path(source).is_relative_to(tmp_path / "runs")  # never from the work root
            uploads.append(target)
            remote_files[target] = hashlib.sha256(Path(source).read_bytes()).hexdigest()

    monkeypatch.setattr(access, "RemoteClient", Client)
    P.upload(spec, None, ["--view", "rgb"])
    receipt = json.loads((local.parent / "rgb.UPLOADED.json").read_text())
    assert receipt["schema"] == "pick-and-place-v2-upload/1" and set(receipt["files"]) == set(uploads)
    assert len(uploads) == len(BUNDLE_FILES) and not list((tmp_path / "runs").iterdir())
