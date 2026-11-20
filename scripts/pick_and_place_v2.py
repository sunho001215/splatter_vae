"""pick-and-place-v2 upstream producers (operator steps between arrival and queue registration).

Entries (each refuses before the dataset is complete/validated; none launches a job or contacts the Robot PC):

  manifest     canonical S manifest + per-episode prep plan            -> <work>/ingest
  rig-map      Phase-B rig map, only once the task rig is accepted      -> <work>/phaseB-rig-map.json
  publish-rgb  calibration-independent RGB bundle from Phase A          -> <work>/publication/S/rgb
  raw-proof    read-only remote sha256 of the raw copies (snapshot)     -> <work>/remote-raw-hashes.json
  snapshot     NEW ingest snapshot from Phase A/B dispatcher records    -> <work>/ingest-snapshot
  publish      geometry bundle from the snapshot (calibration records)  -> <work>/publication/S/geometry
  upload       checked no-overwrite upload of one staged bundle + receipt (remote write; coordinator only)

Remote Phase-A route (the local host controller needs host Docker, canonical bind mounts and free local MIGs):

  phase-a-remote  compile + stage the 100 Phase-A jobs as remote bundle pnpv2-20261117-phaseA (ops/jobs/rwprep_remote,
                  writable release processed/pnpv2-20261117 only); needs raw-proof        -> <work>/phaseA
  rgb-remote      compile + stage the one-job RGB publication bundle pnpv2-20261117-rgb (waits on the Phase-A state
                  mirror)                                                                  -> <work>/rgb-remote
  rgb-fetch       copy the small RGB bundle documents of the finished remote publication -> <work>/publication/S/rgb
                  (then ``upload -- --view rgb`` publishes them to the canonical directory)

  calib-remote    compile the task LoMa-G rig plan on the remote (planner container) as bundle
                  pnpv2-20261117-calib; its view job waits on the Phase-A state mirror     -> <work>/calib
  rig-fetch       mirror the published accepted rig + reference locally (what ``status`` checks)
  phase-b-remote  compile Phase B on the remote against the accepted rig as bundle pnpv2-20261117-phaseB
                                                                                           -> <work>/phaseB

  geometry-remote stage snapshot/release/publication/native measurement as one admitted remote job
                                                                                           -> <work>/geometry-remote
  geometry-fetch  hash-verified compact closure into runs/realworld-inputs + local status evidence
                  (then ``upload -- --view geometry`` publishes canonical learning-input documents)

Ticks: ``python -m ops.jobs.rwprep_remote tick --config <bundle>/dispatcher.json --apply --loop`` (from main).

Queue rows never call this file; they use scripts/pnpv2_*.py. Status and next commands: scripts/pnpv2.py status.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

from s4d.rwlearn.pnpv2_spec import PREP_DISK_GIB, SPEC, STEM, TASK, load_spec, read_arrival, values  # noqa: E402


def _atomic_json(path, doc):
    # Inline equivalent of the real-world package atomic JSON writer: scripts outside scripts/realworld_* must not import
    # that package (tests/test_realworld_contract_workers.py import boundary).
    import os

    path = Path(path)
    if path.exists():
        raise ValueError(f"{path} exists; outputs are immutable")
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(doc, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with open(tmp, "wb") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _check_data(spec, a):
    from s4d.rwlearn.pnpv2_arrival import check_data

    report = check_data(spec, raw_root=a.raw_root, settle_seconds=a.settle_seconds)
    if not report["passed"]:
        raise ValueError("check-data refuses: " + "; ".join(report["problems"][:5]))
    return report


def manifest(spec, a, argv):
    from s4d.rwlearn.dsreset import prep_bundle, write_bundle
    from s4d.rwlearn.pnpv2_publication import manifest_check
    from s4d.rwlearn.pnpv2_spec import config

    p = values(spec)
    if not spec.get("sam3_prompts") or not spec.get("robot_urdf_sha256"):
        raise ValueError(f"write the reviewed confirmation {p['arrival_doc']} first (prompts, robot URDF sha256)")
    from s4d.rwlearn.pnpv2_arrival import sha256

    if sha256(p["robot_urdf"]) != spec["robot_urdf_sha256"]:
        raise ValueError("confirmed robot URDF sha256 differs from the asset file")
    report = _check_data(spec, a)
    m = manifest_check(report["manifest"], spec)
    template = config(Path(__file__).resolve().parents[1] / "configs/realworld/prep/pick-and-place-v2.yaml", spec)
    template["stages"]["train_data"]["task"] = TASK
    out = Path(p["ingest"])
    if out.exists():
        raise ValueError(f"{out} exists; the ingest bundle is immutable")
    write_bundle(out, {"S": m}, prep_bundle({"S": m}, template, p["remote_processed"]))
    print(json.dumps({"out": str(out), "episodes": len(m["episodes"]), "launched": False}))


def rig_map(spec, a, argv):
    from ops.jobs.pnpv2_registration import rig_evidence

    p = values(spec)
    stage = rig_evidence(spec)
    if stage["status"] != "done":
        raise ValueError(f"task rig not accepted: {stage['detail']}")
    plan = json.loads(Path(p["prep_plan"]).read_text())
    _atomic_json(Path(p["work"]) / "phaseB-rig-map.json", {r["episode"]: p["rig"] for r in plan})
    print(json.dumps({"episodes": len(plan), "rig": p["rig"]}))


def publish_rgb(spec, a, argv):
    from ops.jobs.pnpv2_rgb import publish_rgb as publish

    sub = argparse.ArgumentParser(prog="publish-rgb")
    sub.add_argument("--execute", action="store_true")
    b = sub.parse_args(argv)
    p = values(spec)
    _check_data(spec, a)
    print(json.dumps(publish(spec, p["manifest"], p.get("rgb_source", p["phase_a"]), execute=b.execute), indent=2))


def raw_proof(spec, a, argv):
    """Read-only: sha256 of each canonical raw recording's REMOTE copy; refuses any mismatch."""
    from ops.remote.access import RemoteClient

    p = values(spec)
    m = json.loads(Path(p["manifest"]).read_text())
    client = RemoteClient()
    proofs = {}
    for r in m["episodes"]:
        remote = client._remote_checksum(f"{p['remote_raw_success']}/{r['filename']}", allow_missing=True)
        if remote != r["sha256"]:
            raise ValueError(f"{r['filename']}: remote raw copy missing or different")
        proofs[r["sha256"]] = remote
    _atomic_json(Path(p["work"]) / "remote-raw-hashes.json", proofs)
    print(json.dumps({"verified": len(proofs)}))


def snapshot(spec, a, argv):
    import scripts.realworld_dsreset_snapshot as module
    from s4d.rwlearn.pnpv2_publication import manifest_check

    def read_manifest(path, *, success_only=False):
        # The historical --success-only reader requires drawer/v1 amendments; v2 is the unamended 100-success set.
        if not success_only:
            raise ValueError("pick-and-place-v2 snapshot requires --success-only")
        return manifest_check(json.loads(Path(path).read_text()), spec)

    previous_reader, previous_argv = module.read_manifest, sys.argv
    try:
        module.read_manifest = read_manifest
        sys.argv = [module.__file__, *argv]
        module.main()
    finally:
        module.read_manifest, sys.argv = previous_reader, previous_argv


def publish(spec, a, argv):
    """Geometry bundle from the NEW snapshot, accepting only the committed offline override."""
    from s4d.rwlearn.pnpv2_offline import build_geometry

    sub = argparse.ArgumentParser(prog="publish")
    sub.add_argument("--ingest-manifest", type=Path, required=True)
    b = sub.parse_args(argv)
    p = values(spec)
    print(json.dumps(build_geometry(spec, p["manifest"], b.ingest_manifest, p["geometry_dir_local"])))


def upload(spec, a, argv):
    """Checked, no-overwrite upload of a staged bundle (and RGB episodes) to its canonical remote directory."""
    from ops.remote.access import RemoteClient
    from s4d.rwlearn.pnpv2_arrival import sha256
    from s4d.rwlearn.pnpv2_spec import BUNDLE_FILES

    sub = argparse.ArgumentParser(prog="upload")
    sub.add_argument("--view", choices=("rgb", "geometry"), required=True)
    b = sub.parse_args(argv)
    p = values(spec)
    local, remote = Path(p[f"{b.view}_dir_local"]), p[f"{b.view}_dir"]
    files = [local / n for _, n in BUNDLE_FILES]
    if b.view == "rgb":
        files += sorted(f for f in (local / "episodes").rglob("*") if f.is_file())
    if b.view == "geometry" and (Path(p["work"]) / "geometry-remote").exists():
        from ops.jobs.pnpv2_geometry import mirror_evidence
        mirror_evidence(spec, canonical=False)  # validate the fetched publication BEFORE any canonical upload
    client = RemoteClient()
    sent = {}
    # RemoteClient uploads only from the repository/data roots: stage a verified copy under runs/ for each file.
    import shutil
    import tempfile

    ATTACH_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{STEM}-upload-{b.view}-", dir=ATTACH_ROOT) as tmp:
        for f in files:
            target = f"{remote}/{f.relative_to(local)}"
            digest = sha256(f)
            existing = client._remote_checksum(target, allow_missing=True)
            if existing not in (None, digest):
                raise ValueError(f"refusing to overwrite a different remote file: {target}")
            if existing is None:
                copy = Path(tmp) / f.relative_to(local)
                copy.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(f, copy)
                if sha256(copy) != digest:
                    raise ValueError(f"staging copy differs: {f}")
                client.rsync_upload(copy, target)
                if client._remote_checksum(target) != digest:
                    raise ValueError(f"uploaded file differs: {target}")
            sent[target] = digest
    if b.view == "geometry" and (Path(p["work"]) / "geometry-remote").exists():
        from ops.jobs.pnpv2_geometry import mirror_published
        mirror_published(spec, client)
    _atomic_json(local.parent / f"{b.view}.UPLOADED.json",
                 {"schema": "pick-and-place-v2-upload/1", "view": b.view, "files": sent})
    print(json.dumps({"uploaded": len(sent), "view": b.view}))


def _remote_raw_verified(spec):
    p = values(spec)
    proof = Path(p["work"]) / "remote-raw-hashes.json"
    m = json.loads(Path(p["manifest"]).read_text())
    if not proof.is_file():
        raise ValueError("run --entry raw-proof first: every remote raw copy must be verified before Phase A")
    proofs = json.loads(proof.read_text())
    if TASK == 'ball-hiding':
        from s4d.rwlearn.ball_hiding_membership import amendment
        # The original all-100 raw proof remains immutable and must bind exactly those 100 hashes.
        a = amendment()
        expected = set(a['episodes'].values()) | {r['raw_sha256'] for r in a['excluded'].values()}
        if proofs != {h: h for h in expected}:
            raise ValueError('BH original remote raw proof differs from the exact 100-recording inventory')
        return m
    if any(proofs.get(r["sha256"]) != r["sha256"] for r in m["episodes"]) or len(proofs) != len(m["episodes"]):
        raise ValueError("remote raw proof differs from the canonical manifest")
    return m


ATTACH_ROOT = Path(__file__).resolve().parents[1] / "runs"


def _attachment(link, bundle):
    """(real, link): local bundle attachments live in the repository runs/ directory, the remote client's authorized
    transfer root (staging uploads code/bundle archives from there); ``link`` in the work root points at it."""
    link = Path(link)
    real = ATTACH_ROOT / bundle
    if link.exists() or link.is_symlink() or real.exists():
        raise ValueError(f"{link} or {real} exists; bundle attachments are immutable")
    return real, link


def _link(real, link):
    import os

    os.symlink(real, link, target_is_directory=True)
    for name in ("code.tar", "bundle.tar"):  # staging archives, already published remotely
        (real / name).unlink(missing_ok=True)


def phase_a_remote(spec, a, argv):
    """Phase A (ingest, native depth, FoundationStereo, SAM3) on the remote prep route; launches nothing."""
    from ops.jobs import rwprep_remote
    from ops.jobs.pnpv2_registration import STAGE_IDS

    p = values(spec)
    _check_data(spec, a)
    _remote_raw_verified(spec)
    out, link = _attachment(p["phase_a"], PHASE_A_BUNDLE)
    result = rwprep_remote.prepare([Path(p["prep_plan"])], out, rwprep_remote.RemoteClient(),
                                   release=rwprep_remote_release(spec), stem=STAGE_IDS["phaseA"],
                                   name=PHASE_A_BUNDLE, disk_gib=PREP_DISK_GIB)
    _link(out, link)
    print(json.dumps(result))


PHASE_A_BUNDLE = f"{STEM}-phaseA"


def rwprep_remote_release(spec):
    release = Path(values(spec)["remote_processed"]).name
    if release != STEM:
        raise ValueError("pick-and-place-v2 writes only the pnpv2-20261117 processed release")
    return release


def rgb_remote(spec, a, argv):
    """One-job remote RGB publication bundle; admitted by its tick only after every Phase-A job is done."""
    import shutil

    from ops.jobs import pnpv2_rgb, rwprep_remote

    p = values(spec)
    _check_data(spec, a)
    config = json.loads((Path(p.get("rgb_source", p["phase_a"])) / "dispatcher.json").read_text())
    jobs = json.loads(Path(config["jobs"]).read_text())
    out, link = _attachment(Path(p["work"]) / ("rgb2-remote" if TASK == "ball-hiding" else "rgb-remote"), pnpv2_rgb.RGB_BUNDLE)
    out.mkdir(parents=True)
    shutil.copyfile(p["manifest"], out / "manifest.json")
    if TASK == 'ball-hiding':
        from s4d.rwlearn.ball_hiding_membership import eligible_jobs
        jobs = eligible_jobs(jobs)
    job = pnpv2_rgb.remote_job(spec, out, config["remote"], [j["id"] for j in jobs])
    result = rwprep_remote.stage([job], out, rwprep_remote.RemoteClient(), commit=None if TASK == "ball-hiding" else config["commit"],
                                 release=rwprep_remote_release(spec), name=pnpv2_rgb.RGB_BUNDLE,
                                 image=config["image"], files=("manifest.json",))
    _link(out, link)
    print(json.dumps(result))


def rgb_fetch(spec, a, argv):
    """Copy the finished remote RGB bundle documents into the local staging directory (never the episodes)."""
    import shlex

    from ops.jobs import pnpv2_rgb
    from ops.remote.access import RemoteClient
    from s4d.rwlearn.pnpv2_publication import manifest_check

    p = values(spec)
    state = json.loads((Path(p["work"]) / ("rgb2-remote/state/state.json" if TASK == "ball-hiding" else "rgb-remote/state/state.json")).read_text())["jobs"]
    if (state.get(pnpv2_rgb.RGB_JOB) or {}).get("status") != "done":
        raise ValueError("remote RGB publication job is not done")
    local = Path(p["rgb_dir_local"])
    if local.exists():
        raise ValueError(f"{local} exists; publications are immutable")
    remote = pnpv2_rgb.remote_out(spec)
    script = ("import json,pathlib,sys; d=pathlib.Path(sys.argv[1]); "
              "print(json.dumps({n:(d/n).read_text() for n in sys.argv[2:]}))")
    docs = json.loads(RemoteClient().checked_ssh("python3 -I -c " + shlex.quote(script) + " " + shlex.join(
        [remote, *pnpv2_rgb.DOCS]), idempotent=True).stdout)
    m = manifest_check(json.loads(docs["manifest.json"]), spec)
    if m != json.loads(Path(p["manifest"]).read_text()):
        raise ValueError("remote RGB publication manifest differs from the ingest manifest")
    store = yaml.safe_load(docs["store.yaml"])
    if store.get("kind") != "compact" or len(store.get("paths", [])) != len(m["episodes"]) or not all(
            x.startswith(remote + "/episodes/") for x in store["paths"]):
        raise ValueError("remote RGB store.yaml must name exactly the eligible remote published episodes")
    local.mkdir(parents=True)
    for name, text in docs.items():
        (local / name).write_text(text)
    print(json.dumps({"out": str(local), "files": sorted(docs), "remote": remote}))


REMOTE_RW = "/home/compu/kaist/sunho/realworld"
CALIB_BUNDLE = f"{STEM}-calib2" if TASK == "ball-hiding" else f"{STEM}-calib"
PHASE_B_BUNDLE = f"{STEM}-phaseB"
# The established LoMa/romav2 upstream staged for the dsreset calibrations (read-only; torch-home links into
# calib-loma / calib-dw, which the jobs therefore also mount read-only, as the drawer-open-v2 calibration does).
CALIB_UPSTREAM = f"{REMOTE_RW}/configs/dsreset-upstream-20261114"
CALIB_READONLY = (f"{REMOTE_RW}/calib-loma", f"{REMOTE_RW}/calib-dw", f"{REMOTE_RW}/assets/validation-20261031")
_CALIB_PLAN = """
import json, pathlib, sys
from ops.jobs.rwprep_mig import atomic
from scripts.realworld_dsreset_calibration import plan
phase, task, out, work, upstream, stem, extra = sys.argv[1:8]
out = pathlib.Path(out)
if not out.exists():
    plan(phase, task, out, work, upstream, stem=stem)
c = json.loads((out / 'dispatcher.json').read_text())
if task == 'ball-hiding':
    assert len(sys.argv) == 10, 'BH calibration needs the fresh source pin'
    assert not (out / 'state/state.json').exists(), 'HELD: calibration queue already used'
    c.update(commit=sys.argv[8], code=sys.argv[9])
    atomic(out / 'dispatcher.json', c)
want = list(dict.fromkeys([*c.get('extra_readonly', []), *json.loads(extra)]))
if c.get('extra_readonly') != want:
    assert not (out / 'state/state.json').exists(), 'HELD: calibration queue already used'
    c['extra_readonly'] = want
    atomic(out / 'dispatcher.json', c)
print(json.dumps({'jobs': len(json.loads((out / 'jobs.json').read_text())), 'extra_readonly': want}))
"""
_PHASE_B_PLAN = """
import json, pathlib, subprocess, sys
from ops.jobs.rwprep_mig import atomic
base, out, rig, stem, extra, disk = sys.argv[1:7]
base, out = pathlib.Path(base), pathlib.Path(out)
c = json.loads((base / 'dispatcher.json').read_text())
rows = json.loads((base / 'prep-plan.json').read_text())
if stem.startswith('bh-20261120-'):
    from s4d.rwlearn.ball_hiding_membership import eligible_jobs, amendment
    rows = eligible_jobs(rows)
    assert len(rows) == amendment()['target'], 'BH Phase B requires 99 eligible episodes'
plan_path = base / 'prep-plan.json'
if stem.startswith('bh-20261120-'):
    plan_path = out.parent / (out.name + '.eligible-plan.json')
    if plan_path.exists():
        assert json.loads(plan_path.read_text()) == rows, 'HELD: eligible Phase-B plan differs'
    else:
        atomic(plan_path, rows)
rigmap, doc = out.parent / (out.name + '.rig-map.json'), {r['episode']: rig for r in rows}
if rigmap.exists():
    assert json.loads(rigmap.read_text()) == doc, 'HELD: rig map differs'
else:
    atomic(rigmap, doc)
if not out.exists():
    override = len(sys.argv) > 7 and sys.argv[7] == 'acceptance_override'
    from s4d.rwlearn.pnpv2_offline import phase_b_compiler
    compiler = phase_b_compiler(rig, override)
    args = [sys.executable, compiler, '--phase', 'B', '--id-stem', stem,
            '--plan', str(plan_path), '--out', str(out), '--host', 'remote',
            '--code', '/opt/src/splatter4d', '--rig-map', str(rigmap), '--extra-readonly', extra]
    for k in ('processed', 'raw', 'weights', 'image', 'tools', 'assets'):
        args += ['--' + k, c[k]]
    subprocess.run(args, check=True)
jobs = json.loads((out / 'jobs.json').read_text())
if any(j['disk_gib'] != int(disk) for j in jobs):  # measured declaration (the compiler's default is 32 GiB)
    assert not (out / 'state/state.json').exists(), 'HELD: Phase-B queue already used'
    for j in jobs:
        j['disk_gib'] = int(disk)
    atomic(out / 'jobs.json', jobs)
print(json.dumps({'jobs': len(jobs), 'disk_gib': int(disk)}))
"""


def _phase_a_config(spec):
    return json.loads((Path(values(spec)["phase_a"]) / "dispatcher.json").read_text())


def calib_remote(spec, a, argv):
    """Task LoMa-G rig on the remote route (established planner and automatic G1-G6 publication; no override)."""
    from ops.jobs import rwprep_remote
    from ops.jobs.pnpv2_registration import STAGE_IDS

    p = values(spec)
    base = _phase_a_config(spec)
    if TASK == 'ball-hiding':
        fresh = json.loads((Path(p['work']) / 'rgb2-remote/dispatcher.json').read_text())
        base.update({k: fresh[k] for k in ('code', 'commit')})
    out, link = _attachment(p["calib"], CALIB_BUNDLE)
    remote = f"{REMOTE_RW}/configs/{CALIB_BUNDLE}"
    client = rwprep_remote.RemoteClient()
    print(rwprep_remote.planner(client, base, [
        "-c", _CALIB_PLAN, f"{base['remote']}/dispatcher.json", TASK, remote,
        f"{p['remote_processed']}/calibration/{TASK}", CALIB_UPSTREAM, STAGE_IDS["rig"],
        json.dumps([*CALIB_READONLY]), *([base["commit"], base["code"]] if TASK == "ball-hiding" else [])]))
    print(json.dumps(rwprep_remote.attach(remote, out, client, base)))
    _link(out, link)


def rig_fetch(spec, a, argv):
    """Mirror the remote accepted rig and its reference (sha-checked download); refuses before publication."""
    from ops.jobs import rwprep_remote

    p = values(spec)
    state = json.loads((Path(p["calib"]) / "state/state.json").read_text())["jobs"]
    from ops.jobs.pnpv2_registration import STAGE_IDS

    publish_job = f"{STAGE_IDS['rig']}-{TASK}-publish"
    if (state.get(publish_job) or {}).get("status") != "done":
        raise ValueError(f"calibration publication {publish_job} is not done")
    local = Path(p["rig_local"])
    if local.exists() or local.with_suffix(".reference.npz").exists():
        raise ValueError(f"{local} exists; the rig mirror is immutable")
    client = rwprep_remote.RemoteClient()
    for name in ("automatic-decision.json", "override-decision.json"):
        remote = str(Path(p["rig"]).parent / name)
        if client._remote_checksum(remote, allow_missing=True) is not None:
            client.rsync_download(remote, local.parent / name)
    for remote, target in ((p["rig"], local), (str(Path(p["rig"]).with_suffix(".reference.npz")),
                                               local.with_suffix(".reference.npz"))):
        client.rsync_download(remote, target)
    from ops.jobs.pnpv2_registration import rig_evidence

    print(json.dumps({k: v for k, v in rig_evidence(spec).items() if k != "inputs"}))


def phase_b_remote(spec, a, argv):
    """Phase B compiled on the remote against the offline-accepted task rig (compiler checks every camera)."""
    from ops.jobs import rwprep_remote
    from ops.jobs.pnpv2_registration import STAGE_IDS, rig_evidence

    p = values(spec)
    stage = rig_evidence(spec)
    if stage["status"] != "done":
        raise ValueError(f"task rig not accepted: {stage['detail']}")
    base = {**_phase_a_config(spec), **{k: v for k, v in json.loads(
        (Path(p["calib"]) / "dispatcher.json").read_text()).items() if k in ("code", "commit")}}
    out, link = _attachment(p["phase_b"], PHASE_B_BUNDLE)
    remote = f"{REMOTE_RW}/configs/{PHASE_B_BUNDLE}"
    client = rwprep_remote.RemoteClient()
    print(rwprep_remote.planner(client, base, [
        "-c", _PHASE_B_PLAN, base["remote"], remote, p["rig"], STAGE_IDS["phaseB"],
        f"{p['remote_processed']}/calibration/{TASK}", str(PREP_DISK_GIB),
        "acceptance_override" if stage.get("acceptance_override") else "automatic"]))
    print(json.dumps(rwprep_remote.attach(remote, out, client, base)))
    _link(out, link)


def geometry_remote(spec, a, argv):
    """Stage the remote geometry publisher; no tick, launch, registry or controller changes."""
    from ops.jobs import pnpv2_geometry as geometry, rwprep_remote
    from ops.jobs.pnpv2_registration import STAGE_IDS, dispatcher_evidence, rig_evidence
    from s4d.rwlearn.pnpv2_publication import manifest_check
    from s4d.rwlearn.pnpv2_offline import no_override

    sub = argparse.ArgumentParser(prog="geometry-remote")
    sub.add_argument("--source", type=Path, default=Path("/home/ws/ws/realworld-ops/ingest/manifest.json"))
    b = sub.parse_args(argv)
    p = values(spec)
    _check_data(spec, a)
    m = manifest_check(_remote_raw_verified(spec), spec)
    ids = {r["id"] for r in m["episodes"]}
    phases = {}
    for key, stage in (("phase_a", "phaseA"), ("phase_b", "phaseB")):
        evidence = dispatcher_evidence(p[key], ids, STAGE_IDS[stage])
        if evidence["status"] != "done":
            raise ValueError(f"{stage} not complete: {evidence['detail']}")
        phases[key] = json.loads((Path(p[key]) / "dispatcher.json").read_text())
    rig = rig_evidence(spec)
    if rig["status"] != "done":
        raise ValueError(f"task rig not accepted: {rig['detail']}")
    out, link = _attachment(Path(p["work"]) / "geometry-remote", geometry.BUNDLE)
    rwprep_remote_release(spec)
    source_bytes = b.source.read_bytes()
    source = json.loads(source_bytes)
    names = {r["filename"] for r in m["episodes"]}
    # Freeze only this collection in the new snapshot; never transfer v1/drawer state into it.
    selected = {k: e for k, e in source["episodes"].items() if e.get("task") == TASK
                and Path((e.get("source") or {}).get("path") or e.get("filename") or k).name in names}
    frozen = {"schema": "rw-ingest-manifest", "schema_version": 2, "episodes": selected}
    no_override(frozen)
    client = rwprep_remote.RemoteClient()
    remote_inputs, phase_ids = {}, {}
    for key, config in phases.items():
        remote = config["remote"]
        expected = PHASE_A_BUNDLE if key == "phase_a" else PHASE_B_BUNDLE
        if remote != f"{REMOTE_RW}/configs/{expected}":
            raise ValueError("geometry requires the canonical pnpv2 Phase-A/B bundles")
        jobs = json.loads((Path(p[key]) / "jobs.json").read_text())
        if TASK == 'ball-hiding':
            from s4d.rwlearn.ball_hiding_membership import eligible_jobs
            jobs = eligible_jobs(jobs)
        phase_ids[remote] = [j["id"] for j in jobs]
        for name in ("dispatcher.json", "jobs.json"):
            path = f"{remote}/{name}"
            remote_inputs[path] = client._remote_checksum(path)
    for path in (p["rig"], str(Path(p["rig"]).with_suffix(".reference.npz"))):
        remote_inputs[path] = client._remote_checksum(path)
    from s4d.rwlearn.pnpv2_arrival import sha256
    if remote_inputs[p["rig"]] != sha256(p["rig_local"]) or remote_inputs[
            str(Path(p["rig"]).with_suffix(".reference.npz"))] != sha256(Path(p["rig_local"]).with_suffix(".reference.npz")):
        raise ValueError("remote rig differs from the accepted local mirror")
    import hashlib
    out.mkdir(parents=True)
    geometry.json_once(out / "manifest.json", m)
    geometry.json_once(out / "source-ingest.json", frozen)
    geometry.json_once(out / "path-provenance.json", dict(source=str(b.source),
                       source_sha256=hashlib.sha256(source_bytes).hexdigest(), manifest_sha256=sha256(p["manifest"])))
    job = geometry.remote_job(spec, out, phases["phase_a"]["remote"], phases["phase_b"]["remote"],
                              phase_ids, remote_inputs)
    # stage() defaults to this checkout's HEAD. Phase A's commit predates the geometry worker.
    result = rwprep_remote.stage([job], out, client, release=rwprep_remote_release(spec), name=geometry.BUNDLE,
                                 image=phases["phase_a"]["image"],
                                 files=("manifest.json", "source-ingest.json", "path-provenance.json"))
    _link(out, link)
    print(json.dumps(result))


def geometry_fetch(spec, a, argv):
    from ops.jobs.pnpv2_geometry import fetch
    print(json.dumps(fetch(spec)))


ENTRIES = {"manifest": manifest, "rig-map": rig_map, "publish-rgb": publish_rgb, "raw-proof": raw_proof,
           "snapshot": snapshot, "publish": publish, "upload": upload,
           "phase-a-remote": phase_a_remote, "rgb-remote": rgb_remote, "rgb-fetch": rgb_fetch,
           "calib-remote": calib_remote, "rig-fetch": rig_fetch, "phase-b-remote": phase_b_remote,
           "geometry-remote": geometry_remote, "geometry-fetch": geometry_fetch}


def main(argv=None, *, cpu_only=False):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--entry", choices=sorted(ENTRIES), required=True)
    parser.add_argument("--task-spec", type=Path, default=Path(SPEC))
    parser.add_argument("--raw-root", type=Path, help="override <roots.raw> (layout <raw>/pick-and-place-v2/success)")
    parser.add_argument("--settle-seconds", type=int, help="stable-inventory window when there is no VERIFIED marker")
    parser.add_argument("args", nargs=argparse.REMAINDER)
    a = parser.parse_args(argv)
    spec = load_spec(a.task_spec)
    doc = read_arrival(spec)
    if doc is not None:
        spec = load_spec(a.task_spec, overlay=doc)
    forwarded = a.args[1:] if a.args[:1] == ["--"] else a.args
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _bootstrap import guard_gpus

    if not cpu_only:
        guard_gpus()
    from s4d.rwlearn.pnpv2_runtime import install

    install()
    ENTRIES[a.entry](spec, a, forwarded)


if __name__ == "__main__":
    main()
