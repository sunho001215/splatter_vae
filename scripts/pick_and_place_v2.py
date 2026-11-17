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

from s4d.rwlearn.pnpv2_spec import SPEC, TASK, load_spec, read_arrival, values  # noqa: E402


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
    print(json.dumps(publish(spec, p["manifest"], p["phase_a"], execute=b.execute), indent=2))


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

    module.read_manifest = read_manifest
    sys.argv = [module.__file__, *argv]
    module.main()


def publish(spec, a, argv):
    """Geometry bundle: the established learning_inputs view of the NEW snapshot; refuses acceptance overrides."""
    from s4d.rwlearn.dsreset import learning_inputs
    from s4d.rwlearn.pnpv2_publication import manifest_check, no_override
    from s4d.rwlearn.store import open_store

    sub = argparse.ArgumentParser(prog="publish")
    sub.add_argument("--ingest-manifest", type=Path, required=True)
    b = sub.parse_args(argv)
    p = values(spec)
    m = manifest_check(json.loads(Path(p["manifest"]).read_text()), spec)
    ingest = json.loads(b.ingest_manifest.read_text())
    no_override(ingest)
    paths = []
    for r in m["episodes"]:
        matches = [e for e in ingest["episodes"].values() if e.get("task") == TASK and e.get("sha256") == r["sha256"]]
        if len(matches) != 1:
            raise ValueError(f"{r['id']}: no unique snapshot record")
        paths.append(matches[0]["preprocessing"]["stages"]["train_data"]["outputs"])
    descriptor = {"kind": "compact", "paths": paths}
    docs = learning_inputs(m, ingest, open_store(descriptor))
    no_override(docs["gate_report.json"])
    out = Path(p["geometry_dir_local"])
    out.mkdir(parents=True, exist_ok=False)
    (out / "store.yaml").write_text(yaml.safe_dump(descriptor, sort_keys=False))
    for name, doc in docs.items():
        (out / name).write_text(json.dumps(doc, indent=2) + "\n")
    print(json.dumps({"out": str(out), "episodes": len(paths), "launched": False}))


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
    client = RemoteClient()
    sent = {}
    for f in files:
        target = f"{remote}/{f.relative_to(local)}"
        digest = sha256(f)
        existing = client._remote_checksum(target, allow_missing=True)
        if existing not in (None, digest):
            raise ValueError(f"refusing to overwrite a different remote file: {target}")
        if existing is None:
            client.rsync_upload(f, target)
        sent[target] = digest
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
    if any(proofs.get(r["sha256"]) != r["sha256"] for r in m["episodes"]) or len(proofs) != len(m["episodes"]):
        raise ValueError("remote raw proof differs from the canonical manifest")
    return m


def phase_a_remote(spec, a, argv):
    """Phase A (ingest, native depth, FoundationStereo, SAM3) on the remote prep route; launches nothing."""
    from ops.jobs import rwprep_remote
    from ops.jobs.pnpv2_registration import STAGE_IDS

    p = values(spec)
    _check_data(spec, a)
    _remote_raw_verified(spec)
    out = Path(p["phase_a"])
    if out.exists():
        raise ValueError(f"{out} exists; the Phase-A bundle is immutable")
    result = rwprep_remote.prepare([Path(p["prep_plan"])], out, rwprep_remote.RemoteClient(),
                                   release=rwprep_remote_release(spec), stem=STAGE_IDS["phaseA"],
                                   name=PHASE_A_BUNDLE)
    print(json.dumps(result))


PHASE_A_BUNDLE = "pnpv2-20261117-phaseA"


def rwprep_remote_release(spec):
    release = Path(values(spec)["remote_processed"]).name
    if release != "pnpv2-20261117":
        raise ValueError("pick-and-place-v2 writes only the pnpv2-20261117 processed release")
    return release


def rgb_remote(spec, a, argv):
    """One-job remote RGB publication bundle; admitted by its tick only after every Phase-A job is done."""
    import shutil

    from ops.jobs import pnpv2_rgb, rwprep_remote

    p = values(spec)
    _check_data(spec, a)
    config = json.loads((Path(p["phase_a"]) / "dispatcher.json").read_text())
    jobs = json.loads(Path(config["jobs"]).read_text())
    out = Path(p["work"]) / "rgb-remote"
    if out.exists():
        raise ValueError(f"{out} exists; the RGB bundle is immutable")
    out.mkdir(parents=True)
    shutil.copyfile(p["manifest"], out / "manifest.json")
    job = pnpv2_rgb.remote_job(spec, out, config["remote"], [j["id"] for j in jobs])
    result = rwprep_remote.stage([job], out, rwprep_remote.RemoteClient(), commit=config["commit"],
                                 release=rwprep_remote_release(spec), name=pnpv2_rgb.RGB_BUNDLE,
                                 image=config["image"], files=("manifest.json",))
    print(json.dumps(result))


def rgb_fetch(spec, a, argv):
    """Copy the finished remote RGB bundle documents into the local staging directory (never the episodes)."""
    import shlex

    from ops.jobs import pnpv2_rgb
    from ops.remote.access import RemoteClient
    from s4d.rwlearn.pnpv2_publication import manifest_check

    p = values(spec)
    state = json.loads((Path(p["work"]) / "rgb-remote/state/state.json").read_text())["jobs"]
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
    if store.get("kind") != "compact" or len(store.get("paths", [])) != 100 or not all(
            x.startswith(remote + "/episodes/") for x in store["paths"]):
        raise ValueError("remote RGB store.yaml must name the 100 remote published episodes")
    local.mkdir(parents=True)
    for name, text in docs.items():
        (local / name).write_text(text)
    print(json.dumps({"out": str(local), "files": sorted(docs), "remote": remote}))


REMOTE_RW = "/home/compu/kaist/sunho/realworld"
CALIB_BUNDLE = "pnpv2-20261117-calib"
PHASE_B_BUNDLE = "pnpv2-20261117-phaseB"
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
base, out, rig, stem, extra = sys.argv[1:6]
base, out = pathlib.Path(base), pathlib.Path(out)
c = json.loads((base / 'dispatcher.json').read_text())
rows = json.loads((base / 'prep-plan.json').read_text())
rigmap, doc = out.parent / (out.name + '.rig-map.json'), {r['episode']: rig for r in rows}
if rigmap.exists():
    assert json.loads(rigmap.read_text()) == doc, 'HELD: rig map differs'
else:
    atomic(rigmap, doc)
if not out.exists():
    args = [sys.executable, 'scripts/realworld_prep_register.py', '--phase', 'B', '--id-stem', stem,
            '--plan', str(base / 'prep-plan.json'), '--out', str(out), '--host', 'remote',
            '--code', '/opt/src/splatter4d', '--rig-map', str(rigmap), '--extra-readonly', extra]
    for k in ('processed', 'raw', 'weights', 'image', 'tools', 'assets'):
        args += ['--' + k, c[k]]
    subprocess.run(args, check=True)
print(json.dumps({'jobs': len(json.loads((out / 'jobs.json').read_text()))}))
"""


def _phase_a_config(spec):
    return json.loads((Path(values(spec)["phase_a"]) / "dispatcher.json").read_text())


def calib_remote(spec, a, argv):
    """Task LoMa-G rig on the remote route (established planner and automatic G1-G6 publication; no override)."""
    from ops.jobs import rwprep_remote
    from ops.jobs.pnpv2_registration import STAGE_IDS

    p = values(spec)
    base = _phase_a_config(spec)
    out = Path(p["calib"])
    if out.exists():
        raise ValueError(f"{out} exists; the calibration attachment is immutable")
    remote = f"{REMOTE_RW}/configs/{CALIB_BUNDLE}"
    client = rwprep_remote.RemoteClient()
    print(rwprep_remote.planner(client, base, [
        "-c", _CALIB_PLAN, f"{base['remote']}/dispatcher.json", TASK, remote,
        f"{p['remote_processed']}/calibration/{TASK}", CALIB_UPSTREAM, STAGE_IDS["rig"],
        json.dumps([*CALIB_READONLY])]))
    print(json.dumps(rwprep_remote.attach(remote, out, client, base)))


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
    for remote, target in ((p["rig"], local), (str(Path(p["rig"]).with_suffix(".reference.npz")),
                                               local.with_suffix(".reference.npz"))):
        client.rsync_download(remote, target)
    from ops.jobs.pnpv2_registration import rig_evidence

    print(json.dumps({k: v for k, v in rig_evidence(spec).items() if k != "inputs"}))


def phase_b_remote(spec, a, argv):
    """Phase B compiled on the remote against the automatically accepted task rig (compiler checks every camera)."""
    from ops.jobs import rwprep_remote
    from ops.jobs.pnpv2_registration import STAGE_IDS, rig_evidence

    p = values(spec)
    stage = rig_evidence(spec)
    if stage["status"] != "done":
        raise ValueError(f"task rig not accepted: {stage['detail']}")
    base = _phase_a_config(spec)
    out = Path(p["phase_b"])
    if out.exists():
        raise ValueError(f"{out} exists; the Phase-B attachment is immutable")
    remote = f"{REMOTE_RW}/configs/{PHASE_B_BUNDLE}"
    client = rwprep_remote.RemoteClient()
    print(rwprep_remote.planner(client, base, [
        "-c", _PHASE_B_PLAN, base["remote"], remote, p["rig"], STAGE_IDS["phaseB"],
        f"{p['remote_processed']}/calibration/{TASK}"]))
    print(json.dumps(rwprep_remote.attach(remote, out, client, base)))


ENTRIES = {"manifest": manifest, "rig-map": rig_map, "publish-rgb": publish_rgb, "raw-proof": raw_proof,
           "snapshot": snapshot, "publish": publish, "upload": upload,
           "phase-a-remote": phase_a_remote, "rgb-remote": rgb_remote, "rgb-fetch": rgb_fetch,
           "calib-remote": calib_remote, "rig-fetch": rig_fetch, "phase-b-remote": phase_b_remote}


def main(argv=None):
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

    guard_gpus()
    from s4d.rwlearn.pnpv2_runtime import install

    install()
    ENTRIES[a.entry](spec, a, forwarded)


if __name__ == "__main__":
    main()
