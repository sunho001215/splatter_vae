"""pick-and-place-v2 upstream producers (operator steps between arrival and queue registration).

Entries (each refuses before the dataset is complete/validated; none launches a job or contacts the Robot PC):

  manifest     canonical S manifest + per-episode prep plan            -> <work>/ingest
  rig-map      Phase-B rig map, only once the task rig is accepted      -> <work>/phaseB-rig-map.json
  publish-rgb  calibration-independent RGB bundle from Phase A          -> <work>/publication/S/rgb
  raw-proof    read-only remote sha256 of the raw copies (snapshot)     -> <work>/remote-raw-hashes.json
  snapshot     NEW ingest snapshot from Phase A/B dispatcher records    -> <work>/ingest-snapshot
  publish      geometry bundle from the snapshot (calibration records)  -> <work>/publication/S/geometry
  upload       checked no-overwrite upload of one staged bundle + receipt (remote write; coordinator only)

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
    from s4d.realworld.io import atomic_write_json

    path = Path(path)
    if path.exists():
        raise ValueError(f"{path} exists; outputs are immutable")
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, doc)


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


ENTRIES = {"manifest": manifest, "rig-map": rig_map, "publish-rgb": publish_rgb, "raw-proof": raw_proof,
           "snapshot": snapshot, "publish": publish, "upload": upload}


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
