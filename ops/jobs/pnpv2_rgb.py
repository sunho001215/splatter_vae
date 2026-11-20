"""Calibration-independent pick-and-place-v2 RGB publication: the drawer-open-v2 RGB route (rgb_inputs.py) for v2.

Inputs are the canonical 100-success manifest and the completed Phase-A dispatcher (normalized episodes). Each
episode is exported RGB-only (``rgb_train_store.write_episode(rgb_only=True)``), then checked with the single-view
contract and M5 multiview-set synchronisation; the gate report passes only the non-geometric consumers (DynaMo,
ReViWo, E2E CNN) and carries a failed calibration hold, so no geometry consumer can train on it. Default is a
read-only preview; ``execute`` writes a NEW staging directory only (never raw, processed or remote).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from s4d.rwlearn.pnpv2_spec import TASK, values

STAGE_COMPLETE = "normalized/_COMPLETE.json"


def require(ok, why):
    if not ok:
        raise ValueError(why)


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def sources(spec, manifest, phase_a):
    """{episode: Phase-A output root}, requiring the dispatcher's done state and the normalized completion record."""
    from ops.jobs.pnpv2_registration import STAGE_IDS, dispatcher_evidence

    stage = dispatcher_evidence(phase_a, {r["id"] for r in manifest["episodes"]}, STAGE_IDS["phaseA"])
    require(stage["status"] == "done", f"Phase A not complete: {stage['detail']}")
    roots = {}
    for job in json.loads((Path(phase_a) / "jobs.json").read_text()):
        cfg = yaml.safe_load(Path(next(p for p in job["inputs"] if p.endswith(".yaml"))).read_text())
        root = Path(cfg["output_root"])
        require((root / "episodes" / job["episode"] / STAGE_COMPLETE).is_file(),
                f"{job['episode']}: normalized stage incomplete")
        roots[job["episode"]] = root
    return roots


def local_report(manifest, store, evidence, provenance):
    """Gate exactly the canonical S set after raw, RGB/action and M5 validation (rgb_inputs.local_report for v2)."""
    from s4d.rwlearn import dsreset_gate as gate
    from s4d.rwlearn.dsreset_identity import raw_identity, store_identities

    ids = [r["id"] for r in manifest["episodes"]]
    n = len(ids)
    require(set(evidence) == set(ids), "missing per-episode RGB QA")
    require(all(evidence[e]["singleview"]["passed"] is True and evidence[e]["m5"]["accepted"] > 0
                and evidence[e].get("raw_validation") for e in ids), "raw/RGB/action/M5 gate failed")
    rgb_families = ("singleview", "multiview_sets")
    families = {f: dict(passed=f in rgb_families, failed_count=0 if f in rgb_families else n,
                        failed_episodes=[] if f in rgb_families else ids) for f in gate.REPORT_FAMILIES}
    identities = {r["id"]: {**raw_identity(TASK, r["path"], r["sha256"]), "store_episode": None}
                  for r in manifest["episodes"]}
    report = dict(
        schema="pick-and-place-v2-rgb-gate/1", passed=True, geometry_passed=False,
        verification_source="VERIFIED/stable raw + rehashed canonical raw + completed Phase A + CPU contracts",
        required_stages=["normalized", "rgb_train_data", "singleview_contract", "m5_sets"],
        tasks={TASK: dict(passed=True, target=n, valid_and_preprocessed=n, episodes=ids, store_episodes=[],
                          episode_identities=identities, excluded=[], problems=[], problem_count=0)},
        families={TASK: families},
        calibration_hold={TASK: dict(passed=False, validated=0, episodes={},
                                     problems=["RGB publication carries no accepted camera geometry"],
                                     problem_count=1)},
        rgb_only_projection=provenance)
    return gate.bind_store_episodes(report, {TASK: store.episodes()}, store_identities(store))


def build(m, roots, out, published, *, raw_path=lambda r: r["path"], inspect=None, preview=None):
    """Export, check and gate the RGB-only bundle of ``m`` into the NEW directory ``out``.

    ``roots`` maps each episode to its completed Phase-A output root; ``published`` is the directory whose
    ``episodes/<ep>`` the published ``store.yaml`` names (same relative layout as ``out``); ``raw_path(row)`` is where
    this host reads the canonical recording (identities keep the manifest path).
    """
    from ops.inference_server.contract_checks import singleview_family
    from s4d.realworld import io as rio
    from s4d.realworld.pipeline import camera_views, load_meta
    from s4d.realworld.rgb_train_store import write_episode
    from s4d.realworld.train_store import CompactStore
    from s4d.rwlearn import dsreset_gate as gate
    from s4d.rwlearn.dsreset import inspect_raw
    from s4d.rwlearn.store import sync_index

    out = Path(out)
    require(set(roots) == {r["id"] for r in m["episodes"]}, "Phase-A roots differ from the manifest")
    out.mkdir(parents=True, exist_ok=False)
    paths, evidence = [], {}
    for r in m["episodes"]:
        ep = r["id"]
        dest = out / "episodes" / ep
        write_episode(roots[ep], ep, dest, task=TASK, semantic=False, rgb_only=True, max_sync_ns=20_000_000)
        store = CompactStore([dest])
        key = store.episodes()[0]
        expected = sorted(v.rig_id for v in camera_views(load_meta(roots[ep], ep)))
        sv = singleview_family(store, key, expected_cameras=expected)
        sets = sync_index(store, key, geometry=False)
        require(sv["passed"] and len(sets.times) > 0, ep + ": RGB/action/M5 contracts failed")
        store.close()
        paths.append(str(dest))
        evidence[ep] = dict(singleview=sv, m5=sets.counts, raw_validation=(inspect or inspect_raw)(raw_path(r)))
    store = CompactStore(paths)
    report = local_report(m, store, evidence, preview)
    require(all(gate.task_passed(report, TASK, c) for c in ("reviwo", "dynamo", "e2e_cnn")), "RGB gate failed")
    require(not any(gate.task_passed(report, TASK, c) for c in ("ours", "sincro", "mvp3d", "snerl")),
            "geometry escaped the RGB publication")
    report["dataset_manifest_sha256"] = hashlib.sha256(json.dumps(m, sort_keys=True).encode()).hexdigest()
    keys = {store.raw_sha256(ep): ep for ep in store.episodes()}
    docs = {"manifest.json": m, "gate_report.json": report, "contracts.json": evidence,
            "split_seed0.json": {s: [keys[r["sha256"]] for r in m["episodes"] if r["split"] == s]
                                 for s in ("train", "validation")},
            "outcomes.json": {ep: True for ep in store.episodes()}}
    store.close()
    for name, doc in docs.items():
        rio.atomic_write_json(out / name, doc)
    # The published descriptor names the canonical copies (same relative layout as ``out``).
    (out / "store.yaml").write_text(yaml.safe_dump(dict(
        kind="compact", paths=[f"{published}/episodes/{Path(x).name}" for x in paths])))
    return paths


def publish_rgb(spec, manifest_path, phase_a, *, execute=False, inspect=None):
    from s4d.rwlearn.pnpv2_publication import manifest_check

    p = values(spec)
    out = Path(p["rgb_dir_local"])
    canonical = p["rgb_dir"]
    m = manifest_check(json.loads(Path(manifest_path).read_text()), spec)
    require(not out.exists(), "RGB staging exists; publications are immutable (inspect it, use a new work root)")
    for raw_root in (p["raw_task"], p["processed"], p["remote_processed"]):
        require(not out.resolve().is_relative_to(Path(raw_root)), "RGB publication must stay outside raw/processed")
    for r in m["episodes"]:
        require(_sha(r["path"]) == r["sha256"], f"{r['id']}: raw sha256 drift")
    roots = sources(spec, m, phase_a)
    preview = dict(schema="pick-and-place-v2-rgb-inputs/1", prepared=False, episodes=len(roots),
                   manifest_sha256=_sha(manifest_path), phase_a=str(phase_a), out=str(out), canonical=canonical)
    if not execute:
        return preview
    paths = build(m, roots, out, canonical, inspect=inspect, preview=preview)
    (out / "store-local.yaml").write_text(yaml.safe_dump(dict(kind="compact", paths=paths)))
    return {**preview, "prepared": True}


# ------------------------------------------------------------------------------------ remote Phase-A route
# With Phase A on the remote (ops/jobs/rwprep_remote, release pnpv2-20261117) the normalized outputs exist only there,
# so the same export runs as a one-job remote bundle inside the prep image; only the small bundle documents come back.
REMOTE_RW = "/home/compu/kaist/sunho/realworld"
REMOTE_RAW = "/home/compu/kaist/sunho/data/real_world"
RGB_BUNDLE = "pnpv2-20261117-rgb"
RGB_JOB = "pnpv2-20261117-rgbpub-pick-and-place-v2-S"
DOCS = ("store.yaml", "manifest.json", "split_seed0.json", "gate_report.json", "outcomes.json", "contracts.json")


def remote_out(spec):
    """Remote RGB publication directory inside the writable pnpv2 processed release (store paths stay under it)."""
    return f"{values(spec)['remote_processed']}/learning-inputs/{TASK}/S/rgb"


def remote_raw(row, raw_root=REMOTE_RAW):
    """The remote copy of a canonical recording: same suffix after ``/real_world/`` (compile_plans rule)."""
    return f"{raw_root}/{str(row['path']).split('/real_world/', 1)[1]}"


def phase_a_roots(bundle, episodes):
    """{episode: output_root} of a remote Phase-A bundle whose (mirrored) state marks every job done."""
    bundle = Path(bundle)
    jobs = json.loads((bundle / "jobs.json").read_text())
    state = json.loads((bundle / "state" / "state.json").read_text())["jobs"]
    require({j["episode"] for j in jobs} == set(episodes), "Phase-A bundle episodes differ from the manifest")
    require(all(j["id"].startswith("pnpv2-20261117-prepA-") for j in jobs), "Phase-A jobs must use the v2 stem")
    require(all((state.get(j["id"]) or {}).get("status") == "done" for j in jobs), "Phase A not complete")
    roots = {}
    for job in jobs:
        cfg = yaml.safe_load(Path(next(p for p in job["inputs"] if p.endswith(".yaml"))).read_text())
        root = Path(cfg["output_root"])
        require((root / "episodes" / job["episode"] / STAGE_COMPLETE).is_file(),
                f"{job['episode']}: normalized stage incomplete")
        roots[job["episode"]] = root
    return roots


def remote_build(manifest_path, phase_a, out, raw_root=REMOTE_RAW):
    """Worker entry (inside the prep image on the remote). Builds into a private staging directory and renames it
    into ``out`` only after every check passed; an existing ``out`` is never modified."""
    import os
    import time

    from s4d.rwlearn.pnpv2_publication import manifest_check
    from s4d.rwlearn.pnpv2_spec import SPEC, load_spec

    spec = load_spec(Path(__file__).resolve().parents[2] / SPEC)
    out = Path(out)
    require(str(out) == remote_out(spec), "remote RGB publication must use the canonical release directory")
    m = manifest_check(json.loads(Path(manifest_path).read_text()), spec)
    if out.exists():
        require(all((out / n).is_file() for n in DOCS), f"{out} exists without a complete bundle; inspect it")
        return {"out": str(out), "reused": True}
    for r in m["episodes"]:
        require(_sha(remote_raw(r, raw_root)) == r["sha256"], f"{r['id']}: remote raw copy missing or different")
    roots = phase_a_roots(phase_a, [r["id"] for r in m["episodes"]])
    staging = out.parent / f".{out.name}.staging-{os.getpid()}-{time.time_ns()}"
    preview = dict(schema="pick-and-place-v2-rgb-inputs/1", prepared=True, episodes=len(roots),
                   manifest_sha256=_sha(manifest_path), phase_a=str(phase_a), out=str(out), canonical=str(out),
                   route="remote-phase-a/1")
    build(m, roots, staging, out, raw_path=lambda r: remote_raw(r, raw_root), preview=preview)
    os.rename(staging, out)  # same filesystem; fails if another attempt published first
    return {"out": str(out), "episodes": len(roots), "reused": False}


def remote_job(spec, bundle_dir, phase_a_remote, phase_a_ids):
    """The one-job remote bundle definition (compiled locally, staged by rwprep_remote.stage)."""
    bundle_dir = Path(bundle_dir)
    manifest = bundle_dir / "manifest.json"
    remote = f"{REMOTE_RW}/configs/{RGB_BUNDLE}"
    return dict(
        id=RGB_JOB, priority=-4, gpu_profile="1g.24gb", gpu_gib=2, ram_gib=32, cpus=4, disk_gib=32,
        inputs={str(manifest): _sha(manifest)}, deps=[],
        external_states={f"{phase_a_remote}/state/state.json": list(phase_a_ids)},
        steps=[{"kind": "python", "argv": ["-m", "ops.jobs.pnpv2_rgb", "remote", "--manifest",
                                           f"{remote}/manifest.json", "--phase-a", phase_a_remote,
                                           "--out", remote_out(spec)]}])


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="pick-and-place-v2 RGB publication worker (remote prep image)")
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("remote")
    r.add_argument("--manifest", type=Path, required=True)
    r.add_argument("--phase-a", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--raw-root", default=REMOTE_RAW)
    a = parser.parse_args(argv)
    print(json.dumps(remote_build(a.manifest, a.phase_a, a.out, a.raw_root)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
