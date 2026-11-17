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


def publish_rgb(spec, manifest_path, phase_a, *, execute=False, inspect=None):
    from s4d.rwlearn.dsreset import inspect_raw
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
    from ops.inference_server.contract_checks import singleview_family
    from s4d.realworld import io as rio
    from s4d.realworld.pipeline import camera_views, load_meta
    from s4d.realworld.rgb_train_store import write_episode
    from s4d.realworld.train_store import CompactStore
    from s4d.rwlearn import dsreset_gate as gate
    from s4d.rwlearn.store import sync_index

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
        evidence[ep] = dict(singleview=sv, m5=sets.counts, raw_validation=(inspect or inspect_raw)(r["path"]))
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
    # The published descriptor names the canonical remote copies (upload keeps the same relative layout).
    published = [f"{canonical}/episodes/{Path(x).name}" for x in paths]
    (out / "store.yaml").write_text(yaml.safe_dump(dict(kind="compact", paths=published)))
    (out / "store-local.yaml").write_text(yaml.safe_dump(dict(kind="compact", paths=paths)))
    return {**preview, "prepared": True}
