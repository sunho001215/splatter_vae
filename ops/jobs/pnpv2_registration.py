"""pick-and-place-v2 registration: evidence-gated waves. Dry run by default; --execute from the adopted main only.

Nothing is registered before the dataset has arrived and ``check-data`` passes on the actual recordings. Each queue
wave is registered once its upstream evidence exists (the established "publish verified inputs before queue
registration" rule), so no barrier row can run before its inputs exist:

* wave ``rgb`` (DynaMo, ReViWo, E2E CNN, E2E CNN noaug) needs check-data + ingest + Phase A + the RGB publication;
  it never waits for the rig, Phase B or the geometry publication.
* wave ``geometry`` (Ours orig/ch01, SinCro, SNeRL, 3D-MVP) additionally needs the accepted task rig, all 100 Phase-B
  jobs, the geometry publication (per-episode calibration_validated records) and the train-only scene measurement.

Publication follows the reviewed registrars (register_dsreset_20261113.py / register_drawer_rgb_20261115.py): the
live queue, registry and inputs are hashed in the dry run and re-verified under stage2-driver.lock, remote-driver.lock
and registry.lock; an exact-ID placement pause is written before the queue append; a pending journal and a proof
record the transaction. Nothing is launched or placed.
"""

from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import json
import re
from contextlib import contextmanager
from pathlib import Path

import yaml

from ops.jobs.pnpv2_plan import STAGES, build_rows, counts
from s4d.rwlearn.pnpv2_spec import (
    BUNDLE_FILES,
    GEOMETRY_METHODS,
    PAUSE,
    RGB_METHODS,
    SPEC,
    STEM,
    TASK,
    V1_PAUSE,
    WRAPPERS,
    dispatcher_stem,
    leftover_tokens,
    row_id,
    values,
)

ROOT = Path(__file__).resolve().parents[2]
MAIN = Path("/home/ws/ws/splatter4d")
QUEUE, REGISTRY, PAUSES = "experiments/queue.yaml", "experiments/registry.jsonl", "runs/setup/admission-pauses.json"
LOCKS = ("runs/setup/stage2-driver.lock", "runs/setup/remote-driver.lock", "experiments/registry.lock")
RGB_CONSUMERS = ("dynamo", "reviwo", "e2e_cnn")
GEOMETRY_CONSUMERS = ("ours", "sincro", "snerl", "mvp3d")
STAGE_IDS = {"phaseA": dispatcher_stem("prepA"), "phaseB": dispatcher_stem("prepB"),
             "rig": dispatcher_stem("calib")}


class Refused(ValueError):
    pass


def require(ok, why):
    if not ok:
        raise Refused(why)


_REVIEWED = []


def _reviewed():
    """The reviewed queue/registry helpers every real-world registrar uses (text append + parsed-equality proof)."""
    if not _REVIEWED:
        path = ROOT / "runs/setup/tools/register_ours_realworld_20261111.py"
        spec = importlib.util.spec_from_file_location("pnpv2_registration_common", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _REVIEWED.append(module)
    return _REVIEWED[0]


def require_main(repo):
    require(Path(repo).resolve() == MAIN and ROOT == MAIN, "only the adopted main checkout may register (--execute)")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def json_sha(value):
    return sha(json.dumps(value, sort_keys=True).encode())


# --------------------------------------------------------------------------------------------- stage evidence
def _stage(status, detail="", **extra):
    return dict(status=status, detail=detail, **extra)


def _manifest_ids(manifest):
    return {r["id"] for r in manifest["episodes"]}


def ingest_evidence(spec, data_report):
    p = values(spec)
    path = Path(p["manifest"])
    if not path.is_file():
        return _stage("pending", "canonical manifest bundle not built")
    from s4d.rwlearn.pnpv2_publication import manifest_check

    m = json.loads(path.read_text())
    try:
        manifest_check(m, spec)
    except ValueError as exc:
        return _stage("blocked", f"ingest manifest invalid: {exc}")
    if data_report is not None and data_report.get("manifest") is not None:
        live = data_report["manifest"]
        if json_sha(m) != json_sha(live):
            return _stage("blocked", "ingest manifest differs from check-data on the arrived recordings")
    plan = Path(p["prep_plan"])
    rows = json.loads(plan.read_text()) if plan.is_file() else []
    if {r.get("episode") for r in rows} != _manifest_ids(m) or any("config" not in r for r in rows):
        return _stage("blocked", "prep-plan.json must hold one fresh config per manifest episode (no reuse)")
    return _stage("done", f"{len(rows)} episodes", inputs={str(path): file_sha(path), str(plan): file_sha(plan)},
                  manifest=m)


def dispatcher_evidence(bundle, episodes, stem):
    bundle = Path(bundle)
    jobs_path, state_path = bundle / "jobs.json", bundle / "state/state.json"
    if not jobs_path.is_file():
        return _stage("pending", f"dispatcher bundle {bundle.name} not compiled")
    jobs = json.loads(jobs_path.read_text())
    ids = [j["id"] for j in jobs]
    if any(not i.startswith(stem + "-") for i in ids):
        return _stage("blocked", f"dispatcher job ids must use the {stem} stem (never a v1 prefix)")
    if {j.get("episode") for j in jobs} != set(episodes):
        return _stage("blocked", "dispatcher episodes differ from the canonical manifest")
    state = json.loads(state_path.read_text()).get("jobs", {}) if state_path.is_file() else {}
    status = {i: (state.get(i) or {}).get("status", "pending") for i in ids}
    done = sum(v == "done" for v in status.values())
    failed = sorted(i for i, v in status.items() if v in ("failed", "unknown"))
    if done == len(ids):
        return _stage("done", f"{done}/{len(ids)} done",
                      inputs={str(jobs_path): file_sha(jobs_path), str(state_path): file_sha(state_path)})
    return _stage("blocked" if failed else "pending", f"{done}/{len(ids)} done" + (f"; held {failed[:3]}" if failed
                                                                                      else ""))


def rig_evidence(spec):
    p = values(spec)
    rig = Path(p["rig_local"])
    reference = rig.with_suffix(".reference.npz")
    if not rig.is_file() or not reference.is_file():
        return _stage("pending", "accepted task rig not published")
    try:
        from s4d.realworld.calib.rig import load_rig, rig_camera_status

        document, _, _ = load_rig(rig)
        held = [c for c in document["cameras"] if rig_camera_status(document, c)[0] != "validated"]
    except (ValueError, KeyError, OSError) as exc:
        return _stage("blocked", f"rig unreadable: {exc}")
    if held or document.get("task") not in (None, TASK) or (document.get("acceptance_override") or None):
        return _stage("blocked", f"rig not automatically accepted for every camera: {held}")
    return _stage("done", f"{len(document['cameras'])} cameras validated",
                  inputs={str(rig): file_sha(rig), str(reference): file_sha(reference)})


def bundle_evidence(spec, view, manifest):
    """The locally staged five-file bundle plus its verified upload receipt; report-level gate per consumer."""
    from s4d.rwlearn import gate
    from s4d.rwlearn.pnpv2_publication import manifest_check, no_override

    p = values(spec)
    d = Path(p[f"{view}_dir_local"])
    names = [n for _, n in BUNDLE_FILES]
    if not all((d / n).is_file() for n in names):
        return _stage("pending", f"{view} publication not staged")
    blobs = {n: (d / n).read_bytes() for n in names}
    m = json.loads(blobs["manifest.json"])
    try:
        manifest_check(m, spec)
    except ValueError as exc:
        return _stage("blocked", f"{view} manifest invalid: {exc}")
    if json_sha(m) != json_sha(manifest):
        return _stage("blocked", f"{view} publication membership differs from the ingest manifest")
    report = json.loads(blobs["gate_report.json"])
    try:
        no_override(report)
    except ValueError as exc:
        return _stage("blocked", str(exc))
    if report.get("dataset_manifest_sha256") != json_sha(m):
        return _stage("blocked", f"{view} gate report is bound to another manifest")
    consumers = RGB_CONSUMERS if view == "rgb" else GEOMETRY_CONSUMERS
    failed = [c for c in consumers if not gate.task_passed(report, TASK, c)]
    if failed:
        return _stage("blocked", f"{view} gate refuses {failed}: {gate.task_problems(report, TASK, failed[0])[:2]}")
    if view == "rgb" and any(gate.task_passed(report, TASK, c) for c in GEOMETRY_CONSUMERS):
        return _stage("blocked", "RGB publication must not carry geometry acceptance")
    saved = report["tasks"][TASK]
    ids = {r["store_episode"]: r for r in saved["episode_identities"].values()}
    by_sha = {r["sha256"]: r for r in m["episodes"]}
    split, outcomes = json.loads(blobs["split_seed0.json"]), json.loads(blobs["outcomes.json"])
    want = {s: sorted(ep for ep, r in ids.items() if by_sha.get(r["raw_sha256"], {}).get("split") == s)
            for s in ("train", "validation")}
    if (len(ids) != 100 or set(ids) != set(saved["store_episodes"]) or {r["raw_sha256"] for r in ids.values()}
            != set(by_sha) or any(sorted(split.get(s, [])) != want[s] for s in want)
            or set(outcomes) != set(ids) or any(v is not True for v in outcomes.values())):
        return _stage("blocked", f"{view} split/outcomes/identities differ from the manifest")
    store = yaml.safe_load(blobs["store.yaml"])
    roots = (p["remote_processed"] + "/", p[f"{view}_dir"] + "/")
    if (store.get("kind") != "compact" or len(store.get("paths", [])) != 100
            or not all(isinstance(x, str) and x.startswith(roots) for x in store["paths"])):
        return _stage("blocked", f"{view} store.yaml must list 100 published compact episodes")
    receipt = d.parent / f"{view}.UPLOADED.json"
    local = {str(d / n): sha(b) for n, b in blobs.items()}
    if not receipt.is_file():
        return _stage("pending", f"{view} publication staged; upload receipt missing", inputs=local)
    sent = json.loads(receipt.read_text())
    remote = {f"{p[f'{view}_dir']}/{n}": sha(b) for n, b in blobs.items()}
    if sent.get("schema") != "pick-and-place-v2-upload/1" or any(sent.get("files", {}).get(k) != v
                                                                 for k, v in remote.items()):
        return _stage("blocked", f"{view} upload receipt differs from the staged bundle")
    return _stage("done", f"{len(ids)} episodes", inputs={**local, str(receipt): file_sha(receipt)},
                  split=split, outcomes=outcomes)


def scene_evidence(spec, manifest, rig_stage):
    p = values(spec)
    path = Path(p["scene_report"])
    if not path.is_file():
        return _stage("pending", "train-only native scene measurement missing")
    from s4d.rwlearn.dsreset_measure import validate_report

    report = json.loads(path.read_text())
    try:
        scene = validate_report(report, manifest)
    except ValueError as exc:
        return _stage("blocked", str(exc))
    rig = Path(p["rig_local"])
    if not rig.is_file() or report.get("native_provenance", {}).get("rig_document_sha256") != file_sha(rig):
        return _stage("blocked", "scene measurement rig differs from the accepted task rig")
    return _stage("done", "lookat/pose_scale/bounds measured", inputs={str(path): file_sha(path)}, scene=scene)


def evidence(spec, data_report, arrival_state=None):
    """Every upstream stage in order. Each stage reads only its own evidence; waves need all prerequisites."""
    out = {}
    arrival = arrival_state or (data_report or {}).get("arrival") or {}
    out["arrival"] = _stage("done" if arrival.get("state") == "complete" else "pending",
                            f"{arrival.get('state', 'not checked')} ({arrival.get('found', 0)}/100"
                            f"{', basis ' + arrival['basis'] if arrival.get('basis') else ''})")
    if data_report is None:
        out["check-data"] = _stage("pending", "not run here (scripts/pnpv2.py check-data hashes every recording)")
    else:
        out["check-data"] = _stage("done" if data_report.get("passed") else "blocked",
                                   "; ".join(data_report.get("problems", [])[:3]) or "passed")
    out["ingest"] = ingest_evidence(spec, data_report)
    manifest = out["ingest"].pop("manifest", None)
    eps = _manifest_ids(manifest) if manifest else set()
    p = values(spec)
    out["phaseA"] = (dispatcher_evidence(p["phase_a"], eps, STAGE_IDS["phaseA"]) if manifest
                     else _stage("pending", "after ingest"))
    a_done = out["phaseA"]["status"] == "done"
    out["rgb-publication"] = bundle_evidence(spec, "rgb", manifest) if a_done else _stage("pending", "after Phase A")
    out["rig"] = rig_evidence(spec) if a_done else _stage("pending", "after Phase A")
    out["phaseB"] = (dispatcher_evidence(p["phase_b"], eps, STAGE_IDS["phaseB"])
                     if out["rig"]["status"] == "done" else _stage("pending", "after the accepted rig"))
    out["geometry-publication"] = (bundle_evidence(spec, "geometry", manifest) if out["phaseB"]["status"] == "done"
                                   else _stage("pending", "after Phase B"))
    rgb, geo = out["rgb-publication"], out["geometry-publication"]
    if geo["status"] == "done" and rgb["status"] == "done" and (rgb["split"], rgb["outcomes"]) != (
            geo["split"], geo["outcomes"]):
        geo.update(status="blocked", detail="RGB and geometry publications disagree on split/outcomes")
    out["scene"] = (scene_evidence(spec, manifest, out["rig"]) if geo["status"] == "done"
                    else _stage("pending", "after the geometry publication"))
    for stage in out.values():
        stage.pop("split", None), stage.pop("outcomes", None)
    return out


def ready_waves(ev):
    return [w for w in ("rgb", "geometry") if all(ev[s]["status"] == "done" for s in STAGES[f"wave:{w}"][1])]


# ------------------------------------------------------------------------------------------------- proposal
def _merged_config(row):
    """(config path, merged config) for rows that pass a pnpv2 template plus --set overrides; else None."""
    args = row["args"]
    if "--config" not in args:
        return None
    from s4d.config import load_config

    path = args[args.index("--config") + 1]
    sets = args[args.index("--set") + 1:] if "--set" in args else []
    return path, load_config([ROOT / path], sets)


def check_rows(rows, current_ids, known, repo):
    """Structural safety of the proposal; returns a list of problems (empty = sound)."""
    problems = []
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        problems.append("duplicate row ids")
    if set(ids) & known:
        problems.append(f"already registered/attempted/owned: {sorted(set(ids) & known)[:3]}")
    for row in rows:
        rid = row["id"]
        if not rid.startswith(STEM + "-") or f"-{TASK}-S" not in rid or "-SF" in rid:
            problems.append(f"{rid}: not a pnpv2 S identity")
        if row["script"] not in WRAPPERS or not (ROOT / row["script"]).is_file():
            problems.append(f"{rid}: unknown entrypoint {row['script']}")
        for dep in row["deps"]:
            if not dep.startswith(STEM + "-") or (dep not in ids and dep not in current_ids):
                problems.append(f"{rid}: dependency {dep} is not a pnpv2 row of this graph")
        for arg in map(str, row["args"]):
            for ref in re.findall(r"(?:^|=|/)runs/([^/]+)/", arg):
                if not ref.startswith(STEM + "-"):
                    problems.append(f"{rid}: references a non-pnpv2 run directory runs/{ref}")
        if (repo / "runs" / rid).exists() or (repo / "runs" / rid).is_symlink():
            problems.append(f"{rid}: run output already exists")
        merged = _merged_config(row)
        if merged is not None and leftover_tokens(merged[1]):
            problems.append(f"{rid}: unresolved template fields {leftover_tokens(merged[1])}")
        method = row["realworld_method"]
        wave = "rgb" if method in RGB_METHODS or method == "rgb" else "geometry"
        if row["pnpv2_wave"] != wave or (method in GEOMETRY_METHODS and row["pnpv2_wave"] != "geometry"):
            problems.append(f"{rid}: wave/method classification mismatch")
    return problems


def placement_problems(rows, current):
    """Every row must resolve its inputs and be admissible on both hosts, exactly as placement will check."""
    from ops.jobs import host_roles, realworld_local

    roles, by_id, problems = host_roles.load_roles(), {**current, **{r["id"]: r for r in rows}}, []
    for row in rows:
        try:
            realworld_local.needed_inputs(row)
        except realworld_local.LocalInputError as exc:
            problems.append(f"{row['id']}: input resolver refuses: {exc}")
        for host in ("local", "remote"):
            reason = host_roles.refusal(row, host, roles, {}, by_id)
            if reason is not None:
                problems.append(f"{row['id']}: {host} placement refuses: {reason}")
    return problems


def inputs_of(rows, ev):
    hashes = {}
    for stage in ev.values():
        hashes.update(stage.get("inputs", {}))
    for row in rows:
        for arg in row["args"]:
            if str(arg).startswith("configs/") and str(arg).endswith(".yaml"):
                hashes[str(ROOT / arg)] = file_sha(ROOT / arg)
        hashes[str(ROOT / row["script"])] = file_sha(ROOT / row["script"])
    return hashes


def prepare(repo, spec, data_report, release, *, waves=None):
    """Dry-run record (no writes). Refuses unless check-data passed on the actual arrived recordings."""
    repo = Path(repo)
    require(data_report is not None and data_report.get("passed") is True,
            "check-data must pass on the actual arrived recordings: " + "; ".join(
                (data_report or {}).get("problems", ["not run"])[:3]))
    require(re.fullmatch(r"[0-9a-f]{40}", str(release.get("commit", ""))), "full 40-hex --commit required")
    require(re.fullmatch(r"sha256:[0-9a-f]{64}", str(release.get("image", ""))), "--image sha256:<64 hex> required")
    R = _reviewed()
    ev = evidence(spec, data_report)
    ready = ready_waves(ev)
    before = (repo / QUEUE).read_bytes()
    registry = (repo / REGISTRY).read_bytes() if (repo / REGISTRY).exists() else b""
    current = {r["id"]: r for r in R.queue_document(before)["jobs"]}
    registered = [w for w in ("rgb", "geometry") if row_id("gate", w) in current]
    selected = [w for w in ready if w not in registered] if waves is None else list(waves)
    for w in selected:
        require(w in ready, f"wave {w} is not ready: " + "; ".join(
            f"{s}={ev[s]['status']} ({ev[s]['detail']})" for s in STAGES[f"wave:{w}"][1] if ev[s]["status"] != "done"))
        require(w not in registered, f"wave {w} is already registered")
    require(selected, "nothing to register: no ready, unregistered wave")
    scene = ev["scene"].get("scene") if "geometry" in selected else None
    rows = build_rows(spec, release, scene=scene, waves=tuple(selected))
    from ops.jobs import host_roles

    known = set(current) | R.registry_ids(registry) | set(host_roles.read_ownership(repo / "runs/setup/ownership"))
    problems = check_rows(rows, set(current), known, repo) + placement_problems(rows, current)
    pauses = json.loads((repo / PAUSES).read_bytes())
    from s4d.rwlearn.pnpv2_arrival import v1_collisions, v1_unpaused

    v1_hits, other_holds, pause_problems = v1_collisions([r["id"] for r in rows], pauses)
    problems += pause_problems + [f"v1 discontinuation prefix/ID matches {i}" for i in v1_hits]
    problems += [f"pick-and-place v1 row not held by {V1_PAUSE} (resumable): {i}"
                 for i in v1_unpaused(current.values(), pauses)]
    require(not problems, "; ".join(problems[:5]))
    candidate = R.queue_candidate(before, rows)
    hashes = inputs_of(rows, ev)
    hashes[str(ROOT / SPEC)] = file_sha(ROOT / SPEC)
    return dict(schema="pick-and-place-v2-registration/1", status="dry-run", registered=False, launched=False,
                task=TASK, variant="S", waves=selected, already_registered=registered, release=release,
                repo=str(repo.resolve()), rows=rows, rows_sha256=json_sha(rows), counts=counts(rows),
                data_manifest_sha256=data_report["manifest_sha256"], evidence={k: {"status": v["status"],
                                                                                    "detail": v["detail"]}
                                                                                for k, v in ev.items()},
                input_sha256=hashes, tool_sha256=file_sha(Path(__file__)), queue_sha256_before=sha(before),
                queue_sha256_after=sha(candidate), registry_sha256=sha(registry), pause=PAUSE,
                other_active_holds=other_holds, v1_pause=V1_PAUSE)


def pause_candidate(raw, rows):
    """Hold exactly this publication's IDs (never a prefix) before the queue becomes visible."""
    doc = json.loads(raw)
    holds = [p for p in doc["pauses"] if p.get("name") == PAUSE]
    require(len(holds) <= 1, "duplicate pnpv2 placement pause")
    if holds:
        hold = holds[0]
        require(not hold.get("prefixes"), "pnpv2 placement pause must hold exact IDs")
    else:
        hold = dict(name=PAUSE, ids=[], prefixes=[],
                    reason="pick-and-place-v2 rows await verified placement (ops/jobs/realworld_local.py place)")
        doc["pauses"].append(hold)
    hold["ids"] = list(dict.fromkeys([*hold["ids"], *[r["id"] for r in rows]]))
    hold["active"] = True
    return (json.dumps(doc, indent=2) + "\n").encode()


@contextmanager
def publication_locks(repo):
    handles = []
    try:
        for name in LOCKS:  # the order of every real-world registrar and the placement publication
            handle = (Path(repo) / name).open("a")
            handles.append(handle)
            fcntl.flock(handle, fcntl.LOCK_EX)
        yield
    finally:
        for handle in reversed(handles):
            handle.close()


def execute(repo, record, *, recheck):
    """Append the dry-run rows under the queue/registry locks. ``recheck()`` re-runs check-data + evidence."""
    repo = Path(repo)
    R = _reviewed()
    require_main(repo)
    require(record.get("status") == "dry-run" and json_sha(record["rows"]) == record["rows_sha256"], "plan changed")
    setup = repo / "runs/setup"
    stem = f"{STEM}-{'+'.join(record['waves'])}-{record['rows_sha256'][:16]}"
    pending, proof = setup / f"{stem}.pending.json", setup / f"{stem}.registration.json"

    def check():
        require(R.current_pin(repo, record["release"]) == record["release"], "release differs from adopted sources")
        require(file_sha(Path(__file__)) == record["tool_sha256"], "registrar changed")
        require(sha((repo / QUEUE).read_bytes()) == record["queue_sha256_before"], "queue changed")
        registry = (repo / REGISTRY).read_bytes() if (repo / REGISTRY).exists() else b""
        require(sha(registry) == record["registry_sha256"], "registry changed")
        require(all(Path(p).is_file() and file_sha(p) == v for p, v in record["input_sha256"].items()),
                "inputs changed")
        require(all(not (repo / "runs" / r["id"]).exists() for r in record["rows"]), "run output appeared")

    with publication_locks(repo):
        require(not pending.exists() and not proof.exists(), "existing journal/proof requires inspection")
        check()
        fresh = recheck()
        require(fresh.get("passed") is True and fresh.get("manifest_sha256") == record["data_manifest_sha256"],
                "check-data no longer passes on the same recordings")
        candidate = R.queue_candidate((repo / QUEUE).read_bytes(), record["rows"])
        require(sha(candidate) == record["queue_sha256_after"], "candidate changed")
        pauses = repo / PAUSES
        pause_before = pauses.read_bytes()
        from s4d.rwlearn.pnpv2_arrival import v1_collisions, v1_unpaused

        queue_rows = R.queue_document((repo / QUEUE).read_bytes())["jobs"]
        v1_hits, _, problems = v1_collisions([r["id"] for r in record["rows"]], json.loads(pause_before))
        require(not v1_hits and not problems and not v1_unpaused(queue_rows, json.loads(pause_before)),
                "v1 discontinuation hold changed, matches v2 rows or no longer holds every v1 row")
        pause_after = pause_candidate(pause_before, record["rows"])
        record = dict(record, pause_sha256_before=sha(pause_before), pause_sha256_after=sha(pause_after))
        R.atomic_write(pending, R.json_bytes(record))
        require(pauses.read_bytes() == pause_before, "pauses changed under locks")
        R.atomic_write(pauses, pause_after)  # fail closed: rows are held before they become visible
        check()
        R.atomic_write(repo / QUEUE, candidate)
        record = dict(record, status="published", registered=True, registered_at=R.now())
        R.atomic_write(proof, R.json_bytes(record))
        pending.unlink()
        R.sync_directory(setup)
    return record


# ----------------------------------------------------------------------------------------------- next steps
PY = "/home/ws/ws/splatter4d/.venv/bin/python -I"


def next_steps(spec, ev):
    """The exact producer command for each pending stage whose prerequisites are done (docs: PICK_AND_PLACE_V2.md)."""
    p = values(spec)
    w = p["work"]
    producer = f"{PY} scripts/pick_and_place_v2.py --entry"
    prep = (f"--processed {p['remote_processed']} --raw {spec['roots']['raw']} --weights <prep weights dir> "
            f"--image $(cat {w}/prep-image-id) --code {w}/prep-code --control-repo /home/ws/ws/splatter4d")
    commands = {
        "arrival": [f"operator: verified pull of Robot-PC teleop_collector/data/{TASK}/success into {p['raw_success']}"
                    " with success.local.sha256 / success.robotpc.sha256 / success.VERIFIED (dsreset transfer layout);"
                    " this tooling never contacts the Robot PC"],
        "check-data": [f"{PY} scripts/pnpv2.py check-data"],
        "ingest": [f"write {p['arrival_doc']} (schema pick-and-place-v2-arrival/1: reviewed sam3_prompts, "
                   "robot_urdf_sha256, confirmed_by)", f"{producer} manifest"],
        "phaseA": [f"{PY} scripts/realworld_prep_register.py --phase A --id-stem {STAGE_IDS['phaseA']} "
                   f"--plan {p['prep_plan']} --out {p['phase_a']} {prep}",
                   f"LOCAL HOST: launch-prep-controller.sh {p['phase_a']} (dsreset RUNBOOK section 3)"],
        "rgb-publication": [f"{producer} publish-rgb -- --execute", f"{producer} upload -- --view rgb"],
        "rig": [f"{PY} scripts/realworld_dsreset_calibration.py plan --id-stem {STAGE_IDS['rig']} --task {TASK} "
                f"--phase-config {p['phase_a']}/dispatcher.json --out {p['calib']} "
                f"--work {p['remote_processed']}/calibration/{TASK} --upstream <calib-upstream>",
                "LOCAL HOST: start its controller (waits for Phase A automatically; automatic G1-G6 acceptance)"],
        "phaseB": [f"{producer} rig-map",
                   f"{PY} scripts/realworld_prep_register.py --phase B --id-stem {STAGE_IDS['phaseB']} "
                   f"--plan {p['prep_plan']} --rig-map {w}/phaseB-rig-map.json --out {p['phase_b']} {prep} "
                   f"--extra-readonly {p['remote_processed']}/calibration/{TASK}"],
        "geometry-publication": [
            f"{producer} raw-proof",
            f"{producer} snapshot -- --success-only --source /home/ws/ws/realworld-ops/ingest/manifest.json "
            f"--manifest {p['manifest']} --prep-config {p['phase_a']}/dispatcher.json --prep-config "
            f"{p['phase_b']}/dispatcher.json --remote-hashes {w}/remote-raw-hashes.json --out-dir {p['snapshot']}",
            f"{PY} scripts/realworld_prep_followup.py records --id-stem {dispatcher_stem('qa')} --task {TASK} "
            f"--base-config {p['phase_b']}/dispatcher.json --rig {p['rig']} --manifest {p['snapshot']}/manifest.json "
            f"--out {w}/records --result {p['remote_processed']}/calibration-records/{TASK}",
            f"{PY} -m s4d.realworld.calib.release apply --plan {p['remote_processed']}/calibration-records/{TASK}/"
            f"plan.json --state-dir {p['snapshot']} [--apply]",
            f"{producer} publish -- --ingest-manifest {p['snapshot']}/manifest.json",
            f"{producer} upload -- --view geometry"],
        "scene": [f"{PY} scripts/realworld_prep_followup.py measure --id-stem {dispatcher_stem('qa')} --task {TASK} "
                  f"--base-config {p['phase_b']}/dispatcher.json --rig {p['rig']} --inputs {p['geometry_dir_local']} "
                  f"--out {w}/measure --result {p['remote_processed']}/measurements/{TASK}"],
    }
    out = {}
    for stage, (kind, needs) in STAGES.items():
        if kind != "external" or ev[stage]["status"] == "done":
            continue
        if all(ev[n]["status"] == "done" for n in needs):
            out[stage] = commands[stage]
    for wave in ready_waves(ev):
        out[f"wave:{wave}"] = [f"{PY} scripts/pnpv2.py register --wave {wave} --commit <main HEAD> --image <role image> "
                               "--release-source local   # dry run; add --execute from the adopted main"]
    return out
