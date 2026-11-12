"""Inventory, deployability checks and inference-host staging of trained real-world FQL policies.

    .venv/bin/python -I ops/policy_serving/staging.py inventory [--evaluate] --out-json INV.json --out-md INV.md
    .venv/bin/python -I ops/policy_serving/staging.py stage RUN [RUN ...] [--execute]   # dry run by default
    .venv/bin/python -I ops/policy_serving/staging.py hook RUN [--execute]              # monitoring hook (Phase 2)
    .venv/bin/python -I ops/policy_serving/staging.py bcflow-variant ID [ID ...] [--decision-json F] [--execute]

The registry and queue are DATA here (read-only JSONL / YAML; no ``ops.campaign`` import). Run directories, FQL
datasets and recordings are only read. ``stage --execute`` copies a deployable policy (checkpoint, encoder export,
deploy-contract sidecar; never training datasets) to the inference host's checkpoint root (sha256-verified, atomic,
never overwriting a differing file), into the server container's RAM root, rewrites the catalog there atomically
and asks the running server to reload it (SIGHUP); ``hook`` re-applies the POL-CRITERIA.md checks first and stages
only a deployable policy. Criteria: ``ops/policy_serving/criteria.py``. Nothing here talks to a robot.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import json
import re
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

if __package__ in (None, ""):  # ``python ops/policy_serving/staging.py``: make the repository importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ops.policy_serving import criteria as C  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CONDITIONS = ("e2e_cnn", "mvp3d_single", "mvp3d_multiview", "dynamo", "reviwo", "sincro", "snerl", "ours")
PRETRAIN_OF = {"mvp3d_single": "mvp3d", "mvp3d_multiview": "mvp3d"}  # FQL condition -> pretraining family
_C = "|".join(CONDITIONS)
FQL_RE = re.compile(rf"^(?P<prefix>.+)-fql-(?P<condition>{_C})-(?P<task>.+)-s(?P<seed>\d+)$")
DATA_RE = re.compile(rf"^(?P<prefix>.+)-fqldata-(?P<condition>{_C})-(?P<task>.+)$")
PRE_RE = re.compile(r"^(?P<prefix>.+)-(?P<condition>dynamo|reviwo|sincro|snerl|mvp3d)-(?P<task>.+)-pre(?:-native)?$")
# ours: the encoder export is its own row (scripts/export_encoder.py) after the scripts/train.py pretraining row
# <prefix>-ours-<task>-pre; the export row stands for the pretraining (criterion a = the -pre run's completion).
OURS_EXPORT_RE = re.compile(r"^(?P<prefix>.+)-(?P<condition>ours)-(?P<task>.+)-export$")
# FQL settings (user decision 2026-11-11 ~23:10, FQLNQ): rows registered with a setting prefix are served under
# DISTINCT ids <condition>-<task>-s<seed>-<setting>, never replacing the alpha-10 ids of the earlier rows; the a100
# comparison baseline is staged only when explicitly asked (``stage``/``hook`` --allow-comparison).
FQL_SETTINGS = {"rwnq-": ("a10nq", 10.0, True), "rwa100-": ("a100", 100.0, False)}
COMPARISON_SETTINGS = ("a100",)
# Inventories record the policy-id rule they were evaluated under; ``stage --inventory`` refuses any other (an inventory
# from before the FQL-setting rule names rwnq rows by the alpha-10 ids and would overwrite those entries).
STAGING_RULE = "fql-settings/1"
# explicit decisions (MVP3D-FQL-DIAG.md, Directive I item 5): never staged as deployable until reruns pass
QUARANTINE = {
    re.compile(r"^rwcal-20261109-fql-mvp3d_(single|multiview)-.+-s\d+$"):
        "3D-MVP frozen read-out offset (offset/signal 84-136): vision-blind policies, 8/12 critics diverged "
        "(MVP3D-FQL-DIAG.md); MVP agent fixing, reruns pending",
}
# supersession recorded in the bookkeeping commits of main (never launched rows)
SUPERSEDED = {
    re.compile(r"^rwng-20261106-"): "superseded by rwng-20261107-* (episode-join fix 782f140; bookkeeping e9338cb: "
                                    "'rwng-20261106-* superseded, paused')",
    re.compile(r"^rwng-20261107-fql(data)?-"): "superseded by rwng-20261107b-* (FQL frame-mapping fix ee07b2e; "
                                               "bookkeeping cfc4b21: 'old fqldata/fql rows superseded, paused')",
}
CATALOG_NAME = "catalog.json"
VALIDATION_DIR = Path("/home/ws/ws/realworld-ops/serving/validation")  # auto-eligibility deployment-validation records
# released rig documents per task (realworld-ops/serving/API.md; DEP's validated replays)
RIG_DOCUMENTS = {"drawer-open": "rig-loma-drawer-open.json", "mug-hanging": "rig-loma-mug-hanging-fx.json"}


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text()) or {}


def _json(path: Path):
    return json.loads(path.read_text()) if path.is_file() else None


# ------------------------------------------------------------------------------------------------ records
@dataclass
class Lifecycle:
    """One registry row's lifecycle from its events (data only)."""
    events: list = field(default_factory=list)

    @property
    def last_launch(self) -> int | None:
        idx = [i for i, e in enumerate(self.events) if e["event"] in ("launching", "launched")]
        return idx[-1] if idx else None

    def after_launch(self, kind: str) -> list:
        start = self.last_launch
        return [e for e in self.events[(start or 0):] if e["event"] == kind] if start is not None else []

    @property
    def synced(self) -> dict | None:
        """The final results sync after the last launch (its ``sync.files``: remote-verified sha256 per file)."""
        done = [e for e in self.after_launch("results_synced") if e.get("final")]
        return done[-1] if done else None

    @property
    def state(self) -> str:
        if not self.events:
            return "never_launched"
        kinds = [e["event"] for e in self.events]
        if kinds[-1] == "reset":
            return "reset"
        if self.synced is not None:
            codes = [e.get("code") for e in self.after_launch("exited") + self.after_launch("remote_exited")]
            return "completed_synced" if codes and all(c == 0 for c in codes) else "synced_nonzero_exit"
        if "failed" in kinds[(self.last_launch or 0):]:
            return "failed"
        launch = self.events[self.last_launch] if self.last_launch is not None else {}
        exits = self.after_launch("exited")
        if launch.get("host", "local") == "local" and exits and not self.after_launch("remote_exited") and \
                all(e.get("code") == 0 for e in exits) and kinds[-1] == "exited":
            # a LOCAL run that exited 0 has its results in place (no sync step; the scheduler records results_synced
            # only for remote runs); completion itself is still checked from completion.json + config (criterion b)
            return "completed_local"
        if exits or self.after_launch("remote_exited"):
            return "exited_unsynced"
        return "running"

    def remote_sha256(self, name: str) -> str | None:
        s = self.synced
        return None if s is None else (s.get("sync") or {}).get("files", {}).get(name)


def read_registry(path: Path, prefix: str = "rw") -> dict[str, Lifecycle]:
    rows: dict[str, Lifecycle] = {}
    with open(path) as f:
        for line in f:
            if f'"id": "{prefix}' not in line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.setdefault(e["id"], Lifecycle()).events.append(e)
    return rows


def read_queue(path: Path, prefix: str = "rw") -> dict[str, dict]:
    return {j["id"]: j for j in (_yaml(path).get("jobs") or []) if str(j.get("id", "")).startswith(prefix)}


def _arg(job: dict | None, key: str) -> str | None:
    for a in (job or {}).get("args", []):
        if isinstance(a, str) and a.startswith(key + "="):
            return a.split("=", 1)[1]
    return None


def superseded(run: str) -> str | None:
    return next((why for rx, why in SUPERSEDED.items() if rx.search(run)), None)


def quarantine_decision(run: str) -> str | None:
    return next((why for rx, why in QUARANTINE.items() if rx.search(run)), None)


def fql_setting(run: str) -> str | None:
    """The FQL setting of a run id (``FQL_SETTINGS`` prefix), None for the earlier rows (alpha 10, no normalisation)."""
    return next((v[0] for k, v in FQL_SETTINGS.items() if run.startswith(k)), None)


def policy_id(condition: str, task: str, seed: int, setting: str | None = None) -> str:
    return f"{condition}-{task}-s{seed}" + (f"-{setting}" if setting else "")


def setting_check(run: str, config: dict) -> str | None:
    """A setting-prefixed run must have trained with exactly its setting (agent.alpha, agent.normalize_q_loss of its
    own config.yaml); the reason of a mismatch, else None."""
    spec = next((v for k, v in FQL_SETTINGS.items() if run.startswith(k)), None)
    if spec is None or not config:
        return None
    agent = config.get("agent") or {}
    alpha, normalize = agent.get("alpha"), agent.get("normalize_q_loss")
    if alpha is None or float(alpha) != spec[1] or normalize is not spec[2]:
        return (f"config.yaml agent.alpha {alpha!r} / normalize_q_loss {normalize!r} differ from the {spec[0]} setting "
                f"(alpha {spec[1]}, normalize_q_loss {spec[2]})")
    return None


# ------------------------------------------------------------------------------------------------ inventory
@dataclass
class Sources:
    data_root: Path              # the checkout whose runs/ holds synced run directories (read-only)
    registry: Path
    queue: Path
    splits: Path                 # <task>/split_seed0.json
    raw_root: Path               # raw recordings (read-only)
    rig_dir: Path | None = None  # released task rig documents (rig-loma-<task>*.json)

    @property
    def runs(self) -> Path:
        return self.data_root / "runs"

    def rig(self, task: str) -> Path:
        """The task's released rig document (calibration of the deploy session): RIG_DOCUMENTS, else the unique
        ``rig-loma-<task>.json``."""
        if self.rig_dir is None:
            raise ValueError("no rig directory configured (--rig-dir)")
        p = self.rig_dir / RIG_DOCUMENTS.get(task, f"rig-loma-{task}.json")
        if not p.is_file():
            raise ValueError(f"no released rig document {p}")
        return p

    def split(self, task: str) -> dict:
        return json.loads((self.splits / task / "split_seed0.json").read_text())


def _file_record(src: Sources, life: Lifecycle | None, run: str, name: str, *, hash_local: bool = True) -> dict:
    p = src.runs / run / name
    remote = life.remote_sha256(name) if life else None
    rec = {"path": f"runs/{run}/{name}", "local": p.is_file(), "remote_verified_sha256": remote}
    if p.is_file() and hash_local:
        rec["sha256"] = sha256_file(p)
        rec["matches_remote"] = None if remote is None else rec["sha256"] == remote
    elif remote is not None:
        rec["sha256"] = remote  # REMOTE runs/<id>/<name>, sha256 verified on the remote by the sync
    return rec


def pretraining_row(src: Sources, run: str, life: Lifecycle | None, job: dict | None) -> dict:
    m = PRE_RE.match(run)
    d = src.runs / run
    config = _yaml(d / "config.yaml") if (d / "config.yaml").is_file() else {}
    completion = _json(d / "completion.json")
    a = C.pretraining_complete(config, completion) if config else C.check(False, reason="no local config.yaml")
    state = life.state if life else "never_launched"
    if state not in ("completed_synced", "completed_local"):
        a = {**a, "pass": False, "reason": f"registry state {state}" + (f"; {a['reason']}" if "reason" in a else "")}
    return {"run": run, "kind": "encoder_pretraining", "condition": m["condition"], "task": m["task"],
            "registry_state": state, "superseded": superseded(run), "budget": {k: a.get(k) for k in ("kind", "budget")},
            "completed": completion, "a": a, "encoder": _file_record(src, life, run, "encoder.pt")}


def ours_export_row(src: Sources, run: str, life: Lifecycle | None, pre_life: Lifecycle | None) -> dict:
    """ours: the export row (its encoder.pt) with criterion a of its pretraining run ``<prefix>-ours-<task>-pre``
    (scripts/train.py config.yaml + completion.json); both rows must be completed and synced."""
    m = OURS_EXPORT_RE.match(run)
    pre = run[:-len("-export")] + "-pre"
    d = src.runs / pre
    config = _yaml(d / "config.yaml") if (d / "config.yaml").is_file() else {}
    completion = _json(d / "completion.json")
    a = C.pretraining_complete(config, completion) if config else C.check(False, reason=f"no local {pre}/config.yaml")
    states = {pre: pre_life.state if pre_life else "never_launched", run: life.state if life else "never_launched"}
    bad = {k: v for k, v in states.items() if v not in ("completed_synced", "completed_local")}
    if bad:
        a = {**a, "pass": False, "reason": f"registry state {bad}" + (f"; {a['reason']}" if "reason" in a else "")}
    return {"run": run, "kind": "encoder_pretraining", "condition": m["condition"], "task": m["task"],
            "pretraining_run": pre, "registry_state": states[run], "superseded": superseded(run),
            "budget": {k: a.get(k) for k in ("kind", "budget")}, "completed": completion, "a": a,
            "encoder": _file_record(src, life, run, "encoder.pt")}


def fql_row(src: Sources, run: str, life: Lifecycle | None, job: dict | None) -> dict:
    m = FQL_RE.match(run)
    d = src.runs / run
    config = _yaml(d / "config.yaml") if (d / "config.yaml").is_file() else {}
    completion = _json(d / "completion.json")
    state = life.state if life else "never_launched"
    b = C.fql_complete(config, completion) if config else C.check(False, reason="no local config.yaml",
                                                                    budget=(job or {}).get("fql_budget"))
    if state not in ("completed_synced", "completed_local"):
        b = {**b, "pass": False, "reason": f"registry state {state}" + (f"; {b['reason']}" if "reason" in b else "")}
    setting = fql_setting(run)
    if (wrong := setting_check(run, config)) is not None:
        b = {**b, "pass": False, "reason": wrong + (f"; {b['reason']}" if "reason" in b else "")}
    dataset = ((config.get("data") or {}).get("dataset")) or _arg(job, "data.dataset")
    policy = _file_record(src, life, run, "policy.pt")
    if state == "completed_local":
        policy["local_completion"] = True  # criteria.checkpoint_matches_completion: no remote sync for local runs
    return {"run": run, "kind": "fql", "condition": m["condition"], "task": m["task"], "seed": int(m["seed"]),
            "fql_setting": setting,
            "policy_id": policy_id(m["condition"], m["task"], int(m["seed"]), setting), "registry_state": state,
            "superseded": superseded(run), "quarantine_decision": quarantine_decision(run), "dataset": dataset,
            "budget_steps": b.get("budget"), "completed": completion, "b": b,
            "policy": policy}


def inventory(src: Sources) -> dict:
    """Every real-world pretraining / FQL-dataset / FQL row of the registry + queue, with completion records."""
    reg, queue = read_registry(src.registry), read_queue(src.queue)
    ids = sorted(set(reg) | set(queue))
    pre, data, fql, other = [], [], [], []
    for run in ids:
        life, job = reg.get(run), queue.get(run)
        if PRE_RE.match(run):
            pre.append(pretraining_row(src, run, life, job))
        elif OURS_EXPORT_RE.match(run):
            pre.append(ours_export_row(src, run, life, reg.get(run[:-len("-export")] + "-pre")))
        elif FQL_RE.match(run):
            fql.append(fql_row(src, run, life, job))
        elif DATA_RE.match(run):
            m = DATA_RE.match(run)
            data.append({"run": run, "kind": "fql_dataset", "condition": m["condition"], "task": m["task"],
                         "registry_state": life.state if life else "never_launched", "superseded": superseded(run),
                         "dataset_local": (src.runs / run / "dataset.h5").is_file()})
        else:
            other.append({"run": run, "registry_state": life.state if life else "never_launched"})
    return {"pretraining": pre, "fql_datasets": data, "fql": fql, "other": other}


# ------------------------------------------------------------------------------------------------ evaluation
def _dataset_attrs(path: Path) -> dict | None:
    import h5py

    if not path.is_file():
        return None
    with h5py.File(path, "r") as f:
        return {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f.attrs.items()
                if k in ("condition", "export_path", "export_sha256", "format", "camera_orders", "image_hw")}


def pretrain_run_of(src: Sources, row: dict) -> str:
    """The pretraining run of an FQL row: the export recorded in its (local) FQL dataset, else the row-id convention
    ``<prefix>-<family>-<task>-pre`` (ours: its export row ``-export``; datasets not built yet)."""
    if row.get("dataset"):
        attrs = _dataset_attrs(src.data_root / row["dataset"])
        if attrs and attrs.get("export_path"):
            return Path(attrs["export_path"]).parent.name
    prefix = row["run"].split("-fql-")[0]
    suffix = "-export" if row["condition"] == "ours" else "-pre"
    return f"{prefix}-{PRETRAIN_OF.get(row['condition'], row['condition'])}-{row['task']}{suffix}"


def write_sidecars(src: Sources, runs: list[str], out_dir: Path) -> dict:
    """d: ``scripts/check_realworld_deploy.py contracts`` (the repository's sidecar writer) for ``runs``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-I", str(REPO / "scripts/check_realworld_deploy.py"), "contracts", "--runs-root",
           str(src.runs), "--out-dir", str(out_dir), *runs]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        return {run: C.check(False, reason=f"sidecar writer failed: {r.stderr.strip()[-400:]}") for run in runs}
    written = json.loads(r.stdout) if r.stdout.strip().startswith("{") else {}
    res = {}
    for run in runs:
        p = out_dir / f"{run}.deploy_contract.json"
        res[run] = C.check(p.is_file(), sidecar=str(p), sha256=sha256_file(p) if p.is_file() else None,
                           camera_orders=(written.get(run) or {}).get("camera_orders"))
    return res


def evaluate(src: Sources, rows: list[dict], pre_rows: dict[str, dict], *, sidecar_dir: Path, probe: bool = True,
             log=print) -> dict[str, dict]:
    """Criteria a-f for FQL rows whose a/b allow it (candidates); every other row is classified from a/b alone."""
    import torch

    from s4d.rwlearn.deploy import DeployPolicy

    torch.set_num_threads(min(8, torch.get_num_threads()))
    out, candidates = {}, []
    for r in rows:
        checks = {"b": r["b"]}
        if r["condition"] != "e2e_cnn":
            pre_run = pretrain_run_of(src, r)
            r["pretrain_run"] = pre_run
            checks["a"] = pre_rows[pre_run]["a"] if pre_run in pre_rows else \
                C.check(False, reason=f"pretraining row {pre_run} not in the registry/queue")
        out[r["run"]] = checks
        if r["superseded"] is None and checks["b"]["pass"] and checks.get("a", {"pass": True})["pass"]:
            candidates.append(r)
    sidecars = write_sidecars(src, [r["run"] for r in candidates], sidecar_dir) if candidates else {}
    samples: dict[tuple, list] = {}
    e1_cache: dict[tuple, dict] = {}
    for r in candidates:
        run, checks = r["run"], out[r["run"]]
        log(f"[evaluate] {run}")
        ck = src.runs / run / "policy.pt"
        payload = torch.load(ck, map_location="cpu", weights_only=False)
        meta, norm = payload["meta"], C.feature_norm_of(payload)
        checks["b"] = C.checkpoint_matches_completion(checks["b"], payload.get("step"), r["policy"])
        from ops.policy_serving.config import LEGACY_SETTING, agent_setting_matches

        agent_cfg = (payload.get("agent") or {}).get("config")
        if checks["b"]["pass"] and not agent_setting_matches(r.get("fql_setting") or LEGACY_SETTING, agent_cfg):
            # the WEIGHTS must have trained with the run id's FQL setting (eligibility follows the setting)
            got = {k: (agent_cfg or {}).get(k) for k in ("alpha", "normalize_q_loss")}
            checks["b"] = {**checks["b"], "pass": False, "reason": f"checkpoint agent.config {got} is not the "
                           f"{r.get('fql_setting') or LEGACY_SETTING} setting"}
        del payload
        if not checks["b"]["pass"]:  # not the completed artefact: no further criteria
            continue
        ds = src.runs / Path(meta["dataset"]).parent.name / "dataset.h5"
        attrs = _dataset_attrs(ds)
        enc = None
        if meta.get("export_path"):
            pre_run = Path(meta["export_path"]).parent.name
            enc_rec = (pre_rows.get(pre_run) or {}).get("encoder") or {}
            enc = enc_rec.get("sha256")
            r["export"] = {"path": meta["export_path"], "pretrain_run": pre_run, "sha256": enc}
            if pre_run in pre_rows:
                checks["a"] = pre_rows[pre_run]["a"]  # the run that actually produced the export
        checks["c"] = C.export_chain(meta, attrs, enc)
        checks["d"] = sidecars.get(run, C.check(False, reason="no sidecar"))
        metrics = [json.loads(x) for x in (src.runs / run / "metrics.jsonl").read_text().splitlines() if x.strip()]
        checks["f"] = C.stability(metrics, int(r["budget_steps"]))
        if r["condition"] != "e2e_cnn":
            key = (ds, None if norm is None else hashlib.sha256(norm[0].tobytes() + norm[1].tobytes()).hexdigest())
            if key not in e1_cache:
                e1_cache[key] = C.feature_conditioning(ds, src.split(r["task"])["train"], norm) if ds.is_file() \
                    else C.check(None, reason=f"dataset {ds} not local")
            checks["e1"] = {**e1_cache[key], "dataset": str(ds.relative_to(src.data_root))}
        if r["quarantine_decision"]:
            checks["e2"] = C.check(None, reason="not run: quarantined by decision (MVP3D-FQL-DIAG.md probe)")
        elif probe and checks["c"]["pass"] and checks["d"]["pass"]:
            export = src.data_root / meta["export_path"] if meta.get("export_path") else None
            try:
                pol = DeployPolicy(ck, device="cpu", export_path=export, training=checks["d"]["sidecar"])
            except Exception as exc:  # noqa: BLE001 - a load failure is a c failure, reported
                checks["c"] = {**checks["c"], "pass": False, "reason": f"DeployPolicy load failed: {exc}"}
            else:
                checks["c"]["deploy_load_cpu"] = True
                spec = pol.input_spec
                cams = sorted({c for o in spec.trained_camera_orders for c in o})
                views = list(spec.trained_camera_orders[0]) if spec.views > 1 else None
                depth = "depth" in spec.modalities
                rig = src.rig(r["task"]) if spec.needs_calibration or depth else None
                key = (r["task"], tuple(views or cams), depth, None if rig is None else str(rig))
                try:
                    if key not in samples:
                        log(f"[probe] sampling {C.E2_SAMPLES} held-out frames for {key}")
                        samples[key] = C.probe_samples(src.split(r["task"])["validation"], src.raw_root, cams,
                                                       views=views, depth=depth, rig_document=rig)
                    checks["e2"] = C.visual_probe(pol, samples[key])
                except Exception as exc:  # noqa: BLE001 - an unevaluable probe never passes
                    checks["e2"] = C.check(None, reason=f"probe failed: {type(exc).__name__}: {exc}")
                del pol
        else:
            checks["e2"] = C.check(None, reason="not run (c/d failed or --no-probe)")
    return out


def classify_rows(rows: list[dict], checks: dict[str, dict]) -> None:
    for r in rows:
        if r["superseded"]:
            r["class"] = {"status": "superseded", "reasons": [r["superseded"]], "flags": []}
            continue
        r["checks"] = checks.get(r["run"], {"b": r["b"]})
        r["class"] = C.classify(r["condition"], r["checks"], quarantine=r["quarantine_decision"])
        if r["run"] not in checks and r["class"]["status"] == "quarantined" and not r["quarantine_decision"]:
            r["class"] = {"status": "unevaluated", "reasons": ["criteria c-f not run (inventory without --evaluate)"],
                          "flags": []}


# ------------------------------------------------------------------------------------------------ catalog
def catalog_entry(row: dict, input_spec: dict | None = None, validation: dict | None = None) -> dict:
    """Catalog entry of a classified FQL row (paths relative to the checkpoint root, mirroring ``runs/``).
    ``input_spec``: the advertised unbound spec (``server.describe_entry`` of the local files; the server checks it
    against the loaded policy)."""
    exp = row.get("export") or {}
    status = row["class"]["status"]
    # the FQL setting and identity come from the RUN ID, never from a cached label: an inventory row whose label
    # disagrees (e.g. a missing fql_setting on an rwnq run) is refused instead of being granted another setting's
    # usage (Codex review of 1847fdd, P0)
    m, setting = FQL_RE.match(str(row.get("run") or "")), fql_setting(str(row.get("run") or ""))
    if m is None or (m["condition"], m["task"], int(m["seed"])) != (row.get("condition"), row.get("task"),
                                                                    int(row.get("seed", -1))) \
            or row.get("fql_setting") != setting:
        raise ValueError(f"inventory row {row.get('run')!r}: condition/task/seed/fql_setting "
                         f"{(row.get('condition'), row.get('task'), row.get('seed'), row.get('fql_setting'))} differ "
                         f"from its run id (setting {setting}): re-evaluate")
    criteria = C.criteria_summary(row["condition"], row.get("checks") or {})
    failed = {k for k, v in criteria.items() if isinstance(v, dict) and v.get("pass") is False}
    out = {"checkpoint": row["policy"]["path"], "sha256": row["policy"]["sha256"],
           "export": exp.get("path"), "training_contract": f"contracts/{row['run']}.deploy_contract.json",
           "condition": row["condition"], "task": row["task"], "seed": row["seed"], "run": row["run"],
           "status": status, "reasons": row["class"]["reasons"],
           "flags": list(row["class"]["flags"]) + ([f"fql_setting_{setting}"] if setting else []),
           "criteria": criteria,
           **usage_of(status, row["condition"], setting, checkpoint_sha256=row["policy"]["sha256"],
                      failed_criteria=failed, validation=validation)}
    if status == "restricted" and out["execution_eligible"] and (input_spec or {}).get("condition") != row["condition"]:
        # a restricted grant is published only with the checkpoint's described input_spec of the same condition (the
        # server binds it to the weights; the validator refuses a grant without it): otherwise dry run / replay only
        out = {k: v for k, v in out.items() if k not in ("usage", "execution_eligible", "execution_decision",
                                                          "usage_reasons", "deploy_validation")}
        out.update(usage_of(status, row["condition"], setting))
    if input_spec is not None:
        out["input_spec"] = input_spec
    return out


EXECUTION_DECISION = {"decided_by": "user", "at": "2026-11-10T21:10+09:00",
                      "text": "I think you should make execution_eligible=True for the trained policies. (scope since the "
                              "user decision 2026-11-12 ~16:00: the end-to-end CNN policies, which stay eligible after "
                              "the established validation; legacy frozen-encoder policies are no longer eligible)"}
NOT_ELIGIBLE = "not execution-eligible"
# User decisions 2026-11-12 ~16:00 (relayed by the coordinator), replacing the 2026-11-10 / 2026-11-12 ~00:20 rules:
# * legacy frozen-encoder FQL policies (alpha 10, normalize_q_loss false; every condition but e2e_cnn) are NOT
#   execution-eligible and are never re-enabled automatically (checkpoints / results stay; only eligibility changes);
# * E2E CNN policies stay eligible (they passed the established deployment validation);
# * new-setting FQL policies (a10nq: alpha 10 + normalize_q_loss true) of EVERY condition (ours, reviwo, sincro, dynamo,
#   snerl, mvp3d_single, mvp3d_multiview) become eligible AUTOMATICALLY once trained and once the deployment validation
#   passed (``deploy_validation`` record bound to the checkpoint: offline deploy-vs-training equality, server direct ==
#   network, robot-PC client replay == direct); no per-policy approval; manual execution through the existing workflow;
# * the a100 comparison setting stays NOT eligible; the blanket 3D-MVP block is removed (old vision-blind 3D-MVP stays
#   quarantined by QUARANTINE; the legacy native-depth 3D-MVP reruns are legacy frozen-encoder policies).
LEGACY_FROZEN_REASON = ("user decision 2026-11-12 ~16:00: legacy frozen-encoder FQL policies (alpha 10, "
                        "normalize_q_loss false) are NOT execution-eligible and never re-enabled automatically "
                        "(checkpoints and results kept)")
COMPARISON_REASON = "FQL comparison setting a100: NOT execution-eligible unless separately authorized"
AWAITING_VALIDATION = ("new-setting (a10nq) policy: execution-eligible automatically once its deployment validation "
                       "passes (offline equality, server direct == network, robot-PC client replay); not yet passed")
AUTO_NQ_DECISION = {"decided_by": "user", "at": "2026-11-12 ~16:00 KST (relayed by the coordinator)",
                    "text": "AUTOMATICALLY ENABLE new-setting FQL policies (agent.alpha=10 + agent.normalize_q_loss=true) "
                            "as soon as training finishes AND the required deployment validations pass (Ours, ReViWo, "
                            "SinCro, DynaMo, SNeRL, 3D-MVP single, 3D-MVP multi-view); no per-policy user approval; "
                            "available for manual execution through the existing workflow only"}
VALIDATION_CHECKS = ("l2_offline", "server_network", "robotpc_client")  # == config.DEPLOY_VALIDATION_CHECKS


def setting_reason(setting: str) -> str:
    return COMPARISON_REASON if setting == "a100" else AWAITING_VALIDATION if setting == "a10nq" else \
        f"FQL setting {setting}: not execution-eligible"


def exclusions(condition: str | None, setting: str | None) -> list[str]:
    """Reasons an entry of this condition / FQL setting (None = legacy) can never be execution-eligible by rule (the
    validation-gated a10nq setting is not an exclusion); [] otherwise."""
    if setting is None or setting == "legacy":
        return [] if condition == "e2e_cnn" else [LEGACY_FROZEN_REASON]
    return [] if setting == "a10nq" else [setting_reason(setting)]


def validation_record(path: Path | None, run: str, checkpoint_sha256: str, *, training_contract_sha256: str | None = None,
                      input_spec: dict | None = None) -> dict | None:
    """The deployment-validation record of a run (``<dir>/<run>.json``, written by the auto-eligibility pipeline) as a
    compact catalog summary, or None unless it is well formed, passed overall and in every check, and bound to this
    run, these weights, this training contract and this input_spec (a re-staged contract or spec needs a new record)."""
    from ops.policy_serving.config import input_spec_digest, valid_sha256

    if path is None:
        return None
    f = Path(path) / f"{run}.json"
    if not f.is_file():
        return None
    try:
        rec = json.loads(f.read_text())
    except ValueError:
        return None
    checks = rec.get("checks") if isinstance(rec, dict) else None
    if not isinstance(checks, dict) or set(checks) != set(VALIDATION_CHECKS) or rec.get("pass") is not True or \
            not all(isinstance(checks[k], dict) and checks[k].get("pass") is True for k in VALIDATION_CHECKS) or \
            rec.get("run") != run or rec.get("checkpoint_sha256") != checkpoint_sha256 or \
            not valid_sha256(rec.get("training_contract_sha256")) or \
            (training_contract_sha256 is not None and rec.get("training_contract_sha256") != training_contract_sha256) or \
            (input_spec is not None and rec.get("input_spec_sha256") != input_spec_digest(input_spec)):
        return None
    return {"run": run, "checkpoint_sha256": checkpoint_sha256, "pass": True, "at": rec.get("at"),
            "code": rec.get("code"), "record": str(f), "training_contract_sha256": rec["training_contract_sha256"],
            "input_spec_sha256": rec.get("input_spec_sha256"),
            "checks": {k: {kk: vv for kk, vv in checks[k].items() if kk in ("pass", "compared", "equal", "windows")}
                       for k in VALIDATION_CHECKS}}


# DynaMo: its restricted status (e1 LayerNorm-spread FAILED, not waived) alone does not block execution (user decisions
# 2026-11-12 Task 3 and ~16:00); scope = the new-setting DynaMo policies (legacy DynaMo is a legacy frozen-encoder
# policy). Every other check stays, including the deployment validation. Validator allowlist:
# config.RESTRICTED_EXECUTION_SCOPES.
RESTRICTED_EXECUTION_DECISIONS = {
    "dynamo": {"decided_by": "user", "at": "2026-11-12 (Task 3) and ~16:00 KST (relayed by the coordinator)",
               "text": "DynaMo must support actual execution on the Robot PC, not just dry-run testing. The exception "
                       "stops DynaMo being blocked solely by representation quality / the failed LayerNorm-spread "
                       "criterion; extended to new-setting (a10nq) DynaMo policies; keep the diagnostic results and "
                       "warnings and every other validation. (the e1 LayerNorm-spread criterion stays FAILED, not "
                       "waived)",
               "scope": {"condition": "dynamo", "fql_setting": "a10nq"},
               "acknowledged_failures": ["e1_layernorm_spread"]}}
RESTRICTED_EXECUTION_WARNINGS = {
    "dynamo": "WARNING: execution-eligible by user decision 2026-11-12, NOT a validated-performance claim; DIAG "
              "assessed the legacy DynaMo policies NOT healthy enough for an execution test as-is (DIAG-REPORT.md 4). "
              "Known offline findings (legacy setting): e1 LayerNorm-spread FAIL (0.33-0.36 < 0.5); one-step actor "
              "24-30 mm off the demonstrations with an optimistic critic (realworld-ops/diag/DIAG-REPORT.md 3, 7, 8); "
              "representation weaker than ReViWo/SinCro, view-specific, most sensitive to the camera colour start-up "
              "cast (realworld-ops/diag/DYNAMO-REPRESENTATION.md)"}


def restricted_grant(status: str, condition: str | None, setting: str | None, checkpoint_sha256: str | None,
                     failed_criteria=None) -> dict | None:
    """The user execution decision covering a ``restricted`` entry (bound to its checkpoint), or None: only for a
    (condition, setting) pair in the validator's allowlist with a recorded decision whose scope names it, a valid
    checkpoint sha256, and published FAILED criteria that are non-empty and all acknowledged (the validator's rule).
    ``setting``: the run id's FQL setting (None = legacy). The deployment validation is checked by ``usage_of``."""
    from ops.policy_serving.config import LEGACY_SETTING, RESTRICTED_EXECUTION_SCOPES, valid_sha256

    decision = RESTRICTED_EXECUTION_DECISIONS.get(condition)
    setting = setting or LEGACY_SETTING
    if status != "restricted" or decision is None or (condition, setting) not in RESTRICTED_EXECUTION_SCOPES or \
            decision["scope"] != {"condition": condition, "fql_setting": setting} or \
            not valid_sha256(checkpoint_sha256) or not failed_criteria or \
            not set(failed_criteria) <= set(decision["acknowledged_failures"]):
        return None
    return {**copy.deepcopy(decision), "checkpoint_sha256": checkpoint_sha256}


def usage_of(status: str, condition: str | None = None, setting: str | None = None, *,
             checkpoint_sha256: str | None = None, failed_criteria=None, validation: dict | None = None) -> dict:
    """Usage restriction of a staged entry (user decisions 2026-11-10 21:10, 2026-11-12 Task 3 and ~16:00); a missing
    field means not eligible for every consumer (fail closed); execution on the robot PC still needs its manual mode.
    * quarantined: not usable; anything but deployable / restricted: dry run / replay only;
    * e2e_cnn (legacy E2E setting) deployable: execution-eligible (``EXECUTION_DECISION``);
    * legacy frozen-encoder (every other condition, no FQL-setting prefix): NOT eligible, never re-enabled;
    * a100: NOT eligible;
    * a10nq deployable: eligible (``AUTO_NQ_DECISION`` bound to the checkpoint) once ``validation`` (the passed
      deployment-validation summary, ``validation_record``) is given; before that dry run / replay only;
    * a10nq restricted (e1 LN spread only, criterion kept FAILED): eligible only with the scoped user decision of its
      condition (``restricted_grant``: DynaMo) AND the deployment validation; its warning travels in usage_reasons."""
    if status == "quarantined":
        return {"usage": "none", "execution_eligible": False,
                "usage_reasons": ["quarantined: refused by the server unless --allow-quarantined (pipeline tests)",
                                  NOT_ELIGIBLE]}
    if status not in ("deployable", "restricted"):
        return {"usage": "dry_run_replay_only", "execution_eligible": False, "usage_reasons": [NOT_ELIGIBLE]}
    lead = [C.LN_SPREAD_DECISION] if status == "restricted" else []
    held = exclusions(condition, setting)
    if held:
        return {"usage": "dry_run_replay_only", "execution_eligible": False, "execution_decision": None,
                "usage_reasons": lead + held + [NOT_ELIGIBLE]}
    validated = validation is not None and validation.get("checkpoint_sha256") == checkpoint_sha256 and \
        validation.get("pass") is True
    if status == "deployable" and condition == "e2e_cnn" and setting in (None, "legacy"):
        return {"usage": "execution", "execution_eligible": True, "execution_decision": dict(EXECUTION_DECISION),
                "usage_reasons": [f"user decision {EXECUTION_DECISION['at']} (E2E CNN kept eligible by the user "
                                  "decision 2026-11-12 ~16:00): deployable (all criteria passed) -> execution-eligible; "
                                  "robot-PC manual execution mode still required"]}
    grant = restricted_grant(status, condition, setting, checkpoint_sha256, failed_criteria) \
        if status == "restricted" else None
    if status == "restricted" and grant is None:
        return {"usage": "dry_run_replay_only", "execution_eligible": False, "execution_decision": None,
                "usage_reasons": lead + [NOT_ELIGIBLE]}
    if not validated:
        return {"usage": "dry_run_replay_only", "execution_eligible": False, "execution_decision": None,
                "usage_reasons": lead + [AWAITING_VALIDATION, NOT_ELIGIBLE]}
    if grant is not None:
        return {"usage": "execution", "execution_eligible": True, "execution_decision": grant,
                "deploy_validation": validation,
                "usage_reasons": [C.LN_SPREAD_DECISION, RESTRICTED_EXECUTION_WARNINGS[condition],
                                  f"user decision {grant['at']}: restricted {condition} ({setting}) -> "
                                  f"execution-eligible with the FAILED criteria {grant['acknowledged_failures']} "
                                  "acknowledged and the deployment validation passed; robot-PC manual execution mode "
                                  "still required"]}
    return {"usage": "execution", "execution_eligible": True,
            "execution_decision": {**AUTO_NQ_DECISION, "checkpoint_sha256": checkpoint_sha256},
            "deploy_validation": validation,
            "usage_reasons": [f"user decision {AUTO_NQ_DECISION['at']}: new-setting (a10nq) FQL policy, deployable and "
                              "deployment-validated -> execution-eligible automatically; robot-PC manual execution "
                              "mode still required"]}


def local_input_spec(src: Sources, row: dict, sidecar_dir: Path) -> dict:
    from ops.policy_serving.server import describe_entry

    return describe_entry({"checkpoint": str(src.runs / row["run"] / "policy.pt"),
                           "training_contract": str(sidecar_dir / f"{row['run']}.deploy_contract.json")})


def build_catalog(entries: dict[str, dict], *, aliases: dict[str, str], criteria_sha256: str | None) -> dict:
    from ops.policy_serving.config import CATALOG_SCHEMA

    return {"schema": CATALOG_SCHEMA, "criteria_sha256": criteria_sha256,
            "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "policies": dict(sorted(entries.items())), "aliases": dict(sorted(aliases.items()))}


BCFLOW_REASON = ("EXPERIMENTAL bc_flow actor (DIAG 2026-11-11, realworld-ops/diag/bcflow): the checkpoint's BC flow "
                 "instead of its one-step policy; criteria a-f were evaluated for the one-step actor; dry run / replay "
                 "only unless a user decision names bc_flow")


def bcflow_variant(base_id: str, base: dict, *, decision: dict | None = None) -> tuple[str, dict]:
    """``(<base_id>+bcflow, entry)``: the same staged files served with ``actor: bc_flow`` under its own id (never
    replacing ``base_id``). Not eligible unless ``decision`` (a user decision whose text names bc_flow) is given AND
    the base entry is deployable; a non-servable base gives a quarantined variant (withdrawn with it)."""
    from ops.policy_serving.config import BCFLOW_FLAG, BCFLOW_SUFFIX, SERVABLE_STATUSES

    if base.get("actor", "onestep") != "onestep" or base_id.endswith(BCFLOW_SUFFIX):
        raise ValueError(f"{base_id}: a bc_flow variant derives from a one-step entry")
    keep = ("checkpoint", "sha256", "export", "training_contract", "dataset", "condition", "task", "seed", "run",
            "input_spec", "criteria")
    entry = {k: base[k] for k in keep if k in base}
    entry.update(actor="bc_flow", flags=sorted({*base.get("flags", []), BCFLOW_FLAG}),
                 note=f"experimental BC-flow actor of {base_id}")
    if base["status"] not in SERVABLE_STATUSES:
        entry.update(status="quarantined", reasons=[f"base entry {base_id} is {base['status']}", BCFLOW_REASON],
                     usage="none", execution_eligible=False, usage_reasons=[NOT_ELIGIBLE])
    else:
        entry.update(status=base["status"], reasons=list(base.get("reasons", [])) or ([BCFLOW_REASON]
                     if base["status"] != "deployable" else []))
        # exclusions are re-derived from the base's condition and run id, so an existing entry without the key (or an
        # older catalog) cannot carry a 3D-MVP / new-setting base to eligibility (Codex review of 618ed2f)
        setting = fql_setting(str(base.get("run") or ""))
        excluded = exclusions(base.get("condition"), setting)
        if setting == "a10nq":  # the auto-eligibility covers the validated ONE-STEP policy, not its experimental BC flow
            excluded = [AWAITING_VALIDATION.split(":")[0] + ": its +bcflow variant has no deployment validation of its "
                        "own (dry run / replay only)"]
        blocked = bool(excluded) or base.get("execution_eligible") is not True
        if decision is not None and base["status"] == "deployable" and not blocked and \
                decision.get("checkpoint_sha256", base["sha256"]) == base["sha256"]:
            if decision.get("decided_by") != "user" or "bc_flow" not in str(decision.get("text", "")):
                raise ValueError("a bc_flow execution decision is a user decision whose text names bc_flow")
            # the grant is bound to these weights: a re-staged base (other checkpoint) drops it
            entry.update(usage="execution", execution_eligible=True,
                         execution_decision={**decision, "checkpoint_sha256": base["sha256"]},
                         usage_reasons=[f"user decision {decision.get('at')}: experimental bc_flow actor -> "
                                        "execution-eligible; robot-PC manual execution mode still required"])
        else:  # an ineligible base (3D-MVP, new FQL setting, ...) keeps its variant ineligible with its reasons
            held = (excluded or [r for r in base.get("usage_reasons", []) if r != NOT_ELIGIBLE]) \
                if blocked and base["status"] == "deployable" else []
            entry.update(usage="dry_run_replay_only", execution_eligible=False, usage_reasons=[BCFLOW_REASON, *held])
    return base_id + BCFLOW_SUFFIX, entry


def sync_bcflow_variants(entries: dict[str, dict]) -> dict[str, dict]:
    """Re-derive every ``+bcflow`` entry from its CURRENT base (files, labels, withdrawal follow the base; a granted
    bc_flow decision is kept only while the base stays deployable with the SAME checkpoint sha256, and is dropped for
    good once the variant is withdrawn). Returns a new dict."""
    from ops.policy_serving.config import BCFLOW_SUFFIX

    out = dict(entries)
    for pid, e in entries.items():
        if pid.endswith(BCFLOW_SUFFIX):
            base_id = pid[:-len(BCFLOW_SUFFIX)]
            base = entries.get(base_id) or {"status": "withdrawn", **{k: e[k] for k in ("checkpoint", "sha256",
                                                                                           "condition", "task")}}
            decision = e.get("execution_decision") if e.get("execution_eligible") is True else None
            out[pid] = bcflow_variant(base_id, base, decision=decision)[1]
    return out


def legacy_aliases(entries: dict[str, dict]) -> dict[str, str]:
    """The pre-catalog IDs ``<condition>-<task>`` -> the staged ``-s0`` entry (a quarantined one is refused by the
    server with its reasons instead of an unknown-id error)."""
    return {pid[:-3]: pid for pid in entries if pid.endswith("-s0")}


# ------------------------------------------------------------------------------------------------ host staging
@dataclass
class Host:
    """The inference host: an SSH alias with an existing master, the host checkpoint root (under its home) and the
    server container with its RAM checkpoint root."""
    ssh: str = "irsl"
    root: str = "shkim/serving/checkpoints"
    stage_dir: str = "shkim/serving/.staging-checkpoints"
    container: str = "shkim_rw"
    ram_root: str = "/ram/serving/checkpoints"
    server_match: str = "^/opt/envs/s4d/bin/python -I -B ops/policy_serving/server.py serve"
    bwlimit: str = "10m"

    def run(self, cmd: str, *, stdin: str | None = None, check: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(["ssh", "-o", "BatchMode=yes", self.ssh, cmd], input=stdin, capture_output=True,
                              text=True, check=check)


def staging_files(src: Sources, row: dict, sidecar_dir: Path) -> dict[str, Path]:
    """Files a policy needs on the inference host (relative path -> local source); never datasets."""
    files = {row["policy"]["path"]: src.runs / row["run"] / "policy.pt",
             f"contracts/{row['run']}.deploy_contract.json": sidecar_dir / f"{row['run']}.deploy_contract.json"}
    for name in ("config.yaml", "completion.json", "provenance.json"):
        p = src.runs / row["run"] / name
        if p.is_file():
            files[f"runs/{row['run']}/{name}"] = p
    exp = row.get("export")
    if exp:
        files[exp["path"]] = src.data_root / exp["path"]
        for name in ("completion.json", "provenance.json", "config.yaml"):
            p = src.data_root / Path(exp["path"]).parent / name
            if p.is_file():
                files[f"{Path(exp['path']).parent}/{name}"] = p
    return files


def _verified(host: Host, where: str, shas: dict[str, str]) -> set[str]:
    """Relative paths already present under ``where`` with the expected sha256 (host root or ``container:path``)."""
    sums = "".join(f"{h}  {rel}\n" for rel, h in shas.items())
    cmd = f"cd {where} && sha256sum -c - 2>/dev/null"
    if where.startswith("container:"):
        cmd = f"docker exec -i {host.container} sh -c {shlex.quote(f'cd {where[10:]} && sha256sum -c - 2>/dev/null')}"
    r = host.run(f"mkdir -p {where} 2>/dev/null; {cmd}" if not where.startswith("container:") else cmd,
                 stdin=sums, check=False)
    return {line[:-4] for line in r.stdout.splitlines() if line.endswith(": OK")}


def stage_files(host: Host, files: dict[str, Path], *, execute: bool, log=print) -> dict[str, str]:
    """Upload what the host root lacks (resumable rsync into the staging dir), verify sha256 on the host, publish
    atomically (an existing final file with another sha256 is REFUSED, never overwritten), then copy what the
    container RAM root lacks and verify every file there. Returns {relative path: sha256}. Dry run: only the plan."""
    shas = {rel: sha256_file(p) for rel, p in sorted(files.items())}
    log(f"[stage] {len(shas)} files, {sum(p.stat().st_size for p in files.values()) / 1e6:.1f} MB"
        + ("" if execute else " (dry run)"))
    if not execute:
        return shas
    sums = "".join(f"{h}  {rel}\n" for rel, h in shas.items())
    have = _verified(host, host.root, shas)
    todo = {rel: h for rel, h in shas.items() if rel not in have}
    log(f"[stage] host root has {len(have)} of {len(shas)}; uploading {len(todo)} "
        f"({sum(files[r].stat().st_size for r in todo) / 1e6:.1f} MB)")
    if todo:
        host.run(f"mkdir -p {host.stage_dir}")
        with tempfile.TemporaryDirectory() as tmp:
            for rel in todo:  # one tree of symlinks -> one rsync with relative paths
                link = Path(tmp) / rel
                link.parent.mkdir(parents=True, exist_ok=True)
                link.symlink_to(files[rel].resolve())
            subprocess.run(["rsync", "-aL", "--partial-dir=.rsync-partial", f"--bwlimit={host.bwlimit}",
                            "-e", "ssh -o BatchMode=yes", f"{tmp}/", f"{host.ssh}:{host.stage_dir}/"], check=True)
        host.run(f"cd {host.stage_dir} && sha256sum --quiet -c -", stdin="".join(f"{h}  {r}\n" for r, h in
                                                                               todo.items()))
        publish = []
        for rel, h in todo.items():
            s, d = shlex.quote(f"{host.stage_dir}/{rel}"), shlex.quote(f"{host.root}/{rel}")
            publish.append(f"mkdir -p $(dirname {d}); if [ -e {d} ]; then [ \"$(sha256sum < {d} | cut -d' ' -f1)\" = "
                           f"{h} ] || {{ echo REFUSE {rel}; exit 3; }}; rm -f {s}; else mv {s} {d}; fi; chmod a-w {d}")
        host.run("set -eu\n" + "\n".join(publish))
    host.run(f"cd {host.root} && sha256sum --quiet -c -", stdin=sums)
    in_ram = _verified(host, f"container:{host.ram_root}", shas)
    copy = [rel for rel in shas if rel not in in_ram]
    if copy:
        rels = " ".join(shlex.quote(r) for r in copy)
        host.run(f"tar -C {host.root} -cf - {rels} | docker exec -i {host.container} tar -C {host.ram_root} -xf -")
    host.run(f"docker exec -i {host.container} sh -c 'cd {host.ram_root} && sha256sum --quiet -c -'", stdin=sums)
    log(f"[stage] verified {len(shas)} files on {host.ssh}:{host.root} and {host.container}:{host.ram_root} "
        f"({len(copy)} copied into RAM)")
    return shas


def _catalog_text(host: Host, where: str) -> str | None:
    """The published catalog text on the host root or (``container``) in the container RAM root; None if absent."""
    path = f"{host.ram_root}/{CATALOG_NAME}" if where == "container" else f"{host.root}/{CATALOG_NAME}"
    cmd = f"docker exec {host.container} cat {path}" if where == "container" else f"cat {path}"
    r = host.run(cmd, check=False)
    return r.stdout if r.returncode == 0 and r.stdout.strip() else None


def read_host_catalog(host: Host) -> dict:
    text = _catalog_text(host, "host")
    return json.loads(text) if text else {"policies": {}, "aliases": {}}


def catalog_copies_agree(host: Host) -> bool:
    """The host and container copies are byte-identical (a publication interrupted between them is not)."""
    return _catalog_text(host, "host") == _catalog_text(host, "container")


def write_host_catalog(host: Host, catalog: dict, *, execute: bool, log=print) -> None:
    """Atomic replace on the host root and in the container RAM root (complete write to a unique temporary name,
    then rename; the container copy is verified byte-identical)."""
    import uuid

    from ops.policy_serving.config import validate_catalog

    validate_catalog(catalog)
    text = json.dumps(catalog, indent=1) + "\n"
    if not execute:
        log(f"[catalog] would write {len(catalog['policies'])} policies, {len(catalog['aliases'])} aliases (dry run)")
        return
    tmp = f".{CATALOG_NAME}.{uuid.uuid4().hex}.part"
    host.run(f"chmod u+w {host.root} && cat > {host.root}/{tmp} && mv {host.root}/{tmp} {host.root}/{CATALOG_NAME}",
             stdin=text)
    host.run(f"docker exec -i {host.container} sh -c 'cat > {host.ram_root}/{tmp} && "
             f"mv {host.ram_root}/{tmp} {host.ram_root}/{CATALOG_NAME}'", stdin=text)
    if _catalog_text(host, "container") != text:
        raise RuntimeError("container catalog differs from the published host catalog after writing")
    log(f"[catalog] {len(catalog['policies'])} policies written (sha256 {hashlib.sha256(text.encode()).hexdigest()})")


RELOAD_SCRIPT = """for p in $(pgrep -f {match}); do
  m=$(sed -n 's/^SigCgt:[[:space:]]*//p' /proc/$p/status)
  if [ $(( 0x$m & 1 )) -eq 1 ]; then kill -HUP $p && echo "reloaded $p"; else echo "skip $p (no SIGHUP handler)"; fi
done"""


def signal_reload(host: Host, *, execute: bool, log=print) -> None:
    """SIGHUP to the running server (catalog reload, no restart) ONLY if it catches SIGHUP (``/proc/<pid>/status``
    SigCgt bit 1): an older server without the handler would be terminated by the default action."""
    script = RELOAD_SCRIPT.format(match=shlex.quote(host.server_match))
    cmd = f"docker exec {host.container} sh -c {shlex.quote(script)}"
    if not execute:
        log(f"[reload] would run: {cmd}")
        return
    r = host.run(cmd, check=False)
    log("[reload] " + (r.stdout.strip().replace("\n", "; ") or "no running server matched (nothing reloaded)"))


# ------------------------------------------------------------------------------------------------ reports
def markdown(inv: dict, *, criteria_sha256: str | None) -> str:
    L = [f"# Real-world policy inventory ({datetime.datetime.now().astimezone():%Y-%m-%d %H:%M %Z})", "",
         f"Generated by `ops/policy_serving/staging.py inventory` (criteria POL-CRITERIA.md sha256 `{criteria_sha256}`)."
         " Registry/queue read as data; run files read-only.", "",
         "## (A) Encoder pretraining", "",
         "| run | condition | task | registry | budget | completed | complete (a) | encoder.pt sha256 | note |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in inv["pretraining"]:
        c, e = r["completed"] or {}, r["encoder"]
        done = c.get("epochs_completed", c.get("steps"))
        L.append(f"| {r['run']} | {r['condition']} | {r['task']} | {r['registry_state']} | "
                 f"{r['budget'].get('budget')} {r['budget'].get('kind') or ''} | {c.get('status', '-')} {done or ''} | "
                 f"{'yes' if r['a']['pass'] else 'no'} | {(e.get('sha256') or '-')[:12]}{'' if e['local'] else ' (remote)'}"
                 f" | {r['superseded'] or r['a'].get('reason', '')} |")
    L += ["", "## (B) FQL policies", "",
          "| policy id | run | registry | steps / budget | policy.pt sha256 | class | 5090 catalog | reasons |",
          "|---|---|---|---|---|---|---|---|"]
    for r in inv["fql"]:
        c = r["completed"] or {}
        cls = r.get("class", {})
        why = ", ".join(cls.get("flags", [])) + ("; " if cls.get("flags") and cls.get("reasons") else "") + \
            "; ".join(cls.get("reasons", []))
        L.append(f"| {r['policy_id']} | {r['run']} | {r['registry_state']} | {c.get('steps', '-')} / "
                 f"{r['budget_steps']} | {(r['policy'].get('sha256') or '-')[:12]}"
                 f"{'' if r['policy']['local'] else ' (remote)'} | **{cls.get('status', '?')}** | "
                 f"{r.get('host_catalog') or '-'} | {why} |")
    L += ["", "## (C) Classes", ""]
    for status in ("deployable", "restricted", "incomplete", "quarantined", "superseded"):
        ids = [r["policy_id"] if status != "superseded" else r["run"] for r in inv["fql"]
               if r.get("class", {}).get("status") == status]
        L.append(f"* **{status}** ({len(ids)}): " + (", ".join(ids) if ids else "none"))
    L.append("* FQL settings (FQLNQ 2026-11-11): rwnq-* rows are served as `<id>-a10nq` (flag fql_setting_a10nq) next to "
             "the alpha-10 ids; rwa100-* comparison rows are staged only with --allow-comparison")
    L.append("* restricted = every criterion passes except the e1 LayerNorm-spread gate (FAILED, not waived): served "
             "for dry run / replay only (amendment A2). No catalog entry is execution-eligible (Directive I).")
    L += ["", "## Check values (candidates)", "",
          "| policy id | c export chain | d sidecar | e1 offset / LN std | e2 D_vis/S, D_vis/D_noise | f final q / late "
          "median / peak |", "|---|---|---|---|---|---|"]
    for r in inv["fql"]:
        k = r.get("checks") or {}
        if "c" not in k:
            continue
        e1, e2, f = k.get("e1") or {}, k.get("e2") or {}, k.get("f") or {}
        e1s = "-" if not e1 or e1.get("pass") is None and "offset_to_signal" not in e1 else \
            f"{e1['offset_to_signal']:.3g} / {e1['ln_unit_std_raw']:.3g}"
        e2s = e2.get("reason", "-") if e2.get("pass") is None else \
            f"{_fmt(e2['d_vis_over_spread'])}, {_fmt(e2['d_vis_over_noise'])} ({'pass' if e2['pass'] else 'FAIL'})"
        peak = f.get("peak_q_mean") or [None, None]
        L.append(f"| {r['policy_id']} | {'pass' if k['c']['pass'] else 'FAIL'} | "
                 f"{'pass' if k['d']['pass'] else 'FAIL'} | {e1s} | {e2s} | "
                 f"{_fmt(f.get('final_q_mean'))} / {_fmt(f.get('late_q_mean_median'))} / {_fmt(peak[1])}@{peak[0]} "
                 f"({'pass' if f.get('pass') else 'FAIL'}) |")
    return "\n".join(L) + "\n"


def _fmt(x) -> str:
    return "-" if x is None else f"{x:.3g}"


# ------------------------------------------------------------------------------------------------ CLI
def _sources(a) -> Sources:
    return Sources(a.data_root, a.data_root / "experiments/registry.jsonl", a.data_root / "experiments/queue.yaml",
                   a.splits, a.raw_root, a.rig_dir)


def _evaluated(a, runs: list[str] | None = None) -> dict:
    src = _sources(a)
    inv = inventory(src)
    pre = {r["run"]: r for r in inv["pretraining"]}
    rows = [r for r in inv["fql"] if runs is None or r["run"] in runs]
    checks = evaluate(src, rows, pre, sidecar_dir=a.sidecar_dir, probe=not a.no_probe,
                      log=lambda m: print(m, file=sys.stderr, flush=True)) if a.evaluate else {}
    classify_rows(inv["fql"], checks)
    return inv


def cmd_inventory(a) -> int:
    if a.from_json:  # refresh the annotations of an already evaluated inventory (no re-evaluation)
        inv = json.loads(a.from_json.read_text())
        for r in inv["fql"]:
            r.pop("host_catalog", None)
    else:
        inv = _evaluated(a)
    if a.host_catalog:  # read-only: which runs the inference host's catalog serves (id + status)
        cat = read_host_catalog(Host(ssh=a.ssh)).get("policies") or {}
        for r in inv["fql"]:
            e = cat.get(r["policy_id"])
            if e is not None and e.get("run") == r["run"]:
                r["host_catalog"] = f"{r['policy_id']} ({e['status']}, {e.get('usage', 'dry_run_replay_only')})"
    crit = inv.get("criteria_sha256") if a.from_json else (sha256_file(a.criteria) if a.criteria.is_file() else None)
    inv["criteria_sha256"] = crit
    if not a.from_json:
        inv["staging_rule"] = STAGING_RULE
    if a.out_json:
        a.out_json.write_text(json.dumps(inv, indent=1, default=str) + "\n")
    text = markdown(inv, criteria_sha256=crit)
    if a.out_md:
        a.out_md.write_text(text)
    else:
        print(text)
    return 0


def _identity_errors(src: Sources, row: dict, sidecar_dir: Path) -> list[str]:
    """Every staged input must be the one the criteria were evaluated on: policy.pt, the deploy-contract sidecar and
    the encoder export (sha256 recorded by the evaluation)."""
    errs = []
    if sha256_file(src.runs / row["run"] / "policy.pt") != row["policy"]["sha256"]:
        errs.append("policy.pt changed since the evaluation")
    d = (row.get("checks") or {}).get("d") or {}
    sidecar = sidecar_dir / f"{row['run']}.deploy_contract.json"
    if not d.get("sha256") or not sidecar.is_file() or sha256_file(sidecar) != d["sha256"]:
        errs.append("deploy-contract sidecar missing or changed since the evaluation")
    exp = row.get("export")
    if exp and (not (src.data_root / exp["path"]).is_file() or sha256_file(src.data_root / exp["path"]) != exp["sha256"]):
        errs.append("encoder export missing or changed since the evaluation")
    return errs


def _evaluated_shas(row: dict, sidecar_dir: Path) -> dict[str, str]:
    """Relative staged path -> the sha256 the criteria were evaluated on (policy, sidecar, export)."""
    out = {row["policy"]["path"]: row["policy"]["sha256"],
           f"contracts/{row['run']}.deploy_contract.json": (row.get("checks") or {}).get("d", {}).get("sha256")}
    if row.get("export"):
        out[row["export"]["path"]] = row["export"]["sha256"]
    return out


def withdrawn_entry(entry: dict, row: dict) -> dict:
    """A staged entry whose run no longer passes: quarantined with the re-evaluation's reasons (never left
    selectable); the staged files stay (immutable) but the server refuses it."""
    entry = {k: v for k, v in entry.items() if k not in ("execution_decision", "deploy_validation")}
    return {**entry, "status": "quarantined", "usage": "none", "execution_eligible": False, "flags": [],
            "criteria": C.criteria_summary(row["condition"], row.get("checks") or {}),
            "usage_reasons": ["withdrawn by re-evaluation"],
            "reasons": [f"re-evaluation {datetime.date.today()}: {row['class']['status']}"] + row["class"]["reasons"]}


def cmd_stage(a, *, hook: bool = False) -> int:
    """``stage``: stage rows the inventory classifies deployable / restricted (``hook``: exactly the given completed
    runs, re-evaluated; a run that does not pass is reported, NOT staged, and an existing catalog entry of that run is
    withdrawn = quarantined) and merge them into the host catalog under a local lock; then reload."""
    import fcntl

    comparison = [r for r in a.runs if fql_setting(r) in COMPARISON_SETTINGS]
    if comparison and not getattr(a, "allow_comparison", False):  # before any evaluation, host read or lock
        print(json.dumps({"error": f"comparison-baseline runs {comparison} are staged only with --allow-comparison "
                                   "(FQLNQ 2026-11-11: never staged unless asked)"}))
        return 2
    if getattr(a, "inventory", None):  # a just-evaluated inventory (stage only; the hook always re-evaluates)
        inv = json.loads(a.inventory.read_text())
        if inv.get("criteria_sha256") != (sha256_file(a.criteria) if a.criteria.is_file() else None):
            print(json.dumps({"error": "inventory was evaluated under other criteria"}))
            return 2
        if inv.get("staging_rule") != STAGING_RULE:
            print(json.dumps({"error": f"inventory was evaluated under staging rule {inv.get('staging_rule')!r}, not "
                                       f"{STAGING_RULE!r}: re-evaluate it (policy ids depend on the FQL setting)"}))
            return 2
    else:
        a.evaluate = True
        inv = _evaluated(a, a.runs)
    rows = {r["run"]: r for r in inv["fql"]}
    missing = [r for r in a.runs if r not in rows]
    if missing:
        print(json.dumps({"error": f"unknown FQL runs {missing}"}))
        return 2
    wrong = {r: rows[r].get("policy_id") for r in a.runs if rows[r].get("policy_id") != policy_id(
        rows[r]["condition"], rows[r]["task"], int(rows[r]["seed"]), fql_setting(r))}
    if wrong:  # before any catalog mutation: never let a run take another setting's id
        print(json.dumps({"error": f"policy ids differ from the FQL-setting rule (re-evaluate): {wrong}"}))
        return 2

    src = _sources(a)
    host = Host(ssh=a.ssh, bwlimit=a.bwlimit)
    a.lock_file.parent.mkdir(parents=True, exist_ok=True)
    log = lambda m: print(m, file=sys.stderr, flush=True)  # noqa: E731
    with open(a.lock_file, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)  # one catalog read-merge-write transaction at a time (concurrent hooks)
        catalog = read_host_catalog(host) if a.execute or a.read_host else {"policies": {}, "aliases": {}}
        crit = sha256_file(a.criteria) if a.criteria.is_file() else None

        def publish(policies: dict) -> None:
            policies.update(sync_bcflow_variants(policies))  # in place: later merge phases see the derived variants
            aliases = {k: v for k, v in {**(catalog.get("aliases") or {}), **legacy_aliases(policies)}.items()
                       if v in policies}
            unchanged = policies == (catalog.get("policies") or {}) and aliases == (catalog.get("aliases") or {})
            if unchanged and (not a.execute or catalog_copies_agree(host)):
                log("[catalog] unchanged: nothing written, no reload (idempotent re-run)")
            elif policies:
                write_host_catalog(host, build_catalog(policies, aliases=aliases, criteria_sha256=crit),
                                   execute=a.execute, log=log)
                signal_reload(host, execute=a.execute, log=log)
                catalog.update(policies=dict(policies), aliases=aliases)

        def labelled(row) -> bool:
            return row["class"]["status"] == "quarantined" and getattr(a, "label_quarantined", False) and \
                row["policy"]["local"] and (a.sidecar_dir / f"{row['run']}.deploy_contract.json").is_file()

        entries = dict(catalog.get("policies") or {})
        report, files, evaluated = {}, {}, {}
        for run in a.runs:  # 1. withdrawals, PUBLISHED FIRST: nothing below (checks, uploads) can hold them back
            row = rows[run]
            status = row["class"]["status"]
            report[run] = {"policy_id": row["policy_id"], "status": status, "reasons": row["class"]["reasons"]}
            if status not in ("deployable", "restricted") and not labelled(row) and \
                    (entries.get(row["policy_id"]) or {}).get("run") == run and \
                    entries[row["policy_id"]]["status"] != "quarantined":
                entries[row["policy_id"]] = withdrawn_entry(entries[row["policy_id"]], row)
                report[run]["catalog_status"] = "withdrawn (quarantined)"
        if entries != (catalog.get("policies") or {}):
            publish(entries)
        try:  # 2. new entries: identity checks under the lock, preparation, upload, hash comparison
            changed = {r: errs for r in a.runs if rows[r]["class"]["status"] in ("deployable", "restricted")
                       and (errs := _identity_errors(src, rows[r], a.sidecar_dir))}
            if changed:
                raise RuntimeError(f"inputs changed since the evaluation (re-evaluate): {changed}")
            for run in a.runs:
                row = rows[run]
                if row["class"]["status"] in ("deployable", "restricted") or labelled(row):
                    files.update(staging_files(src, row, a.sidecar_dir))
                    evaluated.update(_evaluated_shas(row, a.sidecar_dir))
                    spec = local_input_spec(src, row, a.sidecar_dir)
                    entries[row["policy_id"]] = catalog_entry(row, spec, validation_record(
                        getattr(a, "validation_dir", None), run, row["policy"]["sha256"],
                        training_contract_sha256=((row.get("checks") or {}).get("d") or {}).get("sha256"),
                        input_spec=spec))
                    report[run]["staged"] = bool(a.execute)
                    report[run]["catalog_status"] = row["class"]["status"]
            if files:
                uploaded = stage_files(host, files, execute=a.execute, log=log)
                bad = sorted(rel for rel, h in evaluated.items() if uploaded.get(rel) != h)
                if bad:
                    raise RuntimeError(f"staged files differ from the evaluated inputs: {bad}")
        except Exception as exc:  # the withdrawals are already published; no new entry is
            print(json.dumps({"error": f"staging failed: {type(exc).__name__}: {exc}", "runs": report},
                             default=str))
            return 2
        publish(entries)
    print(json.dumps({"execute": bool(a.execute), "runs": report}, indent=1, default=str))
    return 0 if all(r["status"] in ("deployable", "restricted") for r in report.values()) or not hook else 3


USAGE_KEYS = ("usage", "execution_eligible", "execution_decision", "usage_reasons", "deploy_validation")


def relabel_entry(pid: str, entry: dict, validation_dir: Path | None, sidecar_dir: Path | None = None) -> dict:
    """An existing one-step catalog entry with its usage re-derived under the CURRENT rules (status, criteria,
    artefacts, reasons unchanged except the restricted-class decision line, which follows criteria.LN_SPREAD_DECISION);
    a10nq entries take their deployment-validation record from ``validation_dir``. Entries that are not deployable /
    restricted (quarantined, withdrawn) are returned unchanged."""
    if entry.get("status") not in ("deployable", "restricted") or entry.get("actor", "onestep") != "onestep":
        return dict(entry)
    setting = fql_setting(str(entry.get("run") or ""))
    failed = {k for k, v in (entry.get("criteria") or {}).items() if isinstance(v, dict) and v.get("pass") is False}
    sidecar = None if sidecar_dir is None else Path(sidecar_dir) / Path(str(entry.get("training_contract"))).name
    contract_sha = sha256_file(sidecar) if sidecar is not None and sidecar.is_file() else None
    # an a10nq entry is relabelled eligible only against the CURRENT staged contract (local sidecar copy) and spec
    validation = validation_record(validation_dir, entry["run"], entry["sha256"], input_spec=entry.get("input_spec"),
                                   training_contract_sha256=contract_sha) \
        if setting == "a10nq" and contract_sha is not None else None
    u = usage_of(entry["status"], entry.get("condition"), setting, checkpoint_sha256=entry.get("sha256"),
                 failed_criteria=failed, validation=validation)
    if entry["status"] == "restricted" and u["execution_eligible"] and \
            (entry.get("input_spec") or {}).get("condition") != entry.get("condition"):
        u = usage_of(entry["status"], entry.get("condition"), setting)  # a restricted grant needs the bound spec
    out = {k: v for k, v in entry.items() if k not in USAGE_KEYS}
    out.update(u)
    if entry["status"] == "restricted":
        out["reasons"] = [C.LN_SPREAD_DECISION if (r.startswith("user decision 2026-11-") and "LayerNorm-spread" in r)
                          else r for r in entry.get("reasons", [])]
    return out


def relabel_catalog(catalog: dict, validation_dir: Path | None, sidecar_dir: Path | None = None) -> dict:
    """The catalog with every one-step entry relabelled (``relabel_entry``) and the bc_flow variants re-derived from
    their bases; aliases, criteria hash and artefacts unchanged."""
    entries = {pid: relabel_entry(pid, e, validation_dir, sidecar_dir)
               for pid, e in (catalog.get("policies") or {}).items()}
    entries = sync_bcflow_variants(entries)
    aliases = {k: v for k, v in (catalog.get("aliases") or {}).items() if v in entries}
    return build_catalog(entries, aliases=aliases, criteria_sha256=catalog.get("criteria_sha256"))


def cmd_relabel(a) -> int:
    """``relabel``: re-derive the usage / execution eligibility of every catalog entry under the current rules (no
    criteria re-evaluation, no file staging): ``--from-json/--out-json`` transforms a local copy; ``--execute`` relabels
    the host catalog under the staging lock and reloads the servers."""
    import fcntl

    from ops.policy_serving.config import validate_catalog

    def diff(old: dict, new: dict) -> dict:
        return {pid: {k: [old["policies"].get(pid, {}).get(k), e.get(k)] for k in ("usage", "execution_eligible")
                      if old["policies"].get(pid, {}).get(k) != e.get(k)}
                for pid, e in new["policies"].items()
                if any(old["policies"].get(pid, {}).get(k) != e.get(k) for k in USAGE_KEYS + ("reasons",))}

    if a.from_json:
        old = json.loads(a.from_json.read_text())
        new = relabel_catalog(old, a.validation_dir, a.sidecar_dir)
        validate_catalog(json.loads(json.dumps(new)))
        if a.out_json:
            a.out_json.write_text(json.dumps(new, indent=1) + "\n")
        print(json.dumps({"changed": diff(old, new), "eligible": sorted(
            p for p, e in new["policies"].items() if e.get("execution_eligible") is True)}, indent=1))
        return 0
    host = Host(ssh=a.ssh)
    a.lock_file.parent.mkdir(parents=True, exist_ok=True)
    log = lambda m: print(m, file=sys.stderr, flush=True)  # noqa: E731
    with open(a.lock_file, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        old = read_host_catalog(host)
        new = relabel_catalog(old, a.validation_dir, a.sidecar_dir)
        changed = diff(old, new)
        if changed or not catalog_copies_agree(host):
            write_host_catalog(host, new, execute=a.execute, log=log)
            signal_reload(host, execute=a.execute, log=log)
    print(json.dumps({"execute": bool(a.execute), "changed": changed}, indent=1))
    return 0


def cmd_bcflow(a) -> int:
    """``bcflow-variant``: add/refresh ``<id>+bcflow`` entries (experimental BC-flow actor) of staged one-step entries
    in the host catalog (no files are copied: the variant serves the base entry's staged files), then reload."""
    import fcntl

    decision = json.loads(a.decision_json.read_text()) if a.decision_json else None
    if decision is not None and len(a.policies) != 1:
        print(json.dumps({"error": "a bc_flow execution decision is bound to ONE checkpoint: give one policy id"}))
        return 2
    host = Host(ssh=a.ssh)
    a.lock_file.parent.mkdir(parents=True, exist_ok=True)
    with open(a.lock_file, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        catalog = read_host_catalog(host)
        entries = dict(catalog.get("policies") or {})
        report = {}
        for pid in a.policies:
            if pid not in entries:
                print(json.dumps({"error": f"{pid} is not in the host catalog"}))
                return 2
            bound = (decision or {}).get("checkpoint_sha256")
            if bound is not None and bound != entries[pid].get("sha256"):
                print(json.dumps({"error": f"the decision is bound to {bound}, {pid} serves {entries[pid].get('sha256')}"}))
                return 2
            vid, entry = bcflow_variant(pid, entries[pid], decision=decision)
            entries[vid] = entry
            report[vid] = {k: entry.get(k) for k in ("status", "usage", "execution_eligible")}
        aliases = {k: v for k, v in (catalog.get("aliases") or {}).items() if v in entries}
        write_host_catalog(host, build_catalog(sync_bcflow_variants(entries), aliases=aliases,
                                               criteria_sha256=catalog.get("criteria_sha256")),
                           execute=a.execute, log=lambda m: print(m, file=sys.stderr, flush=True))
        signal_reload(host, execute=a.execute, log=lambda m: print(m, file=sys.stderr, flush=True))
    print(json.dumps({"execute": bool(a.execute), "variants": report}, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ops/policy_serving/staging.py", description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", type=Path, default=Path("/home/ws/ws/splatter4d"),
                    help="checkout holding runs/ + experiments/ (read-only)")
    ap.add_argument("--splits", type=Path,
                    default=Path("/home/ws/ws/realworld-ops/calib-loma/ops/results/learning_inputs"))
    ap.add_argument("--raw-root", type=Path, default=Path("/home/ws/data/real_world"))
    ap.add_argument("--rig-dir", type=Path, default=Path("/home/ws/ws/realworld-ops/calib-loma/rel/rig"))
    ap.add_argument("--sidecar-dir", type=Path, default=Path("/home/ws/ws/realworld-ops/serving/pol-contracts"))
    ap.add_argument("--criteria", type=Path, default=Path("/home/ws/ws/realworld-ops/serving/POL-CRITERIA.md"))
    ap.add_argument("--no-probe", action="store_true", help="skip e2 (rows then cannot be deployable)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("inventory")
    i.add_argument("--evaluate", action="store_true", help="run criteria c-f on the completed candidates")
    i.add_argument("--out-json", type=Path)
    i.add_argument("--out-md", type=Path)
    i.add_argument("--host-catalog", action="store_true", help="annotate rows served by the inference host's catalog")
    i.add_argument("--from-json", type=Path, help="re-annotate this evaluated inventory instead of evaluating")
    i.add_argument("--ssh", default="irsl")
    for name in ("stage", "hook"):
        s = sub.add_parser(name)
        s.add_argument("runs", nargs="+")
        s.add_argument("--execute", action="store_true", help="really stage (default: dry run)")
        s.add_argument("--allow-comparison", action="store_true",
                       help="also stage FQL comparison-baseline rows (rwa100-*); refused without it")
        s.add_argument("--read-host", action="store_true", help="dry run: read the host catalog for the merge")
        if name == "stage":
            s.add_argument("--inventory", type=Path, help="use this evaluated inventory JSON instead of re-evaluating")
            s.add_argument("--label-quarantined", action="store_true",
                           help="also stage QUARANTINED runs as labelled catalog entries (refused unless the server "
                                "runs with --allow-quarantined; never deployable)")
        s.add_argument("--ssh", default="irsl")
        s.add_argument("--lock-file", type=Path, default=Path("/home/ws/ws/realworld-ops/serving/.pol-staging.lock"))
        s.add_argument("--bwlimit", default="10m")
        s.add_argument("--validation-dir", type=Path, default=VALIDATION_DIR,
                       help="deployment-validation records <run>.json (auto-eligibility pipeline)")
    r = sub.add_parser("relabel", help="re-derive every catalog entry's execution eligibility under the current rules")
    r.add_argument("--execute", action="store_true", help="relabel the HOST catalog + reload (default: dry run)")
    r.add_argument("--from-json", type=Path, help="transform this local catalog copy instead of the host catalog")
    r.add_argument("--out-json", type=Path, help="with --from-json: write the relabelled catalog here")
    r.add_argument("--validation-dir", type=Path, default=VALIDATION_DIR)
    r.add_argument("--ssh", default="irsl")
    r.add_argument("--lock-file", type=Path, default=Path("/home/ws/ws/realworld-ops/serving/.pol-staging.lock"))
    b = sub.add_parser("bcflow-variant", help="EXPERIMENTAL: serve staged policies' BC flow as <id>+bcflow")
    b.add_argument("policies", nargs="+", help="staged one-step policy ids")
    b.add_argument("--decision-json", type=Path, help="user decision {decided_by: user, at, text naming bc_flow}; "
                                                      "without it the variants are dry run / replay only")
    b.add_argument("--execute", action="store_true", help="really write the catalog + reload (default: dry run)")
    b.add_argument("--ssh", default="irsl")
    b.add_argument("--lock-file", type=Path, default=Path("/home/ws/ws/realworld-ops/serving/.pol-staging.lock"))
    a = ap.parse_args(argv)
    if a.cmd == "inventory":
        return cmd_inventory(a)
    if a.cmd == "bcflow-variant":
        return cmd_bcflow(a)
    if a.cmd == "relabel":
        return cmd_relabel(a)
    return cmd_stage(a, hook=a.cmd == "hook")


if __name__ == "__main__":
    raise SystemExit(main())
