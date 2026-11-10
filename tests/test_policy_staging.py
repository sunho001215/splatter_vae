"""Deployability criteria + inference-host staging of real-world FQL policies (``ops.policy_serving.criteria`` /
``ops.policy_serving.staging``; realworld-ops/serving/POL-CRITERIA.md). CPU only, labelled FIXTURE policies and a
synthetic registry/queue/runs tree; no SSH: every host command is refused in dry runs (asserted)."""

from __future__ import annotations

import json
import subprocess

import h5py
import numpy as np
import pytest
import yaml

from ops.policy_serving import criteria as C
from ops.policy_serving import staging as S
from ops.policy_serving.config import validate_catalog
from s4d.rwlearn.deploy.fixtures import FIXTURE_NATIVE_HW, make_fixture_policy, session_choices, synthetic_frames


# ------------------------------------------------------------------------------------------------ a, b
@pytest.mark.parametrize("config,completion,ok", [
    ({"train": {"num_epochs": 40, "max_global_steps": None}}, {"status": "completed", "epochs_completed": 40}, True),
    ({"train": {"num_epochs": 40, "max_global_steps": None}}, {"status": "completed", "epochs_completed": 39}, False),
    ({"train": {"num_epochs": 1000, "max_global_steps": 100001}}, {"status": "completed", "steps": 100001}, True),
    ({"train": {"max_steps": 2399999}}, {"status": "completed", "steps": 2399999}, True),  # index convention
    ({"train": {"max_steps": 2399999}}, {"status": "completed", "steps": 2000000}, False),
    ({"train": {"epochs": 15, "max_steps": None}, "run": {"total_steps": 66975}},
     {"status": "completed", "steps": 66975}, True),
    ({"train": {"num_epochs": 50, "max_global_steps": 300000}}, {"status": "paused", "steps": 300000}, False),
    ({"train": {"num_epochs": 40}}, None, False),
    ({"train": {}}, {"status": "completed", "steps": 5}, False),  # no budget key: never assumed complete
])
def test_pretraining_completion_against_the_runs_own_budget(config, completion, ok):
    assert C.pretraining_complete(config, completion)["pass"] is ok


def test_fql_completion_needs_the_configured_offline_steps():
    cfg = {"train": {"offline_steps": 1_000_000}}
    assert C.fql_complete(cfg, {"status": "completed", "steps": 1_000_000})["pass"]
    assert not C.fql_complete(cfg, {"status": "completed", "steps": 999_999})["pass"]
    assert not C.fql_complete(cfg, None)["pass"]


# ------------------------------------------------------------------------------------------------ c
def test_export_chain_requires_one_identical_sha256():
    meta = {"condition": "dynamo", "export_sha256": "a" * 64, "export_path": "runs/p-pre/encoder.pt"}
    assert C.export_chain(meta, {"export_sha256": "a" * 64}, "a" * 64)["pass"]
    assert not C.export_chain(meta, {"export_sha256": "b" * 64}, "a" * 64)["pass"]
    assert not C.export_chain(meta, {"export_sha256": "a" * 64}, None)["pass"]
    assert C.export_chain({"condition": "e2e_cnn"}, {"condition": "e2e_cnn"}, None)["pass"]
    assert not C.export_chain({"condition": "e2e_cnn"}, {"condition": "dynamo"}, None)["pass"]


# ------------------------------------------------------------------------------------------------ e1
def _feature_dataset(path, offset: float, rows=60, dim=32, seed=0):
    rng = np.random.default_rng(seed)
    with h5py.File(path, "w") as f:
        for ep in ("t/t-000000", "t/t-000001"):
            for tl in ("cam_a", "cam_b"):
                f.create_dataset(f"episodes/{ep}/{tl}/features",
                                 data=(offset + rng.standard_normal((rows, dim))).astype(np.float16))
    return ["t/t-000000", "t/t-000001"]


def test_feature_conditioning_separates_an_offset_dominated_readout(tmp_path):
    eps = _feature_dataset(tmp_path / "ok.h5", 0.0)
    good = C.feature_conditioning(tmp_path / "ok.h5", eps)
    assert good["pass"] and good["offset_to_signal"] < 1 and good["ln_unit_std_raw"] > 0.5 and good["rows"] == 240
    _feature_dataset(tmp_path / "off.h5", 50.0)
    bad = C.feature_conditioning(tmp_path / "off.h5", eps)
    assert not bad["pass"] and not bad["offset_pass"] and bad["offset_to_signal"] > C.E1_OFFSET_MAX
    again = C.feature_conditioning(tmp_path / "ok.h5", eps)  # fixed random stream: reproducible evidence
    assert again == good


# ------------------------------------------------------------------------------------------------ e2
def _samples(policy, n=12):
    choices = session_choices(policy, "s")
    spec = policy.input_spec.bind(**choices)
    cams = list(spec.camera_ids)
    session = {k: choices[k] for k in ("calibration", "native_depth_correction") if k in choices}
    out = []
    for i in range(n):
        frames = synthetic_frames(spec, 3, seed=100 + i, start_seq=10 * i, start_ns=10 ** 9 * i)
        t = max(f[-1].timestamp_ns for f in frames.values())
        out.append({"episode": f"t/t-{i:06d}", "cameras": cams, "index": 2,
                    "native_hw": dict.fromkeys(cams, FIXTURE_NATIVE_HW), "frames": frames, "session": session,
                    "pose7": np.array([0.4, 0.0, 0.3, 0, 0, 0, 1.0]) + 0.01 * i * np.eye(7)[0],
                    "pose_time_ns": t - 1000, "gripper_register": 3.0 * i, "gripper_time_ns": t - 2000})
    return out


@pytest.mark.parametrize("condition", ["reviwo", "e2e_cnn", "sincro", "mvp3d_single", "mvp3d_multiview"])
def test_visual_probe_uses_the_deploy_path_and_measures_vision(tmp_path, condition):
    policy = make_fixture_policy(tmp_path / condition, condition)
    r = C.visual_probe(policy, _samples(policy))  # asserts its cached path == DeployPolicy.act bitwise
    assert r["samples"] == 12 and r["d_vis"] > 0 and r["d_noise"] > 0 and r["action_spread"] > 0
    assert r["pass"] == (r["d_vis"] >= C.E2_SPREAD_FRACTION * r["action_spread"]
                         and r["d_vis"] >= C.E2_NOISE_FRACTION * r["d_noise"])
    assert policy.live_sessions == ()  # probe sessions are ended


def test_visual_probe_fails_a_vision_blind_policy(tmp_path, monkeypatch):
    import torch

    policy = make_fixture_policy(tmp_path / "blind", "reviwo")
    real = policy.offline.features
    monkeypatch.setattr(policy.offline, "features", lambda obs: torch.zeros_like(real(obs)))
    r = C.visual_probe(policy, _samples(policy))
    assert r["d_vis"] == 0 and not r["pass"]


# ------------------------------------------------------------------------------------------------ f
def _metrics(qs, budget=1_000_000, extra=None):
    steps = np.linspace(5000, budget, len(qs)).astype(int)
    rows = [{"step": int(s), "training/critic/q_mean": float(q), "training/critic/q_max": float(q) + 1,
             "validation/critic/q_mean": float(q), "training/actor/actor_loss": 1.0} for s, q in zip(steps, qs)]
    if extra:
        rows[-1].update(extra)
    return rows


def test_stability_rules():
    assert C.stability(_metrics([-40, -20, -5, -3, -2]), 1_000_000)["pass"]
    runaway = C.stability(_metrics([-40, 150, -5, -3, -2]), 1_000_000)
    assert not runaway["pass"] and not runaway["f2_no_runaway"]
    late = C.stability(_metrics([-40, -20, -5, 5, 8]), 1_000_000)
    assert not late["pass"] and not late["f3_late_feasible"] and late["f2_no_runaway"]
    nan = C.stability(_metrics([-40, -20, -5], extra={"training/actor/actor_loss": float("nan")}), 1_000_000)
    assert not nan["pass"] and not nan["f1_finite"]
    transient = C.stability(_metrics([-40, -20, -5, -3, 24, -2, -2, -2, -2, -2]), 1_000_000)
    assert transient["pass"] and transient["peak_q_mean"][1] == 24 and transient["first_q_max_positive"] is not None
    assert not C.stability([], 1_000_000)["pass"]


# ------------------------------------------------------------------------------------------------ classes
def _passing(**over):
    base = {"a": C.check(True), "b": C.check(True), "c": C.check(True), "d": C.check(True), "f": C.check(True),
            "e1": C.check(True, offset_to_signal=1.0, ln_unit_std_raw=0.8, offset_pass=True, ln_pass=True),
            "e2": C.check(True, d_vis=0.2, action_spread=0.3, d_noise=0.2)}
    base.update(over)
    return base


def test_classification_rules():
    assert C.classify("dynamo", _passing())["status"] == "deployable"
    att = C.classify("dynamo", _passing(e1=C.check(False, offset_to_signal=2.5, ln_unit_std_raw=0.33,
                                                   offset_pass=True, ln_pass=False)))
    assert att["status"] == "deployable" and att["flags"] == [C.ATTENUATED]
    off = C.classify("mvp3d_single", _passing(e1=C.check(False, offset_to_signal=120, ln_unit_std_raw=0.01,
                                                         offset_pass=False, ln_pass=False)))
    assert off["status"] == "quarantined"
    assert C.classify("dynamo", _passing(e2=C.check(False, d_vis=0.0, action_spread=0.3, d_noise=0.2)))["status"] \
        == "quarantined"
    assert C.classify("dynamo", _passing(b=C.check(False, reason="running")))["status"] == "incomplete"
    e2e = _passing()
    del e2e["a"], e2e["e1"]
    assert C.classify("e2e_cnn", e2e)["status"] == "deployable"
    assert C.classify("dynamo", _passing(), quarantine="decision")["status"] == "quarantined"
    assert C.classify("ours", {})["status"] == "fixture-only"
    assert C.classify("dynamo", _passing(f=C.check(None)))["status"] == "quarantined"  # never deployable unchecked


# ------------------------------------------------------------------------------------------------ inventory
def _event(run, kind, **kw):
    return json.dumps({"event": kind, "id": run, **kw}) + "\n"


def _done(run, files):
    return [_event(run, "launching"), _event(run, "launched"), _event(run, "remote_exited", code=0),
            _event(run, "results_synced", final=True, sync={"files": files}), _event(run, "exited", code=0)]


@pytest.fixture
def tree(tmp_path):
    runs = tmp_path / "runs"
    exp = tmp_path / "experiments"
    exp.mkdir()
    lines, jobs = [], []

    def put(run, files):
        d = runs / run
        d.mkdir(parents=True)
        for name, content in files.items():
            (d / name).write_text(content if isinstance(content, str) else json.dumps(content))
        return {name: S.sha256_file(d / name) for name in files}

    fql_cfg = yaml.safe_dump({"train": {"offline_steps": 1000}, "data": {"dataset": "runs/x-fqldata-dynamo-t/d.h5"}})
    done = put("x-fql-dynamo-t-s0", {"config.yaml": fql_cfg, "completion.json": {"status": "completed",
                                                                                     "steps": 1000},
                                     "policy.pt": "weights"})
    lines += _done("x-fql-dynamo-t-s0", done)
    put("x-fql-dynamo-t-s1", {"config.yaml": fql_cfg})
    lines += [_event("x-fql-dynamo-t-s1", "launching"), _event("x-fql-dynamo-t-s1", "launched")]
    pre = put("x-dynamo-t-pre", {"config.yaml": yaml.safe_dump({"train": {"num_epochs": 2}}),
                                 "completion.json": {"status": "completed", "epochs_completed": 2},
                                 "encoder.pt": "enc"})
    lines += _done("x-dynamo-t-pre", pre)
    jobs.append({"id": "rwng-20261106-fql-dynamo-t-s0", "args": ["--set", "data.dataset=runs/q/d.h5"]})
    jobs.append({"id": "rwcal-20261109-fql-mvp3d_single-t-s0", "args": []})
    (exp / "registry.jsonl").write_text("".join(lines).replace('"id": "x-', '"id": "rwx-'))
    (exp / "queue.yaml").write_text(yaml.safe_dump({"jobs": jobs}))
    for d in list(runs.iterdir()):
        d.rename(runs / ("rw" + d.name))
    return S.Sources(tmp_path, exp / "registry.jsonl", exp / "queue.yaml", tmp_path / "splits", tmp_path / "raw")


def test_inventory_reads_registry_and_runs_as_data(tree):
    inv = S.inventory(tree)
    fql = {r["run"]: r for r in inv["fql"]}
    assert set(fql) == {"rwx-fql-dynamo-t-s0", "rwx-fql-dynamo-t-s1", "rwng-20261106-fql-dynamo-t-s0",
                        "rwcal-20261109-fql-mvp3d_single-t-s0"}
    done = fql["rwx-fql-dynamo-t-s0"]
    assert done["registry_state"] == "completed_synced" and done["b"]["pass"] and done["policy_id"] == "dynamo-t-s0"
    assert done["policy"]["matches_remote"] is True
    assert fql["rwx-fql-dynamo-t-s1"]["registry_state"] == "running" and not fql["rwx-fql-dynamo-t-s1"]["b"]["pass"]
    assert fql["rwng-20261106-fql-dynamo-t-s0"]["superseded"]
    assert fql["rwcal-20261109-fql-mvp3d_single-t-s0"]["quarantine_decision"]
    pre = inv["pretraining"][0]
    assert pre["run"] == "rwx-dynamo-t-pre" and pre["a"]["pass"] and pre["encoder"]["matches_remote"]
    S.classify_rows(inv["fql"], {})
    status = {r["run"]: r["class"]["status"] for r in inv["fql"]}
    assert status == {"rwx-fql-dynamo-t-s0": "unevaluated", "rwx-fql-dynamo-t-s1": "incomplete",
                      "rwng-20261106-fql-dynamo-t-s0": "superseded",
                      "rwcal-20261109-fql-mvp3d_single-t-s0": "incomplete"}
    assert "(B) FQL policies" in S.markdown(inv, criteria_sha256=None)


def test_lifecycle_states():
    life = S.Lifecycle([{"event": "launching"}, {"event": "launched"}, {"event": "exited", "code": 1},
                        {"event": "failed"}])
    assert life.state == "failed"
    assert S.Lifecycle([{"event": "launched"}, {"event": "reset"}]).state == "reset"
    relaunch = S.Lifecycle([{"event": "launched"}, {"event": "results_synced", "final": True, "sync": {}},
                            {"event": "exited", "code": 0}, {"event": "launched"}])
    assert relaunch.state == "running" and relaunch.synced is None  # a sync before the last launch does not count


# ------------------------------------------------------------------------------------------------ catalog + host
def _row(status="deployable", seed=0):
    return {"run": f"r-fql-dynamo-t-s{seed}", "condition": "dynamo", "task": "t", "seed": seed,
            "policy_id": f"dynamo-t-s{seed}", "policy": {"path": f"runs/r-fql-dynamo-t-s{seed}/policy.pt",
                                                        "sha256": "c" * 64},
            "export": {"path": "runs/r-dynamo-t-pre/encoder.pt", "pretrain_run": "r-dynamo-t-pre", "sha256": "e" * 64},
            "class": {"status": status, "reasons": [] if status == "deployable" else ["x"], "flags": []}}


def test_catalog_entries_aliases_and_schema():
    entries = {"dynamo-t-s0": S.catalog_entry(_row()), "dynamo-t-s1": S.catalog_entry(_row(seed=1)),
               "dynamo-u-s0": {**S.catalog_entry(_row(status="quarantined")), "task": "u"}}
    aliases = S.legacy_aliases(entries)
    assert aliases == {"dynamo-t": "dynamo-t-s0", "dynamo-u": "dynamo-u-s0"}  # s0 only; quarantined: refused w/ reasons
    cat = S.build_catalog(entries, aliases=aliases, criteria_sha256="f" * 64)
    validate_catalog(json.loads(json.dumps(cat)))
    e = cat["policies"]["dynamo-t-s0"]
    assert e["training_contract"] == "contracts/r-fql-dynamo-t-s0.deploy_contract.json"
    assert e["export"] == "runs/r-dynamo-t-pre/encoder.pt" and e["checkpoint"].startswith("runs/")


def test_staging_files_never_include_datasets_and_dry_run_touches_no_host(tmp_path, monkeypatch):
    src = S.Sources(tmp_path, tmp_path / "r.jsonl", tmp_path / "q.yaml", tmp_path, tmp_path)
    row = _row()
    for rel in (row["policy"]["path"], row["export"]["path"], "runs/r-fql-dynamo-t-s0/config.yaml",
                "runs/r-fql-dynamo-t-s0/dataset.h5"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(rel)
    (tmp_path / "sc").mkdir()
    (tmp_path / "sc" / "r-fql-dynamo-t-s0.deploy_contract.json").write_text("{}")
    files = S.staging_files(src, row, tmp_path / "sc")
    assert not any(k.endswith(".h5") for k in files)
    assert {"runs/r-fql-dynamo-t-s0/policy.pt", "runs/r-dynamo-t-pre/encoder.pt",
            "contracts/r-fql-dynamo-t-s0.deploy_contract.json"} <= set(files)

    def refuse(*a, **k):
        raise AssertionError(f"dry run ran a command: {a}")

    monkeypatch.setattr(subprocess, "run", refuse)
    host = S.Host()
    shas = S.stage_files(host, files, execute=False, log=lambda m: None)
    assert shas["runs/r-fql-dynamo-t-s0/policy.pt"] == S.sha256_file(tmp_path / row["policy"]["path"])
    cat = S.build_catalog({"dynamo-t-s0": S.catalog_entry(row)}, aliases={}, criteria_sha256=None)
    S.write_host_catalog(host, cat, execute=False, log=lambda m: None)
    S.signal_reload(host, execute=False, log=lambda m: None)
