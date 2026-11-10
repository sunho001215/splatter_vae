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
    assert again == good and not good["normalized"]
    # amendment A1: a policy persisting a training-split standardisation is checked on what its head sees
    import torch

    payload = {"meta": {"feature_norm": {"kind": "train_standardize"}},
               "agent": {"modules": {"critic": {"encoder.feat_mean": torch.full((32,), 50.0),
                                                "encoder.feat_std": torch.ones(32)}}}}
    norm = C.feature_norm_of(payload)
    fixed = C.feature_conditioning(tmp_path / "off.h5", eps, norm)
    assert fixed["normalized"] and fixed["pass"] and fixed["offset_to_signal"] < 1
    assert C.feature_norm_of({"meta": {}, "agent": {}}) is None
    with pytest.raises(ValueError):
        C.feature_norm_of({"meta": {"feature_norm": {"kind": "x"}}, "agent": {}})


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
    ln = _passing(e1=C.check(False, offset_to_signal=2.5, ln_unit_std_raw=0.33, offset_pass=True, ln_pass=False))
    res = C.classify("dynamo", ln)  # amendment A2: the failed gate stays FAILED; dry run / replay only
    assert res["status"] == "restricted" and "LayerNorm-spread FAIL" in res["reasons"][0]
    assert res["reasons"][-1] == C.LN_SPREAD_DECISION and res["flags"] == []
    summary = C.criteria_summary("dynamo", {**ln, "e2": C.check(True, d_vis=0.2, action_spread=0.3, d_noise=0.01,
                                                              d_vis_over_spread=0.667, d_vis_over_noise=20.0,
                                                              samples=192)})
    assert summary["e1_layernorm_spread"] == {"pass": False, "value": 0.33, "threshold": ">= 0.5"}
    assert summary["e1_offset_to_signal"]["pass"] is True and summary["e2_visual_sensitivity"]["pass"] is True
    full = C.criteria_summary("dynamo", {"a": C.check(True, kind="epochs", budget=40, done=40),
                                         "b": C.check(True, steps=1000, budget=1000, checkpoint_step=1000,
                                                      synced_artifact=True),
                                         "c": C.check(True, export_sha256={"policy_meta": "e", "dataset": "e",
                                                                           "pretrain_encoder": "e"},
                                                      deploy_load_cpu=True),
                                         "d": C.check(True, sha256="s")})
    assert full["a_pretraining_complete"]["done"] == 40 and full["b_fql_complete"]["checkpoint_step"] == 1000
    assert full["c_export_chain_and_load"]["export_sha256"] == "e" and full["d_deploy_contract"]["sidecar_sha256"] == "s"
    ln_e2_fail = {**ln, "e2": C.check(False, d_vis=0.0, action_spread=0.3, d_noise=0.2)}
    assert C.classify("dynamo", ln_e2_fail)["status"] == "quarantined"  # LN failure + no visual use: quarantined
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


def test_usage_restrictions_eligibility_and_validator_rules():
    dep = S.usage_of("deployable")  # user decision 2026-11-10 21:10: deployable -> execution-eligible
    assert dep["execution_eligible"] is True and dep["usage"] == "execution"
    assert dep["execution_decision"]["decided_by"] == "user" and dep["execution_decision"]["text"]
    for status in ("restricted", "quarantined", "incomplete", "superseded"):
        u = S.usage_of(status)
        assert u["execution_eligible"] is False and u["usage"] != "execution" and "execution_decision" not in u
    assert S.usage_of("restricted")["usage"] == "dry_run_replay_only"
    assert S.usage_of("restricted")["usage_reasons"][0] == C.LN_SPREAD_DECISION
    assert S.usage_of("quarantined")["usage"] == "none"
    base = {**S.catalog_entry(_row()), "checkpoint": "runs/x/policy.pt"}
    assert base["execution_eligible"] is True

    def cat(**over):
        e = {**base, **over}
        return {"schema": "s4d-policy-catalog/1", "policies": {"p": {k: v for k, v in e.items() if v is not None}},
                "aliases": {}}

    validate_catalog(cat())  # deployable + usage execution + user decision
    with pytest.raises(ValueError, match="states its reasons"):
        validate_catalog(cat(status="restricted", reasons=[]))
    for over in ({"status": "restricted", "reasons": ["x"]}, {"status": "quarantined", "reasons": ["x"]}):
        with pytest.raises(ValueError, match="never execution-eligible"):
            validate_catalog(cat(**over))
    for decision in (None, {}, {"decided_by": "coordinator", "at": "t", "text": "x"},
                     {"decided_by": "user", "at": "t", "text": " "}):
        with pytest.raises(ValueError, match="execution_decision"):
            validate_catalog(cat(execution_decision=decision))
    with pytest.raises(ValueError, match="usage 'execution'"):
        validate_catalog(cat(usage="dry_run_replay_only"))  # eligible without execution usage
    with pytest.raises(ValueError, match="requires execution_eligible"):
        validate_catalog(cat(execution_eligible=False, execution_decision=None))  # execution usage, not eligible
    validate_catalog(cat(execution_eligible=False, usage="dry_run_replay_only", execution_decision=None))
    with pytest.raises(ValueError, match="non-finite"):  # would break the JSON wire format for every client
        validate_catalog(cat(criteria={"e2_visual_sensitivity": {"d_vis_over_noise": float("inf")}}))
    with pytest.raises(ValueError, match="usage"):
        validate_catalog(cat(usage="anything"))
    with pytest.raises(ValueError, match="true or false"):
        validate_catalog(cat(execution_eligible="yes"))


def test_completion_must_be_the_evaluated_checkpoint():
    b = C.fql_complete({"train": {"offline_steps": 1000}}, {"status": "completed", "steps": 1000})
    assert C.checkpoint_matches_completion(b, 1000, {"matches_remote": True})["pass"]
    mid = C.checkpoint_matches_completion(b, 500, {"matches_remote": True})
    assert not mid["pass"] and "step 500" in mid["reason"]
    other = C.checkpoint_matches_completion(b, 1000, {"matches_remote": False})
    assert not other["pass"] and "remote-verified" in other["reason"]
    assert not C.checkpoint_matches_completion(b, 1000, {"matches_remote": None})["pass"]


def test_constant_action_policy_fails_visual_sensitivity(tmp_path, monkeypatch):
    policy = make_fixture_policy(tmp_path / "const", "reviwo")
    real = policy.offline.act_from_features

    def constant(*a, **k):
        out = real(*a, **k)
        out["action_norm"] = np.zeros_like(out["action_norm"])
        return out

    monkeypatch.setattr(policy.offline, "act_from_features", constant)
    r = C.visual_probe(policy, _samples(policy))
    assert r["action_spread"] == 0 and r["d_vis"] == 0 and not r["pass"]


class FakeHost(S.Host):
    """In-memory inference host: host + container catalog copies; every other command succeeds."""

    def __init__(self, slow_read: float = 0.0):
        super().__init__()
        self.copies, self.writes, self.reloads, self.slow_read = {}, 0, 0, slow_read
        self.fail_container_write = False

    def run(self, cmd, *, stdin=None, check=True):
        import time as _t

        out, rc = "", 0
        if cmd.startswith(f"cat {self.root}/"):
            _t.sleep(self.slow_read)
            out = self.copies.get("host", "")
            rc = 0 if out else 1
        elif cmd.startswith(f"docker exec {self.container} cat "):
            out = self.copies.get("container", "")
            rc = 0 if out else 1
        elif cmd.startswith("chmod u+w"):
            self.copies["host"] = stdin
            self.writes += 1
        elif cmd.startswith(f"docker exec -i {self.container} sh -c 'cat >"):
            if self.fail_container_write:
                raise subprocess.CalledProcessError(1, cmd)
            self.copies["container"] = stdin
        elif "pgrep" in cmd:
            self.reloads += 1
        return subprocess.CompletedProcess(cmd, rc, out, "")


def _stage_env(tmp_path, monkeypatch, runs=("r-fql-e2e_cnn-t-s0",), status="deployable"):
    import argparse

    data, sc = tmp_path / "data", tmp_path / "sc"
    rows = []
    for run in runs:
        (data / "runs" / run).mkdir(parents=True, exist_ok=True)
        (data / "runs" / run / "policy.pt").write_text(run)
        sc.mkdir(exist_ok=True)
        (sc / f"{run}.deploy_contract.json").write_text("{}")
        seed = int(run[-1])
        f_ok = {"pass": True, "final_q_mean": -40.0, "late_q_mean_median": -41.0, "peak_q_mean": [5000, -12.0]}
        rows.append({"run": run, "condition": "e2e_cnn", "task": "t", "seed": seed, "policy_id": f"e2e_cnn-t-s{seed}",
                     "policy": {"path": f"runs/{run}/policy.pt", "sha256": S.sha256_file(data / "runs" / run /
                                                                                         "policy.pt"), "local": True},
                     "export": None, "class": {"status": status, "reasons": [] if status == "deployable" else ["x"],
                                               "flags": []},
                     "checks": {"d": {"pass": True, "sha256": S.sha256_file(sc / f"{run}.deploy_contract.json")},
                                "f": f_ok}})
    crit = tmp_path / "criteria.md"
    crit.write_text("criteria")
    inv = tmp_path / "inv.json"
    inv.write_text(json.dumps({"criteria_sha256": S.sha256_file(crit), "fql": rows}))
    monkeypatch.setattr(S, "stage_files", lambda host, files, **k: {r: S.sha256_file(p) for r, p in files.items()})
    monkeypatch.setattr(S, "local_input_spec", lambda *a: None)
    args = argparse.Namespace(inventory=inv, criteria=crit, runs=list(runs), sidecar_dir=sc, ssh="irsl", bwlimit="1m",
                              lock_file=tmp_path / "stage.lock", execute=True, read_host=False,
                              label_quarantined=False, data_root=data, splits=tmp_path, raw_root=tmp_path,
                              rig_dir=None, no_probe=False, evaluate=False)
    return args, inv


def test_stage_publishes_idempotently_repairs_partial_publication_and_withdraws(tmp_path, monkeypatch):
    host = FakeHost()
    monkeypatch.setattr(S, "Host", lambda **k: host)
    args, inv = _stage_env(tmp_path, monkeypatch)
    assert S.cmd_stage(args) == 0 and host.writes == 1 and host.copies["host"] == host.copies["container"]
    entry = json.loads(host.copies["host"])["policies"]["e2e_cnn-t-s0"]
    assert entry["status"] == "deployable" and entry["execution_eligible"] is True and entry["execution_decision"]
    assert S.cmd_stage(args) == 0 and host.writes == 1  # unchanged: nothing written, no reload
    host.copies["container"] = ""  # a publication interrupted after the host copy
    assert S.cmd_stage(args) == 0 and host.writes == 2 and host.copies["host"] == host.copies["container"]
    data = json.loads(inv.read_text())  # re-evaluation now fails: the staged entry of that run is withdrawn
    data["fql"][0]["class"] = {"status": "quarantined", "reasons": ["f: q_mean runaway"], "flags": []}
    data["fql"][0]["checks"]["f"] = {"pass": False, "final_q_mean": float("nan"), "late_q_mean_median": 150.0,
                                     "peak_q_mean": [900, float("inf")]}  # non-finite measurements must not block
    inv.write_text(json.dumps(data))
    assert S.cmd_stage(args, hook=True) == 3
    entry = json.loads(host.copies["host"])["policies"]["e2e_cnn-t-s0"]
    assert entry["status"] == "quarantined" and entry["usage"] == "none" and "f: q_mean runaway" in entry["reasons"]
    assert entry["execution_eligible"] is False and "execution_decision" not in entry  # withdrawn: never eligible
    f = entry["criteria"]["f_stability"]  # the re-evaluation's results replace the staged (passing) ones
    assert f["pass"] is False and f["final_q_mean"] is None and f["peak_q_mean"] == [900, None]


def test_stage_refuses_inputs_changed_since_the_evaluation(tmp_path, monkeypatch):
    host = FakeHost()
    monkeypatch.setattr(S, "Host", lambda **k: host)
    args, _ = _stage_env(tmp_path, monkeypatch)
    (args.sidecar_dir / "r-fql-e2e_cnn-t-s0.deploy_contract.json").write_text('{"camera_orders": [["other"]]}')
    assert S.cmd_stage(args) == 2 and host.writes == 0


def test_concurrent_stage_transactions_do_not_lose_updates(tmp_path, monkeypatch):
    import threading

    host = FakeHost(slow_read=0.3)  # both would read the same old catalog without the lock
    monkeypatch.setattr(S, "Host", lambda **k: host)
    args, _ = _stage_env(tmp_path, monkeypatch, runs=("r-fql-e2e_cnn-t-s0", "r-fql-e2e_cnn-t-s1"))
    a0 = type(args)(**{**vars(args), "runs": ["r-fql-e2e_cnn-t-s0"]})
    a1 = type(args)(**{**vars(args), "runs": ["r-fql-e2e_cnn-t-s1"]})
    ts = [threading.Thread(target=S.cmd_stage, args=(a,)) for a in (a0, a1)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(30)
    assert set(json.loads(host.copies["host"])["policies"]) == {"e2e_cnn-t-s0", "e2e_cnn-t-s1"}


def test_withdrawal_is_published_even_if_another_runs_upload_fails(tmp_path, monkeypatch):
    host = FakeHost()
    monkeypatch.setattr(S, "Host", lambda **k: host)
    args, inv = _stage_env(tmp_path, monkeypatch, runs=("r-fql-e2e_cnn-t-s0", "r-fql-e2e_cnn-t-s1"))
    assert S.cmd_stage(args) == 0
    data = json.loads(inv.read_text())
    data["fql"][0]["class"] = {"status": "quarantined", "reasons": ["re-evaluated: fails"], "flags": []}
    inv.write_text(json.dumps(data))

    def broken(*a, **k):
        raise RuntimeError("upload failed")

    monkeypatch.setattr(S, "stage_files", broken)
    assert S.cmd_stage(args) == 2
    pol = json.loads(host.copies["host"])["policies"]
    assert pol["e2e_cnn-t-s0"]["status"] == "quarantined" and pol["e2e_cnn-t-s1"]["status"] == "deployable"


def test_probe_ratios_stay_json_safe_when_the_noise_change_is_zero(tmp_path, monkeypatch):
    import torch

    policy = make_fixture_policy(tmp_path / "nz", "reviwo")
    real = policy.offline.act_from_features
    monkeypatch.setattr(policy.offline, "act_from_features", lambda f, p, m, noise=None, **k: real(
        f, p, m, noise=torch.zeros_like(noise) if noise is not None else None, **k))  # noise ignored
    r = C.visual_probe(policy, _samples(policy))
    assert r["d_noise"] == 0 and r["d_vis_over_noise"] is None and r["pass"]
    summary = C.criteria_summary("reviwo", {"e2": r})
    json.dumps(summary, allow_nan=False)


def test_withdrawal_is_published_before_another_runs_identity_failure(tmp_path, monkeypatch):
    host = FakeHost()
    monkeypatch.setattr(S, "Host", lambda **k: host)
    args, inv = _stage_env(tmp_path, monkeypatch, runs=("r-fql-e2e_cnn-t-s0", "r-fql-e2e_cnn-t-s1"))
    assert S.cmd_stage(args) == 0
    data = json.loads(inv.read_text())
    data["fql"][0]["class"] = {"status": "quarantined", "reasons": ["re-evaluated: fails"], "flags": []}
    inv.write_text(json.dumps(data))
    (args.sidecar_dir / "r-fql-e2e_cnn-t-s1.deploy_contract.json").write_text('{"changed": true}')
    assert S.cmd_stage(args) == 2  # B's inputs changed ...
    pol = json.loads(host.copies["host"])["policies"]
    assert pol["e2e_cnn-t-s0"]["status"] == "quarantined"  # ... but A's withdrawal is published
