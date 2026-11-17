"""pick-and-place-v2 deployment hooks (2026-11-18), main line: pnp-v2 FQL ids (pnpv2-<date>-fql-<method>-pick-and-place-
v2-S-s<k>, ops/jobs/pnpv2_plan.py) parse to their task / method / seed / setting and get the policy ids the serving /
coordinator release lines use; they are NEVER execution-eligible without a deployment-validation record bound to their
checkpoint / contract / input_spec and become eligible with one; the earlier pick-and-place (v1) ids keep their parse,
setting and policy ids. This line has no e2e_cnn_noaug serving alias: its pnp-v2 rows are not recognized here.
Pure CPU, no host access."""

from __future__ import annotations

import json

import pytest

from ops.policy_serving import config as CFG
from ops.policy_serving import criteria as K
from ops.policy_serving import staging as S

PNP = "pick-and-place-v2"
PLAN = {"ours_orig": ("ours", "a10nq", "ours_orig", "Original"), "ours_ch01": ("ours", "a10nq", "ours_ch01", "Chamfer0.1"),
        "dynamo": ("dynamo", "a10nq", "dynamo", None), "reviwo": ("reviwo", "a10nq", "reviwo", None),
        "sincro": ("sincro", "a10nq", "sincro", None), "snerl": ("snerl", "a10nq", "snerl", None),
        "mvp3d_single": ("mvp3d_single", "a10nq", "mvp3d_multiview", None),
        "mvp3d_multiview": ("mvp3d_multiview", "a10nq", "mvp3d_multiview", None),
        "e2e_cnn": ("e2e_cnn", None, None, None)}


def run_of(method, seed):
    return f"pnpv2-20261117-fql-{method}-{PNP}-S-s{seed}"


def expected_id(method, seed):
    cond, setting, _, recipe = PLAN[method]
    if recipe:
        return f"ours-{PNP}-success-{recipe}-scratch-ckpt400k-s{seed}-a10nq"
    return f"{cond}-{PNP}-s{seed}" + (f"-{setting}" if setting else "")


@pytest.mark.parametrize("method", sorted(PLAN))
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_a_pnpv2_id_parses_to_its_task_method_seed_and_setting(method, seed):
    run = run_of(method, seed)
    cond, setting, export, _ = PLAN[method]
    for m in (S.FQL_RE.fullmatch(run), CFG.FQL_RUN_RE.fullmatch(run)):
        assert (m["condition"], m["task"], m["variant"], int(m["seed"])) == (cond, PNP, "S", seed)
    p = S.pnpv2(run)
    assert (p["method"], p["condition"], p["task"], p["seed"], p["setting"]) == (method, cond, PNP, seed, setting)
    assert p["export"] == (None if export is None else f"pnpv2-20261117-export-{export}-{PNP}-S")
    assert S.fql_setting(run) == setting and S._pnpv2_servable(run)
    assert S.pnpv2_policy_id(p) == expected_id(method, seed)


def test_noaug_and_unplanned_pnpv2_ids_are_not_recognized_here():
    noaug = run_of("e2e_cnn_noaug", 0)
    assert S.FQL_RE.fullmatch(noaug) is None and not S._pnpv2_servable(noaug)
    for run in ("pnpv2-20261117-fql-ours-pick-and-place-v2-S-s0", "pnpv2-20261117-fql-dynamo-pick-and-place-v2-SF-s0"):
        with pytest.raises(ValueError):
            S.pnpv2(run)
        assert not S._pnpv2_servable(run)


V1 = {"rwnq-20261111-fql-dynamo-pick-and-place-s1": ("dynamo-pick-and-place-s1-a10nq", "a10nq"),
      "rwnq-20261111-fql-reviwo-pick-and-place-s2": ("reviwo-pick-and-place-s2-a10nq", "a10nq"),
      "rwnt-20261110-fql-e2e_cnn-pick-and-place-s1": ("e2e_cnn-pick-and-place-s1", None),
      "rwcal-20261109-fql-mvp3d_single-pick-and-place-s0": ("mvp3d_single-pick-and-place-s0", None)}


@pytest.mark.parametrize("run", sorted(V1))
def test_v1_pick_and_place_ids_stay_as_they_are(run):
    pid, setting = V1[run]
    m = S.FQL_RE.fullmatch(run)
    assert m["task"] == "pick-and-place" and m["arm"] is None and m["variant"] is None and S.pnpv2(run) is None
    assert CFG.FQL_RUN_RE.fullmatch(run)["task"] == "pick-and-place"
    assert S.fql_setting(run) == setting
    assert S.policy_id(m["condition"], m["task"], int(m["seed"]), setting) == pid
    # an rwv2 (S) id keeps this line's earlier parse: no variant group outside pick-and-place-v2
    assert S.FQL_RE.fullmatch("rwv2-20261113-fql-dynamo-pick-and-place-S-s0")["task"] == "pick-and-place-S"


def _row(method, seed, status="deployable"):
    run = run_of(method, seed)
    cond, setting, _, _ = PLAN[method]
    checks = {"b": K.check(True, steps=10, budget=10, checkpoint_step=10),
              "c": K.check(True, export_sha256={"policy_meta": "e"}, deploy_load_cpu=True),
              "d": K.check(True, sha256="s"),
              "e2": K.check(True, d_vis_over_spread=0.7, d_vis_over_noise=30.0, samples=192),
              "f": K.check(True, final_q_mean=-40.0, late_q_mean_median=-41.0, peak_q_mean=[5, -12.0])}
    if cond != "e2e_cnn":
        checks.update(a=K.check(True, kind="epochs", budget=40, done=40),
                      e1=K.check(True, offset_pass=True, ln_pass=status != "restricted", offset_to_signal=1.0,
                                 ln_unit_std_raw=0.8 if status != "restricted" else 0.3))
    return {"run": run, "condition": cond, "task": PNP, "seed": seed, "fql_setting": setting,
            "policy_id": S.pnpv2_policy_id(S.pnpv2(run)),
            "policy": {"path": f"runs/{run}/policy.pt", "sha256": f"{seed + 1:x}" * 64},
            "class": {"status": status, "reasons": ["e1 LN"] if status == "restricted" else [], "flags": []},
            "checks": checks}


def _validation(row, spec):
    return {"run": row["run"], "checkpoint_sha256": row["policy"]["sha256"], "pass": True, "at": "t", "record": "r",
            "training_contract_sha256": "c" * 64, "input_spec_sha256": CFG.input_spec_digest(spec),
            "checks": {k: {"pass": True} for k in CFG.DEPLOY_VALIDATION_CHECKS}}


@pytest.mark.parametrize("method,status", [(m, "deployable") for m in sorted(PLAN)] + [("dynamo", "restricted")])
def test_pnpv2_policies_are_eligible_only_with_their_bound_validation_record(method, status):
    row = _row(method, 0, status)
    spec = {"condition": row["condition"]}
    pid = row["policy_id"]
    bare = S.catalog_entry(row, spec, None)
    assert bare["execution_eligible"] is False and bare["usage"] == "dry_run_replay_only"
    entry = S.catalog_entry(row, spec, _validation(row, spec))
    assert entry["execution_eligible"] is True
    assert entry["execution_decision"]["checkpoint_sha256"] == row["policy"]["sha256"]
    assert CFG.entry_fql_setting(pid, entry) == (PLAN[method][1] or CFG.LEGACY_SETTING)
    out = CFG.validate_catalog(S.build_catalog({pid: entry}, aliases={}, criteria_sha256=None))["policies"][pid]
    assert out["execution_eligible"] is True and CFG.eligibility_evidence_refusal(out) is None
    assert S.catalog_entry(row, spec, {**_validation(row, spec), "checkpoint_sha256": "f" * 64})["execution_eligible"] \
        is False
    if PLAN[method][3]:
        assert f"identity_recipe_{PLAN[method][3]}" in entry["flags"]
    if status == "restricted":
        assert pid == "dynamo-pick-and-place-v2-s0-a10nq"


def test_inventory_reads_pnpv2_rows_and_binds_exports(tmp_path):
    reg = tmp_path / "registry.jsonl"
    ids = ("pnpv2-20261117-pre-ours_ch01-pick-and-place-v2-S", "pnpv2-20261117-export-ours_ch01-pick-and-place-v2-S",
           run_of("ours_ch01", 0), run_of("e2e_cnn_noaug", 0), "rwnq-20261111-fql-dynamo-pick-and-place-s1",
           "rwnq-20261111-fql-ours_orig-pick-and-place-s1")
    reg.write_text("".join(json.dumps(x) + "\n" for rid in ids for x in (
        {"id": rid, "event": "launched", "attempt": 1, "host": "local"}, {"id": rid, "event": "exited", "code": 0})))
    (tmp_path / "q.yaml").write_text("jobs: []\n")
    pre = tmp_path / "runs/pnpv2-20261117-pre-ours_ch01-pick-and-place-v2-S"
    pre.mkdir(parents=True)
    pre.joinpath("config.yaml").write_text("loss: {chamfer: 0.0}\ntrain: {steps: 400000}\n")  # the orig recipe
    src = S.Sources(data_root=tmp_path, registry=reg, queue=tmp_path / "q.yaml", splits=tmp_path, raw_root=tmp_path)
    inv = S.inventory(src)
    assert [r["run"] for r in inv["fql"]] == sorted([run_of("ours_ch01", 0), "rwnq-20261111-fql-dynamo-pick-and-place-s1"])
    fql = {r["run"]: r for r in inv["fql"]}
    assert fql[run_of("ours_ch01", 0)]["policy_id"] == expected_id("ours_ch01", 0)
    others = {r["run"] for r in inv["other"]}
    assert {run_of("e2e_cnn_noaug", 0), "rwnq-20261111-fql-ours_orig-pick-and-place-s1",
            "pnpv2-20261117-pre-ours_ch01-pick-and-place-v2-S"} <= others
    exp = {r["run"]: r for r in inv["pretraining"]}["pnpv2-20261117-export-ours_ch01-pick-and-place-v2-S"]
    assert exp["a"]["pass"] is False and "scratch 400k arm" in exp["a"]["reason"]
    assert S.pretrain_run_of(src, {"run": run_of("mvp3d_single", 1), "condition": "mvp3d_single", "task": PNP}) == \
        f"pnpv2-20261117-export-mvp3d_multiview-{PNP}-S"
    assert S.RIG_DOCUMENTS[PNP] == f"rig-{PNP}-accepted.json"
