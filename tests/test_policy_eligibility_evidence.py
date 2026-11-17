"""Pipeline audit 2026-11-17 (realworld-ops/robotpc-pipeline-audit-20261117/AUDIT.md): F1, every execution-eligible
catalog entry (end-to-end CNN included) needs checkpoint-bound execution evidence, an execution decision alone is
never enough, and a catalog holding such an entry is still served with that entry DEMOTED; F8, the server never
unpickles a catalog checkpoint before its sha256 matched. CPU only; no sockets (the server registry is built, never
started); no host access."""

from __future__ import annotations

import builtins
import hashlib
import json

import pytest
import torch

from ops.policy_serving import config as CFG
from ops.policy_serving import criteria as C
from ops.policy_serving import staging as S
from ops.policy_serving.server import PolicyServer, checkpoint_setting_refusal, describe_entry, verified_checkpoint

SHA = "c" * 64
E2E_ID, E2E_RUN = "e2e_cnn-t-s0", "r-fql-e2e_cnn-t-s0"
SPEC = {"condition": "e2e_cnn"}


def _validation(entry: dict) -> dict:
    """A passed deployment-validation summary bound to the entry's run / checkpoint / contract / input_spec."""
    return {"run": entry["run"], "checkpoint_sha256": entry["sha256"], "pass": True, "at": "t", "record": "r.json",
            "training_contract_sha256": "a" * 64, "input_spec_sha256": CFG.input_spec_digest(entry.get("input_spec")),
            "checks": {k: {"pass": True} for k in CFG.DEPLOY_VALIDATION_CHECKS}}


def _decision_only_e2e(sha: str = SHA) -> dict:
    """The live-catalog form of an E2E entry before F1: eligible by the 2026-11-10 decision alone (no checkpoint
    binding, no deployment validation), criteria all passed."""
    return {"checkpoint": f"runs/{E2E_RUN}/policy.pt", "sha256": sha, "condition": "e2e_cnn", "task": "t", "seed": 0,
            "run": E2E_RUN, "training_contract": f"contracts/{E2E_RUN}.deploy_contract.json", "status": "deployable",
            "reasons": [], "flags": [], "input_spec": SPEC, "criteria": {k: {"pass": True} for k in CFG.ELIGIBLE_CRITERIA},
            "usage": "execution", "execution_eligible": True, "execution_decision": dict(S.EXECUTION_DECISION),
            "usage_reasons": ["user decision 2026-11-10T21:10+09:00: execution-eligible"]}


def _catalog(policies: dict) -> dict:
    return {"schema": CFG.CATALOG_SCHEMA, "policies": policies, "aliases": {}}


def test_an_e2e_entry_with_only_an_execution_decision_is_not_eligible(tmp_path, caplog):
    """F1: the decision alone (or bound to the checkpoint, but without a validation) is never evidence; the catalog
    is NOT refused: the entry is published dry run / replay only with the reason, every other entry stays as is."""
    bare = _decision_only_e2e()
    bound_decision = {**bare, "execution_decision": {**S.EXECUTION_DECISION, "checkpoint_sha256": SHA}}
    validated_unbound = {**bare, "deploy_validation": _validation(bare)}  # decision not bound to the checkpoint
    complete = {**bound_decision, "deploy_validation": _validation(bare)}
    other = {**complete, "seed": 1, "run": "r-fql-e2e_cnn-t-s1", "sha256": "d" * 64,
             "checkpoint": "runs/r-fql-e2e_cnn-t-s1/policy.pt",
             "execution_decision": {**S.EXECUTION_DECISION, "checkpoint_sha256": "d" * 64}}
    other["deploy_validation"] = _validation(other)
    for entry, why in ((bare, "no deploy_validation"), (bound_decision, "no deploy_validation"),
                       (validated_unbound, "execution decision is not bound")):
        with caplog.at_level("WARNING", logger=CFG.__name__):
            out = CFG.validate_catalog(_catalog({E2E_ID: entry, "e2e_cnn-t-s1": other}))  # no raise
        e = out["policies"][E2E_ID]
        assert e["execution_eligible"] is False and e["usage"] == "dry_run_replay_only"
        assert "execution_decision" not in e and "deploy_validation" not in e
        assert e["usage_reasons"][0].startswith(CFG.EVIDENCE_DEMOTION) and why in e["usage_reasons"][0]
        assert e["status"] == "deployable" and e["criteria"] == entry["criteria"]  # labels, not the class, change
        assert out["policies"]["e2e_cnn-t-s1"] == other  # the rest of the catalog is untouched
        assert E2E_ID in caplog.text and why in caplog.text  # the demotion is logged
        caplog.clear()
    assert CFG.validate_catalog(_catalog({E2E_ID: complete}))["policies"][E2E_ID]["execution_eligible"] is True
    for crit in ({k: v for k, v in complete["criteria"].items() if k != "f_stability"},
                 {**complete["criteria"], "e2_visual_sensitivity": {"pass": False}}):
        e = CFG.validate_catalog(_catalog({E2E_ID: {**complete, "criteria": crit}}))["policies"][E2E_ID]
        assert e["execution_eligible"] is False and "criteria" in e["usage_reasons"][0]
    # staging never produces the decision-only form: without a bound record the entry is dry run / replay only
    u = S.usage_of("deployable", "e2e_cnn", None, checkpoint_sha256=SHA)
    assert u["execution_eligible"] is False and u["execution_decision"] is None
    u = S.usage_of("deployable", "e2e_cnn", None, checkpoint_sha256=SHA, validation=_validation(bare))
    assert u["execution_eligible"] is True and u["execution_decision"]["checkpoint_sha256"] == SHA
    # the server applies such a catalog (no crash, nothing refused) and never advertises the entry eligible
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(_catalog({E2E_ID: bare})))
    srv = PolicyServer(catalog=path, checkpoint_root=tmp_path, device="cpu")  # registry only; never started
    info = srv.entries[E2E_ID].info()
    assert srv.entries[E2E_ID].status == "deployable" and info["execution_eligible"] is False
    assert info["usage"] == "dry_run_replay_only" and info["usage_reasons"][0].startswith(CFG.EVIDENCE_DEMOTION)


def test_the_evidence_criteria_are_the_names_criteria_summary_publishes():
    checks = {"a": C.check(True, kind="epochs", budget=40, done=40),
              "b": C.check(True, steps=10, budget=10, checkpoint_step=10),
              "c": C.check(True, export_sha256={"policy_meta": "e"}, deploy_load_cpu=True),
              "d": C.check(True, sha256="s"),
              "e1": C.check(True, offset_pass=True, ln_pass=True, offset_to_signal=1.0, ln_unit_std_raw=0.8),
              "e2": C.check(True, d_vis_over_spread=0.7, d_vis_over_noise=30.0, samples=192),
              "f": C.check(True, final_q_mean=-40.0, late_q_mean_median=-41.0, peak_q_mean=[5, -12.0])}
    assert set(C.criteria_summary("dynamo", checks)) == set(CFG.ELIGIBLE_CRITERIA + CFG.FROZEN_ELIGIBLE_CRITERIA)
    e2e = {k: v for k, v in checks.items() if k not in ("a", "e1")}
    assert set(C.criteria_summary("e2e_cnn", e2e)) == set(CFG.ELIGIBLE_CRITERIA)


class _Trap:
    """Unpickling this creates ``marker`` (``open(marker, "w")``): proof that pickle code ran at load time."""

    def __init__(self, marker):
        self.marker = str(marker)

    def __reduce__(self):
        return builtins.open, (self.marker, "w")


def test_a_checkpoint_sha_mismatch_is_detected_before_unpickling(tmp_path):
    """F8: describe / setting checks (and the server registry that runs them) hash the checkpoint bytes against the
    entry's sha256 BEFORE ``torch.load``; a differing or missing hash never reaches the unpickler."""
    marker = tmp_path / "unpickled"
    ck = tmp_path / "runs" / "x" / "policy.pt"
    ck.parent.mkdir(parents=True)
    torch.save({"meta": _Trap(marker), "agent": {"config": {"alpha": 10.0}}}, ck)
    good = hashlib.sha256(ck.read_bytes()).hexdigest()
    entry = {"checkpoint": "runs/x/policy.pt", "sha256": "d" * 64, "condition": "e2e_cnn", "task": "t", "seed": 0,
             "run": E2E_RUN, "status": "deployable", "reasons": [], "flags": []}
    with pytest.raises(ValueError, match="not unpickled"):
        verified_checkpoint(entry, tmp_path)
    with pytest.raises(ValueError, match="not unpickled"):
        describe_entry(entry, tmp_path)
    assert "not unpickled" in checkpoint_setting_refusal(E2E_ID, entry, tmp_path)
    for sha in (None, "", "x" * 64):  # no valid hash to check against: refused when required (every catalog entry)
        with pytest.raises(ValueError, match="no valid sha256"):
            verified_checkpoint({**entry, "sha256": sha}, tmp_path, require_sha256=True)
    with pytest.raises(ValueError, match="no valid sha256"):
        verified_checkpoint({**entry, "sha256": "x" * 64}, tmp_path, require_sha256=False)  # a named hash is checked
    path = tmp_path / "catalog.json"  # the registry: a tampered file is unavailable, never unpickled
    path.write_text(json.dumps(_catalog({E2E_ID: entry})))
    srv = PolicyServer(catalog=path, checkpoint_root=tmp_path, device="cpu")  # registry only; never started
    assert srv.entries[E2E_ID].status == "error" and "not unpickled" in srv.entries[E2E_ID].reasons[0]
    assert not marker.exists()
    verified_checkpoint({**entry, "sha256": good}, tmp_path)  # control: the matching hash does unpickle the trap
    assert marker.exists()
