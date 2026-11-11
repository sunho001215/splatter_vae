"""FQL settings and ours in policy staging (FQLNQ, user decision 2026-11-11 ~23:10): setting-prefixed rows (rwnq-*,
rwa100-*) are served under DISTINCT ids, never replacing the alpha-10 ids; the a100 comparison is staged only when
asked; a setting row's own config must carry its setting; ours rows chain to their export row and the scripts/train.py
pretraining run. Synthetic registry / queue / runs tree only; no host, SSH or evaluation."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import yaml

from ops.policy_serving import criteria as C
from ops.policy_serving import staging as S
from ops.policy_serving.config import validate_catalog


def _event(run, kind, **kw):
    return json.dumps({"event": kind, "id": run, **kw}) + "\n"


def _done(run, files):
    return [_event(run, "launching"), _event(run, "launched"), _event(run, "remote_exited", code=0),
            _event(run, "results_synced", final=True, sync={"files": files}), _event(run, "exited", code=0)]


def _fql_cfg(alpha, normalize):
    agent = {"alpha": alpha} if normalize is None else {"alpha": alpha, "normalize_q_loss": normalize}
    return yaml.safe_dump({"train": {"offline_steps": 1000}, "agent": agent,
                           "data": {"dataset": "runs/rwx-fqldata-reviwo-t/dataset.h5"}})


@pytest.fixture
def tree(tmp_path):
    runs, exp = tmp_path / "runs", tmp_path / "experiments"
    exp.mkdir()
    lines = []

    def put(run, files, done=True):
        d = runs / run
        d.mkdir(parents=True)
        for name, content in files.items():
            (d / name).write_text(content if isinstance(content, str) else json.dumps(content))
        if done:
            lines.extend(_done(run, {name: S.sha256_file(d / name) for name in files}))

    complete = {"status": "completed", "steps": 1000}
    put("rwx-fql-reviwo-t-s0", {"config.yaml": _fql_cfg(10.0, None), "completion.json": complete, "policy.pt": "w0"})
    put("rwnq-20261111-fql-reviwo-t-s0", {"config.yaml": _fql_cfg(10.0, True), "completion.json": complete,
                                          "policy.pt": "w1"})
    put("rwa100-20261111-fql-reviwo-t-s0", {"config.yaml": _fql_cfg(100.0, False), "completion.json": complete,
                                            "policy.pt": "w2"})
    put("rwnq-20261111-fql-reviwo-t-s1", {"config.yaml": _fql_cfg(10.0, False), "completion.json": complete,
                                          "policy.pt": "w3"})  # mislabelled: the nq id without normalisation
    put("rwx-ours-t-pre", {"config.yaml": yaml.safe_dump({"train": {"steps": 200000}}),
                           "completion.json": {"status": "completed", "steps": 200000}})
    put("rwx-ours-t-export", {"encoder.pt": "enc"})
    put("rwy-ours-t-pre", {"config.yaml": yaml.safe_dump({"train": {"steps": 200000}})}, done=False)
    put("rwy-ours-t-export", {"encoder.pt": "enc"}, done=False)
    (exp / "registry.jsonl").write_text("".join(lines))
    jobs = [{"id": "rwnq-20261111-fql-ours-t-s0", "args": []}, {"id": "rwa100-20261111-fql-ours-t-s2", "args": []},
            {"id": "rwy-ours-t-pre", "args": []}, {"id": "rwy-ours-t-export", "args": []}]  # never launched
    (exp / "queue.yaml").write_text(yaml.safe_dump({"jobs": jobs}))
    return S.Sources(tmp_path, exp / "registry.jsonl", exp / "queue.yaml", tmp_path / "splits", tmp_path / "raw")


def test_setting_rows_get_distinct_policy_ids_and_their_setting_is_checked(tree):
    inv = S.inventory(tree)
    fql = {r["run"]: r for r in inv["fql"]}
    assert fql["rwx-fql-reviwo-t-s0"]["policy_id"] == "reviwo-t-s0" and fql["rwx-fql-reviwo-t-s0"]["fql_setting"] is None
    nq, a100 = fql["rwnq-20261111-fql-reviwo-t-s0"], fql["rwa100-20261111-fql-reviwo-t-s0"]
    assert (nq["policy_id"], nq["fql_setting"]) == ("reviwo-t-s0-a10nq", "a10nq") and nq["b"]["pass"]
    assert (a100["policy_id"], a100["fql_setting"]) == ("reviwo-t-s0-a100", "a100") and a100["b"]["pass"]
    wrong = fql["rwnq-20261111-fql-reviwo-t-s1"]
    assert not wrong["b"]["pass"] and "a10nq setting" in wrong["b"]["reason"]
    assert fql["rwnq-20261111-fql-ours-t-s0"]["policy_id"] == "ours-t-s0-a10nq"  # queued, never launched
    assert len({r["policy_id"] for r in inv["fql"]}) == len(inv["fql"])  # no id is shared by two runs
    assert S.setting_check("rwa100-x-fql-reviwo-t-s0", {"agent": {"alpha": 100.0}}) is not None  # missing flag


def test_setting_entries_are_flagged_and_never_aliased():
    def row(run, pid, setting):
        return {"run": run, "condition": "reviwo", "task": "t", "seed": 0, "policy_id": pid, "fql_setting": setting,
                "policy": {"path": f"runs/{run}/policy.pt", "sha256": "c" * 64},
                "export": {"path": "runs/r-reviwo-t-pre/encoder.pt", "pretrain_run": "r-reviwo-t-pre", "sha256": "e" * 64},
                "class": {"status": "deployable", "reasons": [], "flags": []}}

    entries = {"reviwo-t-s0": S.catalog_entry(row("r-fql-reviwo-t-s0", "reviwo-t-s0", None)),
               "reviwo-t-s0-a10nq": S.catalog_entry(row("rwnq-1-fql-reviwo-t-s0", "reviwo-t-s0-a10nq", "a10nq"))}
    assert entries["reviwo-t-s0"]["flags"] == [] and entries["reviwo-t-s0-a10nq"]["flags"] == ["fql_setting_a10nq"]
    aliases = S.legacy_aliases(entries)
    assert aliases == {"reviwo-t": "reviwo-t-s0"}  # the legacy alias keeps naming the alpha-10 policy
    validate_catalog(json.loads(json.dumps(S.build_catalog(entries, aliases=aliases, criteria_sha256=None))))
    vid, variant = S.bcflow_variant("reviwo-t-s0-a10nq", entries["reviwo-t-s0-a10nq"])
    assert vid == "reviwo-t-s0-a10nq+bcflow" and "fql_setting_a10nq" in variant["flags"]


def test_ours_export_row_carries_the_pretraining_completion(tree):
    inv = S.inventory(tree)
    pre = {r["run"]: r for r in inv["pretraining"]}
    done, open_ = pre["rwx-ours-t-export"], pre["rwy-ours-t-export"]
    assert done["condition"] == "ours" and done["pretraining_run"] == "rwx-ours-t-pre"
    assert done["a"]["pass"] and done["a"]["budget"] == 200000 and done["encoder"]["matches_remote"] is True
    assert not open_["a"]["pass"] and "registry state" in open_["a"]["reason"]
    assert "rwx-ours-t-pre" not in pre  # the scripts/train.py run is reached through its export row
    row = {"run": "rwnq-20261111-fql-ours-t-s0", "condition": "ours", "task": "t", "dataset": None}
    assert S.pretrain_run_of(tree, row) == "rwnq-20261111-ours-t-export"
    assert S.pretrain_run_of(tree, {**row, "condition": "sincro"}) == "rwnq-20261111-sincro-t-pre"
    assert C.pretraining_complete({"train": {"steps": 200000}}, {"status": "completed", "steps": 200000})["pass"]
    assert not C.pretraining_complete({"train": {"steps": 200000}}, {"status": "completed", "steps": 1000})["pass"]


def test_comparison_rows_are_staged_only_when_asked(tmp_path, capsys, monkeypatch):
    def args(**kw):
        return SimpleNamespace(runs=["rwnq-20261111-fql-reviwo-t-s0", "rwa100-20261111-fql-reviwo-t-s0"], **kw)

    def refuse(*a, **k):
        raise AssertionError("must refuse before any evaluation")

    monkeypatch.setattr(S, "_evaluated", refuse)
    assert S.cmd_stage(args(allow_comparison=False)) == 2
    assert "--allow-comparison" in json.loads(capsys.readouterr().out)["error"]
    assert S.cmd_stage(args(), hook=True) == 2  # the hook (stage watcher) never stages it by default either
    for cmd in ("stage", "hook"):  # the CLI default refuses; the flag exists on both commands
        assert S.main(["--data-root", str(tmp_path), cmd, "rwa100-20261111-fql-reviwo-t-s0"]) == 2
        capsys.readouterr()
    monkeypatch.setattr(S, "_evaluated", lambda a, runs: {"fql": []})
    out = S.cmd_stage(args(allow_comparison=True, execute=False, read_host=False, ssh="x", bwlimit="1m",
                           lock_file=tmp_path / "lock", inventory=None, data_root=tmp_path, splits=tmp_path,
                           raw_root=tmp_path, rig_dir=None, sidecar_dir=tmp_path, criteria=tmp_path / "c.md"))
    assert out == 2 and "unknown FQL runs" in capsys.readouterr().out  # past the comparison gate


def test_stage_refuses_inventories_of_another_policy_id_rule(tmp_path, capsys, monkeypatch):
    """Codex review (5ec2b46): an inventory evaluated before the FQL-setting rule names an rwnq run by the alpha-10 id;
    staging it would overwrite that entry (and move its legacy alias). Refused before any host / catalog access."""
    def host(**k):
        raise AssertionError("no host access for a refused inventory")

    monkeypatch.setattr(S, "Host", host)
    crit = tmp_path / "criteria.md"
    crit.write_text("criteria")
    run = "rwnq-20261111-fql-reviwo-t-s0"
    row = {"run": run, "condition": "reviwo", "task": "t", "seed": 0, "policy_id": "reviwo-t-s0",  # old rule
           "class": {"status": "deployable", "reasons": [], "flags": []}}
    base = {"criteria_sha256": S.sha256_file(crit), "fql": [row]}
    args = SimpleNamespace(runs=[run], criteria=crit, allow_comparison=False)
    for doc, message in ((base, "staging rule"), ({**base, "staging_rule": S.STAGING_RULE}, "FQL-setting rule")):
        inv = tmp_path / "inv.json"
        inv.write_text(json.dumps(doc))
        assert S.cmd_stage(SimpleNamespace(**vars(args), inventory=inv)) == 2
        assert message in json.loads(capsys.readouterr().out)["error"]
